"""Application contracts for the bounded background-insights queue (B2a-1).

Drives the real main.py functions -- the upload follow-up dispatcher, the
pending-analysis route, the queue worker's job and the insights helper -- with
an in-memory database, an in-memory Ollama transport and synthetic documents.
No real database, model, network, subprocess or production service is reached;
the worker threads run only these in-process doubles.

What these prove: an upload starts no thread per document and queues only the
committed document's id; pending analysis reads ids only, never content or
filenames, queues them without blocking and reports truthful counts, and keeps
its 401/403 and a sanitized 503; a queued job reloads the CURRENT row by its
exact id, skips deleted or already-analysed documents, closes every connection
and cursor it opens, resolves the trusted helper default exactly once at job
start and sends the same helper payload as before; model capacity is released
only for a call that was never dispatched or whose reply carried the terminal
marker (``done`` exactly True and ``done_reason`` exactly "stop" or
"length"), and is otherwise held -- with its document key -- through every
local cleanup, with no further model call or retry; only a valid "stop"
completion can store an insight; and a failed queue admission leaves an
already committed document committed, with the existing generic warning.

What these do NOT prove: anything about a real Ollama server (whether it
stopped work), a real PostgreSQL, an ASGI server's lifecycle, a restart, or GPU
residency.

REQUIRES THE SOCKETPAIR-AWARE SIBLING GUARD (see test_main_route_harness.py):
entering a TestClient builds a real asyncio loop.
"""

import contextlib
import importlib
import json
import os
import queue as stdlib_queue
import sys
import threading
import time

import dotenv
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation_profiles import GEMMA_CANDIDATE, LLAMA_CONTROL, ModelRegistry, Qualification, default_model_registry

_EMBEDDING_PROFILE_KEYS = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_QUERY_PREFIX",
    "EMBEDDING_DOCUMENT_PREFIX",
    "EMBEDDING_PROFILE_UNVERIFIED_ACK",
)
_ADMIN_SQL = "SELECT is_admin FROM users WHERE token = %s AND COALESCE(is_active, 1) <> 0"
_PENDING_IDS_SQL = (
    "SELECT d.id FROM documents d LEFT JOIN document_insights di ON di.document_id = d.id "
    "WHERE di.id IS NULL ORDER BY d.upload_timestamp DESC LIMIT %s"
)
_LOAD_SQL = (
    "SELECT d.filename, d.content FROM documents d LEFT JOIN document_insights di "
    "ON di.document_id = d.id WHERE d.id = %s AND di.id IS NULL"
)
_UPSERT_PREFIX = "INSERT INTO document_insights"
_BACKFILL_UNAVAILABLE = "Insight analysis is temporarily unavailable. Please try again shortly."
_DOC_TEXT = "SYNTHETIC-AGREEMENT. The lease term is two years and the rent is due monthly. " * 6
_INSIGHTS = json.dumps({"doc_type": "Lease", "keywords": ["lease"], "facts": ["The term is two years."],
                        "entities": ["Synthetic Tenant"]})
_SECRET = "SECRET-DETAIL password=hunter2 http://db.internal"
SYNTHETIC_DIGEST = "5e" * 32
_WAIT = 5.0  # bounded wait for one in-process worker step


def _helper_jobs():
    return importlib.import_module("helper_jobs")


def _admission():
    return importlib.import_module("generation_admission")


@contextlib.contextmanager
def _imported_main():
    previous = sys.modules.pop("main", None)
    try:
        yield importlib.import_module("main")
    finally:
        sys.modules.pop("main", None)
        if previous is not None:
            sys.modules["main"] = previous


@pytest.fixture
def main_module(monkeypatch):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    for key in _EMBEDDING_PROFILE_KEYS:
        monkeypatch.delenv(key, raising=False)
    with _imported_main() as module:
        yield module


# --- in-memory collaborators -------------------------------------------------------


class _RecordingQueue:
    """Stands in for main._INSIGHTS_QUEUE where only admission matters."""

    def __init__(self):
        self.enqueued = []
        self.batches = []

    def start(self):
        return True

    def close(self):
        pass

    def enqueue(self, document_id):
        self.enqueued.append(document_id)
        return _helper_jobs().EnqueueOutcome.ACCEPTED

    def enqueue_batch(self, document_ids):
        ids = list(document_ids)
        self.batches.append(ids)
        return _helper_jobs().BatchAdmission(accepted=len(ids), already_pending=0, capacity_rejected=0)


_REAL_THREAD = threading.Thread
_FOLLOW_UP_NAMES = {"_extract_document_insights", "generate_kb_from_document"}


def _record_follow_up_threads(monkeypatch):
    """Record (and never start) any thread whose target is a document
    follow-up -- the old per-document dispatch. Other threads are untouched."""
    constructed = []

    class _RecordingThread(_REAL_THREAD):
        def __init__(self, *args, target=None, **kwargs):
            self._follow_up = getattr(target, "__name__", "") in _FOLLOW_UP_NAMES
            if self._follow_up:
                constructed.append(target.__name__)
            super().__init__(*args, target=target, **kwargs)

        def start(self):
            if not self._follow_up:
                super().start()

    monkeypatch.setattr(threading, "Thread", _RecordingThread)
    return constructed


class _Cursor:
    def __init__(self, connection):
        self._connection = connection
        self._rows = []
        self.closed = False

    def execute(self, sql, params=()):
        db = self._connection.db
        norm = " ".join(sql.split())
        params = tuple(params)
        db.statements.append((norm, params))
        if norm in db.fail_sql:
            raise RuntimeError(_SECRET)
        self._rows = []
        if norm == _ADMIN_SQL:
            self._rows = [(db.admins[params[0]],)] if params[0] in db.admins else []
        elif norm.startswith("SELECT d.id") and "FROM documents d LEFT JOIN document_insights" in norm:
            # The pending-analysis selection. Answers whatever columns are
            # selected, so a route that still read content would receive it.
            columns = norm.split(" FROM ", 1)[0][len("SELECT "):].split(", ")
            pending = [doc_id for doc_id in db.pending_order if doc_id not in db.insights][: params[-1]]
            self._rows = [
                tuple({"d.id": doc_id, "d.filename": db.documents[doc_id][0], "d.content": db.documents[doc_id][1]}[c]
                      for c in columns)
                for doc_id in pending
            ]
        elif norm == _LOAD_SQL:
            doc_id = params[0]
            if doc_id in db.documents and (doc_id not in db.insights or db.load_ignores_insights):
                self._rows = [db.documents[doc_id]]
        elif norm.startswith(_UPSERT_PREFIX):
            assert "ON CONFLICT (document_id) DO UPDATE" in norm
            document_id, *values = params
            self._connection.pending[document_id] = tuple(values)
        else:
            db.unexpected.append(norm)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def close(self):
        self.closed = True
        self._connection.db.cursor_closes += 1


class _Connection:
    def __init__(self, db):
        self.db = db
        self.pending = {}
        self.closes = 0

    def cursor(self):
        cursor = _Cursor(self)
        self.db.cursors.append(cursor)
        return cursor

    def commit(self):
        if self.db.fail_commit:
            raise RuntimeError(_SECRET)
        self.db.insights.update(self.pending)
        self.pending = {}
        self.db.commits += 1

    def rollback(self):
        self.pending = {}
        self.db.rollbacks += 1

    def close(self):
        self.closes += 1


class _DB:
    """documents + document_insights + users, enough for these routes and jobs."""

    def __init__(self, documents=None, insights=None, admins=None):
        self.documents = dict(documents or {})  # id -> (filename, content)
        self.pending_order = sorted(self.documents, reverse=True)  # upload_timestamp DESC
        self.insights = dict(insights or {})  # document_id -> stored values
        self.admins = dict(admins or {"tok-admin": 1, "tok-user": 0})
        self.statements = []
        self.unexpected = []
        self.connections = []
        self.cursors = []
        self.cursor_closes = 0
        self.commits = 0
        self.rollbacks = 0
        self.fail_sql = set()
        self.fail_commit = False
        self.connect_mode = "ok"  # ok | none | raise
        self.load_ignores_insights = False

    def connect(self):
        if self.connect_mode == "raise":
            raise RuntimeError(_SECRET)
        if self.connect_mode == "none":
            return None
        connection = _Connection(self)
        self.connections.append(connection)
        return connection

    def all_closed_once(self):
        return all(c.closes == 1 for c in self.connections) and all(c.closed for c in self.cursors)


class _Reply:
    def __init__(self, body=None, json_error=False):
        self._body = body
        self._json_error = json_error

    def raise_for_status(self):
        pass

    def json(self):
        if self._json_error:
            raise ValueError("Expecting value: line 1 column 1")
        return self._body


class _Ollama:
    """requests.post double for helper calls. A reply may be a body, a _Reply,
    or an exception raised AFTER the call was submitted (the fake server keeps
    working: ``server_running`` stays true and nothing ever reports it done)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.posts = []
        self.server_running = 0

    def __call__(self, url, json=None, timeout=None):
        self.posts.append({"url": url, "json": json, "timeout": timeout})
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            self.server_running += 1
            raise reply
        return reply if isinstance(reply, _Reply) else _Reply(reply)


def _finished(text, reason="stop"):
    return {"response": text, "done": True, "done_reason": reason}


def _install(monkeypatch, main_module, db, ollama):
    monkeypatch.setattr(main_module, "connect_to_postgres", db.connect)
    monkeypatch.setattr(main_module, "get_active_ollama_base_url", lambda: "http://ollama.invalid")
    monkeypatch.setattr(main_module.requests, "post", ollama)


@pytest.fixture
def worker_queue(monkeypatch, main_module):
    """A real HelperJobQueue running the real _run_insight_job on one worker
    thread, installed as main._INSIGHTS_QUEUE. ``finished`` yields
    (document_id, state) after each job's accounting is complete."""
    helper_jobs = _helper_jobs()
    finished = stdlib_queue.Queue()
    queue = helper_jobs.HelperJobQueue(
        settings=helper_jobs.HelperQueueSettings(workers=1, max_queued=4),
        run_job=lambda document_id, call: main_module._run_insight_job(document_id, call),
        on_finished=lambda document_id, state: finished.put((document_id, state)),
    )
    monkeypatch.setattr(main_module, "_INSIGHTS_QUEUE", queue)
    assert queue.start() is True
    try:
        yield queue, finished
    finally:
        queue.close()


def _next_finished(finished):
    return finished.get(timeout=_WAIT)


def _quiet(finished, seconds=0.3):
    """No further job finishes within a short bounded window."""
    with pytest.raises(stdlib_queue.Empty):
        finished.get(timeout=seconds)


# --- old-bug detection: per-document threads and content-reading backfill ----------


class _RecordingWriteCursor:
    def __init__(self, connection):
        self._connection = connection
        self._row = None

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        self._connection.statements.append(normalized)
        self._row = (501,) if normalized.startswith("INSERT INTO documents") else None

    def fetchone(self):
        return self._row

    def close(self):
        pass


class _RecordingWriteConnection:
    def __init__(self):
        self.statements = []
        self.committed = False
        self.closed = False

    def cursor(self):
        return _RecordingWriteCursor(self)

    def commit(self):
        self.committed = True

    def rollback(self):
        pass

    def close(self):
        self.closed = True


class _FixedEmbeddings:
    def __init__(self, dimension):
        self._dimension = dimension

    def embed_documents(self, texts):
        return [[0.25] * self._dimension for _ in texts]


def _upload_client(monkeypatch, main_module, connection):
    monkeypatch.setattr(main_module, "_initialize_application", lambda: None)
    monkeypatch.setattr(main_module, "_resolve_caller_id", lambda token: "uploader-under-test")
    monkeypatch.setattr(main_module, "connect_to_postgres", lambda: connection)
    monkeypatch.setattr(main_module, "_EMBEDDING_PROVIDER", _FixedEmbeddings(main_module._EMBEDDING_PROFILE.dimension))
    return TestClient(main_module.app)


def test_an_upload_starts_no_follow_up_thread_and_queues_only_its_id(monkeypatch, main_module):
    connection = _RecordingWriteConnection()
    recorder = _RecordingQueue()
    monkeypatch.setattr(main_module, "_INSIGHTS_QUEUE", recorder, raising=False)
    client = _upload_client(monkeypatch, main_module, connection)
    with client:
        threads = _record_follow_up_threads(monkeypatch)
        response = client.post("/upload", files=[("files", ("indexed.txt", _DOC_TEXT.encode(), "text/plain"))])

    assert response.status_code == 200, response.text
    assert connection.committed, "the document committed before any follow-up"
    assert threads == [], "no thread may be started per uploaded document"
    assert recorder.enqueued == [501], "only the committed document id is queued, for insights"
    [result] = response.json()["results"]
    assert result["status"] == "indexed" and result["warnings"] == []
    assert not [sql for sql in connection.statements if "knowledge_base" in sql]


def test_pending_analysis_reads_ids_only_and_starts_no_thread(monkeypatch, main_module):
    db = _DB(documents={7: ("seven.txt", _DOC_TEXT), 9: ("nine.txt", _DOC_TEXT)})
    recorder = _RecordingQueue()
    monkeypatch.setattr(main_module, "_INSIGHTS_QUEUE", recorder, raising=False)
    monkeypatch.setattr(main_module, "connect_to_postgres", db.connect)
    threads = _record_follow_up_threads(monkeypatch)
    client = TestClient(main_module.app)

    response = client.post("/admin/document_insights/analyze_pending", params={"token": "tok-admin", "max_docs": 5})

    pending_selects = [sql for sql, _ in db.statements if "LEFT JOIN document_insights" in sql]
    assert pending_selects == [_PENDING_IDS_SQL], "only document ids are read -- never content or filenames"
    assert threads == [], "no thread may be started per pending document"
    assert response.status_code == 200, response.text
    assert response.json() == {"queued": 2, "already_pending": 0, "capacity_rejected": 0}
    assert recorder.batches == [[9, 7]], "ids in the existing order, nothing else"
    assert db.all_closed_once()


# --- pending analysis -----------------------------------------------------------------


def _pending_client(monkeypatch, main_module, db):
    monkeypatch.setattr(main_module, "connect_to_postgres", db.connect)
    return TestClient(main_module.app)


def _idle_queue(max_queued, queued=()):
    """A started, accepting queue whose worker never runs, so counts are exact."""
    helper_jobs = _helper_jobs()

    class _ParkedThread:
        def __init__(self, target=None, name=None, args=(), daemon=None):
            pass

        def start(self):
            pass

    queue = helper_jobs.HelperJobQueue(
        settings=helper_jobs.HelperQueueSettings(workers=1, max_queued=max_queued),
        run_job=lambda document_id, call: pytest.fail("no job may run here"),
        thread_factory=_ParkedThread,
    )
    assert queue.start() is True
    for document_id in queued:
        assert queue.enqueue(document_id) is helper_jobs.EnqueueOutcome.ACCEPTED
    return queue


def test_a_partial_pending_batch_reports_new_duplicate_and_rejected_counts(monkeypatch, main_module):
    db = _DB(documents={doc_id: (f"doc-{doc_id}.txt", _DOC_TEXT) for doc_id in (5, 6, 7, 8, 9)})
    queue = _idle_queue(max_queued=3, queued=(5,))
    monkeypatch.setattr(main_module, "_INSIGHTS_QUEUE", queue)
    client = _pending_client(monkeypatch, main_module, db)

    response = client.post("/admin/document_insights/analyze_pending", params={"token": "tok-admin", "max_docs": 50})

    assert response.status_code == 200, response.text
    assert response.json() == {"queued": 2, "already_pending": 1, "capacity_rejected": 2}
    snapshot = queue.snapshot()
    assert (snapshot.queued, snapshot.running, snapshot.unresolved) == (3, 0, 0)
    assert db.all_closed_once()


@pytest.mark.parametrize("max_docs, bound", [(0, 1), (-5, 1), (1, 1), (15, 15), (50, 50), (500, 50)])
def test_pending_analysis_keeps_max_docs_between_1_and_50(monkeypatch, main_module, max_docs, bound):
    db = _DB()
    monkeypatch.setattr(main_module, "_INSIGHTS_QUEUE", _idle_queue(max_queued=100))
    client = _pending_client(monkeypatch, main_module, db)

    response = client.post("/admin/document_insights/analyze_pending", params={"token": "tok-admin", "max_docs": max_docs})

    assert response.status_code == 200
    assert [params for sql, params in db.statements if sql == _PENDING_IDS_SQL] == [(bound,)]


@pytest.mark.parametrize("params, status", [({}, 401), ({"token": "tok-user"}, 403), ({"token": "tok-nobody"}, 403)],
                         ids=["missing-token", "non-admin", "unknown-token"])
def test_pending_analysis_keeps_its_authorization(monkeypatch, main_module, params, status):
    db = _DB(documents={7: ("seven.txt", _DOC_TEXT)})
    queue = _idle_queue(max_queued=100)
    monkeypatch.setattr(main_module, "_INSIGHTS_QUEUE", queue)
    client = _pending_client(monkeypatch, main_module, db)

    response = client.post("/admin/document_insights/analyze_pending", params=params)

    assert response.status_code == status
    assert [sql for sql, _ in db.statements if sql == _PENDING_IDS_SQL] == []
    assert queue.snapshot().queued == 0
    assert db.all_closed_once()


@pytest.mark.parametrize("failure", ["queue-unavailable", "connect-none", "connect-raises", "select-raises"])
def test_pending_analysis_dependency_failures_are_a_sanitized_503_with_no_accepted_work(
    monkeypatch, main_module, capsys, failure
):
    helper_jobs = _helper_jobs()
    db = _DB(documents={7: ("seven.txt", _DOC_TEXT)})
    queue = _idle_queue(max_queued=100)
    if failure == "queue-unavailable":
        queue.close()
    elif failure == "connect-none":
        db.connect_mode = "none"
    elif failure == "connect-raises":
        db.connect_mode = "raise"
    else:
        db.fail_sql.add(_PENDING_IDS_SQL)
    monkeypatch.setattr(main_module, "_INSIGHTS_QUEUE", queue)
    client = _pending_client(monkeypatch, main_module, db)
    capsys.readouterr()

    response = client.post("/admin/document_insights/analyze_pending", params={"token": "tok-admin"})

    assert response.status_code == 503
    assert response.json() == {"detail": _BACKFILL_UNAVAILABLE}
    assert queue.snapshot().queued == 0
    assert queue.enqueue(7) is (
        helper_jobs.EnqueueOutcome.UNAVAILABLE if failure == "queue-unavailable" else helper_jobs.EnqueueOutcome.ACCEPTED
    )
    assert db.all_closed_once()
    out = capsys.readouterr().out
    assert "SECRET-DETAIL" not in response.text + out and "hunter2" not in response.text + out


# --- the queued job: current-row reload, cleanup and call accounting -------------------


def test_a_valid_insight_is_stored_and_capacity_released(monkeypatch, main_module, worker_queue, capsys):
    queue, finished = worker_queue
    CallState = _admission().CallState
    db = _DB(documents={11: ("eleven.txt", _DOC_TEXT)})
    ollama = _Ollama(_finished(_INSIGHTS))
    _install(monkeypatch, main_module, db, ollama)

    assert queue.enqueue(11) is _helper_jobs().EnqueueOutcome.ACCEPTED
    assert _next_finished(finished) == (11, CallState.OBSERVED_TERMINAL)

    assert 11 in db.insights and db.insights[11][0] == "Lease"
    [load] = [params for sql, params in db.statements if sql == _LOAD_SQL]
    assert load == (11,), "the current row is reloaded by its exact id"
    [post] = ollama.posts
    assert post["json"] == {
        "model": "llama3.1:8b", "prompt": post["json"]["prompt"], "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": 1200}, "keep_alive": "5m",
    }
    assert post["timeout"] == (5, 120.0)
    assert db.all_closed_once() and len(db.connections) == 2, "load and upsert each closed their own"
    snapshot = queue.snapshot()
    assert (snapshot.running, snapshot.unresolved, snapshot.free_capacity) == (0, 0, 1)
    assert queue.enqueue(11) is _helper_jobs().EnqueueOutcome.ACCEPTED, "a finished key is released"
    assert "SYNTHETIC-AGREEMENT" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "reply",
    [_finished(_INSIGHTS, "length"), _finished("   "), _finished(None),
     {"thinking": _INSIGHTS, "response": "", "done": True, "done_reason": "stop"}],
    ids=["length", "blank", "non-string", "thinking-only"],
)
def test_a_terminal_reply_that_is_not_a_valid_insight_frees_capacity_and_writes_nothing(
    monkeypatch, main_module, worker_queue, reply
):
    queue, finished = worker_queue
    db = _DB(documents={12: ("twelve.txt", _DOC_TEXT)}, insights={})
    db.load_ignores_insights = True
    db.insights[12] = ("Older", "[]", "[]", "[]", "2026-01-01T00:00:00")  # a row written meanwhile
    _install(monkeypatch, main_module, db, _Ollama(reply))

    queue.enqueue(12)

    assert _next_finished(finished) == (12, _admission().CallState.OBSERVED_TERMINAL)
    assert db.insights[12][0] == "Older", "an older valid row is never overwritten"
    assert not [sql for sql, _ in db.statements if sql.startswith(_UPSERT_PREFIX)]
    assert queue.snapshot().free_capacity == 1


_UNKNOWN_REPLIES = {
    "done-missing": {"response": _INSIGHTS, "done_reason": "stop"},
    "done-false": {"response": _INSIGHTS, "done": False, "done_reason": "stop"},
    "done-string": {"response": _INSIGHTS, "done": "true", "done_reason": "stop"},
    "done-one": {"response": _INSIGHTS, "done": 1, "done_reason": "stop"},
    "reason-missing": {"response": _INSIGHTS, "done": True},
    "reason-load": {"response": _INSIGHTS, "done": True, "done_reason": "load"},
    "reason-case": {"response": _INSIGHTS, "done": True, "done_reason": "STOP"},
    "error-body": {"error": "runner busy", "done": True, "done_reason": "stop"},
    "not-an-object": [_INSIGHTS],
    "malformed-json": _Reply(json_error=True),
    "transport-error": ConnectionError("connection reset after the request was sent"),
    "read-timeout": TimeoutError("read timed out"),
}


@pytest.mark.parametrize("case", sorted(_UNKNOWN_REPLIES))
def test_unobserved_termination_holds_capacity_and_key_through_every_local_cleanup(
    monkeypatch, main_module, worker_queue, case
):
    """The fake server may still be generating: an exception, a normal end of
    the HTTP body without the terminal marker, or an error body never frees
    the slot, the document key or permits another model call."""
    queue, finished = worker_queue
    helper_jobs = _helper_jobs()
    db = _DB(documents={13: ("thirteen.txt", _DOC_TEXT), 14: ("fourteen.txt", _DOC_TEXT)})
    ollama = _Ollama(_UNKNOWN_REPLIES[case])
    _install(monkeypatch, main_module, db, ollama)

    queue.enqueue(13)
    assert _next_finished(finished) == (13, _admission().CallState.UNKNOWN)

    assert queue.enqueue(13) is helper_jobs.EnqueueOutcome.DUPLICATE, "the unresolved key is retained"
    assert queue.enqueue(14) is helper_jobs.EnqueueOutcome.ACCEPTED, "other ids may wait within Q"
    _quiet(finished)
    queue.close()  # local cleanup: intake closed, workers signalled
    assert queue.start() is True  # a later lifespan in the same process
    _quiet(finished)

    snapshot = queue.snapshot()
    assert (snapshot.unresolved, snapshot.free_capacity, snapshot.queued) == (1, 0, 1)
    assert len(ollama.posts) == 1, "no next model call and no retry while the call is unresolved"
    assert not [sql for sql, _ in db.statements if sql.startswith(_UPSERT_PREFIX)]
    assert db.all_closed_once()


@pytest.mark.parametrize("failure", ["deleted", "already-analysed", "connect-none", "connect-raises", "load-raises"])
def test_a_pre_dispatch_failure_or_skip_frees_capacity_without_a_model_call(
    monkeypatch, main_module, worker_queue, capsys, failure
):
    queue, finished = worker_queue
    db = _DB(documents={15: ("fifteen.txt", _DOC_TEXT)})
    if failure == "deleted":
        db.documents.clear()
    elif failure == "already-analysed":
        db.insights[15] = ("Done", "[]", "[]", "[]", "2026-01-01T00:00:00")
    elif failure == "connect-none":
        db.connect_mode = "none"
    elif failure == "connect-raises":
        db.connect_mode = "raise"
    else:
        db.fail_sql.add(_LOAD_SQL)
    ollama = _Ollama()
    _install(monkeypatch, main_module, db, ollama)
    capsys.readouterr()

    queue.enqueue(15)

    assert _next_finished(finished) == (15, _admission().CallState.NEVER_DISPATCHED)
    assert ollama.posts == []
    assert queue.snapshot().free_capacity == 1
    assert db.all_closed_once()
    out = capsys.readouterr().out
    assert "SECRET-DETAIL" not in out and "fifteen" not in out and "SYNTHETIC-AGREEMENT" not in out


def _qualified_gemma_default():
    base = default_model_registry()
    gemma = __import__("dataclasses").replace(
        base.get(GEMMA_CANDIDATE),
        expected_digest=SYNTHETIC_DIGEST,
        qualification=Qualification(think_false_verified=True, combined_capacity_verified=True,
                                    quality_verified=True, evidence_ref="synthetic test identity"),
    )
    return ModelRegistry(profiles=(base.get(LLAMA_CONTROL), gemma), default_profile_id=GEMMA_CANDIDATE)


def test_an_identity_failure_at_job_start_frees_capacity_without_a_model_call(monkeypatch, main_module, worker_queue):
    queue, finished = worker_queue
    db = _DB(documents={16: ("sixteen.txt", _DOC_TEXT)})
    ollama = _Ollama()
    _install(monkeypatch, main_module, db, ollama)
    monkeypatch.setattr(main_module, "_MODEL_REGISTRY", _qualified_gemma_default())
    lookups = []
    monkeypatch.setattr(main_module, "_installed_model_digest", lambda tag: lookups.append(tag) or "77" * 32)

    queue.enqueue(16)

    assert _next_finished(finished) == (16, _admission().CallState.NEVER_DISPATCHED)
    assert lookups == ["gemma4:31b"] and ollama.posts == []
    assert queue.snapshot().free_capacity == 1


def test_a_qualified_candidate_job_keeps_one_snapshot_and_its_expected_and_observed_digest(
    monkeypatch, main_module, worker_queue
):
    queue, finished = worker_queue
    db = _DB(documents={17: ("seventeen.txt", _DOC_TEXT)})
    ollama = _Ollama(_finished(_INSIGHTS))
    _install(monkeypatch, main_module, db, ollama)
    monkeypatch.setattr(main_module, "_MODEL_REGISTRY", _qualified_gemma_default())
    monkeypatch.setattr(main_module, "_installed_model_digest", lambda tag: "sha256:" + SYNTHETIC_DIGEST.upper())
    resolved = []
    real_resolve = main_module._resolve_helper_generation

    def recording_resolve():
        snapshot = real_resolve()
        resolved.append(snapshot)
        return snapshot

    monkeypatch.setattr(main_module, "_resolve_helper_generation", recording_resolve)

    queue.enqueue(17)

    assert _next_finished(finished) == (17, _admission().CallState.OBSERVED_TERMINAL)
    [snapshot] = resolved
    assert (snapshot.profile_id, snapshot.selection) == (GEMMA_CANDIDATE, "default")
    assert (snapshot.expected_digest, snapshot.observed_digest) == (SYNTHETIC_DIGEST, SYNTHETIC_DIGEST)
    [post] = ollama.posts
    assert post["json"]["model"] == "gemma4:31b" and post["json"]["think"] is False
    assert post["json"]["options"] == {"temperature": 0.1, "num_ctx": 8192, "num_predict": 1200}
    assert post["timeout"] == (5, 150.0)
    assert 17 in db.insights


@pytest.mark.parametrize("failure", ["no-json-object", "connect-none", "connect-raises", "upsert-raises", "commit-raises"])
def test_a_terminal_reply_then_parse_or_database_failure_frees_capacity_and_cleans_up(
    monkeypatch, main_module, worker_queue, capsys, failure
):
    queue, finished = worker_queue
    db = _DB(documents={18: ("eighteen.txt", _DOC_TEXT)})
    reply = _finished("no braces in this reply at all") if failure == "no-json-object" else _finished(_INSIGHTS)
    _install(monkeypatch, main_module, db, _Ollama(reply))
    real_connect = db.connect
    if failure in ("connect-none", "connect-raises"):
        calls = {"n": 0}

        def connect():
            calls["n"] += 1
            if calls["n"] == 1:
                return real_connect()  # the job's load succeeds
            if failure == "connect-none":
                return None
            raise RuntimeError(_SECRET)

        monkeypatch.setattr(main_module, "connect_to_postgres", connect)
    elif failure == "upsert-raises":
        original_execute = _Cursor.execute

        def execute(self, sql, params=()):
            if " ".join(sql.split()).startswith(_UPSERT_PREFIX):
                raise RuntimeError(_SECRET)
            return original_execute(self, sql, params)

        monkeypatch.setattr(_Cursor, "execute", execute)
    elif failure == "commit-raises":
        db.fail_commit = True
    capsys.readouterr()

    queue.enqueue(18)

    assert _next_finished(finished) == (18, _admission().CallState.OBSERVED_TERMINAL)
    assert 18 not in db.insights
    assert queue.snapshot().free_capacity == 1
    assert db.all_closed_once(), "every opened connection and cursor is closed exactly once"
    if failure in ("upsert-raises", "commit-raises"):
        assert db.rollbacks == 1, "an uncommitted insight write is rolled back"
    out = capsys.readouterr().out
    assert "SECRET-DETAIL" not in out and "SYNTHETIC-AGREEMENT" not in out


# --- upload follow-up with a real ingestion service -------------------------------------


def test_a_failed_queue_admission_keeps_the_committed_document_with_the_generic_warning(monkeypatch, main_module):
    helper_jobs = _helper_jobs()

    class _Unstartable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(_SECRET)

    unavailable = helper_jobs.HelperJobQueue(
        settings=helper_jobs.HelperQueueSettings(workers=1, max_queued=4),
        run_job=lambda document_id, call: pytest.fail("no job may run"),
        thread_factory=_Unstartable,
    )
    monkeypatch.setattr(main_module, "_INSIGHTS_QUEUE", unavailable)
    connection = _RecordingWriteConnection()
    client = _upload_client(monkeypatch, main_module, connection)

    with client:  # the lifespan's worker start fails; the application still serves
        response = client.post("/upload", files=[("files", ("kept.txt", _DOC_TEXT.encode(), "text/plain"))])
        version = client.get("/version")

    assert version.status_code == 200
    assert response.status_code == 200, response.text
    [result] = response.json()["results"]
    assert result["status"] == "indexed" and result["document_id"] == 501
    assert result["warnings"] == ["Post-commit follow-up dispatch failed."]
    assert connection.committed and connection.closed
    assert not [sql for sql in connection.statements if "knowledge_base" in sql]
    assert unavailable.snapshot().accepting is False
    assert "SECRET-DETAIL" not in response.text
