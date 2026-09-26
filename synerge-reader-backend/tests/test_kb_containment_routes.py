"""Route-level contracts for the PR #54 knowledge-base containment.

The knowledge base has no owner column and holds answers and Q&A derived from
users' private documents, so until it is user-owned:

* ``GET /knowledge_base`` is admin-only, identified by an
  ``Authorization: Bearer`` header (not a ``?token=`` query parameter), and a
  refused caller causes no knowledge-base query at all;
* ``POST /submit_correction`` and ``PUT /put_ratings`` require a valid Bearer
  identity and touch only the caller's own history row -- an anonymous or
  invalid-token request issues no history write at all, and a cross-user,
  ownerless, or missing-row request changes and commits nothing;
* no correction, from any caller, is copied into the knowledge base;
* ``POST /ask`` neither reads the knowledge base nor puts its text into the
  model prompt, in any answer mode, for an authenticated, anonymous, or
  invalid-token caller -- while the caller's own document evidence, the
  answer stream, citations, and history persistence keep working.

The upload and /ask auto-write removals are also proven where those routes
are already exercised: tests/test_main_upload_route_adapter.py and
tests/test_main_citation_wiring.py.

These drive the real FastAPI app through ``TestClient`` against an in-memory
fake database that emulates the owner predicate (``user_id = NULL`` never
matches) and records every statement, so "nothing was written" is asserted
from the statements actually issued, not from the source text.

REQUIRES THE SOCKETPAIR-AWARE SIBLING GUARD, for the reason
tests/test_main_route_harness.py documents: entering a ``TestClient`` builds a
real asyncio event loop, whose Windows self-pipe uses ``socket.socketpair()``.
``main._initialize_application`` is replaced before any client is built, and
``main.connect_to_postgres``, ``main._EMBEDDING_PROVIDER`` and (for /ask)
``main.post_ollama`` are replaced with in-memory doubles, so there is no
database, Ollama, network, or subprocess.
"""

import ast
import contextlib
import copy
import importlib
import json
import os
from pathlib import Path
import sys

import dotenv
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


_MAIN_PY_PATH = Path(__file__).resolve().parents[1] / "main.py"

# Duplicated from the other TestClient files on purpose (see
# tests/test_main_upload_route_adapter.py): a shared conftest.py would add
# collection-wide state to every test module.
_EMBEDDING_PROFILE_KEYS = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_QUERY_PREFIX",
    "EMBEDDING_DOCUMENT_PREFIX",
    "EMBEDDING_PROFILE_UNVERIFIED_ACK",
)

# token -> (user id, is_admin)
_USERS = {
    "tok-admin": ("admin-id", 1),
    "tok-alice": ("alice-id", 0),
    "tok-bob": ("bob-id", 0),
}

_INITIAL_HISTORY = {
    1: {"user_id": "alice-id", "answer": "alice answer", "comment": None, "rating": None},
    2: {"user_id": "bob-id", "answer": "bob answer", "comment": None, "rating": None},
    3: {"user_id": None, "answer": "ownerless answer", "comment": None, "rating": None},
    4: {"user_id": "admin-id", "answer": "admin answer", "comment": None, "rating": None},
}

# One KB row whose answer and source filename stand in for private content.
_KB_ROW = (
    11, "kb question", "kb original", "private kb answer", "2026-09-01",
    None, "User", 0, "private-contract.pdf", "document",
)

# id -> (filename, owner, content). Each scope holds exactly one short document,
# so the real planner supplies it whole (complete-document evidence), which
# needs no embedding. The leading marker identifies whose text reached a prompt.
_DOCUMENTS = {
    10: ("alice-lease.txt", "alice-id", "ALICE-DOC: the lease term is two years."),
    20: ("shared-notes.txt", None, "OWNERLESS-DOC: the notice period is thirty days."),
    30: ("bob-nda.txt", "bob-id", "BOB-DOC: the confidentiality period is five years."),
}
_DOC_MARKERS = ("ALICE-DOC", "OWNERLESS-DOC", "BOB-DOC")

# What a knowledge-base read would return. It must never reach a prompt.
_KB_SENTINEL = "KB-SENTINEL answer derived from another user's private document"

_SCOPE_OWNERLESS = (
    "SELECT id, filename, title, length(content) FROM documents "
    "WHERE user_id IS NULL ORDER BY id"
)
_SCOPE_OWNED = (
    "SELECT id, filename, title, length(content) FROM documents "
    "WHERE user_id = %s ORDER BY id"
)
_LOAD_OWNERLESS = "SELECT content FROM documents WHERE id = %s AND user_id IS NULL"
_LOAD_OWNED = "SELECT content FROM documents WHERE id = %s AND user_id = %s"
_INSERT_HISTORY = (
    "INSERT INTO chat_history (ts, selected_text, question, answer, user_id) "
    "VALUES (%s, %s, %s, %s, %s) RETURNING id"
)

_SELECT_OWNED_ROW = "SELECT id FROM chat_history WHERE id = %s AND user_id = %s"
_UPDATE_ANSWER = (
    "UPDATE chat_history SET answer = %s, comment = %s "
    "WHERE id = %s AND user_id = %s RETURNING id"
)
_UPDATE_RATING = (
    "UPDATE chat_history SET rating = %s, comment = %s "
    "WHERE id = %s AND user_id = %s RETURNING id"
)


class _FakeDB:
    """Committed state shared by every connection the app opens."""

    def __init__(self):
        self.history = copy.deepcopy(_INITIAL_HISTORY)
        self.history_inserts = []
        self.statements = []
        self.unexpected = []
        self.connections = []
        self.commits = 0

    def connect(self):
        connection = _FakeConnection(self)
        self.connections.append(connection)
        return connection


class _FakeConnection:
    def __init__(self, db):
        self._db = db
        self._pending = {}
        self._pending_inserts = []
        self.closed = False

    def row(self, chat_id):
        if chat_id in self._pending:
            return self._pending[chat_id]
        return self._db.history.get(chat_id)

    def cursor(self):
        return _FakeCursor(self, self._db)

    def commit(self):
        self._db.history.update(self._pending)
        self._db.history_inserts.extend(self._pending_inserts)
        self._pending = {}
        self._pending_inserts = []
        self._db.commits += 1

    def rollback(self):
        self._pending = {}
        self._pending_inserts = []

    def close(self):
        self.closed = True


class _FakeCursor:
    def __init__(self, connection, db):
        self._connection = connection
        self._db = db
        self._rows = []

    def _owned(self, chat_id, user_id):
        row = self._connection.row(chat_id)
        # SQL semantics: user_id = %s never matches a NULL owner.
        return row if row is not None and row["user_id"] is not None and row["user_id"] == user_id else None

    def execute(self, sql, params=()):
        norm = " ".join(sql.split())
        self._db.statements.append((norm, tuple(params or ())))
        self._rows = []
        if norm == "SELECT is_admin FROM users WHERE token = %s":
            user = _USERS.get(params[0])
            self._rows = [(user[1],)] if user else []
        elif norm == "SELECT id FROM users WHERE token = %s":
            user = _USERS.get(params[0])
            self._rows = [(user[0],)] if user else []
        elif norm.startswith("SELECT id, question, original_answer") and "FROM knowledge_base" in norm:
            self._rows = [_KB_ROW]
        elif norm in (_SCOPE_OWNERLESS, _SCOPE_OWNED):
            owner = None if norm == _SCOPE_OWNERLESS else params[0]
            self._rows = [
                (doc_id, filename, None, len(content))
                for doc_id, (filename, doc_owner, content) in sorted(_DOCUMENTS.items())
                if doc_owner == owner
            ]
        elif norm in (_LOAD_OWNERLESS, _LOAD_OWNED):
            owner = None if norm == _LOAD_OWNERLESS else params[1]
            document = _DOCUMENTS.get(params[0])
            self._rows = [(document[2],)] if document and document[1] == owner else []
        elif norm == _INSERT_HISTORY:
            self._connection._pending_inserts.append(tuple(params))
            self._rows = [(900 + len(self._db.history_inserts) + len(self._connection._pending_inserts),)]
        elif norm == _SELECT_OWNED_ROW:
            chat_id, user_id = params
            self._rows = [(chat_id,)] if self._owned(chat_id, user_id) else []
        elif norm in (_UPDATE_ANSWER, _UPDATE_RATING):
            value, comment, chat_id, user_id = params
            row = self._owned(chat_id, user_id)
            if row is not None:
                field = "answer" if norm == _UPDATE_ANSWER else "rating"
                self._connection._pending[chat_id] = {**row, field: value, "comment": comment}
                self._rows = [(chat_id,)]
        else:
            self._db.unexpected.append(norm)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def close(self):
        pass


def _kb_statements(db):
    return [sql for sql, _ in db.statements if "knowledge_base" in sql]


def _history_writes(db):
    return [
        sql for sql, _ in db.statements
        if sql.startswith(("UPDATE", "INSERT", "DELETE")) and "chat_history" in sql
    ]


def _assert_nothing_written(db):
    assert _history_writes(db) == [], "a refused request must issue no history write"
    assert _kb_statements(db) == [], "a refused request must issue no knowledge-base statement"
    assert db.commits == 0
    assert db.history == _INITIAL_HISTORY
    assert db.unexpected == []
    assert all(connection.closed for connection in db.connections)


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


@pytest.fixture
def db():
    return _FakeDB()


@pytest.fixture
def embedding_calls(monkeypatch, main_module):
    """Any embedding means a knowledge-base write was being prepared."""
    calls = []

    class _Tripwire:
        def embed_documents(self, texts):
            calls.append(list(texts))
            raise AssertionError("no containment route may embed anything")

        def embed_query(self, text):
            calls.append([text])
            raise AssertionError("no containment route may embed anything")

    monkeypatch.setattr(main_module, "_EMBEDDING_PROVIDER", _Tripwire())
    return calls


@pytest.fixture
def client(monkeypatch, main_module, db, embedding_calls):
    # MANDATORY PATCHING RULE: replace the initializer BEFORE the TestClient
    # is constructed, so startup never reaches psycopg2.connect.
    monkeypatch.setattr(main_module, "_initialize_application", lambda: None)
    monkeypatch.setattr(main_module, "connect_to_postgres", db.connect)
    test_client = TestClient(main_module.app)
    try:
        with test_client:
            yield test_client
    finally:
        test_client.close()


_NO_VALID_IDENTITY = pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer tok-bogus"}, {"Authorization": "tok-alice"}],
    ids=["anonymous", "invalid-token", "malformed-header"],
)

_NOT_THE_CALLERS_ROW = pytest.mark.parametrize(
    ("token", "chat_id"),
    [("tok-alice", 2), ("tok-alice", 3), ("tok-admin", 1), ("tok-alice", 999)],
    ids=["other-users-row", "ownerless-row", "admin-on-a-users-row", "missing-row"],
)


# --- GET /knowledge_base -----------------------------------------------------


_ADMIN_CHECK = "SELECT is_admin FROM users WHERE token = %s"


def _assert_kb_read_refused(response, db, status):
    assert response.status_code == status
    assert _kb_statements(db) == [], "the admin check must run before any KB query"
    for private in ("private kb answer", "private-contract.pdf", "kb question"):
        assert private not in response.text
    assert all(connection.closed for connection in db.connections)


@pytest.mark.parametrize(
    "headers",
    [
        {},
        # An admin's real token without the Bearer scheme, or under another
        # scheme, is still malformed: the header, not the token alone, counts.
        {"Authorization": "tok-admin"},
        {"Authorization": "Basic tok-admin"},
    ],
    ids=["missing", "malformed-no-scheme", "malformed-wrong-scheme"],
)
def test_knowledge_base_read_without_a_bearer_token_is_401_before_any_query(client, db, headers):
    response = client.get("/knowledge_base", headers=headers)

    _assert_kb_read_refused(response, db, 401)
    assert db.statements == [], "no usable token means no admin lookup either"


@pytest.mark.parametrize(
    "token", ["tok-bogus", "tok-alice"], ids=["invalid-token", "ordinary-user"]
)
def test_knowledge_base_read_by_a_non_admin_bearer_token_is_403(client, db, token):
    response = client.get("/knowledge_base", headers={"Authorization": f"Bearer {token}"})

    _assert_kb_read_refused(response, db, 403)
    assert db.statements == [(_ADMIN_CHECK, (token,))], (
        "the existing admin check must run with the Bearer token, and nothing after it"
    )


def test_a_query_string_token_is_no_longer_accepted(client, db):
    """Even an admin token is refused in the URL form, which leaks into browser
    history and access logs; only the Authorization header is read."""
    response = client.get("/knowledge_base", params={"token": "tok-admin"})

    _assert_kb_read_refused(response, db, 401)
    assert db.statements == []


def test_an_admin_bearer_token_reads_the_knowledge_base(client, db):
    response = client.get("/knowledge_base", headers={"Authorization": "Bearer tok-admin"})

    assert response.status_code == 200, response.text
    [entry] = response.json()
    assert entry["id"] == 11
    assert entry["answer"] == "private kb answer"
    assert entry["context_text"] == "private-contract.pdf"
    assert db.statements[0] == (_ADMIN_CHECK, ("tok-admin",)), (
        "the admin check must run first, with the Bearer token"
    )
    assert [sql.split(" ", 1)[0] for sql in _kb_statements(db)] == ["SELECT"]
    assert all(connection.closed for connection in db.connections)


# --- POST /submit_correction -------------------------------------------------


def _correct(client, chat_id, headers):
    return client.post(
        "/submit_correction",
        json={"chat_id": chat_id, "corrected_answer": "corrected text", "comment": "fix"},
        headers=headers,
    )


@_NO_VALID_IDENTITY
def test_correction_without_a_valid_identity_writes_nothing(client, db, embedding_calls, headers):
    response = _correct(client, 1, headers)

    assert response.status_code == 401
    _assert_nothing_written(db)
    assert embedding_calls == []


@_NOT_THE_CALLERS_ROW
def test_correction_of_a_row_the_caller_does_not_own_writes_nothing(
    client, db, embedding_calls, token, chat_id
):
    response = _correct(client, chat_id, {"Authorization": f"Bearer {token}"})

    assert response.status_code == 404, (
        "another user's, an ownerless, and a missing row must be indistinguishable"
    )
    _assert_nothing_written(db)
    assert embedding_calls == []


def test_an_owner_correction_updates_only_the_owners_history(client, db, embedding_calls):
    response = _correct(client, 1, {"Authorization": "Bearer tok-alice"})

    assert response.status_code == 200, response.text
    assert response.json() == {"message": "Correction saved to your history", "chat_id": 1}
    assert db.history[1] == {
        "user_id": "alice-id", "answer": "corrected text", "comment": "fix", "rating": None,
    }
    assert db.commits == 1
    assert {k: v for k, v in db.history.items() if k != 1} == {
        k: v for k, v in _INITIAL_HISTORY.items() if k != 1
    }
    scoped = [(sql, params) for sql, params in db.statements if "chat_history" in sql]
    assert scoped == [
        (_SELECT_OWNED_ROW, (1, "alice-id")),
        (_UPDATE_ANSWER, ("corrected text", "fix", 1, "alice-id")),
    ], "both the SELECT and the UPDATE must be scoped to the caller's user id"
    assert _kb_statements(db) == [] and embedding_calls == [], (
        "a correction must not be copied into the shared knowledge base"
    )
    assert db.unexpected == []


def test_an_admins_own_correction_is_not_copied_into_the_knowledge_base(
    client, db, embedding_calls
):
    response = _correct(client, 4, {"Authorization": "Bearer tok-admin"})

    assert response.status_code == 200, response.text
    assert db.history[4]["answer"] == "corrected text"
    assert _kb_statements(db) == [] and embedding_calls == []


# --- PUT /put_ratings --------------------------------------------------------


def _rate(client, chat_id, headers):
    return client.put(
        "/put_ratings",
        json={"id": chat_id, "rating": 5, "comment": ""},
        headers=headers,
    )


@_NO_VALID_IDENTITY
def test_rating_without_a_valid_identity_writes_nothing(client, db, headers):
    response = _rate(client, 1, headers)

    assert response.status_code == 401
    _assert_nothing_written(db)


@_NOT_THE_CALLERS_ROW
def test_rating_a_row_the_caller_does_not_own_writes_nothing(client, db, token, chat_id):
    response = _rate(client, chat_id, {"Authorization": f"Bearer {token}"})

    assert response.status_code == 404
    # The rating route has no separate SELECT: its one statement is the
    # owner-scoped UPDATE, which matches no row here, so nothing changes.
    caller_id = _USERS[token][0]
    assert [(sql, params) for sql, params in db.statements if "chat_history" in sql] == [
        (_UPDATE_RATING, (5, "", chat_id, caller_id)),
    ]
    assert db.commits == 0
    assert db.history == _INITIAL_HISTORY
    assert _kb_statements(db) == [] and db.unexpected == []
    assert all(connection.closed for connection in db.connections)


def test_an_owner_rating_updates_only_the_owners_history(client, db):
    response = _rate(client, 1, {"Authorization": "Bearer tok-alice"})

    assert response.status_code == 200, response.text
    assert response.json() == {"message": "Rating updated", "id": 1}
    assert db.history[1]["rating"] == 5
    assert db.commits == 1
    assert {k: v for k, v in db.history.items() if k != 1} == {
        k: v for k, v in _INITIAL_HISTORY.items() if k != 1
    }
    assert [(sql, params) for sql, params in db.statements if "chat_history" in sql] == [
        (_UPDATE_RATING, (5, "", 1, "alice-id")),
    ]
    assert _kb_statements(db) == [] and db.unexpected == []


# --- POST /ask ---------------------------------------------------------------


_ANSWER_TOKENS = ("The term ", "is stated in ", "the document [C1].")


class _FakeGeneration:
    """A streamed /api/generate response: NDJSON lines cut into small chunks."""

    def __init__(self, tokens):
        self._payload = "".join(json.dumps({"response": token}) + "\n" for token in tokens)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, decode_unicode=True, chunk_size=32):
        for index in range(0, len(self._payload), chunk_size):
            yield self._payload[index : index + chunk_size]


class _FakeVerifierReply:
    def raise_for_status(self):
        pass

    def json(self):
        return {"response": "supported"}


@pytest.fixture
def ollama_calls(monkeypatch, main_module):
    calls = []

    def post_ollama(endpoint, payload, *, stream=False, timeout=60):
        calls.append({"endpoint": endpoint, "payload": payload, "stream": stream})
        return _FakeGeneration(_ANSWER_TOKENS) if stream else _FakeVerifierReply()

    monkeypatch.setattr(main_module, "post_ollama", post_ollama)
    return calls


@pytest.fixture
def kb_helper_calls(monkeypatch, main_module):
    """Replace every knowledge-base helper with a recorder. A reader would hand
    back the sentinel, so a regression shows up both as a recorded call and
    as sentinel text in the model prompt."""
    calls = []

    def get_relevant_knowledge_base(question, limit=3):
        calls.append("get_relevant_knowledge_base")
        return [{"id": 71, "question": "KB question?", "answer": _KB_SENTINEL,
                 "context": "", "corrected_by": "User", "usage_count": 0,
                 "relevance_score": 0.99}]

    monkeypatch.setattr(main_module, "get_relevant_knowledge_base", get_relevant_knowledge_base)
    monkeypatch.setattr(main_module, "increment_kb_usage", lambda ids: calls.append("increment_kb_usage"))
    monkeypatch.setattr(main_module, "auto_save_to_kb", lambda *args: calls.append("auto_save_to_kb"))
    return calls


_NOT_ASSERTED = object()


@pytest.mark.parametrize("mode", ["document_qa", "structured_json", "model_reasoning"])
@pytest.mark.parametrize(
    ("auth_token", "own_marker", "own_filename", "history_owner"),
    [
        ("tok-alice", "ALICE-DOC", "alice-lease.txt", "alice-id"),
        (None, "OWNERLESS-DOC", "shared-notes.txt", None),
        # Which owner an invalid-token history row gets is a separate, known
        # open issue; this containment test deliberately does not pin it.
        ("tok-forged", None, None, _NOT_ASSERTED),
    ],
    ids=["authenticated", "anonymous", "invalid-token"],
)
def test_ask_neither_queries_nor_injects_the_knowledge_base(
    client, db, ollama_calls, kb_helper_calls, embedding_calls,
    mode, auth_token, own_marker, own_filename, history_owner,
):
    response = client.post(
        "/ask",
        json={
            "question": "How long is the term?",
            "model": "local-test-model",
            "auth_token": auth_token,
            "mode": mode,
        },
    )

    assert response.status_code == 200, response.text
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert events[-1] == {"type": "done", "ok": True}, "the answer still completes"

    # No knowledge base anywhere: no helper call, no SQL, no text in any
    # model request, and nothing embedded to search it with.
    assert kb_helper_calls == []
    assert _kb_statements(db) == [] and db.unexpected == []
    assert embedding_calls == []
    [generation] = [call for call in ollama_calls if call["stream"]]
    prompt = generation["payload"]["prompt"]
    assert "knowledge_base_corrections" not in prompt
    for call in ollama_calls:
        assert _KB_SENTINEL not in json.dumps(call["payload"])

    # Document evidence still works, scoped to the caller only.
    [evidence] = [event for event in events if event["type"] == "evidence"]
    if own_marker is None:
        assert evidence["citations"] == [], "an invalid token gets no document evidence"
    else:
        assert own_marker in prompt
        assert [citation["filename"] for citation in evidence["citations"]] == [own_filename]
    for marker in _DOC_MARKERS:
        if marker != own_marker:
            assert marker not in prompt, f"{marker} reached another caller's prompt"

    # The streamed answer and its history row are unchanged behavior.
    streamed = "".join(event["text"] for event in events if event["type"] == "delta")
    assert streamed == "".join(_ANSWER_TOKENS)
    assert [event["type"] for event in events].count("entry_id") == 1
    assert len(db.history_inserts) == 1
    if history_owner is not _NOT_ASSERTED:
        assert db.history_inserts[0][4] == history_owner
    assert all(connection.closed for connection in db.connections)


# --- no knowledge-base reader or automatic writer remains wired --------------


def test_no_route_or_helper_calls_a_kb_reader_or_automatic_writer():
    """Supplements the behavior tests: these helpers stay defined for the later
    user-owned knowledge base, but nothing may reference them. The admin KB
    editor uses its own endpoints and none of these helpers."""
    tree = ast.parse(_MAIN_PY_PATH.read_text(encoding="utf-8"), filename=str(_MAIN_PY_PATH))
    helpers = {
        "get_relevant_knowledge_base",
        "increment_kb_usage",
        "auto_save_to_kb",
        "generate_kb_from_document",
    }
    references = [
        (node.id, node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id in helpers
    ]
    assert references == [], f"a knowledge-base helper is wired again: {references}"


def test_this_file_contacts_no_external_service():
    this_path = Path(__file__).resolve()
    tree = ast.parse(this_path.read_text(encoding="utf-8"), filename=str(this_path))
    forbidden = {"subprocess", "docker", "requests", "psycopg2", "socket", "httpx"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert imported.isdisjoint(forbidden), sorted(imported & forbidden)
