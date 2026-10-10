"""Route-level contracts for CP2A: /version and durable answer provenance.

Drives the real FastAPI app through ``TestClient`` against an in-memory fake
database that emulates PostgreSQL's handling of the two new columns (a UUID
must parse; JSONB must be valid JSON and reads back decoded), and replaces
``post_ollama`` and the installed-model lookup with in-memory doubles. No
database, Ollama, model, network, or subprocess is reached. Every build SHA
and model digest is a synthetic test identity.

What these prove:

* /version reports exactly the injected release identity -- full, short,
  known/unknown, normalized -- and nothing internal, without touching the
  database or Ollama;
* a successful answer is saved with a server-generated trace id and
  provenance (schema 3) in the same INSERT, describing the snapshotted
  profile and the shared selection and epoch it was admitted under (the
  Llama control with no observed digest; a qualified Gemma selection with its
  observed digest), the release, prompt versions, configured embedding
  profile, server timing, and the stream's own metrics;
* configuration swapped mid-stream does not reach a running answer's record;
* failed, unfinished, empty and whitespace-only generations save no row and
  no provenance; a failed INSERT rolls back; a provenance failure saves no
  answer row at all; verification still precedes persistence;
* client-sent look-alike fields never reach the record;
* owner history keeps its shape and owner scope; only the admin chat-history
  route returns provenance, legacy rows read back as null, stored version 1
  and 2 records come back exactly as saved (and stay readable with their own
  vocabulary), and no other code path reads the column.

The generation ledger installed here is what the lifespan installs once this
process owns generation, without a database session.

REQUIRES THE SOCKETPAIR-AWARE SIBLING GUARD (see test_main_route_harness.py):
entering a TestClient builds a real asyncio loop.
"""

import ast
import contextlib
import copy
import dataclasses
import importlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import dotenv
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from answer_provenance import AnswerProvenanceError, read_answer_provenance
from generation_admission import DispatchIntents, GenerationLedger, SharedSelection
from generation_profiles import (
    GEMMA_CANDIDATE,
    LLAMA_CONTROL,
    LLAMA_CONTROL_DIGEST,
    ModelRegistry,
    Qualification,
    default_model_registry,
)
from rag_model_profiles import default_embedding_profile, explicit_unverified_embedding_profile
from release_metadata import API_CONTRACT_VERSION, APP_VERSION, release_metadata_from_env

_MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"

_EMBEDDING_PROFILE_KEYS = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_QUERY_PREFIX",
    "EMBEDDING_DOCUMENT_PREFIX",
    "EMBEDDING_PROFILE_UNVERIFIED_ACK",
)

SYNTHETIC_SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER_SYNTHETIC_SHA = "fedcba9876543210fedcba9876543210fedcba98"
SYNTHETIC_DIGEST = "5e" * 32

_ACTIVE_ID = "SELECT id FROM users WHERE token = %s AND COALESCE(is_active, 1) <> 0"
_ACTIVE_ADMIN = "SELECT is_admin FROM users WHERE token = %s AND COALESCE(is_active, 1) <> 0"
_SCOPE_OWNED = (
    "SELECT id, filename, title, length(content) FROM documents "
    "WHERE user_id = %s ORDER BY id"
)
_LOAD_OWNED = "SELECT content FROM documents WHERE id = %s AND user_id = %s"
_INSERT_HISTORY = (
    "INSERT INTO chat_history (ts, selected_text, question, answer, user_id, trace_id, answer_provenance) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id"
)
_LIST_HISTORY = (
    "SELECT id, ts, selected_text, question, answer FROM chat_history "
    "WHERE user_id = %s ORDER BY id DESC LIMIT 20"
)
_ADMIN_COUNT = "SELECT COUNT(*) FROM chat_history ch"
_ADMIN_LIST = (
    "SELECT ch.id, ch.user_id, u.username, ch.ts, ch.question, ch.answer, ch.rating, ch.comment, "
    "ch.trace_id, ch.answer_provenance FROM chat_history ch LEFT JOIN users u ON ch.user_id = u.id "
    "ORDER BY ch.id DESC LIMIT %s OFFSET %s"
)

# token -> (user id, is_admin)
_USERS = {"tok-admin": ("admin-id", 1), "tok-alice": ("alice-id", 0), "tok-bob": ("bob-id", 0)}
_DOCUMENTS = {
    10: ("alice-lease.txt", "alice-id", "ALICE-DOC: the lease term is two years."),
    40: ("admin-memo.txt", "admin-id", "ADMIN-DOC: the notice period is thirty days."),
}

# Rows saved before CP2A: no trace id, no provenance.
_LEGACY_ROWS = {
    1: dict(user_id="alice-id", ts="2026-09-01T10:00:00", selected_text="", question="legacy alice question",
            answer="legacy alice answer", rating=4, comment="ok", trace_id=None, answer_provenance=None),
    2: dict(user_id="bob-id", ts="2026-09-02T10:00:00", selected_text="", question="legacy bob question",
            answer="legacy bob answer", rating=None, comment=None, trace_id=None, answer_provenance=None),
}


class _FakeDB:
    def __init__(self, log):
        self.rows = copy.deepcopy(_LEGACY_ROWS)
        self.log = log
        self.statements = []
        self.unexpected = []
        self.commits = 0
        self.rollbacks = 0
        self.fail_insert = False

    def connect(self):
        return _FakeConnection(self)


class _FakeConnection:
    def __init__(self, db):
        self._db = db
        self._pending = {}

    def cursor(self):
        return _FakeCursor(self, self._db)

    def commit(self):
        self._db.rows.update(self._pending)
        self._pending = {}
        self._db.commits += 1

    def rollback(self):
        self._pending = {}
        self._db.rollbacks += 1

    def close(self):
        pass


class _FakeCursor:
    def __init__(self, connection, db):
        self._connection = connection
        self._db = db
        self._rows = []

    def execute(self, sql, params=()):
        norm = " ".join(sql.split())
        params = tuple(params or ())
        db = self._db
        db.statements.append(norm)
        db.log.append(("sql", norm))
        self._rows = []
        user = _USERS.get(params[0]) if params and isinstance(params[0], str) else None
        if norm == _ACTIVE_ID:
            self._rows = [(user[0],)] if user else []
        elif norm == _ACTIVE_ADMIN:
            self._rows = [(user[1],)] if user else []
        elif norm == _SCOPE_OWNED:
            self._rows = [
                (doc_id, name, None, len(content))
                for doc_id, (name, owner, content) in sorted(_DOCUMENTS.items())
                if owner == params[0]
            ]
        elif norm == _LOAD_OWNED:
            document = _DOCUMENTS.get(params[0])
            self._rows = [(document[2],)] if document and document[1] == params[1] else []
        elif norm == _INSERT_HISTORY:
            if db.fail_insert:
                raise RuntimeError("psycopg2 insert failed: password=hunter2 host=db.internal")
            ts, selected, question, answer, user_id, trace_id, provenance = params
            # What PostgreSQL would do with these parameters: a UUID column
            # rejects anything that is not a UUID, JSONB rejects invalid JSON,
            # and psycopg2 hands JSONB back decoded.
            uuid.UUID(trace_id)
            decoded = json.loads(provenance)
            entry_id = 700 + len(db.rows) + len(self._connection._pending) + 1
            self._connection._pending[entry_id] = dict(
                user_id=user_id, ts=ts, selected_text=selected, question=question, answer=answer,
                rating=None, comment=None, trace_id=trace_id, answer_provenance=decoded,
                raw_provenance=provenance,
            )
            self._rows = [(entry_id,)]
        elif norm == _LIST_HISTORY:
            self._rows = [
                (row_id, row["ts"], row["selected_text"], row["question"], row["answer"])
                for row_id, row in sorted(db.rows.items(), reverse=True)
                if row["user_id"] is not None and row["user_id"] == params[0]
            ]
        elif norm == _ADMIN_COUNT:
            self._rows = [(len(db.rows),)]
        elif norm == _ADMIN_LIST:
            limit, offset = params
            usernames = {user_id: user_id.split("-")[0] for user_id, _ in _USERS.values()}
            self._rows = [
                (row_id, row["user_id"], usernames.get(row["user_id"]), row["ts"], row["question"],
                 row["answer"], row["rating"], row["comment"], row["trace_id"],
                 copy.deepcopy(row["answer_provenance"]))
                for row_id, row in sorted(db.rows.items(), reverse=True)
            ][offset:offset + limit]
        else:
            db.unexpected.append(norm)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def close(self):
        pass


def _ndjson(*records):
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


# Ollama reports these two counters here; it reports no durations, so those
# must be stored as null, never as 0.
_DONE = {"response": "", "done": True, "done_reason": "stop", "prompt_eval_count": 900, "eval_count": 12}
_ANSWER = "The lease term is two years [C1]."


class _FakeStream:
    def __init__(self, payload, chunk_size=7):
        data = payload.encode("utf-8")
        self._chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, decode_unicode=True, chunk_size=32):
        yield from self._chunks


class _FakeReply:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


class _Ollama:
    def __init__(self, log):
        self.log = log
        self.calls = []
        self.stream_payload = _ndjson({"response": _ANSWER}, _DONE)
        self.verifier_body = {"response": "supported", "done": True, "done_reason": "stop"}

    def __call__(self, endpoint, payload, *, stream=False, timeout=60):
        self.calls.append({"payload": payload, "stream": stream, "timeout": timeout})
        self.log.append(("ollama", "generate" if stream else "verify"))
        if stream:
            return _FakeStream(self.stream_payload)
        return _FakeReply(self.verifier_body)

    @property
    def verifications(self):
        return [call for call in self.calls if not call["stream"]]


@contextlib.contextmanager
def _imported_main():
    previous = sys.modules.pop("main", None)
    try:
        yield importlib.import_module("main")
    finally:
        sys.modules.pop("main", None)
        if previous is not None:
            sys.modules["main"] = previous


def _neutral_environment(monkeypatch, build_sha):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    for key in _EMBEDDING_PROFILE_KEYS:
        monkeypatch.delenv(key, raising=False)
    if build_sha is None:
        monkeypatch.delenv("SYNERGE_BUILD_SHA", raising=False)
    else:
        monkeypatch.setenv("SYNERGE_BUILD_SHA", build_sha)


@pytest.fixture
def log():
    return []


@pytest.fixture
def main_module(monkeypatch):
    # The release operator's injected value, read once at import as in production.
    _neutral_environment(monkeypatch, SYNTHETIC_SHA)
    with _imported_main() as module:
        yield module


class _NoIntents(DispatchIntents):
    def record_dispatch(self, intent):
        pass

    def record_outcome(self, intent, state):
        pass


def _ready_generation(monkeypatch, main_module, profile_id=LLAMA_CONTROL, epoch=1):
    """Install a generation ledger this process owns, admitting under the
    shared selection (profile_id, epoch)."""
    ledger = GenerationLedger(main_module._ADMISSION_SETTINGS, clock=time.monotonic)
    assert ledger.install_owner(intents=_NoIntents(), owner_seq=1, selection=SharedSelection(profile_id, epoch),
                                paused_by=None, recovered={})
    monkeypatch.setattr(main_module, "_GENERATION_LEDGER", ledger)
    return ledger


@pytest.fixture(autouse=True)
def generation(monkeypatch, main_module):
    return _ready_generation(monkeypatch, main_module)


@pytest.fixture
def db(log):
    return _FakeDB(log)


@pytest.fixture
def ollama(monkeypatch, main_module, log):
    double = _Ollama(log)
    monkeypatch.setattr(main_module, "post_ollama", double)
    return double


@pytest.fixture
def inventory(monkeypatch, main_module):
    state = {"digest": SYNTHETIC_DIGEST, "calls": []}

    def installed_model_digest(tag):
        state["calls"].append(tag)
        return state["digest"]

    monkeypatch.setattr(main_module, "_installed_model_digest", installed_model_digest)
    return state


class _NoEmbedding:
    def embed_query(self, text):
        raise AssertionError("these requests are answered from complete documents")

    def embed_documents(self, texts):
        raise AssertionError("these requests are answered from complete documents")


@contextlib.contextmanager
def _client_for(monkeypatch, module, db):
    monkeypatch.setattr(module, "_initialize_application", lambda: None)
    monkeypatch.setattr(module, "connect_to_postgres", db.connect)
    monkeypatch.setattr(module, "_EMBEDDING_PROVIDER", _NoEmbedding())
    test_client = TestClient(module.app)
    try:
        with test_client:
            yield test_client
    finally:
        test_client.close()


@pytest.fixture
def client(monkeypatch, main_module, db, ollama, inventory):
    with _client_for(monkeypatch, main_module, db) as test_client:
        yield test_client


def _events(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def _ask(client, token="tok-alice", **fields):
    body = {"question": "How long is the lease term?", "mode": "document_qa", "auth_token": token}
    body.update(fields)
    return client.post("/ask", json=body)


def _saved(db):
    """Rows this test's answers saved (legacy rows excluded)."""
    return {row_id: row for row_id, row in db.rows.items() if row_id not in _LEGACY_ROWS}


def _only_saved(db):
    [(entry_id, row)] = _saved(db).items()
    return entry_id, row


def _qualified_gemma_registry(default=LLAMA_CONTROL):
    base = default_model_registry()
    gemma = dataclasses.replace(
        base.get(GEMMA_CANDIDATE),
        expected_digest=SYNTHETIC_DIGEST,
        qualification=Qualification(
            think_false_verified=True, combined_capacity_verified=True,
            quality_verified=True, evidence_ref="synthetic test identity",
        ),
    )
    return ModelRegistry(
        profiles=(base.get(LLAMA_CONTROL), gemma),
        default_profile_id=default,
    )


# --- /version -----------------------------------------------------------------

_VERSION_KEYS = {"app_version", "build_sha", "build_sha_short", "build_known", "display_version"}


def _version_response(monkeypatch, build_sha):
    _neutral_environment(monkeypatch, build_sha)
    db = _FakeDB([])
    with _imported_main() as module, _client_for(monkeypatch, module, db) as test_client:
        response = test_client.get("/version")
    return response, db


@pytest.mark.parametrize(
    "injected",
    [SYNTHETIC_SHA, SYNTHETIC_SHA.upper() + "\n"],
    ids=["full-sha", "uppercase-with-newline"],
)
def test_version_reports_the_injected_build(monkeypatch, injected):
    response, db = _version_response(monkeypatch, injected)

    assert response.status_code == 200
    assert response.json() == {
        "app_version": "0.1.0",
        "build_sha": SYNTHETIC_SHA,
        "build_sha_short": "0123456",
        "build_known": True,
        "display_version": "v0.1.0 · 0123456",
    }
    assert response.headers["cache-control"] == "no-store"
    assert db.statements == [], "the version endpoint touches no database"


@pytest.mark.parametrize("injected", [None, "", "c236c6e", "not-a-sha", "ab" * 32], ids=[
    "missing", "empty", "abbreviated", "garbage", "sha256-length",
])
def test_version_reports_an_unknown_build_rather_than_guessing(monkeypatch, injected):
    response, _ = _version_response(monkeypatch, injected)

    assert response.status_code == 200
    assert response.json() == {
        "app_version": "0.1.0",
        "build_sha": None,
        "build_sha_short": None,
        "build_known": False,
        "display_version": "v0.1.0 · build unknown",
    }


def test_openapi_keeps_the_api_contract_version_while_version_reports_the_release(
    main_module, client, db, ollama, inventory
):
    assert API_CONTRACT_VERSION == "2.0.0" and APP_VERSION == "0.1.0"
    assert main_module.app.version == API_CONTRACT_VERSION
    assert client.get("/openapi.json").json()["info"]["version"] == API_CONTRACT_VERSION

    response = client.get("/version")

    assert response.status_code == 200
    assert response.json()["app_version"] == APP_VERSION
    assert main_module.app.version != response.json()["app_version"], (
        "the API contract version and the application release are separate identities"
    )
    assert db.statements == [] and ollama.calls == [] and inventory["calls"] == []


def test_version_is_public_and_exposes_nothing_internal(client, db, ollama, inventory):
    response = client.get("/version")

    assert response.status_code == 200
    assert set(response.json()) == _VERSION_KEYS
    lowered = response.text.lower()
    for internal in ("llama", "gemma", "qwen", "profile", "ollama", "11434", "postgres", "http",
                     "/app", "\\\\", "embed", "digest", "token", "rag-answer", "claim-support"):
        assert internal not in lowered, internal
    assert db.statements == [] and ollama.calls == [] and inventory["calls"] == []


# --- provenance on a successful answer ----------------------------------------


def test_a_shared_llama_answer_is_saved_with_server_provenance(client, db, ollama):
    response = _ask(client, "tok-alice", active_document_id=10)

    events = _events(response)
    assert events[-1] == {"type": "done", "ok": True}
    entry_id, row = _only_saved(db)
    assert events[-2] == {"type": "entry_id", "entry_id": entry_id}, "entry_id behaviour is unchanged"
    assert row["user_id"] == "alice-id" and row["answer"] == _ANSWER

    trace_id = row["trace_id"]
    assert str(uuid.UUID(trace_id)) == trace_id and uuid.UUID(trace_id).version == 4
    provenance = row["answer_provenance"]
    assert provenance["schema_version"] == 3
    assert provenance["trace_id"] == trace_id, "the column and the record name the same attempt"
    assert provenance["release"] == {"app_version": "0.1.0", "build_sha": SYNTHETIC_SHA}
    assert provenance["generation"] == {
        "profile_id": "llama_control",
        "model_tag": "llama3.1:8b",
        "selection": "shared_active",
        "selection_epoch": 1,
        "expected_digest": LLAMA_CONTROL_DIGEST,
        "observed_digest": None,
        "requested_options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1200},
        "think": None,
        "keep_alive": "5m",
        "answer_read_timeout_s": 60.0,
    }
    assert provenance["prompt"] == {
        "answer_prompt_version": "rag-answer-v1",
        "answer_mode": "document_qa",
        "claim_support_prompt_version": "claim-support-v1",
    }
    assert provenance["verification"] == {
        "profile_id": "llama_control",
        "model_tag": "llama3.1:8b",
        "expected_digest": LLAMA_CONTROL_DIGEST,
        "observed_digest": None,
        "requested_options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 16},
        "think": None,
        "keep_alive": "5m",
        "read_timeout_s": 30.0,
        "calls_attempted": len(ollama.verifications),
    }
    assert len(ollama.verifications) == 1
    embedding = default_embedding_profile()
    assert provenance["retrieval"] == {
        "evidence_mode": "complete_document",
        "configured_embedding_profile": {
            "profile_id": embedding.profile_id, "provider": "ollama",
            "model": "mxbai-embed-large:335m", "dimension": 1024,
        },
    }
    assert provenance["ollama_metrics"] == {
        "done_reason": "stop", "prompt_eval_count": 900, "eval_count": 12,
        "load_duration_ns": None, "prompt_eval_duration_ns": None,
        "eval_duration_ns": None, "total_duration_ns": None,
    }
    timing = provenance["timing"]
    assert timing["origin"] == "ask_request_received"
    marks = [timing[f"{name}_ms"] for name in (
        "generation_request_sent", "first_answer_token", "generation_complete", "verification_complete")]
    assert all(isinstance(value, int) and value >= 0 for value in marks)
    assert marks == sorted(marks), "milestones are monotonic"


def test_the_answer_stream_carries_no_provenance(client, db):
    response = _ask(client, "tok-alice", active_document_id=10)

    _, row = _only_saved(db)
    assert row["trace_id"] not in response.text
    for leaked in ("answer_provenance", "trace_id", "rag-answer", SYNTHETIC_SHA, LLAMA_CONTROL_DIGEST):
        assert leaked not in response.text, leaked
    [evidence] = [e for e in _events(response) if e["type"] == "evidence"]
    assert evidence["generation"] == {"profile_id": "llama_control", "model": "llama3.1:8b"}, (
        "the answer UI keeps exactly the model label it already had"
    )


def test_a_shared_gemma_selection_records_its_epoch_and_observed_digest(
    monkeypatch, main_module, client, db, ollama, inventory
):
    monkeypatch.setattr(main_module, "_MODEL_REGISTRY", _qualified_gemma_registry())
    _ready_generation(monkeypatch, main_module, GEMMA_CANDIDATE, 3)
    # The inventory reports the digest in another spelling; the record keeps
    # the normalized observation, separately from the registry's pin.
    inventory["digest"] = "sha256:" + SYNTHETIC_DIGEST.upper()

    response = _ask(client, "tok-admin", active_document_id=40)

    assert _events(response)[-1] == {"type": "done", "ok": True}
    _, row = _only_saved(db)
    generation = row["answer_provenance"]["generation"]
    assert generation["profile_id"] == "gemma_candidate"
    assert generation["model_tag"] == "gemma4:31b"
    assert (generation["selection"], generation["selection_epoch"]) == ("shared_active", 3)
    assert generation["expected_digest"] == SYNTHETIC_DIGEST
    assert generation["observed_digest"] == SYNTHETIC_DIGEST
    assert generation["think"] is False
    assert generation["answer_read_timeout_s"] == 150.0
    assert inventory["calls"] == ["gemma4:31b"]
    verification = row["answer_provenance"]["verification"]
    assert verification == {
        "profile_id": "gemma_candidate",
        "model_tag": "gemma4:31b",
        "expected_digest": SYNTHETIC_DIGEST,
        "observed_digest": SYNTHETIC_DIGEST,
        "requested_options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 16},
        "think": False,
        "keep_alive": "5m",
        "read_timeout_s": 150.0,
        "calls_attempted": 1,
    }, "the record matches the verifier call that actually ran"
    [verifier_call] = [call for call in ollama.calls if not call["stream"]]
    assert verifier_call["payload"]["model"] == verification["model_tag"]
    assert verifier_call["payload"]["options"] == verification["requested_options"]
    assert verifier_call["payload"]["think"] is verification["think"]
    assert verifier_call["timeout"] == verification["read_timeout_s"]


def test_each_answer_attempt_gets_its_own_trace_id(client, db):
    _ask(client, "tok-alice", active_document_id=10)
    _ask(client, "tok-alice", active_document_id=10)

    trace_ids = [row["trace_id"] for row in _saved(db).values()]
    assert len(trace_ids) == 2 and len(set(trace_ids)) == 2


def test_the_record_describes_the_snapshot_even_if_configuration_changes_mid_stream(
    monkeypatch, main_module, client, db
):
    real_scope = main_module._resolve_authorized_scope
    swapped_embedding = explicit_unverified_embedding_profile(
        provider="ollama", model="nomic-embed-text:v1.5", dimension=768, query_prefix="", document_prefix="",
    )

    def scope_then_swap(token):
        # Runs inside the stream, after the snapshot and before generation.
        monkeypatch.setattr(main_module, "_MODEL_REGISTRY", _qualified_gemma_registry(default=GEMMA_CANDIDATE))
        monkeypatch.setattr(main_module, "_RELEASE", release_metadata_from_env({"SYNERGE_BUILD_SHA": OTHER_SYNTHETIC_SHA}))
        monkeypatch.setattr(main_module, "_EMBEDDING_PROFILE", swapped_embedding)
        return real_scope(token)

    monkeypatch.setattr(main_module, "_resolve_authorized_scope", scope_then_swap)

    _ask(client, "tok-alice", active_document_id=10)

    assert main_module._RELEASE.build_sha == OTHER_SYNTHETIC_SHA, "the swap really happened mid-stream"
    _, row = _only_saved(db)
    provenance = row["answer_provenance"]
    assert provenance["release"]["build_sha"] == SYNTHETIC_SHA
    assert provenance["generation"]["model_tag"] == "llama3.1:8b"
    assert (provenance["generation"]["selection"], provenance["generation"]["selection_epoch"]) == (
        "shared_active", 1,
    )
    assert provenance["retrieval"]["configured_embedding_profile"]["model"] == "mxbai-embed-large:335m"


# --- nothing is saved for an unfinished answer --------------------------------

_FAILING_STREAMS = {
    "thinking-only": _ndjson({"thinking": "I think it is two years.", "response": ""}, _DONE),
    "empty": _ndjson(_DONE),
    "whitespace-only": _ndjson({"response": " "}, {"response": "\n"}, {"response": "\t\r"}, _DONE),
    "output-limit": _ndjson({"response": "The lease term is"}, dict(_DONE, done_reason="length")),
    "model-error": _ndjson({"response": "The lease "}, {"error": "runner crashed: /srv/models/blob"}),
    "no-final-record": _ndjson({"response": "The lease term is two years."}),
    "malformed-line": _ndjson({"response": "The lease "}) + '{"response": "te\n' + _ndjson(_DONE),
    # The final record arrived after a full answer, but only "stop" says the
    # model finished it.
    "done-reason-missing": _ndjson({"response": _ANSWER}, {k: v for k, v in _DONE.items() if k != "done_reason"}),
    "done-reason-non-string": _ndjson({"response": _ANSWER}, dict(_DONE, done_reason=7)),
    "done-reason-load": _ndjson({"response": _ANSWER}, dict(_DONE, done_reason="load")),
    "done-reason-unknown": _ndjson({"response": _ANSWER}, dict(_DONE, done_reason="something_new")),
}


@pytest.mark.parametrize("case", sorted(_FAILING_STREAMS))
def test_an_unfinished_or_empty_generation_saves_no_row_and_no_provenance(client, db, ollama, case):
    ollama.stream_payload = _FAILING_STREAMS[case]

    response = _ask(client, "tok-alice", active_document_id=10)

    events = _events(response)
    assert events[-1] == {"type": "done", "ok": False}
    assert not [e for e in events if e["type"] in ("verification", "entry_id")]
    assert _INSERT_HISTORY not in db.statements and db.commits == 0
    assert db.rows == _LEGACY_ROWS
    assert ollama.verifications == []
    assert db.unexpected == []


def test_a_failed_insert_rolls_back_and_saves_nothing(client, db):
    db.fail_insert = True

    response = _ask(client, "tok-alice", active_document_id=10)

    events = _events(response)
    assert [e["type"] for e in events][-2:] == ["error", "done"]
    assert events[-2]["code"] == "internal_error" and events[-1] == {"type": "done", "ok": False}
    assert not [e for e in events if e["type"] == "entry_id"]
    assert db.commits == 0 and db.rollbacks == 1
    assert db.rows == _LEGACY_ROWS
    assert "hunter2" not in response.text and "db.internal" not in response.text


@pytest.mark.parametrize("step", ["build_answer_provenance", "answer_provenance_json"])
def test_a_provenance_failure_saves_no_answer_row(monkeypatch, main_module, client, db, ollama, log, step):
    def refuse(*args, **kwargs):
        raise AnswerProvenanceError("simulated provenance failure")

    monkeypatch.setattr(main_module, step, refuse)

    response = _ask(client, "tok-alice", active_document_id=10)

    events = _events(response)
    assert [e for e in events if e["type"] == "verification"], "verification still ran first"
    assert events[-2]["code"] == "internal_error" and events[-1] == {"type": "done", "ok": False}
    assert _INSERT_HISTORY not in db.statements, "no answer row without its provenance"
    assert db.commits == 0 and db.rows == _LEGACY_ROWS
    verified_at = log.index(("ollama", "verify"))
    assert [entry for entry in log[verified_at:] if entry[0] == "sql"] == [], (
        "the failure is caught before any persistence statement"
    )
    assert "simulated provenance failure" not in response.text


def test_verification_still_precedes_persistence(client, db, log):
    response = _ask(client, "tok-alice", active_document_id=10)

    types = [e["type"] for e in _events(response)]
    assert types.index("evidence") < types.index("verification") < types.index("entry_id") < types.index("done")
    insert_at = log.index(("sql", _INSERT_HISTORY))
    assert [entry for entry in log[:insert_at] if entry[0] == "ollama"] == [
        ("ollama", "generate"), ("ollama", "verify"),
    ], "generation, then verification, then the single INSERT"
    assert [entry for entry in log[insert_at:] if entry[0] == "ollama"] == []


# --- client spoofing ----------------------------------------------------------

_SPOOFED_FIELDS = {
    "trace_id": "11111111-1111-4111-8111-111111111111",
    "answer_provenance": {"schema_version": 1, "generation": {"model_tag": "spoofed:latest"}},
    "provenance": {"spoofed": True},
    "model": "gemma4:31b",
    "model_tag": "spoofed:latest",
    "model_digest": "f" * 64,
    "expected_digest": "f" * 64,
    "observed_digest": "f" * 64,
    "options": {"temperature": 1.9, "num_ctx": 32768, "num_predict": 4000},
    "temperature": 1.9,
    "num_ctx": 32768,
    "num_predict": 4000,
    "think": True,
    "prompt_version": "spoofed-prompt-v9",
    "answer_prompt_version": "spoofed-prompt-v9",
    "embedding_profile": "spoofed-embedder",
    "embedding_model": "spoofed-embedder",
    "app_version": "9.9.9",
    "build_sha": "f" * 40,
    "timing": {"first_answer_token_ms": 1},
    "time_to_first_token_ms": 1,
    "selection": "admin_override",
    "assignment": "admin_override",
    "is_admin": True,
}


def _comparable(provenance):
    """A record minus the two values that differ per attempt by design."""
    return {key: value for key, value in provenance.items() if key not in ("trace_id", "timing")}


def test_client_look_alike_fields_never_reach_the_record(client, db):
    clean = _ask(client, "tok-alice", active_document_id=10)
    spoofed = _ask(client, "tok-alice", active_document_id=10, **_SPOOFED_FIELDS)

    assert clean.status_code == spoofed.status_code == 200
    (_, clean_row), (_, spoofed_row) = sorted(_saved(db).items())
    assert _comparable(spoofed_row["answer_provenance"]) == _comparable(clean_row["answer_provenance"])
    assert spoofed_row["trace_id"] != _SPOOFED_FIELDS["trace_id"]
    assert spoofed_row["answer_provenance"]["trace_id"] == spoofed_row["trace_id"]
    assert spoofed_row["answer_provenance"]["generation"]["selection"] == "shared_active"
    # The random server trace id is removed first so it cannot match by chance.
    stored = spoofed_row["raw_provenance"].replace(spoofed_row["trace_id"], "")
    for value in ("spoofed", "9.9.9", "f" * 40, "11111111-1111-4111-8111-111111111111", "gemma4", "1.9", "32768"):
        assert value not in stored, value


# --- exposure: owner history, admin history, legacy rows ----------------------


def test_owner_history_keeps_its_shape_and_owner_scope(client, db):
    _ask(client, "tok-alice", active_document_id=10)

    response = client.post("/history", json={"token": "tok-alice"})

    assert response.status_code == 200
    items = response.json()
    assert [item["question"] for item in items] == ["How long is the lease term?", "legacy alice question"]
    for item in items:
        assert set(item) == {"id", "timestamp", "selected_text", "question", "answer"}, (
            "owner history returns exactly what it did before CP2A"
        )
    assert "legacy bob" not in response.text
    for leaked in ("trace_id", "provenance", "llama", SYNTHETIC_SHA):
        assert leaked not in response.text


def test_admin_chat_history_returns_provenance_and_null_for_legacy_rows(client, db):
    _ask(client, "tok-alice", active_document_id=10)
    entry_id, row = _only_saved(db)

    response = client.get("/admin/chat_history", params={"token": "tok-admin"})

    assert response.status_code == 200
    items = {item["id"]: item for item in response.json()["items"]}
    assert set(items) == {entry_id, 1, 2}
    assert items[entry_id]["trace_id"] == row["trace_id"]
    assert items[entry_id]["answer_provenance"] == row["answer_provenance"]
    for legacy_id, legacy in _LEGACY_ROWS.items():
        item = items[legacy_id]
        assert item["trace_id"] is None and item["answer_provenance"] is None, (
            "a legacy row is 'not recorded' -- never filled with current values"
        )
        assert (item["question"], item["answer"], item["rating"], item["comment"]) == (
            legacy["question"], legacy["answer"], legacy["rating"], legacy["comment"],
        )
    assert db.rows[1] == _LEGACY_ROWS[1] and db.rows[2] == _LEGACY_ROWS[2], "legacy rows unmodified"


# A row saved by the schema-version-1 writer (synthetic values): its
# verification block has no digest, think or read-timeout field.
_V1_TRACE_ID = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
_V1_ROW = dict(
    user_id="alice-id", ts="2026-10-05T10:00:00", selected_text="", question="v1 alice question",
    answer="v1 alice answer", rating=None, comment=None, trace_id=_V1_TRACE_ID,
    answer_provenance={
        "schema_version": 1,
        "trace_id": _V1_TRACE_ID,
        "release": {"app_version": "0.1.0", "build_sha": None},
        "generation": {
            "profile_id": "llama_control", "model_tag": "llama3.1:8b", "selection": "default",
            "expected_digest": LLAMA_CONTROL_DIGEST, "observed_digest": None,
            "requested_options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1200},
            "think": None, "keep_alive": "5m", "answer_read_timeout_s": 60.0,
        },
        "verification": {
            "profile_id": "llama_control", "model_tag": "llama3.1:8b",
            "requested_options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 16},
            "keep_alive": "5m", "calls_attempted": 1,
        },
    },
)


# A row saved by the schema-version-2 writer under the retired per-request
# override (synthetic values).
_V2_TRACE_ID = "9b2f5d1e-3c4a-4e8b-9f10-2a3b4c5d6e7f"
_V2_ROW = dict(
    user_id="admin-id", ts="2026-10-08T10:00:00", selected_text="", question="v2 admin question",
    answer="v2 admin answer", rating=None, comment=None, trace_id=_V2_TRACE_ID,
    answer_provenance={
        "schema_version": 2,
        "trace_id": _V2_TRACE_ID,
        "release": {"app_version": "0.1.0", "build_sha": None},
        "generation": {
            "profile_id": "gemma_candidate", "model_tag": "gemma4:31b", "selection": "admin_override",
            "expected_digest": SYNTHETIC_DIGEST, "observed_digest": SYNTHETIC_DIGEST,
            "requested_options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1200},
            "think": False, "keep_alive": "5m", "answer_read_timeout_s": 150.0,
        },
        "verification": {
            "profile_id": "gemma_candidate", "model_tag": "gemma4:31b", "expected_digest": SYNTHETIC_DIGEST,
            "observed_digest": SYNTHETIC_DIGEST,
            "requested_options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 16},
            "think": False, "keep_alive": "5m", "read_timeout_s": 150.0, "calls_attempted": 2,
        },
    },
)


def test_admin_history_returns_stored_v1_and_v2_records_exactly_as_saved(client, db):
    db.rows[3] = copy.deepcopy(_V1_ROW)
    db.rows[4] = copy.deepcopy(_V2_ROW)
    _ask(client, "tok-alice", active_document_id=10)
    [new_id] = [row_id for row_id in _saved(db) if row_id not in (3, 4)]

    response = client.get("/admin/chat_history", params={"token": "tok-admin"})

    assert response.status_code == 200
    items = {item["id"]: item for item in response.json()["items"]}
    assert items[3]["answer_provenance"] == _V1_ROW["answer_provenance"], "returned exactly as stored"
    assert items[4]["answer_provenance"] == _V2_ROW["answer_provenance"], "returned exactly as stored"
    assert (items[3]["trace_id"], items[4]["trace_id"]) == (_V1_TRACE_ID, _V2_TRACE_ID)
    stored_v1 = items[3]["answer_provenance"]["verification"]
    for added in ("expected_digest", "observed_digest", "think", "read_timeout_s"):
        assert added not in stored_v1, f"{added} must not be synthesized for a version 1 row"
    for stored_id in (3, 4):
        assert "selection_epoch" not in items[stored_id]["answer_provenance"]["generation"]
    assert db.rows[3] == _V1_ROW and db.rows[4] == _V2_ROW, "stored version 1/2 rows are never rewritten"
    # Each stored version stays readable with its own selection vocabulary.
    assert read_answer_provenance(items[3]["answer_provenance"])["selection"] == "default"
    assert read_answer_provenance(items[4]["answer_provenance"])["selection"] == "admin_override"
    current = items[new_id]["answer_provenance"]
    assert current["schema_version"] == 3 and current["verification"]["read_timeout_s"] == 30.0
    assert read_answer_provenance(current)["selection_epoch"] == 1


def test_the_new_verifier_internals_reach_neither_the_stream_nor_owner_history(
    monkeypatch, main_module, client, db, inventory
):
    monkeypatch.setattr(main_module, "_MODEL_REGISTRY", _qualified_gemma_registry())
    _ready_generation(monkeypatch, main_module, GEMMA_CANDIDATE, 3)
    response = _ask(client, "tok-admin", active_document_id=40)
    assert _events(response)[-1] == {"type": "done", "ok": True}
    _, row = _only_saved(db)
    assert row["answer_provenance"]["verification"]["observed_digest"] == SYNTHETIC_DIGEST

    history = client.post("/history", json={"token": "tok-admin"})

    assert history.status_code == 200
    for text in (response.text, history.text):
        for leaked in (SYNTHETIC_DIGEST, "read_timeout_s", "expected_digest", "observed_digest", "keep_alive"):
            assert leaked not in text, leaked
    [evidence] = [e for e in _events(response) if e["type"] == "evidence"]
    assert evidence["generation"] == {"profile_id": "gemma_candidate", "model": "gemma4:31b"}


@pytest.mark.parametrize("token, status", [(None, 401), ("tok-alice", 403)], ids=["anonymous", "ordinary-user"])
def test_only_an_admin_can_read_provenance(client, db, token, status):
    _ask(client, "tok-alice", active_document_id=10)

    params = {"token": token} if token else {}
    response = client.get("/admin/chat_history", params=params)

    assert response.status_code == status
    assert "answer_provenance" not in response.text and "llama" not in response.text


def test_no_other_code_path_reads_or_writes_the_provenance_column():
    tree = ast.parse(_MAIN_PATH.read_text(encoding="utf-8"))
    owners = set()
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and (
                "answer_provenance" in node.value or "trace_id" in node.value.replace("[Ask] trace_id=", "")
            ):
                owners.add(function.name)
    # ask_question's nested stream_generate holds the INSERT.
    assert owners <= {"ask_question", "stream_generate", "admin_chat_history"}, owners
    assert {"stream_generate", "admin_chat_history"} <= owners
