"""Route-level contracts for backend answer-model resolution on POST /ask.

Drives the real FastAPI app through ``TestClient`` against an in-memory fake
database, and replaces ``post_ollama`` with an in-memory double, so no
database, Ollama, network, or subprocess is reached. Model digests are
synthetic test identities, never production values.

What these prove: every frontend /ask caller shape resolves the backend
default and its legacy model string never reaches a request; an explicit
profile is refused for anonymous, ordinary, forged-role and suspended callers
before any evidence is read or model called; a trusted admin may choose only a
registered, qualified profile whose installed digest matches; resolution is
fixed before streaming; generation options are native and bounded; separate
thinking, model errors, malformed or unfinished streams and the output limit
never produce a persisted success; verification uses the fixed Llama call;
background helpers no longer probe models and are bounded.

REQUIRES THE SOCKETPAIR-AWARE SIBLING GUARD (see test_main_route_harness.py):
entering a TestClient builds a real asyncio loop.
"""

import contextlib
import dataclasses
import importlib
import json
import os
import sys

import dotenv
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation_profiles import (
    GEMMA_CANDIDATE,
    LLAMA_CONTROL,
    ModelRegistry,
    Qualification,
    default_model_registry,
)

_EMBEDDING_PROFILE_KEYS = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_QUERY_PREFIX",
    "EMBEDDING_DOCUMENT_PREFIX",
    "EMBEDDING_PROFILE_UNVERIFIED_ACK",
)

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

# token -> (user id, is_admin, active)
_USERS = {
    "tok-admin": ("admin-id", 1, True),
    "tok-alice": ("alice-id", 0, True),
    "tok-suspended-admin": ("gone-id", 1, False),
}
_DOCUMENTS = {
    10: ("alice-lease.txt", "alice-id", "ALICE-DOC: the lease term is two years."),
    40: ("admin-memo.txt", "admin-id", "ADMIN-DOC: the notice period is thirty days."),
}


class _FakeDB:
    def __init__(self):
        self.statements = []
        self.unexpected = []
        self.inserts = []
        self.commits = 0
        self.connections = []
        self.fail_admin_lookup = False

    def connect(self):
        connection = _FakeConnection(self)
        self.connections.append(connection)
        return connection


class _FakeConnection:
    def __init__(self, db):
        self._db = db
        self._pending = []
        self.closed = False

    def cursor(self):
        return _FakeCursor(self, self._db)

    def commit(self):
        self._db.inserts.extend(self._pending)
        self._pending = []
        self._db.commits += 1

    def rollback(self):
        self._pending = []

    def close(self):
        self.closed = True


class _FakeCursor:
    def __init__(self, connection, db):
        self._connection = connection
        self._db = db
        self._rows = []

    def execute(self, sql, params=()):
        norm = " ".join(sql.split())
        self._db.statements.append(norm)
        self._rows = []
        user = _USERS.get(params[0]) if params and isinstance(params[0], str) else None
        active = user if user and user[2] else None
        if norm == _ACTIVE_ID:
            self._rows = [(active[0],)] if active else []
        elif norm == _ACTIVE_ADMIN:
            if self._db.fail_admin_lookup:
                raise RuntimeError("simulated lookup failure with secret detail")
            self._rows = [(active[1],)] if active else []
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
            entry_id = 700 + len(self._db.inserts) + len(self._connection._pending) + 1
            self._connection._pending.append((entry_id, tuple(params)))
            self._rows = [(entry_id,)]
        else:
            self._db.unexpected.append(norm)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def close(self):
        pass


def _ndjson(*records):
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


_DONE = {"response": "", "done": True, "done_reason": "stop", "prompt_eval_count": 900, "eval_count": 12}
_ANSWER = "The lease term is two years [C1]."


class _FakeStream:
    """A streamed /api/generate response delivered as small byte chunks."""

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
        # Ollama's NDJSON content type carries no charset, so requests hands
        # back raw bytes even with decode_unicode=True.
        yield from self._chunks


class _FakeReply:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


class _Ollama:
    def __init__(self):
        self.calls = []
        self.stream_payload = _ndjson({"response": _ANSWER}, _DONE)
        self.verifier_body = {"response": "supported", "done": True, "done_reason": "stop"}

    def __call__(self, endpoint, payload, *, stream=False, timeout=60):
        self.calls.append({"endpoint": endpoint, "payload": payload, "stream": stream, "timeout": timeout})
        if stream:
            return _FakeStream(self.stream_payload)
        return _FakeReply(self.verifier_body)

    @property
    def generations(self):
        return [call["payload"] for call in self.calls if call["stream"]]

    @property
    def verifications(self):
        return [call["payload"] for call in self.calls if not call["stream"]]


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
def ollama(monkeypatch, main_module):
    double = _Ollama()
    monkeypatch.setattr(main_module, "post_ollama", double)
    return double


@pytest.fixture
def inventory(monkeypatch, main_module):
    """The installed-digest lookup, recorded. Never a real /api/tags call."""
    state = {"digest": SYNTHETIC_DIGEST, "error": None, "calls": []}

    def installed_model_digest(tag):
        state["calls"].append(tag)
        if state["error"]:
            raise state["error"]
        return state["digest"]

    monkeypatch.setattr(main_module, "_installed_model_digest", installed_model_digest)
    return state


@pytest.fixture
def client(monkeypatch, main_module, db, ollama, inventory):
    class _NoEmbedding:
        def embed_query(self, text):
            raise AssertionError("these requests are answered from complete documents")

        def embed_documents(self, texts):
            raise AssertionError("these requests are answered from complete documents")

    monkeypatch.setattr(main_module, "_initialize_application", lambda: None)
    monkeypatch.setattr(main_module, "connect_to_postgres", db.connect)
    monkeypatch.setattr(main_module, "_EMBEDDING_PROVIDER", _NoEmbedding())
    test_client = TestClient(main_module.app)
    try:
        with test_client:
            yield test_client
    finally:
        test_client.close()


def _qualified_gemma_registry(default=LLAMA_CONTROL):
    """A synthetic registry in which Gemma's gates are all recorded, pinned to
    a synthetic digest. Production Gemma stays unqualified."""
    base = default_model_registry()
    gemma = dataclasses.replace(
        base.get(GEMMA_CANDIDATE),
        expected_digest=SYNTHETIC_DIGEST,
        qualification=Qualification(
            think_false_verified=True,
            combined_capacity_verified=True,
            quality_verified=True,
            evidence_ref="synthetic test identity",
        ),
    )
    return ModelRegistry(
        profiles=(base.get(LLAMA_CONTROL), gemma),
        default_profile_id=default,
        verifier_profile_id=LLAMA_CONTROL,
    )


def _events(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def _ask(client, token="tok-alice", **fields):
    body = {"question": "How long is the lease term?", "mode": "document_qa", "auth_token": token}
    body.update(fields)
    return client.post("/ask", json=body)


_BOUNDED_ANSWER_OPTIONS = {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1200}
_BOUNDED_VERIFIER_OPTIONS = {"temperature": 0.0, "num_ctx": 8192, "num_predict": 16}


def _assert_llama_answer_payload(payload):
    assert payload["model"] == "llama3.1:8b"
    assert payload["options"] == _BOUNDED_ANSWER_OPTIONS
    assert payload["keep_alive"] == "5m"
    assert payload["stream"] is True
    assert set(payload) == {"model", "prompt", "stream", "options", "keep_alive"}, (
        "no think field for Llama and no ineffective top-level settings"
    )


def _assert_refused_before_any_work(db, ollama):
    assert ollama.calls == [], "no model may be called"
    assert _SCOPE_OWNED not in db.statements and _LOAD_OWNED not in db.statements, (
        "no evidence may be read"
    )
    assert _INSERT_HISTORY not in db.statements and db.commits == 0
    assert db.unexpected == []
    assert all(connection.closed for connection in db.connections)


# --- every /ask caller resolves the default ----------------------------------

# The request shapes GridApp.jsx sends today, each with the legacy model string
# its task mapping attaches (several of them name qwen3:latest).
_CALLER_SHAPES = {
    "fetchSuggestions": dict(
        question="Generate exactly 4 specific questions a lawyer would ask about this document.",
        model="qwen3:latest", mode="structured_json", active_document_name="alice-lease.txt",
        active_document_id=10, selected_text="", selections=[],
    ),
    "sendMessage-document_qa": dict(
        model="qwen3:latest", mode="document_qa", active_document_name="alice-lease.txt",
        active_document_id=10, document_ids=[], selected_text="", selections=[],
    ),
    "sendMessage-all-documents": dict(
        model="llama3.1:8b", mode="document_qa", active_document_name=None,
        active_document_id=None, document_ids=[10], selected_text="", selections=[],
    ),
    "sendMessage-model_reasoning": dict(
        model="qwen3:latest", mode="model_reasoning", selected_text="the lease term",
        selections=[{"id": "s1", "text": "the lease term is two years", "document_name": "alice-lease.txt", "document_id": 10}],
    ),
    "runToolQuery": dict(
        question="Return JSON.", model="llama3.1:8b", mode="structured_json",
        active_document_id=10, document_ids=[], selected_text="", selections=[],
    ),
    "explainDiffPair": dict(
        question="Explain the change.", model="qwen3:latest", mode="model_reasoning",
        active_document_name=None, selected_text="",
        selections=[
            {"id": "a", "document_name": "Selection 1", "text": "two years", "document_id": 10},
            {"id": "b", "document_name": "Selection 2", "text": "three years", "document_id": None},
        ],
    ),
    "older-client-without-model": dict(mode="document_qa", active_document_id=10),
    "hostile-arbitrary-tag": dict(model="evil/unapproved:latest", mode="document_qa", active_document_id=10),
    "gemma-tag-as-legacy-string": dict(model="gemma4:31b", mode="document_qa", active_document_id=10),
}


@pytest.mark.parametrize("shape", sorted(_CALLER_SHAPES))
def test_every_caller_shape_uses_the_backend_default(client, db, ollama, inventory, shape):
    response = _ask(client, "tok-alice", **_CALLER_SHAPES[shape])

    assert response.status_code == 200, response.text
    events = _events(response)
    assert events[-1] == {"type": "done", "ok": True}
    [evidence] = [e for e in events if e["type"] == "evidence"]
    assert evidence["generation"] == {"profile_id": "llama_control", "model": "llama3.1:8b"}

    [generation] = ollama.generations
    _assert_llama_answer_payload(generation)
    for payload in ollama.calls:
        rendered = json.dumps(payload["payload"])
        for legacy in ("qwen3", "evil/unapproved", "gemma4"):
            assert legacy not in rendered, "a legacy model string must never be forwarded"
    for payload in ollama.verifications:
        assert payload["model"] == "llama3.1:8b"
        assert payload["options"] == _BOUNDED_VERIFIER_OPTIONS

    assert _ACTIVE_ADMIN not in db.statements, "an ordinary answer costs no admin lookup"
    assert inventory["calls"] == []
    assert db.unexpected == []


def test_a_default_answer_commits_history_before_entry_id_and_done(client, db, ollama):
    response = _ask(client, "tok-alice", active_document_id=10)

    events = _events(response)
    types = [e["type"] for e in events]
    assert types[0] == "delta" and types[-2:] == ["entry_id", "done"]
    assert types.index("evidence") < types.index("verification") < types.index("entry_id")
    [(entry_id, params)] = db.inserts
    assert db.commits == 1
    assert events[-2] == {"type": "entry_id", "entry_id": entry_id}
    assert params[3] == _ANSWER and params[4] == "alice-id"
    [verification] = [e for e in events if e["type"] == "verification"]
    assert [c["status"] for c in verification["claims"]] == ["supported"]
    assert len(ollama.verifications) == 1


# --- explicit overrides: who may ask -----------------------------------------


@pytest.mark.parametrize("profile_id", [LLAMA_CONTROL, GEMMA_CANDIDATE, "qwen3_latest"])
@pytest.mark.parametrize(
    "forged",
    [{}, {"is_admin": True}, {"role": "admin", "admin": True, "user": {"is_admin": True}}],
    ids=["plain", "forged-is_admin", "forged-role"],
)
def test_an_ordinary_users_explicit_profile_is_refused_before_inference(
    client, db, ollama, inventory, profile_id, forged
):
    response = _ask(client, "tok-alice", active_document_id=10, model_profile_id=profile_id, **forged)

    assert response.status_code == 403
    assert response.json() == {"detail": "Model selection requires an administrator account."}
    _assert_refused_before_any_work(db, ollama)
    assert inventory["calls"] == []


@pytest.mark.parametrize(
    "token, status",
    [(None, 401), ("tok-bogus", 401), ("tok-suspended-admin", 401)],
    ids=["anonymous", "unknown-token", "suspended-admin"],
)
def test_unidentified_callers_are_refused_before_any_admin_lookup(client, db, ollama, token, status):
    response = _ask(client, token, model_profile_id=GEMMA_CANDIDATE, is_admin=True)

    assert response.status_code == status
    assert _ACTIVE_ADMIN not in db.statements
    _assert_refused_before_any_work(db, ollama)


def test_an_empty_profile_id_is_rejected_by_validation(client, db, ollama):
    response = _ask(client, "tok-admin", model_profile_id="")
    assert response.status_code == 422
    assert ollama.calls == [] and db.statements == []


def test_a_failed_admin_lookup_fails_closed_without_leaking(client, db, ollama):
    db.fail_admin_lookup = True
    response = _ask(client, "tok-admin", model_profile_id=LLAMA_CONTROL)

    assert response.status_code == 503
    assert "secret" not in response.text
    _assert_refused_before_any_work(db, ollama)


# --- explicit overrides: what an admin may choose ----------------------------


def test_admin_unknown_profile_is_400(client, db, ollama):
    response = _ask(client, "tok-admin", model_profile_id="qwen3_latest")
    assert response.status_code == 400
    _assert_refused_before_any_work(db, ollama)


def test_admin_cannot_select_the_unqualified_production_gemma(client, db, ollama, inventory):
    response = _ask(client, "tok-admin", model_profile_id=GEMMA_CANDIDATE)
    assert response.status_code == 409
    _assert_refused_before_any_work(db, ollama)
    assert inventory["calls"] == []


def test_admin_may_explicitly_choose_the_default(client, db, ollama):
    response = _ask(client, "tok-admin", model_profile_id=LLAMA_CONTROL, active_document_id=40)
    assert response.status_code == 200
    assert _events(response)[-1] == {"type": "done", "ok": True}
    [generation] = ollama.generations
    _assert_llama_answer_payload(generation)


def test_admin_override_of_a_qualified_candidate_with_its_exact_digest(
    monkeypatch, main_module, client, db, ollama, inventory
):
    monkeypatch.setattr(main_module, "_MODEL_REGISTRY", _qualified_gemma_registry())

    response = _ask(client, "tok-admin", model_profile_id=GEMMA_CANDIDATE, active_document_id=40)

    assert response.status_code == 200, response.text
    events = _events(response)
    assert events[-1] == {"type": "done", "ok": True}
    [evidence] = [e for e in events if e["type"] == "evidence"]
    assert evidence["generation"] == {"profile_id": "gemma_candidate", "model": "gemma4:31b"}
    [generation] = ollama.generations
    assert generation["model"] == "gemma4:31b"
    assert generation["options"] == _BOUNDED_ANSWER_OPTIONS
    assert generation["think"] is False
    # Verification stays on the fixed, bounded Llama call.
    assert ollama.verifications and all(
        v["model"] == "llama3.1:8b" and v["options"] == _BOUNDED_VERIFIER_OPTIONS and "think" not in v
        for v in ollama.verifications
    )
    assert inventory["calls"] == ["gemma4:31b"]


def test_an_ordinary_user_still_cannot_reach_a_qualified_candidate(
    monkeypatch, main_module, client, db, ollama
):
    monkeypatch.setattr(main_module, "_MODEL_REGISTRY", _qualified_gemma_registry())
    assert _ask(client, "tok-alice", model_profile_id=GEMMA_CANDIDATE).status_code == 403
    response = _ask(client, "tok-alice", model="gemma4:31b", active_document_id=10)
    [generation] = ollama.generations
    assert generation["model"] == "llama3.1:8b", "a legacy string cannot reach the candidate"
    assert response.status_code == 200


@pytest.mark.parametrize(
    "digest, error, status",
    [
        (None, None, 409),                 # tag absent
        ("77" * 32, None, 409),            # re-pulled: digest changed
        (None, ConnectionError("x"), 503),  # inventory unreadable
    ],
    ids=["absent", "digest-changed", "inventory-unavailable"],
)
def test_a_qualified_candidate_with_the_wrong_identity_is_refused(
    monkeypatch, main_module, client, db, ollama, inventory, digest, error, status
):
    monkeypatch.setattr(main_module, "_MODEL_REGISTRY", _qualified_gemma_registry())
    inventory["digest"], inventory["error"] = digest, error

    response = _ask(client, "tok-admin", model_profile_id=GEMMA_CANDIDATE, active_document_id=40)

    assert response.status_code == status
    _assert_refused_before_any_work(db, ollama)


def test_resolution_is_fixed_before_streaming(monkeypatch, main_module, client, db, ollama):
    """A registry change while the answer streams cannot reach that answer."""
    real_scope = main_module._resolve_authorized_scope
    gemma_default = _qualified_gemma_registry(default=GEMMA_CANDIDATE)

    def scope_then_swap(token):
        # Runs inside the stream, after resolution and before generation.
        monkeypatch.setattr(main_module, "_MODEL_REGISTRY", gemma_default)
        return real_scope(token)

    monkeypatch.setattr(main_module, "_resolve_authorized_scope", scope_then_swap)

    response = _ask(client, "tok-alice", active_document_id=10)

    assert main_module._MODEL_REGISTRY is gemma_default, "the swap really happened mid-stream"
    [generation] = ollama.generations
    _assert_llama_answer_payload(generation)
    [answer_call] = [call for call in ollama.calls if call["stream"]]
    assert answer_call["timeout"] == 60.0, "the timeout was snapshotted with the profile too"
    [evidence] = [e for e in _events(response) if e["type"] == "evidence"]
    assert evidence["generation"]["model"] == "llama3.1:8b"


# --- a qualified candidate as the default (synthetic registry only) -----------


def test_a_qualified_gemma_default_needs_no_admin_but_verifies_identity_first(
    monkeypatch, main_module, client, db, ollama, inventory
):
    monkeypatch.setattr(main_module, "_MODEL_REGISTRY", _qualified_gemma_registry(default=GEMMA_CANDIDATE))

    response = _ask(client, "tok-alice", model="llama3.1:8b", active_document_id=10)

    assert response.status_code == 200, response.text
    assert _ACTIVE_ADMIN not in db.statements, "an omitted profile never needs admin authority"
    assert inventory["calls"] == ["gemma4:31b"], "the default's identity is checked per answer"
    [evidence] = [e for e in _events(response) if e["type"] == "evidence"]
    assert evidence["generation"] == {"profile_id": "gemma_candidate", "model": "gemma4:31b"}
    [answer_call] = [call for call in ollama.calls if call["stream"]]
    assert answer_call["payload"]["model"] == "gemma4:31b"
    assert answer_call["payload"]["think"] is False
    assert answer_call["timeout"] == 150.0


@pytest.mark.parametrize(
    "digest, error, status",
    [
        (None, None, 409),
        ("77" * 32, None, 409),
        (None, ConnectionError("x"), 503),
        ("not-a-digest", None, 503),
    ],
    ids=["absent", "digest-changed", "inventory-unavailable", "malformed-digest"],
)
def test_a_qualified_default_with_the_wrong_identity_fails_before_evidence_or_inference(
    monkeypatch, main_module, client, db, ollama, inventory, digest, error, status
):
    monkeypatch.setattr(main_module, "_MODEL_REGISTRY", _qualified_gemma_registry(default=GEMMA_CANDIDATE))
    inventory["digest"], inventory["error"] = digest, error

    response = _ask(client, "tok-alice", active_document_id=10)

    assert response.status_code == status
    assert _ACTIVE_ADMIN not in db.statements
    _assert_refused_before_any_work(db, ollama)


# --- the answer transport's finite read timeout -------------------------------


def test_each_resolved_profile_reaches_the_transport_with_its_own_timeout(
    monkeypatch, main_module, client, db, ollama, inventory
):
    _ask(client, "tok-alice", active_document_id=10)
    monkeypatch.setattr(main_module, "_MODEL_REGISTRY", _qualified_gemma_registry())
    _ask(client, "tok-admin", model_profile_id=GEMMA_CANDIDATE, active_document_id=40,
         answer_read_timeout_s=999, timeout=999)

    answers = [(c["payload"]["model"], c["timeout"]) for c in ollama.calls if c["stream"]]
    assert answers == [("llama3.1:8b", 60.0), ("gemma4:31b", 150.0)], (
        "a request field cannot change the transport timeout"
    )
    assert {c["timeout"] for c in ollama.calls if not c["stream"]} == {30}, (
        "the verifier keeps its existing finite timeout"
    )


def test_post_ollama_uses_the_value_as_the_requests_read_timeout(monkeypatch, main_module):
    sent = []
    monkeypatch.setattr(main_module, "get_active_ollama_base_url", lambda: "http://ollama.invalid:11434")
    monkeypatch.setattr(
        main_module.requests, "post",
        lambda url, json=None, stream=False, timeout=None: sent.append((url, stream, timeout)),
    )

    main_module.post_ollama("/api/generate", {}, stream=True, timeout=150.0)
    main_module.post_ollama("/api/generate", {}, timeout=30)

    assert sent == [
        ("http://ollama.invalid:11434/api/generate", True, (main_module.OLLAMA_CONNECT_TIMEOUT, 150.0)),
        ("http://ollama.invalid:11434/api/generate", False, (main_module.OLLAMA_CONNECT_TIMEOUT, 30)),
    ]
    assert main_module.OLLAMA_CONNECT_TIMEOUT == 1.5, "the connect timeout stays short"


# --- what a generation stream must be to count as an answer ------------------


_FAILING_STREAMS = {
    "thinking-only": (_ndjson({"thinking": "I think the answer is two years.", "response": ""}, _DONE), "generation_empty"),
    "empty": (_ndjson(_DONE), "generation_empty"),
    # A completed stream (done_reason "stop", not the output limit) whose only
    # answer text is whitespace: a space, a newline, then a tab and a carriage
    # return. It is as empty as no text at all and must not be persisted.
    "whitespace-only": (
        _ndjson({"response": " "}, {"response": "\n"}, {"response": "\t\r"}, _DONE),
        "generation_empty",
    ),
    "output-limit": (
        _ndjson({"response": "The lease term is"}, dict(_DONE, done_reason="length")),
        "generation_output_limit",
    ),
    "model-error-mid-stream": (
        _ndjson({"response": "The lease "}, {"error": "runner crashed: /srv/models/blob"}),
        "generation_incomplete",
    ),
    "no-final-record": (_ndjson({"response": "The lease term is two years."}), "generation_incomplete"),
    "malformed-line": (
        _ndjson({"response": "The lease "}) + '{"response": "te\n' + _ndjson(_DONE),
        "generation_incomplete",
    ),
}


@pytest.mark.parametrize("case", sorted(_FAILING_STREAMS))
def test_unfinished_or_empty_generation_never_persists(client, db, ollama, case):
    ollama.stream_payload, code = _FAILING_STREAMS[case]

    response = _ask(client, "tok-alice", active_document_id=10)

    events = _events(response)
    assert events[-1] == {"type": "done", "ok": False}
    errors = [e for e in events if e["type"] == "error"]
    assert [e["code"] for e in errors] == [code]
    assert not [e for e in events if e["type"] in ("verification", "entry_id")]
    assert _INSERT_HISTORY not in db.statements and db.commits == 0
    assert ollama.verifications == [], "nothing unfinished is verified"
    assert "runner crashed" not in response.text and "/srv/models" not in response.text
    assert "I think" not in response.text, "thinking text never reaches the client"


def test_thinking_alongside_an_answer_is_kept_out_of_everything(client, db, ollama):
    ollama.stream_payload = _ndjson(
        {"thinking": "SECRET-REASONING about [C1]. ", "response": ""},
        {"response": "The lease term is “two” years [C1]."},
        _DONE,
    )

    response = _ask(client, "tok-alice", active_document_id=10)

    events = _events(response)
    assert events[-1] == {"type": "done", "ok": True}
    assert "SECRET-REASONING" not in response.text
    [(_, params)] = db.inserts
    assert params[3] == "The lease term is “two” years [C1]."
    assert all("SECRET-REASONING" not in json.dumps(v) for v in ollama.verifications)


# --- the verifier -------------------------------------------------------------


@pytest.mark.parametrize(
    "body, expected",
    [
        ({"thinking": "supported", "response": "unsupported", "done": True}, "unsupported"),
        ({"thinking": "supported", "response": "", "done": True}, None),
        ({"response": "supported", "done": True, "done_reason": "length"}, None),
        (["supported"], None),
    ],
    ids=["thinking-ignored", "thinking-only", "output-limit", "not-an-object"],
)
def test_the_verifier_reads_only_a_complete_response(main_module, ollama, body, expected):
    ollama.verifier_body = body
    records = [type("R", (), {"citation_id": "C1", "evidence_text": "Term: two years."})()]

    status = main_module._ollama_claim_verifier(
        "The term is two years [C1].", records, main_module._MODEL_REGISTRY.verifier_call()
    )

    assert status == expected
    [call] = ollama.calls
    assert call["payload"]["model"] == "llama3.1:8b"
    assert call["payload"]["options"] == _BOUNDED_VERIFIER_OPTIONS


def test_a_verifier_error_reply_is_unavailable_never_supported(client, db, ollama):
    ollama.verifier_body = {"error": "out of memory at /srv/secret"}

    response = _ask(client, "tok-alice", active_document_id=10)

    events = _events(response)
    assert events[-1] == {"type": "done", "ok": True}, "the answer itself still completed"
    [verification] = [e for e in events if e["type"] == "verification"]
    assert [(c["status"], c["reason"]) for c in verification["claims"]] == [
        ("unverified", "verifier_unavailable")
    ]
    assert "/srv/secret" not in response.text


# --- background helpers -------------------------------------------------------


class _HelperReply:
    def __init__(self, text):
        self._text = text

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": self._text, "done": True, "done_reason": "stop"}


def _record_helper_posts(monkeypatch, main_module, reply_text):
    posts = []

    def post(url, json=None, timeout=None):
        posts.append({"url": url, "json": json})
        return _HelperReply(reply_text)

    monkeypatch.setattr(main_module, "get_active_ollama_base_url", lambda: "http://ollama.invalid")
    monkeypatch.setattr(main_module.requests, "post", post)
    return posts


def _assert_bounded_helper(payload, temperature):
    assert payload == {
        "model": "llama3.1:8b",
        "prompt": payload["prompt"],
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": 8192, "num_predict": 1200},
        "keep_alive": "5m",
    }
    assert payload["prompt"] != "Say OK", "no model-probing generation"


def test_insight_extraction_is_one_bounded_llama_call(monkeypatch, main_module, db):
    monkeypatch.setattr(main_module, "connect_to_postgres", db.connect)
    posts = _record_helper_posts(
        monkeypatch, main_module, '{"doc_type": "Lease", "keywords": [], "facts": [], "entities": []}'
    )

    main_module._extract_document_insights(10, "alice-lease.txt", "The lease term is two years. " * 20)

    assert [p["url"] for p in posts] == ["http://ollama.invalid/api/generate"]
    _assert_bounded_helper(posts[0]["json"], 0.1)


def test_kb_pair_generation_is_bounded_and_never_probes(monkeypatch, main_module):
    posts = _record_helper_posts(monkeypatch, main_module, "Q: What is the term?\nA: Two years in total.")

    pairs = main_module._generate_qa_pairs_from_text("The lease term is two years. " * 400, "label")

    assert pairs, "the helper still produces pairs"
    assert len(posts) == 2, "one call per excerpt and no probe calls"
    for post in posts:
        _assert_bounded_helper(post["json"], 0.2)
