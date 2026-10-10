"""Route-level contracts for backend answer-model resolution on POST /ask.

Drives the real FastAPI app through ``TestClient`` against an in-memory fake
database, and replaces ``post_ollama`` with an in-memory double, so no
database, Ollama, network, or subprocess is reached. Model digests are
synthetic test identities, never production values.

What these prove: every frontend /ask caller shape answers with the shared
selection and its legacy model string never reaches a request; a per-request
model_profile_id is refused with 400 for every caller -- an administrator, a
forged role, an anonymous request -- before any identity lookup, evidence or
model call; a qualified candidate answers only as the shared selection and
only with its exact installed digest, and an unusable identity is a 503 with
no fallback model; the snapshot is fixed before streaming; generation options
are native and bounded; separate thinking, model errors, malformed or
unfinished streams, the output limit and any completion reason other than
"stop" never produce a persisted success; verification uses the verifier call
fixed in the answer's own snapshot (the model that wrote the answer, with
bounded verifier options and its own read timeout) and grades only a finished
reply; a native generation call is sent only for an admitted job, after its
durable intent, with its read timeout clamped to the job's budget; background
helpers no longer probe models, build their snapshot once per job from the
shared selection they were admitted under, generate and write nothing when
that identity is unusable, send no further call after one whose end was not
observed, are bounded, parse or store only a finished reply, and log fixed
categories and counts rather than content.

The ledger installed here is what the lifespan installs once this process
owns generation, without a database session (ownership itself is proven in
tests/test_generation_selection.py and the integration tests).

REQUIRES THE SOCKETPAIR-AWARE SIBLING GUARD (see test_main_route_harness.py):
entering a TestClient builds a real asyncio loop.
"""

import contextlib
import dataclasses
import importlib
import json
import os
import sys
import time

import dotenv
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation_admission import (
    CallState,
    DispatchIntents,
    DispatchRefused,
    GenerationLedger,
    SharedSelection,
    generation_scope,
)
from generation_profiles import (
    GEMMA_CANDIDATE,
    LLAMA_CONTROL,
    ModelRegistry,
    Qualification,
    default_model_registry,
    shared_generation_snapshot,
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


class _Intents(DispatchIntents):
    def __init__(self):
        self.events = []

    def record_dispatch(self, intent):
        self.events.append(("dispatch", intent.call_role, intent.model_tag))

    def record_outcome(self, intent, state):
        self.events.append(("outcome", state))


def _ready_generation(monkeypatch, main_module, profile_id=LLAMA_CONTROL, epoch=1):
    """Install a generation ledger this process owns, admitting under the
    shared selection (profile_id, epoch). Returns (ledger, intents)."""
    ledger = GenerationLedger(main_module._ADMISSION_SETTINGS, clock=time.monotonic)
    intents = _Intents()
    assert ledger.install_owner(intents=intents, owner_seq=1, selection=SharedSelection(profile_id, epoch),
                                paused_by=None, recovered={})
    monkeypatch.setattr(main_module, "_GENERATION_LEDGER", ledger)
    return ledger, intents


@pytest.fixture(autouse=True)
def generation(monkeypatch, main_module):
    """Every test starts with the shared control selection admitted here."""
    return _ready_generation(monkeypatch, main_module)


def _select_gemma(monkeypatch, main_module, default=LLAMA_CONTROL):
    """The synthetic qualified registry, with Gemma as the shared selection."""
    monkeypatch.setattr(main_module, "_MODEL_REGISTRY", _qualified_gemma_registry(default=default))
    return _ready_generation(monkeypatch, main_module, GEMMA_CANDIDATE, 3)


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
    )


def _default_verifier(main_module):
    """The verifier call a shared-control answer's snapshot carries, built
    from the application's own registry the way /ask builds it."""
    return shared_generation_snapshot(
        main_module._MODEL_REGISTRY.get(LLAMA_CONTROL), epoch=1, observed_digest=None,
    ).verifier


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


# --- every /ask caller answers with the shared selection ------------------------

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
def test_every_caller_shape_uses_the_shared_selection(client, db, ollama, inventory, shape):
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


# --- the per-request override is retired for every caller ------------------------


@pytest.mark.parametrize("profile_id", [LLAMA_CONTROL, GEMMA_CANDIDATE, "qwen3_latest"])
@pytest.mark.parametrize(
    "token, forged",
    [("tok-alice", {}), ("tok-alice", {"is_admin": True, "role": "admin", "user": {"is_admin": True}}),
     ("tok-admin", {}), (None, {}), ("tok-suspended-admin", {})],
    ids=["ordinary", "forged-role", "admin", "anonymous", "suspended-admin"],
)
def test_any_per_request_profile_is_refused_before_any_lookup_or_inference(
    client, db, ollama, inventory, generation, profile_id, token, forged
):
    response = _ask(client, token, active_document_id=10, model_profile_id=profile_id, **forged)

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Per-request model selection is not supported. An administrator selects the shared answer model."
    }
    assert db.statements == [], "refused before the caller identity, evidence or history is read"
    assert ollama.calls == [] and inventory["calls"] == []
    assert generation[0].snapshot().answers_running == 0, "no admission slot was taken"


def test_an_empty_profile_id_is_rejected_by_validation(client, db, ollama):
    response = _ask(client, "tok-admin", model_profile_id="")
    assert response.status_code == 422
    assert ollama.calls == [] and db.statements == []


# --- a qualified candidate as the shared selection (synthetic registry only) -------


def test_a_qualified_gemma_selection_answers_with_its_exact_digest(
    monkeypatch, main_module, client, db, ollama, inventory
):
    _select_gemma(monkeypatch, main_module)

    response = _ask(client, "tok-admin", active_document_id=40)

    assert response.status_code == 200, response.text
    events = _events(response)
    assert events[-1] == {"type": "done", "ok": True}
    [evidence] = [e for e in events if e["type"] == "evidence"]
    assert evidence["generation"] == {"profile_id": "gemma_candidate", "model": "gemma4:31b"}
    [generation] = ollama.generations
    assert generation["model"] == "gemma4:31b"
    assert generation["options"] == _BOUNDED_ANSWER_OPTIONS
    assert generation["think"] is False
    # The answer is verified by the model that wrote it: the same exact tag and
    # think policy, with the bounded verifier options and Gemma's role timeout.
    assert ollama.verifications and all(
        v["model"] == "gemma4:31b" and v["options"] == _BOUNDED_VERIFIER_OPTIONS and v["think"] is False
        for v in ollama.verifications
    )
    assert {c["timeout"] for c in ollama.calls if not c["stream"]} == {150.0}
    assert inventory["calls"] == ["gemma4:31b"], "verification makes no second inventory lookup"
    assert _ACTIVE_ADMIN not in db.statements, "no request carries admin authority for a model"


def test_a_legacy_model_string_cannot_reach_the_candidate(monkeypatch, main_module, client, db, ollama):
    monkeypatch.setattr(main_module, "_MODEL_REGISTRY", _qualified_gemma_registry())
    response = _ask(client, "tok-alice", model="gemma4:31b", active_document_id=10)
    [generation] = ollama.generations
    assert generation["model"] == "llama3.1:8b", "only the shared selection decides the model"
    assert response.status_code == 200


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
    assert response.status_code == 200
    [generation] = ollama.generations
    _assert_llama_answer_payload(generation)
    [answer_call] = [call for call in ollama.calls if call["stream"]]
    assert answer_call["timeout"] == 60.0, "the timeout was snapshotted with the profile too"
    [evidence] = [e for e in _events(response) if e["type"] == "evidence"]
    assert evidence["generation"]["model"] == "llama3.1:8b"


def test_the_verifier_is_fixed_with_the_answer_even_if_the_registry_changes_mid_stream(
    monkeypatch, main_module, client, db, ollama, inventory
):
    """A Gemma-default registry swapped in after resolution cannot redirect
    the Llama answer's verification, and verification makes no lookup."""
    real_scope = main_module._resolve_authorized_scope

    def scope_then_swap(token):
        monkeypatch.setattr(main_module, "_MODEL_REGISTRY", _qualified_gemma_registry(default=GEMMA_CANDIDATE))
        return real_scope(token)

    monkeypatch.setattr(main_module, "_resolve_authorized_scope", scope_then_swap)

    response = _ask(client, "tok-alice", active_document_id=10)

    assert _events(response)[-1] == {"type": "done", "ok": True}
    [verification] = [call for call in ollama.calls if not call["stream"]]
    assert verification["payload"]["model"] == "llama3.1:8b" and "think" not in verification["payload"]
    assert verification["payload"]["options"] == _BOUNDED_VERIFIER_OPTIONS
    assert verification["timeout"] == 30.0
    assert inventory["calls"] == []


def test_a_gemma_answer_keeps_its_verifier_if_the_registry_changes_mid_stream(
    monkeypatch, main_module, client, db, ollama, inventory
):
    _select_gemma(monkeypatch, main_module)
    real_scope = main_module._resolve_authorized_scope

    def scope_then_swap(token):
        # The production registry, in which Gemma is unqualified again.
        monkeypatch.setattr(main_module, "_MODEL_REGISTRY", default_model_registry())
        return real_scope(token)

    monkeypatch.setattr(main_module, "_resolve_authorized_scope", scope_then_swap)

    response = _ask(client, "tok-admin", active_document_id=40)

    assert _events(response)[-1] == {"type": "done", "ok": True}
    [verification] = [call for call in ollama.calls if not call["stream"]]
    assert verification["payload"]["model"] == "gemma4:31b" and verification["payload"]["think"] is False
    assert verification["payload"]["options"] == _BOUNDED_VERIFIER_OPTIONS
    assert verification["timeout"] == 150.0
    assert inventory["calls"] == ["gemma4:31b"], "one lookup, when the snapshot was built; none per verifier call"


def test_a_qualified_gemma_selection_answer_is_verified_by_gemma(
    monkeypatch, main_module, client, db, ollama, inventory
):
    _select_gemma(monkeypatch, main_module, default=GEMMA_CANDIDATE)

    response = _ask(client, "tok-alice", active_document_id=10)

    events = _events(response)
    assert events[-1] == {"type": "done", "ok": True}
    [verification] = [call for call in ollama.calls if not call["stream"]]
    assert verification["payload"] == {
        "model": "gemma4:31b", "prompt": verification["payload"]["prompt"], "stream": False,
        "options": _BOUNDED_VERIFIER_OPTIONS, "keep_alive": "5m", "think": False,
    }
    assert verification["timeout"] == 150.0
    [claims] = [e["claims"] for e in events if e["type"] == "verification"]
    assert [c["status"] for c in claims] == ["supported"]
    assert _ACTIVE_ADMIN not in db.statements and inventory["calls"] == ["gemma4:31b"]


def test_a_qualified_gemma_selection_needs_no_admin_but_verifies_identity_first(
    monkeypatch, main_module, client, db, ollama, inventory
):
    _select_gemma(monkeypatch, main_module)

    response = _ask(client, "tok-alice", model="llama3.1:8b", active_document_id=10)

    assert response.status_code == 200, response.text
    assert _ACTIVE_ADMIN not in db.statements, "an ordinary answer never needs admin authority"
    assert inventory["calls"] == ["gemma4:31b"], "the selection's identity is checked per answer"
    [evidence] = [e for e in _events(response) if e["type"] == "evidence"]
    assert evidence["generation"] == {"profile_id": "gemma_candidate", "model": "gemma4:31b"}
    [answer_call] = [call for call in ollama.calls if call["stream"]]
    assert answer_call["payload"]["model"] == "gemma4:31b"
    assert answer_call["payload"]["think"] is False
    assert answer_call["timeout"] == 150.0


@pytest.mark.parametrize(
    "digest, error, status",
    [
        (None, None, 503),
        ("77" * 32, None, 503),
        (None, ConnectionError("x"), 503),
        ("not-a-digest", None, 503),
    ],
    ids=["absent", "digest-changed", "inventory-unavailable", "malformed-digest"],
)
def test_a_qualified_selection_with_the_wrong_identity_fails_before_evidence_or_inference(
    monkeypatch, main_module, client, db, ollama, inventory, generation, digest, error, status
):
    ledger, _ = _select_gemma(monkeypatch, main_module)
    inventory["digest"], inventory["error"] = digest, error

    response = _ask(client, "tok-alice", active_document_id=10)

    assert response.status_code == status
    assert response.json()["detail"] in {detail for _, detail in main_module._MODEL_SELECTION_REFUSALS.values()}
    assert _ACTIVE_ADMIN not in db.statements
    _assert_refused_before_any_work(db, ollama)
    snap = ledger.snapshot()
    assert (snap.answers_running, snap.unresolved_answers) == (0, 0), "the refused answer returned its slot"


# --- the answer transport's finite read timeout -------------------------------


def test_each_selected_profile_reaches_the_transport_with_its_own_timeout(
    monkeypatch, main_module, client, db, ollama, inventory
):
    _ask(client, "tok-alice", active_document_id=10)
    _select_gemma(monkeypatch, main_module)
    _ask(client, "tok-admin", active_document_id=40, answer_read_timeout_s=999, timeout=999)

    answers = [(c["payload"]["model"], c["timeout"]) for c in ollama.calls if c["stream"]]
    assert answers == [("llama3.1:8b", 60.0), ("gemma4:31b", 150.0)], (
        "a request field cannot change the transport timeout"
    )
    verifications = [(c["payload"]["model"], c["timeout"]) for c in ollama.calls if not c["stream"]]
    assert verifications == [("llama3.1:8b", 30.0), ("gemma4:31b", 150.0)], (
        "each verifier call carries its own profile's finite role timeout"
    )


def _admitted_scope(ledger):
    ticket = ledger.try_acquire_helper()
    ticket.bind_snapshot(shared_generation_snapshot(default_model_registry().get(LLAMA_CONTROL), epoch=1,
                                                    observed_digest=None))
    return ticket


def test_post_ollama_sends_generation_only_for_an_admitted_job_after_its_intent(monkeypatch, main_module, generation):
    ledger, intents = generation
    sent = []
    monkeypatch.setattr(main_module, "get_active_ollama_base_url", lambda: "http://ollama.invalid:11434")

    def post(url, json=None, stream=False, timeout=None):
        sent.append((url, stream, timeout, list(intents.events)))
        return type("R", (), {"json": lambda self: {"done": True, "done_reason": "stop"},
                              "raise_for_status": lambda self: None})()

    monkeypatch.setattr(main_module.requests, "post", post)

    with pytest.raises(DispatchRefused):
        main_module.post_ollama("/api/generate", {}, stream=True, timeout=150.0)
    assert sent == [], "no admitted job on this thread: nothing is sent"

    ticket = _admitted_scope(ledger)
    with generation_scope(ticket):
        main_module.post_ollama("/api/generate", {}, stream=True, timeout=150.0)
        assert ticket.observed({"done": True, "done_reason": "stop"})  # the body observes the stream
        main_module.post_ollama("/api/generate", {}, timeout=30).json()
    main_module.post_ollama("/api/embed", {}, timeout=30)  # embeddings are not generation

    assert [entry[:3] for entry in sent] == [
        ("http://ollama.invalid:11434/api/generate", True, (main_module.OLLAMA_CONNECT_TIMEOUT, 150.0)),
        ("http://ollama.invalid:11434/api/generate", False, (main_module.OLLAMA_CONNECT_TIMEOUT, 30)),
        ("http://ollama.invalid:11434/api/embed", False, (main_module.OLLAMA_CONNECT_TIMEOUT, 30)),
    ]
    assert sent[0][3] == [("dispatch", "answer", "llama3.1:8b")], "the durable intent precedes the POST"
    assert sent[1][3][-1] == ("dispatch", "verifier", "llama3.1:8b")
    assert intents.events[-1] == ("outcome", CallState.OBSERVED_TERMINAL), "the verifier reply was observed"
    assert main_module.OLLAMA_CONNECT_TIMEOUT == 1.5, "the connect timeout stays short"
    ledger.release(ticket)


def test_post_ollama_clamps_the_read_timeout_to_the_jobs_remaining_budget(monkeypatch, main_module, generation):
    ledger, intents = generation
    sent = []
    monkeypatch.setattr(main_module, "get_active_ollama_base_url", lambda: "http://ollama.invalid:11434")
    monkeypatch.setattr(main_module.requests, "post",
                        lambda url, json=None, stream=False, timeout=None: sent.append(timeout))
    ticket = _admitted_scope(ledger)
    ticket.deadline = time.monotonic() + 20.0
    with generation_scope(ticket):
        main_module.post_ollama("/api/generate", {}, stream=True, timeout=150.0)
    assert sent and 19.0 <= sent[0][1] <= 20.0, "a read can never outlast the job's whole budget"
    ticket.deadline = time.monotonic() - 1.0
    assert ticket.observed({"done": True, "done_reason": "stop"})
    with generation_scope(ticket), pytest.raises(DispatchRefused):
        main_module.post_ollama("/api/generate", {}, stream=True, timeout=150.0)
    assert len(sent) == 1, "a spent budget sends nothing"
    ledger.release(ticket)


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
    # A final record arrived, with an answer streamed before it, but it does
    # not say the model finished: only done_reason "stop" does.
    "done-reason-missing": (
        _ndjson({"response": _ANSWER}, {k: v for k, v in _DONE.items() if k != "done_reason"}),
        "generation_incomplete",
    ),
    "done-reason-non-string": (_ndjson({"response": _ANSWER}, dict(_DONE, done_reason=7)), "generation_incomplete"),
    "done-reason-load": (_ndjson({"response": _ANSWER}, dict(_DONE, done_reason="load")), "generation_incomplete"),
    "done-reason-unknown": (
        _ndjson({"response": _ANSWER}, dict(_DONE, done_reason="something_new")),
        "generation_incomplete",
    ),
    "done-not-true": (_ndjson({"response": _ANSWER}, dict(_DONE, done="true")), "generation_incomplete"),
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
        # A finished reply, so only the thinking/response split decides it.
        ({"thinking": "supported", "response": "unsupported", "done": True, "done_reason": "stop"}, "unsupported"),
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
        "The term is two years [C1].", records, _default_verifier(main_module)
    )

    assert status == expected
    [call] = ollama.calls
    assert call["payload"]["model"] == "llama3.1:8b"
    assert call["payload"]["options"] == _BOUNDED_VERIFIER_OPTIONS
    assert call["timeout"] == 30.0, "the captured verifier read timeout reaches the transport"


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


# Each would read "supported" if completion were not checked.
_UNFINISHED_VERIFIER_BODIES = {
    "done-missing": {"response": "supported", "done_reason": "stop"},
    "done-false": {"response": "supported", "done": False, "done_reason": "stop"},
    "done-string": {"response": "supported", "done": "true", "done_reason": "stop"},
    "reason-missing": {"response": "supported", "done": True},
    "reason-non-string": {"response": "supported", "done": True, "done_reason": 7},
    "reason-load": {"response": "supported", "done": True, "done_reason": "load"},
    "reason-unknown": {"response": "supported", "done": True, "done_reason": "something_new"},
    "blank-response": {"response": "   ", "done": True, "done_reason": "stop"},
    "thinking-only-finished": {"thinking": "supported", "response": "", "done": True, "done_reason": "stop"},
}


@pytest.mark.parametrize("case", sorted(_UNFINISHED_VERIFIER_BODIES))
def test_an_unfinished_verifier_reply_is_never_a_status(main_module, ollama, case):
    ollama.verifier_body = _UNFINISHED_VERIFIER_BODIES[case]
    records = [type("R", (), {"citation_id": "C1", "evidence_text": "Term: two years."})()]

    status = main_module._ollama_claim_verifier(
        "The term is two years [C1].", records, _default_verifier(main_module)
    )

    assert status is None


@pytest.mark.parametrize("case", ["done-false", "reason-missing", "reason-load", "reason-unknown"])
def test_an_unfinished_verifier_reply_is_malformed_and_the_answer_still_saves(client, db, ollama, case):
    ollama.verifier_body = _UNFINISHED_VERIFIER_BODIES[case]

    response = _ask(client, "tok-alice", active_document_id=10)

    events = _events(response)
    assert events[-1] == {"type": "done", "ok": True}, "a verifier reply never decides the answer"
    [verification] = [e for e in events if e["type"] == "verification"]
    assert [(c["status"], c["reason"]) for c in verification["claims"]] == [
        ("unverified", "verifier_malformed")
    ]
    assert len(ollama.verifications) == 1
    assert [e["type"] for e in events if e["type"] == "entry_id"] == ["entry_id"]
    [(_, params)] = db.inserts
    assert params[3] == _ANSWER


def test_a_verifier_error_outranks_its_completion_fields(client, db, ollama):
    ollama.verifier_body = {
        "error": "SECRET-VERIFIER-ERROR at /srv/models", "response": "supported",
        "done": True, "done_reason": "stop",
    }

    response = _ask(client, "tok-alice", active_document_id=10)

    events = _events(response)
    assert events[-1] == {"type": "done", "ok": True}
    [verification] = [e for e in events if e["type"] == "verification"]
    assert [(c["status"], c["reason"]) for c in verification["claims"]] == [
        ("unverified", "verifier_unavailable")
    ]
    assert "SECRET-VERIFIER-ERROR" not in response.text and "/srv/models" not in response.text
    assert len(db.inserts) == 1


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
        posts.append({"url": url, "json": json, "timeout": timeout})
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
    assert posts[0]["timeout"] == (5, 120.0), "5 s connect, the control's 120 s helper read timeout"


def test_kb_pair_generation_is_bounded_and_never_probes(monkeypatch, main_module):
    posts = _record_helper_posts(monkeypatch, main_module, "Q: What is the term?\nA: Two years in total.")

    pairs = main_module._generate_qa_pairs_from_text("The lease term is two years. " * 400, "label")

    assert pairs, "the helper still produces pairs"
    assert len(posts) == 2, "one call per excerpt and no probe calls"
    for post in posts:
        _assert_bounded_helper(post["json"], 0.2)
        assert post["timeout"] == (5, 120.0)


# --- background helpers: only a finished reply is parsed or stored -----------


class _HelperBody:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def _helper_replies(monkeypatch, main_module, *replies):
    """Serve each helper /api/generate call the next reply: a body, or an
    exception the transport raises."""
    posts = []
    queue = list(replies)

    def post(url, json=None, timeout=None):
        posts.append({"url": url, "json": json, "timeout": timeout})
        reply = queue.pop(0)
        if callable(reply) and not isinstance(reply, BaseException):
            reply = reply()  # a reply that acts (e.g. replaces the registry) when served
        if isinstance(reply, BaseException):
            raise reply
        return _HelperBody(reply)

    monkeypatch.setattr(main_module, "get_active_ollama_base_url", lambda: "http://ollama.invalid")
    monkeypatch.setattr(main_module.requests, "post", post)
    return posts


def _finished(text):
    return {"response": text, "done": True, "done_reason": "stop"}


def _unfinished_replies(text):
    """Replies whose ``response`` would parse and be used if completion were
    not checked first."""
    return {
        "done-missing": {"response": text, "done_reason": "stop"},
        "done-false": {"response": text, "done": False, "done_reason": "stop"},
        "reason-missing": {"response": text, "done": True},
        "reason-non-string": {"response": text, "done": True, "done_reason": 7},
        "reason-load": {"response": text, "done": True, "done_reason": "load"},
        "reason-unknown": {"response": text, "done": True, "done_reason": "something_new"},
        "reason-length": {"response": text, "done": True, "done_reason": "length"},
        "blank-response": {"response": " \n\t", "done": True, "done_reason": "stop"},
        "thinking-only": {"thinking": text, "response": "", "done": True, "done_reason": "stop"},
        "not-an-object": [text],
        "model-error": {"error": "runner crashed", "response": text, "done": True, "done_reason": "stop"},
    }


_UNFINISHED_CASES = sorted(_unfinished_replies(""))
# Replies that carry the protocol's terminal marker (the call ended) although
# they are not a usable answer; every other unfinished reply leaves the call's
# end unobserved.
_TERMINAL_UNFINISHED = {"reason-length", "blank-response", "thinking-only"}
_INSIGHT_DOCUMENT_TEXT = "The lease term is two years. " * 20
_REPLACEMENT_INSIGHTS = json.dumps(
    {"doc_type": "Replacement", "keywords": ["new"], "facts": ["A replacement fact."], "entities": ["Bob"]}
)
_EXISTING_INSIGHT = (
    "Lease", json.dumps(["lease term"]), json.dumps(["The term is two years."]),
    json.dumps(["Alice"]), "2026-01-01T00:00:00",
)


class _InsightsTable:
    """document_insights holding one existing valid row. The helper's
    INSERT ... ON CONFLICT (document_id) DO UPDATE is applied on commit, the
    way PostgreSQL would apply it, so an overwrite would be visible."""

    def __init__(self, rows):
        self.rows = dict(rows)
        self.statements = []
        self.connections = 0
        self.commits = 0
        self.connection_closes = 0
        self.cursor_closes = 0

    def connect(self):
        self.connections += 1
        return _InsightsConnection(self)


class _InsightsConnection:
    def __init__(self, table):
        self._table = table
        self.pending = {}

    def cursor(self):
        return _InsightsCursor(self)

    def commit(self):
        self._table.rows.update(self.pending)
        self.pending = {}
        self._table.commits += 1

    def rollback(self):
        self.pending = {}

    def close(self):
        self._table.connection_closes += 1


class _InsightsCursor:
    def __init__(self, connection):
        self._connection = connection

    def execute(self, sql, params=()):
        norm = " ".join(sql.split())
        self._connection._table.statements.append(norm)
        assert norm.startswith("INSERT INTO document_insights"), f"unexpected SQL: {norm}"
        assert "ON CONFLICT (document_id) DO UPDATE" in norm
        document_id, *values = params
        self._connection.pending[document_id] = tuple(values)

    def close(self):
        self._connection._table.cursor_closes += 1


def test_a_finished_insight_reply_upserts_the_row(monkeypatch, main_module):
    """Control for the test below: this fake really applies the upsert."""
    table = _InsightsTable({10: _EXISTING_INSIGHT})
    monkeypatch.setattr(main_module, "connect_to_postgres", table.connect)
    _helper_replies(monkeypatch, main_module, _finished(_REPLACEMENT_INSIGHTS))

    main_module._extract_document_insights(10, "alice-lease.txt", _INSIGHT_DOCUMENT_TEXT)

    assert table.commits == 1 and len(table.statements) == 1
    assert table.rows[10][:4] == (
        "Replacement", json.dumps(["new"]), json.dumps(["A replacement fact."]), json.dumps(["Bob"])
    )
    assert (table.cursor_closes, table.connection_closes) == (1, 1), "cursor and connection closed once"


@pytest.mark.parametrize("case", _UNFINISHED_CASES)
def test_an_unfinished_insight_reply_stores_nothing_and_keeps_the_existing_row(monkeypatch, main_module, case):
    table = _InsightsTable({10: _EXISTING_INSIGHT})
    monkeypatch.setattr(main_module, "connect_to_postgres", table.connect)
    posts = _helper_replies(monkeypatch, main_module, _unfinished_replies(_REPLACEMENT_INSIGHTS)[case])

    main_module._extract_document_insights(10, "alice-lease.txt", _INSIGHT_DOCUMENT_TEXT)

    assert len(posts) == 1
    assert table.connections == 0, "no database connection is opened for an unfinished reply"
    assert table.statements == [] and table.commits == 0, "no INSERT and no ON CONFLICT update"
    assert table.rows == {10: _EXISTING_INSIGHT}, "the existing valid row is untouched"


def test_an_unfinished_insight_reply_leaves_the_document_answerable(monkeypatch, main_module, client, db, ollama):
    connections_before, statements_before = len(db.connections), list(db.statements)
    _helper_replies(
        monkeypatch, main_module,
        {"response": _REPLACEMENT_INSIGHTS, "done": True, "done_reason": "load"},
    )

    main_module._extract_document_insights(10, "alice-lease.txt", _INSIGHT_DOCUMENT_TEXT)

    assert len(db.connections) == connections_before and db.statements == statements_before, (
        "the insight helper issued no SQL, so the committed document is untouched"
    )
    response = _ask(client, "tok-alice", active_document_id=10)
    events = _events(response)
    assert events[-1] == {"type": "done", "ok": True}, "the document still answers"
    assert len(db.inserts) == 1


_QA_TEXT = "The lease term is two years. " * 400
_QA_REPLY_ONE = "Q: What is the term?\nA: The term is two years in total."
_QA_REPLY_TWO = "Q: Who is the tenant?\nA: Alice is the tenant of record."
_QA_PAIR_ONE = {"question": "What is the term?", "answer": "The term is two years in total."}
_QA_PAIR_TWO = {"question": "Who is the tenant?", "answer": "Alice is the tenant of record."}


def test_finished_qa_replies_keep_every_pair(monkeypatch, main_module):
    _helper_replies(monkeypatch, main_module, _finished(_QA_REPLY_ONE), _finished(_QA_REPLY_TWO))
    assert main_module._generate_qa_pairs_from_text(_QA_TEXT, "label") == [_QA_PAIR_ONE, _QA_PAIR_TWO]


@pytest.mark.parametrize("case", _UNFINISHED_CASES)
def test_unfinished_qa_replies_contribute_no_pairs(monkeypatch, main_module, generation, case):
    posts = _helper_replies(
        monkeypatch, main_module,
        _unfinished_replies(_QA_REPLY_ONE)[case], _unfinished_replies(_QA_REPLY_TWO)[case],
    )

    assert main_module._generate_qa_pairs_from_text(_QA_TEXT, "label") == []
    if case in _TERMINAL_UNFINISHED:
        assert len(posts) == 2, "a terminal (if unusable) reply freed the slot: each excerpt was asked once"
    else:
        assert len(posts) == 1, "an unobserved end holds the slot: no second call is sent"
        assert generation[0].snapshot().unresolved_helpers == 1


@pytest.mark.parametrize("case", _UNFINISHED_CASES + ["transport-error"])
@pytest.mark.parametrize("invalid_part", [1, 2])
def test_only_finished_qa_replies_contribute_pairs(monkeypatch, main_module, case, invalid_part):
    def invalid(text):
        if case == "transport-error":
            return main_module.requests.exceptions.ConnectionError("simulated")
        return _unfinished_replies(text)[case]

    if invalid_part == 1:
        # After a first call whose end was not observed, the second excerpt is
        # never sent in that slot, so it cannot contribute either.
        replies = (invalid(_QA_REPLY_ONE), _finished(_QA_REPLY_TWO))
        expected = [_QA_PAIR_TWO] if case in _TERMINAL_UNFINISHED else []
    else:
        replies, expected = (_finished(_QA_REPLY_ONE), invalid(_QA_REPLY_TWO)), [_QA_PAIR_ONE]
    _helper_replies(monkeypatch, main_module, *replies)

    assert main_module._generate_qa_pairs_from_text(_QA_TEXT, "label") == expected


# --- background helpers log categories and counts, never content -------------

_PRIVATE = "PRIVATE-SENTINEL-3c9e"
_PRIVATE_TEXT = f"{_PRIVATE} The lease term is two years and the tenant is {_PRIVATE}. " * 20
_PRIVATE_INSIGHTS = json.dumps({
    "doc_type": f"{_PRIVATE}-Lease",
    "keywords": [f"{_PRIVATE}-keyword"],
    "facts": [f"{_PRIVATE} pays rent monthly."],
    "entities": [f"{_PRIVATE} Holdings"],
})
_PRIVATE_QA = f"Q: Who is {_PRIVATE}?\nA: {_PRIVATE} is the tenant of record here."


def _assert_logs_are_private(capsys):
    captured = capsys.readouterr()
    for stream in (captured.out, captured.err):
        assert _PRIVATE not in stream, f"private text reached a log line: {stream!r}"
        assert "://" not in stream, f"an endpoint reached a log line: {stream!r}"
    return captured.out


def _private_failure(main_module, kind):
    """A reply, or a transport exception, carrying private text and an endpoint."""
    if kind == "transport-error":
        return main_module.requests.exceptions.ConnectionError(
            f"{_PRIVATE} http://ollama.internal:11434/api/generate refused"
        )
    if kind == "model-error":
        return {"error": f"{_PRIVATE} out of memory at /srv/models/blob"}
    if kind == "unfinished":
        return {"response": _PRIVATE_INSIGHTS, "done": True, "done_reason": "load"}
    if kind == "unparseable":
        return _finished(f"{_PRIVATE} has no JSON here at all")
    raise AssertionError(kind)


def test_a_stored_insight_logs_counts_only(monkeypatch, main_module, capsys):
    table = _InsightsTable({})
    monkeypatch.setattr(main_module, "connect_to_postgres", table.connect)
    _helper_replies(monkeypatch, main_module, _finished(_PRIVATE_INSIGHTS))
    capsys.readouterr()

    main_module._extract_document_insights(10, f"{_PRIVATE}-lease.pdf", _PRIVATE_TEXT)

    assert table.commits == 1, "the insight was stored"
    out = _assert_logs_are_private(capsys)
    assert out.strip() == "[Insights] Stored: 1 keywords, 1 facts, 1 entities"


@pytest.mark.parametrize("kind", ["transport-error", "model-error", "unfinished", "unparseable"])
def test_a_failed_insight_logs_no_private_text(monkeypatch, main_module, capsys, kind):
    table = _InsightsTable({})
    monkeypatch.setattr(main_module, "connect_to_postgres", table.connect)
    _helper_replies(monkeypatch, main_module, _private_failure(main_module, kind))
    capsys.readouterr()

    main_module._extract_document_insights(10, f"{_PRIVATE}-lease.pdf", _PRIVATE_TEXT)

    assert table.statements == []
    assert _assert_logs_are_private(capsys).strip(), "the failure is still logged"


def test_an_insight_database_failure_logs_no_private_text(monkeypatch, main_module, capsys):
    def refuse():
        raise RuntimeError(f"{_PRIVATE} postgresql://user:pw@db.internal/synerge")

    monkeypatch.setattr(main_module, "connect_to_postgres", refuse)
    _helper_replies(monkeypatch, main_module, _finished(_PRIVATE_INSIGHTS))
    capsys.readouterr()

    main_module._extract_document_insights(10, f"{_PRIVATE}-lease.pdf", _PRIVATE_TEXT)

    assert _assert_logs_are_private(capsys).strip() == "[Insights] Extraction failed (non-critical): RuntimeError"


@pytest.mark.parametrize("helper", ["insights", "qa"])
def test_an_unreachable_model_server_logs_no_private_text(monkeypatch, main_module, capsys, helper):
    def unreachable():
        raise RuntimeError(f"Ollama is not reachable. Checked http://10.0.0.5:11434: {_PRIVATE}")

    monkeypatch.setattr(main_module, "get_active_ollama_base_url", unreachable)
    capsys.readouterr()

    if helper == "insights":
        main_module._extract_document_insights(10, f"{_PRIVATE}-lease.pdf", _PRIVATE_TEXT)
    else:
        assert main_module._generate_qa_pairs_from_text(_PRIVATE_TEXT * 10, f"{_PRIVATE}.example") == []

    assert _assert_logs_are_private(capsys).strip()


def test_finished_qa_replies_log_counts_only(monkeypatch, main_module, capsys):
    _helper_replies(monkeypatch, main_module, _finished(_PRIVATE_QA), _finished(_PRIVATE_QA))
    capsys.readouterr()

    pairs = main_module._generate_qa_pairs_from_text(_PRIVATE_TEXT * 10, f"{_PRIVATE}.example")

    assert len(pairs) == 2 and all(_PRIVATE in pair["answer"] for pair in pairs), "the pairs keep their content"
    _assert_logs_are_private(capsys)


@pytest.mark.parametrize("kind", ["transport-error", "model-error", "unfinished"])
def test_failed_qa_replies_log_no_private_text(monkeypatch, main_module, capsys, kind):
    reply = _private_failure(main_module, kind)
    if kind == "unfinished":
        reply = {"response": _PRIVATE_QA, "done": True}
    _helper_replies(monkeypatch, main_module, reply, reply)
    capsys.readouterr()

    pairs = main_module._generate_qa_pairs_from_text(_PRIVATE_TEXT * 10, f"{_PRIVATE}.example")

    assert pairs == []
    assert _assert_logs_are_private(capsys).strip(), "the failures are still logged"


# --- background helpers: the shared selection, captured once per job -----------


def _assert_gemma_helper(payload, temperature):
    assert payload == {
        "model": "gemma4:31b",
        "prompt": payload["prompt"],
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": 8192, "num_predict": 1200},
        "keep_alive": "5m",
        "think": False,
    }


def _no_admin_lookup(monkeypatch, main_module):
    assert not hasattr(main_module, "_caller_is_admin"), "no generation path consults admin authority"


def test_control_helpers_make_no_inventory_or_admin_lookup(monkeypatch, main_module, inventory):
    _no_admin_lookup(monkeypatch, main_module)
    table = _InsightsTable({})
    monkeypatch.setattr(main_module, "connect_to_postgres", table.connect)
    posts = _helper_replies(
        monkeypatch, main_module,
        _finished(_REPLACEMENT_INSIGHTS), _finished(_QA_REPLY_ONE), _finished(_QA_REPLY_TWO),
    )

    main_module._extract_document_insights(10, "alice-lease.txt", _INSIGHT_DOCUMENT_TEXT)
    pairs = main_module._generate_qa_pairs_from_text(_QA_TEXT, "label")

    assert inventory["calls"] == [], "the unqualified control is never looked up"
    assert pairs == [_QA_PAIR_ONE, _QA_PAIR_TWO] and table.commits == 1
    _assert_bounded_helper(posts[0]["json"], 0.1)
    for post in posts[1:]:
        _assert_bounded_helper(post["json"], 0.2)
    assert {post["timeout"] for post in posts} == {(5, 120.0)}


def test_a_qualified_candidate_selection_routes_both_helpers_with_its_identity(monkeypatch, main_module, inventory):
    _select_gemma(monkeypatch, main_module)
    _no_admin_lookup(monkeypatch, main_module)
    table = _InsightsTable({})
    monkeypatch.setattr(main_module, "connect_to_postgres", table.connect)
    posts = _helper_replies(
        monkeypatch, main_module,
        _finished(_REPLACEMENT_INSIGHTS), _finished(_QA_REPLY_ONE), _finished(_QA_REPLY_TWO),
    )

    main_module._extract_document_insights(10, "alice-lease.txt", _INSIGHT_DOCUMENT_TEXT)
    assert inventory["calls"] == ["gemma4:31b"], "the insight job read the identity once, before generating"
    pairs = main_module._generate_qa_pairs_from_text(_QA_TEXT, "label")
    assert inventory["calls"] == ["gemma4:31b", "gemma4:31b"], "one read per job, never one per excerpt"

    assert pairs == [_QA_PAIR_ONE, _QA_PAIR_TWO] and table.commits == 1
    _assert_gemma_helper(posts[0]["json"], 0.1)
    for post in posts[1:]:
        _assert_gemma_helper(post["json"], 0.2)
    assert {post["timeout"] for post in posts} == {(5, 150.0)}, "5 s connect, Gemma's helper read timeout"


def test_helpers_follow_the_shared_selection_their_job_was_admitted_under(
    monkeypatch, main_module, client, db, ollama, inventory
):
    table = _InsightsTable({})
    monkeypatch.setattr(main_module, "connect_to_postgres", table.connect)
    posts = _helper_replies(monkeypatch, main_module, _finished(_REPLACEMENT_INSIGHTS), _finished(_REPLACEMENT_INSIGHTS))
    main_module._extract_document_insights(40, "admin-memo.txt", _INSIGHT_DOCUMENT_TEXT)
    _select_gemma(monkeypatch, main_module)  # a committed switch: later jobs capture the new selection
    main_module._extract_document_insights(40, "admin-memo.txt", _INSIGHT_DOCUMENT_TEXT)

    _assert_bounded_helper(posts[0]["json"], 0.1)
    _assert_gemma_helper(posts[1]["json"], 0.1)
    assert [post["timeout"] for post in posts] == [(5, 120.0), (5, 150.0)]
    assert inventory["calls"] == ["gemma4:31b"], "only the Gemma job read its identity"


_UNUSABLE_HELPER_IDENTITIES = {
    "absent": (None, None, "ModelIdentityMismatchError"),
    "digest-changed": ("77" * 32, None, "ModelIdentityMismatchError"),
    "inventory-unavailable": (
        None, ConnectionError(f"{_PRIVATE} http://ollama.internal:11434 refused"), "ModelIdentityUnavailableError",
    ),
    "malformed-digest": ("not-a-digest", None, "ModelIdentityUnavailableError"),
}


@pytest.mark.parametrize("case", sorted(_UNUSABLE_HELPER_IDENTITIES))
def test_an_unusable_candidate_identity_stops_both_helpers_before_generation(
    monkeypatch, main_module, inventory, capsys, case
):
    digest, error, refusal = _UNUSABLE_HELPER_IDENTITIES[case]
    _select_gemma(monkeypatch, main_module)
    inventory["digest"], inventory["error"] = digest, error
    table = _InsightsTable({10: _EXISTING_INSIGHT})
    monkeypatch.setattr(main_module, "connect_to_postgres", table.connect)
    posts = _helper_replies(monkeypatch, main_module)  # nothing queued: any request would be recorded
    capsys.readouterr()

    main_module._extract_document_insights(10, f"{_PRIVATE}-lease.pdf", _PRIVATE_TEXT)
    pairs = main_module._generate_qa_pairs_from_text(_PRIVATE_TEXT * 10, f"{_PRIVATE}.example")

    assert posts == [], "no generation request was made"
    assert pairs == [], "Q&A contributes no pairs"
    assert table.connections == 0 and table.statements == [] and table.commits == 0, "no INSERT or upsert"
    assert table.rows == {10: _EXISTING_INSIGHT}, "the existing row is untouched"
    assert inventory["calls"] == ["gemma4:31b", "gemma4:31b"]
    assert _assert_logs_are_private(capsys).splitlines() == [
        f"[Insights] Helper model unavailable ({refusal}); insight extraction skipped",
        f"[KB] Helper model unavailable ({refusal}); Q&A generation skipped",
    ]


def test_an_ineligible_selection_stops_both_helpers_without_any_lookup(monkeypatch, main_module, inventory, capsys):
    # The persisted selection names Gemma, but this process's registry has the
    # production (unqualified) Gemma: no snapshot, and no fallback model.
    _ready_generation(monkeypatch, main_module, GEMMA_CANDIDATE, 3)
    table = _InsightsTable({10: _EXISTING_INSIGHT})
    monkeypatch.setattr(main_module, "connect_to_postgres", table.connect)
    posts = _helper_replies(monkeypatch, main_module)
    capsys.readouterr()

    main_module._extract_document_insights(10, "alice-lease.txt", _INSIGHT_DOCUMENT_TEXT)
    pairs = main_module._generate_qa_pairs_from_text(_QA_TEXT, "label")

    assert posts == [] and pairs == [] and inventory["calls"] == []
    assert table.connections == 0 and table.rows == {10: _EXISTING_INSIGHT}
    assert _assert_logs_are_private(capsys).splitlines() == [
        "[Insights] Helper model unavailable (ProfileNotQualifiedError); insight extraction skipped",
        "[KB] Helper model unavailable (ProfileNotQualifiedError); Q&A generation skipped",
    ]


@pytest.mark.parametrize("case", ["reason-load", "reason-length", "thinking-only", "model-error", "not-an-object"])
def test_an_unfinished_candidate_insight_reply_stores_nothing(monkeypatch, main_module, inventory, case):
    _select_gemma(monkeypatch, main_module)
    table = _InsightsTable({10: _EXISTING_INSIGHT})
    monkeypatch.setattr(main_module, "connect_to_postgres", table.connect)
    posts = _helper_replies(monkeypatch, main_module, _unfinished_replies(_REPLACEMENT_INSIGHTS)[case])

    main_module._extract_document_insights(10, "alice-lease.txt", _INSIGHT_DOCUMENT_TEXT)

    [post] = posts
    _assert_gemma_helper(post["json"], 0.1)
    assert table.connections == 0 and table.statements == [] and table.rows == {10: _EXISTING_INSIGHT}


def test_the_qa_job_keeps_its_snapshot_across_excerpts(monkeypatch, main_module, inventory):
    gemma_default = _qualified_gemma_registry(default=GEMMA_CANDIDATE)

    def swap_then_reply():
        # Served for the first excerpt: the registry changes mid-job.
        monkeypatch.setattr(main_module, "_MODEL_REGISTRY", gemma_default)
        return _finished(_QA_REPLY_ONE)

    posts = _helper_replies(monkeypatch, main_module, swap_then_reply, _finished(_QA_REPLY_TWO))

    pairs = main_module._generate_qa_pairs_from_text(_QA_TEXT, "label")

    assert main_module._MODEL_REGISTRY is gemma_default, "the swap really happened between excerpts"
    assert pairs == [_QA_PAIR_ONE, _QA_PAIR_TWO]
    for post in posts:
        _assert_bounded_helper(post["json"], 0.2)
        assert post["timeout"] == (5, 120.0)
    assert inventory["calls"] == [], "the job was not resolved again for its second excerpt"


def test_a_candidate_qa_job_keeps_its_snapshot_if_the_registry_reverts(monkeypatch, main_module, inventory):
    _select_gemma(monkeypatch, main_module)
    production = default_model_registry()

    def revert_then_reply():
        monkeypatch.setattr(main_module, "_MODEL_REGISTRY", production)
        return _finished(_QA_REPLY_ONE)

    posts = _helper_replies(monkeypatch, main_module, revert_then_reply, _finished(_QA_REPLY_TWO))

    assert main_module._generate_qa_pairs_from_text(_QA_TEXT, "label") == [_QA_PAIR_ONE, _QA_PAIR_TWO]
    assert main_module._MODEL_REGISTRY is production
    for post in posts:
        _assert_gemma_helper(post["json"], 0.2)
    assert {post["timeout"] for post in posts} == {(5, 150.0)}
    assert inventory["calls"] == ["gemma4:31b"]


def test_a_helper_with_no_free_slot_or_no_ownership_generates_nothing(monkeypatch, main_module, capsys):
    posts = _helper_replies(monkeypatch, main_module)
    ledger = GenerationLedger(main_module._ADMISSION_SETTINGS, clock=time.monotonic)  # never owned
    monkeypatch.setattr(main_module, "_GENERATION_LEDGER", ledger)
    capsys.readouterr()

    main_module._extract_document_insights(10, "alice-lease.txt", _INSIGHT_DOCUMENT_TEXT)
    assert main_module._generate_qa_pairs_from_text(_QA_TEXT, "label") == []

    assert posts == []
    assert capsys.readouterr().out.splitlines() == [
        "[Insights] Helper model capacity unavailable; insight extraction skipped",
        "[KB] Helper model capacity unavailable; Q&A generation skipped",
    ]
