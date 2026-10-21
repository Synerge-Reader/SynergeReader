"""Runtime routing contracts for main.py: the experimental profiles on the
compatibility runtime, everything else on the primary runtime.

Drives the real FastAPI app (``TestClient``) and the real ``post_ollama``,
helper and inventory code against an in-memory database and a fake HTTP
transport (``requests.get`` / ``requests.post`` replaced), so no database,
Ollama, network or subprocess is reached. Digests are the operator-supplied
pins; inference is never run.

What these prove:

* every call site -- the streamed answer, its verifier calls, both background
  helpers, the identity lookup and the loaded-model list -- reaches the runtime
  the job's snapshot names, and only that one; the embedder and the
  control/qualified profiles stay on the primary runtime;
* the compatibility endpoint comes from server configuration only, is
  validated, and an unset, malformed, disabled, unreachable or mismatched
  runtime refuses without any fallback to another runtime or model, and
  without leaving a dispatched call unresolved;
* the tested think settings are sent, thinking text never reaches the
  stream, the saved answer or its provenance, and a thinking-only or
  length-limited reply is not an answer;
* provenance version 4 records the runtime key (never a URL), the experimental
  flag and the think value that was sent;
* a restart with the experimental tier off (or without its endpoint) keeps
  the persisted experimental selection -- no automatic switch or reseed --
  and refuses to answer with it while reporting it honestly to the admin.

What these do NOT prove: anything about real Ollama, the compatibility
container, GPU residency or answer quality.

REQUIRES THE SOCKETPAIR-AWARE SIBLING GUARD (see test_main_route_harness.py).
"""

import contextlib
import importlib
import json
import os
import sys
import time

import dotenv
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation_admission import (  # noqa: E402
    DispatchIntents,
    DispatchRefused,
    GenerationLedger,
    JobKind,
    SharedSelection,
    generation_scope,
)
from generation_profiles import (  # noqa: E402
    COMPAT_RUNTIME,
    GEMMA_CANDIDATE,
    GEMMA_CANDIDATE_DIGEST,
    INELIGIBLE_EXPERIMENTAL_DISABLED,
    INELIGIBLE_RUNTIME_NOT_CONFIGURED,
    MISTRAL35_DIGEST,
    MISTRAL35_EXPERIMENTAL,
    NEMOTRON35_DIGEST,
    NEMOTRON35_EXPERIMENTAL,
    PRIMARY_RUNTIME,
    QWEN38_DIGEST,
    QWEN38_EXPERIMENTAL,
)
from generation_selection import OUTCOME_COMMITTED, OwnerRecovery, OwnerState, SelectionRecord  # noqa: E402

PRIMARY_URL = "http://primary.invalid:11434"
COMPAT_URL = "http://synergereader-compat-ollama:11434"
SECRET_THOUGHT = "SECRET-THOUGHT-never-shown"
_ANSWER = "The lease term is two years [C1]."

_EMBEDDING_PROFILE_KEYS = (
    "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_DIMENSION", "EMBEDDING_QUERY_PREFIX",
    "EMBEDDING_DOCUMENT_PREFIX", "EMBEDDING_PROFILE_UNVERIFIED_ACK",
)
_ACTIVE_ID = "SELECT id FROM users WHERE token = %s AND COALESCE(is_active, 1) <> 0"
_SCOPE_OWNED = "SELECT id, filename, title, length(content) FROM documents WHERE user_id = %s ORDER BY id"
_LOAD_OWNED = "SELECT content FROM documents WHERE id = %s AND user_id = %s"
_INSERT_HISTORY = (
    "INSERT INTO chat_history (ts, selected_text, question, answer, user_id, trace_id, answer_provenance) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id"
)
_DOCUMENT = (10, "alice-lease.txt", "alice-id", "ALICE-DOC: the lease term is two years.")


# --- an in-memory database --------------------------------------------------------


class _DB:
    def __init__(self):
        self.inserts = []

    def connect(self):
        return _Connection(self)


class _Connection:
    def __init__(self, db):
        self.db, self.pending = db, []

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        self.db.inserts.extend(self.pending)
        self.pending = []

    def rollback(self):
        self.pending = []

    def close(self):
        pass


class _Cursor:
    def __init__(self, connection):
        self.connection, self.rows = connection, []

    def execute(self, sql, params=()):
        norm = " ".join(sql.split())
        doc_id, name, owner, content = _DOCUMENT
        self.rows = []
        if norm == _ACTIVE_ID:
            self.rows = [("alice-id",)] if params and params[0] == "tok-alice" else []
        elif norm == _SCOPE_OWNED:
            self.rows = [(doc_id, name, None, len(content))] if params[0] == owner else []
        elif norm == _LOAD_OWNED:
            self.rows = [(content,)] if params == (doc_id, owner) else []
        elif norm == _INSERT_HISTORY:
            self.connection.pending.append(tuple(params))
            self.rows = [(900 + len(self.connection.db.inserts) + 1,)]

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)

    def close(self):
        pass


# --- a fake HTTP transport ---------------------------------------------------------


def _ndjson(*records):
    return "".join(json.dumps(record) + "\n" for record in records).encode()


class _Stream:
    def __init__(self, data):
        self.chunks = [data[i:i + 9] for i in range(0, len(data), 9)]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, decode_unicode=True, chunk_size=32):
        yield from self.chunks


class _Reply:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


class _Transport:
    """requests.get/post for both runtimes, recorded by URL."""

    def __init__(self):
        self.gets, self.posts = [], []
        self.compat_down = False
        self.installed = {
            PRIMARY_URL: {"gemma4:31b": GEMMA_CANDIDATE_DIGEST},
            COMPAT_URL: {"qwen3.8:27b": QWEN38_DIGEST, "nemotron-3.5-lightning:30b": NEMOTRON35_DIGEST,
                         "mistral-medium-3.5:128b": MISTRAL35_DIGEST},
        }
        self.loaded = {PRIMARY_URL: [], COMPAT_URL: []}
        done = {"response": "", "done": True, "done_reason": "stop", "prompt_eval_count": 950, "eval_count": 300}
        self.stream = _ndjson({"thinking": SECRET_THOUGHT, "response": ""}, {"thinking": " more", "response": ""},
                              {"response": _ANSWER}, done)
        self.verifier = {"thinking": SECRET_THOUGHT, "response": "supported", "done": True, "done_reason": "stop"}
        self.qa = {"thinking": SECRET_THOUGHT, "done": True, "done_reason": "stop",
                   "response": "Q: How long is the lease?\nA: The lease runs for two years from January first."}
        self.helper = {"thinking": SECRET_THOUGHT, "done": True, "done_reason": "stop",
                       "response": json.dumps({"doc_type": "Lease", "keywords": ["lease"], "facts": ["two years"],
                                               "entities": ["Alice"]})}

    def _base(self, url):
        for base in (PRIMARY_URL, COMPAT_URL):
            if url.startswith(base + "/"):
                if base == COMPAT_URL and self.compat_down:
                    raise ConnectionError("SECRET-DETAIL connection refused")
                return base, url[len(base):]
        raise AssertionError(f"a request left the configured runtimes: {url}")

    def get(self, url, timeout=None, **kwargs):
        self.gets.append(url)
        base, path = self._base(url)
        if path == "/api/version":
            return _Reply({"version": "0.32.12"})
        if path == "/api/tags":
            return _Reply({"models": [{"name": tag, "model": tag, "digest": digest}
                                      for tag, digest in self.installed[base].items()]})
        if path == "/api/ps":
            return _Reply({"models": list(self.loaded[base])})
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url, json=None, stream=False, timeout=None, **kwargs):
        base, path = self._base(url)
        self.posts.append({"base": base, "path": path, "payload": json, "stream": stream, "timeout": timeout})
        assert path in ("/api/generate", "/api/embed"), url
        if path == "/api/embed":
            return _Reply({"embeddings": [[0.0] * 1024]})
        if stream:
            return _Stream(self.stream)
        prompt = json["prompt"]
        if "question-answer pairs" in prompt:
            return _Reply(self.qa)
        return _Reply(self.helper if "JSON object" in prompt else self.verifier)

    def generation_posts(self):
        return [p for p in self.posts if p["path"] == "/api/generate"]


# --- main, imported with the deployment's server settings -------------------------------


@contextlib.contextmanager
def _imported_main():
    previous = sys.modules.pop("main", None)
    try:
        yield importlib.import_module("main")
    finally:
        sys.modules.pop("main", None)
        if previous is not None:
            sys.modules["main"] = previous


def _load_main(monkeypatch, *, compat_url=COMPAT_URL, enabled=True):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    for key in _EMBEDDING_PROFILE_KEYS:
        monkeypatch.delenv(key, raising=False)
    if compat_url is None:
        monkeypatch.delenv("SYNERGE_COMPAT_OLLAMA_URL", raising=False)
    else:
        monkeypatch.setenv("SYNERGE_COMPAT_OLLAMA_URL", compat_url)
    if enabled:
        monkeypatch.setenv("SYNERGE_EXPERIMENTAL_MODELS", "1")
    else:
        monkeypatch.delenv("SYNERGE_EXPERIMENTAL_MODELS", raising=False)
    return _imported_main()


class _Intents(DispatchIntents):
    def __init__(self):
        self.events = []

    def record_dispatch(self, intent):
        self.events.append(("dispatch", intent.call_role, intent.profile_id, intent.model_tag))

    def record_outcome(self, intent, state):
        self.events.append(("outcome", state))


def _install(monkeypatch, main, profile_id, epoch=3):
    ledger = GenerationLedger(main._ADMISSION_SETTINGS, clock=time.monotonic, selection_policy=main._selection_policy)
    intents = _Intents()
    assert ledger.install_owner(intents=intents, owner_seq=1, selection=SharedSelection(profile_id, epoch),
                                paused_by=None, recovered={})
    monkeypatch.setattr(main, "_GENERATION_LEDGER", ledger)
    return ledger, intents


@pytest.fixture
def transport():
    return _Transport()


@pytest.fixture
def env(monkeypatch, transport, request):
    settings = getattr(request, "param", {})
    with _load_main(monkeypatch, **settings) as main:
        db = _DB()
        monkeypatch.setattr(main, "_initialize_application", lambda: None)
        monkeypatch.setattr(main, "connect_to_postgres", db.connect)
        monkeypatch.setattr(main, "get_active_ollama_base_url", lambda: PRIMARY_URL)
        monkeypatch.setattr(main.requests, "get", transport.get)
        monkeypatch.setattr(main.requests, "post", transport.post)
        client = TestClient(main.app)
        with client:
            yield main, client, db


def _ask(client, **fields):
    body = {"question": "How long is the lease term?", "mode": "document_qa", "auth_token": "tok-alice",
            "active_document_id": 10}
    body.update(fields)
    return client.post("/ask", json=body)


def _events(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


# --- server configuration ----------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("http://synergereader-compat-ollama:11434", "http://synergereader-compat-ollama:11434"),
        ("http://synergereader-compat-ollama:11434/", "http://synergereader-compat-ollama:11434"),
        ("  http://10.0.0.5:11434 ", "http://10.0.0.5:11434"),
        ("", None),
        ("synergereader-compat-ollama:11434", None),
        ("https://synergereader-compat-ollama:11434", None),
        ("http://synergereader-compat-ollama", None),
        ("http://user:secret@synergereader-compat-ollama:11434", None),
        ("http://synergereader-compat-ollama:11434/api", None),
        ("http://synergereader-compat-ollama:11434?x=1", None),
        ("http://synergereader-compat-ollama:99999", None),
        ("http://bad_host:11434", None),
    ],
)
def test_the_compatibility_endpoint_is_a_validated_server_setting(monkeypatch, value, expected):
    with _load_main(monkeypatch, compat_url=None) as main:
        assert main._compat_runtime_url(value) == expected


def test_defaults_leave_every_experimental_profile_unselectable(monkeypatch):
    with _load_main(monkeypatch, compat_url=None, enabled=False) as main:
        assert main._COMPAT_RUNTIME_URL is None and main._EXPERIMENTAL_MODELS_ENABLED is False
        assert main._MODEL_REGISTRY.configured_runtimes == frozenset({PRIMARY_RUNTIME})
        registry = main._MODEL_REGISTRY
        assert [p.profile_id for p in registry.profiles if registry.is_selectable(p)] == [
            "llama_control", "gemma_candidate"]


def test_a_malformed_endpoint_disables_the_runtime_and_is_never_logged(monkeypatch, capsys):
    with _load_main(monkeypatch, compat_url="http://user:SECRET-PASSWORD@compat:11434") as main:
        assert main._COMPAT_RUNTIME_URL is None
        assert COMPAT_RUNTIME not in main._MODEL_REGISTRY.configured_runtimes
    printed = capsys.readouterr().out
    assert "not a valid private runtime URL" in printed and "SECRET-PASSWORD" not in printed


# --- an experimental answer, end to end -------------------------------------------------------


def test_a_qwen_answer_its_verifier_and_identity_use_only_the_compatibility_runtime(monkeypatch, env, transport):
    main, client, db = env
    _, intents = _install(monkeypatch, main, QWEN38_EXPERIMENTAL)
    events = _events(_ask(client))
    assert events[-1] == {"type": "done", "ok": True}, events

    generation = transport.generation_posts()
    assert [(p["base"], p["stream"]) for p in generation] == [(COMPAT_URL, True), (COMPAT_URL, False)]
    answer, verifier = (p["payload"] for p in generation)
    assert answer["model"] == verifier["model"] == "qwen3.8:27b"
    assert answer["think"] is True and verifier["think"] is True
    assert answer["options"] == {"temperature": 0.2, "num_ctx": 8192, "num_predict": 2048}
    assert verifier["options"]["num_predict"] == 512
    assert generation[0]["timeout"][1] <= 180.0 and generation[1]["timeout"][1] <= 180.0
    assert f"{COMPAT_URL}/api/tags" in transport.gets and f"{PRIMARY_URL}/api/tags" not in transport.gets
    assert all(p["base"] == COMPAT_URL for p in transport.posts), "nothing went to the primary runtime"
    assert [event[:3] for event in intents.events if event[0] == "dispatch"] == [
        ("dispatch", "answer", QWEN38_EXPERIMENTAL), ("dispatch", "verifier", QWEN38_EXPERIMENTAL)]

    evidence = next(e for e in events if e["type"] == "evidence")
    assert evidence["generation"] == {"profile_id": QWEN38_EXPERIMENTAL, "model": "qwen3.8:27b",
                                      "experimental": True}
    text = json.dumps(events)
    assert SECRET_THOUGHT not in text, "thinking never reaches the stream"
    [row] = db.inserts
    assert row[3] == _ANSWER and SECRET_THOUGHT not in row[3]
    provenance = json.loads(row[6])
    assert provenance["schema_version"] == 4
    assert provenance["generation"]["runtime_key"] == COMPAT_RUNTIME
    assert provenance["generation"]["experimental"] is True and provenance["generation"]["think"] is True
    assert provenance["generation"]["observed_digest"] == QWEN38_DIGEST
    assert provenance["verification"]["runtime_key"] == COMPAT_RUNTIME
    assert provenance["verification"]["requested_options"]["num_predict"] == 512
    assert SECRET_THOUGHT not in row[6] and "://" not in row[6] and "thinking" not in row[6]


def test_mistral_sends_its_tested_effort_level_with_its_larger_bounded_timeout(monkeypatch, env, transport):
    main, client, db = env
    _install(monkeypatch, main, MISTRAL35_EXPERIMENTAL)
    assert _events(_ask(client))[-1] == {"type": "done", "ok": True}
    answer, verifier = (p for p in transport.generation_posts())
    assert answer["payload"]["think"] == "high" and verifier["payload"]["think"] == "high"
    assert 180.0 < answer["timeout"][1] <= 420.0, "longer than Gemma's 150 s, clamped to the 900 s job budget"
    provenance = json.loads(db.inserts[0][6])
    assert provenance["generation"]["think"] == "high"
    assert provenance["generation"]["answer_read_timeout_s"] == 420.0


@pytest.mark.parametrize(
    "stream, code",
    [
        (_ndjson({"thinking": SECRET_THOUGHT, "response": ""}, {"response": "", "done": True, "done_reason": "stop"}),
         "generation_empty"),
        (_ndjson({"thinking": SECRET_THOUGHT, "response": ""}, {"response": "", "done": True, "done_reason": "length"}),
         "generation_output_limit"),
        (_ndjson({"response": "The lease"}, {"response": "", "done": True, "done_reason": "length"}),
         "generation_output_limit"),
    ],
    ids=["thinking-only", "thinking-cut-off", "answer-cut-off"],
)
def test_a_thinking_only_or_length_limited_reply_is_never_an_answer(monkeypatch, env, transport, stream, code):
    main, client, db = env
    _install(monkeypatch, main, QWEN38_EXPERIMENTAL)
    transport.stream = stream
    events = _events(_ask(client))
    assert events[-1] == {"type": "done", "ok": False}
    assert any(e["type"] == "error" and e["code"] == code for e in events), events
    assert db.inserts == [] and SECRET_THOUGHT not in json.dumps(events)
    assert len(transport.generation_posts()) == 1, "no verifier call for an unfinished answer"


def test_a_verifier_whose_thinking_used_its_allowance_leaves_the_claim_unverified(monkeypatch, env, transport):
    main, client, db = env
    _install(monkeypatch, main, QWEN38_EXPERIMENTAL)
    transport.verifier = {"thinking": SECRET_THOUGHT, "response": "", "done": True, "done_reason": "length"}
    events = _events(_ask(client))
    assert events[-1] == {"type": "done", "ok": True}
    verification = next(e for e in events if e["type"] == "verification")
    assert {claim["status"] for claim in verification["claims"]} == {"unverified"}
    assert SECRET_THOUGHT not in json.dumps(events)


# --- refusals, never a fallback --------------------------------------------------------------


def test_an_unreachable_compatibility_runtime_refuses_before_the_identity_read(monkeypatch, env, transport):
    main, client, db = env
    ledger, intents = _install(monkeypatch, main, QWEN38_EXPERIMENTAL)
    transport.compat_down = True
    response = _ask(client)
    assert (response.status_code, response.json()["detail"]) == (
        503, "Could not verify the model. Please try again shortly.")
    assert transport.posts == [], "no model call anywhere -- no fallback to the primary runtime"
    assert not any(url.startswith(PRIMARY_URL) for url in transport.gets)
    assert "SECRET-DETAIL" not in response.text and COMPAT_URL not in response.text
    assert ledger.snapshot().answers_running == 0 and intents.events == []


def test_a_runtime_lost_after_the_identity_read_fails_closed_without_an_unresolved_call(monkeypatch, env, transport):
    main, client, db = env
    ledger, intents = _install(monkeypatch, main, QWEN38_EXPERIMENTAL)
    real_get = transport.get

    def lost_after_identity(url, timeout=None, **kwargs):
        reply = real_get(url, timeout=timeout, **kwargs)
        if url == f"{COMPAT_URL}/api/tags":
            # The runtime goes away and the 30 s reachability cache expires.
            transport.compat_down = True
            main._COMPAT_RUNTIME_CHECKED_AT = 0.0
        return reply

    monkeypatch.setattr(main.requests, "get", lost_after_identity)
    events = _events(_ask(client))
    assert events[-1] == {"type": "done", "ok": False}
    [error] = [e for e in events if e["type"] == "error"]
    assert error["code"] == "generation_unreachable" and "experimental model's Ollama runtime" in error["message"]
    assert "SECRET-DETAIL" not in json.dumps(events) and COMPAT_URL not in json.dumps(events)
    assert transport.posts == [], "nothing was sent anywhere -- no fallback to the primary runtime or Gemma"
    assert not [e for e in intents.events if e[0] == "dispatch"], "no intent: the call was never dispatched"
    # The job thread returns its slot in its own `finally`, just after the last line was
    # delivered: wait (bounded) for that, rather than reading the ledger at the same instant.
    snap = _settled(ledger)
    assert (snap.unresolved_answers, snap.unresolved_helpers, snap.answers_running) == (0, 0, 0)
    assert db.inserts == []


def _settled(ledger, limit_s=5.0):
    deadline = time.monotonic() + limit_s
    snap = ledger.snapshot()
    while (snap.answers_running or snap.helpers_running) and time.monotonic() < deadline:
        time.sleep(0.01)
        snap = ledger.snapshot()
    return snap


def test_a_mismatched_compatibility_identity_refuses_before_any_model_call(monkeypatch, env, transport):
    main, client, db = env
    _install(monkeypatch, main, QWEN38_EXPERIMENTAL)
    transport.installed[COMPAT_URL]["qwen3.8:27b"] = "77" * 32
    transport.installed[PRIMARY_URL]["qwen3.8:27b"] = QWEN38_DIGEST  # the right digest on the wrong runtime
    response = _ask(client)
    assert response.status_code == 503
    assert response.json()["detail"] == "The selected answer model is not installed with its qualified identity."
    assert transport.posts == [] and f"{PRIMARY_URL}/api/tags" not in transport.gets


@pytest.mark.parametrize(
    "env, detail",
    [
        ({"enabled": False}, "The selected experimental answer model is disabled on this server."),
        ({"compat_url": None}, "The selected answer model's runtime is not configured on this server."),
    ],
    indirect=["env"],
    ids=["disabled", "no-endpoint"],
)
def test_a_persisted_experimental_selection_on_a_server_that_cannot_serve_it_refuses(monkeypatch, env, transport, detail):
    main, client, db = env
    _install(monkeypatch, main, MISTRAL35_EXPERIMENTAL)
    response = _ask(client)
    assert (response.status_code, response.json()["detail"]) == (503, detail)
    assert transport.posts == [] and transport.gets == [], "no identity lookup, no model call, no fallback"


@pytest.mark.parametrize("env", [{"enabled": False}, {"compat_url": None}], indirect=True, ids=["disabled", "no-endpoint"])
def test_gemma_keeps_answering_when_the_compatibility_runtime_is_off(monkeypatch, env, transport):
    main, client, db = env
    _install(monkeypatch, main, GEMMA_CANDIDATE)
    assert _events(_ask(client))[-1] == {"type": "done", "ok": True}
    assert {p["base"] for p in transport.posts} == {PRIMARY_URL}
    provenance = json.loads(db.inserts[0][6])
    assert provenance["generation"]["runtime_key"] == PRIMARY_RUNTIME and provenance["generation"]["think"] is False


def test_gemma_and_the_embedder_stay_on_the_primary_runtime_while_experimental_is_enabled(monkeypatch, env, transport):
    main, client, db = env
    ledger, _ = _install(monkeypatch, main, GEMMA_CANDIDATE)
    assert _events(_ask(client, model="qwen3.8:27b"))[-1] == {"type": "done", "ok": True}, (
        "the legacy model string never routes anywhere"
    )
    assert {p["base"] for p in transport.posts} == {PRIMARY_URL}
    assert not any(url.startswith(COMPAT_URL) for url in transport.gets)
    # The embedder is not generation: it stays on the primary runtime even
    # while an experimental job is the calling thread's admitted job.
    _install(monkeypatch, main, QWEN38_EXPERIMENTAL)
    ticket = main._GENERATION_LEDGER.try_acquire_helper()
    main._helper_generation(ticket)
    with generation_scope(ticket):
        main.post_ollama("/api/embed", {"model": "mxbai-embed-large:335m", "input": ["x"]}, timeout=30)
    assert transport.posts[-1]["base"] == PRIMARY_URL and transport.posts[-1]["path"] == "/api/embed"
    main._GENERATION_LEDGER.release(ticket)


def test_a_generation_payload_can_never_name_another_model_or_runtime(monkeypatch, env, transport):
    main, client, db = env
    _install(monkeypatch, main, QWEN38_EXPERIMENTAL)
    ticket = main._GENERATION_LEDGER.try_acquire_helper()
    main._helper_generation(ticket)
    with generation_scope(ticket):
        for payload in ({"model": "gemma4:31b"}, {"model": "mistral-medium-3.5:128b"}, {}):
            with pytest.raises(DispatchRefused):
                main.post_ollama("/api/generate", payload, stream=True, timeout=10)
    assert transport.posts == []
    main._GENERATION_LEDGER.release(ticket)


# --- background helpers ------------------------------------------------------------------------


class _InsightsDB:
    def __init__(self):
        self.writes = []

    def connect(self):
        db = self

        class _C:
            def cursor(self):
                class _K:
                    def execute(self, sql, params=()):
                        db.writes.append(params)

                    def close(self):
                        pass
                return _K()

            def commit(self):
                pass

            def rollback(self):
                pass

            def close(self):
                pass
        return _C()


def test_both_helpers_use_the_helper_jobs_runtime_and_bounded_thinking_allowance(monkeypatch, env, transport):
    main, client, db = env
    _install(monkeypatch, main, QWEN38_EXPERIMENTAL)
    insights = _InsightsDB()
    monkeypatch.setattr(main, "connect_to_postgres", insights.connect)
    text = "The lease between Alice and Bob runs for two years from the first of January. " * 10
    main._extract_document_insights(10, "alice-lease.txt", text)
    pairs = main._generate_qa_pairs_from_text(text * 3, "alice-lease.txt")

    helper_posts = transport.generation_posts()
    assert helper_posts and {p["base"] for p in helper_posts} == {COMPAT_URL}
    for post in helper_posts:
        assert post["stream"] is False
        assert post["payload"]["model"] == "qwen3.8:27b" and post["payload"]["think"] is True
        assert post["payload"]["options"]["num_predict"] == 2048
        assert post["timeout"][1] <= 300.0
    assert insights.writes and SECRET_THOUGHT not in json.dumps(insights.writes, default=str)
    assert pairs and SECRET_THOUGHT not in json.dumps(pairs)
    assert f"{COMPAT_URL}/api/tags" in transport.gets


def test_a_helper_on_an_unreachable_compatibility_runtime_sends_nothing(monkeypatch, env, transport, capsys):
    main, client, db = env
    _install(monkeypatch, main, QWEN38_EXPERIMENTAL)
    transport.compat_down = True
    main._extract_document_insights(10, "alice-lease.txt", "x" * 400)
    assert transport.posts == []
    printed = capsys.readouterr().out
    assert "SECRET-DETAIL" not in printed and COMPAT_URL not in printed


# --- Nemotron's own role allowances (candidate check xm-20261020T030140Z-dc2e7a43) ------------
#
# There its verifier generated exactly 512 tokens and its insights helper exactly 2,048, and
# neither reply was usable. Only those two allowances change; strict completion and parsing,
# one call per claim or excerpt, and the refusal to store an unfinished reply all stay.

_HELPER_TEXT = "The lease between Alice and Bob runs for two years from the first of January. " * 10
_NEMOTRON_ID = NEMOTRON35_EXPERIMENTAL


def _cut_off(eval_count, prompt_eval_count=1259):
    return {"thinking": SECRET_THOUGHT * 40, "response": "", "done": True, "done_reason": "length",
            "prompt_eval_count": prompt_eval_count, "eval_count": eval_count}


def _completion_lines(printed, role):
    return [line for line in printed.splitlines()
            if line.startswith(f"[Generation] completion role={role} ")]


def test_a_nemotron_answer_keeps_its_settings_and_its_verifier_gets_its_own_allowance(monkeypatch, env, transport):
    main, client, db = env
    _install(monkeypatch, main, _NEMOTRON_ID)
    assert _events(_ask(client))[-1] == {"type": "done", "ok": True}

    answer, verifier = transport.generation_posts()
    assert answer["payload"]["model"] == verifier["payload"]["model"] == "nemotron-3.5-lightning:30b"
    assert answer["payload"]["think"] is True and verifier["payload"]["think"] is True, "thinking stays on"
    assert answer["payload"]["options"] == {"temperature": 0.2, "num_ctx": 8192, "num_predict": 2048}
    assert verifier["payload"]["options"] == {"temperature": 0.0, "num_ctx": 8192, "num_predict": 2048}
    assert answer["timeout"][1] <= 180.0 and verifier["timeout"][1] <= 180.0, "deadlines unchanged"
    provenance = json.loads(db.inserts[0][6])
    assert provenance["generation"]["requested_options"]["num_predict"] == 2048
    assert provenance["verification"]["requested_options"] == {
        "temperature": 0.0, "num_ctx": 8192, "num_predict": 2048}
    assert provenance["verification"]["think"] is True


def test_a_nemotron_verifier_cut_off_at_its_allowance_is_malformed_logged_and_not_retried(
        monkeypatch, env, transport, capsys):
    main, client, db = env
    _install(monkeypatch, main, _NEMOTRON_ID)
    transport.verifier = _cut_off(2048)
    capsys.readouterr()

    events = _events(_ask(client))

    assert events[-1] == {"type": "done", "ok": True}, "a verifier reply never decides the answer"
    verification = next(e for e in events if e["type"] == "verification")
    assert [(c["status"], c["reason"]) for c in verification["claims"]] == [("unverified", "verifier_malformed")]
    assert len(transport.generation_posts()) == 2, "one answer and one verifier call: no retry"
    assert db.inserts[0][3] == _ANSWER, "the completed answer is still saved"
    printed = capsys.readouterr().out
    assert _completion_lines(printed, "verifier") == [
        f"[Generation] completion role=verifier profile={_NEMOTRON_ID} result=output_limit done_reason=length "
        "prompt_eval_count=1259 eval_count=2048 num_predict=2048 at_output_limit=yes "
        f"thinking_chars={len(SECRET_THOUGHT) * 40} response_chars=0"
    ]
    assert SECRET_THOUGHT not in printed and "two years" not in printed, "no reasoning or prompt text is logged"


@pytest.mark.parametrize(
    "reply",
    ["The claim is supported by the evidence.", "supported or unsupported", '{"status": "supported"}'],
    ids=["prose", "two-statuses", "json"],
)
def test_a_finished_but_malformed_nemotron_verdict_stays_unverified(monkeypatch, env, transport, capsys, reply):
    main, client, db = env
    _install(monkeypatch, main, _NEMOTRON_ID)
    transport.verifier = {"thinking": SECRET_THOUGHT, "response": reply, "done": True, "done_reason": "stop",
                          "prompt_eval_count": 1259, "eval_count": 700}
    capsys.readouterr()

    events = _events(_ask(client))

    verification = next(e for e in events if e["type"] == "verification")
    assert [(c["status"], c["reason"]) for c in verification["claims"]] == [("unverified", "verifier_malformed")]
    [line] = _completion_lines(capsys.readouterr().out, "verifier")
    assert " result=completed done_reason=stop " in line and " at_output_limit=no " in line
    assert f" response_chars={len(reply)}" in line and reply not in line


def test_nemotron_helpers_get_its_own_allowance_with_thinking_on(monkeypatch, env, transport):
    main, client, db = env
    _install(monkeypatch, main, _NEMOTRON_ID)
    insights = _InsightsDB()
    monkeypatch.setattr(main, "connect_to_postgres", insights.connect)

    main._extract_document_insights(10, "alice-lease.txt", _HELPER_TEXT)
    pairs = main._generate_qa_pairs_from_text(_HELPER_TEXT * 3, "alice-lease.txt")

    helper_posts = transport.generation_posts()
    assert len(helper_posts) == 3, "one insights call and one call per Q&A excerpt"
    for post in helper_posts:
        assert post["payload"]["model"] == "nemotron-3.5-lightning:30b" and post["payload"]["think"] is True
        assert post["payload"]["options"]["num_predict"] == 4096
        assert post["payload"]["options"]["num_ctx"] == 8192
        assert post["timeout"][1] <= 300.0, "the helper read timeout is unchanged"
    assert len(insights.writes) == 1 and pairs


_UNSTORABLE_INSIGHTS = {
    "cut-off-at-allowance": _cut_off(4096, prompt_eval_count=1218),
    "cut-off-mid-json": {"thinking": SECRET_THOUGHT, "response": '{"doc_type": "Lease", "keywords": ["le',
                         "done": True, "done_reason": "length", "prompt_eval_count": 1218, "eval_count": 4096},
    "finished-truncated-json": {"response": '{"doc_type": "Lease", "keywords": ["lease"', "done": True,
                                "done_reason": "stop", "prompt_eval_count": 1218, "eval_count": 900},
    "finished-no-json": {"response": "The document is a lease.", "done": True, "done_reason": "stop",
                         "prompt_eval_count": 1218, "eval_count": 900},
    "thinking-only": {"thinking": SECRET_THOUGHT, "response": "", "done": True, "done_reason": "stop",
                      "prompt_eval_count": 1218, "eval_count": 3000},
    "model-error": {"error": "SECRET-DETAIL runner crashed", "done": True, "done_reason": "stop"},
}


@pytest.mark.parametrize("case", sorted(_UNSTORABLE_INSIGHTS))
def test_an_unusable_nemotron_insights_reply_is_never_stored_or_retried(monkeypatch, env, transport, capsys, case):
    main, client, db = env
    ledger, _ = _install(monkeypatch, main, _NEMOTRON_ID)
    insights = _InsightsDB()
    monkeypatch.setattr(main, "connect_to_postgres", insights.connect)
    transport.helper = _UNSTORABLE_INSIGHTS[case]
    capsys.readouterr()

    main._extract_document_insights(10, "alice-lease.txt", _HELPER_TEXT)

    assert insights.writes == [], "nothing is stored"
    assert len(transport.generation_posts()) == 1, "one call: no automatic retry"
    printed = capsys.readouterr().out
    [line] = _completion_lines(printed, "insights")
    assert f" profile={_NEMOTRON_ID} " in line and " num_predict=4096 " in line
    if case.startswith("cut-off"):
        assert " result=output_limit done_reason=length " in line and " at_output_limit=yes " in line
        assert "[Insights] Reply was not a completed response; nothing stored" in printed
    for private in (SECRET_THOUGHT, "SECRET-DETAIL", "two years", "Lease"):
        assert private not in printed, "no prompt, reply, reasoning or error text is logged"
    # A reply with a terminal marker freed its slot; an error reply keeps its end unobserved.
    assert ledger.snapshot().unresolved_helpers == (1 if case == "model-error" else 0)


def test_a_failed_nemotron_insights_write_stores_nothing_and_is_logged(monkeypatch, env, transport, capsys):
    main, client, db = env
    _install(monkeypatch, main, _NEMOTRON_ID)
    transport.helper = dict(transport.helper, prompt_eval_count=1218, eval_count=1500)

    class _FailingConnection:
        rolled_back = False

        def cursor(self):
            class _Cursor:
                def execute(self, sql, params=()):
                    raise RuntimeError("SECRET-DETAIL constraint violated")

                def close(self):
                    pass
            return _Cursor()

        def commit(self):
            raise AssertionError("nothing may be committed")

        def rollback(self):
            _FailingConnection.rolled_back = True

        def close(self):
            pass

    monkeypatch.setattr(main, "connect_to_postgres", _FailingConnection)
    capsys.readouterr()

    main._extract_document_insights(10, "alice-lease.txt", _HELPER_TEXT)

    printed = capsys.readouterr().out
    assert _FailingConnection.rolled_back, "the uncommitted write is rolled back"
    [line] = _completion_lines(printed, "insights")
    assert " result=completed " in line and " eval_count=1500 num_predict=4096 at_output_limit=no " in line
    assert "[Insights] Extraction failed (non-critical): RuntimeError" in printed
    assert "SECRET-DETAIL" not in printed and SECRET_THOUGHT not in printed
    assert len(transport.generation_posts()) == 1


def test_a_nemotron_qa_reply_cut_off_at_its_allowance_gives_no_pairs(monkeypatch, env, transport, capsys):
    main, client, db = env
    _install(monkeypatch, main, _NEMOTRON_ID)
    transport.qa = _cut_off(4096, prompt_eval_count=1100)
    capsys.readouterr()

    assert main._generate_qa_pairs_from_text(_HELPER_TEXT * 3, "alice-lease.txt") == []

    printed = capsys.readouterr().out
    lines = _completion_lines(printed, "qa_pairs")
    assert len(lines) == 2 == len(transport.generation_posts()), "each excerpt asked once; no retry"
    assert all(" result=output_limit " in line and " num_predict=4096 " in line for line in lines)
    assert SECRET_THOUGHT not in printed


@pytest.mark.parametrize("profile_id", [QWEN38_EXPERIMENTAL, MISTRAL35_EXPERIMENTAL])
def test_qwen_and_mistral_role_calls_keep_the_shared_allowances(monkeypatch, env, transport, profile_id):
    main, client, db = env
    _install(monkeypatch, main, profile_id)
    insights = _InsightsDB()
    monkeypatch.setattr(main, "connect_to_postgres", insights.connect)
    main._extract_document_insights(10, "alice-lease.txt", _HELPER_TEXT)
    [helper] = transport.generation_posts()
    assert helper["payload"]["options"]["num_predict"] == 2048

    monkeypatch.setattr(main, "connect_to_postgres", db.connect)
    assert _events(_ask(client))[-1] == {"type": "done", "ok": True}
    _, answer, verifier = transport.generation_posts()
    assert answer["payload"]["options"]["num_predict"] == 2048
    assert verifier["payload"]["options"] == {"temperature": 0.0, "num_ctx": 8192, "num_predict": 512}


# --- inventory and loaded-model list per runtime ----------------------------------------------


def test_identity_and_residency_reads_go_to_the_named_runtime(monkeypatch, env, transport):
    main, client, db = env
    transport.loaded[COMPAT_URL] = [{"name": "qwen3.8:27b", "model": "qwen3.8:27b", "digest": QWEN38_DIGEST}]
    assert main._installed_model_digest("qwen3.8:27b", COMPAT_RUNTIME) == QWEN38_DIGEST
    assert main._installed_model_digest("qwen3.8:27b") is None, "the primary runtime does not have it"
    assert main._resident_models(COMPAT_RUNTIME)[0]["name"] == "qwen3.8:27b"
    assert main._resident_models(PRIMARY_RUNTIME) == []
    with pytest.raises(RuntimeError):
        main._runtime_base_url("elsewhere")


# --- restart with the experimental tier switched off -------------------------------------------


class _RecoveredStore(DispatchIntents):
    """A store whose acquisition finds an experimental selection persisted by
    an earlier process (seeding never overwrites it; see
    test_generation_selection.py)."""

    def __init__(self, recovery):
        self.recovery, self.default_profile_ids, self.released = recovery, [], 0

    def try_acquire(self, default_profile_id):
        self.default_profile_ids.append(default_profile_id)
        return self.recovery

    def check_owner(self):
        return True

    def release(self):
        self.released += 1

    def record_dispatch(self, intent):
        raise AssertionError("nothing is dispatched for a selection this server cannot serve")

    def record_outcome(self, intent, state):
        raise AssertionError("nothing is dispatched for a selection this server cannot serve")


@pytest.mark.parametrize(
    "env, reason, detail",
    [({"enabled": False}, INELIGIBLE_EXPERIMENTAL_DISABLED,
      "The selected experimental answer model is disabled on this server."),
     ({"compat_url": None}, INELIGIBLE_RUNTIME_NOT_CONFIGURED,
      "The selected answer model's runtime is not configured on this server.")],
    indirect=["env"], ids=["disabled", "no-endpoint"],
)
def test_a_restart_keeps_the_persisted_experimental_selection_and_refuses_to_answer_with_it(
        monkeypatch, env, transport, reason, detail):
    main, client, db = env
    persisted = SelectionRecord(MISTRAL35_EXPERIMENTAL, None, "active", 6, None, None, None, OUTCOME_COMMITTED)
    store = _RecoveredStore(OwnerRecovery(owner_seq=4, selection=persisted,
                                          unresolved_jobs={JobKind.ANSWER: 0, JobKind.HELPER: 0}, unresolved_calls=0))
    monkeypatch.setattr(main, "_new_generation_store", lambda: store)
    owner = main._GENERATION_OWNER
    assert owner.start()
    try:
        deadline = time.monotonic() + 5.0
        while owner.state is not OwnerState.OWNER:
            assert time.monotonic() < deadline, "ownership was not installed"
            time.sleep(0.005)
        default = main._MODEL_REGISTRY.get(main._MODEL_REGISTRY.default_profile_id)
        assert store.default_profile_ids == [default.profile_id], "the seed default, applied only if no row exists"
        assert not default.experimental and default.runtime_key == PRIMARY_RUNTIME
        snap = main._GENERATION_LEDGER.snapshot()
        assert snap.selection == SharedSelection(MISTRAL35_EXPERIMENTAL, 6), "kept; no automatic switch or reseed"
        assert snap.selection_job_limit == 1, "the conservative one-job policy still applies"

        response = _ask(client)
        assert (response.status_code, response.json()["detail"]) == (503, detail)
        assert transport.posts == [] and transport.gets == [], "no inventory read, model call or fallback"

        status = main._GENERATION_RUNTIME.status()
        assert status["active"]["profile_id"] == MISTRAL35_EXPERIMENTAL
        assert (status["active"]["eligible"], status["active"]["experimental"]) == (False, True)
        [mistral] = [c for c in status["choices"] if c["profile_id"] == MISTRAL35_EXPERIMENTAL]
        assert mistral["active"] is True and mistral["ineligible_reason"] == reason
        assert "://" not in json.dumps(status), "the status names runtime keys, never endpoints"
    finally:
        owner.stop()
