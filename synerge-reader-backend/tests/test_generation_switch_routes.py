"""Route-level contracts for the shared model switcher.

Drives the real FastAPI app through ``TestClient`` with the real ledger,
switch coordinator, owned answer job and ``post_ollama``; only the database,
the Ollama HTTP transport (``requests.post``), the persisted selection store
(in memory, with the PostgreSQL store's compare-and-set rules) and the
read-only inventory and loaded-model probes are in-memory doubles. No
database, model, network or subprocess is reached. The registry is SYNTHETIC:
Gemma's qualification gates are set here for testing only, except where a test
names the production registry.

What these prove: the admin generation routes authorize server-side -- from
the Authorization header, with the existing admin lookup -- before any
coordinator, inventory or model work, and ignore browser role claims; a
switch answers 202, keeps the committed profile active while it drains, pauses
new answers with a fixed retryable 503, and only after commit do answers,
their verifiers and helpers use the new profile at the new epoch, each model
call durably recorded before its POST; duplicate, unknown, unqualified,
same-profile and mismatched-identity requests change nothing and never fall
back to another model; an abort restores the unchanged profile at a new epoch;
the residency barrier reads Ollama's loaded-model list strictly -- a reply
without an explicit models list, or with an entry that names no model, aborts
the switch instead of committing it, while a confirmed empty list or a list of
identified models is accepted; a call whose end was never observed holds
capacity and blocks switching; a
burst of answers waiting for capacity keeps the admin and status routes
responsive and is refused with a fixed busy 503; an answer cancelled during
its identity lookup returns its slot; a verifier call that fails after it was
sent keeps the slot unresolved and stops further verifier calls while the
finished answer is still saved; and the interactive URL import is refused at
once, never queued, when no helper slot is free.

REQUIRES THE SOCKETPAIR-AWARE SIBLING GUARD (see test_main_route_harness.py).
"""

import contextlib
import dataclasses
import importlib
import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import dotenv
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation_admission import CallState, DispatchIntents, GenerationLedger, SharedSelection  # noqa: E402
from generation_coordinator import CoordinatorSettings, GenerationRuntime, SwitchCoordinator  # noqa: E402
from generation_profiles import (  # noqa: E402
    GEMMA_CANDIDATE,
    LLAMA_CONTROL,
    LLAMA_CONTROL_DIGEST,
    ModelRegistry,
    Qualification,
    default_model_registry,
)
from generation_selection import OwnerState, SelectionRecord  # noqa: E402

SYNTHETIC_GEMMA_DIGEST = "5e" * 32
BOOT = "00000000-0000-4000-8000-000000000001"
_WAIT = 5.0
_EMBEDDING_PROFILE_KEYS = (
    "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_DIMENSION", "EMBEDDING_QUERY_PREFIX",
    "EMBEDDING_DOCUMENT_PREFIX", "EMBEDDING_PROFILE_UNVERIFIED_ACK",
)
_ACTIVE_ID = "SELECT id FROM users WHERE token = %s AND COALESCE(is_active, 1) <> 0"
_ACTIVE_ADMIN = "SELECT is_admin FROM users WHERE token = %s AND COALESCE(is_active, 1) <> 0"
_SCOPE_OWNED = "SELECT id, filename, title, length(content) FROM documents WHERE user_id = %s ORDER BY id"
_LOAD_OWNED = "SELECT content FROM documents WHERE id = %s AND user_id = %s"
_USERNAME = "SELECT username FROM users WHERE token = %s"
_USERS = {"tok-admin": ("admin-id", 1), "tok-alice": ("alice-id", 0)}
_DOCUMENT = (10, "lease.txt", "alice-id", "The lease term is two years.")


def _registry(qualified=True):
    base = default_model_registry()
    gemma = dataclasses.replace(
        base.get(GEMMA_CANDIDATE), expected_digest=SYNTHETIC_GEMMA_DIGEST,
        qualification=Qualification(think_false_verified=True, combined_capacity_verified=qualified,
                                    quality_verified=qualified, evidence_ref="synthetic test identity"),
    )
    return ModelRegistry(profiles=(base.get(LLAMA_CONTROL), gemma), default_profile_id=LLAMA_CONTROL)


# --- in-memory collaborators -------------------------------------------------------


class _DB:
    def __init__(self):
        self.statements = []
        self.inserts = []
        self.fail_admin_lookup = False

    def connect(self):
        return _Connection(self)


class _Connection:
    def __init__(self, db):
        self.db = db
        self.pending = []

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
        self.connection = connection
        self.rows = []

    def execute(self, sql, params=()):
        db = self.connection.db
        norm = " ".join(sql.split())
        db.statements.append(norm)
        user = _USERS.get(params[0]) if params and isinstance(params[0], str) else None
        if norm == _ACTIVE_ADMIN:
            if db.fail_admin_lookup:
                raise RuntimeError("SECRET-DETAIL lookup failed")
            self.rows = [(user[1],)] if user else []
        elif norm == _ACTIVE_ID:
            self.rows = [(user[0],)] if user else []
        elif norm == _SCOPE_OWNED:
            self.rows = [(_DOCUMENT[0], _DOCUMENT[1], None, len(_DOCUMENT[3]))] if params[0] == _DOCUMENT[2] else []
        elif norm == _LOAD_OWNED:
            self.rows = [(_DOCUMENT[3],)] if tuple(params) == (_DOCUMENT[0], _DOCUMENT[2]) else []
        elif norm == _USERNAME:
            self.rows = [("admin",)]
        elif norm.startswith("INSERT INTO chat_history"):
            self.connection.pending.append(tuple(params))
            self.rows = [(900 + len(db.inserts) + len(self.connection.pending),)]
        else:
            raise AssertionError(f"unexpected SQL: {norm}")

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)

    def close(self):
        pass


class _Store(DispatchIntents):
    """The persisted selection and dispatch intents, in memory."""

    def __init__(self):
        self.record = SelectionRecord(LLAMA_CONTROL, None, "active", 1, None, None, None, None)
        self.boot_id = BOOT
        self.events = []
        self.open_calls = {}

    def record_dispatch(self, intent):
        self.events.append(("dispatch", intent.call_role, intent.model_tag, intent.selection_epoch))
        self.open_calls[intent.call_id] = "dispatching"

    def record_outcome(self, intent, state):
        self.events.append(("outcome", state))
        if state is CallState.UNKNOWN:
            self.open_calls[intent.call_id] = "unknown"
        else:
            self.open_calls.pop(intent.call_id, None)

    def read_selection(self):
        return self.record

    def count_unresolved(self):
        return len(self.open_calls)

    def begin_switch(self, *, expected, target_profile_id, switch_id):
        r = self.record
        if r.state != "active" or (r.active_profile_id, r.epoch) != (expected.active_profile_id, expected.epoch):
            return None
        self.record = SelectionRecord(r.active_profile_id, target_profile_id, "switching", r.epoch + 1, switch_id,
                                      BOOT, r.last_switch_id, r.last_outcome)
        return self.record

    def commit_switch(self, *, switching):
        r = self.record
        if (r.state, r.switch_id, r.epoch, r.target_profile_id) != ("switching", switching.switch_id, switching.epoch,
                                                                   switching.target_profile_id):
            return None
        self.record = SelectionRecord(r.target_profile_id, None, "active", r.epoch + 1, None, None, r.switch_id,
                                      "committed")
        return self.record

    def abort_switch(self, *, switching, reason):
        r = self.record
        if r.state != "switching" or (r.switch_id, r.epoch) != (switching.switch_id, switching.epoch):
            return None
        self.record = SelectionRecord(r.active_profile_id, None, "active", r.epoch + 1, None, None, r.switch_id,
                                      "aborted:" + reason)
        return self.record


class _Probes:
    def __init__(self):
        self.digests = {"llama3.1:8b": LLAMA_CONTROL_DIGEST, "gemma4:31b": SYNTHETIC_GEMMA_DIGEST}
        self.resident = []
        self.inventory = []
        self.residency_reads = 0

    def installed_digest(self, tag):
        self.inventory.append(tag)
        return self.digests.get(tag)

    def resident_models(self):
        self.residency_reads += 1
        return [{"name": tag, "model": tag} for tag in self.resident]


class _Ollama:
    """``requests.post`` for /api/generate. Streams are NDJSON bytes; a gate
    holds a stream open; ``final`` False ends a stream without its final
    record (the server may still be working)."""

    def __init__(self, store):
        self.store = store
        self.posts = []
        self.gate = None
        self.final = True
        self.started = threading.Event()

    def __call__(self, url, json=None, stream=False, timeout=None):
        self.posts.append({"model": json["model"], "stream": stream, "timeout": timeout, "think": json.get("think"),
                           "recorded_before": list(self.store.events)})
        if url.endswith("/api/generate") and stream:
            return _Stream(self)
        return _Reply({"response": "supported", "done": True, "done_reason": "stop"})


class _Stream:
    def __init__(self, ollama):
        self.ollama = ollama

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, decode_unicode=True, chunk_size=32):
        yield (json.dumps({"response": "The lease term is two years [C1]."}) + "\n").encode()
        self.ollama.started.set()
        if self.ollama.gate is not None:
            assert self.ollama.gate.wait(_WAIT)
        if self.ollama.final:
            yield (json.dumps({"response": "", "done": True, "done_reason": "stop"}) + "\n").encode()


class _Reply:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return dict(self.body)


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
def env(monkeypatch):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    for key in _EMBEDDING_PROFILE_KEYS:
        monkeypatch.delenv(key, raising=False)
    with _imported_main() as main:
        db, store, probes = _DB(), _Store(), _Probes()
        ollama = _Ollama(store)
        ledger = GenerationLedger(main._ADMISSION_SETTINGS, clock=time.monotonic)
        assert ledger.install_owner(intents=store, owner_seq=1, selection=store.record.shared, paused_by=None,
                                    recovered={})
        owner = SimpleNamespace(store=store, state=OwnerState.OWNER, recovery=SimpleNamespace(selection=store.record),
                                start=lambda: True, stop=ledger.fence)
        coordinator = SwitchCoordinator(
            ledger=ledger, owner=owner, registry_provider=lambda: main._MODEL_REGISTRY,
            installed_digest=probes.installed_digest, resident_models=probes.resident_models,
            settings=CoordinatorSettings(drain_timeout_s=2.0, residency_timeout_s=1.0, residency_poll_s=0.02,
                                         check_interval_s=0.01),
            log=lambda line: None,
        )
        runtime = GenerationRuntime(ledger=ledger, owner=owner, coordinator=coordinator)
        monkeypatch.setattr(main, "_MODEL_REGISTRY", _registry())
        monkeypatch.setattr(main, "_GENERATION_LEDGER", ledger)
        monkeypatch.setattr(main, "_GENERATION_RUNTIME", runtime)
        monkeypatch.setattr(main, "_initialize_application", lambda: coordinator.start())
        monkeypatch.setattr(main, "connect_to_postgres", db.connect)
        monkeypatch.setattr(main, "get_active_ollama_base_url", lambda: "http://ollama.invalid")
        monkeypatch.setattr(main.requests, "post", ollama)
        monkeypatch.setattr(main, "_installed_model_digest", probes.installed_digest)
        client = TestClient(main.app)
        with client:
            yield SimpleNamespace(main=main, client=client, db=db, store=store, probes=probes, ollama=ollama,
                                  ledger=ledger, coordinator=coordinator)


def _admin(path, env, method="get", token="tok-admin", **kwargs):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return getattr(env.client, method)(path, headers=headers, **kwargs)


def _ask(env, token="tok-alice"):
    response = env.client.post("/ask", json={"question": "How long is the lease term?", "mode": "document_qa",
                                             "auth_token": token, "active_document_id": 10})
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    return response, events


def _wait_for(predicate):
    deadline = time.monotonic() + _WAIT
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.01)


def _status(env):
    response = _admin("/admin/generation", env)
    assert response.status_code == 200, response.text
    return response.json()


# --- authorization --------------------------------------------------------------------


_ROUTES = [("get", "/admin/generation", None), ("post", "/admin/generation/select", {"profile_id": GEMMA_CANDIDATE}),
           ("post", "/admin/generation/abort", None)]


@pytest.mark.parametrize("method, path, body", _ROUTES, ids=["status", "select", "abort"])
@pytest.mark.parametrize("case", ["no-header", "non-admin", "query-token", "forged-role", "lookup-fails"])
def test_generation_routes_authorize_server_side_before_any_work(env, method, path, body, case):
    kwargs = {"json": body} if body is not None else {}
    if case == "no-header":
        response = getattr(env.client, method)(path, **kwargs)
        expected = 401
    elif case == "non-admin":
        response = _admin(path, env, method, token="tok-alice", **kwargs)
        expected = 403
    elif case == "query-token":
        response = getattr(env.client, method)(path, params={"token": "tok-admin"}, **kwargs)
        expected = 401
    elif case == "forged-role":
        claims = {"Authorization": "Bearer tok-alice", "X-Role": "admin"}
        if method == "get":
            response = env.client.get(path, params={"is_admin": "true", "role": "admin"}, headers=claims)
        else:
            response = env.client.post(path, json=dict(body or {}, is_admin=True, role="admin"), headers=claims)
        expected = 403
    else:
        env.db.fail_admin_lookup = True
        response = _admin(path, env, method, **kwargs)
        expected = 503
    assert response.status_code == expected
    assert "SECRET" not in response.text
    assert env.probes.inventory == [] and env.probes.residency_reads == 0, "no privileged work before authorization"
    assert env.coordinator.status()["switch"] is None and env.store.record.epoch == 1
    assert env.ollama.posts == []


# --- a switch, end to end ---------------------------------------------------------------


def test_a_switch_drains_with_the_old_snapshot_then_answers_verifies_and_helps_with_the_new_one(env):
    status = _status(env)
    assert status["active"]["profile_id"] == LLAMA_CONTROL and status["admission"] == "open"
    assert {c["profile_id"]: c["eligible"] for c in status["choices"]} == {LLAMA_CONTROL: True, GEMMA_CANDIDATE: True}

    # Old work in flight: a helper admitted under the Llama selection.
    old = env.ledger.try_acquire_helper()
    env.probes.resident = ["llama3.1:8b"]

    accepted = _admin("/admin/generation/select", env, "post", json={"profile_id": GEMMA_CANDIDATE})
    assert accepted.status_code == 202
    switch_id = accepted.json()["switch_id"]
    assert _admin("/admin/generation/select", env, "post", json={"profile_id": GEMMA_CANDIDATE}).status_code == 409
    _wait_for(lambda: (_status(env)["switch"] or {}).get("phase") == "draining")
    assert _status(env)["active"]["profile_id"] == LLAMA_CONTROL, "not active before the commit"

    paused, _ = _ask(env)
    assert paused.status_code == 503 and paused.headers["retry-after"] == "2"
    assert paused.json() == {"detail": "The answer model is being switched. Please retry shortly."}
    assert old.selection == SharedSelection(LLAMA_CONTROL, 1), "admitted work keeps its own selection"

    env.ledger.release(old)  # drained; the previous model is still listed as loaded
    _wait_for(lambda: (_status(env)["switch"] or {}).get("phase") == "awaiting_residency")
    assert env.store.record.state == "switching"
    env.probes.resident = []  # natural keep-alive expiry
    _wait_for(lambda: _status(env)["last_switch"] is not None)

    status = _status(env)
    assert status["last_switch"] == {"switch_id": switch_id, "target_profile_id": GEMMA_CANDIDATE,
                                     "outcome": "committed", "reason": None}
    assert status["active"]["profile_id"] == GEMMA_CANDIDATE and status["active"]["epoch"] == 3

    response, events = _ask(env)
    assert response.status_code == 200 and events[-1] == {"type": "done", "ok": True}
    [evidence] = [e for e in events if e["type"] == "evidence"]
    assert evidence["generation"] == {"profile_id": GEMMA_CANDIDATE, "model": "gemma4:31b"}
    answer, verifier = env.ollama.posts
    assert (answer["model"], answer["stream"], answer["think"]) == ("gemma4:31b", True, False)
    assert (verifier["model"], verifier["stream"], verifier["think"]) == ("gemma4:31b", False, False)
    assert answer["recorded_before"][-1] == ("dispatch", "answer", "gemma4:31b", 3), "intent before the POST"
    assert verifier["recorded_before"][-1] == ("dispatch", "verifier", "gemma4:31b", 3)
    [(_, _, _, _, _, _, provenance)] = env.db.inserts
    generation = json.loads(provenance)["generation"]
    assert (generation["selection"], generation["selection_epoch"]) == ("shared_active", 3)
    _wait_for(lambda: env.ledger.snapshot().answers_running == 0)
    assert env.store.count_unresolved() == 0, "both observed calls retired their intents"

    env.main._extract_document_insights(10, "lease.txt", "The lease term is two years. " * 20)
    assert env.ollama.posts[-1]["model"] == "gemma4:31b", "a helper admitted after the commit uses the new model"


def test_an_abort_restores_the_unchanged_profile_at_a_new_epoch(env):
    old = env.ledger.try_acquire_helper()
    assert _admin("/admin/generation/select", env, "post", json={"profile_id": GEMMA_CANDIDATE}).status_code == 202
    _wait_for(lambda: (_status(env)["switch"] or {}).get("phase") == "draining")
    assert _admin("/admin/generation/abort", env, "post").status_code == 202
    _wait_for(lambda: _status(env)["last_switch"] is not None)
    status = _status(env)
    assert status["last_switch"]["outcome"] == "aborted" and status["last_switch"]["reason"] == "admin_abort"
    assert (status["active"]["profile_id"], status["active"]["epoch"]) == (LLAMA_CONTROL, 3)
    assert env.store.record.last_outcome == "aborted:admin_abort"
    assert _admin("/admin/generation/abort", env, "post").status_code == 409, "nothing left to abort"
    env.ledger.release(old)
    response, events = _ask(env)
    assert response.status_code == 200 and env.ollama.posts[0]["model"] == "llama3.1:8b"


def _ps_entry(tag, digest):
    return {"name": tag, "model": tag, "size": 1, "digest": digest, "expires_at": "2026-10-10T00:05:00Z",
            "size_vram": 1}


@pytest.mark.parametrize(
    "payload, outcome, reason",
    [
        ({}, "aborted", "residency_unavailable"),
        ({"models": [{}]}, "aborted", "residency_unavailable"),
        ({"models": []}, "committed", None),
        ({"models": [_ps_entry("mxbai-embed-large:latest", "ab" * 32)]}, "committed", None),
        ({"models": [_ps_entry("llama3.1:8b", LLAMA_CONTROL_DIGEST)]}, "aborted", "residency_timeout"),
    ],
    ids=["no-models-key", "entry-without-identity", "confirmed-empty", "other-model-resident",
         "previous-model-resident"],
)
def test_the_residency_barrier_trusts_only_a_readable_ollama_inventory(env, monkeypatch, payload, outcome, reason):
    reads = []

    def ps(url, timeout=None):
        reads.append(url)
        return _Reply(payload)

    monkeypatch.setattr(env.main.requests, "get", ps)
    monkeypatch.setattr(env.coordinator, "_resident_models", env.main._resident_models)
    accepted = _admin("/admin/generation/select", env, "post", json={"profile_id": GEMMA_CANDIDATE})
    assert accepted.status_code == 202
    _wait_for(lambda: _status(env)["last_switch"] is not None)

    status = _status(env)
    assert status["last_switch"] == {"switch_id": accepted.json()["switch_id"], "target_profile_id": GEMMA_CANDIDATE,
                                     "outcome": outcome, "reason": reason}
    committed = outcome == "committed"
    assert (status["active"]["profile_id"], status["active"]["epoch"]) == (
        GEMMA_CANDIDATE if committed else LLAMA_CONTROL, 3), "unreadable inventory never commits the switch"
    assert env.store.record.last_outcome == ("committed" if committed else "aborted:" + reason)
    assert status["admission"] == "open"
    assert reads and set(reads) == {"http://ollama.invalid/api/ps"}, "only the read-only loaded-model list"
    assert env.ollama.posts == []


@pytest.mark.parametrize("target, status_code", [("qwen3_latest", 400), (LLAMA_CONTROL, 409)],
                         ids=["unknown", "already-active"])
def test_an_unknown_or_already_active_profile_is_refused(env, target, status_code):
    assert _admin("/admin/generation/select", env, "post", json={"profile_id": target}).status_code == status_code
    assert env.store.record.epoch == 1 and env.probes.inventory == []


def test_an_unqualified_profile_is_refused_and_a_mismatched_identity_fails_without_fallback(env, monkeypatch):
    monkeypatch.setattr(env.main, "_MODEL_REGISTRY", _registry(qualified=False))
    assert _admin("/admin/generation/select", env, "post", json={"profile_id": GEMMA_CANDIDATE}).status_code == 409
    # The released production Gemma is eligible, but only with its pinned identity.
    monkeypatch.setattr(env.main, "_MODEL_REGISTRY", default_model_registry())
    assert {c["profile_id"]: c["eligible"] for c in _status(env)["choices"]} == {LLAMA_CONTROL: True,
                                                                                  GEMMA_CANDIDATE: True}
    env.probes.digests["gemma4:31b"] = "77" * 32
    assert _admin("/admin/generation/select", env, "post", json={"profile_id": GEMMA_CANDIDATE}).status_code == 202
    _wait_for(lambda: _status(env)["last_switch"] is not None)
    assert _status(env)["last_switch"]["reason"] == "target_identity_mismatch"
    assert env.store.record.epoch == 1 and _status(env)["admission"] == "open"
    response, _ = _ask(env)
    assert response.status_code == 200 and env.ollama.posts[0]["model"] == "llama3.1:8b"


def test_a_call_whose_end_was_never_observed_holds_capacity_and_blocks_switching(env):
    env.ollama.final = False  # the stream ends without its final record
    response, events = _ask(env)
    assert events[-1] == {"type": "done", "ok": False} and env.db.inserts == []
    _wait_for(lambda: env.ledger.snapshot().answers_running == 0)
    status = _status(env)
    assert status["capacity"]["unresolved_answers"] == 1
    assert env.store.events[-1] == ("outcome", CallState.UNKNOWN) and env.store.count_unresolved() == 1
    refused = _admin("/admin/generation/select", env, "post", json={"profile_id": GEMMA_CANDIDATE})
    assert refused.status_code == 409 and "unknown" in refused.json()["detail"]


def test_a_burst_of_waiting_answers_keeps_the_admin_routes_responsive(env):
    env.ollama.gate = threading.Event()
    results = []

    def ask():
        results.append(_ask(env)[0].status_code)

    holders = [threading.Thread(target=ask) for _ in range(2)]
    for thread in holders:
        thread.start()
    _wait_for(lambda: env.ledger.snapshot().answers_running == 2)
    waiters = [threading.Thread(target=ask) for _ in range(6)]
    for thread in waiters:
        thread.start()
    _wait_for(lambda: env.ledger.snapshot().answer_waiters == env.main._ADMISSION_SETTINGS.answer_waiters)
    started = time.monotonic()
    status = _status(env)
    version = env.client.get("/version")
    assert time.monotonic() - started < 1.0, "admin and public routes answer while answers wait"
    assert version.status_code == 200 and status["capacity"]["answer_waiters"] == 4
    for thread in waiters:
        thread.join(_WAIT)
    assert sorted(results) == [503] * 6, "two refused at once (busy), four after the bounded wait"
    env.ollama.gate.set()
    for thread in holders:
        thread.join(_WAIT)
    assert sorted(results) == [200, 200] + [503] * 6


def test_the_url_import_is_refused_at_once_when_no_helper_slot_is_free(env, monkeypatch):
    monkeypatch.setattr(env.main, "_safe_fetch_url", lambda url: SimpleNamespace(
        raise_for_status=lambda: None, headers={"Content-Type": "text/html"},
        raw=SimpleNamespace(read=lambda *a, **k: ("<p>" + "The lease term is two years. " * 30 + "</p>").encode()),
    ))
    held = env.ledger.try_acquire_helper()
    started = time.monotonic()
    response = env.client.post("/knowledge_base/import_url", json={"url": "https://example.org/a", "token": "tok-admin"})
    assert response.status_code == 503 and response.headers["retry-after"] == "2"
    assert time.monotonic() - started < 1.0 and env.ollama.posts == [], "refused at once, nothing sent"
    env.ledger.release(held)
    monkeypatch.setattr(env.main, "_save_kb_pairs", lambda pairs, *args: len(pairs))
    env.ollama.posts.clear()

    def reply(url, json=None, stream=False, timeout=None):
        env.ollama.posts.append({"model": json["model"]})
        return _Reply({"response": "Q: What is the term?\nA: The term is two years in total.", "done": True,
                       "done_reason": "stop"})

    monkeypatch.setattr(env.main.requests, "post", reply)
    response = env.client.post("/knowledge_base/import_url", json={"url": "https://example.org/a", "token": "tok-admin"})
    assert response.status_code == 200 and response.json()["added"] >= 1
    assert {post["model"] for post in env.ollama.posts} == {"llama3.1:8b"}
    _wait_for(lambda: env.ledger.snapshot().helpers_running == 0)


def _select_gemma_now(env):
    env.ledger.pause("switch-under-test")
    env.ledger.resume("switch-under-test", SharedSelection(GEMMA_CANDIDATE, 3))


def test_cancelling_an_answer_during_its_identity_lookup_returns_its_slot(env, monkeypatch):
    import asyncio

    from schemas import AskRequest

    _select_gemma_now(env)
    entered, release = threading.Event(), threading.Event()

    def slow_lookup(tag):
        entered.set()
        assert release.wait(_WAIT)
        return SYNTHETIC_GEMMA_DIGEST

    monkeypatch.setattr(env.main, "_installed_model_digest", slow_lookup)
    monkeypatch.setattr(env.main, "_resolve_caller_id", lambda token: "alice-id")

    async def ask_then_cancel():
        task = asyncio.ensure_future(env.main.ask_question(AskRequest(question="q", auth_token="tok-alice")))
        while not entered.is_set():
            await asyncio.sleep(0.01)
        assert env.ledger.snapshot().answers_running == 1, "the slot is held while the identity is read"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(ask_then_cancel())
    release.set()
    assert env.ledger.snapshot().answers_running == 0, "the cancelled answer returned its slot"
    assert env.ollama.posts == [] and env.store.events == [], "nothing was dispatched"


def test_a_verifier_failure_after_dispatch_holds_the_slot_and_stops_further_verifier_calls(env, monkeypatch):
    def flaky(url, json=None, stream=False, timeout=None):
        env.ollama.posts.append({"stream": stream})
        if stream:
            return _TwoClaimStream()
        raise ConnectionError("SECRET-DETAIL reset after the request was sent")

    monkeypatch.setattr(env.main.requests, "post", flaky)
    response, events = _ask(env)

    assert events[-1] == {"type": "done", "ok": True}, "a verifier failure never decides the answer"
    [verification] = [e for e in events if e["type"] == "verification"]
    assert [c["status"] for c in verification["claims"]] == ["unverified", "unverified"]
    assert [post["stream"] for post in env.ollama.posts] == [True, False], "no second verifier call after it"
    assert "SECRET" not in response.text and len(env.db.inserts) == 1
    _wait_for(lambda: env.ledger.snapshot().answers_running == 0)
    assert env.ledger.snapshot().unresolved_answers == 1, "the unobserved verifier call keeps the slot"
    assert env.store.count_unresolved() == 1


class _TwoClaimStream(_Stream):
    def __init__(self):
        pass

    def iter_content(self, decode_unicode=True, chunk_size=32):
        text = "The lease term is two years [C1]. Rent is due monthly [C1]."
        yield (json.dumps({"response": text}) + "\n").encode()
        yield (json.dumps({"response": "", "done": True, "done_reason": "stop"}) + "\n").encode()
