"""Unit contracts for generation_coordinator.py, the admin model switch.

No main import, database, model, network or subprocess: the persisted
selection is an in-memory store with the same compare-and-set rules as the
PostgreSQL one (whose SQL is proven on a real database in
tests/test_generation_selection_integration.py), the ledger is the real one,
and the inventory and loaded-model list are read-only fakes. The registry is
SYNTHETIC: Gemma's qualification gates are set here for testing only.

What these prove: a switch commits only after every admitted job drained with
its own snapshot, no unresolved work exists in memory or durably, and the
previous model left the loaded-model list by itself -- and only then admits
the new profile at a new epoch; a drain timeout, an unobserved call, durable
unresolved rows, a residency that never clears or cannot be read (a reply
without an explicit models list, or any listed entry without a usable model
identity, is unreadable; an explicitly empty list is a confirmed empty
inventory), or an admin abort each abort it, restoring the unchanged profile
at a bumped epoch that fences any late commit; a bad target identity changes
nothing; an abort is
refused rather than acknowledged while a commit holds the fence; an ambiguous
commit is settled by re-reading persisted state, and stays paused when that
read fails; a switch left by a previous process keeps admission paused until
an administrator aborts it; requests are validated in memory; and the
coordinator only ever READS the inventory and the loaded-model list.
"""

import dataclasses
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation_admission import (  # noqa: E402
    AdmissionRefusal,
    AdmissionRefused,
    AdmissionSettings,
    CallState,
    DispatchIntents,
    GenerationLedger,
    GenerationTicket,
    SharedSelection,
)
from generation_coordinator import CoordinatorSettings, SwitchCoordinator  # noqa: E402
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
_TERMINAL = {"response": "x", "done": True, "done_reason": "stop"}


def _registry(qualified=True):
    base = default_model_registry()
    gemma = dataclasses.replace(
        base.get(GEMMA_CANDIDATE),
        expected_digest=SYNTHETIC_GEMMA_DIGEST,
        qualification=Qualification(think_false_verified=True, combined_capacity_verified=qualified,
                                    quality_verified=qualified, evidence_ref="synthetic test identity"),
    )
    return ModelRegistry(profiles=(base.get(LLAMA_CONTROL), gemma), default_profile_id=LLAMA_CONTROL)


class _MemoryStore(DispatchIntents):
    """The PostgreSQL store's selection semantics, in memory."""

    def __init__(self, record=None):
        self.record = record or SelectionRecord(LLAMA_CONTROL, None, "active", 1, None, None, None, None)
        self.unresolved_rows = 0
        self.boot_id = BOOT
        self.fail = set()
        self.ambiguous = set()
        self.writes = []

    def _check(self, name):
        if name in self.fail:
            raise RuntimeError("SECRET-DETAIL connection reset")

    def record_dispatch(self, intent):
        pass

    def record_outcome(self, intent, state):
        pass

    def read_selection(self):
        self._check("read")
        return self.record

    def count_unresolved(self):
        self._check("count")
        return self.unresolved_rows

    def begin_switch(self, *, expected, target_profile_id, switch_id):
        self._check("begin")
        r = self.record
        if r.state != "active" or r.active_profile_id != expected.active_profile_id or r.epoch != expected.epoch:
            return None
        self.record = SelectionRecord(r.active_profile_id, target_profile_id, "switching", r.epoch + 1, switch_id,
                                      self.boot_id, r.last_switch_id, r.last_outcome)
        self.writes.append("begin")
        return self.record

    def commit_switch(self, *, switching):
        self._check("commit")
        r = self.record
        if (r.state, r.target_profile_id, r.epoch, r.switch_id, r.switch_owner_boot_id) != (
            "switching", switching.target_profile_id, switching.epoch, switching.switch_id,
            switching.switch_owner_boot_id,
        ):
            return None
        self.record = SelectionRecord(r.target_profile_id, None, "active", r.epoch + 1, None, None, r.switch_id,
                                      "committed")
        self.writes.append("commit")
        if "commit" in self.ambiguous:
            raise RuntimeError("the commit landed but its acknowledgement was lost")
        return self.record

    def abort_switch(self, *, switching, reason):
        self._check("abort")
        r = self.record
        if r.state != "switching" or r.switch_id != switching.switch_id or r.epoch != switching.epoch:
            return None
        self.record = SelectionRecord(r.active_profile_id, None, "active", r.epoch + 1, None, None, r.switch_id,
                                      "aborted:" + reason)
        self.writes.append("abort:" + reason)
        return self.record


class _Probes:
    """Read-only inventory and loaded-model list."""

    def __init__(self):
        self.digests = {"llama3.1:8b": LLAMA_CONTROL_DIGEST, "gemma4:31b": SYNTHETIC_GEMMA_DIGEST}
        self.loaded = [{"name": "llama3.1:8b", "model": "llama3.1:8b", "digest": LLAMA_CONTROL_DIGEST}]
        self.loaded_polls_left = 0  # polls during which the old model is still listed
        self.inventory_calls = []
        self.residency_calls = 0
        self.residency_error = None

    def installed_digest(self, tag):
        self.inventory_calls.append(tag)
        value = self.digests.get(tag)
        if isinstance(value, Exception):
            raise value
        return value

    def resident_models(self):
        self.residency_calls += 1
        if self.residency_error is not None:
            raise self.residency_error
        if self.loaded_polls_left > 0:
            self.loaded_polls_left -= 1
            return list(self.loaded)
        return []


def _setup(record=None, *, paused_by=None, recovered=None, **settings):
    store = _MemoryStore(record)
    ledger = GenerationLedger(AdmissionSettings(total_slots=2, answer_slots=2, helper_slots=1), clock=time.monotonic)
    assert ledger.install_owner(intents=store, owner_seq=1, selection=store.record.shared, paused_by=paused_by,
                                recovered=recovered or {})
    owner = SimpleNamespace(store=store, state=OwnerState.OWNER,
                            recovery=SimpleNamespace(selection=store.record))
    probes = _Probes()
    values = dict(drain_timeout_s=1.0, residency_timeout_s=1.0, residency_poll_s=0.02, check_interval_s=0.01)
    values.update(settings)
    registry = _registry()
    coordinator = SwitchCoordinator(
        ledger=ledger, owner=owner, registry_provider=lambda: registry,
        installed_digest=probes.installed_digest, resident_models=probes.resident_models,
        settings=CoordinatorSettings(**values), log=lambda line: None,
    )
    assert coordinator.start()
    return SimpleNamespace(store=store, ledger=ledger, owner=owner, probes=probes, coordinator=coordinator)


def _outcome(env):
    deadline = time.monotonic() + _WAIT
    while time.monotonic() < deadline:
        last = env.coordinator.status()["last_switch"]
        if last is not None:
            return last
        time.sleep(0.01)
    raise AssertionError("the switch did not finish")


def _phase(env, phase):
    deadline = time.monotonic() + _WAIT
    while time.monotonic() < deadline:
        switch = env.coordinator.status()["switch"]
        if switch is not None and switch["phase"] == phase:
            return
        time.sleep(0.005)
    raise AssertionError(f"the switch never reached {phase}")


def _running_answer(ledger):
    ticket = ledger.enqueue_answer(lambda: None)
    assert isinstance(ticket, GenerationTicket) and ledger.start(ticket)
    ticket.bind_snapshot(SimpleNamespace(profile_id=ticket.selection.profile_id,
                                         selection_epoch=ticket.selection.epoch, model_tag="llama3.1:8b"))
    return ticket


@pytest.fixture
def stop_all():
    envs = []
    yield envs.append
    for env in envs:
        env.coordinator.stop()


def test_a_switch_commits_only_after_drain_and_residency_then_admits_the_new_profile(stop_all):
    env = _setup()
    stop_all(env)
    old = _running_answer(env.ledger)
    env.probes.loaded_polls_left = 3

    accepted = env.coordinator.request_switch(GEMMA_CANDIDATE)
    assert (accepted.accepted, accepted.status_code) == (True, 202)
    _phase(env, "draining")
    with pytest.raises(AdmissionRefused) as refused:
        env.ledger.enqueue_answer(lambda: None)
    assert refused.value.reason is AdmissionRefusal.SWITCHING
    status = env.coordinator.status()
    assert status["active"]["profile_id"] == LLAMA_CONTROL, "active only changes at the commit"
    assert env.store.record.state == "switching" and env.store.record.epoch == 2

    old.dispatching("answer")  # the admitted answer keeps its own snapshot and finishes
    assert old.observed(_TERMINAL)
    assert old.selection == SharedSelection(LLAMA_CONTROL, 1)
    env.ledger.release(old)

    assert _outcome(env) == {"switch_id": accepted.switch_id, "target_profile_id": GEMMA_CANDIDATE,
                             "outcome": "committed", "reason": None}
    assert env.probes.residency_calls >= 4, "the barrier waited until the previous model was no longer listed"
    assert env.store.record == SelectionRecord(GEMMA_CANDIDATE, None, "active", 3, None, None, accepted.switch_id,
                                               "committed")
    assert env.store.writes == ["begin", "commit"]
    assert env.probes.inventory_calls == ["gemma4:31b", "gemma4:31b"], "target identity read before begin and commit"
    new = env.ledger.try_acquire_helper()
    assert new.selection == SharedSelection(GEMMA_CANDIDATE, 3)
    env.ledger.release(new)
    status = env.coordinator.status()
    assert status["active"] == {"profile_id": GEMMA_CANDIDATE, "display_name": "Gemma 4 31B", "model": "gemma4:31b",
                                "epoch": 3, "registered": True}
    assert status["switch"] is None and status["admission"] == "open"


@pytest.mark.parametrize("cause", ["drain-timeout", "unknown-call", "durable-unresolved", "admin-abort"])
def test_a_switch_aborts_back_to_the_unchanged_profile_at_a_new_epoch(stop_all, cause):
    env = _setup(drain_timeout_s=0.3)
    stop_all(env)
    old = _running_answer(env.ledger)
    if cause == "durable-unresolved":
        env.store.unresolved_rows = 1
    accepted = env.coordinator.request_switch(GEMMA_CANDIDATE)
    _phase(env, "draining")
    if cause == "unknown-call":
        old.dispatching("answer")  # never observed to finish
        env.ledger.release(old)
    elif cause == "durable-unresolved":
        old.dispatching("answer")
        old.observed(_TERMINAL)
        env.ledger.release(old)
    elif cause == "admin-abort":
        aborted = env.coordinator.abort_switch()
        assert (aborted.accepted, aborted.status_code) == (True, 202)

    reason = {"drain-timeout": "drain_timeout", "unknown-call": "uncertain_work",
              "durable-unresolved": "uncertain_work", "admin-abort": "admin_abort"}[cause]
    assert _outcome(env)["outcome"] == "aborted" and _outcome(env)["reason"] == reason
    assert env.store.record.active_profile_id == LLAMA_CONTROL and env.store.record.state == "active"
    assert env.store.record.epoch == 3 and env.store.record.last_outcome == "aborted:" + reason
    assert env.ledger.snapshot().selection == SharedSelection(LLAMA_CONTROL, 3)
    assert "commit" not in env.store.writes
    if not old.released:
        old.dispatching("answer")
        old.observed(_TERMINAL)
        env.ledger.release(old)
    # The abort's epoch bump fences any late commit of the aborted switch.
    stale = SelectionRecord(LLAMA_CONTROL, GEMMA_CANDIDATE, "switching", 2, accepted.switch_id, BOOT, None, None)
    assert env.store.commit_switch(switching=stale) is None


@pytest.mark.parametrize("problem", ["never-clears", "unreadable"])
def test_unresolved_residency_aborts_rather_than_admitting_the_target(stop_all, problem):
    env = _setup(residency_timeout_s=0.2)
    stop_all(env)
    if problem == "never-clears":
        env.probes.loaded_polls_left = 10_000
    else:
        env.probes.residency_error = ConnectionError("SECRET-DETAIL ps unavailable")
    env.coordinator.request_switch(GEMMA_CANDIDATE)
    outcome = _outcome(env)
    assert outcome["outcome"] == "aborted"
    assert outcome["reason"] == ("residency_timeout" if problem == "never-clears" else "residency_unavailable")
    assert env.ledger.snapshot().selection.profile_id == LLAMA_CONTROL
    assert env.store.writes == ["begin", "abort:" + outcome["reason"]]


@pytest.mark.parametrize(
    "listed",
    [
        [{}],
        [{"size": 1, "expires_at": "2026-10-10T00:05:00Z"}],
        [{"name": "", "model": "  "}],
        [{"name": 7, "model": "llama3.1:8b"}],
        [{"name": "llama3.1:8b", "model": "llama3.1:8b", "digest": "not-a-digest"}],
        [{"name": "mxbai-embed-large:latest", "model": "mxbai-embed-large:latest"}, {}],
    ],
    ids=["empty-entry", "no-identity-fields", "blank-tags", "non-string-tag", "malformed-digest", "one-bad-of-two"],
)
def test_a_listed_model_without_a_usable_identity_aborts_and_never_commits(stop_all, listed):
    env = _setup()
    stop_all(env)
    env.probes.loaded = listed
    env.probes.loaded_polls_left = 10_000
    env.coordinator.request_switch(GEMMA_CANDIDATE)
    outcome = _outcome(env)
    assert (outcome["outcome"], outcome["reason"]) == ("aborted", "residency_unavailable")
    assert env.store.writes == ["begin", "abort:residency_unavailable"], "never committed from unreadable inventory"
    assert env.ledger.snapshot().selection == SharedSelection(LLAMA_CONTROL, 3)


@pytest.mark.parametrize(
    "payload",
    [{}, {"models": [{}]}, {"models": None}, {"models": {"name": "llama3.1:8b"}}, [], None, "models",
     {"models": ["llama3.1:8b"]}, {"models": [{"model": "llama3.1:8b", "digest": None}]}],
    ids=["no-models-key", "entry-without-identity", "null-models", "models-not-a-list", "list-reply", "null-reply",
         "text-reply", "entry-not-an-object", "null-digest"],
)
def test_an_ollama_ps_reply_without_a_readable_inventory_is_refused(payload):
    from generation_coordinator import loaded_model_entries

    with pytest.raises(ValueError):
        loaded_model_entries(payload)


def test_a_confirmed_inventory_is_accepted_empty_or_with_identified_models():
    from generation_coordinator import loaded_model_entries

    assert loaded_model_entries({"models": []}) == [], "an explicitly empty list is a confirmed empty inventory"
    listed = [{"name": "llama3.1:8b", "model": "llama3.1:8b", "digest": LLAMA_CONTROL_DIGEST, "size": 1},
              {"name": "mxbai-embed-large:latest", "model": "mxbai-embed-large:latest"}]
    assert loaded_model_entries({"models": listed}) == listed


@pytest.mark.parametrize(
    "digest, reason",
    [("77" * 32, "target_identity_mismatch"), (None, "target_identity_mismatch"),
     (ConnectionError("SECRET-DETAIL"), "target_identity_unavailable")],
    ids=["re-pulled", "absent", "inventory-unavailable"],
)
def test_a_bad_target_identity_fails_without_changing_anything(stop_all, digest, reason):
    env = _setup()
    stop_all(env)
    env.probes.digests["gemma4:31b"] = digest
    env.coordinator.request_switch(GEMMA_CANDIDATE)
    assert _outcome(env)["outcome"] == "failed" and _outcome(env)["reason"] == reason
    assert env.store.writes == [] and env.store.record.epoch == 1
    assert env.ledger.snapshot().admission == "open", "nothing was paused"
    assert env.probes.residency_calls == 0


def test_an_abort_is_refused_while_a_commit_holds_the_fence(stop_all):
    env = _setup()
    stop_all(env)
    env.coordinator._fence_lock.acquire()  # a commit compare-and-set in progress
    try:
        refused = env.coordinator.abort_switch()
    finally:
        env.coordinator._fence_lock.release()
    assert (refused.accepted, refused.status_code) == (False, 409), "never acknowledged, so no late commit follows it"
    assert env.coordinator.abort_switch().status_code == 409, "nothing to abort"


def test_an_ambiguous_commit_is_settled_by_re_reading_and_stays_paused_when_unreadable(stop_all):
    env = _setup()
    stop_all(env)
    env.store.ambiguous.add("commit")
    env.coordinator.request_switch(GEMMA_CANDIDATE)
    assert _outcome(env)["outcome"] == "committed", "the landed commit is recognised from the persisted row"
    assert env.ledger.snapshot().selection == SharedSelection(GEMMA_CANDIDATE, 3)

    env2 = _setup()
    stop_all(env2)
    env2.store.ambiguous.add("commit")
    original_read = env2.store.read_selection

    def unreadable():
        if "commit" in env2.store.writes:
            raise RuntimeError("SECRET-DETAIL")
        return original_read()

    env2.store.read_selection = unreadable
    env2.coordinator.request_switch(GEMMA_CANDIDATE)
    assert _outcome(env2) ["reason"] == "commit_ambiguous"
    assert env2.ledger.snapshot().admission == "paused", "an unconfirmed commit keeps admission paused (fail closed)"


def test_a_switch_left_by_a_previous_process_stays_paused_until_aborted(stop_all):
    interrupted = SelectionRecord(LLAMA_CONTROL, GEMMA_CANDIDATE, "switching", 4,
                                  "00000000-0000-4000-8000-0000000000aa",
                                  "00000000-0000-4000-8000-0000000000bb", None, None)
    env = _setup(interrupted, paused_by=interrupted.switch_id)
    stop_all(env)
    assert env.ledger.snapshot().admission == "paused", "startup never resets switching to active"
    status = env.coordinator.status()
    assert status["switch"]["origin"] == "interrupted" and status["switch"]["target_profile_id"] == GEMMA_CANDIDATE
    assert env.coordinator.request_switch(GEMMA_CANDIDATE).status_code == 409
    assert env.coordinator.abort_switch().status_code == 202
    deadline = time.monotonic() + _WAIT
    while env.ledger.snapshot().admission != "open" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert env.store.record == SelectionRecord(LLAMA_CONTROL, None, "active", 5, None, None, interrupted.switch_id,
                                               "aborted:interrupted_switch")
    assert env.ledger.snapshot().selection == SharedSelection(LLAMA_CONTROL, 5)
    assert env.coordinator.status()["last_switch"]["reason"] == "interrupted_switch"


def test_switch_requests_are_validated_in_memory(stop_all):
    env = _setup()
    stop_all(env)
    assert env.coordinator.request_switch("qwen3_latest").status_code == 400
    assert env.coordinator.request_switch(LLAMA_CONTROL).status_code == 409, "already active"
    unqualified = _registry(qualified=False)
    env.coordinator._registry = lambda: unqualified
    assert env.coordinator.request_switch(GEMMA_CANDIDATE).status_code == 409, "an unqualified Gemma is refused"
    env.coordinator._registry = _registry
    held = _setup()
    stop_all(held)
    ticket = _running_answer(held.ledger)
    ticket.dispatching("answer")
    held.ledger.release(ticket)
    refused = held.coordinator.request_switch(GEMMA_CANDIDATE)
    assert refused.status_code == 409 and "unknown" in refused.detail
    fenced = _setup()
    stop_all(fenced)
    fenced.ledger.fence()
    assert fenced.coordinator.request_switch(GEMMA_CANDIDATE).status_code == 503
    assert env.probes.inventory_calls == [], "validation does no inventory or database work"


def test_one_switch_at_a_time(stop_all):
    env = _setup()
    stop_all(env)
    old = _running_answer(env.ledger)
    assert env.coordinator.request_switch(GEMMA_CANDIDATE).accepted
    assert env.coordinator.request_switch(GEMMA_CANDIDATE).status_code == 409
    old.dispatching("answer")
    old.observed(_TERMINAL)
    env.ledger.release(old)
    assert _outcome(env)["outcome"] == "committed"


def test_the_status_reports_choices_and_counts_without_io(stop_all):
    env = _setup()
    stop_all(env)
    status = env.coordinator.status()
    assert status["choices"] == [
        {"profile_id": LLAMA_CONTROL, "display_name": "Llama 3.1 8B", "model": "llama3.1:8b", "eligible": True,
         "active": True},
        {"profile_id": GEMMA_CANDIDATE, "display_name": "Gemma 4 31B", "model": "gemma4:31b", "eligible": True,
         "active": False},
    ]
    assert status["ownership"] == "owner" and status["settings_provisional"] is True
    assert set(status["capacity"]) == {"answers_running", "helpers_running", "unresolved_answers",
                                       "unresolved_helpers", "answer_waiters", "total_slots", "answer_slots",
                                       "helper_slots"}
    assert env.probes.inventory_calls == [] and env.probes.residency_calls == 0 and env.store.writes == []
