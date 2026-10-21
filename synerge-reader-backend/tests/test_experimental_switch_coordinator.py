"""Cross-runtime switching contracts for generation_coordinator.py.

No main import, database, model, network or subprocess: an in-memory store
with the PostgreSQL store's compare-and-set rules, the real ledger, and
read-only fakes for each runtime's installed inventory and loaded-model list.

What these prove, for the primary and compatibility runtimes:

* a switch target's identity is read on the target's OWN runtime -- a tag on
  another runtime never vouches for it -- and a mismatch fails before
  admission is paused;
* the residency barrier waits for the previous model to leave the runtime
  that served it, in both directions and between two compatibility profiles,
  and also for any other registered generation model in the shared GPU pool;
  an embedder or another client's model never blocks it, and the target's own
  model may already be resident ON ITS OWN RUNTIME only -- a copy of the
  target's model (by tag or digest) on any other runtime blocks until it
  leaves, or the switch aborts at the residency deadline;
* an unreadable loaded-model list, or a previous profile whose runtime is not
  configured, aborts with the unchanged profile at a bumped epoch; so do a
  timeout and an administrator's abort; nothing falls back to another model;
* review B2: with the experimental tier off the barrier reads only the
  primary, previous and target runtimes, so a stopped compatibility runtime
  never blocks a Gemma/Llama switch in either direction; with the tier on, or
  for a persisted experimental selection, the compatibility runtime is read;
* review B1: the bounded return to control -- an experimental profile on
  the compatibility runtime back to Gemma or Llama, when only that runtime is
  unavailable or unconfigured -- keeps polling every other runtime, commits
  only after the full 420 s window (controlled clocks; no real waiting), and
  reports ``residency_unverified``; a malformed list, an answering-but-failing
  runtime, an unreadable primary, a lingering model ever seen, uncertain
  work, a short window or keep-alive, and every switch toward or between
  experimental profiles still fail closed; an administrator's abort and the
  commit's compare-and-set fence still hold;
* in-flight jobs keep their immutable snapshot through the drain, and the
  committed experimental selection admits one job at a time;
* the coordinator only ever READS inventories and loaded-model lists: no
  unload, warm-up or service operation exists here;
* requests to disabled or unconfigured experimental profiles are refused in
  memory with fixed messages, and the status says why.
"""

import dataclasses
import os
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation_admission import (  # noqa: E402
    AdmissionSettings,
    DispatchIntents,
    GenerationLedger,
    GenerationTicket,
    SelectionPolicy,
    SharedSelection,
)
from generation_coordinator import (  # noqa: E402
    RESIDENCY_UNVERIFIED,
    CoordinatorSettings,
    RuntimeInventoryUnavailable,
    SwitchCoordinator,
    keep_alive_seconds,
)
from generation_profiles import (  # noqa: E402
    COMPAT_RUNTIME,
    GEMMA_CANDIDATE,
    GEMMA_CANDIDATE_DIGEST,
    LLAMA_CONTROL,
    LLAMA_CONTROL_DIGEST,
    MISTRAL35_DIGEST,
    MISTRAL35_EXPERIMENTAL,
    NEMOTRON35_DIGEST,
    NEMOTRON35_EXPERIMENTAL,
    PRIMARY_RUNTIME,
    QWEN38_DIGEST,
    QWEN38_EXPERIMENTAL,
    ModelRegistry,
    default_model_registry,
    shared_generation_snapshot,
)
from generation_selection import OwnerState, SelectionRecord  # noqa: E402

BOOT = "00000000-0000-4000-8000-000000000002"
BOTH = frozenset({PRIMARY_RUNTIME, COMPAT_RUNTIME})
_WAIT = 5.0
_TERMINAL = {"response": "x", "done": True, "done_reason": "stop"}
EMBEDDER = {"name": "mxbai-embed-large:335m", "model": "mxbai-embed-large:335m", "digest": "e1" * 32}
TAGS = {
    LLAMA_CONTROL: ("llama3.1:8b", LLAMA_CONTROL_DIGEST),
    GEMMA_CANDIDATE: ("gemma4:31b", GEMMA_CANDIDATE_DIGEST),
    QWEN38_EXPERIMENTAL: ("qwen3.8:27b", QWEN38_DIGEST),
    NEMOTRON35_EXPERIMENTAL: ("nemotron-3.5-lightning:30b", NEMOTRON35_DIGEST),
    MISTRAL35_EXPERIMENTAL: ("mistral-medium-3.5:128b", MISTRAL35_DIGEST),
}


def _entry(profile_id):
    tag, digest = TAGS[profile_id]
    return {"name": tag, "model": tag, "digest": digest}


class _Store(DispatchIntents):
    def __init__(self, record):
        self.record = record
        self.boot_id = BOOT
        self.writes = []

    def record_dispatch(self, intent):
        pass

    def record_outcome(self, intent, state):
        pass

    def read_selection(self):
        return self.record

    def count_unresolved(self):
        return 0

    def begin_switch(self, *, expected, target_profile_id, switch_id):
        r = self.record
        if r.state != "active" or r.active_profile_id != expected.active_profile_id or r.epoch != expected.epoch:
            return None
        self.record = SelectionRecord(r.active_profile_id, target_profile_id, "switching", r.epoch + 1, switch_id,
                                      self.boot_id, r.last_switch_id, r.last_outcome)
        self.writes.append("begin")
        return self.record

    def commit_switch(self, *, switching):
        r = self.record
        if (r.state, r.target_profile_id, r.epoch, r.switch_id) != (
            "switching", switching.target_profile_id, switching.epoch, switching.switch_id,
        ):
            return None
        self.record = SelectionRecord(r.target_profile_id, None, "active", r.epoch + 1, None, None, r.switch_id,
                                      "committed")
        self.writes.append("commit")
        return self.record

    def abort_switch(self, *, switching, reason):
        r = self.record
        if r.state != "switching" or r.switch_id != switching.switch_id or r.epoch != switching.epoch:
            return None
        self.record = SelectionRecord(r.active_profile_id, None, "active", r.epoch + 1, None, None, r.switch_id,
                                      "aborted:" + reason)
        self.writes.append("abort:" + reason)
        return self.record


class _Runtimes:
    """Per-runtime, read-only inventory and loaded-model list."""

    def __init__(self):
        self.installed = {
            PRIMARY_RUNTIME: {TAGS[p][0]: TAGS[p][1] for p in (LLAMA_CONTROL, GEMMA_CANDIDATE)},
            COMPAT_RUNTIME: {TAGS[p][0]: TAGS[p][1]
                             for p in (QWEN38_EXPERIMENTAL, NEMOTRON35_EXPERIMENTAL, MISTRAL35_EXPERIMENTAL)},
        }
        # runtime -> list of [entry, polls_left] (None: listed until changed)
        self.loaded = {PRIMARY_RUNTIME: [[EMBEDDER, None]], COMPAT_RUNTIME: []}
        self.errors = {}
        self.inventory_calls = []
        self.residency_calls = []
        self.on_read = None  # called with the runtime key before each read (controlled clocks)

    def listed_for(self, runtime_key, profile_id, polls):
        self.loaded[runtime_key].append([_entry(profile_id), polls])

    def installed_digest(self, tag, runtime_key):
        self.inventory_calls.append((tag, runtime_key))
        return self.installed[runtime_key].get(tag)

    def resident_models(self, runtime_key):
        if self.on_read is not None:
            self.on_read(runtime_key)
        self.residency_calls.append(runtime_key)
        error = self.errors.get(runtime_key)
        if error is not None:
            raise error
        listed = []
        for item in self.loaded[runtime_key]:
            entry, polls = item
            if polls is None:
                listed.append(dict(entry))
            elif polls > 0:
                item[1] = polls - 1
                listed.append(dict(entry))
        return listed


def _policy(registry):
    def resolve(selection):
        values = registry.admission_policy(selection.profile_id)
        return None if values in (None, (None, None, None)) else SelectionPolicy(*values)
    return resolve


def _setup(active=GEMMA_CANDIDATE, *, registry=None, residency_timeout_s=1.0, clock=time.monotonic, store=None):
    registry = registry or default_model_registry(experimental_enabled=True, configured_runtimes=BOTH)
    store = store or _Store(SelectionRecord(active, None, "active", 4, None, None, None, None))
    ledger = GenerationLedger(AdmissionSettings(), clock=time.monotonic, selection_policy=_policy(registry))
    assert ledger.install_owner(intents=store, owner_seq=1, selection=store.record.shared, paused_by=None,
                                recovered={})
    owner = SimpleNamespace(store=store, state=OwnerState.OWNER, recovery=SimpleNamespace(selection=store.record))
    runtimes = _Runtimes()
    logs = []
    coordinator = SwitchCoordinator(
        ledger=ledger, owner=owner, registry_provider=lambda: registry,
        installed_digest=runtimes.installed_digest, resident_models=runtimes.resident_models,
        settings=CoordinatorSettings(drain_timeout_s=2.0, residency_timeout_s=residency_timeout_s,
                                     residency_poll_s=0.01 if clock is not time.monotonic else 0.02,
                                     check_interval_s=0.01),
        clock=clock,
        log=logs.append,
    )
    assert coordinator.start()
    return SimpleNamespace(store=store, ledger=ledger, runtimes=runtimes, coordinator=coordinator, registry=registry,
                           logs=logs)


@pytest.fixture
def envs():
    made = []
    yield made
    for env in made:
        env.coordinator.stop()


def _outcome(env):
    deadline = time.monotonic() + _WAIT
    while time.monotonic() < deadline:
        last = env.coordinator.status()["last_switch"]
        if last is not None:
            return last["outcome"], last["reason"]
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


def _switch(env, target):
    outcome = env.coordinator.request_switch(target)
    assert (outcome.accepted, outcome.status_code) == (True, 202), outcome.detail
    return _outcome(env)


# --- in both directions and between compatibility profiles ----------------------------


def test_gemma_to_mistral_reads_the_target_on_its_runtime_and_waits_for_gemma_to_leave(envs):
    env = _setup(GEMMA_CANDIDATE)
    envs.append(env)
    env.runtimes.listed_for(PRIMARY_RUNTIME, GEMMA_CANDIDATE, polls=3)
    assert _switch(env, MISTRAL35_EXPERIMENTAL) == ("committed", None)
    assert env.runtimes.inventory_calls == [("mistral-medium-3.5:128b", COMPAT_RUNTIME)] * 2, (
        "target identity read on the compatibility runtime, before begin and before commit"
    )
    assert env.runtimes.residency_calls.count(PRIMARY_RUNTIME) >= 4, "waited for Gemma to leave the primary runtime"
    assert COMPAT_RUNTIME in env.runtimes.residency_calls, "the shared GPU pool's other runtime was read too"
    assert env.store.record.active_profile_id == MISTRAL35_EXPERIMENTAL and env.store.record.epoch == 6
    snap = env.ledger.snapshot()
    assert snap.selection == SharedSelection(MISTRAL35_EXPERIMENTAL, 6) and snap.selection_job_limit == 1
    status = env.coordinator.status()
    assert status["active"]["runtime"] == COMPAT_RUNTIME and status["active"]["experimental"] is True


def test_mistral_back_to_gemma_waits_for_mistral_to_leave_the_compatibility_runtime(envs):
    env = _setup(MISTRAL35_EXPERIMENTAL)
    envs.append(env)
    env.runtimes.listed_for(COMPAT_RUNTIME, MISTRAL35_EXPERIMENTAL, polls=3)
    assert _switch(env, GEMMA_CANDIDATE) == ("committed", None)
    assert env.runtimes.inventory_calls == [("gemma4:31b", PRIMARY_RUNTIME)] * 2
    assert env.runtimes.residency_calls.count(COMPAT_RUNTIME) >= 4
    assert env.ledger.snapshot().selection_job_limit is None, "Gemma's admission returns to the shared settings"


def test_between_two_compatibility_profiles_the_previous_one_must_leave(envs):
    env = _setup(QWEN38_EXPERIMENTAL)
    envs.append(env)
    env.runtimes.listed_for(COMPAT_RUNTIME, QWEN38_EXPERIMENTAL, polls=4)
    assert _switch(env, NEMOTRON35_EXPERIMENTAL) == ("committed", None)
    assert env.runtimes.residency_calls.count(COMPAT_RUNTIME) >= 5


def test_an_embedder_or_another_clients_model_never_blocks_and_the_target_may_already_be_loaded(envs):
    env = _setup(GEMMA_CANDIDATE)
    envs.append(env)
    env.runtimes.loaded[PRIMARY_RUNTIME].append([{"name": "someone-else:7b", "model": "someone-else:7b"}, None])
    env.runtimes.listed_for(COMPAT_RUNTIME, QWEN38_EXPERIMENTAL, polls=None)  # the target, already resident
    assert _switch(env, QWEN38_EXPERIMENTAL) == ("committed", None)


@pytest.mark.parametrize(
    "runtime_key, lingering",
    [(PRIMARY_RUNTIME, LLAMA_CONTROL), (COMPAT_RUNTIME, NEMOTRON35_EXPERIMENTAL)],
    ids=["llama-on-primary", "nemotron-on-compat"],
)
def test_another_registered_generation_model_in_the_shared_pool_blocks_until_timeout(envs, runtime_key, lingering):
    env = _setup(GEMMA_CANDIDATE, residency_timeout_s=0.3)
    envs.append(env)
    env.runtimes.listed_for(runtime_key, lingering, polls=None)
    assert _switch(env, QWEN38_EXPERIMENTAL) == ("aborted", "residency_timeout")
    assert env.store.record.active_profile_id == GEMMA_CANDIDATE, "the unchanged profile, at a bumped epoch"
    assert env.store.record.epoch == 6 and env.store.writes == ["begin", "abort:residency_timeout"]
    assert env.ledger.snapshot().selection == SharedSelection(GEMMA_CANDIDATE, 6)


# --- the target's own model on the wrong runtime (review R1) ---------------------------
#
# The target may already be loaded where it will SERVE. A copy of the target's model on
# any other runtime is a conflicting generator in the shared GPU pool: it blocks like
# any other registered model until it leaves by its own expiry, or the switch aborts
# at the residency deadline. Nothing is unloaded.

WRONG_RUNTIME = [
    pytest.param(GEMMA_CANDIDATE, MISTRAL35_EXPERIMENTAL, PRIMARY_RUNTIME, id="gemma-to-mistral,mistral-on-primary"),
    pytest.param(QWEN38_EXPERIMENTAL, GEMMA_CANDIDATE, COMPAT_RUNTIME, id="qwen-to-gemma,gemma-on-compat"),
]


def _drained(env, runtime_key, profile_id):
    """True once the fake no longer lists profile_id's entry on runtime_key."""
    tag = TAGS[profile_id][0]
    return all(not (entry["name"] == tag and (polls is None or polls > 0))
               for entry, polls in env.runtimes.loaded[runtime_key])


@pytest.mark.parametrize("active, target, wrong_runtime", WRONG_RUNTIME)
def test_a_target_copy_on_the_wrong_runtime_blocks_until_it_leaves(envs, active, target, wrong_runtime):
    env = _setup(active)
    envs.append(env)
    env.runtimes.listed_for(wrong_runtime, target, polls=4)
    assert _switch(env, target) == ("committed", None)
    assert _drained(env, wrong_runtime, target), "committed only after the wrong-runtime copy had left"
    assert env.runtimes.residency_calls.count(wrong_runtime) >= 5


@pytest.mark.parametrize("active, target, wrong_runtime", WRONG_RUNTIME)
def test_a_target_copy_that_stays_on_the_wrong_runtime_aborts_at_the_deadline(envs, active, target, wrong_runtime):
    env = _setup(active, residency_timeout_s=0.3)
    envs.append(env)
    env.runtimes.listed_for(wrong_runtime, target, polls=None)
    assert _switch(env, target) == ("aborted", "residency_timeout")
    assert env.store.record.active_profile_id == active, "the unchanged profile, at a bumped epoch"
    assert env.store.writes == ["begin", "abort:residency_timeout"]
    assert env.ledger.snapshot().selection == SharedSelection(active, 6)


@pytest.mark.parametrize(
    "active, target, own_runtime, wrong_runtime",
    [(GEMMA_CANDIDATE, MISTRAL35_EXPERIMENTAL, COMPAT_RUNTIME, PRIMARY_RUNTIME),
     (QWEN38_EXPERIMENTAL, GEMMA_CANDIDATE, PRIMARY_RUNTIME, COMPAT_RUNTIME)],
    ids=["mistral-on-both", "gemma-on-both"],
)
def test_a_target_on_both_runtimes_waits_only_for_the_wrong_copy(envs, active, target, own_runtime, wrong_runtime):
    env = _setup(active)
    envs.append(env)
    env.runtimes.listed_for(own_runtime, target, polls=None)  # stays: it is where the target serves
    env.runtimes.listed_for(wrong_runtime, target, polls=3)
    assert _switch(env, target) == ("committed", None)
    assert _drained(env, wrong_runtime, target) and not _drained(env, own_runtime, target)

    stuck = _setup(active, residency_timeout_s=0.3)
    envs.append(stuck)
    stuck.runtimes.listed_for(own_runtime, target, polls=None)
    stuck.runtimes.listed_for(wrong_runtime, target, polls=None)
    assert _switch(stuck, target) == ("aborted", "residency_timeout")


@pytest.mark.parametrize(
    "active, target, own_runtime",
    [(GEMMA_CANDIDATE, MISTRAL35_EXPERIMENTAL, COMPAT_RUNTIME), (QWEN38_EXPERIMENTAL, GEMMA_CANDIDATE, PRIMARY_RUNTIME)],
    ids=["mistral-on-compat", "gemma-on-primary"],
)
def test_a_target_already_resident_on_its_own_runtime_never_blocks(envs, active, target, own_runtime):
    env = _setup(active, residency_timeout_s=0.5)
    envs.append(env)
    env.runtimes.listed_for(own_runtime, target, polls=None)
    started = time.monotonic()
    assert _switch(env, target) == ("committed", None)
    assert time.monotonic() - started < 0.45, "no wait at all for the target where it serves"


@pytest.mark.parametrize(
    "where, entry, outcome",
    [
        # The target's digest under another name, on the wrong runtime: still the target's weights.
        (PRIMARY_RUNTIME, {"name": "registry.example/mistral:latest", "digest": MISTRAL35_DIGEST},
         ("aborted", "residency_timeout")),
        # The target's tag on the wrong runtime, whatever digest it reports.
        (PRIMARY_RUNTIME, {"name": "mistral-medium-3.5:128b", "digest": "ab" * 32}, ("aborted", "residency_timeout")),
        (PRIMARY_RUNTIME, {"model": "mistral-medium-3.5:128b"}, ("aborted", "residency_timeout")),
        # The same aliases where the target serves are its own copy: allowed.
        (COMPAT_RUNTIME, {"name": "registry.example/mistral:latest", "digest": MISTRAL35_DIGEST},
         ("committed", None)),
        (COMPAT_RUNTIME, {"model": "mistral-medium-3.5:128b"}, ("committed", None)),
    ],
    ids=["digest-alias-wrong", "tag-alias-wrong", "model-key-wrong", "digest-alias-own", "model-key-own"],
)
def test_target_tag_and_digest_aliases_are_judged_by_runtime(envs, where, entry, outcome):
    env = _setup(GEMMA_CANDIDATE, residency_timeout_s=0.3)
    envs.append(env)
    env.runtimes.loaded[where].append([entry, None])
    assert _switch(env, MISTRAL35_EXPERIMENTAL) == outcome


def test_embedding_only_lists_on_both_runtimes_never_block(envs):
    env = _setup(QWEN38_EXPERIMENTAL)
    envs.append(env)
    env.runtimes.loaded[COMPAT_RUNTIME].append([dict(EMBEDDER), None])
    assert _switch(env, GEMMA_CANDIDATE) == ("committed", None)


@pytest.mark.parametrize(
    "runtime_key, reply",
    [(PRIMARY_RUNTIME, [{"digest": "not-a-digest", "name": "x"}]), (COMPAT_RUNTIME, [{"size": 1}]),
     (COMPAT_RUNTIME, {"models": "nope"})],
    ids=["bad-digest", "nameless-entry", "not-a-list"],
)
def test_a_malformed_list_on_either_runtime_aborts_and_never_commits(envs, runtime_key, reply):
    env = _setup(GEMMA_CANDIDATE)
    envs.append(env)
    real = env.runtimes.resident_models
    env.coordinator._resident_models = lambda key: reply if key == runtime_key else real(key)
    assert _switch(env, MISTRAL35_EXPERIMENTAL) == ("aborted", "residency_unavailable")
    assert env.store.record.active_profile_id == GEMMA_CANDIDATE


# --- fail closed --------------------------------------------------------------------


def test_an_unreadable_compatibility_list_aborts_a_switch_in_either_direction(envs):
    to_compat = _setup(GEMMA_CANDIDATE)
    envs.append(to_compat)
    to_compat.runtimes.errors[COMPAT_RUNTIME] = RuntimeError("SECRET-DETAIL connection refused")
    assert _switch(to_compat, MISTRAL35_EXPERIMENTAL) == ("aborted", "residency_unavailable")
    assert to_compat.store.record.active_profile_id == GEMMA_CANDIDATE

    back = _setup(QWEN38_EXPERIMENTAL)
    envs.append(back)
    back.runtimes.errors[COMPAT_RUNTIME] = RuntimeError("down")
    assert _switch(back, GEMMA_CANDIDATE) == ("aborted", "residency_unavailable")
    assert back.store.record.active_profile_id == QWEN38_EXPERIMENTAL, "never switched without proof Qwen left"


def test_a_previous_profile_on_an_unconfigured_runtime_cannot_be_proven_gone(envs):
    """A persisted experimental selection on a server without its runtime,
    with a residency window SHORTER than the return to control requires
    (1 s < 420 s): switching back to Gemma is accepted but aborts at the
    barrier, because the compatibility runtime's list cannot be read here.
    With the full window the same switch commits unverified (see the
    return-to-control tests below)."""
    env = _setup(QWEN38_EXPERIMENTAL, registry=default_model_registry())
    envs.append(env)
    assert env.coordinator.status()["active"]["eligible"] is False
    assert _switch(env, GEMMA_CANDIDATE) == ("aborted", "residency_unavailable")
    assert COMPAT_RUNTIME not in env.runtimes.residency_calls
    assert env.store.record.active_profile_id == QWEN38_EXPERIMENTAL


def test_a_target_installed_only_on_another_runtime_fails_before_any_pause(envs):
    env = _setup(GEMMA_CANDIDATE)
    envs.append(env)
    del env.runtimes.installed[COMPAT_RUNTIME]["qwen3.8:27b"]
    env.runtimes.installed[PRIMARY_RUNTIME]["qwen3.8:27b"] = QWEN38_DIGEST
    assert _switch(env, QWEN38_EXPERIMENTAL) == ("failed", "target_identity_mismatch")
    assert env.runtimes.inventory_calls == [("qwen3.8:27b", COMPAT_RUNTIME)]
    assert env.store.writes == [] and env.ledger.snapshot().admission == "open"


def test_an_admin_abort_while_waiting_for_the_compatibility_runtime(envs):
    env = _setup(QWEN38_EXPERIMENTAL, residency_timeout_s=5.0)
    envs.append(env)
    env.runtimes.listed_for(COMPAT_RUNTIME, QWEN38_EXPERIMENTAL, polls=None)
    assert env.coordinator.request_switch(GEMMA_CANDIDATE).accepted
    _phase(env, "awaiting_residency")
    assert env.coordinator.abort_switch().accepted
    assert _outcome(env) == ("aborted", "admin_abort")
    assert env.store.record.active_profile_id == QWEN38_EXPERIMENTAL


def test_an_in_flight_experimental_job_keeps_its_snapshot_and_drains_before_the_switch(envs):
    env = _setup(QWEN38_EXPERIMENTAL)
    envs.append(env)
    ticket = env.ledger.enqueue_answer(lambda: None)
    assert isinstance(ticket, GenerationTicket) and env.ledger.start(ticket)
    profile = env.registry.get(QWEN38_EXPERIMENTAL)
    ticket.bind_snapshot(shared_generation_snapshot(profile, epoch=4, observed_digest=QWEN38_DIGEST,
                                                    registry=env.registry))
    assert env.coordinator.request_switch(GEMMA_CANDIDATE).accepted
    _phase(env, "draining")
    assert ticket.snapshot.runtime_key == COMPAT_RUNTIME and ticket.selection == SharedSelection(QWEN38_EXPERIMENTAL, 4)
    ticket.dispatching("answer")
    assert ticket.observed(_TERMINAL)
    env.ledger.release(ticket)
    assert _outcome(env) == ("committed", None)


def test_the_coordinator_only_reads_inventories(envs):
    env = _setup(GEMMA_CANDIDATE)
    envs.append(env)
    assert _switch(env, MISTRAL35_EXPERIMENTAL) == ("committed", None)
    assert set(env.runtimes.residency_calls) <= {PRIMARY_RUNTIME, COMPAT_RUNTIME}
    assert all(runtime in (PRIMARY_RUNTIME, COMPAT_RUNTIME) for _, runtime in env.runtimes.inventory_calls)
    assert not hasattr(env.coordinator, "_unload") and not hasattr(env.coordinator, "_warm_up")


# --- requests and status -------------------------------------------------------------


def test_disabled_or_unconfigured_experimental_targets_are_refused_in_memory(envs):
    disabled = _setup(GEMMA_CANDIDATE, registry=default_model_registry(configured_runtimes=BOTH))
    unconfigured = _setup(GEMMA_CANDIDATE, registry=default_model_registry(experimental_enabled=True))
    envs.extend([disabled, unconfigured])
    refused = disabled.coordinator.request_switch(QWEN38_EXPERIMENTAL)
    assert (refused.accepted, refused.status_code, refused.detail) == (
        False, 409, "Experimental model profiles are disabled on this server.")
    refused = unconfigured.coordinator.request_switch(MISTRAL35_EXPERIMENTAL)
    assert (refused.accepted, refused.status_code, refused.detail) == (
        False, 409, "That model profile's runtime is not configured on this server.")
    for env in (disabled, unconfigured):
        assert env.runtimes.inventory_calls == [] and env.store.writes == []


def test_the_status_reports_experimental_choices_honestly(envs):
    env = _setup(GEMMA_CANDIDATE)
    envs.append(env)
    status = env.coordinator.status()
    assert status["experimental_enabled"] is True
    choices = {c["profile_id"]: c for c in status["choices"]}
    for profile_id in (QWEN38_EXPERIMENTAL, NEMOTRON35_EXPERIMENTAL, MISTRAL35_EXPERIMENTAL):
        choice = choices[profile_id]
        assert (choice["eligible"], choice["experimental"], choice["evidence"], choice["runtime"]) == (
            True, True, "experimental", COMPAT_RUNTIME)
        assert "://" not in repr(choice), "a runtime key, never an endpoint"
    assert choices[MISTRAL35_EXPERIMENTAL]["runtime_smoke_verdict"] == "REVIEW_REQUIRED"
    assert "91.27%" in choices[MISTRAL35_EXPERIMENTAL]["review_note"]
    assert choices[QWEN38_EXPERIMENTAL]["runtime_smoke_verdict"] == "PASS"
    assert choices[GEMMA_CANDIDATE]["evidence"] == "qualified" and choices[LLAMA_CONTROL]["evidence"] == "control"


# --- review B2: with the tier off, the compatibility runtime never blocks Gemma/Llama ----------

TIER_OFF_BOTH = dict(experimental_enabled=False, configured_runtimes=BOTH)  # URL set, tier off


@pytest.mark.parametrize("active, target", [(GEMMA_CANDIDATE, LLAMA_CONTROL), (LLAMA_CONTROL, GEMMA_CANDIDATE)],
                         ids=["gemma-to-llama", "llama-to-gemma"])
@pytest.mark.parametrize("failure", [RuntimeInventoryUnavailable("down"), RuntimeError("answered 500"),
                                     ValueError("malformed")], ids=["down", "error", "malformed"])
def test_with_the_tier_off_a_failing_compat_runtime_never_blocks_a_primary_switch(envs, active, target, failure):
    env = _setup(active, registry=default_model_registry(**TIER_OFF_BOTH))
    envs.append(env)
    env.runtimes.errors[COMPAT_RUNTIME] = failure
    env.runtimes.listed_for(PRIMARY_RUNTIME, active, polls=2)
    assert _switch(env, target) == ("committed", None)
    assert COMPAT_RUNTIME not in env.runtimes.residency_calls, "nothing this backend loads can be there"
    assert env.runtimes.residency_calls.count(PRIMARY_RUNTIME) >= 3, "the primary is still read until clear"
    assert env.store.record.active_profile_id == target


def test_with_the_tier_on_the_compat_runtime_is_still_read_for_a_primary_switch(envs):
    down = _setup(GEMMA_CANDIDATE)
    envs.append(down)
    down.runtimes.errors[COMPAT_RUNTIME] = RuntimeInventoryUnavailable("down")
    assert _switch(down, LLAMA_CONTROL) == ("aborted", "residency_unavailable")
    assert down.store.record.active_profile_id == GEMMA_CANDIDATE

    lingering = _setup(GEMMA_CANDIDATE, residency_timeout_s=0.3)
    envs.append(lingering)
    lingering.runtimes.listed_for(COMPAT_RUNTIME, NEMOTRON35_EXPERIMENTAL, polls=None)
    assert _switch(lingering, LLAMA_CONTROL) == ("aborted", "residency_timeout")
    assert COMPAT_RUNTIME in lingering.runtimes.residency_calls


def test_a_persisted_experimental_selection_with_the_tier_off_still_reads_its_own_runtime(envs):
    env = _setup(QWEN38_EXPERIMENTAL, registry=default_model_registry(**TIER_OFF_BOTH))
    envs.append(env)
    env.runtimes.listed_for(COMPAT_RUNTIME, QWEN38_EXPERIMENTAL, polls=3)
    assert _switch(env, GEMMA_CANDIDATE) == ("committed", None), "verified: Qwen was seen to leave"
    assert env.runtimes.residency_calls.count(COMPAT_RUNTIME) >= 4


# --- review B1: the bounded return to control (controlled clocks) ------------------------------
#
# The clock only moves when the test moves it: by a fixed step after each residency poll (the
# coordinator's wait between polls), or not at all. Nothing waits seven real minutes.

WINDOW = 420.0


class _Clock:
    def __init__(self, step=0.0):
        self.now = 1000.0
        self.step = step
        self.primary_reads = []  # controlled time of each primary-runtime read

    def __call__(self):
        return self.now

    def on_read(self, runtime_key):
        if runtime_key == PRIMARY_RUNTIME:
            self.primary_reads.append(self.now)

    def tick(self):
        self.now += self.step


class _TimedStore(_Store):
    """Records the controlled time of the commit, and can count unresolved rows on a script."""

    def __init__(self, record, clock, unresolved=None):
        super().__init__(record)
        self.clock = clock
        self.commit_time = None
        self.unresolved = list(unresolved or [])

    def count_unresolved(self):
        return self.unresolved.pop(0) if self.unresolved else 0

    def commit_switch(self, *, switching):
        self.commit_time = self.clock.now
        return super().commit_switch(switching=switching)


def _recovery(active=QWEN38_EXPERIMENTAL, *, step=30.0, registry=None, window=WINDOW, unresolved=None,
              store_cls=_TimedStore):
    clock = _Clock(step)
    store = store_cls(SelectionRecord(active, None, "active", 4, None, None, None, None), clock, unresolved)
    env = _setup(active, registry=registry, residency_timeout_s=window, clock=clock, store=store)
    env.clock = clock
    env.runtimes.on_read = clock.on_read
    real_wait = env.coordinator._wait

    def wait_then_tick(switch, seconds):
        real_wait(switch, seconds)
        clock.tick()
    env.coordinator._wait = wait_then_tick
    return env


@pytest.mark.parametrize("target", [GEMMA_CANDIDATE, LLAMA_CONTROL], ids=["to-gemma", "to-llama"])
def test_compat_down_return_to_control_commits_unverified_only_after_the_full_window(envs, target):
    env = _recovery()
    envs.append(env)
    env.runtimes.errors[COMPAT_RUNTIME] = RuntimeInventoryUnavailable("connection refused")
    assert _switch(env, target) == ("committed", RESIDENCY_UNVERIFIED)
    started = env.clock.primary_reads[0]
    assert env.store.commit_time - started >= WINDOW, "never before the full window after draining"
    assert len(env.clock.primary_reads) >= WINDOW / 30.0, "the primary runtime was polled throughout"
    assert env.runtimes.residency_calls.count(COMPAT_RUNTIME) >= WINDOW / 30.0, "compat kept being tried"
    assert env.store.writes == ["begin", "commit"] and env.store.record.active_profile_id == target
    assert env.ledger.snapshot().selection == SharedSelection(target, 6)
    assert env.ledger.snapshot().admission == "open"
    unverified = [line for line in env.logs if "residency unverified" in line]
    assert unverified and "does not prove the previous model unloaded" in unverified[0]
    assert env.coordinator.status()["last_switch"]["reason"] == RESIDENCY_UNVERIFIED


def test_compat_unconfigured_return_to_control_commits_unverified_after_the_full_window(envs):
    """Review R3: a restart without SYNERGE_COMPAT_OLLAMA_URL kept the persisted Qwen selection."""
    env = _recovery(registry=default_model_registry())  # primary only, tier off
    envs.append(env)
    assert env.coordinator.status()["active"]["eligible"] is False
    assert _switch(env, GEMMA_CANDIDATE) == ("committed", RESIDENCY_UNVERIFIED)
    assert COMPAT_RUNTIME not in env.runtimes.residency_calls, "an unconfigured runtime is never contacted"
    assert env.store.commit_time - env.clock.primary_reads[0] >= WINDOW


def test_the_deadline_is_enforced_and_polling_continues_while_the_clock_stands_still(envs):
    env = _recovery(step=0.0)  # time does not move by itself
    envs.append(env)
    env.runtimes.errors[COMPAT_RUNTIME] = RuntimeInventoryUnavailable("down")
    assert env.coordinator.request_switch(GEMMA_CANDIDATE).accepted
    _phase(env, "awaiting_residency_unverified")
    deadline = time.monotonic() + _WAIT
    while len(env.clock.primary_reads) < 25 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(env.clock.primary_reads) >= 25, "still polling every runtime it needs"
    env.clock.now += WINDOW - 1.0  # one second short of the window
    reads = len(env.clock.primary_reads)
    while len(env.clock.primary_reads) < reads + 10 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert env.coordinator.status()["switch"]["phase"] == "awaiting_residency_unverified"
    assert env.store.writes == ["begin"], "no commit before the full window"
    env.clock.now += 1.0
    assert _outcome(env) == ("committed", RESIDENCY_UNVERIFIED)


def test_compat_coming_back_clean_during_the_window_commits_verified(envs):
    env = _recovery(step=10.0)
    envs.append(env)
    env.runtimes.errors[COMPAT_RUNTIME] = RuntimeInventoryUnavailable("down")

    def recover(runtime_key):
        env.clock.on_read(runtime_key)
        if len(env.clock.primary_reads) == 5:
            env.runtimes.errors.pop(COMPAT_RUNTIME, None)  # back, nothing loaded
    env.runtimes.on_read = recover
    assert _switch(env, GEMMA_CANDIDATE) == ("committed", None), "a readable clean list verifies residency"
    assert env.store.commit_time - env.clock.primary_reads[0] < WINDOW


def test_an_administrator_can_abort_the_unverified_wait(envs):
    env = _recovery(step=0.0)
    envs.append(env)
    env.runtimes.errors[COMPAT_RUNTIME] = RuntimeInventoryUnavailable("down")
    assert env.coordinator.request_switch(GEMMA_CANDIDATE).accepted
    _phase(env, "awaiting_residency_unverified")
    assert env.coordinator.abort_switch().accepted
    assert _outcome(env) == ("aborted", "admin_abort")
    assert env.store.record.active_profile_id == QWEN38_EXPERIMENTAL
    assert env.store.writes == ["begin", "abort:admin_abort"]
    env.clock.now += 10 * WINDOW
    time.sleep(0.05)
    assert env.store.writes == ["begin", "abort:admin_abort"], "no late commit after an acknowledged abort"


class _ConflictStore(_TimedStore):
    """The selection changed under the switch: the commit's compare-and-set fails."""

    def commit_switch(self, *, switching):
        self.commit_time = self.clock.now
        return None


def test_the_commit_fence_still_holds_for_an_unverified_return(envs):
    env = _recovery(store_cls=_ConflictStore)
    envs.append(env)
    env.runtimes.errors[COMPAT_RUNTIME] = RuntimeInventoryUnavailable("down")
    assert _switch(env, GEMMA_CANDIDATE) == ("aborted", "selection_conflict")
    assert env.store.record.active_profile_id == QWEN38_EXPERIMENTAL


# Each must stay fail-closed even with the full window and the compatibility runtime down.
def _fail_closed_cases():
    down = RuntimeInventoryUnavailable("down")
    return {
        "toward-experimental": dict(active=GEMMA_CANDIDATE, target=QWEN38_EXPERIMENTAL,
                                    errors={COMPAT_RUNTIME: down}, outcome=("aborted", "residency_unavailable")),
        "between-experimental": dict(active=QWEN38_EXPERIMENTAL, target=MISTRAL35_EXPERIMENTAL,
                                     errors={COMPAT_RUNTIME: down}, outcome=("aborted", "residency_unavailable")),
        "primary-unavailable": dict(errors={COMPAT_RUNTIME: down, PRIMARY_RUNTIME: RuntimeInventoryUnavailable("x")},
                                    outcome=("aborted", "residency_unavailable")),
        "primary-malformed": dict(errors={COMPAT_RUNTIME: down}, replies={PRIMARY_RUNTIME: [{"size": 1}]},
                                  outcome=("aborted", "residency_unavailable")),
        "compat-malformed": dict(replies={COMPAT_RUNTIME: [{"size": 1}]},
                                 outcome=("aborted", "residency_unavailable")),
        "compat-not-a-list": dict(replies={COMPAT_RUNTIME: {"models": "nope"}},
                                  outcome=("aborted", "residency_unavailable")),
        "compat-answered-with-an-error": dict(errors={COMPAT_RUNTIME: RuntimeError("HTTP 500")},
                                              outcome=("aborted", "residency_unavailable")),
        "compat-lists-qwen-until-the-deadline": dict(listed=[(COMPAT_RUNTIME, QWEN38_EXPERIMENTAL)],
                                                     outcome=("aborted", "residency_timeout")),
        "registered-model-lingers-on-primary": dict(errors={COMPAT_RUNTIME: down},
                                                    listed=[(PRIMARY_RUNTIME, MISTRAL35_EXPERIMENTAL)],
                                                    outcome=("aborted", "residency_timeout")),
        "uncertain-work-appears-in-the-window": dict(errors={COMPAT_RUNTIME: down}, unresolved=[0, 1],
                                                     outcome=("aborted", "uncertain_work")),
    }


@pytest.mark.parametrize("case", sorted(_fail_closed_cases()))
def test_everything_outside_the_bounded_return_stays_fail_closed(envs, case):
    spec = _fail_closed_cases()[case]
    active = spec.get("active", QWEN38_EXPERIMENTAL)
    env = _recovery(active, unresolved=spec.get("unresolved"))
    envs.append(env)
    env.runtimes.errors.update(spec.get("errors", {}))
    for runtime_key, profile_id in spec.get("listed", []):
        env.runtimes.listed_for(runtime_key, profile_id, polls=None)
    replies = spec.get("replies", {})
    if replies:
        real = env.runtimes.resident_models

        def resident(key):
            if key in replies:
                env.clock.on_read(key)
                return replies[key]
            return real(key)
        env.coordinator._resident_models = resident
    assert _switch(env, spec.get("target", GEMMA_CANDIDATE)) == spec["outcome"]
    assert env.store.record.active_profile_id == active, "the unchanged profile, at a bumped epoch"
    assert "commit" not in env.store.writes
    assert not any("residency unverified" in line for line in env.logs)


def test_lingering_seen_on_a_readable_list_then_unavailable_never_commits_unverified(envs):
    env = _recovery()
    envs.append(env)
    env.runtimes.listed_for(COMPAT_RUNTIME, QWEN38_EXPERIMENTAL, polls=None)

    def go_dark(runtime_key):
        env.clock.on_read(runtime_key)
        # Qwen was listed on the first three compat reads; from the fourth the runtime is gone.
        if runtime_key == COMPAT_RUNTIME and env.runtimes.residency_calls.count(COMPAT_RUNTIME) == 3:
            env.runtimes.errors[COMPAT_RUNTIME] = RuntimeInventoryUnavailable("crashed")
    env.runtimes.on_read = go_dark
    assert _switch(env, GEMMA_CANDIDATE) == ("aborted", "residency_timeout")
    assert env.store.record.active_profile_id == QWEN38_EXPERIMENTAL
    assert COMPAT_RUNTIME in env.runtimes.errors, "the runtime did go dark before the deadline"


@pytest.mark.parametrize("window, keep_alive", [(WINDOW - 1.0, "5m"), (WINDOW, "10m"), (WINDOW, "7m")],
                         ids=["window-below-420", "keep-alive-10m", "keep-alive-equals-window"])
def test_a_window_that_does_not_outlast_the_previous_keep_alive_stays_fail_closed(envs, window, keep_alive):
    base = default_model_registry(experimental_enabled=True, configured_runtimes=BOTH)
    profiles = tuple(dataclasses.replace(p, keep_alive=keep_alive) if p.profile_id == QWEN38_EXPERIMENTAL else p
                     for p in base.profiles)
    registry = ModelRegistry(profiles=profiles, default_profile_id=base.default_profile_id,
                             experimental_enabled=True, configured_runtimes=BOTH)
    env = _recovery(registry=registry, window=window)
    envs.append(env)
    env.runtimes.errors[COMPAT_RUNTIME] = RuntimeInventoryUnavailable("down")
    assert _switch(env, GEMMA_CANDIDATE) == ("aborted", "residency_unavailable")
    assert env.store.record.active_profile_id == QWEN38_EXPERIMENTAL


@pytest.mark.parametrize("value, seconds", [("5m", 300.0), ("300s", 300.0), ("1h", 3600.0), ("10m", 600.0),
                                            ("0m", None), ("5", None), ("m", None), (None, None), ("05m", None)])
def test_keep_alive_seconds(value, seconds):
    assert keep_alive_seconds(value) == seconds
