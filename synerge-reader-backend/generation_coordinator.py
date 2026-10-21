"""The administrator's shared model switch, run by one dedicated coordinator.

A switch request is validated in memory and queued; the HTTP route answers
202 at once. The coordinator's own thread -- never a request worker or the
event loop -- then, in order:

1. reads the target's installed identity (read-only inventory) on the
   target's OWN runtime and refuses a target whose tag is absent, re-pulled
   or unreadable there;
2. pauses new generation admission in this process, then moves the persisted
   selection active -> switching with a compare-and-set on the exact state,
   active profile and epoch it paused;
3. waits, bounded, for every admitted answer and helper job to finish with
   the snapshot it was admitted under -- nothing is rerouted or cancelled;
   any job that ends with an unobserved model call, and any unresolved call
   row in the database, aborts the switch;
4. waits, bounded, for the previous generation model to leave the
   loaded-model list of the runtime that served it, by its own keep-alive
   expiry (read-only ``/api/ps``). The runtimes share one GPU pool, so the
   loaded-model list of EVERY configured runtime is read on each poll, and
   the barrier also waits for any other registered generation model (the
   target's own excepted) to leave any of them; querying only the
   destination would miss a model still holding memory elsewhere. An
   embedding model or another client's model is not a registered generation
   model and never blocks the barrier: no list has to become empty. An empty
   job count is not proof that weights left GPU memory, so a model that stays
   listed, or a list that cannot be read -- a reply without an explicit
   ``models`` list, any entry without a usable model identity, or a runtime
   the previous profile needs that is not configured -- aborts the switch.
   No unload, warm-up or service operation is ever sent, so a committed
   switch selects the model; it does not claim the model was loaded;
5. re-reads the target identity and commits switching -> active with a
   compare-and-set on state, target, epoch, switch id and switch-owner boot
   id; only then is admission reopened with the new profile and epoch.

An abort (administrator, timeout or uncertainty) moves switching -> active
with the unchanged profile and a bumped epoch, which fences any late commit.
The administrator's abort and the commit serialize on one fence lock, so an
acknowledged abort can never be followed by a commit. After an ambiguous
database outcome the persisted state is re-read; if it cannot be read the
process stays paused (fail closed).

Preconditions this module cannot enforce: Ollama serves this one application
(or other clients are coordinated by the operator), and embedding weights are
outside its control. All waits and polls are bounded and every value here is
provisional, not measured on the production host.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from generation_admission import GenerationLedger, JobKind, SharedSelection
from generation_profiles import (
    INELIGIBLE_EXPERIMENTAL_DISABLED,
    INELIGIBLE_RUNTIME_NOT_CONFIGURED,
    ModelIdentityMismatchError,
    ModelSelectionError,
    check_installed_identity,
    normalize_digest,
)

_INELIGIBLE_DETAILS = {
    INELIGIBLE_EXPERIMENTAL_DISABLED: "Experimental model profiles are disabled on this server.",
    INELIGIBLE_RUNTIME_NOT_CONFIGURED: "That model profile's runtime is not configured on this server.",
}
from generation_selection import (
    STATE_ACTIVE,
    STATE_SWITCHING,
    GenerationOwner,
    SelectionRecord,
)

PHASES = ("queued", "verifying_target", "draining", "awaiting_residency", "committing", "aborting")
OUTCOMES = ("committed", "aborted", "failed")


@dataclass(frozen=True)
class CoordinatorSettings:
    """Provisional, unmeasured bounds for one switch."""

    drain_timeout_s: float = 480.0
    residency_timeout_s: float = 420.0
    residency_poll_s: float = 5.0
    check_interval_s: float = 0.25

    def __post_init__(self) -> None:
        for name, low, high in (
            ("drain_timeout_s", 0.01, 3600.0),
            ("residency_timeout_s", 0.01, 3600.0),
            ("residency_poll_s", 0.01, 300.0),
            ("check_interval_s", 0.005, 10.0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
                raise ValueError(f"{name} must be a number in [{low}, {high}]")


@dataclass(frozen=True)
class SwitchRequestOutcome:
    accepted: bool
    status_code: int
    detail: str
    switch_id: Optional[str] = None


class _Switch:
    __slots__ = ("switch_id", "target_profile_id", "phase", "abort_requested", "old", "switching",
                 "committed", "outcome", "reason")

    def __init__(self, target_profile_id: str) -> None:
        self.switch_id = str(uuid.uuid4())
        self.target_profile_id = target_profile_id
        self.phase = "queued"
        self.abort_requested = False
        self.old: Optional[SharedSelection] = None
        self.switching: Optional[SelectionRecord] = None
        self.committed = False
        self.outcome: Optional[str] = None
        self.reason: Optional[str] = None


_AMBIGUOUS = object()


def _listed_identity(entry: object) -> tuple:
    """The (tags, digest) one loaded-model entry names. ValueError when the
    entry is malformed -- not an object, a tag that is not a string, a digest
    that is not a digest -- or names no model tag at all."""
    if not isinstance(entry, dict):
        raise ValueError("unreadable loaded-model entry")
    tags = []
    for key in ("name", "model"):
        if key in entry:
            value = entry[key]
            if not isinstance(value, str):
                raise ValueError("unreadable loaded-model entry")
            if value.strip():
                tags.append(value)
    digest = None
    if "digest" in entry:
        digest = normalize_digest(entry["digest"])
        if digest is None:
            raise ValueError("unreadable loaded-model entry")
    if not tags:
        raise ValueError("loaded-model entry without a model identity")
    return tuple(tags), digest


def loaded_model_entries(payload: object) -> list:
    """The entries of one ``/api/ps`` reply, or ValueError when the reply is
    not a readable inventory. A successful HTTP response proves nothing by
    itself: the reply must carry an explicit ``models`` list -- an explicitly
    empty list is a confirmed empty inventory -- and every listed entry must
    name its model."""
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("unreadable loaded-model list")
    models = payload["models"]
    for entry in models:
        _listed_identity(entry)
    return models


class SwitchCoordinator:
    """``installed_digest(tag, runtime_key)`` reads one runtime's installed
    identity of ``tag``; ``resident_models(runtime_key)`` reads one runtime's
    loaded-model list. Both are read-only and are always called with the
    runtime a profile actually uses."""

    def __init__(
        self,
        *,
        ledger: GenerationLedger,
        owner: GenerationOwner,
        registry_provider: Callable[[], object],
        installed_digest: Callable[[str, str], Optional[str]],
        resident_models: Callable[[str], list],
        settings: CoordinatorSettings = CoordinatorSettings(),
        clock: Callable[[], float] = time.monotonic,
        thread_factory: Callable[..., object] = threading.Thread,
        log: Callable[[str], None] = print,
    ) -> None:
        self._ledger = ledger
        self._owner = owner
        self._registry = registry_provider
        self._installed_digest = installed_digest
        self._resident_models = resident_models
        self._settings = settings
        self._clock = clock
        self._thread_factory = thread_factory
        self._log = log
        self._lock = threading.Lock()
        self._work = threading.Condition(self._lock)
        # Serializes the commit compare-and-set against an administrator's
        # abort, so an acknowledged abort can never be followed by a commit.
        self._fence_lock = threading.Lock()
        self._pending: Optional[_Switch] = None
        self._last: Optional[dict] = None
        self._interrupted_abort = False
        self._started = False
        self._stopping = False

    # --- lifecycle -------------------------------------------------------------------

    def start(self) -> bool:
        with self._lock:
            if self._started:
                return False
            self._started = True
        try:
            thread = self._thread_factory(target=self._run, name="generation-coordinator", daemon=True)
            thread.start()
        except Exception:
            with self._lock:
                self._stopping = True
            self._log("[Generation] switch coordinator could not start; switching unavailable")
            return False
        return True

    def stop(self) -> None:
        with self._lock:
            self._started = True
            self._stopping = True
            self._work.notify_all()

    # --- requests (event loop; in-memory only) ------------------------------------------

    def request_switch(self, target_profile_id: object) -> SwitchRequestOutcome:
        registry = self._registry()
        profile = registry.get(target_profile_id) if isinstance(target_profile_id, str) else None
        if profile is None:
            return SwitchRequestOutcome(False, 400, "Unknown model profile.")
        reason = registry.ineligible_reason(profile)
        if reason is not None:
            return SwitchRequestOutcome(
                False, 409, _INELIGIBLE_DETAILS.get(reason, "That model profile is not qualified for selection.")
            )
        snap = self._ledger.snapshot()
        with self._lock:
            if self._stopping or not self._started or not snap.owner:
                return SwitchRequestOutcome(False, 503, "Generation ownership is not available on this server.")
            if self._pending is not None:
                return SwitchRequestOutcome(False, 409, "A model switch is already in progress.")
            if snap.paused_by is not None:
                return SwitchRequestOutcome(False, 409, "An interrupted model switch must be aborted first.")
            if snap.selection is None:
                return SwitchRequestOutcome(False, 503, "Generation ownership is not available on this server.")
            if snap.selection.profile_id == profile.profile_id:
                return SwitchRequestOutcome(False, 409, "That model profile is already active.")
            if snap.unresolved_answers or snap.unresolved_helpers:
                return SwitchRequestOutcome(
                    False, 409,
                    "Model work whose outcome is unknown is holding capacity; switching is blocked until it is resolved.",
                )
            switch = _Switch(profile.profile_id)
            self._pending = switch
            self._work.notify_all()
        self._log(f"[Generation] switch {switch.switch_id} queued")
        return SwitchRequestOutcome(True, 202, "Model switch accepted.", switch.switch_id)

    def abort_switch(self) -> SwitchRequestOutcome:
        # Never waits: the commit holds the fence lock only for its own
        # compare-and-set, and an abort that cannot take it is refused rather
        # than acknowledged, so no acknowledged abort is ever followed by a
        # commit and the event loop is never blocked here.
        if not self._fence_lock.acquire(blocking=False):
            return SwitchRequestOutcome(False, 409, "The model switch is committing; check its status.")
        try:
            with self._lock:
                switch = self._pending
                if switch is not None:
                    if switch.committed:
                        return SwitchRequestOutcome(False, 409, "The model switch has already been committed.")
                    switch.abort_requested = True
                    self._work.notify_all()
                    return SwitchRequestOutcome(True, 202, "Abort requested.", switch.switch_id)
                snap = self._ledger.snapshot()
                if snap.paused_by is not None and snap.owner:
                    self._interrupted_abort = True
                    self._work.notify_all()
                    return SwitchRequestOutcome(True, 202, "Abort requested.", snap.paused_by)
        finally:
            self._fence_lock.release()
        return SwitchRequestOutcome(False, 409, "No model switch is in progress.")

    def status(self) -> dict:
        snap = self._ledger.snapshot()
        registry = self._registry()
        recovery = self._owner.recovery
        with self._lock:
            switch = self._pending
            current = None
            if switch is not None:
                current = {
                    "switch_id": switch.switch_id,
                    "target_profile_id": switch.target_profile_id,
                    "phase": switch.phase,
                    "abort_requested": switch.abort_requested,
                    "origin": "this_process",
                }
            elif snap.paused_by is not None:
                interrupted = recovery.selection if recovery is not None else None
                current = {
                    "switch_id": snap.paused_by,
                    "target_profile_id": interrupted.target_profile_id if interrupted is not None else None,
                    "phase": "aborting" if self._interrupted_abort else "interrupted",
                    "abort_requested": self._interrupted_abort,
                    "origin": "interrupted",
                }
            last = dict(self._last) if self._last is not None else None
        active = None
        if snap.selection is not None:
            profile = registry.get(snap.selection.profile_id)
            active = {
                "profile_id": snap.selection.profile_id,
                "display_name": profile.display_name if profile is not None else None,
                "model": profile.model_tag if profile is not None else None,
                "epoch": snap.selection.epoch,
                "registered": profile is not None,
                "experimental": bool(profile is not None and profile.experimental),
                "runtime": profile.runtime_key if profile is not None else None,
                # A persisted selection this server would refuse to answer
                # with (disabled experimental profile, unconfigured runtime).
                "eligible": bool(profile is not None and registry.is_selectable(profile)),
            }
        choices = []
        for profile in registry.profiles:
            smoke = profile.runtime_smoke
            choices.append({
                "profile_id": profile.profile_id,
                "display_name": profile.display_name,
                "model": profile.model_tag,
                "eligible": registry.is_selectable(profile),
                "ineligible_reason": registry.ineligible_reason(profile),
                "active": active is not None and profile.profile_id == active["profile_id"],
                # Backend-owned key only; never an endpoint.
                "runtime": profile.runtime_key,
                "evidence": profile.evidence_tier,
                "experimental": profile.experimental,
                "runtime_smoke_verdict": smoke.verdict if smoke is not None else None,
                "review_note": smoke.review_note if smoke is not None else None,
            })
        return {
            "ownership": self._owner.state.value,
            "admission": snap.admission,
            "active": active,
            "switch": current,
            "last_switch": last,
            "choices": choices,
            "capacity": {
                "answers_running": snap.answers_running,
                "helpers_running": snap.helpers_running,
                "unresolved_answers": snap.unresolved_answers,
                "unresolved_helpers": snap.unresolved_helpers,
                "answer_waiters": snap.answer_waiters,
                "total_slots": snap.total_slots,
                "answer_slots": snap.answer_slots,
                "helper_slots": snap.helper_slots,
                "selection_job_limit": snap.selection_job_limit,
            },
            "experimental_enabled": bool(getattr(registry, "experimental_enabled", False)),
            "settings_provisional": True,
        }

    # --- the coordinator thread ------------------------------------------------------------

    def _run(self) -> None:
        while True:
            with self._lock:
                while not self._stopping and not self._interrupted_abort and (
                    self._pending is None or self._pending.outcome is not None
                ):
                    self._work.wait()
                if self._stopping:
                    return
                interrupted = self._interrupted_abort
                switch = self._pending
            try:
                if interrupted:
                    self._abort_interrupted()
                else:
                    self._execute(switch)
            except Exception:
                self._log("[Generation] switch coordinator step failed; admission stays as it was")
                if not interrupted and switch.outcome is None:
                    self._finish(switch, "failed", "coordinator_error")
            finally:
                if interrupted:
                    with self._lock:
                        self._interrupted_abort = False

    def _finish(self, switch: _Switch, outcome: str, reason: Optional[str]) -> None:
        switch.outcome = outcome
        switch.reason = reason
        with self._lock:
            self._last = {
                "switch_id": switch.switch_id,
                "target_profile_id": switch.target_profile_id,
                "outcome": outcome,
                "reason": reason,
            }
            if self._pending is switch:
                self._pending = None
        self._log(f"[Generation] switch {switch.switch_id} {outcome}" + (f": {reason}" if reason else ""))

    def _wait(self, switch: _Switch, seconds: float) -> None:
        with self._lock:
            self._work.wait_for(lambda: switch.abort_requested or self._stopping, timeout=max(0.0, seconds))

    def _verify_target(self, profile) -> Optional[str]:
        runtime_key = profile.runtime_key
        try:
            check_installed_identity(profile, lambda tag: self._installed_digest(tag, runtime_key))
        except ModelIdentityMismatchError:
            return "target_identity_mismatch"
        except ModelSelectionError:
            return "target_identity_unavailable"
        return None

    def _execute(self, switch: _Switch) -> None:
        store = self._owner.store
        registry = self._registry()
        target = registry.get(switch.target_profile_id)
        if store is None or target is None or not registry.is_selectable(target):
            return self._finish(switch, "failed", "ownership_unavailable" if store is None else "target_unavailable")

        switch.phase = "verifying_target"
        refusal = self._verify_target(target)
        if refusal is not None:
            return self._finish(switch, "failed", refusal)
        if switch.abort_requested:
            return self._finish(switch, "aborted", "admin_abort")

        old = self._ledger.pause(switch.switch_id)
        if old is None:
            return self._finish(switch, "failed", "selection_conflict")
        switch.old = old
        expected = SelectionRecord(old.profile_id, None, STATE_ACTIVE, old.epoch, None, None, None, None)
        try:
            switching = store.begin_switch(expected=expected, target_profile_id=target.profile_id,
                                           switch_id=switch.switch_id)
        except Exception:
            switching = self._reread_begin(store, switch, old)
            if switching is _AMBIGUOUS:
                return self._finish(switch, "failed", "begin_ambiguous")
        if switching is None:
            self._ledger.resume(switch.switch_id, old)
            return self._finish(switch, "failed", "selection_conflict")
        switch.switching = switching

        # Drain: every admitted job finishes with its own snapshot.
        switch.phase = "draining"
        deadline = self._clock() + self._settings.drain_timeout_s
        while not self._ledger.wait_drained(self._settings.check_interval_s):
            if self._stopping:
                return self._finish(switch, "failed", "stopped")
            if switch.abort_requested:
                return self._abort(store, switch, "admin_abort")
            if self._ledger.unknown_since_pause():
                return self._abort(store, switch, "uncertain_work")
            if self._clock() >= deadline:
                return self._abort(store, switch, "drain_timeout")
        if switch.abort_requested:
            return self._abort(store, switch, "admin_abort")
        if self._ledger.unknown_since_pause() or self._ledger.unresolved_total():
            return self._abort(store, switch, "uncertain_work")
        try:
            unresolved_rows = store.count_unresolved()
        except Exception:
            unresolved_rows = None
        if unresolved_rows != 0:
            return self._abort(store, switch, "uncertain_work")

        # Residency: the previous model must leave by its own expiry.
        switch.phase = "awaiting_residency"
        previous = registry.get(old.profile_id)
        if previous is None:
            return self._abort(store, switch, "previous_profile_unknown")
        deadline = self._clock() + self._settings.residency_timeout_s
        while True:
            try:
                resident = self._generation_resident(registry, previous, target)
            except Exception:
                return self._abort(store, switch, "residency_unavailable")
            if not resident:
                break
            if switch.abort_requested:
                return self._abort(store, switch, "admin_abort")
            if self._clock() >= deadline:
                return self._abort(store, switch, "residency_timeout")
            self._wait(switch, min(self._settings.residency_poll_s, max(0.0, deadline - self._clock())))
            if self._stopping:
                return self._finish(switch, "failed", "stopped")

        switch.phase = "committing"
        refusal = self._verify_target(target)
        if refusal is not None:
            return self._abort(store, switch, refusal)
        with self._fence_lock:
            if switch.abort_requested:
                committed = None
            else:
                try:
                    committed = store.commit_switch(switching=switching)
                except Exception:
                    committed = self._reread_commit(store, switch)
                if committed is not None and committed is not _AMBIGUOUS:
                    switch.committed = True
        if committed is _AMBIGUOUS:
            return self._finish(switch, "failed", "commit_ambiguous")
        if committed is None:
            return self._abort(store, switch, "admin_abort" if switch.abort_requested else "selection_conflict")
        self._ledger.resume(switch.switch_id, committed.shared)
        return self._finish(switch, "committed", None)

    def _listed(self, runtime_key: str) -> list:
        """One runtime's loaded-model entries as (tags, digest); ValueError
        when the list is unreadable. Every entry must be identifiable before
        any is trusted: one entry that names no model makes the list
        unreadable."""
        models = self._resident_models(runtime_key)
        if not isinstance(models, list):
            raise ValueError("unreadable loaded-model list")
        return [_listed_identity(entry) for entry in models]

    @staticmethod
    def _matches(listed: list, model_tag: str, digest: Optional[str]) -> bool:
        for tags, listed_digest in listed:
            if model_tag in tags:
                return True
            if digest is not None and listed_digest == digest:
                return True
        return False

    def _generation_resident(self, registry, previous, target) -> bool:
        """True while a generation model that is not the target, where the
        target will serve, still holds memory in the shared GPU pool;
        ValueError when that cannot be read.

        The previous profile is looked for on the runtime that served it,
        which must be configured here. Every configured runtime's list is
        also checked for every registered generation model, so a switch
        between runtimes -- either way -- and between two compatibility
        profiles waits for whichever model is still loaded. The target's own
        tag and digest are exempt ONLY on the target's runtime, where it may
        already be resident: a copy of the target's model on any other
        runtime is a conflicting generator and blocks like any other.
        Unregistered models (the embedder, other clients) are never waited
        for."""
        runtimes = set(registry.configured_runtimes)
        if previous.runtime_key not in runtimes:
            raise ValueError("the previous profile's runtime is not configured")
        identities = [(tag, digest) for _profile_id, tag, digest in registry.generation_identities()]
        for runtime_key in sorted(runtimes):
            listed = self._listed(runtime_key)
            if runtime_key == previous.runtime_key and self._matches(
                listed, previous.model_tag, previous.expected_digest
            ):
                return True
            if runtime_key == target.runtime_key:
                # Where the target serves, its own tag/digest never blocks.
                listed = [
                    (tags, digest) for tags, digest in listed
                    if target.model_tag not in tags
                    and not (target.expected_digest is not None and digest == target.expected_digest)
                ]
            for tag, digest in identities:
                if self._matches(listed, tag, digest):
                    return True
        return False

    def _read(self, store) -> Optional[SelectionRecord]:
        try:
            return store.read_selection()
        except Exception:
            return None

    def _reread_begin(self, store, switch: _Switch, old: SharedSelection):
        record = self._read(store)
        if record is None:
            return _AMBIGUOUS  # admission stays paused
        if record.state == STATE_SWITCHING and record.switch_id == switch.switch_id:
            return record
        if record.state == STATE_ACTIVE and record.epoch == old.epoch and record.active_profile_id == old.profile_id:
            return None
        return _AMBIGUOUS

    def _reread_commit(self, store, switch: _Switch):
        record = self._read(store)
        if record is None:
            return _AMBIGUOUS
        if (record.state == STATE_ACTIVE and record.last_switch_id == switch.switch_id
                and record.last_outcome == "committed"):
            return record
        if record.state == STATE_SWITCHING and record.switch_id == switch.switch_id:
            return None  # the commit did not land; abort instead
        return _AMBIGUOUS

    def _abort(self, store, switch: _Switch, reason: str) -> None:
        switch.phase = "aborting"
        try:
            aborted = store.abort_switch(switching=switch.switching, reason=reason)
        except Exception:
            aborted = _AMBIGUOUS
        if aborted is _AMBIGUOUS or aborted is None:
            record = self._read(store)
            if (record is not None and record.state == STATE_ACTIVE
                    and record.last_switch_id == switch.switch_id
                    and record.last_outcome == "aborted:" + reason):
                aborted = record
            else:
                # Not confirmed: admission stays paused (fail closed).
                return self._finish(switch, "failed", "abort_ambiguous")
        self._ledger.resume(switch.switch_id, aborted.shared)
        return self._finish(switch, "aborted", reason)

    def _abort_interrupted(self) -> None:
        store = self._owner.store
        paused_by = self._ledger.paused_by
        record = self._read(store) if store is not None else None
        if (record is None or paused_by is None or record.state != STATE_SWITCHING
                or record.switch_id != paused_by):
            self._log("[Generation] interrupted switch could not be aborted; admission stays paused")
            return
        try:
            aborted = store.abort_switch(switching=record, reason="interrupted_switch")
        except Exception:
            aborted = None
        if aborted is None:
            confirmed = self._read(store)
            if confirmed is None or confirmed.state != STATE_ACTIVE or confirmed.last_switch_id != paused_by:
                self._log("[Generation] interrupted switch abort not confirmed; admission stays paused")
                return
            aborted = confirmed
        self._ledger.resume(paused_by, aborted.shared)
        with self._lock:
            self._last = {"switch_id": paused_by, "target_profile_id": record.target_profile_id,
                          "outcome": "aborted", "reason": "interrupted_switch"}
        self._log(f"[Generation] interrupted switch {paused_by} aborted")


class GenerationRuntime:
    """Ownership plus the switch coordinator, as one lifecycle."""

    def __init__(self, *, ledger: GenerationLedger, owner: GenerationOwner, coordinator: SwitchCoordinator) -> None:
        self.ledger = ledger
        self.owner = owner
        self.coordinator = coordinator

    def start(self) -> None:
        self.owner.start()
        self.coordinator.start()

    def stop(self) -> None:
        self.coordinator.stop()
        self.owner.stop()

    def status(self) -> dict:
        return self.coordinator.status()

    def request_switch(self, target_profile_id: object) -> SwitchRequestOutcome:
        return self.coordinator.request_switch(target_profile_id)

    def abort_switch(self) -> SwitchRequestOutcome:
        return self.coordinator.abort_switch()


__all__ = [
    "CoordinatorSettings",
    "GenerationRuntime",
    "JobKind",
    "SwitchCoordinator",
    "SwitchRequestOutcome",
    "loaded_model_entries",
]
