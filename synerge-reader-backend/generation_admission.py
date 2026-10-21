"""Admission and call accounting for every generation call this process makes.

Two layers live here.

``HelperCall`` / ``HelperCapacity`` (B2a-1) are the original process-local
helper accounting: pure, no I/O, no clock, no logging and no thread creation.
They remain as the default capacity of a stand-alone ``HelperJobQueue``.

``GenerationLedger`` / ``GenerationTicket`` (below) are the shared admission
the application uses for answers, their verifier calls and every helper job.
The ledger never performs I/O itself; the one boundary it calls is the
injected ``DispatchIntents`` sink, always OUTSIDE its lock, which records a
durable intent before any model POST (see generation_selection.py).

The question both layers answer is: may a job start another model call, given
what this process has OBSERVED about the calls it already submitted? A call is
in exactly one of three states:

* ``NEVER_DISPATCHED`` -- no model POST was invoked for it;
* ``OBSERVED_TERMINAL`` -- the reply body carried the model protocol's
  terminal marker (see ``is_terminal_generate_reply``);
* ``UNKNOWN`` -- submission was attempted and termination was not observed.

Capacity comes back only for the first two. An ``UNKNOWN`` call keeps its slot:
a transport exception, a normal end of the HTTP body without the marker, an
error body, ``response.close()``, a timer, a ``finally`` block, a client
disconnect, shutdown or the passage of time are not evidence that the model
stopped working, so none of them releases the slot. ``HelperCapacity`` keeps
that knowledge in memory only. The ledger's tickets also record a durable
intent before each dispatch, so a restarted owner counts every call whose end
was never observed as still holding capacity (no automatic clearing).
"""

from __future__ import annotations

import collections
import contextlib
import enum
import math
import re
import threading
import uuid
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Union


class CallState(enum.Enum):
    NEVER_DISPATCHED = "never_dispatched"
    OBSERVED_TERMINAL = "observed_terminal"
    UNKNOWN = "unknown"


# The completion reasons that end a native /api/generate reply. "length" ends
# the model's work too (the output limit was reached); it frees capacity but
# is never accepted as an answer -- answer acceptance stays with
# ollama_generation.completed_generate_response, which requires "stop".
TERMINAL_DONE_REASONS = ("stop", "length")

MAX_SLOTS = 64


class AdmissionSettingsError(ValueError):
    """A capacity setting is not a usable positive integer."""


def require_positive_int(name: str, value: object, *, maximum: int) -> int:
    """``value`` if it is a real int in [1, maximum]; never a bool, float or str."""
    if type(value) is not int or not 1 <= value <= maximum:
        raise AdmissionSettingsError(f"{name} must be an integer in [1, {maximum}]; got {value!r}")
    return value


def is_terminal_generate_reply(body: object) -> bool:
    """True only for a reply that says the model finished this call.

    The body must be a dict with no ``error`` field, ``done`` must be exactly
    ``True`` (not a truthy look-alike) and ``done_reason`` exactly one of
    ``TERMINAL_DONE_REASONS`` (a plain str, case-sensitive). A thinking field,
    the response text and every other field are ignored here.
    """
    if not isinstance(body, dict) or "error" in body:
        return False
    if body.get("done") is not True:
        return False
    reason = body.get("done_reason")
    return type(reason) is str and reason in TERMINAL_DONE_REASONS


class HelperCall:
    """What one job observed about the model calls it made.

    Handed to exactly one job, which reports ``dispatching()`` immediately
    before it invokes a model POST and ``observed(body)`` with the decoded
    reply body when it has one. Only that job's thread uses it.
    """

    __slots__ = ("_dispatched", "_awaiting_terminal", "_unknown", "_finished")

    def __init__(self) -> None:
        self._dispatched = 0
        self._awaiting_terminal = False
        self._unknown = False
        self._finished = False

    def dispatching(self) -> None:
        """A model POST is about to be invoked for this job."""
        if self._finished:
            raise RuntimeError("this call record is already finished")
        if self._awaiting_terminal:
            # The previous call never showed its terminal marker.
            self._unknown = True
        self._dispatched += 1
        self._awaiting_terminal = True

    def observed(self, body: object) -> bool:
        """Report a decoded reply body; True if it proved the call terminal."""
        if self._finished or not self._awaiting_terminal:
            return False
        if is_terminal_generate_reply(body):
            self._awaiting_terminal = False
            return True
        return False

    @property
    def state(self) -> CallState:
        if self._dispatched == 0:
            return CallState.NEVER_DISPATCHED
        if self._unknown or self._awaiting_terminal:
            return CallState.UNKNOWN
        return CallState.OBSERVED_TERMINAL


@dataclass(frozen=True)
class CapacitySnapshot:
    slots: int
    in_use: int
    unresolved: int

    @property
    def free(self) -> int:
        return self.slots - self.in_use - self.unresolved


class HelperCapacity:
    """``slots`` concurrent background model calls for this process.

    ``try_acquire`` never waits; it issues a fresh ``HelperCall`` ticket.
    ``finish`` accepts only a ticket that THIS capacity issued and that is
    still outstanding, and returns its slot unless the job's call ended
    ``UNKNOWN``, in which case the slot stays unresolved for good. A ticket
    issued by another capacity, a record constructed directly and an already
    finished ticket are refused before this ledger, the issuing ledger or the
    ticket changes. Only outstanding tickets are tracked -- never more than
    ``slots`` -- and a finished ticket is forgotten.
    """

    def __init__(self, slots: int) -> None:
        self._slots = require_positive_int("slots", slots, maximum=MAX_SLOTS)
        self._lock = threading.Lock()
        self._outstanding = set()  # tickets issued here and not yet finished
        self._unresolved = 0

    @property
    def free(self) -> int:
        with self._lock:
            return self._slots - len(self._outstanding) - self._unresolved

    def try_acquire(self) -> Optional[HelperCall]:
        with self._lock:
            if len(self._outstanding) + self._unresolved >= self._slots:
                return None
            call = HelperCall()
            self._outstanding.add(call)
        return call

    def finish(self, call: HelperCall) -> CallState:
        if not isinstance(call, HelperCall):
            raise TypeError("finish() needs the HelperCall this capacity issued")
        with self._lock:
            if call not in self._outstanding:
                if call._finished:
                    raise RuntimeError("this call record is already finished")
                raise RuntimeError("this call record is not outstanding at this capacity")
            state = call.state
            self._outstanding.remove(call)
            call._finished = True
            if state is CallState.UNKNOWN:
                self._unresolved += 1
        return state

    def snapshot(self) -> CapacitySnapshot:
        with self._lock:
            return CapacitySnapshot(
                slots=self._slots, in_use=len(self._outstanding), unresolved=self._unresolved
            )


# ===========================================================================
# Shared generation admission: answers, their verifier calls, every helper
# ===========================================================================
#
# One ledger per process. Only the process that owns generation (see
# generation_selection.GenerationOwner) installs a DispatchIntents sink into
# it; until then, and for ever after the owner fences it, nothing is admitted
# and no model call may be dispatched.
#
# Bounds (AdmissionSettings): at most ``total_slots`` admitted-or-unresolved
# jobs of either kind, at most ``answer_slots`` answers and ``helper_slots``
# helper jobs, at most ``answer_waiters`` answers waiting for up to
# ``answer_wait_s``. A freed slot goes to waiting answers first, in arrival
# order; helpers take a slot only when no answer is waiting. Unresolved work
# counts against every bound. While a profile with a ``SelectionPolicy`` is
# the shared selection (an experimental profile: one combined job), its
# ``max_jobs`` tightens every bound and its budgets replace the shared ones
# for jobs granted under it; it never widens a bound.
#
# This module stays free of clocks and event loops, so nothing time-based in
# it can release capacity: the ledger is handed a clock only to stamp job
# deadlines, and the asynchronous two-second wait lives in answer_jobs.py on
# top of the synchronous waiter API below (enqueue_answer /
# settle_answer_waiter). A waiter's ``wake`` callback is how a grant made on
# any thread reaches its event loop.
#
# Every admitted job captures the shared selection -- profile id and epoch --
# at the moment its slot is granted, under the same lock that grants it, so a
# job can never mix a slot from one selection with a model from another. A
# switch pauses admission (``pause``) and installs the new selection only once
# every admitted job has drained (``resume``).

_PROFILE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
MAX_SELECTION_EPOCH = 2**62
MAX_ANSWER_WAITERS = 1024
CALL_ROLES = ("answer", "verifier", "helper")


class JobKind(enum.Enum):
    ANSWER = "answer"
    HELPER = "helper"


class AdmissionRefusal(enum.Enum):
    # Every slot and waiting place is taken by running work, or the wait for
    # a slot ran out.
    BUSY = "busy"
    # The capacity is held by work whose end was never observed; waiting
    # cannot help until it is resolved.
    UNRESOLVED = "unresolved"
    # A model switch has paused new generation.
    SWITCHING = "switching"
    # This process does not own generation (yet, or any more).
    UNAVAILABLE = "unavailable"


class AdmissionRefused(Exception):
    """Admission was refused for ``reason``. Carries no other detail."""

    def __init__(self, reason: AdmissionRefusal) -> None:
        super().__init__(reason.value)
        self.reason = reason


class DispatchRefused(RuntimeError):
    """A model call was NOT sent. Fixed messages only."""


def _require_real(name: str, value: object, low: float, high: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise AdmissionSettingsError(f"{name} must be a finite number in [{low}, {high}]; got {value!r}")
    return float(value)


def require_selection_epoch(value: object) -> int:
    if type(value) is not int or not 1 <= value <= MAX_SELECTION_EPOCH:
        raise AdmissionSettingsError("a selection epoch must be an integer of at least 1")
    return value


@dataclass(frozen=True)
class AdmissionSettings:
    """Validated admission bounds. Every default here is PROVISIONAL: chosen
    for a single-host first release, not measured on the production GPU host.

    ``answer_wait_s`` is the agreed up-to-two-second wait. The job budgets are
    whole-job wall-clock limits from the grant, covering evidence, every model
    call and persistence; per-read transport timeouts are clamped to what is
    left. ``unstarted_lease_s`` reclaims an admitted answer whose response was
    never started by the server.
    """

    total_slots: int = 2
    answer_slots: int = 2
    helper_slots: int = 1
    answer_waiters: int = 4
    answer_wait_s: float = 2.0
    answer_job_budget_s: float = 420.0
    helper_job_budget_s: float = 420.0
    unstarted_lease_s: float = 15.0

    def __post_init__(self) -> None:
        require_positive_int("total_slots", self.total_slots, maximum=MAX_SLOTS)
        require_positive_int("answer_slots", self.answer_slots, maximum=MAX_SLOTS)
        require_positive_int("helper_slots", self.helper_slots, maximum=MAX_SLOTS)
        require_positive_int("answer_waiters", self.answer_waiters, maximum=MAX_ANSWER_WAITERS)
        if self.answer_slots > self.total_slots or self.helper_slots > self.total_slots:
            raise AdmissionSettingsError("a sublimit cannot exceed total_slots")
        _require_real("answer_wait_s", self.answer_wait_s, 0.01, 30.0)
        _require_real("answer_job_budget_s", self.answer_job_budget_s, 1.0, 3600.0)
        _require_real("helper_job_budget_s", self.helper_job_budget_s, 1.0, 3600.0)
        _require_real("unstarted_lease_s", self.unstarted_lease_s, 0.01, 300.0)

    def limit(self, kind: "JobKind") -> int:
        return self.answer_slots if kind is JobKind.ANSWER else self.helper_slots

    def budget(self, kind: "JobKind") -> float:
        return float(self.answer_job_budget_s if kind is JobKind.ANSWER else self.helper_job_budget_s)


@dataclass(frozen=True)
class SharedSelection:
    """The shared active selection a job is admitted under."""

    profile_id: str
    epoch: int

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not _PROFILE_ID_PATTERN.match(self.profile_id):
            raise AdmissionSettingsError("a selection names a well-formed profile id")
        require_selection_epoch(self.epoch)


@dataclass(frozen=True)
class SelectionPolicy:
    """Admission limits that apply while one profile is the shared selection,
    on top of ``AdmissionSettings`` (they can only tighten its slot limits).

    ``max_jobs`` caps admitted-or-unresolved jobs of either kind together;
    the budgets replace the shared whole-job budgets for jobs granted under
    that selection. None keeps the shared setting. Provisional, unmeasured:
    an experimental profile starts at one combined job at a time."""

    max_jobs: Optional[int] = None
    answer_job_budget_s: Optional[float] = None
    helper_job_budget_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_jobs is not None:
            require_positive_int("max_jobs", self.max_jobs, maximum=MAX_SLOTS)
        for name in ("answer_job_budget_s", "helper_job_budget_s"):
            if getattr(self, name) is not None:
                _require_real(name, getattr(self, name), 1.0, 3600.0)


# What a selection policy resolver that fails, or returns something else,
# is treated as: the most conservative admission, never a wider one.
_FAIL_CLOSED_POLICY = SelectionPolicy(max_jobs=1)


@dataclass(frozen=True)
class DispatchIntent:
    """What is durably recorded immediately before one model POST."""

    call_id: str
    job_id: str
    job_kind: str
    call_role: str
    profile_id: str
    model_tag: str
    selection_epoch: int


class DispatchIntents:
    """The durable boundary a ledger records model calls through.

    ``record_dispatch`` must make the intent durable -- or raise -- BEFORE the
    caller sends the POST; any failure, including an ambiguous one, means the
    call is not sent. ``record_outcome`` is best effort: OBSERVED_TERMINAL
    retires the intent, UNKNOWN marks it as ended without terminal evidence,
    NEVER_DISPATCHED removes an intent whose POST was never sent. A failed
    outcome write leaves the intent in place, which is the conservative
    direction: a later owner counts it as unresolved."""

    def record_dispatch(self, intent: DispatchIntent) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def record_outcome(self, intent: DispatchIntent, state: CallState) -> None:  # pragma: no cover
        raise NotImplementedError


_ADMITTED = "admitted"
_STARTED = "started"
_RELEASED = "released"


def is_terminal_generate_stream(stream: object) -> bool:
    """True only for a streamed reply whose parser saw the protocol's final
    record (``done`` exactly True, ``done_reason`` "stop" or "length") with no
    error line and no malformed line anywhere. Duck-typed over
    ollama_generation.OllamaGenerateStream so this module stays pure."""
    reason = getattr(stream, "done_reason", None)
    return (
        getattr(stream, "done", None) is True
        and type(reason) is str
        and reason in TERMINAL_DONE_REASONS
        and getattr(stream, "model_error", True) is False
        and getattr(stream, "malformed_lines", 1) == 0
    )


class GenerationTicket:
    """One admitted generation job: an answer with its verifier calls, or one
    helper job. Issued only by a GenerationLedger.

    The job's own thread reports ``dispatching(role)`` immediately before each
    model POST -- which records the durable intent first and refuses (raising
    DispatchRefused, POST not sent) when that is not possible -- and
    ``observed(body)`` / ``observed_stream(stream)`` with what came back. Calls
    are strictly sequential: once a call was dispatched and its terminal
    marker not observed, the job may dispatch nothing more, so one slot never
    covers two possibly-running calls.
    """

    __slots__ = (
        "kind", "job_id", "selection", "deadline", "snapshot", "_ledger", "_owner_seq",
        "_phase", "_dispatched", "_awaiting", "_unknown", "_released_state",
    )

    def __init__(self, ledger: "GenerationLedger", kind: JobKind, selection: SharedSelection,
                 owner_seq: int, deadline: float, phase: str) -> None:
        self.kind = kind
        self.job_id = str(uuid.uuid4())
        self.selection = selection
        self.deadline = deadline
        # The frozen AnswerGenerationSnapshot built from ``selection`` for this
        # job (bind_snapshot). Every call this ticket dispatches uses its tag.
        self.snapshot = None
        self._ledger = ledger
        self._owner_seq = owner_seq
        self._phase = phase
        self._dispatched = 0
        self._awaiting: Optional[DispatchIntent] = None
        self._unknown = False
        self._released_state: Optional[CallState] = None

    def bind_snapshot(self, snapshot) -> None:
        """Fix this job's snapshot once; its profile and epoch must be the
        captured selection's."""
        if self.snapshot is not None:
            raise RuntimeError("this ticket's snapshot is already fixed")
        if (
            getattr(snapshot, "profile_id", None) != self.selection.profile_id
            or getattr(snapshot, "selection_epoch", None) != self.selection.epoch
        ):
            raise RuntimeError("a snapshot must come from the ticket's own selection")
        self.snapshot = snapshot

    # --- time ---------------------------------------------------------------------

    @property
    def remaining_s(self) -> float:
        return self.deadline - self._ledger._clock()

    @property
    def expired(self) -> bool:
        return self.remaining_s <= 0

    def read_timeout(self, profile_timeout: float) -> float:
        """The per-read transport timeout for the next read: the profile's own
        value, clamped to this job's remaining budget (never below 0.5 s)."""
        return max(0.5, min(float(profile_timeout), self.remaining_s))

    # --- calls --------------------------------------------------------------------

    @property
    def may_dispatch(self) -> bool:
        return (
            not self._unknown
            and self._awaiting is None
            and not self.expired
            and self._ledger._may_dispatch(self)
        )

    def dispatching(self, role: str = "helper") -> None:
        """Record a durable intent, then permit exactly one model POST."""
        if role not in CALL_ROLES:
            raise ValueError("unknown call role")
        if self._unknown or self._awaiting is not None:
            raise DispatchRefused("an earlier model call of this job was never observed to finish")
        if self.expired:
            raise DispatchRefused("the job's time budget is spent")
        intent = self._ledger._dispatch_intent(self, role)
        intents = self._ledger._intents
        try:
            intents.record_dispatch(intent)
        except Exception:
            intent_failed = True
        else:
            intent_failed = False
        if intent_failed:
            # The POST is never sent, so removing a possibly-written intent is
            # truthful; if that also fails, the intent stays (conservative).
            try:
                intents.record_outcome(intent, CallState.NEVER_DISPATCHED)
            except Exception:
                pass
            raise DispatchRefused("the model call could not be recorded before dispatch")
        self._dispatched += 1
        self._awaiting = intent

    def _settle(self, terminal: bool) -> bool:
        intent = self._awaiting
        if intent is None:
            return False
        if not terminal:
            # A complete reply without the terminal marker: the call's end was
            # not observed. It stays awaiting (UNKNOWN); nothing more may run.
            self._unknown = True
            return False
        self._awaiting = None
        try:
            self._ledger._intents.record_outcome(intent, CallState.OBSERVED_TERMINAL)
        except Exception:
            pass
        return True

    def observed(self, body: object) -> bool:
        """A decoded non-streamed reply body; True if it proved the call ended."""
        return self._settle(is_terminal_generate_reply(body))

    def observed_stream(self, stream: object) -> bool:
        """A fully read streamed reply; True if it proved the call ended."""
        return self._settle(is_terminal_generate_stream(stream))

    @property
    def state(self) -> CallState:
        if self._dispatched == 0:
            return CallState.NEVER_DISPATCHED
        if self._unknown or self._awaiting is not None:
            return CallState.UNKNOWN
        return CallState.OBSERVED_TERMINAL

    @property
    def released(self) -> bool:
        return self._phase == _RELEASED


class AnswerWaiter:
    """One answer waiting for a slot. ``wake`` is called -- from whichever
    thread decided -- once the waiter was granted or refused; the waiting side
    then collects the outcome with ``GenerationLedger.settle_answer_waiter``."""

    __slots__ = ("wake", "status", "ticket", "refusal")

    def __init__(self, wake: Callable[[], None]) -> None:
        self.wake = wake
        self.status = "waiting"  # waiting | granted | collected | refused | abandoned
        self.ticket: Optional[GenerationTicket] = None
        self.refusal: Optional[AdmissionRefusal] = None


@dataclass(frozen=True)
class LedgerSnapshot:
    """Counts only: no ids, no content."""

    admission: str  # open | paused | unavailable
    owner: bool
    selection: Optional[SharedSelection]
    paused_by: Optional[str]
    answers_running: int
    helpers_running: int
    unresolved_answers: int
    unresolved_helpers: int
    answer_waiters: int
    total_slots: int
    answer_slots: int
    helper_slots: int
    # The combined job limit of the current selection's policy (None: the
    # shared settings apply unchanged).
    selection_job_limit: Optional[int] = None


class GenerationLedger:
    """The process's one admission ledger for every generation job.

    ``selection_policy`` (optional, pure, no I/O) maps the shared selection
    to the ``SelectionPolicy`` that applies while it is active. It is
    resolved when a selection is installed or resumed -- never per request --
    and a resolver that fails is treated as one job at a time."""

    def __init__(
        self,
        settings: AdmissionSettings,
        *,
        clock: Callable[[], float],
        selection_policy: Optional[Callable[[SharedSelection], Optional[SelectionPolicy]]] = None,
    ) -> None:
        if not isinstance(settings, AdmissionSettings):
            raise TypeError("settings must be AdmissionSettings")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if selection_policy is not None and not callable(selection_policy):
            raise TypeError("selection_policy must be callable")
        self._settings = settings
        self._clock = clock
        self._selection_policy = selection_policy
        self._policy: Optional[SelectionPolicy] = None
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._intents: Optional[DispatchIntents] = None
        self._owner_seq: Optional[int] = None
        self._fenced = False
        self._selection: Optional[SharedSelection] = None
        self._paused_by: Optional[str] = None
        self._outstanding = {JobKind.ANSWER: set(), JobKind.HELPER: set()}
        self._unresolved = {JobKind.ANSWER: 0, JobKind.HELPER: 0}
        self._unknown_releases = 0
        self._waiters = collections.deque()
        self._listeners = []

    @property
    def settings(self) -> AdmissionSettings:
        return self._settings

    def _resolve_policy(self, selection: Optional[SharedSelection]) -> Optional[SelectionPolicy]:
        """Outside the lock: the policy for ``selection``, or None."""
        if selection is None or self._selection_policy is None:
            return None
        try:
            policy = self._selection_policy(selection)
        except Exception:
            return _FAIL_CLOSED_POLICY
        if policy is None or isinstance(policy, SelectionPolicy):
            return policy
        return _FAIL_CLOSED_POLICY

    # --- ownership (the owner manager) ---------------------------------------------

    def install_owner(
        self,
        *,
        intents: DispatchIntents,
        owner_seq: int,
        selection: Optional[SharedSelection],
        paused_by: Optional[str],
        recovered: Mapping[JobKind, int],
    ) -> bool:
        """Open admission for the owner, once per process lifetime.

        ``recovered`` is the unresolved work a previous owner left durable: it
        holds capacity exactly like this process's own UNKNOWN calls. A
        ``paused_by`` switch (left switching by a previous boot) keeps
        admission paused until an administrator aborts it."""
        if not isinstance(intents, DispatchIntents) or type(owner_seq) is not int or owner_seq < 1:
            raise TypeError("install_owner needs a DispatchIntents sink and a positive owner sequence")
        if selection is not None and not isinstance(selection, SharedSelection):
            raise TypeError("selection must be a SharedSelection or None")
        counts = {kind: int(recovered.get(kind, 0)) for kind in JobKind}
        if any(count < 0 for count in counts.values()):
            raise ValueError("recovered counts are non-negative")
        policy = self._resolve_policy(selection)
        with self._lock:
            if self._fenced or self._intents is not None:
                return False
            self._intents = intents
            self._owner_seq = owner_seq
            self._selection = selection
            self._policy = policy
            self._paused_by = paused_by
            for kind, count in counts.items():
                self._unresolved[kind] += count
            grants = self._grant_waiters_locked()
            self._changed.notify_all()
        self._deliver(grants, ())
        self._notify_listeners()
        return True

    def fence(self) -> None:
        """Permanently stop admission and dispatch in this process (ownership
        lost or shut down). Waiting answers are refused; running jobs may
        finish reading what they already sent but cannot dispatch again."""
        with self._lock:
            self._fenced = True
            refused = self._refuse_waiters_locked(AdmissionRefusal.UNAVAILABLE)
            self._changed.notify_all()
        self._deliver((), refused)
        self._notify_listeners()

    @property
    def fenced(self) -> bool:
        with self._lock:
            return self._fenced

    def _ready_locked(self) -> bool:
        return not self._fenced and self._intents is not None

    def _may_dispatch(self, ticket: GenerationTicket) -> bool:
        with self._lock:
            return (
                self._ready_locked()
                and ticket._owner_seq == self._owner_seq
                and ticket._phase == _STARTED
                and ticket in self._outstanding[ticket.kind]
            )

    def _dispatch_intent(self, ticket: GenerationTicket, role: str) -> DispatchIntent:
        snapshot = ticket.snapshot
        if snapshot is None:
            raise DispatchRefused("the job's model snapshot is not fixed")
        if not self._may_dispatch(ticket):
            raise DispatchRefused("this process may not dispatch for this job")
        return DispatchIntent(
            call_id=str(uuid.uuid4()),
            job_id=ticket.job_id,
            job_kind=ticket.kind.value,
            call_role=role,
            profile_id=ticket.selection.profile_id,
            model_tag=snapshot.model_tag,
            selection_epoch=ticket.selection.epoch,
        )

    # --- the switch coordinator -------------------------------------------------------

    def pause(self, switch_id: str) -> Optional[SharedSelection]:
        """Pause new admission for ``switch_id``; the selection being left, or
        None (and nothing changes) when this process cannot pause now."""
        with self._lock:
            if not self._ready_locked() or self._paused_by is not None or self._selection is None:
                return None
            self._paused_by = switch_id
            self._unknown_releases = 0
            refused = self._refuse_waiters_locked(AdmissionRefusal.SWITCHING)
            current = self._selection
        self._deliver((), refused)
        return current

    def resume(self, switch_id: str, selection: SharedSelection) -> bool:
        """End ``switch_id``'s pause with ``selection`` (the committed target,
        or the unchanged profile at the aborted epoch). Refused when another
        switch, or none, holds the pause."""
        if not isinstance(selection, SharedSelection):
            raise TypeError("selection must be a SharedSelection")
        policy = self._resolve_policy(selection)
        with self._lock:
            if self._fenced or self._paused_by is None or self._paused_by != switch_id:
                return False
            self._selection = selection
            self._policy = policy
            self._paused_by = None
            grants = self._grant_waiters_locked()
            self._changed.notify_all()
        self._deliver(grants, ())
        self._notify_listeners()
        return True

    @property
    def paused_by(self) -> Optional[str]:
        with self._lock:
            return self._paused_by

    def unknown_since_pause(self) -> int:
        """Jobs released UNKNOWN since the current pause began."""
        with self._lock:
            return self._unknown_releases

    def wait_drained(self, timeout: float) -> bool:
        """True once no job of either kind is admitted or running (within
        ``timeout`` seconds). Unresolved work is not running work; the
        coordinator checks it separately."""
        deadline = self._clock() + max(0.0, timeout)
        with self._changed:
            while any(self._outstanding[kind] for kind in JobKind):
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return False
                self._changed.wait(min(remaining, 0.25))
            return True

    def unresolved_total(self) -> int:
        with self._lock:
            return sum(self._unresolved.values())

    def snapshot(self) -> LedgerSnapshot:
        with self._lock:
            if not self._ready_locked():
                admission = "unavailable"
            elif self._paused_by is not None:
                admission = "paused"
            else:
                admission = "open"
            return LedgerSnapshot(
                admission=admission,
                owner=self._ready_locked(),
                selection=self._selection,
                paused_by=self._paused_by,
                answers_running=len(self._outstanding[JobKind.ANSWER]),
                helpers_running=len(self._outstanding[JobKind.HELPER]),
                unresolved_answers=self._unresolved[JobKind.ANSWER],
                unresolved_helpers=self._unresolved[JobKind.HELPER],
                answer_waiters=len(self._waiters),
                total_slots=self._settings.total_slots,
                answer_slots=self._settings.answer_slots,
                helper_slots=self._settings.helper_slots,
                selection_job_limit=self._policy.max_jobs if self._policy is not None else None,
            )

    def add_listener(self, listener: Callable[[], None]) -> None:
        """Called (outside every ledger lock) whenever capacity or the
        selection may have changed, e.g. to wake helper queue workers."""
        with self._lock:
            self._listeners.append(listener)

    # --- admission -----------------------------------------------------------------

    def _refusal_locked(self) -> Optional[AdmissionRefusal]:
        if not self._ready_locked() or self._selection is None:
            return AdmissionRefusal.UNAVAILABLE
        if self._paused_by is not None:
            return AdmissionRefusal.SWITCHING
        return None

    def _total_limit_locked(self) -> int:
        cap = self._policy.max_jobs if self._policy is not None else None
        return self._settings.total_slots if cap is None else min(self._settings.total_slots, cap)

    def _kind_limit_locked(self, kind: JobKind) -> int:
        cap = self._policy.max_jobs if self._policy is not None else None
        limit = self._settings.limit(kind)
        return limit if cap is None else min(limit, cap)

    def _budget_locked(self, kind: JobKind) -> float:
        policy = self._policy
        if policy is not None:
            override = policy.answer_job_budget_s if kind is JobKind.ANSWER else policy.helper_job_budget_s
            if override is not None:
                return float(override)
        return self._settings.budget(kind)

    def _has_capacity_locked(self, kind: JobKind) -> bool:
        used = sum(len(jobs) for jobs in self._outstanding.values()) + sum(self._unresolved.values())
        used_kind = len(self._outstanding[kind]) + self._unresolved[kind]
        return used < self._total_limit_locked() and used_kind < self._kind_limit_locked(kind)

    def _exhausted_by_unresolved_locked(self, kind: JobKind) -> bool:
        return (
            sum(self._unresolved.values()) >= self._total_limit_locked()
            or self._unresolved[kind] >= self._kind_limit_locked(kind)
        )

    def _new_ticket_locked(self, kind: JobKind, phase: str) -> GenerationTicket:
        ticket = GenerationTicket(
            self, kind, self._selection, self._owner_seq,
            self._clock() + self._budget_locked(kind), phase,
        )
        self._outstanding[kind].add(ticket)
        return ticket

    def _grant_waiters_locked(self):
        grants = []
        while self._waiters and self._refusal_locked() is None and self._has_capacity_locked(JobKind.ANSWER):
            waiter = self._waiters.popleft()
            waiter.ticket = self._new_ticket_locked(JobKind.ANSWER, _ADMITTED)
            waiter.status = "granted"
            grants.append(waiter)
        return grants

    def _refuse_waiters_locked(self, refusal: AdmissionRefusal):
        refused = list(self._waiters)
        self._waiters.clear()
        for waiter in refused:
            waiter.status = "refused"
            waiter.refusal = refusal
        return refused

    def _deliver(self, grants, refused) -> None:
        for waiter in (*grants, *refused):
            try:
                waiter.wake()
            except Exception:
                # The waiting side can no longer be reached (its event loop is
                # closed): it will never collect the grant, so a granted slot
                # is returned here. A later settle finds nothing to collect.
                ticket = self._abandon_granted(waiter)
                if ticket is not None:
                    self.release_unstarted(ticket)

    def _abandon_granted(self, waiter: AnswerWaiter) -> Optional[GenerationTicket]:
        with self._lock:
            if waiter.status != "granted":
                return None
            waiter.status = "abandoned"
            return waiter.ticket

    def enqueue_answer(self, wake: Callable[[], None]) -> Union[GenerationTicket, AnswerWaiter]:
        """An ADMITTED answer ticket at once, a registered AnswerWaiter, or
        AdmissionRefused. A waiter is refused at once (never queued) when the
        answer capacity is held by unresolved work or every waiting place is
        taken."""
        if not callable(wake):
            raise TypeError("wake must be callable")
        with self._lock:
            refusal = self._refusal_locked()
            if refusal is not None:
                raise AdmissionRefused(refusal)
            if not self._waiters and self._has_capacity_locked(JobKind.ANSWER):
                return self._new_ticket_locked(JobKind.ANSWER, _ADMITTED)
            if self._exhausted_by_unresolved_locked(JobKind.ANSWER):
                raise AdmissionRefused(AdmissionRefusal.UNRESOLVED)
            if len(self._waiters) >= self._settings.answer_waiters:
                raise AdmissionRefused(AdmissionRefusal.BUSY)
            waiter = AnswerWaiter(wake)
            self._waiters.append(waiter)
            return waiter

    def settle_answer_waiter(self, waiter: AnswerWaiter) -> Optional[GenerationTicket]:
        """Decide a waiter's outcome once, under the lock: its granted ticket
        (the grant wins even if it landed at the deadline), or None -- in which
        case a still-waiting entry is withdrawn and can no longer be granted.
        The caller owns a returned ticket, and must release it if it cannot
        use it."""
        with self._lock:
            if waiter.status == "granted":
                waiter.status = "collected"
                return waiter.ticket
            if waiter.status == "waiting":
                self._waiters.remove(waiter)
                waiter.status = "abandoned"
        return None

    def try_acquire_helper(self) -> Optional[GenerationTicket]:
        """A STARTED helper ticket, or None. Never waits; never takes a slot
        while an answer is waiting for one."""
        with self._lock:
            if self._refusal_locked() is not None or self._waiters:
                return None
            if not self._has_capacity_locked(JobKind.HELPER):
                return None
            return self._new_ticket_locked(JobKind.HELPER, _STARTED)

    def helper_free(self) -> int:
        with self._lock:
            if self._refusal_locked() is not None:
                return 0
            used = sum(len(jobs) for jobs in self._outstanding.values()) + sum(self._unresolved.values())
            used_kind = len(self._outstanding[JobKind.HELPER]) + self._unresolved[JobKind.HELPER]
            return max(0, min(self._total_limit_locked() - used, self._kind_limit_locked(JobKind.HELPER) - used_kind))

    # --- answer handoff and release ----------------------------------------------------

    def start(self, ticket: GenerationTicket) -> bool:
        """ADMITTED -> STARTED for an answer whose response began. False if
        the ticket was already released (lease expiry, unstarted release)."""
        with self._lock:
            if ticket._ledger is not self or ticket._phase != _ADMITTED or ticket not in self._outstanding[ticket.kind]:
                return False
            ticket._phase = _STARTED
            return True

    def release_unstarted(self, ticket: GenerationTicket) -> bool:
        """Return an ADMITTED ticket's slot (nothing was dispatched). A no-op
        returning False for any ticket that already started or was released,
        so every path may call it."""
        with self._lock:
            if ticket._ledger is not self or ticket._phase != _ADMITTED or ticket not in self._outstanding[ticket.kind]:
                return False
            self._outstanding[ticket.kind].remove(ticket)
            ticket._phase = _RELEASED
            ticket._released_state = CallState.NEVER_DISPATCHED
            grants = self._grant_waiters_locked()
            self._changed.notify_all()
        self._deliver(grants, ())
        self._notify_listeners()
        return True

    def release(self, ticket: GenerationTicket) -> CallState:
        """Finish a STARTED ticket, exactly once, by what its calls showed.
        UNKNOWN keeps the slot unresolved (durably marked, best effort)."""
        if not isinstance(ticket, GenerationTicket):
            raise TypeError("release() needs a GenerationTicket this ledger issued")
        with self._lock:
            if ticket._ledger is not self or ticket not in self._outstanding[ticket.kind]:
                raise RuntimeError("this ticket is not outstanding at this ledger")
            if ticket._phase != _STARTED:
                raise RuntimeError("only a started ticket is released by its job")
            state = ticket.state
            self._outstanding[ticket.kind].remove(ticket)
            ticket._phase = _RELEASED
            ticket._released_state = state
            if state is CallState.UNKNOWN:
                self._unresolved[ticket.kind] += 1
                self._unknown_releases += 1
            grants = self._grant_waiters_locked()
            intents = self._intents
            self._changed.notify_all()
        self._deliver(grants, ())
        if state is CallState.UNKNOWN and ticket._awaiting is not None and intents is not None:
            try:
                intents.record_outcome(ticket._awaiting, CallState.UNKNOWN)
            except Exception:
                pass
        self._notify_listeners()
        return state

    def _notify_listeners(self) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener()
            except Exception:
                pass


class HelperPool:
    """A ledger's helper sublimit in the capacity shape HelperJobQueue uses:
    ``try_acquire`` never waits and issues a STARTED helper ticket that
    captured the shared selection at that moment; ``finish`` releases it."""

    def __init__(self, ledger: GenerationLedger) -> None:
        self._ledger = ledger

    @property
    def free(self) -> int:
        return self._ledger.helper_free()

    def try_acquire(self) -> Optional[GenerationTicket]:
        return self._ledger.try_acquire_helper()

    def finish(self, ticket: GenerationTicket) -> CallState:
        return self._ledger.release(ticket)

    def snapshot(self) -> CapacitySnapshot:
        snap = self._ledger.snapshot()
        return CapacitySnapshot(
            slots=snap.helper_slots, in_use=snap.helpers_running, unresolved=snap.unresolved_helpers,
        )

    def add_listener(self, listener: Callable[[], None]) -> None:
        self._ledger.add_listener(listener)


# --- the dispatching job of the current thread ----------------------------------------

_SCOPE = threading.local()


@contextlib.contextmanager
def generation_scope(ticket: GenerationTicket, *, stopped: Callable[[], bool] = lambda: False):
    """Mark ``ticket`` as the job this thread dispatches model calls for.
    ``stopped`` reports a job-level stop (e.g. known delivery abandonment)
    after which no new call may be dispatched."""
    previous = getattr(_SCOPE, "value", None)
    _SCOPE.value = (ticket, stopped)
    try:
        yield ticket
    finally:
        _SCOPE.value = previous


def current_generation_ticket() -> Optional[GenerationTicket]:
    value = getattr(_SCOPE, "value", None)
    return value[0] if value else None


def current_dispatch_permitted() -> bool:
    """Whether this thread's job may dispatch another model call now."""
    value = getattr(_SCOPE, "value", None)
    if not value:
        return False
    ticket, stopped = value
    try:
        halted = bool(stopped())
    except Exception:
        halted = True
    return not halted and ticket.may_dispatch
