"""Process-local accounting for background helper model work (B2a-1).

Pure: no I/O, no clock, no logging and no thread creation. A small
``threading.Lock`` guards only in-memory counters and is never held while a
caller does anything else.

The question this module answers is narrow: may a background helper start
another model call, given what this process has OBSERVED about the calls it
already submitted? A call is in exactly one of three states:

* ``NEVER_DISPATCHED`` -- no model POST was invoked for it;
* ``OBSERVED_TERMINAL`` -- the reply body carried the model protocol's
  terminal marker (see ``is_terminal_generate_reply``);
* ``UNKNOWN`` -- submission was attempted and termination was not observed.

Capacity comes back only for the first two. An ``UNKNOWN`` call keeps its slot
for the rest of this process's life: a transport exception, a normal end of the
HTTP body without the marker, an error body, ``response.close()``, a timer, a
``finally`` block, shutdown or the passage of time are not evidence that the
model stopped working, so none of them releases the slot. Nothing here is
persisted; a process restart loses this knowledge, which is an open
release/ownership gate (B2b), not a recovery mechanism.

This is NOT answer admission: /ask is not routed through it.
"""

from __future__ import annotations

import enum
import threading
from dataclasses import dataclass
from typing import Optional


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
