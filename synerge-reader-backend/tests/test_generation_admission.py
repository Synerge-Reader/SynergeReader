"""Unit contracts for generation_admission.py (background helper accounting).

Pure: no main import, database, model, network or subprocess. Threads appear
only in the concurrency test, running in-process code.

What these prove: only a dict without an ``error`` field whose ``done`` is
exactly True and ``done_reason`` exactly "stop" or "length" counts as the
terminal marker; a call is NEVER_DISPATCHED until a POST is reported, and
UNKNOWN unless every dispatched call showed that marker; capacity comes back
for NEVER_DISPATCHED and OBSERVED_TERMINAL calls only, while an UNKNOWN slot
stays unresolved with no clock, timer or retry that could release it; slots are
bounded under concurrent acquisition; and settings must be real positive ints.

``finish`` honours only an outstanding ticket that the SAME capacity issued: a
real ticket from another capacity, a record nobody issued and an already
finished ticket are refused before either ledger or the ticket changes, so no
invalid finish can drive ``in_use`` negative, push ``free`` above the slots or
let acquisitions exceed the slots -- and the real issuer still finishes its
ticket normally.
"""

import ast
import os
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation_admission import (
    AdmissionSettingsError,
    CallState,
    CapacitySnapshot,
    HelperCall,
    HelperCapacity,
    is_terminal_generate_reply,
    require_positive_int,
)

_BACKEND = Path(__file__).resolve().parents[1]


class _StrSubclass(str):
    pass


_TERMINAL = {
    "stop": {"response": "{}", "done": True, "done_reason": "stop"},
    "length": {"response": "{", "done": True, "done_reason": "length"},
    "stop-with-thinking": {"thinking": "hmm", "response": "", "done": True, "done_reason": "stop"},
    "stop-blank-response": {"response": "   ", "done": True, "done_reason": "stop"},
    "stop-non-string-response": {"response": None, "done": True, "done_reason": "stop"},
    "stop-extra-fields": {"response": "x", "done": True, "done_reason": "stop", "eval_count": 3},
}

_NOT_TERMINAL = {
    "done-missing": {"response": "x", "done_reason": "stop"},
    "done-false": {"response": "x", "done": False, "done_reason": "stop"},
    "done-string": {"response": "x", "done": "true", "done_reason": "stop"},
    "done-one": {"response": "x", "done": 1, "done_reason": "stop"},
    "done-float": {"response": "x", "done": 1.0, "done_reason": "stop"},
    "reason-missing": {"response": "x", "done": True},
    "reason-none": {"response": "x", "done": True, "done_reason": None},
    "reason-load": {"response": "x", "done": True, "done_reason": "load"},
    "reason-upper": {"response": "x", "done": True, "done_reason": "STOP"},
    "reason-padded": {"response": "x", "done": True, "done_reason": "stop "},
    "reason-empty": {"response": "x", "done": True, "done_reason": ""},
    "reason-number": {"response": "x", "done": True, "done_reason": 7},
    "reason-str-subclass": {"response": "x", "done": True, "done_reason": _StrSubclass("stop")},
    "error-with-stop": {"error": "runner crashed", "response": "x", "done": True, "done_reason": "stop"},
    "error-null-with-stop": {"error": None, "done": True, "done_reason": "stop"},
    "list": ["done", True],
    "string": '{"done": true, "done_reason": "stop"}',
    "bytes": b'{"done": true, "done_reason": "stop"}',
    "none": None,
}


@pytest.mark.parametrize("case", sorted(_TERMINAL))
def test_the_terminal_marker_is_recognized(case):
    assert is_terminal_generate_reply(_TERMINAL[case]) is True


@pytest.mark.parametrize("case", sorted(_NOT_TERMINAL))
def test_anything_else_is_not_terminal(case):
    assert is_terminal_generate_reply(_NOT_TERMINAL[case]) is False


# --- one job's call record ----------------------------------------------------------


def test_a_job_that_never_dispatched_is_never_dispatched():
    call = HelperCall()
    assert call.observed(_TERMINAL["stop"]) is False, "an observation without a dispatch proves nothing"
    assert call.state is CallState.NEVER_DISPATCHED


def test_a_dispatched_call_is_unknown_until_its_terminal_marker_is_observed():
    call = HelperCall()
    call.dispatching()
    assert call.state is CallState.UNKNOWN
    assert call.observed(_NOT_TERMINAL["reason-load"]) is False
    assert call.state is CallState.UNKNOWN
    assert call.observed(_TERMINAL["length"]) is True
    assert call.state is CallState.OBSERVED_TERMINAL


def test_every_dispatched_call_must_be_observed():
    both = HelperCall()
    both.dispatching()
    both.observed(_TERMINAL["stop"])
    both.dispatching()
    both.observed(_TERMINAL["stop"])
    assert both.state is CallState.OBSERVED_TERMINAL

    first_unobserved = HelperCall()
    first_unobserved.dispatching()
    first_unobserved.dispatching()  # the first call never showed its marker
    first_unobserved.observed(_TERMINAL["stop"])
    assert first_unobserved.state is CallState.UNKNOWN, "a later terminal reply never covers an earlier call"


# --- capacity -----------------------------------------------------------------------


def test_capacity_is_bounded_and_returned_only_for_never_dispatched_or_terminal_calls():
    capacity = HelperCapacity(2)
    first, second = capacity.try_acquire(), capacity.try_acquire()
    assert first is not None and second is not None
    assert capacity.try_acquire() is None, "never more than the configured slots"
    assert capacity.snapshot() == CapacitySnapshot(slots=2, in_use=2, unresolved=0)

    assert capacity.finish(first) is CallState.NEVER_DISPATCHED
    assert capacity.free == 1
    third = capacity.try_acquire()
    third.dispatching()
    third.observed(_TERMINAL["stop"])
    assert capacity.finish(third) is CallState.OBSERVED_TERMINAL
    assert capacity.free == 1

    second.dispatching()  # submitted; the reply never showed the marker
    assert capacity.finish(second) is CallState.UNKNOWN
    assert capacity.snapshot() == CapacitySnapshot(slots=2, in_use=0, unresolved=1)
    assert capacity.free == 1


def test_an_unknown_slot_is_never_released():
    capacity = HelperCapacity(1)
    call = capacity.try_acquire()
    call.dispatching()
    assert capacity.finish(call) is CallState.UNKNOWN
    for _ in range(1000):
        assert capacity.try_acquire() is None
    assert capacity.snapshot().free == 0
    with pytest.raises(RuntimeError):
        capacity.finish(call)  # a second finish cannot launder the slot back
    with pytest.raises(RuntimeError):
        call.dispatching()
    assert call.observed(_TERMINAL["stop"]) is False, "a late observation cannot release a finished record"
    assert capacity.snapshot() == CapacitySnapshot(slots=1, in_use=0, unresolved=1)


def test_finish_accepts_only_its_own_call_records():
    with pytest.raises(TypeError):
        HelperCapacity(1).finish(object())

    # A real ticket, issued by ANOTHER capacity: refused, and the receiving
    # ledger does not gain a slot it never lent.
    issuer, other = HelperCapacity(1), HelperCapacity(1)
    ticket = issuer.try_acquire()
    with pytest.raises(RuntimeError):
        other.finish(ticket)
    assert other.snapshot() == CapacitySnapshot(slots=1, in_use=0, unresolved=0)
    granted = [other.try_acquire() for _ in range(3)]
    assert sum(call is not None for call in granted) == 1, "a limit of one still admits exactly one"
    assert issuer.try_acquire() is None, "the issuer still holds its slot"

    # A record nobody issued: refused, never a negative in_use or spare slot.
    empty = HelperCapacity(1)
    with pytest.raises(RuntimeError):
        empty.finish(HelperCall())
    assert empty.snapshot() == CapacitySnapshot(slots=1, in_use=0, unresolved=0)
    assert empty.free == 1

    assert issuer.finish(ticket) is CallState.NEVER_DISPATCHED, "the real issuer still finishes it"
    assert issuer.snapshot() == CapacitySnapshot(slots=1, in_use=0, unresolved=0)


def _drive(call, state):
    """Bring a fresh call record into ``state`` through its public reports."""
    if state is not CallState.NEVER_DISPATCHED:
        call.dispatching()
    if state is CallState.OBSERVED_TERMINAL:
        assert call.observed(_TERMINAL["stop"]) is True
    assert call.state is state


@pytest.mark.parametrize("state", list(CallState), ids=lambda state: state.value)
def test_a_foreign_ticket_changes_neither_ledger_nor_the_ticket(state):
    issuer, other = HelperCapacity(1), HelperCapacity(1)
    ticket = issuer.try_acquire()
    _drive(ticket, state)
    issuer_before, other_before = issuer.snapshot(), other.snapshot()

    with pytest.raises(RuntimeError):
        other.finish(ticket)

    assert issuer.snapshot() == issuer_before == CapacitySnapshot(slots=1, in_use=1, unresolved=0)
    assert other.snapshot() == other_before == CapacitySnapshot(slots=1, in_use=0, unresolved=0)
    assert ticket.state is state and ticket._finished is False, "the ticket itself is untouched"
    held = other.try_acquire()
    assert held is not None and other.try_acquire() is None, "the other ledger still lends exactly one slot"
    assert issuer.try_acquire() is None, "the issuer's slot is still held by its ticket"

    assert issuer.finish(ticket) is state, "its real issuer still finishes it normally"
    unresolved = 1 if state is CallState.UNKNOWN else 0
    assert issuer.snapshot() == CapacitySnapshot(slots=1, in_use=0, unresolved=unresolved)
    assert other.finish(held) is CallState.NEVER_DISPATCHED
    assert other.snapshot() == CapacitySnapshot(slots=1, in_use=0, unresolved=0)


@pytest.mark.parametrize("held", [0, 1], ids=["nothing-held", "one-held"])
@pytest.mark.parametrize("state", list(CallState), ids=lambda state: state.value)
def test_an_unissued_record_is_refused_without_changing_the_ledger(state, held):
    capacity = HelperCapacity(2)
    tickets = [capacity.try_acquire() for _ in range(held)]
    before = capacity.snapshot()
    stray = HelperCall()  # constructed directly: no capacity ever issued it
    _drive(stray, state)

    with pytest.raises(RuntimeError):
        capacity.finish(stray)

    assert capacity.snapshot() == before == CapacitySnapshot(slots=2, in_use=held, unresolved=0)
    assert stray.state is state and stray._finished is False
    granted = [capacity.try_acquire() for _ in range(3)]
    assert sum(call is not None for call in granted) == 2 - held, "acquisitions never exceed the slots"
    for ticket in tickets:
        assert capacity.finish(ticket) is CallState.NEVER_DISPATCHED


@pytest.mark.parametrize("state", list(CallState), ids=lambda state: state.value)
def test_a_finished_ticket_is_refused_by_its_issuer_and_by_any_other_capacity(state):
    issuer, other = HelperCapacity(1), HelperCapacity(1)
    ticket = issuer.try_acquire()
    _drive(ticket, state)
    assert issuer.finish(ticket) is state
    issuer_after, other_after = issuer.snapshot(), other.snapshot()

    for capacity in (issuer, other, issuer):
        with pytest.raises(RuntimeError):
            capacity.finish(ticket)

    assert issuer.snapshot() == issuer_after and other.snapshot() == other_after
    assert ticket.observed(_TERMINAL["stop"]) is False, "a late observation changes nothing"
    if state is CallState.UNKNOWN:
        assert issuer_after == CapacitySnapshot(slots=1, in_use=0, unresolved=1)
        assert issuer.try_acquire() is None, "the unresolved slot stays held"
    else:
        assert issuer_after == CapacitySnapshot(slots=1, in_use=0, unresolved=0)
        assert issuer.try_acquire() is not None


def test_no_invalid_finish_can_break_the_slot_bound():
    """Legal work interleaved with every kind of invalid finish: in_use never
    goes negative, free never exceeds the slots, live tickets never exceed the
    slots, and an UNKNOWN slot stays held throughout."""
    slots = 2
    capacity, other = HelperCapacity(slots), HelperCapacity(slots)
    foreign = other.try_acquire()
    live, finished = [], []

    def bounded():
        snapshot = capacity.snapshot()
        assert 0 <= snapshot.in_use <= slots and 0 <= snapshot.free <= slots
        assert snapshot.in_use + snapshot.unresolved <= slots
        assert snapshot.in_use == len(live)

    def refuse_every_invalid_record():
        for record in (foreign, HelperCall(), *finished):
            with pytest.raises(RuntimeError):
                capacity.finish(record)
            bounded()

    for round_number in range(4):
        while (ticket := capacity.try_acquire()) is not None:
            live.append(ticket)
            bounded()
        assert len(live) == slots
        refuse_every_invalid_record()
        for index, ticket in enumerate(live):
            _drive(ticket, CallState.OBSERVED_TERMINAL if (round_number + index) % 2 else CallState.NEVER_DISPATCHED)
        while live:
            ticket = live.pop()
            capacity.finish(ticket)
            finished.append(ticket)
            bounded()
        refuse_every_invalid_record()

    unknown = capacity.try_acquire()
    live.append(unknown)
    _drive(unknown, CallState.UNKNOWN)
    live.remove(unknown)
    assert capacity.finish(unknown) is CallState.UNKNOWN
    finished.append(unknown)
    refuse_every_invalid_record()
    assert capacity.snapshot() == CapacitySnapshot(slots=slots, in_use=0, unresolved=1)
    granted = [capacity.try_acquire() for _ in range(3)]
    assert sum(call is not None for call in granted) == slots - 1
    assert other.snapshot() == CapacitySnapshot(slots=slots, in_use=1, unresolved=0)


def test_concurrent_finishes_of_one_ticket_release_its_slot_exactly_once():
    issuer, other = HelperCapacity(1), HelperCapacity(1)
    ticket = issuer.try_acquire()
    barrier = threading.Barrier(8)
    outcomes = []
    lock = threading.Lock()

    def finish(capacity):
        barrier.wait(timeout=5)
        try:
            outcome = capacity.finish(ticket)
        except RuntimeError:
            outcome = "refused"
        with lock:
            outcomes.append((capacity is issuer, outcome))

    threads = [threading.Thread(target=finish, args=(issuer if n % 2 else other,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(outcomes) == 8
    assert [outcome for _, outcome in outcomes].count(CallState.NEVER_DISPATCHED) == 1
    assert (True, CallState.NEVER_DISPATCHED) in outcomes, "only the issuer can finish it"
    assert issuer.snapshot() == CapacitySnapshot(slots=1, in_use=0, unresolved=0)
    assert other.snapshot() == CapacitySnapshot(slots=1, in_use=0, unresolved=0)


def test_concurrent_acquisition_never_exceeds_the_slots():
    capacity = HelperCapacity(3)
    barrier = threading.Barrier(16)
    acquired = []
    lock = threading.Lock()

    def grab():
        barrier.wait(timeout=5)
        call = capacity.try_acquire()
        if call is not None:
            with lock:
                acquired.append(call)

    threads = [threading.Thread(target=grab) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert len(acquired) == 3
    assert capacity.snapshot().in_use == 3


@pytest.mark.parametrize("value", [0, -1, 65, True, False, 1.0, "1", None])
def test_capacity_settings_must_be_real_positive_ints(value):
    with pytest.raises(AdmissionSettingsError):
        HelperCapacity(value)
    with pytest.raises(AdmissionSettingsError):
        require_positive_int("slots", value, maximum=64)


def test_valid_capacity_settings_are_accepted_unchanged():
    assert require_positive_int("slots", 1, maximum=64) == 1
    assert HelperCapacity(64).snapshot().slots == 64


@pytest.mark.parametrize("module", ["generation_admission.py", "helper_jobs.py"])
def test_no_clock_timer_or_io_can_release_capacity(module):
    """Nothing time-based exists that could reclaim an unresolved slot."""
    tree = ast.parse((_BACKEND / module).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"time", "datetime", "sched", "asyncio", "requests", "socket", "psycopg2", "subprocess", "os"}
    assert imported.isdisjoint(forbidden), sorted(imported & forbidden)
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not names & {"Timer", "sleep", "monotonic", "time"}
