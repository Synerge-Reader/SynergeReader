"""Unit contracts for generation_admission.py: the original helper accounting
and, in the second half, the shared GenerationLedger.

Pure: no main import, database, model, network or subprocess. Threads appear
only in the concurrency tests, running in-process code.

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


# ===========================================================================
# The shared GenerationLedger: answers, their verifier calls and helpers
# ===========================================================================
#
# What these prove: nothing is admitted or dispatched before an owner installs
# a durable intent sink or after the ledger is fenced; the total, answer and
# helper bounds hold together, unresolved work (recovered or this process's)
# counts against all of them and is never released; waiting answers are
# bounded, FIFO and served before helpers; a waiter's outcome is decided once;
# every model call records its durable intent BEFORE it may be sent and is
# refused when that fails; calls are sequential, so one slot never covers two
# possibly-running calls; admission captures the selection, and a pause/resume
# changes it only for jobs admitted afterwards; release is once-only.

from generation_admission import (  # noqa: E402
    AdmissionRefusal,
    AdmissionRefused,
    AdmissionSettings,
    AnswerWaiter,
    DispatchIntents,
    DispatchRefused,
    GenerationLedger,
    GenerationTicket,
    HelperPool,
    JobKind,
    SharedSelection,
    current_dispatch_permitted,
    current_generation_ticket,
    generation_scope,
)


class _Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


class _Intents(DispatchIntents):
    """Records every durable write in order; ``fail`` makes record_dispatch raise."""

    def __init__(self):
        self.events = []
        self.fail = False

    def record_dispatch(self, intent):
        self.events.append(("dispatch", intent.call_id, intent.call_role, intent.profile_id, intent.selection_epoch))
        if self.fail:
            raise RuntimeError("SECRET-DETAIL database unavailable")

    def record_outcome(self, intent, state):
        self.events.append(("outcome", intent.call_id, state))


class _Snapshot:
    def __init__(self, profile_id="llama_control", epoch=1, model_tag="llama3.1:8b"):
        self.profile_id = profile_id
        self.selection_epoch = epoch
        self.model_tag = model_tag


_TERMINAL_BODY = {"response": "x", "done": True, "done_reason": "stop"}


def _ledger(clock=None, *, owned=True, selection=SharedSelection("llama_control", 1), recovered=None, **settings):
    values = dict(total_slots=2, answer_slots=2, helper_slots=1, answer_waiters=2)
    values.update(settings)
    ledger = GenerationLedger(AdmissionSettings(**values), clock=clock or _Clock())
    intents = _Intents()
    if owned:
        assert ledger.install_owner(intents=intents, owner_seq=1, selection=selection, paused_by=None,
                                    recovered=recovered or {})
    return ledger, intents


def _admit_answer(ledger):
    """An answer ticket admitted at once (no waiting), already started."""
    admitted = ledger.enqueue_answer(lambda: None)
    assert isinstance(admitted, GenerationTicket)
    assert ledger.start(admitted)
    admitted.bind_snapshot(_Snapshot(admitted.selection.profile_id, admitted.selection.epoch))
    return admitted


def _helper(ledger):
    ticket = ledger.try_acquire_helper()
    assert ticket is not None
    ticket.bind_snapshot(_Snapshot(ticket.selection.profile_id, ticket.selection.epoch))
    return ticket


def test_nothing_is_admitted_before_an_owner_installs_the_ledger_or_after_it_is_fenced():
    ledger, _ = _ledger(owned=False)
    with pytest.raises(AdmissionRefused) as refused:
        ledger.enqueue_answer(lambda: None)
    assert refused.value.reason is AdmissionRefusal.UNAVAILABLE
    assert ledger.try_acquire_helper() is None

    ledger, intents = _ledger()
    running = _admit_answer(ledger)
    ledger.fence()
    with pytest.raises(AdmissionRefused) as refused:
        ledger.enqueue_answer(lambda: None)
    assert refused.value.reason is AdmissionRefusal.UNAVAILABLE
    with pytest.raises(DispatchRefused):
        running.dispatching("answer")
    assert intents.events == [], "a fenced (stale) owner records and sends nothing"
    assert not ledger.install_owner(intents=_Intents(), owner_seq=2, selection=SharedSelection("llama_control", 1),
                                    paused_by=None, recovered={}), "ownership is installed once per process"


def test_total_answer_and_helper_bounds_hold_together():
    ledger, _ = _ledger(total_slots=2, answer_slots=2, helper_slots=1)
    helper = _helper(ledger)
    answer = _admit_answer(ledger)
    assert ledger.try_acquire_helper() is None, "H=1"
    admitted = ledger.enqueue_answer(lambda: None)
    assert isinstance(admitted, AnswerWaiter), "T=2 is reached by one helper plus one answer"
    assert ledger.settle_answer_waiter(admitted) is None
    assert ledger.release(helper) is CallState.NEVER_DISPATCHED
    second = _admit_answer(ledger)
    snap = ledger.snapshot()
    assert (snap.answers_running, snap.helpers_running) == (2, 0)
    assert ledger.try_acquire_helper() is None, "G answers fill T"
    for ticket in (answer, second):
        ledger.release(ticket)
    assert ledger.snapshot().answers_running == 0


def test_unresolved_work_holds_capacity_across_every_bound_and_is_never_released():
    ledger, _ = _ledger(recovered={JobKind.ANSWER: 1})
    answer = _admit_answer(ledger)
    answer.dispatching("answer")  # never observed: UNKNOWN
    assert ledger.release(answer) is CallState.UNKNOWN
    snap = ledger.snapshot()
    assert (snap.unresolved_answers, snap.answers_running) == (2, 0)
    with pytest.raises(AdmissionRefused) as refused:
        ledger.enqueue_answer(lambda: None)
    assert refused.value.reason is AdmissionRefusal.UNRESOLVED, "waiting cannot help: refused at once"
    assert ledger.try_acquire_helper() is None, "unresolved answers fill T for helpers too"
    with pytest.raises(RuntimeError):
        ledger.release(answer)
    assert ledger.unresolved_total() == 2


def test_waiting_answers_are_bounded_fifo_and_served_before_helpers():
    ledger, _ = _ledger(total_slots=1, answer_slots=1, helper_slots=1, answer_waiters=2)
    holder = _admit_answer(ledger)
    woken = []
    first = ledger.enqueue_answer(lambda: woken.append("first"))
    second = ledger.enqueue_answer(lambda: woken.append("second"))
    with pytest.raises(AdmissionRefused) as refused:
        ledger.enqueue_answer(lambda: None)
    assert refused.value.reason is AdmissionRefusal.BUSY
    assert ledger.release(holder) is CallState.NEVER_DISPATCHED
    assert woken == ["first"] and ledger.try_acquire_helper() is None, "a waiting answer takes the slot"
    granted = ledger.settle_answer_waiter(first)
    assert isinstance(granted, GenerationTicket)
    assert ledger.settle_answer_waiter(first) is None, "a grant is collected once"
    assert ledger.release_unstarted(granted)
    assert woken == ["first", "second"]
    assert ledger.release_unstarted(ledger.settle_answer_waiter(second))
    assert ledger.try_acquire_helper() is not None, "with no answer waiting, a helper may take the slot"


def test_a_waiter_outcome_is_decided_once_and_an_unreachable_waiter_returns_its_slot():
    ledger, _ = _ledger(total_slots=1, answer_slots=1, helper_slots=1)
    holder = _admit_answer(ledger)
    withdrawn = ledger.enqueue_answer(lambda: None)
    assert ledger.settle_answer_waiter(withdrawn) is None  # its deadline passed first
    ledger.release(holder)
    assert ledger.snapshot().answers_running == 0, "a withdrawn waiter is never granted"

    holder = _admit_answer(ledger)

    def closed_loop():
        raise RuntimeError("event loop is closed")

    unreachable = ledger.enqueue_answer(closed_loop)
    ledger.release(holder)
    snap = ledger.snapshot()
    assert (snap.answers_running, snap.answer_waiters) == (0, 0), "the grant nobody can collect is returned"
    assert ledger.settle_answer_waiter(unreachable) is None


def test_every_call_records_a_durable_intent_before_it_may_be_sent():
    ledger, intents = _ledger()
    answer = _admit_answer(ledger)
    answer.dispatching("answer")
    [(kind, call_id, role, profile, epoch)] = intents.events
    assert (kind, role, profile, epoch) == ("dispatch", "answer", "llama_control", 1)
    assert answer.observed({"response": "x", "done": True, "done_reason": "length"}), "length is terminal"
    assert intents.events[-1] == ("outcome", call_id, CallState.OBSERVED_TERMINAL)
    answer.dispatching("verifier")
    assert answer.observed(_TERMINAL_BODY)
    assert ledger.release(answer) is CallState.OBSERVED_TERMINAL

    failing = _admit_answer(ledger)
    intents.fail = True
    with pytest.raises(DispatchRefused) as refused:
        failing.dispatching("answer")
    assert "SECRET" not in str(refused.value) and refused.value.__cause__ is None
    assert intents.events[-1][0] == "outcome" and intents.events[-1][2] is CallState.NEVER_DISPATCHED, (
        "the possibly written intent of a call that was never sent is removed"
    )
    assert failing.state is CallState.NEVER_DISPATCHED
    assert ledger.release(failing) is CallState.NEVER_DISPATCHED and ledger.unresolved_total() == 0


def test_calls_are_sequential_and_an_unobserved_call_blocks_the_next_one():
    ledger, intents = _ledger()
    answer = _admit_answer(ledger)
    answer.dispatching("answer")
    with pytest.raises(DispatchRefused):
        answer.dispatching("verifier")
    assert not answer.may_dispatch
    assert not answer.observed({"response": "x", "done": True, "done_reason": "load"})
    with pytest.raises(DispatchRefused):
        answer.dispatching("verifier")
    assert ledger.release(answer) is CallState.UNKNOWN
    dispatched = [e for e in intents.events if e[0] == "dispatch"]
    assert len(dispatched) == 1
    assert intents.events[-1] == ("outcome", dispatched[0][1], CallState.UNKNOWN), "durably marked unknown"


def test_a_stream_counts_as_terminal_only_with_its_final_record_and_no_bad_line():
    from ollama_generation import OllamaGenerateStream

    cases = {
        b'{"response":"a"}\n{"response":"","done":true,"done_reason":"stop"}\n': True,
        b'{"response":"a"}\n{"response":"","done":true,"done_reason":"length"}\n': True,
        b'{"response":"a"}\n': False,
        b'{"response":"a"}\n{"error":"x"}\n': False,
        b'{"response":"a"}\nnot json\n{"response":"","done":true,"done_reason":"stop"}\n': False,
        b'{"response":"","done":true,"done_reason":"load"}\n': False,
    }
    for data, terminal in cases.items():
        stream = OllamaGenerateStream()
        stream.feed(data)
        stream.close()
        ledger, _ = _ledger()  # an UNKNOWN case keeps its slot for good
        ticket = _admit_answer(ledger)
        ticket.dispatching("answer")
        assert ticket.observed_stream(stream) is terminal
        assert ledger.release(ticket) is (CallState.OBSERVED_TERMINAL if terminal else CallState.UNKNOWN)


def test_admission_captures_the_selection_and_a_switch_changes_it_only_for_later_jobs():
    ledger, _ = _ledger()
    old = _admit_answer(ledger)
    paused = ledger.pause("switch-1")
    assert paused == SharedSelection("llama_control", 1)
    for refusal_check in (lambda: ledger.enqueue_answer(lambda: None),):
        with pytest.raises(AdmissionRefused) as refused:
            refusal_check()
        assert refused.value.reason is AdmissionRefusal.SWITCHING
    assert ledger.try_acquire_helper() is None
    assert ledger.pause("switch-2") is None, "one pause at a time"
    assert not ledger.resume("switch-2", SharedSelection("gemma_candidate", 3))
    old.dispatching("answer")  # admitted work keeps running with its own selection
    assert old.observed(_TERMINAL_BODY)
    assert ledger.resume("switch-1", SharedSelection("gemma_candidate", 3))
    new = ledger.try_acquire_helper()
    assert (old.selection, new.selection) == (SharedSelection("llama_control", 1), SharedSelection("gemma_candidate", 3))
    ledger.release(old)
    ledger.release(new)


def test_a_paused_waiter_is_refused_with_the_switching_reason():
    ledger, _ = _ledger(total_slots=1, answer_slots=1)
    holder = _admit_answer(ledger)
    woken = []
    waiter = ledger.enqueue_answer(lambda: woken.append(1))
    assert ledger.pause("switch-1") is not None
    assert woken == [1] and ledger.settle_answer_waiter(waiter) is None
    assert waiter.refusal is AdmissionRefusal.SWITCHING
    ledger.release(holder)


def test_release_is_once_only_and_handoff_transitions_are_atomic():
    ledger, _ = _ledger()
    other, _ = _ledger()
    admitted = ledger.enqueue_answer(lambda: None)
    with pytest.raises(RuntimeError):
        ledger.release(admitted)  # an unstarted ticket is released only as unstarted
    assert not other.start(admitted) and not other.release_unstarted(admitted), "a foreign ledger refuses it"
    assert ledger.release_unstarted(admitted)
    assert not ledger.release_unstarted(admitted), "idempotent"
    assert not ledger.start(admitted), "an expired (released) ticket never starts late"
    started = _admit_answer(ledger)
    assert not ledger.release_unstarted(started), "a started ticket is never reclaimed by the lease"
    assert ledger.release(started) is CallState.NEVER_DISPATCHED
    with pytest.raises(RuntimeError):
        ledger.release(started)
    with pytest.raises(TypeError):
        ledger.release(HelperCall())
    assert ledger.snapshot().answers_running == 0


def test_a_snapshot_must_come_from_the_tickets_own_selection():
    ledger, _ = _ledger()
    ticket = ledger.try_acquire_helper()
    with pytest.raises(DispatchRefused):
        ticket.dispatching("helper")  # no snapshot fixed yet
    with pytest.raises(RuntimeError):
        ticket.bind_snapshot(_Snapshot("gemma_candidate", 1))
    with pytest.raises(RuntimeError):
        ticket.bind_snapshot(_Snapshot("llama_control", 2))
    ticket.bind_snapshot(_Snapshot())
    with pytest.raises(RuntimeError):
        ticket.bind_snapshot(_Snapshot())
    ledger.release(ticket)


def test_the_job_budget_clamps_reads_and_refuses_dispatch_once_spent():
    clock = _Clock(100.0)
    ledger, intents = _ledger(clock, answer_job_budget_s=60.0)
    ticket = _admit_answer(ledger)
    assert ticket.read_timeout(150.0) == 60.0
    clock.now = 140.0
    assert ticket.read_timeout(150.0) == 20.0 and ticket.read_timeout(5.0) == 5.0
    clock.now = 160.0
    assert ticket.expired and ticket.read_timeout(150.0) == 0.5
    with pytest.raises(DispatchRefused):
        ticket.dispatching("answer")
    assert intents.events == [] and not ticket.may_dispatch
    ledger.release(ticket)


def test_the_thread_scope_names_the_dispatching_job_and_its_stop():
    ledger, _ = _ledger()
    ticket = _admit_answer(ledger)
    assert current_generation_ticket() is None and not current_dispatch_permitted()
    stopped = {"value": False}
    with generation_scope(ticket, stopped=lambda: stopped["value"]):
        assert current_generation_ticket() is ticket and current_dispatch_permitted()
        stopped["value"] = True
        assert not current_dispatch_permitted(), "known abandonment stops new dispatch"
    assert current_generation_ticket() is None
    ledger.release(ticket)


@pytest.mark.parametrize(
    "fields",
    [dict(total_slots=0), dict(answer_slots=3, total_slots=2), dict(helper_slots=True), dict(answer_wait_s=0.0),
     dict(answer_wait_s=float("nan")), dict(answer_job_budget_s=0.5), dict(unstarted_lease_s="1")],
    ids=["zero-total", "sublimit-above-total", "bool", "zero-wait", "nan-wait", "budget-too-small", "str-lease"],
)
def test_admission_settings_are_validated(fields):
    with pytest.raises(AdmissionSettingsError):
        AdmissionSettings(**fields)


def test_the_helper_pool_is_the_helper_sublimit_and_announces_changes():
    ledger, _ = _ledger()
    pool = HelperPool(ledger)
    changes = []
    pool.add_listener(lambda: changes.append(1))
    ticket = pool.try_acquire()
    assert isinstance(ticket, GenerationTicket) and ticket.kind is JobKind.HELPER
    assert pool.free == 0 and pool.snapshot() == CapacitySnapshot(slots=1, in_use=1, unresolved=0)
    assert pool.finish(ticket) is CallState.NEVER_DISPATCHED
    assert changes and pool.free == 1


def test_concurrent_admission_never_exceeds_the_bounds():
    ledger, _ = _ledger(total_slots=3, answer_slots=2, helper_slots=2, answer_waiters=8)
    peak = {"answers": 0, "helpers": 0, "total": 0}
    lock = threading.Lock()

    def worker(kind):
        for _ in range(200):
            if kind == "helper":
                ticket = ledger.try_acquire_helper()
            else:
                admitted = ledger.enqueue_answer(lambda: None)
                ticket = admitted if isinstance(admitted, GenerationTicket) else ledger.settle_answer_waiter(admitted)
                if ticket is not None:
                    ledger.start(ticket)
            if ticket is None:
                continue
            snap = ledger.snapshot()
            with lock:
                peak["answers"] = max(peak["answers"], snap.answers_running)
                peak["helpers"] = max(peak["helpers"], snap.helpers_running)
                peak["total"] = max(peak["total"], snap.answers_running + snap.helpers_running)
            ledger.release(ticket)

    threads = [threading.Thread(target=worker, args=(kind,)) for kind in ("answer", "answer", "helper", "helper")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert peak["answers"] <= 2 and peak["helpers"] <= 2 and peak["total"] <= 3
    snap = ledger.snapshot()
    assert (snap.answers_running, snap.helpers_running, snap.answer_waiters) == (0, 0, 0)
