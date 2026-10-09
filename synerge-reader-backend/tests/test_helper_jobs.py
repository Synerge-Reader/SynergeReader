"""Unit contracts for helper_jobs.py, the bounded id-only insights queue.

Pure: no main import, database, model, network or subprocess. Worker threads
run only injected fake callbacks; every wait is bounded by an event timeout.

What these prove: construction starts nothing; at most ``workers`` threads are
ever created -- never one per document or caller -- and ``start`` is
idempotent; admission is immediate under bursts and bounded by ``max_queued``;
items are positive int ids only; dedupe covers queued, running and unresolved
ids; admitted ids run FIFO; no queue lock is held while a job runs; a job whose
call was never dispatched or was observed terminal frees its slot, while an
UNKNOWN call holds its slot and id through close/start with no further job,
retry or busy loop; ``close`` never waits for a running job; and a failed
worker start leaves the queue unavailable.

Start/close publication: a ``close`` that lands inside ``start``'s window --
after it reserved worker tokens, while it starts threads outside the lock --
supersedes that start, which then reports not-ready and leaves intake closed
(proved with a real idle worker and a fake job, not only a no-op thread); a
second ``start`` meanwhile fails fast instead of reporting another attempt's
reserved tokens as ready, and a failure of the pending start never leaves an
accepting, workerless queue; a later explicit ``start`` reopens the same queue
with running and UNKNOWN accounting intact and never more than ``workers``
live threads.
"""

import os
import queue as stdlib_queue
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation_admission import AdmissionSettingsError, CallState
from helper_jobs import (
    BatchAdmission,
    EnqueueOutcome,
    HelperJobQueue,
    HelperQueueSettings,
    require_document_id,
)

_WAIT = 5.0
_TERMINAL = {"response": "{}", "done": True, "done_reason": "stop"}


class _Factory:
    """Counts worker threads created through the queue's thread factory."""

    def __init__(self, fail_after=None):
        self.created = []
        self.fail_after = fail_after

    def __call__(self, *args, **kwargs):
        if self.fail_after is not None and len(self.created) >= self.fail_after:
            raise RuntimeError("SECRET-DETAIL cannot start a thread")
        thread = threading.Thread(*args, **kwargs)
        self.created.append(thread)
        return thread


class _Jobs:
    """run_job double: per-document behaviour, a record of what ran, and gates."""

    def __init__(self):
        self.ran = []
        self.gates = {}
        self.started = {}
        self.behaviour = {}
        self.lock_states = []
        self.threads = {}

    def gate(self, document_id):
        self.gates[document_id] = threading.Event()
        self.started[document_id] = threading.Event()
        return self.gates[document_id]

    def __call__(self, document_id, call):
        self.ran.append(document_id)
        self.threads[document_id] = threading.current_thread()
        if self.queue is not None:
            # Lock-free I/O: neither the queue lock nor the capacity lock is
            # held while a job runs (a non-blocking acquire succeeds). Checked
            # before the test thread is released to touch the queue.
            free = self.queue._lock.acquire(blocking=False)
            if free:
                self.queue._lock.release()
            capacity_free = self.queue._capacity._lock.acquire(blocking=False)
            if capacity_free:
                self.queue._capacity._lock.release()
            self.lock_states.append(free and capacity_free)
        if document_id in self.started:
            self.started[document_id].set()
        if document_id in self.gates:
            assert self.gates[document_id].wait(_WAIT)
        behaviour = self.behaviour.get(document_id, "terminal")
        if behaviour == "terminal":
            call.dispatching()
            call.observed(_TERMINAL)
        elif behaviour == "unknown":
            call.dispatching()  # submitted; the fake server is still working
        elif behaviour == "raise-before-dispatch":
            raise RuntimeError("SECRET-DETAIL load failed")
        elif behaviour == "raise-after-dispatch":
            call.dispatching()
            raise ConnectionError("SECRET-DETAIL reset after send")

    queue = None


def _queue(jobs, *, workers=1, max_queued=4, factory=None, logs=None):
    finished = stdlib_queue.Queue()
    queue = HelperJobQueue(
        settings=HelperQueueSettings(workers=workers, max_queued=max_queued),
        run_job=jobs,
        thread_factory=factory or _Factory(),
        on_finished=lambda document_id, state: finished.put((document_id, state)),
        log=(logs.append if logs is not None else (lambda line: None)),
    )
    jobs.queue = queue
    return queue, finished


@pytest.fixture
def cleanup():
    queues = []
    yield queues.append
    for queue in queues:
        queue.close()


# --- settings and items ---------------------------------------------------------------


def test_default_settings_are_one_worker_and_one_hundred_ids():
    assert HelperQueueSettings() == HelperQueueSettings(workers=1, max_queued=100)


@pytest.mark.parametrize("field", ["workers", "max_queued"])
@pytest.mark.parametrize("value", [0, -1, True, 1.0, "1", None, 10**6])
def test_settings_must_be_bounded_positive_ints(field, value):
    with pytest.raises(AdmissionSettingsError):
        HelperQueueSettings(**{field: value})


@pytest.mark.parametrize("value", [0, -3, True, False, "5", 5.0, None, (5,)])
def test_only_positive_int_document_ids_are_items(value):
    with pytest.raises(ValueError):
        require_document_id(value)
    queue, _ = _queue(_Jobs())
    with pytest.raises(ValueError):
        queue.enqueue(value)


def test_construction_starts_no_thread_and_admits_nothing():
    factory = _Factory()
    jobs = _Jobs()
    queue, _ = _queue(jobs, factory=factory)

    assert factory.created == [] and jobs.ran == []
    assert queue.enqueue(1) is EnqueueOutcome.UNAVAILABLE
    assert queue.enqueue_batch([1, 2]) is None
    assert queue.snapshot().accepting is False and queue.snapshot().queued == 0


# --- bounds, burst and FIFO -------------------------------------------------------------


def test_a_burst_is_admitted_immediately_up_to_the_bound_with_no_extra_threads(cleanup):
    factory = _Factory()
    jobs = _Jobs()
    gate = jobs.gate(1)
    queue, finished = _queue(jobs, factory=factory, max_queued=5)
    cleanup(queue)
    assert queue.start() is True
    queue.enqueue(1)
    assert jobs.started[1].wait(_WAIT), "the worker is busy with document 1"

    outcomes = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def burst(offset):
        barrier.wait(timeout=_WAIT)
        local = [queue.enqueue(offset * 1000 + n) for n in range(2, 102)]
        with lock:
            outcomes.extend(local)

    callers = [threading.Thread(target=burst, args=(k,)) for k in range(8)]
    began = time.monotonic()
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=_WAIT)
    elapsed = time.monotonic() - began

    assert outcomes.count(EnqueueOutcome.ACCEPTED) == 5, "never more than max_queued waiting ids"
    assert outcomes.count(EnqueueOutcome.FULL) == 800 - 5
    assert elapsed < _WAIT, "admission never waits for the running job"
    assert len(factory.created) == 1, "no thread per document or per caller"
    snapshot = queue.snapshot()
    assert (snapshot.queued, snapshot.running, snapshot.workers) == (5, 1, 1)

    gate.set()
    states = [finished.get(timeout=_WAIT) for _ in range(6)]
    assert all(state is CallState.OBSERVED_TERMINAL for _, state in states)
    assert len(factory.created) == 1


def test_admitted_ids_run_in_fifo_order_without_holding_a_lock(cleanup):
    jobs = _Jobs()
    gate = jobs.gate(10)
    queue, finished = _queue(jobs)
    cleanup(queue)
    queue.start()
    queue.enqueue(10)
    assert jobs.started[10].wait(_WAIT)
    for document_id in (13, 11, 12):
        assert queue.enqueue(document_id) is EnqueueOutcome.ACCEPTED

    gate.set()
    order = [finished.get(timeout=_WAIT)[0] for _ in range(4)]

    assert order == [10, 13, 11, 12] == jobs.ran
    assert all(jobs.lock_states), "no queue or capacity lock is held while a job runs"


def test_dedupe_covers_queued_and_running_ids_and_releases_finished_ones(cleanup):
    jobs = _Jobs()
    gate = jobs.gate(20)
    queue, finished = _queue(jobs)
    cleanup(queue)
    queue.start()
    queue.enqueue(20)
    assert jobs.started[20].wait(_WAIT)

    assert queue.enqueue(20) is EnqueueOutcome.DUPLICATE, "running"
    assert queue.enqueue(21) is EnqueueOutcome.ACCEPTED
    assert queue.enqueue(21) is EnqueueOutcome.DUPLICATE, "queued"
    assert queue.enqueue_batch([22, 22, 21, 20]) == BatchAdmission(accepted=1, already_pending=3, capacity_rejected=0)

    gate.set()
    assert {finished.get(timeout=_WAIT)[0] for _ in range(3)} == {20, 21, 22}
    assert queue.enqueue(20) is EnqueueOutcome.ACCEPTED, "a finished id may be queued again"
    finished.get(timeout=_WAIT)


def test_a_full_queue_rejects_and_a_batch_reports_truthful_counts(cleanup):
    jobs = _Jobs()
    gate = jobs.gate(30)
    queue, finished = _queue(jobs, max_queued=2)
    cleanup(queue)
    queue.start()
    queue.enqueue(30)
    assert jobs.started[30].wait(_WAIT)
    assert queue.enqueue(31) is EnqueueOutcome.ACCEPTED

    assert queue.enqueue_batch([31, 32, 33, 34]) == BatchAdmission(accepted=1, already_pending=1, capacity_rejected=2)
    assert queue.enqueue(35) is EnqueueOutcome.FULL
    gate.set()
    assert sorted(finished.get(timeout=_WAIT)[0] for _ in range(3)) == [30, 31, 32]


# --- call outcomes -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "behaviour, state",
    [("terminal", CallState.OBSERVED_TERMINAL), ("raise-before-dispatch", CallState.NEVER_DISPATCHED)],
)
def test_a_never_dispatched_or_terminal_job_frees_its_slot_and_id(cleanup, behaviour, state):
    jobs = _Jobs()
    jobs.behaviour[40] = behaviour
    queue, finished = _queue(jobs)
    cleanup(queue)
    queue.start()
    queue.enqueue(40)

    assert finished.get(timeout=_WAIT) == (40, state)
    queue.enqueue(41)
    assert finished.get(timeout=_WAIT) == (41, CallState.OBSERVED_TERMINAL), "the slot came back"
    snapshot = queue.snapshot()
    assert (snapshot.unresolved, snapshot.free_capacity) == (0, 1)


@pytest.mark.parametrize("behaviour", ["unknown", "raise-after-dispatch"])
def test_an_unknown_call_holds_its_slot_and_id_with_no_further_job(cleanup, behaviour):
    jobs = _Jobs()
    jobs.behaviour[50] = behaviour
    logs = []
    factory = _Factory()
    queue, finished = _queue(jobs, factory=factory, logs=logs)
    cleanup(queue)
    queue.start()
    queue.enqueue(50)

    assert finished.get(timeout=_WAIT) == (50, CallState.UNKNOWN)
    assert queue.enqueue(50) is EnqueueOutcome.DUPLICATE, "the unresolved id is retained"
    assert queue.enqueue(51) is EnqueueOutcome.ACCEPTED
    with pytest.raises(stdlib_queue.Empty):
        finished.get(timeout=0.3)

    queue.close()
    assert queue.start() is True, "a later start reopens intake but cannot bypass the held slot"
    with pytest.raises(stdlib_queue.Empty):
        finished.get(timeout=0.3)

    assert jobs.ran == [50], "no next job and no retry"
    snapshot = queue.snapshot()
    assert (snapshot.unresolved, snapshot.free_capacity, snapshot.queued) == (1, 0, 1)
    assert snapshot.workers == 1 and len(factory.created) <= 2
    assert logs == ["[Insights] Background model call unresolved; capacity held: 1"]


# --- lifecycle --------------------------------------------------------------------------


def test_start_is_idempotent_and_never_exceeds_the_worker_count(cleanup):
    factory = _Factory()
    queue, _ = _queue(_Jobs(), workers=2, factory=factory)
    cleanup(queue)

    assert queue.start() is True and queue.start() is True
    assert len(factory.created) == 2
    assert queue.snapshot().workers == 2


def test_close_does_not_wait_for_a_running_job_and_the_job_still_finishes(cleanup):
    jobs = _Jobs()
    gate = jobs.gate(60)
    factory = _Factory()
    queue, finished = _queue(jobs, factory=factory)
    cleanup(queue)
    queue.start()
    queue.enqueue(60)
    assert jobs.started[60].wait(_WAIT)

    began = time.monotonic()
    queue.close()
    assert time.monotonic() - began < 1.0, "close never joins or waits"
    assert queue.enqueue(61) is EnqueueOutcome.UNAVAILABLE
    assert queue.snapshot().workers == 1, "the busy worker is still live"
    assert queue.start() is True
    assert len(factory.created) == 1, "a live worker is never duplicated"

    gate.set()
    assert finished.get(timeout=_WAIT) == (60, CallState.OBSERVED_TERMINAL)


def test_closed_idle_workers_exit_and_a_later_start_replaces_them(cleanup):
    factory = _Factory()
    queue, finished = _queue(_Jobs(), factory=factory)
    cleanup(queue)
    queue.start()
    queue.close()
    factory.created[0].join(timeout=_WAIT)
    assert not factory.created[0].is_alive() and queue.snapshot().workers == 0

    assert queue.start() is True
    assert len(factory.created) == 2 and queue.snapshot().workers == 1
    queue.enqueue(70)
    assert finished.get(timeout=_WAIT) == (70, CallState.OBSERVED_TERMINAL)


@pytest.mark.parametrize("fail_after", [0, 1])
def test_a_failed_worker_start_leaves_the_queue_unavailable(cleanup, fail_after):
    logs = []
    factory = _Factory(fail_after=fail_after)
    jobs = _Jobs()
    queue, _ = _queue(jobs, workers=2, factory=factory, logs=logs)
    cleanup(queue)

    assert queue.start() is False
    assert queue.enqueue(80) is EnqueueOutcome.UNAVAILABLE
    assert queue.enqueue_batch([81]) is None
    assert logs == ["[Insights] Queue worker could not start; background insights unavailable"]
    for thread in factory.created:
        thread.join(timeout=_WAIT)
        assert not thread.is_alive(), "a worker started before the failure exits"
    assert jobs.ran == []


def test_a_failing_finish_callback_does_not_kill_the_worker(cleanup):
    jobs = _Jobs()
    seen = []

    def explode(document_id, state):
        seen.append(document_id)
        raise RuntimeError("callback failed")

    queue = HelperJobQueue(
        settings=HelperQueueSettings(workers=1, max_queued=4), run_job=jobs, on_finished=explode,
        log=lambda line: None,
    )
    jobs.queue = queue
    cleanup(queue)
    queue.start()
    queue.enqueue(90)
    queue.enqueue(91)
    deadline = time.monotonic() + _WAIT
    while len(seen) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert seen == [90, 91]


# --- start/close publication ------------------------------------------------------------


class _WindowFactory(_Factory):
    """Real worker threads. The NEXT thread's ``start()`` runs
    ``window(thread, real_start)`` instead: code that executes inside
    HelperJobQueue.start() after it reserved that worker's token and before it
    publishes its outcome. Later threads start normally."""

    def __init__(self):
        super().__init__()
        self.window = None

    def __call__(self, *args, **kwargs):
        thread = super().__call__(*args, **kwargs)
        window, self.window = self.window, None
        if window is not None:
            real_start = thread.start
            thread.start = lambda: window(thread, real_start)
        return thread


class _ObservedCondition(threading.Condition):
    """The queue's condition over the queue's own lock, reporting when a worker
    parks in it. ``parked`` is set while that lock is still held, so nothing
    else can take the lock until the worker is really waiting."""

    def __init__(self, lock):
        super().__init__(lock)
        self.parked = threading.Event()

    def wait(self, timeout=None):
        self.parked.set()
        return super().wait(timeout)


class _LockCheckingLog(list):
    """A log list that also records whether a queue or capacity lock was held
    while each line was written."""

    def __init__(self):
        super().__init__()
        self.queue = None
        self.unlocked = []

    def append(self, line):
        self.unlocked.append(_unlocked(self.queue))
        super().append(line)


def _unlocked(queue):
    """Neither the queue lock nor the capacity lock is held by anyone."""
    free = queue._lock.acquire(blocking=False)
    if free:
        queue._lock.release()
    capacity_free = queue._capacity._lock.acquire(blocking=False)
    if capacity_free:
        queue._capacity._lock.release()
    return free and capacity_free


@pytest.mark.parametrize("moment", ["before-the-worker-runs", "while-the-worker-is-idle"])
def test_a_close_during_startup_keeps_intake_closed_and_dispatches_nothing(cleanup, moment):
    jobs = _Jobs()
    logs = []
    factory = _WindowFactory()
    queue, finished = _queue(jobs, factory=factory, logs=logs)
    cleanup(queue)
    wake = queue._wake = _ObservedCondition(queue._lock)
    window = {}

    def close_inside_the_window(thread, real_start):
        window["unlocked"] = _unlocked(queue)
        if moment == "before-the-worker-runs":
            queue.close()
            real_start()
        else:
            real_start()
            window["parked"] = wake.parked.wait(_WAIT)  # the real worker is idle in the queue's wait
            queue.close()
        thread.join(_WAIT)  # test-only bounded wait: the worker saw the close and exited
        window["worker exited"] = not thread.is_alive()

    factory.window = close_inside_the_window
    started = queue.start()
    snapshot = queue.snapshot()
    admitted = queue.enqueue(1)
    batch = queue.enqueue_batch([2, 3])

    expected_window = {"unlocked": True, "worker exited": True}
    if moment == "while-the-worker-is-idle":
        expected_window["parked"] = True
    assert window == expected_window, "threads start outside every lock; the worker really exited"
    assert (started, snapshot.accepting, snapshot.workers, admitted, batch) == (
        False, False, 0, EnqueueOutcome.UNAVAILABLE, None
    ), "a start overtaken by close never claims readiness or admits work nobody can run"
    with pytest.raises(stdlib_queue.Empty):
        finished.get(timeout=0.3)
    assert jobs.ran == [], "no fake model job was dispatched"
    assert logs == [], "a superseded start is not reported as a worker-start failure"

    assert queue.start() is True, "a later explicit start reopens the same queue"
    assert queue.enqueue(4) is EnqueueOutcome.ACCEPTED
    assert finished.get(timeout=_WAIT) == (4, CallState.OBSERVED_TERMINAL)
    assert jobs.ran == [4]
    assert len(factory.created) == 2 and queue.snapshot().workers == 1


@pytest.mark.parametrize("pending", ["starts", "fails"])
def test_a_concurrent_start_never_reports_a_pending_startup_as_ready(cleanup, pending):
    jobs = _Jobs()
    log = _LockCheckingLog()
    factory = _WindowFactory()
    queue, finished = _queue(jobs, factory=factory, logs=log)
    log.queue = queue
    cleanup(queue)
    reserved, release = threading.Event(), threading.Event()
    window = {}

    def slow_thread_start(thread, real_start):
        window["unlocked"] = _unlocked(queue)
        reserved.set()
        window["released"] = release.wait(_WAIT)
        if pending == "fails":
            raise RuntimeError("SECRET-DETAIL cannot start a thread")
        real_start()

    factory.window = slow_thread_start
    first_result = []
    first = threading.Thread(target=lambda: first_result.append(queue.start()))
    first.start()
    try:
        assert reserved.wait(_WAIT), "the first start reserved its worker and is starting it"
        began = time.monotonic()
        second = queue.start()
        elapsed = time.monotonic() - began
        during = queue.snapshot()
        admitted = queue.enqueue(1)
    finally:
        release.set()
        first.join(_WAIT)
    assert not first.is_alive()

    assert (second, during.accepting, admitted) == (False, False, EnqueueOutcome.UNAVAILABLE), (
        "another start's reserved tokens are not live workers: nothing is ready or admitted yet"
    )
    assert elapsed < 1.0, "the second start fails fast instead of waiting for the first"
    assert len(factory.created) == 1, "the second start created no thread of its own"
    assert window == {"unlocked": True, "released": True}

    if pending == "starts":
        assert first_result == [True]
        assert queue.enqueue(2) is EnqueueOutcome.ACCEPTED
        assert finished.get(timeout=_WAIT) == (2, CallState.OBSERVED_TERMINAL)
        assert queue.snapshot().workers == 1 and log == []
        assert jobs.ran == [2]
    else:
        assert first_result == [False]
        snapshot = queue.snapshot()
        assert (snapshot.accepting, snapshot.workers) == (False, 0), "never an accepting, workerless queue"
        assert queue.enqueue(3) is EnqueueOutcome.UNAVAILABLE
        assert log == ["[Insights] Queue worker could not start; background insights unavailable"]
        assert log.unlocked == [True], "the failure is logged outside every lock"
        assert jobs.ran == []
        assert queue.start() is True, "once the failed attempt settled, an explicit start reopens the queue"
        assert queue.enqueue(4) is EnqueueOutcome.ACCEPTED
        assert finished.get(timeout=_WAIT) == (4, CallState.OBSERVED_TERMINAL)
        assert len(factory.created) == 2 and queue.snapshot().workers == 1
        assert jobs.ran == [4]


def test_a_later_start_after_an_interrupted_one_keeps_running_and_unknown_work(cleanup):
    jobs = _Jobs()
    jobs.behaviour[50] = "unknown"
    gate = jobs.gate(60)
    logs = []
    factory = _WindowFactory()
    queue, finished = _queue(jobs, workers=2, factory=factory, logs=logs)
    cleanup(queue)
    try:
        assert queue.start() is True
        queue.enqueue(50)
        assert finished.get(timeout=_WAIT) == (50, CallState.UNKNOWN)
        queue.enqueue(60)
        assert jobs.started[60].wait(_WAIT)
        assert queue.enqueue(61) is EnqueueOutcome.ACCEPTED, "it waits: one slot unresolved, one running"
        busy = jobs.threads[60]
        [idle] = [thread for thread in factory.created if thread is not busy]

        queue.close()
        idle.join(_WAIT)  # test-only bounded wait
        assert not idle.is_alive() and queue.snapshot().workers == 1, "only the busy worker remains"

        window = {}

        def close_inside_the_window(thread, real_start):
            real_start()
            queue.close()
            thread.join(_WAIT)
            window["worker exited"] = not thread.is_alive()

        factory.window = close_inside_the_window
        interrupted = queue.start()
        assert (interrupted, window, queue.snapshot().accepting) == (False, {"worker exited": True}, False)
        assert queue.enqueue(62) is EnqueueOutcome.UNAVAILABLE

        assert queue.start() is True, "the same queue reopens once the interrupted start has settled"
        snapshot = queue.snapshot()
        assert (snapshot.accepting, snapshot.running, snapshot.unresolved, snapshot.queued) == (True, 1, 1, 1)
        assert (snapshot.workers, snapshot.free_capacity) == (2, 0)
        for document_id in (50, 60, 61):
            assert queue.enqueue(document_id) is EnqueueOutcome.DUPLICATE
        assert len(factory.created) == 4
        assert sum(thread.is_alive() for thread in factory.created) == 2, "never more than `workers` live threads"
        assert busy.is_alive(), "the busy worker was neither replaced nor duplicated"

        gate.set()
        done = {finished.get(timeout=_WAIT), finished.get(timeout=_WAIT)}
        assert done == {(60, CallState.OBSERVED_TERMINAL), (61, CallState.OBSERVED_TERMINAL)}
    finally:
        gate.set()

    assert jobs.ran == [50, 60, 61], "the UNKNOWN call is never retried or replaced"
    snapshot = queue.snapshot()
    assert (snapshot.unresolved, snapshot.free_capacity, snapshot.queued, snapshot.running) == (1, 1, 0, 0)
    assert logs == ["[Insights] Background model call unresolved; capacity held: 1"]
