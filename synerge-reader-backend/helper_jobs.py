"""Bounded, ID-only queue for background document insights (B2a-1).

One process-local queue holds at most ``max_queued`` waiting document ids and
runs them on a fixed set of ``workers`` threads. It replaces starting one
daemon thread per uploaded or pending document.

Capacity: by default the workers share a private
``generation_admission.HelperCapacity(workers)``. The application injects the
shared ledger's helper sublimit instead (``generation_admission.HelperPool``):
then a worker takes a job only when the ledger grants a helper ticket, which
captures the shared model selection at that moment -- never when the id was
queued -- and the ledger's answer priority, switch pauses and unresolved work
all apply. The injected capacity's change notifications wake idle workers.

What an item is: a positive ``int`` document id -- nothing else. No document
text, filename, token, reply or closure over them is ever stored; the job
callback reloads whatever it needs when it actually starts.

Concurrency rules:

* ``enqueue``/``enqueue_batch`` never block on work: one short critical section
  over in-memory state, safe to call from the event loop.
* A worker takes the next id only when the capacity grants it a call record,
  then runs the injected ``run_job(document_id, call)`` OUTSIDE every lock. No
  lock is held across SQL, HTTP, model resolution, a response close, a join or
  a log line. The only lock nesting is queue lock -> capacity lock, inside the
  grant; capacity notifications are delivered outside the capacity's lock.
* Dedupe covers queued, running and unresolved ids, so retained state is
  bounded by ``max_queued + workers``. Order among admitted ids is FIFO.
* An ``UNKNOWN`` call (see generation_admission) keeps its capacity slot and its
  document id for the rest of the process. Workers then wait on a condition
  -- no busy loop, no timer, no automatic retry, and no replacement worker
  that could bypass the held slot.
* ``close`` stops intake and signals idle workers, and never waits for them;
  a running job finishes on its own. ``start`` is idempotent: it reopens
  intake and tops the live workers back up to ``workers``, keeping every
  accounting record, so a later lifespan in the same process can neither
  duplicate live workers nor clear unresolved work.
* ``start`` reserves worker tokens under the lock, starts threads outside it,
  then publishes its outcome. Every ``start`` and ``close`` replaces an
  in-memory lifecycle marker, and a ``start`` reopens intake only if its marker
  is still current: a ``close`` that lands while it is starting threads wins,
  and that ``start`` reports not-ready. Another ``start`` during that window
  fails fast instead of reporting reserved tokens as live workers; once the
  pending one has settled, an explicit ``start`` may reopen the same queue.

Constructing a queue performs no I/O and starts no thread. Nothing is
persisted: queued ids are lost when the process exits (recoverable through
admin pending analysis, which reselects documents without insights).
"""

from __future__ import annotations

import collections
import enum
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from generation_admission import CallState, HelperCall, HelperCapacity, require_positive_int

MAX_WORKERS = 8
MAX_QUEUED = 10_000

_QUEUED = "queued"
_RUNNING = "running"
_UNRESOLVED = "unresolved"


class EnqueueOutcome(enum.Enum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    FULL = "full"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class HelperQueueSettings:
    """``workers`` threads (and as many background model slots) and at most
    ``max_queued`` waiting ids. Validated on construction."""

    workers: int = 1
    max_queued: int = 100

    def __post_init__(self) -> None:
        require_positive_int("workers", self.workers, maximum=MAX_WORKERS)
        require_positive_int("max_queued", self.max_queued, maximum=MAX_QUEUED)


@dataclass(frozen=True)
class BatchAdmission:
    accepted: int
    already_pending: int
    capacity_rejected: int


@dataclass(frozen=True)
class QueueSnapshot:
    accepting: bool
    queued: int
    running: int
    unresolved: int
    workers: int
    free_capacity: int


def require_document_id(value: object) -> int:
    """A document id is a positive int (``documents.id`` is SERIAL); never a bool or str."""
    if type(value) is not int or value < 1:
        raise ValueError("a queued document id must be a positive integer")
    return value


class HelperJobQueue:
    def __init__(
        self,
        *,
        settings: HelperQueueSettings,
        run_job: Callable[[int, HelperCall], None],
        thread_factory: Callable[..., object] = threading.Thread,
        on_finished: Optional[Callable[[int, CallState], None]] = None,
        log: Callable[[str], None] = print,
        capacity: Optional[object] = None,
    ) -> None:
        if not isinstance(settings, HelperQueueSettings):
            raise TypeError("settings must be HelperQueueSettings")
        if not callable(run_job):
            raise TypeError("run_job must be callable")
        if capacity is not None and not all(
            callable(getattr(capacity, name, None)) for name in ("try_acquire", "finish", "snapshot")
        ):
            raise TypeError("capacity needs try_acquire, finish and snapshot")
        self._settings = settings
        self._run_job = run_job
        self._thread_factory = thread_factory
        self._on_finished = on_finished
        self._log = log
        self._capacity = capacity if capacity is not None else HelperCapacity(settings.workers)
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._queued = collections.deque()
        self._keys = {}  # document id -> _QUEUED | _RUNNING | _UNRESOLVED
        self._accepting = False
        self._stopping = False
        self._workers = set()  # one opaque token per live (or reserved) worker
        # Replaced by every start() and close(); a start() publishes its outcome
        # only while its own marker is still the current one.
        self._lifecycle = object()
        # True while one start() is between reserving tokens and publishing.
        self._starting = False
        add_listener = getattr(self._capacity, "add_listener", None)
        if callable(add_listener):
            add_listener(self._capacity_changed)

    def _capacity_changed(self) -> None:
        """A shared capacity may have freed a slot or changed its selection:
        idle workers re-check. Called outside the capacity's own lock."""
        with self._lock:
            self._wake.notify_all()

    # --- lifecycle ---------------------------------------------------------------

    def start(self) -> bool:
        """Open intake with ``workers`` live worker threads.

        True only if this call started every missing worker and no ``close``
        superseded it meanwhile. False when a worker could not be started
        (intake closed, one fixed log line) or a ``close`` landed while workers
        were being started (intake stays closed). Also False -- at once,
        without waiting and changing nothing -- while another ``start`` is
        still starting its workers.
        """
        with self._lock:
            if self._starting:
                return False
            self._starting = True
            attempt = self._lifecycle = object()
            self._stopping = False
            # Reserved under the lock, so starts cannot exceed workers.
            tokens = [object() for _ in range(self._settings.workers - len(self._workers))]
            self._workers.update(tokens)
        started = 0
        try:
            for token in tokens:
                worker = self._thread_factory(
                    target=self._worker_loop, args=(token,), name="insights-worker", daemon=True
                )
                worker.start()
                started += 1
        except Exception:
            # Never bound or logged: only the fact of a failed start is kept.
            pass
        finally:
            failed = started < len(tokens)
            with self._lock:
                self._workers.difference_update(tokens[started:])
                self._starting = False
                current = self._lifecycle is attempt
                if current and failed:
                    self._accepting = False
                    self._stopping = True
                    self._wake.notify_all()
                elif current:
                    self._accepting = True
                # Superseded: the close() that replaced the marker already closed
                # intake and set stopping, so the workers this call started exit
                # unless a later start() reopens the queue first (then they serve).
        if failed:
            self._log("[Insights] Queue worker could not start; background insights unavailable")
        return current and not failed

    def close(self) -> None:
        """Stop intake and signal idle workers. Never joins or waits. A
        ``start`` still starting workers is superseded and cannot reopen
        intake when it finishes."""
        with self._lock:
            self._lifecycle = object()
            self._accepting = False
            self._stopping = True
            self._wake.notify_all()

    # --- admission -----------------------------------------------------------------

    def enqueue(self, document_id: object) -> EnqueueOutcome:
        doc = require_document_id(document_id)
        with self._lock:
            outcome = self._admit(doc)
            if outcome is EnqueueOutcome.ACCEPTED:
                self._wake.notify()
        return outcome

    def enqueue_batch(self, document_ids: Iterable[object]) -> Optional[BatchAdmission]:
        """Admit ids in order, atomically. None, with nothing admitted, when the
        queue is not accepting."""
        docs = [require_document_id(value) for value in document_ids]
        accepted = already_pending = rejected = 0
        with self._lock:
            if not self._accepting:
                return None
            for doc in docs:
                outcome = self._admit(doc)
                if outcome is EnqueueOutcome.ACCEPTED:
                    accepted += 1
                elif outcome is EnqueueOutcome.DUPLICATE:
                    already_pending += 1
                else:
                    rejected += 1
            if accepted:
                self._wake.notify_all()
        return BatchAdmission(accepted=accepted, already_pending=already_pending, capacity_rejected=rejected)

    def _admit(self, doc: int) -> EnqueueOutcome:
        # Caller holds self._lock.
        if not self._accepting:
            return EnqueueOutcome.UNAVAILABLE
        if doc in self._keys:
            return EnqueueOutcome.DUPLICATE
        if len(self._queued) >= self._settings.max_queued:
            return EnqueueOutcome.FULL
        self._queued.append(doc)
        self._keys[doc] = _QUEUED
        return EnqueueOutcome.ACCEPTED

    def snapshot(self) -> QueueSnapshot:
        with self._lock:
            states = list(self._keys.values())
            accepting, queued, workers = self._accepting, len(self._queued), len(self._workers)
        capacity = self._capacity.snapshot()
        return QueueSnapshot(
            accepting=accepting,
            queued=queued,
            running=states.count(_RUNNING),
            unresolved=states.count(_UNRESOLVED),
            workers=workers,
            free_capacity=capacity.free,
        )

    # --- workers -------------------------------------------------------------------

    def _next_job(self, token):
        """(document id, call) once both an id and a slot are available, or
        None when this worker must exit. Waits on the condition, never spins."""
        with self._lock:
            while True:
                if self._stopping:
                    self._workers.discard(token)
                    return None
                if self._queued:
                    # None when no slot is free (or a shared ledger is paused,
                    # unowned or serving a waiting answer first): wait.
                    call = self._capacity.try_acquire()
                    if call is not None:
                        doc = self._queued.popleft()
                        self._keys[doc] = _RUNNING
                        return doc, call
                self._wake.wait()

    def _worker_loop(self, token) -> None:
        try:
            while True:
                job = self._next_job(token)
                if job is None:
                    return
                doc, call = job
                try:
                    self._run_job(doc, call)
                except Exception:
                    # The job owns its own fixed-category logging; nothing about
                    # the document or the error is recorded here.
                    pass
                finally:
                    state = self._capacity.finish(call)
                    with self._lock:
                        if state is CallState.UNKNOWN:
                            self._keys[doc] = _UNRESOLVED
                            unresolved = sum(1 for value in self._keys.values() if value == _UNRESOLVED)
                        else:
                            del self._keys[doc]
                        self._wake.notify_all()
                if state is CallState.UNKNOWN:
                    self._log(f"[Insights] Background model call unresolved; capacity held: {unresolved}")
                if self._on_finished is not None:
                    try:
                        self._on_finished(doc, state)
                    except Exception:
                        pass
        finally:
            with self._lock:
                self._workers.discard(token)
