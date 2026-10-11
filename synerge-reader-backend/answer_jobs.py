"""One owned answer job per admitted /ask, and its bounded delivery.

Ownership. ``acquire_answer`` admits an answer: it waits on the event loop --
holding no thread, no thread-pool token and no lock -- for at most the agreed
``answer_wait_s``, then the caller gets a ticket or a fixed refusal. The
ticket then belongs to exactly one ``AnswerJob`` for the answer's whole life:
evidence, generation, every verifier call and persistence run on ONE
dedicated job thread under that one ticket, and the job releases it exactly
once, by what its model calls showed (generation_admission). The response's
own ASGI ``__call__`` starts the job; an answer whose response is never
started is reclaimed by a start lease, and a response called after that lease
expired answers 503 without starting anything.

Delivery. The job thread puts complete NDJSON lines into a per-answer
``AnswerChannel`` that never blocks the job. Its byte bound counts the ENCODED
bytes of every line -- evidence, deltas, verification and control events --
and keeps a reserve for the terminal control events (error, entry_id, done),
so protocol order and a committed ``entry_id``/``done`` are never squeezed
out. A line that would exceed the bound is never silently truncated: delivery
is abandoned (``delivery_overflow``) and the client is told so explicitly. The
sender is the response itself: it awaits lines on the event loop and bounds
every ``send`` -- the response start, every body, and both sends of the
not-started 503 -- by ``send_stall_s``; a stalled or disconnected client
abandons delivery. Client abandonment is reported as exactly that, never as a
server cancellation.

After known abandonment the job keeps reading the model stream it already
sent to its terminal record (so the call can be observed as finished and its
slot returned), dispatches no new verifier call and begins no answer INSERT.
A transaction that already began is completed and kept.

Time. The ticket carries a whole-job wall-clock budget from its grant. Per-read
transport timeouts are clamped to what is left, the job checks the budget
between phases and between stream chunks, and the sender independently ends
the response with a fixed ``answer_timeout`` error once the budget (plus a
short grace, longer while persistence is in progress) has passed -- even if
the job thread is blocked before the first byte or inside a read. A timeout,
a disconnect or a closed connection is never treated as proof that Ollama
stopped: an unobserved call keeps its capacity as UNKNOWN.
"""

from __future__ import annotations

import asyncio
import collections
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from starlette.responses import Response

from generation_admission import (
    AdmissionRefusal,
    AdmissionRefused,
    AnswerWaiter,
    GenerationLedger,
    GenerationTicket,
    generation_scope,
)

# Fixed client-visible errors this module owns. Codes and messages are
# constants; nothing from an exception ever reaches them.
ANSWER_TIMEOUT_ERROR = {
    "code": "answer_timeout",
    "message": "The answer took longer than its time limit and was stopped.",
}
DELIVERY_INTERRUPTED_ERROR = {
    "code": "answer_delivery_interrupted",
    "message": "The answer could not be delivered in time.",
}
INTERNAL_ERROR = {
    "code": "internal_error",
    "message": "Something went wrong while answering. Please try again.",
}
NOT_DISPATCHED_ERROR = {
    "code": "generation_not_dispatched",
    "message": "The answer could not be started safely. Please retry shortly.",
}
NOT_STARTED_DETAIL = "The answer could not be started. Please retry shortly."

# Why delivery ended early. Client abandonment is never server cancellation.
CLIENT_DISCONNECTED = "client_disconnected"
CLIENT_STALLED = "client_stalled"
DELIVERY_OVERFLOW = "delivery_overflow"
DELIVERY_CANCELLED = "delivery_cancelled"

_CONTROL_PREFIXES = ('{"type": "error"', '{"type": "entry_id"', '{"type": "done"')
_DONE_PREFIX = '{"type": "done"'


def _line(event_type: str, **payload) -> str:
    return json.dumps({"type": event_type, **payload}, ensure_ascii=True) + "\n"


def _terminal_pair(error: dict) -> list:
    return [_line("error", **error), _line("done", ok=False)]


@dataclass(frozen=True)
class DeliverySettings:
    """Provisional, unmeasured delivery bounds.

    ``max_buffered_bytes`` bounds the encoded bytes waiting for the client;
    ``reserved_bytes`` of it are kept for terminal control events. A whole
    answer at the production limits (8 evidence items, 1200 output tokens) is
    far below the default, so a full buffer means the client stopped reading.
    """

    max_buffered_bytes: int = 256 * 1024
    reserved_bytes: int = 2048
    send_stall_s: float = 10.0
    deadline_grace_s: float = 2.0
    persist_grace_s: float = 15.0

    def __post_init__(self) -> None:
        for name, low, high in (("max_buffered_bytes", 4096, 64 * 1024 * 1024), ("reserved_bytes", 512, 65536)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if self.reserved_bytes * 2 > self.max_buffered_bytes:
            raise ValueError("reserved_bytes must leave room for ordinary events")
        for name, low, high in (("send_stall_s", 0.01, 600.0), ("deadline_grace_s", 0.0, 60.0),
                                ("persist_grace_s", 0.0, 120.0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or not low <= value <= high:
                raise ValueError(f"{name} must be a number in [{low}, {high}]")


async def acquire_answer(ledger: GenerationLedger) -> GenerationTicket:
    """An ADMITTED answer ticket, or AdmissionRefused, within the ledger's
    ``answer_wait_s``. The wait holds no thread. A cancelled waiter returns
    any slot it was granted; a grant that lands at the deadline wins."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def wake() -> None:
        loop.call_soon_threadsafe(_set_future, future)

    admitted = ledger.enqueue_answer(wake)
    if isinstance(admitted, GenerationTicket):
        return admitted
    waiter: AnswerWaiter = admitted
    try:
        await asyncio.wait_for(asyncio.shield(future), timeout=ledger.settings.answer_wait_s)
    except asyncio.TimeoutError:
        pass
    except BaseException:
        ticket = ledger.settle_answer_waiter(waiter)
        if ticket is not None:
            ledger.release_unstarted(ticket)
        raise
    ticket = ledger.settle_answer_waiter(waiter)
    if ticket is None:
        raise AdmissionRefused(waiter.refusal or AdmissionRefusal.BUSY)
    return ticket


def _set_future(future) -> None:
    if not future.done():
        future.set_result(None)


class AnswerChannel:
    """A bounded, thread-safe queue of complete encoded NDJSON lines from one
    job thread to one sender. ``put`` never blocks."""

    def __init__(self, settings: DeliverySettings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._items = collections.deque()  # (bytes, is_done)
        self._buffered = 0
        self._done_queued = False
        self._abandoned: Optional[str] = None
        self._wake: Optional[Callable[[], None]] = None
        self.peak_buffered = 0

    def attach(self, wake: Callable[[], None]) -> None:
        with self._lock:
            self._wake = wake
        self._notify()

    def _notify(self) -> None:
        wake = self._wake
        if wake is not None:
            try:
                wake()
            except Exception:
                pass

    def _append_locked(self, line: str) -> None:
        data = line.encode("ascii")
        is_done = line.startswith(_DONE_PREFIX)
        self._items.append((data, is_done))
        self._buffered += len(data)
        self.peak_buffered = max(self.peak_buffered, self._buffered)
        if is_done:
            self._done_queued = True

    def put(self, line: str) -> bool:
        """Queue one complete line; False if it was not queued (delivery is
        over, or the line would exceed the bound -- which abandons delivery
        and queues the explicit interruption pair instead)."""
        if not isinstance(line, str) or not line.endswith("\n"):
            raise ValueError("a channel line is one complete NDJSON line")
        size = len(line.encode("ascii"))
        control = line.startswith(_CONTROL_PREFIXES)
        with self._lock:
            if self._done_queued or self._abandoned is not None:
                return False
            limit = self._settings.max_buffered_bytes - (0 if control else self._settings.reserved_bytes)
            if self._buffered + size > limit:
                self._abandoned = DELIVERY_OVERFLOW
                for terminal in _terminal_pair(DELIVERY_INTERRUPTED_ERROR):
                    self._append_locked(terminal)
                queued = False
            else:
                self._append_locked(line)
                queued = True
        self._notify()
        return queued

    def terminate(self, error: dict) -> bool:
        """End delivery with ``error`` + ``done(ok=false)`` unless a done line
        is already queued (then the job's own terminal is delivered)."""
        with self._lock:
            if self._done_queued:
                return False
            for terminal in _terminal_pair(error):
                self._append_locked(terminal)
        self._notify()
        return True

    def abandon(self, reason: str) -> None:
        """Nothing more will be delivered: drop what is queued."""
        with self._lock:
            if self._abandoned is None:
                self._abandoned = reason
            if reason != DELIVERY_OVERFLOW:
                self._items.clear()
                self._buffered = 0
        self._notify()

    def take(self):
        """The next (bytes, is_done), or None."""
        with self._lock:
            if not self._items:
                return None
            data, is_done = self._items.popleft()
            self._buffered -= len(data)
            return data, is_done

    @property
    def abandoned(self) -> Optional[str]:
        with self._lock:
            return self._abandoned

    @property
    def done_queued(self) -> bool:
        with self._lock:
            return self._done_queued


class AnswerJob:
    """One admitted answer, from its ticket's handoff to its last event."""

    def __init__(
        self,
        *,
        ledger: GenerationLedger,
        ticket: GenerationTicket,
        body: Callable[["AnswerJob"], Iterator[str]],
        settings: DeliverySettings,
        clock: Callable[[], float] = time.monotonic,
        thread_factory: Callable[..., object] = threading.Thread,
        log: Callable[[str], None] = print,
    ) -> None:
        self._ledger = ledger
        self.ticket = ticket
        self._body = body
        self.settings = settings
        self._clock = clock
        self._thread_factory = thread_factory
        self._log = log
        self.channel = AnswerChannel(settings)
        self._lock = threading.Lock()
        self._abandoned: Optional[str] = None
        self._persisting = False
        self._lease = None
        self.finished = threading.Event()

    # --- handoff (event loop) ------------------------------------------------------

    def arm_start_lease(self, loop, seconds: float) -> None:
        """Reclaim the ticket if the response is never started."""
        self._lease = loop.call_later(seconds, self._lease_expired)

    def _lease_expired(self) -> None:
        if self._ledger.release_unstarted(self.ticket):
            self._log("[Ask] admitted answer was never started; slot returned")

    def start(self) -> bool:
        """ADMITTED -> STARTED and start the job thread; False (slot returned,
        nothing run) when the lease already expired or the thread cannot
        start."""
        lease, self._lease = self._lease, None
        if lease is not None:
            lease.cancel()
        if not self._ledger.start(self.ticket):
            return False
        try:
            thread = self._thread_factory(target=self._run, name="answer-job", daemon=True)
            thread.start()
        except Exception:
            # The thread never ran, so nothing was dispatched: NEVER_DISPATCHED.
            self._ledger.release(self.ticket)
            self.finished.set()
            self._log("[Ask] answer job thread could not start; slot returned")
            return False
        return True

    def release_if_unstarted(self) -> None:
        lease, self._lease = self._lease, None
        if lease is not None:
            lease.cancel()
        self._ledger.release_unstarted(self.ticket)

    # --- state shared by the job and the sender --------------------------------------

    def abandon(self, reason: str) -> None:
        with self._lock:
            if self._abandoned is None:
                self._abandoned = reason
        self.channel.abandon(reason)

    @property
    def abandoned(self) -> bool:
        with self._lock:
            if self._abandoned is not None:
                return True
        return self.channel.abandoned is not None

    @property
    def abandon_reason(self) -> Optional[str]:
        with self._lock:
            reason = self._abandoned
        return reason or self.channel.abandoned

    @property
    def expired(self) -> bool:
        return self.ticket.expired

    def begin_persistence(self) -> bool:
        """The commit point decision, made once: False -- begin no INSERT --
        after known abandonment or past the budget."""
        with self._lock:
            if self._abandoned is not None or self.channel.abandoned is not None or self.ticket.expired:
                return False
            self._persisting = True
            return True

    def observed_stream(self, stream) -> bool:
        return self.ticket.observed_stream(stream)

    def delivery_remaining(self) -> float:
        with self._lock:
            grace = self.settings.deadline_grace_s + (self.settings.persist_grace_s if self._persisting else 0.0)
        return self.ticket.deadline + grace - self._clock()

    # --- the job thread ----------------------------------------------------------------

    def _run(self) -> None:
        try:
            with generation_scope(self.ticket, stopped=lambda: self.abandoned):
                for line in self._body(self):
                    self.channel.put(line)
        except BaseException:
            self._log("[Ask] answer job failed")
            self.channel.terminate(INTERNAL_ERROR)
        finally:
            try:
                self._ledger.release(self.ticket)
            except Exception:
                self._log("[Ask] answer job release failed")
            if not self.channel.done_queued and self.channel.abandoned is None:
                self.channel.terminate(INTERNAL_ERROR)
            self.finished.set()


class AdmittedAnswerResponse(Response):
    """The streamed /ask response for one AnswerJob. Speaks ASGI directly so
    every send can be bounded and a disconnect is noticed while waiting."""

    def __init__(self, job: AnswerJob, *, media_type: str, headers: Optional[dict] = None) -> None:
        self.job = job
        self.status_code = 200
        self.media_type = media_type
        self.background = None
        self.init_headers(headers)

    async def __call__(self, scope, receive, send) -> None:
        job = self.job
        try:
            if not job.start():
                await self._send_not_started(send, job)
                return
            loop = asyncio.get_running_loop()
            ready = asyncio.Event()

            def wake() -> None:
                try:
                    loop.call_soon_threadsafe(ready.set)
                except RuntimeError:
                    pass

            job.channel.attach(wake)
            start = {"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers}
            if not await self._bounded_send(send, start, job):
                return  # abandoned: the started job drains on its own and releases its ticket once
            disconnected = asyncio.ensure_future(self._listen(receive, job))
            try:
                await self._deliver(send, ready, disconnected)
            finally:
                disconnected.cancel()
        except BaseException:
            # The server stopped delivering (task cancelled or a failing
            # send): the job is told delivery is over and drains on its own.
            job.abandon(DELIVERY_CANCELLED)
            raise
        finally:
            job.release_if_unstarted()

    async def _send_not_started(self, send, job: AnswerJob) -> None:
        # Nothing ran and the slot is already back; only the reply is left,
        # and it is bounded like every other send.
        body = json.dumps({"detail": NOT_STARTED_DETAIL}).encode("ascii")
        start = {
            "type": "http.response.start",
            "status": 503,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                        (b"retry-after", b"2")],
        }
        if await self._bounded_send(send, start, job):
            await self._bounded_send(send, {"type": "http.response.body", "body": body, "more_body": False}, job)

    @staticmethod
    async def _listen(receive, job: AnswerJob) -> None:
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                job.abandon(CLIENT_DISCONNECTED)
                return

    async def _bounded_send(self, send, message, job: AnswerJob) -> bool:
        try:
            await asyncio.wait_for(send(message), timeout=job.settings.send_stall_s)
        except asyncio.TimeoutError:
            job.abandon(CLIENT_STALLED)
            return False
        except OSError:
            job.abandon(CLIENT_DISCONNECTED)
            return False
        return True

    async def _deliver(self, send, ready: asyncio.Event, disconnected) -> None:
        job = self.job
        while True:
            if disconnected.done() or (job.channel.abandoned and job.channel.abandoned != DELIVERY_OVERFLOW):
                return
            item = job.channel.take()
            if item is None:
                remaining = job.delivery_remaining()
                if remaining <= 0:
                    if job.channel.terminate(ANSWER_TIMEOUT_ERROR):
                        self._log_timeout()
                    continue
                ready.clear()
                item = job.channel.take()
                if item is None:
                    waiter = asyncio.ensure_future(ready.wait())
                    try:
                        await asyncio.wait({waiter, disconnected}, timeout=remaining,
                                           return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        waiter.cancel()
                    continue
            data, is_done = item
            if not await self._bounded_send(send, {"type": "http.response.body", "body": data, "more_body": True}, job):
                return
            if is_done:
                await self._bounded_send(send, {"type": "http.response.body", "body": b"", "more_body": False}, job)
                return

    def _log_timeout(self) -> None:
        print("[Ask] answer delivery ended at the job's time budget")
