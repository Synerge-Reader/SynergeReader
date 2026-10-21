"""Unit contracts for answer_jobs.py: admission waits, the owned answer job
and its bounded delivery.

No main import, database, model, network or subprocess. Each test runs a real
asyncio event loop (the guard's loopback socketpair) and drives
``AdmittedAnswerResponse`` with an in-memory ASGI ``send``/``receive``; job
bodies are fake generators that report fake model calls to the real ledger.
Every wait is bounded.

What these prove: an answer waits for a slot on the event loop for at most the
configured time, holding no thread, and a cancelled waiter returns a slot it
was granted; the channel bounds ENCODED bytes of every event, keeps room for
terminal control events, and replaces an overflowing line with an explicit
interruption instead of truncating; an answer whose response never starts is
reclaimed by its lease and a late start answers 503 without running anything;
a failed thread start returns the slot; a disconnect before the first byte
runs nothing, a disconnect or stall mid-stream lets the job drain its model
call to the terminal record while nothing more is delivered; every ASGI send
is bounded -- a stalled response start abandons a started answer as a client
stall (no further dispatch or INSERT, an unobserved call stays UNKNOWN), and
a stalled start or body of the not-started 503 ends at the same bound; the
sender ends the response at the whole-job budget even while the job thread is
blocked, without treating that as proof the model stopped; an idle stream
carries content-free heartbeat lines (never after ``done``, never extending
the budget); and each ticket is released exactly once.

What these do NOT prove: a real ASGI server's (uvicorn) disconnect or
back-pressure behaviour, or anything about Ollama.
"""

import asyncio
import json
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from answer_jobs import (  # noqa: E402
    ANSWER_TIMEOUT_ERROR,
    CLIENT_DISCONNECTED,
    CLIENT_STALLED,
    DELIVERY_CANCELLED,
    DELIVERY_INTERRUPTED_ERROR,
    DELIVERY_OVERFLOW,
    HEARTBEAT_LINE,
    AdmittedAnswerResponse,
    AnswerChannel,
    AnswerJob,
    DeliverySettings,
    acquire_answer,
)
from ask_stream import delta_event, done_event, entry_id_event, error_event, evidence_event  # noqa: E402
from generation_admission import (  # noqa: E402
    AdmissionRefusal,
    AdmissionRefused,
    AdmissionSettings,
    CallState,
    DispatchIntents,
    GenerationLedger,
    GenerationTicket,
    SharedSelection,
    current_dispatch_permitted,
)

_WAIT = 5.0
_TERMINAL = {"response": "x", "done": True, "done_reason": "stop"}


class _Intents(DispatchIntents):
    def __init__(self):
        self.events = []

    def record_dispatch(self, intent):
        self.events.append(("dispatch", intent.call_role))

    def record_outcome(self, intent, state):
        self.events.append(("outcome", state))


class _Snapshot:
    profile_id = "llama_control"
    selection_epoch = 1
    model_tag = "llama3.1:8b"


def _ledger(**settings):
    values = dict(total_slots=2, answer_slots=2, helper_slots=1, answer_waiters=2, answer_wait_s=0.2)
    values.update(settings)
    ledger = GenerationLedger(AdmissionSettings(**values), clock=time.monotonic)
    intents = _Intents()
    assert ledger.install_owner(intents=intents, owner_seq=1, selection=SharedSelection("llama_control", 1),
                                paused_by=None, recovered={})
    return ledger, intents


def _ticket(ledger):
    admitted = ledger.enqueue_answer(lambda: None)
    assert isinstance(admitted, GenerationTicket)
    admitted.bind_snapshot(_Snapshot())
    return admitted


class _Factory:
    def __init__(self, fail=False):
        self.created = []
        self.fail = fail

    def __call__(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("SECRET-DETAIL no threads")
        thread = threading.Thread(*args, **kwargs)
        self.created.append(thread)
        return thread


def _job(ledger, ticket, body, factory=None, **delivery):
    values = dict(send_stall_s=0.3, deadline_grace_s=0.05, persist_grace_s=0.2)
    values.update(delivery)
    return AnswerJob(ledger=ledger, ticket=ticket, body=body, settings=DeliverySettings(**values),
                     thread_factory=factory or _Factory(), log=lambda line: None)


class _Asgi:
    """An in-memory ASGI peer: records every message; ``disconnect`` (an event)
    makes receive() report http.disconnect; ``stall_after`` blocks every body
    send after that many bodies; ``stall_start`` blocks the response start."""

    def __init__(self, *, disconnect_now=False, stall_after=None, stall_start=False):
        self.messages = []
        self.disconnect = None
        self.disconnect_now = disconnect_now
        self.stall_after = stall_after
        self.stall_start = stall_start

    async def receive(self):
        if self.disconnect_now:
            return {"type": "http.disconnect"}
        while not self.disconnect.is_set():
            await asyncio.sleep(0.01)
        return {"type": "http.disconnect"}

    async def send(self, message):
        if self.stall_start and message["type"] == "http.response.start":
            await asyncio.sleep(3600)
        bodies = [m for m in self.messages if m["type"] == "http.response.body"]
        if self.stall_after is not None and message["type"] == "http.response.body" and len(bodies) >= self.stall_after:
            await asyncio.sleep(3600)
        self.messages.append(message)

    @property
    def status(self):
        return self.messages[0]["status"]

    @property
    def body(self):
        return b"".join(m.get("body", b"") for m in self.messages if m["type"] == "http.response.body")

    @property
    def events(self):
        return [json.loads(line) for line in self.body.decode().splitlines() if line.strip()]

    @property
    def completed(self):
        return any(m["type"] == "http.response.body" and not m.get("more_body") for m in self.messages)


def _serve(job, asgi, after_start=None):
    async def run():
        asgi.disconnect = asyncio.Event()
        response = AdmittedAnswerResponse(job, media_type="application/x-ndjson")
        task = asyncio.ensure_future(response({"type": "http"}, asgi.receive, asgi.send))
        if after_start is not None:
            await after_start(task, asgi)
        try:
            await asyncio.wait_for(task, _WAIT)
        except asyncio.CancelledError:
            pass

    asyncio.run(run())


def _count_releases(ledger):
    """One (path, state) entry per slot the ledger actually returned."""
    returned = []
    release, release_unstarted = ledger.release, ledger.release_unstarted

    def counted_release(ticket):
        state = release(ticket)
        returned.append(("release", state))
        return state

    def counted_release_unstarted(ticket):
        done = release_unstarted(ticket)
        if done:
            returned.append(("release_unstarted", CallState.NEVER_DISPATCHED))
        return done

    ledger.release, ledger.release_unstarted = counted_release, counted_release_unstarted
    return returned


# --- admission waits ----------------------------------------------------------------


def test_an_answer_waits_on_the_loop_holding_no_thread_and_takes_a_freed_slot():
    ledger, _ = _ledger(answer_wait_s=2.0)
    holders = [_ticket(ledger), _ticket(ledger)]

    async def run():
        threads_before = threading.active_count()
        task = asyncio.ensure_future(acquire_answer(ledger))
        await asyncio.sleep(0.1)
        assert not task.done() and ledger.snapshot().answer_waiters == 1
        assert threading.active_count() == threads_before, "a waiting answer holds no thread"
        threading.Thread(target=ledger.release_unstarted, args=(holders[0],)).start()
        return await asyncio.wait_for(task, _WAIT)

    granted = asyncio.run(run())
    assert isinstance(granted, GenerationTicket)
    assert ledger.snapshot().answers_running == 2
    for ticket in (granted, holders[1]):
        ledger.release_unstarted(ticket)


def test_the_wait_ends_busy_at_its_bound_and_leaves_no_waiter_behind():
    ledger, _ = _ledger(answer_wait_s=0.1)
    holders = [_ticket(ledger), _ticket(ledger)]

    async def run():
        started = time.monotonic()
        with pytest.raises(AdmissionRefused) as refused:
            await acquire_answer(ledger)
        return refused.value.reason, time.monotonic() - started

    reason, waited = asyncio.run(run())
    assert reason is AdmissionRefusal.BUSY and 0.09 <= waited < 1.0
    assert ledger.snapshot().answer_waiters == 0
    ledger.release_unstarted(holders[0])
    assert ledger.snapshot().answers_running == 1, "nobody collects the freed slot late"
    ledger.release_unstarted(holders[1])


def test_a_cancelled_waiter_returns_a_slot_it_was_granted():
    ledger, _ = _ledger(answer_wait_s=2.0)
    holders = [_ticket(ledger), _ticket(ledger)]

    async def run():
        task = asyncio.ensure_future(acquire_answer(ledger))
        await asyncio.sleep(0.05)
        ledger.release_unstarted(holders[0])  # granted to the waiter...
        task.cancel()  # ...which is cancelled before it collects the grant
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    snap = ledger.snapshot()
    assert (snap.answers_running, snap.answer_waiters) == (1, 0), "the granted slot came back"
    ledger.release_unstarted(holders[1])


# --- the bounded channel ------------------------------------------------------------


def test_the_channel_bounds_encoded_bytes_and_keeps_room_for_terminal_events():
    settings = DeliverySettings(max_buffered_bytes=4096, reserved_bytes=1024)
    channel = AnswerChannel(settings)
    evidence = evidence_event([{"citation_id": "C1", "excerpt": "é" * 400}], mode="hybrid_retrieval")
    assert len(evidence.encode()) > 400 * 2, "non-ASCII text is counted as its escaped bytes"
    assert channel.put(evidence)
    while channel.put(delta_event("x" * 40)):
        pass
    assert channel.abandoned == DELIVERY_OVERFLOW
    assert channel.peak_buffered <= settings.max_buffered_bytes + 512
    lines = []
    while (item := channel.take()) is not None:
        lines.append(item[0].decode())
    assert all(line.endswith("\n") for line in lines), "every queued line is complete: nothing is truncated"
    assert [json.loads(l)["type"] for l in lines[-2:]] == ["error", "done"]
    assert json.loads(lines[-2]) == {"type": "error", **DELIVERY_INTERRUPTED_ERROR}
    assert not channel.put(entry_id_event(1)), "nothing is accepted after the interruption"


def test_terminal_control_events_use_the_reserve_and_keep_their_order():
    settings = DeliverySettings(max_buffered_bytes=4096, reserved_bytes=1024)
    delta = delta_event("y" * 20)

    def filled():
        channel = AnswerChannel(settings)
        while channel._buffered + len(delta) <= settings.max_buffered_bytes - settings.reserved_bytes:
            assert channel.put(delta)
        return channel

    ordinary = filled()
    assert not ordinary.put(delta), "an ordinary event may not use the reserve"
    assert ordinary.abandoned == DELIVERY_OVERFLOW

    committed = filled()
    assert committed.put(entry_id_event(4242)) and committed.put(done_event(ok=True)), "committed entry_id/done fit"
    assert committed.abandoned is None and committed.done_queued
    items = []
    while (item := committed.take()) is not None:
        items.append(item)
    assert [json.loads(data)["type"] for data, _ in items[-2:]] == ["entry_id", "done"] and items[-1][1] is True


# --- handoff and the job thread -----------------------------------------------------------


def _body_with_one_call(lines_before=(), *, terminal=True, gate=None):
    def body(job):
        yield delta_event("")
        for line in lines_before:
            yield line
        if job.abandoned:
            return
        job.ticket.dispatching("answer")
        if gate is not None:
            assert gate.wait(_WAIT)
        if terminal:
            job.ticket.observed(_TERMINAL)
        yield delta_event("The answer.")
        if not job.begin_persistence():
            return
        yield entry_id_event(7)
        yield done_event(ok=True)

    return body


def test_a_streamed_answer_is_delivered_in_order_and_its_ticket_released_once():
    ledger, intents = _ledger()
    ticket = _ticket(ledger)
    job = _job(ledger, ticket, _body_with_one_call([evidence_event([], mode="none")]))
    asgi = _Asgi()
    _serve(job, asgi)
    assert job.finished.wait(_WAIT)
    assert asgi.status == 200 and asgi.completed
    assert [e["type"] for e in asgi.events] == ["delta", "evidence", "delta", "entry_id", "done"]
    assert intents.events == [("dispatch", "answer"), ("outcome", CallState.OBSERVED_TERMINAL)]
    assert ticket.released and ledger.snapshot().answers_running == 0
    with pytest.raises(RuntimeError):
        ledger.release(ticket)


def test_an_answer_never_started_is_reclaimed_and_a_late_start_runs_nothing():
    ledger, _ = _ledger()
    ticket = _ticket(ledger)
    factory = _Factory()
    job = _job(ledger, ticket, _body_with_one_call(), factory)

    async def lease_then_late_start():
        job.arm_start_lease(asyncio.get_running_loop(), 0.05)
        await asyncio.sleep(0.2)
        assert ledger.snapshot().answers_running == 0, "the lease returned the slot"
        asgi = _Asgi(disconnect_now=True)
        await AdmittedAnswerResponse(job, media_type="application/x-ndjson")({"type": "http"}, asgi.receive, asgi.send)
        return asgi

    asgi = asyncio.run(lease_then_late_start())
    assert asgi.status == 503 and json.loads(asgi.body)["detail"]
    assert factory.created == [], "no job thread was started for an expired admission"


def test_a_failed_thread_start_returns_the_slot_and_answers_503():
    ledger, intents = _ledger()
    ticket = _ticket(ledger)
    job = _job(ledger, ticket, _body_with_one_call(), _Factory(fail=True))
    asgi = _Asgi()
    _serve(job, asgi)
    assert asgi.status == 503 and b"SECRET" not in asgi.body
    assert intents.events == [] and ticket.released and ledger.snapshot().answers_running == 0


def test_a_disconnect_before_the_first_byte_runs_nothing():
    ledger, intents = _ledger()
    ticket = _ticket(ledger)
    gate = threading.Event()

    def body(job):
        assert gate.wait(_WAIT)  # the client goes away before any work starts
        yield delta_event("")
        if job.abandoned:
            return
        raise AssertionError("no evidence or model work after the disconnect")

    job = _job(ledger, ticket, body)

    async def disconnect(task, asgi):
        asgi.disconnect.set()
        await asyncio.sleep(0.1)
        gate.set()

    asgi = _Asgi()
    _serve(job, asgi, after_start=disconnect)
    assert job.finished.wait(_WAIT)
    assert job.abandon_reason == CLIENT_DISCONNECTED
    assert intents.events == [] and ticket.state is CallState.NEVER_DISPATCHED
    assert ledger.snapshot().answers_running == 0 and not asgi.completed


def test_a_mid_stream_disconnect_drains_the_call_and_persists_nothing():
    ledger, intents = _ledger()
    ticket = _ticket(ledger)
    gate = threading.Event()
    persisted = []

    def body(job):
        yield delta_event("")
        job.ticket.dispatching("answer")
        yield delta_event("partial ")
        assert gate.wait(_WAIT)  # the model keeps generating after the client left
        yield delta_event("rest")
        job.ticket.observed(_TERMINAL)  # drained to the terminal record
        verified = current_dispatch_permitted()
        persisted.append((job.begin_persistence(), verified))

    job = _job(ledger, ticket, body)

    async def disconnect(task, asgi):
        while not any(m["type"] == "http.response.body" and b"partial" in m["body"] for m in asgi.messages):
            await asyncio.sleep(0.01)
        asgi.disconnect.set()
        await asyncio.sleep(0.1)
        gate.set()

    asgi = _Asgi()
    _serve(job, asgi, after_start=disconnect)
    assert job.finished.wait(_WAIT)
    assert job.abandon_reason == CLIENT_DISCONNECTED
    assert persisted == [(False, False)], "no INSERT is begun and no verifier call may be dispatched"
    assert b"rest" not in asgi.body and not asgi.completed
    assert ticket.state is CallState.OBSERVED_TERMINAL and ledger.snapshot().answers_running == 0
    assert intents.events[-1] == ("outcome", CallState.OBSERVED_TERMINAL)


def test_a_stalled_client_is_abandoned_at_the_send_bound_while_the_job_continues():
    ledger, _ = _ledger()
    ticket = _ticket(ledger)

    def body(job):
        yield delta_event("")
        job.ticket.dispatching("answer")
        job.ticket.observed(_TERMINAL)
        for _ in range(3):
            yield delta_event("chunk")

    job = _job(ledger, ticket, body, send_stall_s=0.2)
    asgi = _Asgi(stall_after=1)
    started = time.monotonic()
    _serve(job, asgi)
    assert time.monotonic() - started < 2.0, "the stall bound, not the client, ended delivery"
    assert job.finished.wait(_WAIT)
    assert job.abandon_reason == CLIENT_STALLED and not asgi.completed
    assert ticket.state is CallState.OBSERVED_TERMINAL and ticket.released


@pytest.mark.parametrize("observed", [True, False], ids=["drained-to-terminal", "never-observed"])
def test_a_stalled_response_start_abandons_a_started_answer_at_the_send_bound(observed):
    ledger, intents = _ledger()
    ticket = _ticket(ledger)
    returned = _count_releases(ledger)
    gate = threading.Event()
    after_abandonment = []

    def body(job):
        yield delta_event("")
        job.ticket.dispatching("answer")
        assert gate.wait(_WAIT)  # the model keeps generating while the headers are stuck
        if observed:
            job.ticket.observed(_TERMINAL)
        after_abandonment.append((job.abandon_reason, current_dispatch_permitted(), job.begin_persistence()))
        yield delta_event("The answer.")

    job = _job(ledger, ticket, body, send_stall_s=0.2)

    async def release_the_model_once_delivery_ended(task, asgi):
        await asyncio.wait({task}, timeout=_WAIT)
        gate.set()

    asgi = _Asgi(stall_start=True)
    started = time.monotonic()
    _serve(job, asgi, after_start=release_the_model_once_delivery_ended)
    assert time.monotonic() - started < 2.0, "the send bound, not the client, ended delivery"
    assert job.finished.wait(_WAIT)
    assert asgi.messages == [], "the response start never completed, so nothing was delivered"
    assert after_abandonment == [(CLIENT_STALLED, False, False)], "no verifier dispatch and no INSERT after it"
    state = CallState.OBSERVED_TERMINAL if observed else CallState.UNKNOWN
    assert ticket.state is state and returned == [("release", state)], "released once, by what the call showed"
    assert intents.events == [("dispatch", "answer"), ("outcome", state)]
    snap = ledger.snapshot()
    assert snap.answers_running == 0 and snap.unresolved_answers == (0 if observed else 1), "UNKNOWN is never cleared"


@pytest.mark.parametrize("stalled", ["start", "body"], ids=["stalled-start", "stalled-body"])
@pytest.mark.parametrize("why", ["lease-expired", "thread-start-failed"])
def test_both_sends_of_the_not_started_503_are_bounded(why, stalled):
    ledger, intents = _ledger()
    ticket = _ticket(ledger)
    returned = _count_releases(ledger)
    factory = _Factory(fail=why == "thread-start-failed")
    job = _job(ledger, ticket, _body_with_one_call(), factory, send_stall_s=0.2)
    asgi = _Asgi(stall_start=stalled == "start", stall_after=0 if stalled == "body" else None)

    async def respond():
        if why == "lease-expired":
            job.arm_start_lease(asyncio.get_running_loop(), 0.05)
            await asyncio.sleep(0.2)
            assert returned == [("release_unstarted", CallState.NEVER_DISPATCHED)], "the lease returned the slot"
        started = time.monotonic()
        response = AdmittedAnswerResponse(job, media_type="application/x-ndjson")
        await asyncio.wait_for(response({"type": "http"}, asgi.receive, asgi.send), _WAIT)
        return time.monotonic() - started

    elapsed = asyncio.run(respond())
    assert elapsed < 2.0, "the send bound, not the client, ended the 503"
    assert job.abandon_reason == CLIENT_STALLED
    if stalled == "start":
        assert asgi.messages == []
    else:
        assert [m["type"] for m in asgi.messages] == ["http.response.start"] and asgi.status == 503
    assert factory.created == [] and intents.events == [], "nothing ran and nothing was dispatched"
    assert [state for _, state in returned] == [CallState.NEVER_DISPATCHED], "the slot came back exactly once"
    assert ticket.released and ledger.snapshot().answers_running == 0


def test_the_sender_ends_the_answer_at_the_job_budget_while_the_job_is_blocked():
    ledger, intents = _ledger(answer_job_budget_s=1.0)
    ticket = _ticket(ledger)
    gate = threading.Event()
    job = _job(ledger, ticket, _body_with_one_call(terminal=False, gate=gate))
    asgi = _Asgi()
    started = time.monotonic()
    _serve(job, asgi)
    elapsed = time.monotonic() - started
    assert 0.9 <= elapsed < 2.5, "ended at the budget plus its grace, not when the job finished"
    assert asgi.completed and asgi.events[-2:] == [{"type": "error", **ANSWER_TIMEOUT_ERROR},
                                                   {"type": "done", "ok": False}]
    gate.set()  # the blocked read returns later without the terminal record
    assert job.finished.wait(_WAIT)
    assert ticket.state is CallState.UNKNOWN, "a timeout is not proof the model stopped: the slot stays held"
    assert ledger.snapshot().unresolved_answers == 1
    assert intents.events[-1] == ("outcome", CallState.UNKNOWN)


def test_a_server_cancelled_delivery_is_not_reported_as_a_client_abandonment():
    ledger, _ = _ledger()
    ticket = _ticket(ledger)
    gate = threading.Event()
    job = _job(ledger, ticket, _body_with_one_call(gate=gate))

    async def cancel(task, asgi):
        await asyncio.sleep(0.1)
        task.cancel()
        await asyncio.sleep(0.05)
        gate.set()

    _serve(job, _Asgi(), after_start=cancel)
    assert job.finished.wait(_WAIT)
    assert job.abandon_reason == DELIVERY_CANCELLED
    assert ticket.state is CallState.OBSERVED_TERMINAL and ticket.released


def test_persistence_is_decided_once_and_never_begun_after_abandonment_or_expiry():
    ledger, _ = _ledger(answer_job_budget_s=1.0)
    job = _job(ledger, _ticket(ledger), _body_with_one_call())
    assert job.begin_persistence() is True, "the commit point is reached"
    job.abandon(CLIENT_DISCONNECTED)
    assert job.abandoned, "abandonment after the commit point does not undo a begun INSERT"
    other = _job(ledger, _ticket(ledger), _body_with_one_call())
    other.abandon(CLIENT_STALLED)
    assert other.begin_persistence() is False
    for item in (job, other):
        ledger.release_unstarted(item.ticket)


# --- heartbeats ---------------------------------------------------------------------


def test_an_idle_stream_carries_contentless_heartbeats_until_the_answer_resumes():
    ledger, intents = _ledger()
    ticket = _ticket(ledger)
    gate = threading.Event()
    job = _job(ledger, ticket, _body_with_one_call(gate=gate), heartbeat_s=0.1)
    asgi = _Asgi()

    async def release_after_idle(task, peer):
        await asyncio.sleep(0.45)  # the model call is silent, as during a cold load
        gate.set()

    _serve(job, asgi, after_start=release_after_idle)
    assert job.finished.wait(_WAIT) and asgi.completed
    bodies = [m["body"] for m in asgi.messages if m["type"] == "http.response.body"]
    beats = [i for i, body in enumerate(bodies) if body == HEARTBEAT_LINE]
    assert len(beats) >= 2, "one line per idle interval while the call was silent"
    types = [e["type"] for e in asgi.events]
    assert types[0] == "delta" and types[-3:] == ["delta", "entry_id", "done"]
    assert set(types[1:-3]) == {"heartbeat"}, "heartbeats only fill the silence; content order is unchanged"
    done_at = next(i for i, body in enumerate(bodies) if b'"done"' in body)
    assert bodies[done_at + 1:] == [b""], "nothing after the done line but the end of the body"
    assert intents.events == [("dispatch", "answer"), ("outcome", CallState.OBSERVED_TERMINAL)]
    assert ticket.released


def test_a_prompt_answer_carries_no_heartbeat():
    ledger, _ = _ledger()
    job = _job(ledger, _ticket(ledger), _body_with_one_call([evidence_event([], mode="none")]))
    asgi = _Asgi()
    _serve(job, asgi)
    assert job.finished.wait(_WAIT)
    assert HEARTBEAT_LINE not in [m.get("body") for m in asgi.messages]
    assert "heartbeat" not in [e["type"] for e in asgi.events]


def test_a_heartbeat_does_not_extend_the_job_budget():
    ledger, intents = _ledger(answer_job_budget_s=1.0)
    ticket = _ticket(ledger)
    gate = threading.Event()
    job = _job(ledger, ticket, _body_with_one_call(terminal=False, gate=gate), heartbeat_s=0.1)
    asgi = _Asgi()
    started = time.monotonic()
    _serve(job, asgi)
    assert time.monotonic() - started < 2.5
    assert "heartbeat" in [e["type"] for e in asgi.events]
    assert asgi.events[-2:] == [{"type": "error", **ANSWER_TIMEOUT_ERROR}, {"type": "done", "ok": False}]
    gate.set()
    assert job.finished.wait(_WAIT)
    assert ticket.state is CallState.UNKNOWN and ledger.snapshot().unresolved_answers == 1


@pytest.mark.parametrize("value", [0, 0.0, 91, float("nan"), float("inf"), True, "15"])
def test_the_heartbeat_interval_is_bounded_inside_the_proxy_idle_limit(value):
    with pytest.raises(ValueError):
        DeliverySettings(heartbeat_s=value)
