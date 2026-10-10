"""Runtime contracts for main.py import safety and FastAPI lifespan startup.

These tests execute ``main.py`` for real, but they never start an event loop.
``main.lifespan`` performs only synchronous work around its ``yield``, so both
halves can be driven directly (see ``_drive_without_event_loop``). That keeps
this file free of sockets: on Windows every asyncio event loop builds its
self-pipe with ``socket.socketpair()``, which is a loopback ``socket.connect``
that the Phase-B safety boundary blocks.

No database, Ollama, network, or subprocess access happens here: ``init_db``
and ``_initialize_application`` are replaced with recorders or a raiser in
every test that reaches them.
"""

import importlib
import importlib.util
import os
from pathlib import Path
import sys
import threading
import time
import uuid

import dotenv
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dbSetup
from dbSetup import VectorSchemaError
from generation_admission import CallState
from helper_jobs import EnqueueOutcome, HelperJobQueue, HelperQueueSettings


_MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"
_EMBEDDING_PROFILE_KEYS = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_QUERY_PREFIX",
    "EMBEDDING_DOCUMENT_PREFIX",
    "EMBEDDING_PROFILE_UNVERIFIED_ACK",
)

# Distinguishable from any real ``__aenter__`` result, including ``None``.
_NOT_ENTERED = object()

_OPERATOR_PREFIX = "Application startup blocked by vector schema validation:"


def _load_isolated_module(module_file: Path, real_module_name: str):
    """Execute a fresh module copy under a private, temporary name."""
    probe_name = f"_isolated_probe__{real_module_name}__{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(probe_name, module_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    previous = sys.modules.get(probe_name)
    sys.modules[probe_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(probe_name, None)
        else:
            sys.modules[probe_name] = previous
    return module


def _drive_without_event_loop(coro):
    """Run a non-suspending coroutine to completion with no event loop.

    A single ``coro.send(None)`` finishes any coroutine that never hands
    control back to a scheduler; completion arrives as ``StopIteration``,
    whose ``value`` is the coroutine's return value.

    This detects a *suspending* await -- one that actually yields to a loop --
    not every possible ``await``. An ``await`` on an already-resolved awaitable
    completes inside this same single step and is deliberately
    indistinguishable here from straight-line synchronous code.
    """
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    coro.close()
    raise AssertionError(
        "the lifespan suspended on a real await instead of running to "
        "completion, so it can no longer be driven without an event loop; "
        "this contract now requires a real event-loop test harness rather "
        "than this loop-free driver"
    )


@pytest.fixture
def neutral_main_environment(monkeypatch):
    """Keep runtime tests independent of an untracked backend .env file."""
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    for key in _EMBEDDING_PROFILE_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def main_module(neutral_main_environment):
    previous = sys.modules.pop("main", None)
    try:
        module = importlib.import_module("main")
        yield module
    finally:
        sys.modules.pop("main", None)
        if previous is not None:
            sys.modules["main"] = previous


def test_importing_main_does_not_initialize_database(
    monkeypatch,
    neutral_main_environment,
):
    calls = []

    def record_init_db(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(dbSetup, "init_db", record_init_db)

    _load_isolated_module(_MAIN_PATH, "main")

    assert calls == []


def test_configured_lifespan_initializes_once_before_yield(monkeypatch, main_module):
    assert main_module.app.router.lifespan_context is main_module.lifespan, (
        "FastAPI must have captured main.lifespan as the application lifespan "
        "handler; this test drives the object the framework will actually run "
        "at startup, not a lifespan function that merely exists in the module"
    )

    calls = []

    def record_initialize_application(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        main_module,
        "_initialize_application",
        record_initialize_application,
    )

    context = main_module.app.router.lifespan_context(main_module.app)

    _drive_without_event_loop(context.__aenter__())
    # Asserting here, while the lifespan sits suspended at its ``yield``, is
    # what proves the ordering: control reached the yield only after
    # initialization ran. The line-number comparison in
    # test_main_embedding_wiring.py is a source-readability check, not a
    # control-flow proof -- this assertion is that proof.
    assert len(calls) == 1, (
        "the lifespan must initialize the application exactly once before it "
        f"yields control to the server; recorded {len(calls)} call(s)"
    )

    exited = _drive_without_event_loop(context.__aexit__(None, None, None))
    assert not exited, "lifespan shutdown must not suppress exceptions"
    assert len(calls) == 1, (
        "lifespan shutdown must not initialize the application a second time; "
        f"recorded {len(calls)} call(s) in total"
    )


def test_vector_schema_error_is_visible_and_blocks_startup(
    monkeypatch,
    main_module,
    capsys,
):
    error = VectorSchemaError("document_chunks has an incompatible vector dimension")

    def fail_init_db(*args, **kwargs):
        raise error

    monkeypatch.setattr(main_module, "init_db", fail_init_db)

    context = main_module.app.router.lifespan_context(main_module.app)
    capsys.readouterr()

    entered = _NOT_ENTERED
    with pytest.raises(VectorSchemaError) as excinfo:
        entered = _drive_without_event_loop(context.__aenter__())

    assert excinfo.value is error, (
        "the original VectorSchemaError object must propagate out of startup "
        "unwrapped, so the operator sees the real schema failure"
    )
    assert entered is _NOT_ENTERED, (
        "a failed startup must not produce a lifespan value; the application "
        "must never reach the serving state"
    )

    # No __aexit__ after a failed entry: startup never completed, so there is
    # no established lifespan context to unwind.
    captured = capsys.readouterr()
    operator_lines = [
        line for line in captured.err.splitlines() if line.startswith(_OPERATOR_PREFIX)
    ]
    assert len(operator_lines) == 1, (
        "startup must emit exactly one operator-visible stderr line for a "
        f"vector schema failure; found {len(operator_lines)}"
    )
    assert operator_lines[0] == (
        "Application startup blocked by vector schema validation: "
        "document_chunks has an incompatible vector dimension"
    )


class _RecordingGeneration:
    """Stands in for main._GENERATION_RUNTIME: records start/stop, starts no
    thread and opens no database session."""

    def __init__(self, events, start_error=None):
        self._events = events
        self._start_error = start_error

    def start(self):
        self._events.append("generation.start")
        if self._start_error is not None:
            raise self._start_error

    def stop(self):
        self._events.append("generation.stop")


def test_initialize_application_uses_resolved_embedding_dimension(
    monkeypatch,
    main_module,
):
    calls = []

    def record_init_db(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(main_module, "init_db", record_init_db)
    # Initialization ends by starting generation ownership; the recorder keeps
    # this test from starting a real ownership thread.
    monkeypatch.setattr(main_module, "_GENERATION_RUNTIME", _RecordingGeneration([]))

    main_module._initialize_application()

    assert len(calls) == 1, (
        f"_initialize_application() must call init_db exactly once; recorded {len(calls)}"
    )
    args, kwargs = calls[0]
    assert args == (), "init_db must be called with no positional arguments"
    assert kwargs == {
        "expected_dimension": main_module._EMBEDDING_PROFILE.dimension
    }, (
        "init_db must receive exactly the dimension resolved by the embedding "
        "profile, so the schema check can never fall back to a hardcoded value"
    )


# --- generation ownership's lifecycle ---------------------------------------------------


def test_generation_starts_only_after_the_database_and_stops_after_the_queue(monkeypatch, main_module):
    events = []
    monkeypatch.setattr(main_module, "init_db", lambda **kwargs: events.append("init_db"))
    monkeypatch.setattr(main_module, "_GENERATION_RUNTIME", _RecordingGeneration(events))
    monkeypatch.setattr(main_module, "_INSIGHTS_QUEUE", _RecordingInsightsQueue(events))
    context = main_module.app.router.lifespan_context(main_module.app)

    _drive_without_event_loop(context.__aenter__())
    assert events == ["init_db", "generation.start", "queue.start"], (
        "ownership is sought only after the schema initialized, and never waited for"
    )
    _drive_without_event_loop(context.__aexit__(None, None, None))
    assert events[-2:] == ["queue.close", "generation.stop"], "shutdown fences generation and lets ownership go"


def test_a_failed_initialization_never_seeks_generation_ownership(monkeypatch, main_module):
    events = []
    monkeypatch.setattr(main_module, "_GENERATION_RUNTIME", _RecordingGeneration(events))

    def fail_init_db(*args, **kwargs):
        raise VectorSchemaError("document_chunks has an incompatible vector dimension")

    monkeypatch.setattr(main_module, "init_db", fail_init_db)
    with pytest.raises(VectorSchemaError):
        _drive_without_event_loop(main_module.app.router.lifespan_context(main_module.app).__aenter__())
    assert events == []


def test_a_generation_start_failure_never_stops_startup_or_leaks_detail(monkeypatch, main_module, capsys):
    events = []
    monkeypatch.setattr(main_module, "init_db", lambda **kwargs: None)
    monkeypatch.setattr(main_module, "_GENERATION_RUNTIME",
                        _RecordingGeneration(events, start_error=RuntimeError("SECRET-DETAIL")))
    monkeypatch.setattr(main_module, "_INSIGHTS_QUEUE", _RecordingInsightsQueue(events))
    context = main_module.app.router.lifespan_context(main_module.app)
    capsys.readouterr()

    _drive_without_event_loop(context.__aenter__())
    _drive_without_event_loop(context.__aexit__(None, None, None))
    assert events == ["generation.start", "queue.start", "queue.close", "generation.stop"]
    assert "SECRET-DETAIL" not in "".join(capsys.readouterr())


def test_importing_main_starts_no_generation_thread_and_opens_no_session(neutral_main_environment, monkeypatch):
    opened = []
    monkeypatch.setattr(dbSetup, "connect_to_postgres", lambda: opened.append(1))
    before = {thread.name for thread in threading.enumerate()}
    module = _load_isolated_module(_MAIN_PATH, "main")
    after = {thread.name for thread in threading.enumerate()}
    assert opened == [] and not {"generation-owner", "generation-coordinator"} & (after - before)
    assert module._GENERATION_LEDGER.snapshot().admission == "unavailable", "no ownership, no admission"


# --- the background-insights queue's lifecycle (B2a-1) -----------------------------


class _RecordingInsightsQueue:
    """Records the lifespan's queue calls; starts no thread."""

    def __init__(self, events, start_result=True, start_error=None):
        self._events = events
        self._start_result = start_result
        self._start_error = start_error

    def start(self):
        self._events.append("queue.start")
        if self._start_error is not None:
            raise self._start_error
        return self._start_result

    def close(self):
        self._events.append("queue.close")


def test_the_insights_queue_starts_only_after_initialization_and_closes_at_shutdown(monkeypatch, main_module):
    events = []
    monkeypatch.setattr(main_module, "_INSIGHTS_QUEUE", _RecordingInsightsQueue(events))
    monkeypatch.setattr(main_module, "_initialize_application", lambda: events.append("initialize"))
    context = main_module.app.router.lifespan_context(main_module.app)

    _drive_without_event_loop(context.__aenter__())
    assert events == ["initialize", "queue.start"], "workers start only after initialization"

    exited = _drive_without_event_loop(context.__aexit__(None, None, None))
    assert not exited
    assert events == ["initialize", "queue.start", "queue.close"]


def test_a_failed_initialization_starts_no_insights_worker(monkeypatch, main_module, capsys):
    events = []
    monkeypatch.setattr(main_module, "_INSIGHTS_QUEUE", _RecordingInsightsQueue(events))
    error = VectorSchemaError("document_chunks has an incompatible vector dimension")

    def fail_init_db(*args, **kwargs):
        raise error

    monkeypatch.setattr(main_module, "init_db", fail_init_db)
    context = main_module.app.router.lifespan_context(main_module.app)

    with pytest.raises(VectorSchemaError) as excinfo:
        _drive_without_event_loop(context.__aenter__())

    assert excinfo.value is error
    assert events == [], "no queue worker may start when initialization failed"


@pytest.mark.parametrize("failure", ["returns-false", "raises"])
def test_an_insights_worker_start_failure_never_stops_startup(monkeypatch, main_module, capsys, failure):
    events = []
    queue = (
        _RecordingInsightsQueue(events, start_result=False)
        if failure == "returns-false"
        else _RecordingInsightsQueue(events, start_error=RuntimeError("SECRET-DETAIL no threads"))
    )
    monkeypatch.setattr(main_module, "_INSIGHTS_QUEUE", queue)
    monkeypatch.setattr(main_module, "_initialize_application", lambda: None)
    context = main_module.app.router.lifespan_context(main_module.app)
    capsys.readouterr()

    _drive_without_event_loop(context.__aenter__())  # the application still serves
    exited = _drive_without_event_loop(context.__aexit__(None, None, None))

    assert not exited
    assert events == ["queue.start", "queue.close"]
    assert "SECRET-DETAIL" not in "".join(capsys.readouterr())


def test_shutdown_closes_the_queue_without_suppressing_an_exception(monkeypatch, main_module):
    events = []
    monkeypatch.setattr(main_module, "_INSIGHTS_QUEUE", _RecordingInsightsQueue(events))
    monkeypatch.setattr(main_module, "_initialize_application", lambda: None)
    context = main_module.app.router.lifespan_context(main_module.app)
    _drive_without_event_loop(context.__aenter__())
    error = RuntimeError("serving failed")

    # contextlib reports "not suppressed" by returning False once the lifespan
    # re-raised this same exception; a suppressing or replacing lifespan would
    # return True or raise something else instead.
    exited = _drive_without_event_loop(context.__aexit__(RuntimeError, error, None))

    assert exited is False, "lifespan shutdown must not suppress the exception"
    assert events == ["queue.start", "queue.close"], "the queue is still closed"


def test_repeated_lifespans_keep_one_queue_and_never_duplicate_a_busy_worker(monkeypatch, main_module):
    """A second lifespan in the same process reuses the same queue -- its
    accounting is never cleared -- and does not start a second worker while
    the first is still running a job. Shutdown never waits for that job."""
    created = []
    release = threading.Event()
    running = threading.Event()
    finished = []

    def run_job(document_id, call):
        running.set()
        assert release.wait(5)
        call.dispatching()
        call.observed({"done": True, "done_reason": "stop"})

    def thread_factory(*args, **kwargs):
        thread = threading.Thread(*args, **kwargs)
        created.append(thread)
        return thread

    queue = HelperJobQueue(
        settings=HelperQueueSettings(workers=1, max_queued=4), run_job=run_job,
        thread_factory=thread_factory, on_finished=lambda doc, state: finished.append((doc, state)),
        log=lambda line: None,
    )
    monkeypatch.setattr(main_module, "_INSIGHTS_QUEUE", queue)
    monkeypatch.setattr(main_module, "_initialize_application", lambda: None)

    first = main_module.app.router.lifespan_context(main_module.app)
    _drive_without_event_loop(first.__aenter__())
    assert queue.enqueue(5) is EnqueueOutcome.ACCEPTED
    assert running.wait(5)
    _drive_without_event_loop(first.__aexit__(None, None, None))  # returns while the job still runs
    assert queue.enqueue(6) is EnqueueOutcome.UNAVAILABLE

    second = main_module.app.router.lifespan_context(main_module.app)
    _drive_without_event_loop(second.__aenter__())
    assert main_module._INSIGHTS_QUEUE is queue
    assert len(created) == 1, "the busy worker is not duplicated"
    assert queue.enqueue(5) is EnqueueOutcome.DUPLICATE, "the running id is still accounted for"

    release.set()
    deadline = time.monotonic() + 5
    while not finished and time.monotonic() < deadline:
        time.sleep(0.01)
    assert finished == [(5, CallState.OBSERVED_TERMINAL)]
    _drive_without_event_loop(second.__aexit__(None, None, None))
