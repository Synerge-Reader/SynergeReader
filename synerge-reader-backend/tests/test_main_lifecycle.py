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
import uuid

import dotenv
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dbSetup
from dbSetup import VectorSchemaError


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


def test_initialize_application_uses_resolved_embedding_dimension(
    monkeypatch,
    main_module,
):
    calls = []

    def record_init_db(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(main_module, "init_db", record_init_db)

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
