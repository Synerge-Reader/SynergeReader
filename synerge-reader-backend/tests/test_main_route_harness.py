"""Real Starlette/FastAPI TestClient route-harness contracts for main.py.

REQUIRES THE SOCKETPAIR-AWARE SIBLING GUARD. Tests 1–4 construct a
real asyncio event loop, and on Windows every asyncio loop builds its
self-pipe with ``socket.socketpair()``, whose pure-Python emulation performs a
loopback ``socket.connect``. The audited Phase-B guard blocks that call, so
under the audited guard this file FAILS. That is intentional: it must fail
loudly rather than skip, because a skip would breach the zero-skip policy and
would silently hide a missing safety boundary.

The consequence, stated so nobody misreads a red run as a regression:

* The canonical invocation ``... -q -ra tests`` yields 309 tests and is valid
  ONLY under the sibling guard.
* Under the audited guard the suite can only be run as 304 tests, by passing
  ``--ignore=tests/test_main_route_harness.py`` explicitly.

MANDATORY PATCHING RULE. Every test that enters a ``TestClient`` context must
patch ``main._initialize_application`` (or ``main.init_db``) BEFORE entering
it. Unpatched, the real ``init_db`` reaches ``psycopg2.connect``, the guard
raises a ``RuntimeError`` that ``connect_to_postgres``'s
``except psycopg2.DatabaseError`` does not catch, and startup fails for the
wrong reason. This applies to tests 1, 2 and 4; test 3 patches ``init_db`` to
raise a ``VectorSchemaError`` on purpose. Test 5 does not enter a TestClient
context at all (see its docstring), so the rule does not apply to it.

No database, Ollama, external network, subprocess, or production service is
contacted: the initializer is always replaced before startup, and every route
request is served in-process by the ASGI transport.

Evidence produced by this file is LOCAL-ENVIRONMENT ONLY -- Windows, Python
3.11.9, FastAPI 0.135.3, Starlette 1.0.0, anyio 4.13.0, httpx 0.28.1. The
declared runtime in requiredInstall.txt has not been validated.
"""

import contextlib
import importlib
import os
from pathlib import Path
import socket
import subprocess
import sys

import dotenv
from fastapi.testclient import TestClient
import psycopg2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dbSetup import VectorSchemaError


# The neutral-environment and import-isolation logic below is duplicated from
# tests/test_main_lifecycle.py ON PURPOSE. Hoisting it into a shared
# conftest.py would introduce collection-wide state that every one of the ten
# test modules inherits, and would require editing a reviewed file to avoid
# duplicate fixture definitions. That consolidation is a larger change than
# this unit is authorised to make and requires separate review.
_EMBEDDING_PROFILE_KEYS = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_QUERY_PREFIX",
    "EMBEDDING_DOCUMENT_PREFIX",
    "EMBEDDING_PROFILE_UNVERIFIED_ACK",
)

_OPERATOR_PREFIX = "Application startup blocked by vector schema validation:"

_EXPECTED_SOCKETPAIR_MODULE = "sitecustomize"
_EXPECTED_SOCKETPAIR_QUALNAME = "_build_socketpair.<locals>.socketpair"


@contextlib.contextmanager
def _imported_main():
    """Import a fresh ``main`` and restore ``sys.modules`` exactly afterwards.

    Any pre-existing ``sys.modules["main"]`` is preserved and removed for the
    duration, so import order across the suite cannot change what a test sees,
    and no attribute patched on one imported copy can reach another.
    """
    previous = sys.modules.pop("main", None)
    try:
        yield importlib.import_module("main")
    finally:
        sys.modules.pop("main", None)
        if previous is not None:
            sys.modules["main"] = previous


def _guard_module():
    """The active Phase-B guard module, resolved without any filesystem path."""
    guard = sys.modules.get("sitecustomize")
    assert guard is not None, (
        "sitecustomize was not auto-loaded; this file requires a Phase-B guard "
        "directory on PYTHONPATH"
    )
    return guard


def _walk_exception_tree(error: BaseException):
    """Yield one exception and any exceptions nested in an exception group."""
    yield error
    if isinstance(error, BaseExceptionGroup):
        for nested in error.exceptions:
            yield from _walk_exception_tree(nested)


def _assert_blocked(blocked_type, label, operation):
    """The operation must raise the guard's exception with its exact label."""
    with pytest.raises(blocked_type) as excinfo:
        operation()
    assert str(excinfo.value) == f"SYNERGE_GUARD_BLOCKED:{label}", (
        f"{label} was blocked with an unexpected message: {excinfo.value}"
    )


def _connect_to_postgres_port():
    sock = socket.socket()
    try:
        sock.connect(("127.0.0.1", 5432))
    finally:
        sock.close()


def _connect_ex_to_postgres_port():
    sock = socket.socket()
    try:
        sock.connect_ex(("127.0.0.1", 5432))
    finally:
        sock.close()


@pytest.fixture
def neutral_main_environment(monkeypatch):
    """Keep runtime tests independent of an untracked backend .env file."""
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    for key in _EMBEDDING_PROFILE_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def main_module(neutral_main_environment):
    with _imported_main() as module:
        yield module


def test_testclient_context_creates_and_closes_event_loop(monkeypatch, main_module):
    monkeypatch.setattr(main_module, "_initialize_application", lambda: None)
    blocked_type = _guard_module().SynergeGuardBlocked

    client = TestClient(main_module.app)
    try:
        try:
            with client:
                pass
        except blocked_type as exc:
            pytest.fail(
                "the active guard blocked TestClient startup, so no event loop "
                f"could be built: {exc}. This file requires the sibling guard."
            )
    finally:
        client.close()

    assert client.portal is None, (
        "TestClient must release its blocking portal, and with it the event "
        "loop thread, when the context exits"
    )
    assert client.is_closed, "the TestClient transport must be closed afterwards"


def test_testclient_startup_initializes_once_then_serves_openapi(
    monkeypatch,
    main_module,
):
    calls = []
    monkeypatch.setattr(
        main_module,
        "_initialize_application",
        lambda: calls.append(None),
    )

    client = TestClient(main_module.app)
    try:
        with client:
            assert len(calls) == 1, (
                "FastAPI must run the lifespan initializer exactly once during "
                f"startup, before the first request; recorded {len(calls)}"
            )
            response = client.get("/openapi.json")
            try:
                status_code = response.status_code
                payload = response.json()
            finally:
                response.close()
            del response
    finally:
        client.close()
    del client

    assert status_code == 200, f"/openapi.json returned status {status_code}"
    assert payload.get("info", {}).get("title") == "SynergeReader API", (
        "the served schema must identify this application"
    )
    del payload

    assert len(calls) == 1, (
        "serving a request and shutting down must not initialize again; "
        f"recorded {len(calls)} call(s) in total"
    )


def test_vector_schema_error_blocks_testclient_startup_and_is_operator_visible(
    monkeypatch,
    main_module,
    capsys,
):
    # This deliberately overlaps test_main_lifecycle.py's loop-free equivalent.
    # That test proves the coroutine contract; this one proves that Starlette's
    # real ASGI lifespan protocol surfaces the failure and denies a usable
    # client. Neither subsumes the other.
    error = VectorSchemaError("document_chunks has an incompatible vector dimension")

    def fail_init_db(*args, **kwargs):
        raise error

    monkeypatch.setattr(main_module, "init_db", fail_init_db)
    capsys.readouterr()

    client = TestClient(main_module.app)
    entered = False
    raised = None
    try:
        try:
            with client:
                entered = True
        except BaseException as exc:
            raised = exc
    finally:
        client.close()

    assert not entered, "TestClient became usable after a failed startup"
    assert raised is not None, "a failed startup must not pass silently"
    assert any(candidate is error for candidate in _walk_exception_tree(raised)), (
        "the original VectorSchemaError object must be reachable by identity "
        f"through the raised exception; got {type(raised).__name__}: {raised}"
    )

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


def test_route_harness_boundary_remains_armed(monkeypatch, main_module):
    monkeypatch.setattr(main_module, "_initialize_application", lambda: None)
    guard = _guard_module()
    blocked_type = guard.SynergeGuardBlocked

    client = TestClient(main_module.app)
    try:
        with client:
            # Path-free sibling-guard identity: asserting on __module__ and
            # __qualname__ avoids committing a machine-specific temp-directory
            # UUID to the repository and survives guard regeneration.
            assert socket.socketpair.__module__ == _EXPECTED_SOCKETPAIR_MODULE, (
                "socket.socketpair must come from the guard, not the stdlib; "
                f"got {socket.socketpair.__module__!r}"
            )
            assert socket.socketpair.__qualname__ == _EXPECTED_SOCKETPAIR_QUALNAME, (
                "socket.socketpair must be the guard's closure-built "
                f"replacement; got {socket.socketpair.__qualname__!r}"
            )

            _assert_blocked(
                blocked_type,
                "socket.socket.connect",
                _connect_to_postgres_port,
            )
            _assert_blocked(
                blocked_type,
                "socket.socket.connect_ex",
                _connect_ex_to_postgres_port,
            )
            _assert_blocked(
                blocked_type,
                "socket.create_connection",
                lambda: socket.create_connection(("127.0.0.1", 11434)),
            )
            _assert_blocked(
                blocked_type,
                "psycopg2.connect",
                lambda: psycopg2.connect(
                    "postgresql://phase_b_guard:phase_b_guard@127.0.0.1:1/phase_b_guard"
                ),
            )
            _assert_blocked(
                blocked_type,
                "subprocess.Popen",
                lambda: subprocess.Popen(["cmd.exe", "/c", "exit", "0"]),
            )
    finally:
        client.close()


def test_main_module_isolation_is_deterministic(neutral_main_environment):
    """Isolation is proven WITHOUT entering a TestClient context.

    Nothing here starts a lifespan, so the mandatory patching rule does not
    apply and no initializer needs replacing. Two sequential ``_imported_main()``
    uses are enough to show that ``sys.modules`` is restored exactly and that
    attributes patched on one imported copy cannot reach the next.
    """
    baseline_present = "main" in sys.modules
    baseline_module = sys.modules.get("main")

    with _imported_main() as module_a:
        assert sys.modules.get("main") is module_a, (
            "_imported_main() installs its fresh import as sys.modules['main'] for "
            "the duration of the context -- that is how importlib works. The "
            "isolation guarantee proven here is exact restoration afterwards, not "
            "absence during."
        )
        sentinel = object()
        module_a._initialize_application = sentinel
        original_path = Path(module_a.__file__).resolve()

    assert ("main" in sys.modules) is baseline_present, (
        "sys.modules['main'] presence must be restored exactly after the first use"
    )
    assert sys.modules.get("main") is baseline_module, (
        "sys.modules['main'] must be restored to the exact prior object"
    )

    with _imported_main() as module_b:
        assert sys.modules.get("main") is module_b, (
            "the second import must also be published as sys.modules['main'] while "
            "its context is active"
        )
        assert module_b is not module_a, (
            "each _imported_main() use must yield a distinct module object"
        )
        assert Path(module_b.__file__).resolve() == original_path, (
            "both uses must import the same main.py source file"
        )
        assert module_b._initialize_application is not sentinel, (
            "state patched on one imported copy must not leak into the next"
        )
        assert callable(module_b._initialize_application), (
            "the fresh copy must carry its own real initializer"
        )

    assert ("main" in sys.modules) is baseline_present, (
        "sys.modules['main'] presence must be restored exactly after the second use"
    )
    assert sys.modules.get("main") is baseline_module, (
        "sys.modules['main'] must be restored to the exact prior object"
    )
