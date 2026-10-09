"""Authorization contracts for GET /admin/system_status.

The route, the shared ``_require_admin`` helper and its
``_ACTIVE_ADMIN_FLAG_BY_TOKEN`` SQL are compiled out of the CURRENT main.py
source into an isolated namespace, the same AST approach as
tests/test_main_citation_wiring.py. main.py is never imported, so no
module-level setup, lifespan, database, Ollama or network code runs. Every
collaborator the route can reach -- the connection factory, the Ollama
endpoint resolver and ``requests`` -- is an in-memory double that records what
it was asked to do.

What these prove: a missing token is a 401 and an unknown, inactive or
non-admin token a 403 before any health or model work; any failure of the
authorization lookup itself -- connecting, a ``None`` connection, the cursor,
the query, the fetch, or closing the connection -- is a fixed, sanitized 503
that never reaches the health probes, the endpoint lookup or a model HTTP
call; an acquired connection is closed exactly once and ``None`` never; a
cleanup failure never hides an established 401/403; and a verified admin still
receives exactly the existing status payload.

What these do NOT prove: that PostgreSQL itself evaluates the admin predicate
(the rows here are synthetic, and only the exact SQL text and its bound
parameter are checked), or anything about Starlette/ASGI request handling.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from fastapi import HTTPException
import pytest

_MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"

_DEFINITIONS = ("_ACTIVE_ADMIN_FLAG_BY_TOKEN", "_require_admin", "admin_system_status")

_ADMIN_SQL = "SELECT is_admin FROM users WHERE token = %s AND COALESCE(is_active, 1) <> 0"
_UNAVAILABLE = (503, "Could not verify your session. Please try again shortly.")
_SECRET = "SECRET-DETAIL password=hunter2 host=db.internal"

# Synthetic tokens -> the is_admin row the admin lookup returns. An inactive
# account matches no row under the real predicate, so it is modelled as
# no row, exactly like an unknown token.
_ROWS = {
    "tok-admin": (1,),
    "tok-user": (0,),
    "tok-null-flag": (None,),
}


def _defined_name(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


@pytest.fixture(scope="module")
def main_tree():
    return ast.parse(_MAIN_PATH.read_text(encoding="utf-8"), filename=str(_MAIN_PATH))


class _Downstream:
    """Everything the route may do after authorization."""

    def __init__(self):
        self.probe_connects = 0
        self.endpoint_lookups = 0
        self.http_gets = []

    @property
    def touched(self):
        return self.probe_connects + self.endpoint_lookups + len(self.http_gets)

    def endpoint(self):
        self.endpoint_lookups += 1
        return "http://ollama.invalid:11434"

    def get(self, url, timeout=None):
        self.http_gets.append((url, timeout))
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"models": [{"name": "llama3.1:8b", "size": 4_900_000_000}]},
        )


class _ProbeConnection:
    def close(self):
        pass


class _AuthCursor:
    def __init__(self, connection):
        self._connection = connection
        self._row = None

    def execute(self, sql, params=()):
        self._connection.executed.append((" ".join(sql.split()), tuple(params)))
        if self._connection.fail == "execute":
            raise RuntimeError(_SECRET)
        self._row = _ROWS.get(params[0])

    def fetchone(self):
        if self._connection.fail == "fetchone":
            raise RuntimeError(_SECRET)
        return self._row


class _AuthConnection:
    def __init__(self, fail=None, fail_close=False):
        self.fail = fail
        self.fail_close = fail_close
        self.executed = []
        self.close_calls = 0

    def cursor(self):
        if self.fail == "cursor":
            raise RuntimeError(_SECRET)
        return _AuthCursor(self)

    def close(self):
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError(_SECRET)


def _compiled_route(main_tree, auth_connect, downstream):
    """The real route and auth helper, with every collaborator a double. The
    first connect_to_postgres call is the authorization lookup; any later call
    is the route's own database health probe (downstream work)."""
    calls = {"connects": 0}

    def connect_to_postgres():
        calls["connects"] += 1
        if calls["connects"] == 1:
            return auth_connect()
        downstream.probe_connects += 1
        return _ProbeConnection()

    namespace = {
        "app": SimpleNamespace(get=lambda path: (lambda function: function)),
        "Optional": Optional,
        "HTTPException": HTTPException,
        "connect_to_postgres": connect_to_postgres,
        "get_active_ollama_base_url": downstream.endpoint,
        "requests": SimpleNamespace(get=downstream.get),
        "OLLAMA_CONNECT_TIMEOUT": 1.5,
    }
    selected = [node for node in main_tree.body if _defined_name(node) in _DEFINITIONS]
    assert sorted(_defined_name(node) for node in selected) == sorted(_DEFINITIONS), (
        "every definition under test must exist exactly once in the current main.py"
    )
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(_MAIN_PATH), "exec"), namespace)
    assert namespace["_ACTIVE_ADMIN_FLAG_BY_TOKEN"] == _ADMIN_SQL, "the trusted admin predicate is unchanged"
    return namespace["admin_system_status"]


def _call(route, token):
    """The route awaits nothing, so one send() runs it to its result."""
    coroutine = route(token=token)
    try:
        coroutine.send(None)
    except StopIteration as finished:
        return finished.value
    coroutine.close()
    raise AssertionError("admin_system_status suspended; this harness runs no event loop")


def _status_of(main_tree, token, connection_or_factory):
    downstream = _Downstream()
    factory = connection_or_factory if callable(connection_or_factory) else (lambda: connection_or_factory)
    route = _compiled_route(main_tree, factory, downstream)
    with pytest.raises(HTTPException) as excinfo:
        _call(route, token)
    return excinfo.value, downstream


# --- success is unchanged ------------------------------------------------------


def test_a_verified_admin_receives_the_existing_status_payload(main_tree):
    connection = _AuthConnection()
    downstream = _Downstream()
    route = _compiled_route(main_tree, lambda: connection, downstream)

    payload = _call(route, "tok-admin")

    assert payload == {
        "database": {"ok": True},
        "llm_backend": {
            "ok": True,
            "url": "http://ollama.invalid:11434",
            "models": [{"name": "llama3.1:8b", "size_gb": 4.9}],
        },
    }
    assert connection.executed == [(_ADMIN_SQL, ("tok-admin",))], "the shared trusted lookup, synthetic token"
    assert connection.close_calls == 1, "the authorization connection is closed exactly once"
    assert downstream.probe_connects == 1 and downstream.endpoint_lookups == 1
    assert downstream.http_gets == [("http://ollama.invalid:11434/api/tags", (1.5, 5))]


# --- denied: 401 / 403 before any downstream work ---------------------------------


@pytest.mark.parametrize("token", [None, ""], ids=["absent", "empty"])
def test_a_missing_token_is_401_before_any_health_or_model_work(main_tree, token):
    connection = _AuthConnection()

    error, downstream = _status_of(main_tree, token, connection)

    assert (error.status_code, error.detail) == (401, "Unauthorized")
    assert connection.executed == [], "no lookup is spent on a missing token"
    assert connection.close_calls == 1
    assert downstream.touched == 0


@pytest.mark.parametrize("token", ["tok-unknown-or-inactive", "tok-user", "tok-null-flag"],
                         ids=["unknown-or-inactive", "non-admin", "null-admin-flag"])
def test_an_untrusted_token_is_403_before_any_health_or_model_work(main_tree, token):
    connection = _AuthConnection()

    error, downstream = _status_of(main_tree, token, connection)

    assert (error.status_code, error.detail) == (403, "Forbidden: Admin access required")
    assert connection.executed == [(_ADMIN_SQL, (token,))]
    assert connection.close_calls == 1
    assert downstream.touched == 0


# --- an authorization dependency failure is a sanitized 503 ------------------------


def _raising_connect():
    raise RuntimeError(_SECRET)


@pytest.mark.parametrize("failure", ["connect_raises", "connect_returns_none", "cursor", "execute", "fetchone"])
def test_an_authorization_lookup_failure_is_a_sanitized_503_with_no_downstream_work(main_tree, capsys, failure):
    connection = None
    if failure == "connect_raises":
        factory = _raising_connect
    elif failure == "connect_returns_none":
        factory = lambda: None  # noqa: E731
    else:
        connection = _AuthConnection(fail=failure)
        factory = lambda: connection  # noqa: E731
    capsys.readouterr()

    error, downstream = _status_of(main_tree, "tok-admin", factory)

    assert (error.status_code, error.detail) == _UNAVAILABLE
    assert downstream.touched == 0, "no health probe, endpoint discovery or model HTTP call"
    if connection is not None:
        assert connection.close_calls == 1, "an acquired connection is closed exactly once"
    captured = capsys.readouterr()
    for stream in (captured.out, captured.err, str(error.detail)):
        assert "SECRET-DETAIL" not in stream and "hunter2" not in stream and "tok-admin" not in stream


def test_a_close_failure_after_successful_authorization_is_503_and_stops(main_tree, capsys):
    connection = _AuthConnection(fail_close=True)
    capsys.readouterr()

    error, downstream = _status_of(main_tree, "tok-admin", connection)

    assert (error.status_code, error.detail) == _UNAVAILABLE
    assert connection.close_calls == 1
    assert downstream.touched == 0
    assert "SECRET-DETAIL" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "token, fail, expected",
    [
        (None, None, (401, "Unauthorized")),
        ("tok-user", None, (403, "Forbidden: Admin access required")),
        ("tok-admin", "execute", _UNAVAILABLE),
    ],
    ids=["401-kept", "403-kept", "503-kept"],
)
def test_a_close_failure_never_masks_an_established_refusal(main_tree, token, fail, expected):
    connection = _AuthConnection(fail=fail, fail_close=True)

    error, downstream = _status_of(main_tree, token, connection)

    assert (error.status_code, error.detail) == expected
    assert connection.close_calls == 1
    assert downstream.touched == 0
