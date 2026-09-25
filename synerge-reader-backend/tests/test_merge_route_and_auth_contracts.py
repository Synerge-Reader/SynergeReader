"""Runtime contracts for integrating main@fc64a5a (PR #53) into E2.

These import the real ``main`` module -- with ``.env`` loading neutralised and
``connect_to_postgres`` replaced by an in-memory fake -- and read the live
FastAPI route table and helpers. No lifespan runs, so no database
initialisation, event loop, Ollama, network, or subprocess is involved, and the
file runs under the audited guard.

What these prove: the merged app serves exactly main@fc64a5a's route set,
method and path, so the unauthenticated filename delete cannot come back and
main's three document-history routes cannot be lost; those routes read the
Authorization header; the /ask mode contract is exactly the backend's answer
modes; and main's owner-scoped helpers and E2's authorized evidence scope
identify the same user from the same token, while an invalid token fails
closed on both.

What these do NOT prove: production security. In particular, E2's no-token
/ask scope still reads every ownerless document, the knowledge base is still
global, and /history still maps an invalid token to ownerless history. Those
are recorded here as current behaviour, not endorsed, and remain release
blockers on a separate track.
"""

import importlib
import os
from pathlib import Path
import sys
import typing

import dotenv
from fastapi import HTTPException
from fastapi.routing import APIRoute
from pydantic import ValidationError
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from citation_generation import AnswerMode
from schemas import AskRequest


_MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"

# Neutral-environment and import-isolation helpers are duplicated from
# tests/test_main_lifecycle.py on purpose, as the other runtime files do: a
# shared conftest.py would add collection-wide state to every test module.
_EMBEDDING_PROFILE_KEYS = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_QUERY_PREFIX",
    "EMBEDDING_DOCUMENT_PREFIX",
    "EMBEDDING_PROFILE_UNVERIFIED_ACK",
)

# Every (method, path) main@fc64a5a serves, taken from its route decorators.
# Pinned as a set so a restored, renamed, or lost route fails by name, not by
# a count that a swap would leave unchanged.
_MAIN_ROUTES_AT_FC64A5A = frozenset(
    {
        ("GET", "/admin/analytics"),
        ("GET", "/admin/audit_log"),
        ("GET", "/admin/chat_history"),
        ("GET", "/admin/check"),
        ("GET", "/admin/document_insights"),
        ("POST", "/admin/document_insights/analyze_pending"),
        ("GET", "/admin/documents"),
        ("GET", "/admin/overview"),
        ("GET", "/admin/ratings"),
        ("GET", "/admin/ratings/stats"),
        ("GET", "/admin/system_status"),
        ("GET", "/admin/users"),
        ("DELETE", "/admin/users/{user_id}"),
        ("PATCH", "/admin/users/{user_id}"),
        ("POST", "/ask"),
        ("POST", "/convert-docx"),
        ("GET", "/documents"),
        ("GET", "/documents/{document_id}/content"),
        ("POST", "/forgot-password"),
        ("POST", "/google-login"),
        ("POST", "/history"),
        ("GET", "/knowledge_base"),
        ("POST", "/knowledge_base"),
        ("POST", "/knowledge_base/import_url"),
        ("DELETE", "/knowledge_base/{entry_id}"),
        ("PUT", "/knowledge_base/{entry_id}"),
        ("POST", "/login"),
        ("GET", "/me"),
        ("GET", "/me/documents/search"),
        ("DELETE", "/me/documents/{document_id}"),
        ("GET", "/me/stats"),
        ("PUT", "/put_ratings"),
        ("POST", "/register"),
        ("POST", "/resend-verification"),
        ("POST", "/reset-password"),
        ("POST", "/submit_correction"),
        ("GET", "/test"),
        ("POST", "/upload"),
        ("GET", "/verify-email"),
    }
)

_DOCUMENT_HISTORY_ROUTES = (
    ("DELETE", "/me/documents/{document_id}"),
    ("GET", "/documents/{document_id}/content"),
    ("GET", "/me/documents/search"),
)

_ALICE = "11111111-1111-1111-1111-111111111111"
_BOB = "22222222-2222-2222-2222-222222222222"
_TOKENS = {"tok-alice": _ALICE, "tok-bob": _BOB}
# (id, filename, title, char_length, owner)
_DOCUMENTS = (
    (1, "alice-lease.pdf", "Lease", 900, _ALICE),
    (2, "bob-nda.pdf", "NDA", 700, _BOB),
    (3, "ownerless.txt", None, 300, None),
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
        yield importlib.import_module("main")
    finally:
        sys.modules.pop("main", None)
        if previous is not None:
            sys.modules["main"] = previous


def _api_routes(app):
    return {
        (method, route.path)
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    }


def _route(app, method, path):
    [route] = [
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == path and method in route.methods
    ]
    return route


class _FakeCursor:
    """Answers the token lookup and the two documents-scope queries only."""

    def __init__(self):
        self._rows = []

    def execute(self, sql, params=()):
        norm = " ".join(sql.split())
        if norm == "SELECT id FROM users WHERE token = %s":
            owner = _TOKENS.get(params[0])
            self._rows = [(owner,)] if owner else []
        elif "FROM documents" in norm and "WHERE user_id IS NULL" in norm:
            self._rows = [row[:4] for row in _DOCUMENTS if row[4] is None]
        elif "FROM documents" in norm and "WHERE user_id = %s" in norm:
            self._rows = [row[:4] for row in _DOCUMENTS if row[4] == params[0]]
        else:
            raise AssertionError(f"unexpected SQL in test: {norm}")

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def close(self):
        pass


class _FakeConnection:
    def cursor(self):
        return _FakeCursor()

    def close(self):
        pass


# --- route set ---------------------------------------------------------------


def test_the_merged_app_serves_exactly_mains_route_set(main_module):
    served = _api_routes(main_module.app)
    assert served - _MAIN_ROUTES_AT_FC64A5A == set(), "a route main does not serve came back"
    assert _MAIN_ROUTES_AT_FC64A5A - served == set(), "a route main serves was lost"


def test_the_unauthenticated_filename_delete_stays_gone(main_module):
    assert ("DELETE", "/documents/delete") not in _api_routes(main_module.app)
    assert not hasattr(main_module, "DeleteDocumentRequest")
    assert not hasattr(main_module, "delete_document")


@pytest.mark.parametrize(("method", "path"), _DOCUMENT_HISTORY_ROUTES)
def test_document_history_routes_take_the_token_from_the_authorization_header(
    main_module, method, path
):
    dependant = _route(main_module.app, method, path).dependant
    assert "authorization" in {param.alias for param in dependant.header_params}
    assert "token" not in {param.name for param in dependant.query_params}, (
        "a ?token= query parameter would leak into browser history and access logs"
    )


# --- /ask mode contract -------------------------------------------------------


def test_an_ask_request_defaults_to_graded_document_qa():
    assert AskRequest(question="q", model="m").mode == AnswerMode.DOCUMENT_QA.value


def test_the_accepted_modes_are_exactly_the_backend_answer_modes():
    accepted = set(typing.get_args(AskRequest.model_fields["mode"].annotation))
    assert accepted == {mode.value for mode in AnswerMode}


def test_an_unknown_mode_is_rejected_rather_than_guessed():
    with pytest.raises(ValidationError):
        AskRequest(question="q", model="m", mode="unrestricted")


# --- one token, one user, on both authorization paths ------------------------


def test_a_valid_token_identifies_the_same_user_on_both_paths(main_module, monkeypatch):
    monkeypatch.setattr(main_module, "connect_to_postgres", lambda: _FakeConnection())

    owner_id = main_module._user_id_for_token(_FakeCursor(), "tok-alice")
    scope = main_module._resolve_authorized_scope("tok-alice")

    assert owner_id == scope.user_id == _ALICE
    assert scope.established and not scope.anonymous
    assert [document.document_id for document in scope.documents] == [1], (
        "the evidence scope holds only the caller's own documents"
    )


@pytest.mark.parametrize("token", ["tok-forged", "Bearer tok-alice"])
def test_an_invalid_token_fails_closed_on_both_paths(main_module, monkeypatch, token):
    monkeypatch.setattr(main_module, "connect_to_postgres", lambda: _FakeConnection())

    with pytest.raises(HTTPException) as denied:
        main_module._user_id_for_token(_FakeCursor(), token)
    assert denied.value.status_code == 401

    scope = main_module._resolve_authorized_scope(token)
    assert not scope.established and scope.documents == (), (
        "an unknown token must not degrade into the anonymous or another user's scope"
    )


def test_a_missing_token_is_refused_by_owner_routes_but_ask_keeps_its_anonymous_scope(
    main_module, monkeypatch
):
    """Records current behaviour; it is not a security endorsement.

    Owner-scoped document routes refuse a request with no token. /ask keeps
    E2's anonymous scope -- every ownerless document -- which this integration
    deliberately does not change. Whether anonymous /ask should exist at all is
    an open release decision on the security track.
    """
    monkeypatch.setattr(main_module, "connect_to_postgres", lambda: _FakeConnection())

    with pytest.raises(HTTPException) as denied:
        main_module._user_id_for_token(_FakeCursor(), None)
    assert denied.value.status_code == 401

    scope = main_module._resolve_authorized_scope(None)
    assert scope.established and scope.anonymous and scope.user_id is None
    assert [document.document_id for document in scope.documents] == [3], (
        "no owned document is ever part of the anonymous scope"
    )
