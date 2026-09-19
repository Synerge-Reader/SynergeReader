"""Behavior tests for the document search / reopen / delete endpoints.

main.py cannot be imported in a test (it calls init_db(...) at module scope),
so -- like the other tests in this directory -- these read it as source. Unlike
the pure source-contract tests, they go one step further: the handful of
functions under test are lifted out of main.py's AST and executed against a
small in-memory fake database, so the assertions are about what the endpoints
actually DO (who can delete what, what a DB outage returns), not just how the
source is spelled.

No network, no real database, no subprocess, no filesystem writes.
"""

import ast
import asyncio
from pathlib import Path

import pytest
from fastapi import Header, HTTPException

_MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"
_WANTED = {
    "_bearer_token",
    "_open_db",
    "_user_id_for_token",
    "_require_admin",
    "search_my_documents",
    "get_document_content",
    "delete_my_document",
}


class FakeDB:
    """Just enough of Postgres for the statements those endpoints issue."""

    def __init__(self):
        # token -> (user_id, is_admin)
        self.users = {"tok-alice": ("alice", True), "tok-bob": ("bob", False), "tok-carol": ("carol", True)}
        # id -> dict(filename, user_id, content)
        self.documents = {}
        self.chunks = []  # document ids, one per chunk
        self.committed = 0
        self.rolled_back = 0
        self.closed = 0
        self.statements = []

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1

    def close(self):
        self.closed += 1


class FakeCursor:
    def __init__(self, db):
        self.db = db
        self._rows = []
        self.rowcount = 0

    def execute(self, sql, params=()):
        db = self.db
        norm = " ".join(sql.split())
        db.statements.append(norm)
        self._rows = []
        if norm.startswith("SELECT is_admin FROM users WHERE token"):
            u = db.users.get(params[0])
            self._rows = [(1 if u[1] else 0,)] if u else []
        elif norm.startswith("SELECT id FROM users WHERE token"):
            u = db.users.get(params[0])
            self._rows = [(u[0],)] if u else []
        elif norm.startswith("DELETE FROM document_chunks WHERE document_id IN"):
            doc_id, user_id = params
            owned = doc_id in db.documents and db.documents[doc_id]["user_id"] == user_id
            before = len(db.chunks)
            if owned:
                db.chunks = [c for c in db.chunks if c != doc_id]
            self.rowcount = before - len(db.chunks)
        elif norm.startswith("DELETE FROM documents WHERE id = %s AND user_id = %s RETURNING"):
            doc_id, user_id = params
            d = db.documents.get(doc_id)
            if d and d["user_id"] == user_id:
                del db.documents[doc_id]
                self._rows = [(doc_id, d["filename"])]
        elif norm.startswith("SELECT filename, content, upload_timestamp FROM documents WHERE id = %s AND user_id = %s"):
            doc_id, user_id = params
            d = db.documents.get(doc_id)
            self._rows = [(d["filename"], d["content"], "2026-09-01")] if d and d["user_id"] == user_id else []
        elif "FROM documents d" in norm:  # search
            user_id = params[0]
            rows = [
                (i, d["filename"], "2026-09-01", 1, None)
                for i, d in sorted(db.documents.items())
                if d["user_id"] == user_id
            ]
            if len(params) == 3:  # ILIKE variant: (user_id, pattern, limit)
                needle = params[1].strip("%").lower()
                rows = [r for r in rows if needle in r[1].lower()]
            self._rows = rows
        else:
            raise AssertionError(f"unexpected SQL in test: {norm}")

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


@pytest.fixture()
def env():
    tree = ast.parse(_MAIN_PATH.read_text(encoding="utf-8"), filename=str(_MAIN_PATH))
    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in _WANTED]
    assert {f.name for f in funcs} == _WANTED, "expected every endpoint/helper under test to exist in main.py"
    # Decorators are stripped: the routes are exercised as plain functions.
    for f in funcs:
        f.decorator_list = []
    module = ast.Module(body=funcs, type_ignores=[])
    ast.fix_missing_locations(module)
    db = FakeDB()
    ns = {
        "Optional": __import__("typing").Optional,
        "HTTPException": HTTPException,
        "Header": Header,
        "connect_to_postgres": lambda: db,
        "print": lambda *a, **k: None,
    }
    exec(compile(module, str(_MAIN_PATH), "exec"), ns)
    ns["_db"] = db
    return ns


def _run(coro):
    return asyncio.run(coro)


def _seed(db):
    # Two DIFFERENT users each own a document called contract.pdf.
    db.documents = {
        1: {"filename": "contract.pdf", "user_id": "alice", "content": "alice text"},
        2: {"filename": "contract.pdf", "user_id": "bob", "content": "bob text"},
        3: {"filename": "notes.txt", "user_id": "alice", "content": "more alice"},
    }
    db.chunks = [1, 1, 2, 3]


# ---------------------------------------------------------------- delete ----


def test_delete_without_authorization_is_401_and_touches_nothing(env):
    db = env["_db"]
    _seed(db)
    with pytest.raises(HTTPException) as exc:
        _run(env["delete_my_document"](2, authorization=None))
    assert exc.value.status_code == 401
    assert set(db.documents) == {1, 2, 3}
    assert not any(s.startswith("DELETE") for s in db.statements)


@pytest.mark.parametrize("header", ["Bearer nope", "Basic tok-alice", "tok-alice", "Bearer", ""])
def test_delete_with_bad_or_malformed_credentials_is_401(env, header):
    db = env["_db"]
    _seed(db)
    with pytest.raises(HTTPException) as exc:
        _run(env["delete_my_document"](1, authorization=header))
    assert exc.value.status_code == 401
    assert set(db.documents) == {1, 2, 3}


def test_owner_can_delete_own_document_and_its_chunks(env):
    db = env["_db"]
    _seed(db)
    out = _run(env["delete_my_document"](1, authorization="Bearer tok-alice"))
    assert out == {"deleted_id": 1, "filename": "contract.pdf", "chunks_deleted": 2}
    assert set(db.documents) == {2, 3}
    assert db.chunks == [2, 3]
    assert db.committed == 1 and db.closed == 1


def test_cannot_delete_someone_elses_document_and_it_looks_like_not_found(env):
    db = env["_db"]
    _seed(db)
    with pytest.raises(HTTPException) as exc:
        _run(env["delete_my_document"](2, authorization="Bearer tok-alice"))  # doc 2 is bob's
    assert exc.value.status_code == 404
    assert 2 in db.documents and db.chunks.count(2) == 1, "the other user's document and chunks must be untouched"
    assert db.committed == 0 and db.rolled_back >= 1 and db.closed == 1


def test_duplicate_filenames_delete_exactly_the_requested_document(env):
    db = env["_db"]
    _seed(db)
    _run(env["delete_my_document"](2, authorization="Bearer tok-bob"))
    # alice's identically-named document survives; the old filename-based route could hit either.
    assert db.documents[1]["filename"] == "contract.pdf" and db.documents[1]["user_id"] == "alice"
    assert 2 not in db.documents


def test_delete_never_reports_success_for_a_missing_document(env):
    db = env["_db"]
    _seed(db)
    with pytest.raises(HTTPException) as exc:
        _run(env["delete_my_document"](999, authorization="Bearer tok-alice"))
    assert exc.value.status_code == 404


def test_unexpected_db_error_rolls_back_closes_and_does_not_leak_details(env):
    db = env["_db"]
    _seed(db)
    original = FakeCursor.execute

    def flaky(self, sql, params=()):
        if sql.lstrip().startswith("DELETE"):
            raise RuntimeError("connection reset: secret-host.internal:5432")
        return original(self, sql, params)

    FakeCursor.execute = flaky
    try:
        with pytest.raises(HTTPException) as exc:
            _run(env["delete_my_document"](1, authorization="Bearer tok-alice"))
    finally:
        FakeCursor.execute = original
    assert exc.value.status_code == 500
    assert "secret-host" not in str(exc.value.detail), "internal error text must not reach the client"
    assert db.rolled_back >= 1 and db.closed == 1 and 1 in db.documents


# ------------------------------------------------- search / content scoping --


def test_search_and_content_are_admin_only_and_owner_scoped(env):
    db = env["_db"]
    _seed(db)
    # bob is not an admin
    with pytest.raises(HTTPException) as exc:
        _run(env["search_my_documents"](q="", limit=8, authorization="Bearer tok-bob"))
    assert exc.value.status_code == 403
    # alice only ever sees her own documents
    rows = _run(env["search_my_documents"](q="", limit=8, authorization="Bearer tok-alice"))
    assert {r["id"] for r in rows} == {1, 3}
    # and can't open bob's by id
    with pytest.raises(HTTPException) as exc:
        _run(env["get_document_content"](2, authorization="Bearer tok-alice"))
    assert exc.value.status_code == 404
    ok = _run(env["get_document_content"](1, authorization="Bearer tok-alice"))
    assert ok["content"] == "alice text"


def test_search_requires_a_token(env):
    with pytest.raises(HTTPException) as exc:
        _run(env["search_my_documents"](q="x", limit=8, authorization=None))
    assert exc.value.status_code == 401


# ------------------------------------------------------------- DB outage ----


@pytest.mark.parametrize(
    "call",
    [
        lambda e: e["delete_my_document"](1, authorization="Bearer tok-alice"),
        lambda e: e["get_document_content"](1, authorization="Bearer tok-alice"),
        lambda e: e["search_my_documents"](q="", limit=8, authorization="Bearer tok-alice"),
    ],
)
def test_database_outage_is_a_clean_503_not_an_attribute_error(env, call):
    # connect_to_postgres() returns None when the database is unreachable.
    env["connect_to_postgres"] = lambda: None
    with pytest.raises(HTTPException) as exc:
        _run(call(env))
    assert exc.value.status_code == 503


# -------------------------------------------------------- source contracts ---


def test_old_unauthenticated_filename_delete_route_is_gone_and_new_ones_use_headers():
    src = _MAIN_PATH.read_text(encoding="utf-8")
    assert '@app.delete("/documents/delete")' not in src
    assert "class DeleteDocumentRequest" not in src
    tree = ast.parse(src)
    for name in ("search_my_documents", "get_document_content", "delete_my_document"):
        fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name)
        arg_names = [a.arg for a in fn.args.args]
        assert "authorization" in arg_names, f"{name} must read the Authorization header"
        assert "token" not in arg_names, f"{name} must not take a ?token= query parameter"
