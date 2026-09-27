"""Route-level contracts for the PR #54 knowledge-base containment.

The knowledge base has no owner column and holds answers and Q&A derived from
users' private documents, so until it is user-owned:

* ``GET /knowledge_base`` is admin-only, identified by an
  ``Authorization: Bearer`` header (not a ``?token=`` query parameter), and a
  refused caller causes no knowledge-base query at all;
* ``POST /submit_correction`` and ``PUT /put_ratings`` require a valid Bearer
  identity and touch only the caller's own history row -- an anonymous or
  invalid-token request issues no history write at all, and a cross-user,
  ownerless, or missing-row request changes and commits nothing;
* no correction, from any caller, is copied into the knowledge base;
* ``POST /ask`` neither reads the knowledge base nor puts its text into the
  model prompt, in any answer mode -- while the caller's own document
  evidence, the answer stream, citations, and history persistence keep
  working.

Ownership (no anonymous or ownerless records):

* ``POST /ask`` refuses a missing, empty, or invalid token with 401 before it
  reads a document, calls a model, or writes history, and never serves another
  user's or an ownerless (legacy) document to an identified caller;
* ``POST /history`` and ``GET /documents`` return only the caller's own rows,
  refuse missing or invalid credentials with 401, and never return an
  ownerless row;
* deleting a user who still owns documents or chats is refused with 409 and
  moves nothing into the ownerless pool; a user who owns nothing can still be
  deleted.

The upload and /ask auto-write removals are also proven where those routes
are already exercised: tests/test_main_upload_route_adapter.py and
tests/test_main_citation_wiring.py.

These drive the real FastAPI app through ``TestClient`` against an in-memory
fake database that emulates the owner predicate (``user_id = NULL`` never
matches) and records every statement, so "nothing was written" is asserted
from the statements actually issued, not from the source text.

REQUIRES THE SOCKETPAIR-AWARE SIBLING GUARD, for the reason
tests/test_main_route_harness.py documents: entering a ``TestClient`` builds a
real asyncio event loop, whose Windows self-pipe uses ``socket.socketpair()``.
``main._initialize_application`` is replaced before any client is built, and
``main.connect_to_postgres``, ``main._EMBEDDING_PROVIDER`` and (for /ask)
``main.post_ollama`` are replaced with in-memory doubles, so there is no
database, Ollama, network, or subprocess.
"""

import ast
import contextlib
import copy
import importlib
import json
import os
from pathlib import Path
import sys

import bcrypt
import dotenv
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


_MAIN_PY_PATH = Path(__file__).resolve().parents[1] / "main.py"

# Duplicated from the other TestClient files on purpose (see
# tests/test_main_upload_route_adapter.py): a shared conftest.py would add
# collection-wide state to every test module.
_EMBEDDING_PROFILE_KEYS = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_QUERY_PREFIX",
    "EMBEDDING_DOCUMENT_PREFIX",
    "EMBEDDING_PROFILE_UNVERIFIED_ACK",
)

# token -> (user id, is_admin). carol owns no documents and no chats.
_USERS = {
    "tok-admin": ("admin-id", 1),
    "tok-admin2": ("admin2-id", 1),
    "tok-alice": ("alice-id", 0),
    "tok-bob": ("bob-id", 0),
    "tok-carol": ("carol-id", 0),
}

_INITIAL_HISTORY = {
    1: {"user_id": "alice-id", "answer": "alice answer", "comment": None, "rating": None},
    2: {"user_id": "bob-id", "answer": "bob answer", "comment": None, "rating": None},
    3: {"user_id": None, "answer": "ownerless answer", "comment": None, "rating": None},
    4: {"user_id": "admin-id", "answer": "admin answer", "comment": None, "rating": None},
}

# One KB row whose answer and source filename stand in for private content.
_KB_ROW = (
    11, "kb question", "kb original", "private kb answer", "2026-09-01",
    None, "User", 0, "private-contract.pdf", "document",
)

# id -> (filename, owner, content). Each scope holds exactly one short document,
# so the real planner supplies it whole (complete-document evidence), which
# needs no embedding. The leading marker identifies whose text reached a prompt.
_DOCUMENTS = {
    10: ("alice-lease.txt", "alice-id", "ALICE-DOC: the lease term is two years."),
    20: ("shared-notes.txt", None, "OWNERLESS-DOC: the notice period is thirty days."),
    30: ("bob-nda.txt", "bob-id", "BOB-DOC: the confidentiality period is five years."),
}
_DOC_MARKERS = ("ALICE-DOC", "OWNERLESS-DOC", "BOB-DOC")

# What a knowledge-base read would return. It must never reach a prompt.
_KB_SENTINEL = "KB-SENTINEL answer derived from another user's private document"

# Identity lookups. Every protected check must use an active-account form, so
# a token issued before a suspension stops identifying anyone. The plain forms
# are still answered -- regardless of suspension, exactly as SQL would -- for
# the post-check lookups (acting admin, audit actor), so a gate that regressed
# to a plain form would let a suspended token through and fail its test.
_ACTIVE_ID = "SELECT id FROM users WHERE token = %s AND COALESCE(is_active, 1) <> 0"
_ACTIVE_ADMIN = "SELECT is_admin FROM users WHERE token = %s AND COALESCE(is_active, 1) <> 0"
_ACTIVE_ID_AND_NAME = (
    "SELECT id, username FROM users WHERE token = %s AND COALESCE(is_active, 1) <> 0"
)

# Only owner-scoped document SQL is answered. The former anonymous-scope
# statements (``user_id IS NULL``) are deliberately unknown to this fake, so a
# regression that issues one lands in ``db.unexpected`` and fails the test.
_SCOPE_OWNED = (
    "SELECT id, filename, title, length(content) FROM documents "
    "WHERE user_id = %s ORDER BY id"
)
_LOAD_OWNED = "SELECT content FROM documents WHERE id = %s AND user_id = %s"
_INSERT_HISTORY = (
    "INSERT INTO chat_history (ts, selected_text, question, answer, user_id) "
    "VALUES (%s, %s, %s, %s, %s) RETURNING id"
)
_LIST_HISTORY = (
    "SELECT id, ts, selected_text, question, answer FROM chat_history "
    "WHERE user_id = %s ORDER BY id DESC LIMIT 20"
)
_LIST_DOCUMENTS = (
    "SELECT id, filename, upload_timestamp, author, title, publication_date, source, doi_url, "
    "(SELECT COUNT(*) FROM document_chunks WHERE document_id = documents.id) "
    "FROM documents WHERE user_id = %s ORDER BY upload_timestamp DESC"
)

_SELECT_OWNED_ROW = "SELECT id FROM chat_history WHERE id = %s AND user_id = %s"
_UPDATE_ANSWER = (
    "UPDATE chat_history SET answer = %s, comment = %s "
    "WHERE id = %s AND user_id = %s RETURNING id"
)
_UPDATE_RATING = (
    "UPDATE chat_history SET rating = %s, comment = %s "
    "WHERE id = %s AND user_id = %s RETURNING id"
)


class _FakeDB:
    """Committed state shared by every connection the app opens."""

    def __init__(self):
        self.users = dict(_USERS)
        self.inactive = set()
        self.passwords = {}
        self.documents = dict(_DOCUMENTS)
        self.chunks = []
        self.history = copy.deepcopy(_INITIAL_HISTORY)
        self.history_inserts = []
        self.audit_log = []
        self.statements = []
        self.unexpected = []
        self.connections = []
        self.commits = 0

    def connect(self):
        connection = _FakeConnection(self)
        self.connections.append(connection)
        return connection


class _FakeConnection:
    def __init__(self, db):
        self._db = db
        self._reset_pending()
        self.closed = False

    def _reset_pending(self):
        self._pending = {}
        self._pending_inserts = []
        self._pending_user_deletes = []
        self._pending_active = {}
        self._pending_documents = {}
        self._pending_chunks = []
        self._pending_audit = []

    def row(self, chat_id):
        if chat_id in self._pending:
            return self._pending[chat_id]
        return self._db.history.get(chat_id)

    def cursor(self):
        return _FakeCursor(self, self._db)

    def commit(self):
        db = self._db
        db.history.update(self._pending)
        for entry_id, params in self._pending_inserts:
            db.history_inserts.append(params)
            db.history[entry_id] = {
                "user_id": params[4], "answer": params[3], "comment": None, "rating": None,
            }
        for user_id in self._pending_user_deletes:
            db.users = {t: u for t, u in db.users.items() if u[0] != user_id}
        for user_id, active in self._pending_active.items():
            (db.inactive.discard if active else db.inactive.add)(user_id)
        db.documents.update(self._pending_documents)
        db.chunks.extend(self._pending_chunks)
        db.audit_log.extend(self._pending_audit)
        self._reset_pending()
        db.commits += 1

    def rollback(self):
        self._reset_pending()

    def close(self):
        self.closed = True


class _FakeCursor:
    def __init__(self, connection, db):
        self._connection = connection
        self._db = db
        self._rows = []

    def _owned(self, chat_id, user_id):
        row = self._connection.row(chat_id)
        # SQL semantics: user_id = %s never matches a NULL owner.
        return row if row is not None and row["user_id"] is not None and row["user_id"] == user_id else None

    def execute(self, sql, params=()):
        norm = " ".join(sql.split())
        self._db.statements.append((norm, tuple(params or ())))
        self._rows = []
        db = self._db
        users = db.users
        by_id = {user_id: token for token, (user_id, _) in users.items()}
        user = users.get(params[0]) if params and isinstance(params[0], str) else None
        active_user = user if user and user[0] not in db.inactive else None
        if norm == _ACTIVE_ADMIN:
            self._rows = [(active_user[1],)] if active_user else []
        elif norm == _ACTIVE_ID:
            self._rows = [(active_user[0],)] if active_user else []
        elif norm == _ACTIVE_ID_AND_NAME:
            self._rows = [(active_user[0], active_user[0].split("-")[0])] if active_user else []
        elif norm == "SELECT id FROM users WHERE token = %s":
            self._rows = [(user[0],)] if user else []
        elif norm == "SELECT id, username FROM users WHERE token = %s":
            self._rows = [(user[0], user[0].split("-")[0])] if user else []
        elif norm == "SELECT username, email, is_admin, is_active FROM users WHERE token = %s":
            self._rows = [(user[0].split("-")[0], None, user[1], 0 if user[0] in db.inactive else 1)] if user else []
        elif norm == "SELECT password, token, is_active, email_verified FROM users WHERE username = %s":
            token = next((t for t, (uid, _) in users.items() if uid.split("-")[0] == params[0]), None)
            if token and token in db.passwords:
                uid = users[token][0]
                self._rows = [(db.passwords[token], token, 0 if uid in db.inactive else 1, 1)]
        elif norm == "SELECT id, username FROM users WHERE id = %s":
            self._rows = [(params[0], params[0].split("-")[0])] if params[0] in by_id else []
        elif norm == "UPDATE users SET is_active = %s WHERE id = %s":
            self._connection._pending_active[params[1]] = bool(params[0])
        elif norm == "SELECT COUNT(*) FROM chat_history WHERE user_id = %s":
            self._rows = [(sum(1 for row in db.history.values() if row["user_id"] == params[0]),)]
        elif norm == "SELECT COUNT(*) FROM documents WHERE user_id = %s":
            self._rows = [(sum(1 for _, owner, _ in db.documents.values() if owner == params[0]),)]
        elif norm == "DELETE FROM users WHERE id = %s":
            self._connection._pending_user_deletes.append(params[0])
        elif norm.startswith("INSERT INTO admin_audit_log"):
            self._connection._pending_audit.append(tuple(params))
        elif norm.startswith("INSERT INTO documents"):
            # (filename, upload_timestamp, content, author, title,
            #  publication_date, source, doi_url, user_id)
            document_id = 500 + len(db.documents) + len(self._connection._pending_documents)
            self._connection._pending_documents[document_id] = (params[0], params[8], params[2])
            self._rows = [(document_id,)]
        elif norm.startswith("INSERT INTO document_chunks"):
            self._connection._pending_chunks.append(params[0])
        elif norm.startswith("SELECT id, question, original_answer") and "FROM knowledge_base" in norm:
            self._rows = [_KB_ROW]
        elif norm == _SCOPE_OWNED:
            # SQL semantics: user_id = %s never matches a NULL owner.
            self._rows = [
                (doc_id, filename, None, len(content))
                for doc_id, (filename, doc_owner, content) in sorted(db.documents.items())
                if doc_owner is not None and doc_owner == params[0]
            ]
        elif norm == _LOAD_OWNED:
            document = db.documents.get(params[0])
            owned = document and document[1] is not None and document[1] == params[1]
            self._rows = [(document[2],)] if owned else []
        elif norm == _LIST_HISTORY:
            self._rows = [
                (chat_id, f"2026-09-{chat_id}", "", f"question {chat_id}", row["answer"])
                for chat_id, row in sorted(db.history.items(), reverse=True)
                if row["user_id"] is not None and row["user_id"] == params[0]
            ]
        elif norm == _LIST_DOCUMENTS:
            self._rows = [
                (doc_id, filename, "2026-09-01", None, None, None, None, None, 1)
                for doc_id, (filename, owner, _) in sorted(db.documents.items())
                if owner is not None and owner == params[0]
            ]
        elif norm == _INSERT_HISTORY:
            entry_id = 900 + len(db.history_inserts) + len(self._connection._pending_inserts) + 1
            self._connection._pending_inserts.append((entry_id, tuple(params)))
            self._rows = [(entry_id,)]
        elif norm == _SELECT_OWNED_ROW:
            chat_id, user_id = params
            self._rows = [(chat_id,)] if self._owned(chat_id, user_id) else []
        elif norm in (_UPDATE_ANSWER, _UPDATE_RATING):
            value, comment, chat_id, user_id = params
            row = self._owned(chat_id, user_id)
            if row is not None:
                field = "answer" if norm == _UPDATE_ANSWER else "rating"
                self._connection._pending[chat_id] = {**row, field: value, "comment": comment}
                self._rows = [(chat_id,)]
        else:
            self._db.unexpected.append(norm)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def close(self):
        pass


def _kb_statements(db):
    return [sql for sql, _ in db.statements if "knowledge_base" in sql]


def _history_writes(db):
    return [
        sql for sql, _ in db.statements
        if sql.startswith(("UPDATE", "INSERT", "DELETE")) and "chat_history" in sql
    ]


def _assert_nothing_written(db):
    assert _history_writes(db) == [], "a refused request must issue no history write"
    assert _kb_statements(db) == [], "a refused request must issue no knowledge-base statement"
    assert db.commits == 0
    assert db.history == _INITIAL_HISTORY
    assert db.unexpected == []
    assert all(connection.closed for connection in db.connections)


@contextlib.contextmanager
def _imported_main():
    previous = sys.modules.pop("main", None)
    try:
        yield importlib.import_module("main")
    finally:
        sys.modules.pop("main", None)
        if previous is not None:
            sys.modules["main"] = previous


@pytest.fixture
def main_module(monkeypatch):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    for key in _EMBEDDING_PROFILE_KEYS:
        monkeypatch.delenv(key, raising=False)
    with _imported_main() as module:
        yield module


@pytest.fixture
def db():
    return _FakeDB()


@pytest.fixture
def embedding_calls(monkeypatch, main_module):
    """Any embedding means a knowledge-base write was being prepared."""
    calls = []

    class _Tripwire:
        def embed_documents(self, texts):
            calls.append(list(texts))
            raise AssertionError("no containment route may embed anything")

        def embed_query(self, text):
            calls.append([text])
            raise AssertionError("no containment route may embed anything")

    monkeypatch.setattr(main_module, "_EMBEDDING_PROVIDER", _Tripwire())
    return calls


@pytest.fixture
def client(monkeypatch, main_module, db, embedding_calls):
    # MANDATORY PATCHING RULE: replace the initializer BEFORE the TestClient
    # is constructed, so startup never reaches psycopg2.connect.
    monkeypatch.setattr(main_module, "_initialize_application", lambda: None)
    monkeypatch.setattr(main_module, "connect_to_postgres", db.connect)
    test_client = TestClient(main_module.app)
    try:
        with test_client:
            yield test_client
    finally:
        test_client.close()


_NO_VALID_IDENTITY = pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer tok-bogus"}, {"Authorization": "tok-alice"}],
    ids=["anonymous", "invalid-token", "malformed-header"],
)

_NOT_THE_CALLERS_ROW = pytest.mark.parametrize(
    ("token", "chat_id"),
    [("tok-alice", 2), ("tok-alice", 3), ("tok-admin", 1), ("tok-alice", 999)],
    ids=["other-users-row", "ownerless-row", "admin-on-a-users-row", "missing-row"],
)


# --- GET /knowledge_base -----------------------------------------------------


_ADMIN_CHECK = _ACTIVE_ADMIN


def _assert_kb_read_refused(response, db, status):
    assert response.status_code == status
    assert _kb_statements(db) == [], "the admin check must run before any KB query"
    for private in ("private kb answer", "private-contract.pdf", "kb question"):
        assert private not in response.text
    assert all(connection.closed for connection in db.connections)


@pytest.mark.parametrize(
    "headers",
    [
        {},
        # An admin's real token without the Bearer scheme, or under another
        # scheme, is still malformed: the header, not the token alone, counts.
        {"Authorization": "tok-admin"},
        {"Authorization": "Basic tok-admin"},
    ],
    ids=["missing", "malformed-no-scheme", "malformed-wrong-scheme"],
)
def test_knowledge_base_read_without_a_bearer_token_is_401_before_any_query(client, db, headers):
    response = client.get("/knowledge_base", headers=headers)

    _assert_kb_read_refused(response, db, 401)
    assert db.statements == [], "no usable token means no admin lookup either"


@pytest.mark.parametrize(
    "token", ["tok-bogus", "tok-alice"], ids=["invalid-token", "ordinary-user"]
)
def test_knowledge_base_read_by_a_non_admin_bearer_token_is_403(client, db, token):
    response = client.get("/knowledge_base", headers={"Authorization": f"Bearer {token}"})

    _assert_kb_read_refused(response, db, 403)
    assert db.statements == [(_ADMIN_CHECK, (token,))], (
        "the existing admin check must run with the Bearer token, and nothing after it"
    )


def test_a_query_string_token_is_no_longer_accepted(client, db):
    """Even an admin token is refused in the URL form, which leaks into browser
    history and access logs; only the Authorization header is read."""
    response = client.get("/knowledge_base", params={"token": "tok-admin"})

    _assert_kb_read_refused(response, db, 401)
    assert db.statements == []


def test_an_admin_bearer_token_reads_the_knowledge_base(client, db):
    response = client.get("/knowledge_base", headers={"Authorization": "Bearer tok-admin"})

    assert response.status_code == 200, response.text
    [entry] = response.json()
    assert entry["id"] == 11
    assert entry["answer"] == "private kb answer"
    assert entry["context_text"] == "private-contract.pdf"
    assert db.statements[0] == (_ADMIN_CHECK, ("tok-admin",)), (
        "the admin check must run first, with the Bearer token"
    )
    assert [sql.split(" ", 1)[0] for sql in _kb_statements(db)] == ["SELECT"]
    assert all(connection.closed for connection in db.connections)


# --- POST /submit_correction -------------------------------------------------


def _correct(client, chat_id, headers):
    return client.post(
        "/submit_correction",
        json={"chat_id": chat_id, "corrected_answer": "corrected text", "comment": "fix"},
        headers=headers,
    )


@_NO_VALID_IDENTITY
def test_correction_without_a_valid_identity_writes_nothing(client, db, embedding_calls, headers):
    response = _correct(client, 1, headers)

    assert response.status_code == 401
    _assert_nothing_written(db)
    assert embedding_calls == []


@_NOT_THE_CALLERS_ROW
def test_correction_of_a_row_the_caller_does_not_own_writes_nothing(
    client, db, embedding_calls, token, chat_id
):
    response = _correct(client, chat_id, {"Authorization": f"Bearer {token}"})

    assert response.status_code == 404, (
        "another user's, an ownerless, and a missing row must be indistinguishable"
    )
    _assert_nothing_written(db)
    assert embedding_calls == []


def test_an_owner_correction_updates_only_the_owners_history(client, db, embedding_calls):
    response = _correct(client, 1, {"Authorization": "Bearer tok-alice"})

    assert response.status_code == 200, response.text
    assert response.json() == {"message": "Correction saved to your history", "chat_id": 1}
    assert db.history[1] == {
        "user_id": "alice-id", "answer": "corrected text", "comment": "fix", "rating": None,
    }
    assert db.commits == 1
    assert {k: v for k, v in db.history.items() if k != 1} == {
        k: v for k, v in _INITIAL_HISTORY.items() if k != 1
    }
    scoped = [(sql, params) for sql, params in db.statements if "chat_history" in sql]
    assert scoped == [
        (_SELECT_OWNED_ROW, (1, "alice-id")),
        (_UPDATE_ANSWER, ("corrected text", "fix", 1, "alice-id")),
    ], "both the SELECT and the UPDATE must be scoped to the caller's user id"
    assert _kb_statements(db) == [] and embedding_calls == [], (
        "a correction must not be copied into the shared knowledge base"
    )
    assert db.unexpected == []


def test_an_admins_own_correction_is_not_copied_into_the_knowledge_base(
    client, db, embedding_calls
):
    response = _correct(client, 4, {"Authorization": "Bearer tok-admin"})

    assert response.status_code == 200, response.text
    assert db.history[4]["answer"] == "corrected text"
    assert _kb_statements(db) == [] and embedding_calls == []


# --- PUT /put_ratings --------------------------------------------------------


def _rate(client, chat_id, headers):
    return client.put(
        "/put_ratings",
        json={"id": chat_id, "rating": 5, "comment": ""},
        headers=headers,
    )


@_NO_VALID_IDENTITY
def test_rating_without_a_valid_identity_writes_nothing(client, db, headers):
    response = _rate(client, 1, headers)

    assert response.status_code == 401
    _assert_nothing_written(db)


@_NOT_THE_CALLERS_ROW
def test_rating_a_row_the_caller_does_not_own_writes_nothing(client, db, token, chat_id):
    response = _rate(client, chat_id, {"Authorization": f"Bearer {token}"})

    assert response.status_code == 404
    # The rating route has no separate SELECT: its one statement is the
    # owner-scoped UPDATE, which matches no row here, so nothing changes.
    caller_id = _USERS[token][0]
    assert [(sql, params) for sql, params in db.statements if "chat_history" in sql] == [
        (_UPDATE_RATING, (5, "", chat_id, caller_id)),
    ]
    assert db.commits == 0
    assert db.history == _INITIAL_HISTORY
    assert _kb_statements(db) == [] and db.unexpected == []
    assert all(connection.closed for connection in db.connections)


def test_an_owner_rating_updates_only_the_owners_history(client, db):
    response = _rate(client, 1, {"Authorization": "Bearer tok-alice"})

    assert response.status_code == 200, response.text
    assert response.json() == {"message": "Rating updated", "id": 1}
    assert db.history[1]["rating"] == 5
    assert db.commits == 1
    assert {k: v for k, v in db.history.items() if k != 1} == {
        k: v for k, v in _INITIAL_HISTORY.items() if k != 1
    }
    assert [(sql, params) for sql, params in db.statements if "chat_history" in sql] == [
        (_UPDATE_RATING, (5, "", 1, "alice-id")),
    ]
    assert _kb_statements(db) == [] and db.unexpected == []


# --- POST /ask ---------------------------------------------------------------


_ANSWER_TOKENS = ("The term ", "is stated in ", "the document [C1].")


class _FakeGeneration:
    """A streamed /api/generate response: NDJSON lines cut into small chunks."""

    def __init__(self, tokens):
        self._payload = "".join(json.dumps({"response": token}) + "\n" for token in tokens)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, decode_unicode=True, chunk_size=32):
        for index in range(0, len(self._payload), chunk_size):
            yield self._payload[index : index + chunk_size]


class _FakeVerifierReply:
    def raise_for_status(self):
        pass

    def json(self):
        return {"response": "supported"}


@pytest.fixture
def ollama_calls(monkeypatch, main_module):
    calls = []

    def post_ollama(endpoint, payload, *, stream=False, timeout=60):
        calls.append({"endpoint": endpoint, "payload": payload, "stream": stream})
        return _FakeGeneration(_ANSWER_TOKENS) if stream else _FakeVerifierReply()

    monkeypatch.setattr(main_module, "post_ollama", post_ollama)
    return calls


@pytest.fixture
def kb_helper_calls(monkeypatch, main_module):
    """Replace every knowledge-base helper with a recorder. A reader would hand
    back the sentinel, so a regression shows up both as a recorded call and
    as sentinel text in the model prompt."""
    calls = []

    def get_relevant_knowledge_base(question, limit=3):
        calls.append("get_relevant_knowledge_base")
        return [{"id": 71, "question": "KB question?", "answer": _KB_SENTINEL,
                 "context": "", "corrected_by": "User", "usage_count": 0,
                 "relevance_score": 0.99}]

    monkeypatch.setattr(main_module, "get_relevant_knowledge_base", get_relevant_knowledge_base)
    monkeypatch.setattr(main_module, "increment_kb_usage", lambda ids: calls.append("increment_kb_usage"))
    monkeypatch.setattr(main_module, "auto_save_to_kb", lambda *args: calls.append("auto_save_to_kb"))
    return calls


def _ask(client, auth_token, mode="document_qa", document_ids=None):
    body = {
        "question": "How long is the term?",
        "model": "local-test-model",
        "auth_token": auth_token,
        "mode": mode,
    }
    if document_ids is not None:
        body["document_ids"] = document_ids
    return client.post("/ask", json=body)


def _events(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


@pytest.mark.parametrize("mode", ["document_qa", "structured_json", "model_reasoning"])
def test_ask_by_the_owner_uses_only_its_documents_and_no_knowledge_base(
    client, db, ollama_calls, kb_helper_calls, embedding_calls, mode,
):
    response = _ask(client, "tok-alice", mode)

    assert response.status_code == 200, response.text
    events = _events(response)
    assert events[-1] == {"type": "done", "ok": True}, "the answer still completes"

    # No knowledge base anywhere: no helper call, no SQL, no text in any
    # model request, and nothing embedded to search it with.
    assert kb_helper_calls == []
    assert _kb_statements(db) == [] and db.unexpected == []
    assert embedding_calls == []
    [generation] = [call for call in ollama_calls if call["stream"]]
    prompt = generation["payload"]["prompt"]
    assert "knowledge_base_corrections" not in prompt
    for call in ollama_calls:
        assert _KB_SENTINEL not in json.dumps(call["payload"])

    # Document evidence, citations, streaming, and history keep working, and
    # only for the caller's own document.
    [evidence] = [event for event in events if event["type"] == "evidence"]
    assert [citation["filename"] for citation in evidence["citations"]] == ["alice-lease.txt"]
    assert "ALICE-DOC" in prompt
    assert "OWNERLESS-DOC" not in prompt and "BOB-DOC" not in prompt
    streamed = "".join(event["text"] for event in events if event["type"] == "delta")
    assert streamed == "".join(_ANSWER_TOKENS)
    assert [event["type"] for event in events].count("entry_id") == 1
    assert [params[4] for params in db.history_inserts] == ["alice-id"], (
        "the answer's history row belongs to the caller"
    )
    assert all(connection.closed for connection in db.connections)


@pytest.mark.parametrize(
    ("auth_token", "detail"),
    [(None, "Unauthorized"), ("", "Unauthorized"), ("tok-forged", "Invalid session")],
    ids=["missing-token", "empty-token", "invalid-token"],
)
def test_ask_without_a_valid_identity_is_401_before_anything_runs(
    client, db, ollama_calls, kb_helper_calls, embedding_calls, auth_token, detail,
):
    """No anonymous /ask: an unidentified request reads no document -- not even
    an ownerless legacy one -- calls no model, and writes no history row."""
    response = _ask(client, auth_token)

    assert response.status_code == 401
    assert response.json() == {"detail": detail}
    assert ollama_calls == [] and kb_helper_calls == [] and embedding_calls == []
    assert db.history_inserts == [] and db.commits == 0
    expected = [] if not auth_token else [(_ACTIVE_ID, (auth_token,))]
    assert db.statements == expected, "only the identity lookup may run, and only for a supplied token"
    assert all(connection.closed for connection in db.connections)


@pytest.mark.parametrize(
    "document_ids",
    [[20], [30], [20, 30]],
    ids=["ownerless-legacy-document", "other-users-document", "both"],
)
def test_ask_never_serves_another_users_or_an_ownerless_document(
    client, db, ollama_calls, document_ids,
):
    response = _ask(client, "tok-alice", document_ids=document_ids)

    assert response.status_code == 200, response.text
    events = _events(response)
    [evidence] = [event for event in events if event["type"] == "evidence"]
    assert evidence["citations"] == [], "an id outside the caller's own documents is dropped"
    assert "document_not_authorized" in evidence["warnings"]
    [generation] = [call for call in ollama_calls if call["stream"]]
    for marker in _DOC_MARKERS:
        assert marker not in generation["payload"]["prompt"]
    assert [params[4] for params in db.history_inserts] == ["alice-id"]
    assert db.unexpected == []


def test_ask_with_mixed_ids_keeps_only_the_callers_own_document(client, db, ollama_calls):
    response = _ask(client, "tok-alice", document_ids=[10, 20, 30])

    assert response.status_code == 200, response.text
    [evidence] = [event for event in _events(response) if event["type"] == "evidence"]
    assert [citation["filename"] for citation in evidence["citations"]] == ["alice-lease.txt"]
    [generation] = [call for call in ollama_calls if call["stream"]]
    prompt = generation["payload"]["prompt"]
    assert "ALICE-DOC" in prompt
    assert "OWNERLESS-DOC" not in prompt and "BOB-DOC" not in prompt


# --- POST /history -----------------------------------------------------------


_NO_FIELD = object()


def _history(client, token):
    body = {} if token is _NO_FIELD else {"token": token}
    return client.post("/history", json=body)


@pytest.mark.parametrize(
    ("token", "expected_ids"),
    [("tok-alice", [1]), ("tok-bob", [2]), ("tok-admin", [4]), ("tok-carol", [])],
    ids=["owner-alice", "owner-bob", "admin-own-history-only", "user-with-no-history"],
)
def test_history_returns_only_the_callers_own_rows(client, db, token, expected_ids):
    response = _history(client, token)

    assert response.status_code == 200, response.text
    returned = [item["id"] for item in response.json()]
    assert returned == expected_ids
    assert 3 not in returned, "the ownerless legacy row is returned to no one"
    assert db.unexpected == [] and db.commits == 0


@pytest.mark.parametrize(
    ("token", "detail"),
    [(_NO_FIELD, "Unauthorized"), (None, "Unauthorized"), ("tok-forged", "Invalid session")],
    ids=["missing-field", "null-token", "invalid-token"],
)
def test_history_without_a_valid_identity_is_401_and_reads_no_rows(client, db, token, detail):
    response = _history(client, token)

    assert response.status_code == 401
    assert response.json() == {"detail": detail}
    assert not [sql for sql, _ in db.statements if "chat_history" in sql], (
        "no history query may run -- in particular not the old ownerless fallback"
    )
    assert "ownerless answer" not in response.text
    assert all(connection.closed for connection in db.connections)


# --- GET /documents ----------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected_ids"),
    [("tok-alice", [10]), ("tok-bob", [30]), ("tok-carol", []), ("tok-admin", [])],
    ids=["owner-alice", "owner-bob", "user-with-no-documents", "admin-own-documents-only"],
)
def test_documents_lists_only_the_callers_own_metadata(client, db, token, expected_ids):
    response = client.get("/documents", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200, response.text
    listed = response.json()
    assert [document["id"] for document in listed] == expected_ids
    listed_names = {document["filename"] for document in listed}
    for filename in ("alice-lease.txt", "shared-notes.txt", "bob-nda.txt"):
        if filename not in listed_names:
            assert filename not in response.text, f"{filename} leaked into another caller's list"
    assert db.unexpected == []


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "tok-alice"}, {"Authorization": "Bearer tok-forged"}],
    ids=["missing-header", "malformed-header", "invalid-token"],
)
def test_documents_without_a_valid_identity_is_401_and_lists_nothing(client, db, headers):
    response = client.get("/documents", headers=headers)

    assert response.status_code == 401
    assert not [sql for sql, _ in db.statements if "FROM documents" in sql]
    for filename in ("alice-lease.txt", "shared-notes.txt", "bob-nda.txt"):
        assert filename not in response.text
    assert all(connection.closed for connection in db.connections)


# --- DELETE /admin/users/{user_id} -------------------------------------------


def _unlinking_statements(db):
    return [sql for sql, _ in db.statements if "SET user_id = NULL" in sql]


def _user_ids(db):
    return {user_id for user_id, _ in db.users.values()}


@pytest.mark.parametrize("target", ["alice-id", "bob-id"])
def test_deleting_a_user_who_owns_records_is_refused_and_moves_nothing(client, db, target):
    response = client.delete(f"/admin/users/{target}", params={"token": "tok-admin"})

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "1 document(s)" in detail and "1 chat(s)" in detail
    assert _unlinking_statements(db) == [], "no record may be moved into the ownerless pool"
    assert not [sql for sql, _ in db.statements if sql.startswith("DELETE")]
    assert target in _user_ids(db), "the account is kept"
    assert db.history == _INITIAL_HISTORY, "every history row, ownerless ones included, is unchanged"
    assert db.commits == 0 and db.audit_log == []


def test_deleting_a_user_who_owns_nothing_still_works(client, db):
    response = client.delete("/admin/users/carol-id", params={"token": "tok-admin"})

    assert response.status_code == 200, response.text
    assert "carol-id" not in _user_ids(db)
    assert _unlinking_statements(db) == []
    assert db.history == _INITIAL_HISTORY
    assert [entry[3] for entry in db.audit_log] == ["delete_user"]
    assert db.unexpected == []


@pytest.mark.parametrize(
    ("params", "status"),
    [({}, 401), ({"token": "tok-forged"}, 403), ({"token": "tok-alice"}, 403)],
    ids=["missing-token", "invalid-token", "ordinary-user"],
)
def test_deleting_a_user_requires_an_admin(client, db, params, status):
    response = client.delete("/admin/users/carol-id", params=params)

    assert response.status_code == status
    assert "carol-id" in _user_ids(db)
    assert _unlinking_statements(db) == [] and db.commits == 0


# --- release containment: normal active-user flow ----------------------------


@pytest.fixture
def followups(monkeypatch, main_module):
    started = []
    monkeypatch.setattr(
        main_module, "_start_background_task", lambda target, args: started.append(target)
    )
    return started


@pytest.fixture
def fixed_embeddings(monkeypatch, main_module, client):
    """Upload ingestion embeds its chunks; replaces the no-embedding tripwire."""
    dimension = main_module._EMBEDDING_PROFILE.dimension

    class _FixedEmbeddings:
        def embed_documents(self, texts):
            return [[0.25] * dimension for _ in texts]

    monkeypatch.setattr(main_module, "_EMBEDDING_PROVIDER", _FixedEmbeddings())


def test_an_active_user_uploads_asks_with_citations_and_sees_history(
    client, db, ollama_calls, followups, fixed_embeddings, main_module,
):
    """End to end through the real upload, ingestion, /ask, and /history code:
    only the database, embedding provider, model, and threads are doubles."""
    upload = client.post(
        "/upload",
        files=[("files", ("carol-retainer.txt", b"CAROL-DOC: the retainer is ten thousand dollars.\n", "text/plain"))],
        data={"auth_token": "tok-carol"},
    )
    assert upload.status_code == 200, upload.text
    [result] = upload.json()["results"]
    assert result["status"] == "indexed"
    document_id = result["document_id"]
    assert db.documents[document_id][:2] == ("carol-retainer.txt", "carol-id"), (
        "the uploaded document belongs to the uploader"
    )
    assert document_id in db.chunks
    assert followups == [main_module._extract_document_insights]

    ask = client.post(
        "/ask",
        json={"question": "What is the retainer?", "model": "local-test-model",
              "auth_token": "tok-carol", "mode": "document_qa"},
    )
    assert ask.status_code == 200, ask.text
    events = [json.loads(line) for line in ask.text.splitlines() if line.strip()]
    assert events[-1] == {"type": "done", "ok": True}
    [evidence] = [event for event in events if event["type"] == "evidence"]
    assert [citation["filename"] for citation in evidence["citations"]] == ["carol-retainer.txt"]
    assert [citation["document_id"] for citation in evidence["citations"]] == [document_id]
    [verification] = [event for event in events if event["type"] == "verification"]
    assert verification["used_citation_ids"] == ["C1"], "the answer's citation resolves"
    [entry] = [event["entry_id"] for event in events if event["type"] == "entry_id"]
    [generation] = [call for call in ollama_calls if call["stream"]]
    assert "CAROL-DOC" in generation["payload"]["prompt"]
    assert not any(marker in generation["payload"]["prompt"] for marker in _DOC_MARKERS)

    history = client.post("/history", json={"token": "tok-carol"})
    assert history.status_code == 200, history.text
    assert [(item["id"], item["answer"]) for item in history.json()] == [
        (entry, "".join(_ANSWER_TOKENS))
    ]
    documents = client.get("/documents", headers={"Authorization": "Bearer tok-carol"})
    assert [document["id"] for document in documents.json()] == [document_id]
    assert db.unexpected == []


def test_username_and_password_login_still_works(client, db):
    db.passwords["tok-alice"] = bcrypt.hashpw(b"correct horse", bcrypt.gensalt(4)).decode()

    response = client.post("/login", json={"username": "alice", "password": "correct horse"})

    assert response.status_code == 200, response.text
    assert response.json()["token"] == "tok-alice"


# --- release containment: suspended accounts ---------------------------------


def _set_active(client, user_id, active, admin_token="tok-admin"):
    response = client.patch(
        f"/admin/users/{user_id}", json={"token": admin_token, "is_active": active}
    )
    assert response.status_code == 200, response.text


_ALICE_BEARER = {"Authorization": "Bearer tok-alice"}

# Every protected route a signed-in user can call, with the token alice was
# issued before her account was suspended.
_SUSPENDED_USER_CALLS = {
    "ask": lambda c: c.post("/ask", json={"question": "q", "model": "m", "auth_token": "tok-alice"}),
    "upload": lambda c: c.post(
        "/upload", files=[("files", ("x.txt", b"some text", "text/plain"))],
        data={"auth_token": "tok-alice"},
    ),
    "history": lambda c: c.post("/history", json={"token": "tok-alice"}),
    "documents": lambda c: c.get("/documents", headers=_ALICE_BEARER),
    "delete-document": lambda c: c.delete("/me/documents/10", headers=_ALICE_BEARER),
    "put-ratings": lambda c: c.put(
        "/put_ratings", json={"id": 1, "rating": 5, "comment": ""}, headers=_ALICE_BEARER
    ),
    "submit-correction": lambda c: c.post(
        "/submit_correction", json={"chat_id": 1, "corrected_answer": "x", "comment": ""},
        headers=_ALICE_BEARER,
    ),
    "me-stats": lambda c: c.get("/me/stats", params={"token": "tok-alice"}),
}


@pytest.mark.parametrize("call", list(_SUSPENDED_USER_CALLS), ids=list(_SUSPENDED_USER_CALLS))
def test_a_token_issued_before_suspension_is_refused_on_user_routes(
    client, db, ollama_calls, followups, embedding_calls, call,
):
    before = client.get("/documents", headers=_ALICE_BEARER)
    assert before.status_code == 200 and [d["id"] for d in before.json()] == [10], (
        "the token works before the suspension"
    )
    _set_active(client, "alice-id", False)
    statements_before = len(db.statements)

    response = _SUSPENDED_USER_CALLS[call](client)

    assert response.status_code == 401, response.text
    refused = db.statements[statements_before:]
    assert [sql for sql, _ in refused if "FROM users" not in sql] == [], (
        "a suspended token must be refused at the identity check, before any other query"
    )
    assert db.history == _INITIAL_HISTORY and db.history_inserts == []
    assert db.documents == _DOCUMENTS and db.chunks == []
    assert ollama_calls == [] and followups == [] and embedding_calls == []
    assert db.unexpected == []


def test_a_suspended_users_session_endpoints_report_the_suspension(client, db):
    _set_active(client, "alice-id", False)

    assert client.get("/me", params={"token": "tok-alice"}).status_code == 403
    assert client.get("/admin/check", params={"token": "tok-alice"}).json() == {"is_admin": False}
    db.passwords["tok-alice"] = bcrypt.hashpw(b"correct horse", bcrypt.gensalt(4)).decode()
    login = client.post("/login", json={"username": "alice", "password": "correct horse"})
    assert login.status_code == 403, "a suspended account cannot sign in again either"


_ADMIN2_BEARER = {"Authorization": "Bearer tok-admin2"}

# Every admin-gated route family, with the token admin2 was issued before
# another admin suspended them.
_SUSPENDED_ADMIN_CALLS = {
    "knowledge-base": lambda c: c.get("/knowledge_base", headers=_ADMIN2_BEARER),
    "admin-overview": lambda c: c.get("/admin/overview", params={"token": "tok-admin2"}),
    "admin-users": lambda c: c.get("/admin/users", params={"token": "tok-admin2"}),
    "admin-ratings": lambda c: c.get("/admin/ratings", params={"token": "tok-admin2"}),
    "admin-rating-stats": lambda c: c.get("/admin/ratings/stats", params={"token": "tok-admin2"}),
    "document-search": lambda c: c.get("/me/documents/search", headers=_ADMIN2_BEARER),
    "document-content": lambda c: c.get("/documents/10/content", headers=_ADMIN2_BEARER),
    "suspend-a-user": lambda c: c.patch(
        "/admin/users/alice-id", json={"token": "tok-admin2", "is_active": False}
    ),
    "delete-a-user": lambda c: c.delete("/admin/users/carol-id", params={"token": "tok-admin2"}),
}


@pytest.mark.parametrize("call", list(_SUSPENDED_ADMIN_CALLS), ids=list(_SUSPENDED_ADMIN_CALLS))
def test_a_suspended_admins_token_is_refused_by_every_admin_check(client, db, call):
    assert client.get("/knowledge_base", headers=_ADMIN2_BEARER).status_code == 200, (
        "the admin token works before the suspension"
    )
    assert client.get("/admin/check", params={"token": "tok-admin2"}).json() == {"is_admin": True}
    _set_active(client, "admin2-id", False)
    statements_before = len(db.statements)

    response = _SUSPENDED_ADMIN_CALLS[call](client)

    assert response.status_code == 403, response.text
    refused = db.statements[statements_before:]
    assert [sql for sql, _ in refused if "FROM users" not in sql] == [], (
        "a suspended admin must be refused at the admin check, before any other query"
    )
    assert "alice-id" not in db.inactive and "carol-id" in {u for u, _ in db.users.values()}
    assert client.get("/admin/check", params={"token": "tok-admin2"}).json() == {"is_admin": False}
    assert db.unexpected == []


def test_reactivating_an_account_restores_its_existing_token(client, db):
    _set_active(client, "alice-id", False)
    assert client.get("/documents", headers=_ALICE_BEARER).status_code == 401
    _set_active(client, "alice-id", True)

    response = client.get("/documents", headers=_ALICE_BEARER)

    assert response.status_code == 200 and [d["id"] for d in response.json()] == [10]


# --- release containment: Google sign-in and /convert-docx -------------------


@pytest.mark.parametrize("body", [{"token": "a-google-id-token"}, {}], ids=["with-token", "no-body"])
def test_google_sign_in_is_refused_before_any_token_check(client, db, monkeypatch, main_module, body):
    verified = []
    monkeypatch.setattr(
        main_module.id_token, "verify_oauth2_token", lambda *a, **k: verified.append(a) or {}
    )

    response = client.post("/google-login", json=body)

    assert response.status_code == 403
    assert "Google sign-in is temporarily unavailable" in response.json()["detail"]
    assert verified == [], "no Google token may be verified"
    assert db.statements == [], "no account may be read or created"


def test_the_login_form_no_longer_offers_google_sign_in():
    user_auth = (
        Path(__file__).resolve().parents[2]
        / "synerge-reader-frontend" / "src" / "components" / "UserAuth" / "UserAuth.jsx"
    ).read_text(encoding="utf-8")
    assert "const GOOGLE_SIGN_IN_ENABLED = false;" in user_auth
    assert "{GOOGLE_SIGN_IN_ENABLED && GOOGLE_CLIENT_ID && (" in user_auth, (
        "the Google button must render only behind the disabled switch"
    )


@pytest.mark.parametrize(
    "filename", ["contract.docx", "../../outside.docx", "/etc/secret.docx"],
    ids=["plain-name", "traversal-name", "absolute-name"],
)
def test_convert_docx_is_refused_without_reading_or_writing_anything(client, db, filename):
    response = client.post(
        "/convert-docx",
        files=[("file", (filename, b"PK\x03\x04 not really a docx", "application/octet-stream"))],
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "DOCX conversion is disabled."}
    assert db.statements == []


def test_convert_docx_accepts_no_upload_and_cannot_spawn_a_process(main_module):
    [route] = [
        route for route in main_module.app.routes
        if getattr(route, "path", None) == "/convert-docx"
    ]
    dependant = route.dependant
    assert dependant.body_params == [] and dependant.query_params == [], (
        "the disabled route must not parse an upload, so no filename is ever used"
    )
    assert not hasattr(main_module, "subprocess"), "main.py no longer imports subprocess"
    tree = ast.parse(_MAIN_PY_PATH.read_text(encoding="utf-8"), filename=str(_MAIN_PY_PATH))
    [convert] = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "convert_docx_to_pdf"
    ]
    body = ast.Module(body=convert.body, type_ignores=[])  # the decorator is not the body
    called = {
        node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        for node in ast.walk(body) if isinstance(node, ast.Call)
    }
    assert called == {"HTTPException"}, f"the disabled route may only refuse, but calls {called}"


# --- no knowledge-base reader or automatic writer remains wired --------------


def test_no_route_or_helper_calls_a_kb_reader_or_automatic_writer():
    """Supplements the behavior tests: these helpers stay defined for the later
    user-owned knowledge base, but nothing may reference them. The admin KB
    editor uses its own endpoints and none of these helpers."""
    tree = ast.parse(_MAIN_PY_PATH.read_text(encoding="utf-8"), filename=str(_MAIN_PY_PATH))
    helpers = {
        "get_relevant_knowledge_base",
        "increment_kb_usage",
        "auto_save_to_kb",
        "generate_kb_from_document",
    }
    references = [
        (node.id, node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id in helpers
    ]
    assert references == [], f"a knowledge-base helper is wired again: {references}"


def test_this_file_contacts_no_external_service():
    this_path = Path(__file__).resolve()
    tree = ast.parse(this_path.read_text(encoding="utf-8"), filename=str(this_path))
    forbidden = {"subprocess", "docker", "requests", "psycopg2", "socket", "httpx"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert imported.isdisjoint(forbidden), sorted(imported & forbidden)
