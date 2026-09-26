"""Route-adapter contracts for POST /upload (E1b).

This file proves that the /upload route is a thin transport adapter over the
verified ``DocumentIngestionService`` from E1a-2: it reads each uploaded file's
ORIGINAL bytes exactly once, maps the form fields onto the ingestion contract,
preserves input order, composes the service with this application's real
dependencies, returns the service's batch envelope under the service's own
aggregated HTTP status, and dispatches the two pre-existing follow-ups only
through the service's post-commit hook.

REQUIRES THE SOCKETPAIR-AWARE SIBLING GUARD, for exactly the reason
``tests/test_main_route_harness.py`` documents: every test here that enters a
``TestClient`` context builds a real asyncio event loop, and on Windows that
loop builds its self-pipe with ``socket.socketpair()``. Under the audited
Phase-B guard this file FAILS rather than skips, which is intentional -- a skip
would hide a missing safety boundary.

MANDATORY PATCHING RULE. ``main._initialize_application`` is replaced before
any ``TestClient`` context is entered (here, inside the ``upload_client``
fixture, before the client is constructed). Unpatched, the real ``init_db``
reaches ``psycopg2.connect``.

NO EXTERNAL CONTACT. ``DocumentIngestionService`` is substituted with a
recording double in every request-level test, so no parse, chunk, embed, or
write ever runs; ``main.connect_to_postgres`` is replaced with a factory that
fails loudly if anything tries to open a connection, and the tests that need
an uploader lookup replace it with an in-memory fake. There is no
database, Ollama, network, subprocess, container, or production access here.

What these tests do NOT prove: they do not prove the ingestion service's own
parse/chunk/embed/persist behavior (that is ``tests/test_document_ingestion.py``),
they do not exercise pgvector adaptation (the doubles record parameters rather
than adapting them), and they do not validate any production deployment.
"""

import ast
import contextlib
import importlib
import os
from pathlib import Path
import sys

import dotenv
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from document_ingestion import (
    BatchIngestionResult,
    CommittedDocument,
    DocumentMetadata,
    ErrorCategory,
    IngestionResult,
    ResultStatus,
    UploadDocument,
)
from original_file_fixtures import make_docx, make_pdf


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_MAIN_PY_PATH = _REPOSITORY_ROOT / "synerge-reader-backend" / "main.py"
_TESTS_DIR = Path(__file__).resolve().parent

# Duplicated from tests/test_main_route_harness.py on purpose: hoisting this
# into a shared conftest.py would add collection-wide state that every test
# module inherits, which is a larger change than this unit is authorised to
# make.
_EMBEDDING_PROFILE_KEYS = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_QUERY_PREFIX",
    "EMBEDDING_DOCUMENT_PREFIX",
    "EMBEDDING_PROFILE_UNVERIFIED_ACK",
)

_PDF_BYTES = make_pdf(["Original page one", "Original page two"])
_DOCX_BYTES = make_docx(["Original docx paragraph"])
_TXT_BYTES = "Original plain text body.\nSecond line.\n".encode("utf-8")


@contextlib.contextmanager
def _imported_main():
    """Import a fresh ``main`` and restore ``sys.modules`` exactly afterwards."""
    previous = sys.modules.pop("main", None)
    try:
        yield importlib.import_module("main")
    finally:
        sys.modules.pop("main", None)
        if previous is not None:
            sys.modules["main"] = previous


@pytest.fixture
def neutral_main_environment(monkeypatch):
    """Keep these tests independent of an untracked backend .env file."""
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    for key in _EMBEDDING_PROFILE_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def main_module(neutral_main_environment):
    with _imported_main() as module:
        yield module


def _forbidden_connection_factory():
    raise AssertionError(
        "a route-adapter test attempted to open a real database connection"
    )


class _ServiceDouble:
    """Records how the route composed and called the ingestion service."""

    def __init__(self, record, batch_for_uploads, **kwargs):
        self._record = record
        self._batch_for_uploads = batch_for_uploads
        record["constructions"] += 1
        record["kwargs"] = kwargs

    def ingest_batch(self, uploads):
        materialised = list(uploads)
        record = self._record
        record["batches"] += 1
        record["uploads"] = materialised
        return self._batch_for_uploads(materialised)


def _all_indexed(uploads):
    return BatchIngestionResult(
        results=tuple(
            IngestionResult(
                filename=upload.filename or "untitled",
                status=ResultStatus.INDEXED,
                document_id=100 + index,
                chunks_count=index + 1,
            )
            for index, upload in enumerate(uploads)
        ),
        http_status=200,
    )


def _install_service_double(monkeypatch, main_module, batch_for_uploads=_all_indexed):
    record = {"constructions": 0, "batches": 0, "kwargs": None, "uploads": None}

    def factory(**kwargs):
        return _ServiceDouble(record, batch_for_uploads, **kwargs)

    monkeypatch.setattr(main_module, "DocumentIngestionService", factory)
    return record


@pytest.fixture
def upload_client(monkeypatch, main_module):
    # MANDATORY PATCHING RULE: replace the initializer BEFORE the TestClient
    # context is entered, so startup never reaches psycopg2.connect.
    monkeypatch.setattr(main_module, "_initialize_application", lambda: None)
    monkeypatch.setattr(
        main_module, "connect_to_postgres", _forbidden_connection_factory
    )
    client = TestClient(main_module.app)
    try:
        with client:
            yield client
    finally:
        client.close()


# --- 1: an empty request is still a 400, unchanged ---------------------------


def test_no_files_supplied_is_http_400(monkeypatch, main_module, upload_client):
    """An empty request costs nothing: no lookup, no connection, no service.

    The auth_token below is deliberately supplied, because the whole point is
    that a request carrying a token still must not reach the uploader lookup
    (and therefore the database) when it carries no file.
    """
    record = _install_service_double(monkeypatch, main_module)
    touched = []
    monkeypatch.setattr(
        main_module, "_resolve_uploader_id", lambda token: touched.append("lookup")
    )
    monkeypatch.setattr(
        main_module, "_build_ingestion_service", lambda: touched.append("service")
    )

    # Non-file multipart parts keep this a form request while supplying no file
    # at all, which is exactly the "no files supplied" case.
    response = upload_client.post(
        "/upload",
        files={
            "author": (None, "Ada"),
            "auth_token": (None, "a-valid-token"),
        },
    )

    assert response.status_code == 400
    assert touched == [], (
        "an empty request must be rejected before the uploader lookup and "
        f"before any ingestion service is built; reached: {touched}"
    )
    assert record["constructions"] == 0
    # The upload_client fixture points connect_to_postgres at a factory that
    # raises on any call, so a clean 400 -- rather than a 500 -- is itself proof
    # that no database connection factory was invoked.


# --- 2/3/4: the ORIGINAL bytes reach UploadDocument untouched ----------------


def _single_upload(upload_client, record, filename, content, content_type):
    response = upload_client.post(
        "/upload",
        files=[("files", (filename, content, content_type))],
    )
    assert response.status_code == 200, response.text
    uploads = record["uploads"]
    assert len(uploads) == 1
    assert isinstance(uploads[0], UploadDocument)
    return uploads[0]


def test_original_pdf_bytes_reach_upload_document_unchanged(
    monkeypatch, main_module, upload_client
):
    record = _install_service_double(monkeypatch, main_module)
    upload = _single_upload(
        upload_client, record, "contract.pdf", _PDF_BYTES, "application/pdf"
    )

    assert upload.content == _PDF_BYTES, (
        "the route must forward the original PDF bytes, not a re-encoded or "
        "client-extracted substitute"
    )
    assert upload.content[:5] == b"%PDF-", "the PDF magic bytes must survive transport"
    assert upload.filename == "contract.pdf"


def test_original_docx_bytes_reach_upload_document_unchanged(
    monkeypatch, main_module, upload_client
):
    record = _install_service_double(monkeypatch, main_module)
    upload = _single_upload(
        upload_client,
        record,
        "brief.docx",
        _DOCX_BYTES,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert upload.content == _DOCX_BYTES
    assert upload.content[:2] == b"PK", "the DOCX zip container must survive transport"
    assert upload.filename == "brief.docx"


def test_original_txt_bytes_reach_upload_document_unchanged(
    monkeypatch, main_module, upload_client
):
    record = _install_service_double(monkeypatch, main_module)
    upload = _single_upload(
        upload_client, record, "notes.txt", _TXT_BYTES, "text/plain"
    )

    assert upload.content == _TXT_BYTES
    assert upload.filename == "notes.txt"


def test_single_file_field_is_still_accepted(monkeypatch, main_module, upload_client):
    record = _install_service_double(monkeypatch, main_module)

    response = upload_client.post(
        "/upload",
        files=[("file", ("solo.txt", _TXT_BYTES, "text/plain"))],
    )

    assert response.status_code == 200, response.text
    assert [upload.filename for upload in record["uploads"]] == ["solo.txt"]
    assert record["uploads"][0].content == _TXT_BYTES


# --- 5: order and distinctness across a multi-file request ------------------


def test_multiple_files_preserve_order_and_remain_distinct(
    monkeypatch, main_module, upload_client
):
    record = _install_service_double(monkeypatch, main_module)

    response = upload_client.post(
        "/upload",
        files=[
            ("files", ("first.pdf", _PDF_BYTES, "application/pdf")),
            (
                "files",
                (
                    "second.docx",
                    _DOCX_BYTES,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ),
            ),
            ("files", ("third.txt", _TXT_BYTES, "text/plain")),
        ],
    )

    assert response.status_code == 200, response.text
    uploads = record["uploads"]
    assert [upload.filename for upload in uploads] == [
        "first.pdf",
        "second.docx",
        "third.txt",
    ], "the service must receive the files in the order the client sent them"
    assert [upload.content for upload in uploads] == [
        _PDF_BYTES,
        _DOCX_BYTES,
        _TXT_BYTES,
    ], "each file's own original bytes must stay with its own filename"
    assert len({id(upload) for upload in uploads}) == 3

    payload = response.json()
    assert [result["filename"] for result in payload["results"]] == [
        "first.pdf",
        "second.docx",
        "third.txt",
    ], "the response envelope must preserve the same order"


# --- 6: form metadata maps onto DocumentMetadata ----------------------------


def test_form_metadata_reaches_document_metadata(
    monkeypatch, main_module, upload_client
):
    record = _install_service_double(monkeypatch, main_module)

    response = upload_client.post(
        "/upload",
        files=[("files", ("meta.txt", _TXT_BYTES, "text/plain"))],
        data={
            "author": "A. Author",
            "title": "A Title",
            "publication_date": "2026-09-19",
            "source": "A Source",
            "doi_url": "https://doi.example/10.1000/xyz",
        },
    )

    assert response.status_code == 200, response.text
    metadata = record["uploads"][0].metadata
    assert isinstance(metadata, DocumentMetadata)
    assert metadata.author == "A. Author"
    assert metadata.title == "A Title"
    assert metadata.publication_date == "2026-09-19"
    assert metadata.source == "A Source"
    assert metadata.doi_url == "https://doi.example/10.1000/xyz"


def test_absent_metadata_stays_none_rather_than_empty_strings(
    monkeypatch, main_module, upload_client
):
    record = _install_service_double(monkeypatch, main_module)

    response = upload_client.post(
        "/upload",
        files=[("files", ("bare.txt", _TXT_BYTES, "text/plain"))],
    )

    assert response.status_code == 200, response.text
    assert record["uploads"][0].metadata == DocumentMetadata()


# --- 7: uploader identity -- resolved, anonymous, or refused before ingestion


# Planted in every simulated lookup failure: none of it may reach a response
# body or the server log, which carries the exception class name only.
_SEEDED_DB_DETAIL = "password=hunter2 host=db.internal SELECT id FROM users"


class _LookupFailure(Exception):
    pass


class _FakeCursor:
    def __init__(self, row, log, fail_on=None):
        self._row = row
        self._log = log
        self._fail_on = fail_on
        self.closed = False

    def execute(self, sql, params=None):
        self._log.append(("execute", " ".join(sql.split()), params))
        if self._fail_on == "execute":
            raise _LookupFailure(_SEEDED_DB_DETAIL)

    def fetchone(self):
        if self._fail_on == "fetchone":
            raise _LookupFailure(_SEEDED_DB_DETAIL)
        return self._row

    def close(self):
        self.closed = True
        self._log.append(("cursor_close", None, None))


class _FakeConnection:
    def __init__(self, row, log, fail_on=None):
        self._row = row
        self._log = log
        self._fail_on = fail_on
        self.closed = False
        self.cursors = []

    def cursor(self):
        if self._fail_on == "cursor":
            raise _LookupFailure(_SEEDED_DB_DETAIL)
        cursor = _FakeCursor(self._row, self._log, self._fail_on)
        self.cursors.append(cursor)
        return cursor

    def close(self):
        self.closed = True
        self._log.append(("connection_close", None, None))


def test_resolved_uploader_id_reaches_upload_document(
    monkeypatch, main_module, upload_client
):
    record = _install_service_double(monkeypatch, main_module)
    log = []
    connection = _FakeConnection((4242,), log)
    monkeypatch.setattr(main_module, "connect_to_postgres", lambda: connection)

    response = upload_client.post(
        "/upload",
        files=[("files", ("owned.txt", _TXT_BYTES, "text/plain"))],
        data={"auth_token": "a-valid-token"},
    )

    assert response.status_code == 200, response.text
    assert record["uploads"][0].uploader_id == 4242
    assert ("execute", "SELECT id FROM users WHERE token = %s", ("a-valid-token",)) in log


def test_uploader_lookup_always_closes_its_cursor_and_connection(
    monkeypatch, main_module, upload_client
):
    _install_service_double(monkeypatch, main_module)
    log = []
    connection = _FakeConnection((7,), log)
    monkeypatch.setattr(main_module, "connect_to_postgres", lambda: connection)

    response = upload_client.post(
        "/upload",
        files=[("files", ("owned.txt", _TXT_BYTES, "text/plain"))],
        data={"auth_token": "a-valid-token"},
    )

    assert response.status_code == 200, response.text
    assert connection.closed, "the lookup connection must always be closed"
    assert connection.cursors and all(
        cursor.closed for cursor in connection.cursors
    ), "the lookup cursor must always be closed"


def _record_followup_starts(monkeypatch, main_module):
    started = []
    monkeypatch.setattr(
        main_module,
        "_start_background_task",
        lambda target, args: started.append(target),
    )
    return started


def test_unknown_token_is_refused_with_401_before_ingestion(
    monkeypatch, main_module, upload_client, capsys
):
    """A supplied token that matches no user is a failed identity claim, not an
    anonymous upload. Refusing it before the service is built is what keeps it
    from ever becoming an ownerless document."""
    record = _install_service_double(monkeypatch, main_module)
    started = _record_followup_starts(monkeypatch, main_module)
    log = []
    connection = _FakeConnection(None, log)
    monkeypatch.setattr(main_module, "connect_to_postgres", lambda: connection)
    capsys.readouterr()

    response = upload_client.post(
        "/upload",
        files=[("files", ("anon.txt", _TXT_BYTES, "text/plain"))],
        data={"auth_token": "not-a-real-token"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid session"}
    assert record["constructions"] == 0 and record["batches"] == 0, (
        "an unknown token must be refused before any ingestion service is "
        "built, so no ownerless document can be written"
    )
    assert started == [], "no upload follow-up may start for a refused upload"
    assert ("execute", "SELECT id FROM users WHERE token = %s", ("not-a-real-token",)) in log
    assert connection.closed
    assert connection.cursors and all(cursor.closed for cursor in connection.cursors)
    assert "not-a-real-token" not in response.text
    assert "not-a-real-token" not in "".join(capsys.readouterr())


_LOOKUP_FAILURES = ("connect_raises", "connect_returns_none", "cursor", "execute", "fetchone")


@pytest.mark.parametrize("failure", _LOOKUP_FAILURES)
def test_lookup_failure_is_a_generic_503_and_never_reaches_ingestion(
    monkeypatch, main_module, upload_client, capsys, failure
):
    """A lookup that cannot complete must not guess an owner.

    The fake user row exists, so a lookup that fell through would have a real
    owner to find; the refusal comes from the failed step alone. The response
    and the log carry neither the token nor the database exception text.
    """
    record = _install_service_double(monkeypatch, main_module)
    started = _record_followup_starts(monkeypatch, main_module)
    connection = None
    if failure == "connect_raises":
        def connect():
            raise _LookupFailure(_SEEDED_DB_DETAIL)
    elif failure == "connect_returns_none":
        def connect():
            return None
    else:
        connection = _FakeConnection((4242,), [], fail_on=failure)

        def connect():
            return connection
    monkeypatch.setattr(main_module, "connect_to_postgres", connect)
    capsys.readouterr()

    response = upload_client.post(
        "/upload",
        files=[("files", ("owned.txt", _TXT_BYTES, "text/plain"))],
        data={"auth_token": "a-valid-token"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Could not verify your session. Please try again shortly."
    }
    assert record["constructions"] == 0 and record["batches"] == 0, (
        f"a lookup failure at {failure!r} must not reach ingest_batch"
    )
    assert started == [], "no upload follow-up may start for a refused upload"
    if connection is not None:
        assert connection.closed, "an opened lookup connection must be closed"
        assert all(cursor.closed for cursor in connection.cursors)
    captured = "".join(capsys.readouterr())
    for private in ("a-valid-token", _SEEDED_DB_DETAIL):
        assert private not in response.text
        assert private not in captured


@pytest.mark.parametrize(
    ("row", "fail_on", "status"),
    [(None, None, 401), ((4242,), "execute", 503)],
    ids=["unknown-token", "query-failure"],
)
def test_refusals_carry_no_underlying_exception(
    monkeypatch, main_module, row, fail_on, status
):
    connection = _FakeConnection(row, [], fail_on=fail_on)
    monkeypatch.setattr(main_module, "connect_to_postgres", lambda: connection)

    with pytest.raises(main_module.HTTPException) as refused:
        main_module._resolve_uploader_id("a-token")

    assert refused.value.status_code == status
    assert refused.value.__cause__ is None and refused.value.__context__ is None, (
        "the refusal must not chain the database exception"
    )
    assert "a-token" not in str(refused.value.detail)
    assert _SEEDED_DB_DETAIL not in str(refused.value.detail)


@pytest.mark.parametrize("form", [{}, {"auth_token": ""}], ids=["no-field", "empty-field"])
def test_upload_without_token_never_opens_a_lookup_connection(
    monkeypatch, main_module, upload_client, form
):
    record = _install_service_double(monkeypatch, main_module)

    # The fixture's connect_to_postgres raises on any call, so a 200 here is
    # itself the proof that no lookup connection was opened.
    response = upload_client.post(
        "/upload",
        files=[("files", ("anon.txt", _TXT_BYTES, "text/plain"))],
        data=form,
    )

    assert response.status_code == 200, response.text
    assert record["batches"] == 1
    assert record["uploads"][0].uploader_id is None, (
        "no token keeps the existing anonymous upload"
    )


# --- 8: the route composes the REAL service with this app's dependencies ----


def test_route_composes_service_with_expected_dependencies(
    monkeypatch, main_module, upload_client
):
    record = _install_service_double(monkeypatch, main_module)

    response = upload_client.post(
        "/upload",
        files=[("files", ("wired.txt", _TXT_BYTES, "text/plain"))],
    )

    assert response.status_code == 200, response.text
    assert record["constructions"] == 1
    kwargs = record["kwargs"]
    assert set(kwargs) == {
        "connection_factory",
        "embedding_provider",
        "embedding_profile",
        "dispatch_after_commit",
    }, f"unexpected ingestion service composition: {sorted(kwargs)}"
    assert kwargs["connection_factory"] is main_module.connect_to_postgres, (
        "the service must receive connect_to_postgres itself -- it is the only "
        "factory that applies pgvector's register_vector() to the connection"
    )
    assert kwargs["embedding_provider"] is main_module._EMBEDDING_PROVIDER
    assert kwargs["embedding_profile"] is main_module._EMBEDDING_PROFILE
    assert kwargs["dispatch_after_commit"] is main_module._dispatch_upload_followups


@pytest.fixture(scope="module")
def main_source():
    return _MAIN_PY_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def main_tree(main_source):
    return ast.parse(main_source, filename=str(_MAIN_PY_PATH))


def _find_function(node, name):
    for candidate in ast.walk(node):
        if (
            isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
            and candidate.name == name
        ):
            return candidate
    raise AssertionError(f"function {name!r} not found")


def _calls_to_name(node, func_name):
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == func_name
    ]


def test_build_ingestion_service_binds_the_expected_names(main_tree):
    fn = _find_function(main_tree, "_build_ingestion_service")
    calls = _calls_to_name(fn, "DocumentIngestionService")
    assert len(calls) == 1, (
        "_build_ingestion_service must construct DocumentIngestionService exactly once"
    )
    call = calls[0]
    assert not call.args, "the service must be composed with keywords only"
    kwargs = {kw.arg: kw.value for kw in call.keywords}
    assert set(kwargs) == {
        "connection_factory",
        "embedding_provider",
        "embedding_profile",
        "dispatch_after_commit",
    }
    expected = {
        "connection_factory": "connect_to_postgres",
        "embedding_provider": "_EMBEDDING_PROVIDER",
        "embedding_profile": "_EMBEDDING_PROFILE",
        "dispatch_after_commit": "_dispatch_upload_followups",
    }
    for keyword, name in expected.items():
        value = kwargs[keyword]
        assert isinstance(value, ast.Name) and value.id == name, (
            f"{keyword}= must be the bare name {name}"
        )


def test_upload_route_opens_no_connection_of_its_own(main_tree):
    fn = _find_function(main_tree, "upload_documents")
    assert not _calls_to_name(fn, "connect_to_postgres"), (
        "upload_documents must not open a connection directly; the ingestion "
        "write connection belongs to the service's connection factory and the "
        "auth lookup belongs to _resolve_uploader_id"
    )
    for banned in ("psycopg2", "register_vector"):
        assert banned not in ast.dump(fn), (
            f"the upload route must not build its own {banned} path"
        )


def test_upload_route_delegates_parse_chunk_embed_and_persist(main_tree):
    fn = _find_function(main_tree, "upload_documents")
    dumped = ast.dump(fn)
    for banned in (
        "extract_text_from_upload",
        "chunk_document",
        "build_chunk_locator",
        "embed_documents",
        "INSERT INTO documents",
        "INSERT INTO document_chunks",
    ):
        assert banned not in dumped, (
            f"upload_documents must delegate to DocumentIngestionService, but it "
            f"still references {banned!r}"
        )
    assert len(_calls_to_name(fn, "_build_ingestion_service")) == 1


def test_upload_route_reads_each_upload_exactly_once(main_tree):
    fn = _find_function(main_tree, "upload_documents")
    read_calls = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read"
    ]
    assert len(read_calls) == 1, (
        "an UploadFile stream cannot be replayed, so the route must contain "
        f"exactly one .read() call; found {len(read_calls)}"
    )


def test_uploader_lookup_helper_uses_connect_to_postgres(main_tree):
    fn = _find_function(main_tree, "_resolve_uploader_id")
    assert len(_calls_to_name(fn, "connect_to_postgres")) == 1, (
        "the preserved auth lookup must still open its connection via "
        "connect_to_postgres()"
    )
    finally_closes = False
    for try_node in ast.walk(fn):
        if isinstance(try_node, ast.Try) and try_node.finalbody:
            dumped = ast.dump(ast.Module(body=try_node.finalbody, type_ignores=[]))
            if "'close'" in dumped:
                finally_closes = True
    assert finally_closes, (
        "the lookup must close its cursor and connection from a finally block"
    )


# --- 9: the service's aggregated status becomes the real HTTP status --------


def _mixed_batch(http_status):
    if http_status == 200:
        results = (
            IngestionResult(
                filename="a.txt",
                status=ResultStatus.INDEXED,
                document_id=1,
                chunks_count=2,
            ),
        )
    elif http_status == 207:
        results = (
            IngestionResult(
                filename="a.txt",
                status=ResultStatus.INDEXED,
                document_id=1,
                chunks_count=2,
            ),
            IngestionResult(
                filename="b.txt",
                status=ResultStatus.REJECTED,
                error_category=ErrorCategory.EMPTY_FILE,
                error_message="The uploaded file is empty.",
            ),
        )
    elif http_status == 422:
        results = (
            IngestionResult(
                filename="b.txt",
                status=ResultStatus.REJECTED,
                error_category=ErrorCategory.UNSUPPORTED_FILE_TYPE,
                error_message="Upload a PDF, DOCX, or plain text file.",
            ),
        )
    else:
        results = (
            IngestionResult(
                filename="c.txt",
                status=ResultStatus.FAILED,
                error_category=ErrorCategory.DATABASE_UNAVAILABLE,
                error_message="Document storage is temporarily unavailable.",
            ),
        )
    return BatchIngestionResult(results=results, http_status=http_status)


@pytest.mark.parametrize("http_status", [200, 207, 422, 503])
def test_batch_http_status_becomes_the_response_status(
    monkeypatch, main_module, upload_client, http_status
):
    _install_service_double(
        monkeypatch, main_module, lambda uploads: _mixed_batch(http_status)
    )

    response = upload_client.post(
        "/upload",
        files=[("files", ("a.txt", _TXT_BYTES, "text/plain"))],
    )

    assert response.status_code == http_status, (
        "BatchIngestionResult.http_status must be the actual HTTP status, not a "
        f"field buried in a 200 body; got {response.status_code}"
    )
    assert response.json()["http_status"] == http_status


def test_mixed_batch_reports_a_committed_sibling_alongside_a_failure(
    monkeypatch, main_module, upload_client
):
    _install_service_double(monkeypatch, main_module, lambda uploads: _mixed_batch(207))

    response = upload_client.post(
        "/upload",
        files=[
            ("files", ("a.txt", _TXT_BYTES, "text/plain")),
            ("files", ("b.txt", b"", "text/plain")),
        ],
    )

    assert response.status_code == 207
    payload = response.json()
    indexed = [r for r in payload["results"] if r["status"] == "indexed"]
    assert len(indexed) == 1, (
        "a sibling's failure must not erase an already committed document"
    )
    assert indexed[0]["document_id"] == 1
    assert payload["summary"] == {
        "total": 2,
        "indexed": 1,
        "rejected": 1,
        "failed": 0,
    }


# --- 10: the complete envelope, with no raw exception detail ----------------


def test_complete_batch_envelope_is_returned_without_raw_exception_detail(
    monkeypatch, main_module, upload_client
):
    _install_service_double(monkeypatch, main_module, lambda uploads: _mixed_batch(503))

    response = upload_client.post(
        "/upload",
        files=[("files", ("c.txt", _TXT_BYTES, "text/plain"))],
    )

    assert response.status_code == 503
    payload = response.json()
    assert set(payload) == {"results", "summary", "http_status"}
    assert set(payload["summary"]) == {"total", "indexed", "rejected", "failed"}
    assert payload["summary"] == {"total": 1, "indexed": 0, "rejected": 0, "failed": 1}

    result = payload["results"][0]
    for key in (
        "filename",
        "document_id",
        "status",
        "chunks_count",
        "warnings",
        "error_category",
        "error_message",
        "ingestion_contract_version",
    ):
        assert key in result, f"the per-file envelope must carry {key!r}"
    assert result["error_message"] == "Document storage is temporarily unavailable."

    body = response.text
    for leak in ("Traceback", "psycopg2", "DB_CONNECTION_STRING", "SELECT ", "INSERT "):
        assert leak not in body, f"the response must not expose {leak!r}"


# --- 11: post-commit follow-ups -------------------------------------------


def test_dispatcher_forwards_committed_identity_to_both_followups(
    monkeypatch, main_module
):
    started = []
    monkeypatch.setattr(
        main_module,
        "_start_background_task",
        lambda target, args: started.append((target, args)),
    )

    committed = CommittedDocument(
        document_id=77,
        filename="sanitized_name.pdf",
        text="server-extracted text",
        content_sha256="ab" * 32,
        embedding_profile_id="profile-1",
    )
    main_module._dispatch_upload_followups(committed)

    assert [target for target, _ in started] == [
        main_module.generate_kb_from_document,
        main_module._extract_document_insights,
    ], "both pre-existing follow-ups must be preserved"
    for _, args in started:
        assert args == (77, "sanitized_name.pdf", "server-extracted text"), (
            "each follow-up must receive the committed document id, the "
            "service-sanitized filename, and the server-extracted text"
        )


def test_failed_follow_up_starts_surface_after_both_attempts(
    monkeypatch, main_module, capsys
):
    """A follow-up that never started must be visible to the service.

    _start_background_task must let a Thread construction/start failure
    propagate, and the dispatcher must still attempt BOTH follow-ups before
    raising one generic error carrying none of the underlying detail. The
    service then converts that error into its fixed post-commit warning while
    keeping the result indexed -- proven separately by
    test_dispatch_failure_warns_but_does_not_falsify_committed_state in
    tests/test_document_ingestion.py, which this test does not duplicate.

    Only a failed START is detectable here. A failure inside an already-running
    follow-up thread happens after this function has returned and cannot be
    observed synchronously by anything in this path.
    """
    seeded_private_detail = "token=hunter2 /var/secret/doc.pdf"
    attempts = []

    class _UnstartableThread:
        def __init__(self, target=None, args=(), daemon=None):
            attempts.append((target, args))
            raise RuntimeError(f"cannot start thread: {seeded_private_detail}")

    monkeypatch.setattr("threading.Thread", _UnstartableThread)
    capsys.readouterr()

    # 1: the start failure propagates out of _start_background_task itself.
    with pytest.raises(RuntimeError):
        main_module._start_background_task(lambda: None, ())
    assert len(attempts) == 1
    attempts.clear()

    committed = CommittedDocument(
        document_id=77,
        filename="sanitized_name.pdf",
        text="server-extracted text",
        content_sha256="ab" * 32,
        embedding_profile_id="profile-1",
    )

    # 2/3: both follow-ups are attempted, and the raise comes only afterwards.
    with pytest.raises(main_module._PostCommitDispatchError) as excinfo:
        main_module._dispatch_upload_followups(committed)

    assert [target for target, _ in attempts] == [
        main_module.generate_kb_from_document,
        main_module._extract_document_insights,
    ], (
        "the second follow-up must still be attempted when the first one "
        "cannot be started, and both must keep their committed arguments"
    )
    for _, args in attempts:
        assert args == (77, "sanitized_name.pdf", "server-extracted text")
    assert len(attempts) == 2, (
        "the dispatcher must raise only after both attempts, not on the first "
        f"failure; recorded {len(attempts)} attempt(s)"
    )

    # 4: none of the seeded private detail escapes, by message or by chaining.
    assert seeded_private_detail not in str(excinfo.value)
    assert excinfo.value.__cause__ is None and excinfo.value.__context__ is None, (
        "the generic dispatch error must not carry the original exception"
    )
    captured = capsys.readouterr()
    assert seeded_private_detail not in captured.out
    assert seeded_private_detail not in captured.err


def test_route_never_calls_the_followups_directly(main_tree):
    fn = _find_function(main_tree, "upload_documents")
    dumped = ast.dump(fn)
    for name in (
        "generate_kb_from_document",
        "_extract_document_insights",
        "_start_background_task",
        "Thread",
    ):
        assert name not in dumped, (
            "the route must not dispatch follow-ups itself -- they run only "
            f"through the service's post-commit hook, but it references {name!r}"
        )
    dispatcher = _find_function(main_tree, "_dispatch_upload_followups")
    dispatch_dump = ast.dump(dispatcher)
    assert "generate_kb_from_document" in dispatch_dump
    assert "_extract_document_insights" in dispatch_dump


def test_no_followup_is_started_when_the_service_reports_a_failure(
    monkeypatch, main_module, upload_client
):
    started = []
    monkeypatch.setattr(
        main_module,
        "_start_background_task",
        lambda target, args: started.append(target),
    )
    _install_service_double(monkeypatch, main_module, lambda uploads: _mixed_batch(503))

    response = upload_client.post(
        "/upload",
        files=[("files", ("c.txt", _TXT_BYTES, "text/plain"))],
    )

    assert response.status_code == 503
    assert started == [], (
        "post-commit work must be reachable only through the service's "
        "post-commit hook, which never fires for an uncommitted document"
    )


def test_route_logs_no_document_content(main_tree):
    fn = _find_function(main_tree, "upload_documents")
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            for arg in node.args:
                dumped = ast.dump(arg)
                for banned in ("content", "text", "auth_token", "embedding"):
                    assert f"id='{banned}'" not in dumped, (
                        f"the upload route must never log {banned!r}"
                    )


# --- 12: the TestClient harness rule stays enforced -------------------------


_HARNESS_FILES = (
    _TESTS_DIR / "test_main_route_harness.py",
    Path(__file__).resolve(),
)


@pytest.mark.parametrize(
    "harness_path", _HARNESS_FILES, ids=lambda path: Path(path).name
)
def test_testclient_users_patch_initialize_application_first(harness_path):
    source = harness_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(harness_path))

    checked = 0
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        client_calls = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "TestClient"
        ]
        if not client_calls:
            continue
        patch_lines = [
            node.lineno
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setattr"
            and any(
                isinstance(arg, ast.Constant)
                and arg.value in ("_initialize_application", "init_db")
                for arg in node.args
            )
        ]
        assert patch_lines, (
            f"{harness_path.name}:{fn.name} constructs a TestClient without "
            "patching _initialize_application or init_db first"
        )
        assert min(patch_lines) < min(call.lineno for call in client_calls), (
            f"{harness_path.name}:{fn.name} must patch the initializer BEFORE "
            "the TestClient is constructed and entered"
        )
        checked += 1

    assert checked > 0, f"no TestClient user found in {harness_path.name}"


def test_this_file_contacts_no_external_service():
    this_path = Path(__file__).resolve()
    tree = ast.parse(this_path.read_text(encoding="utf-8"), filename=str(this_path))

    forbidden_roots = {"subprocess", "docker", "requests", "psycopg2", "socket"}
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint(forbidden_roots), (
        f"unexpected external-service import(s): "
        f"{sorted(imported_roots & forbidden_roots)}"
    )
