import datetime as dt
import hashlib
import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import document_parser
from document_chunker import DocumentChunk
from document_ingestion import (
    BatchIngestionResult,
    CHUNKER_CONTRACT_VERSION,
    INGESTION_CONTRACT_VERSION,
    PARSER_CONTRACT_VERSION,
    DocumentIngestionService,
    DocumentMetadata,
    ErrorCategory,
    ResultStatus,
    UploadDocument,
)
from document_parser import (
    ExtractionError,
    ParsedDocument,
    ParsedPage,
    UnsupportedFileTypeError,
)
from ollama_embedding_provider import EmbeddingProviderError
from rag_model_profiles import EmbeddingProfile
from original_file_fixtures import make_docx, make_pdf


FIXED_NOW = dt.datetime(2026, 9, 17, 12, 0, tzinfo=dt.timezone.utc)
PROFILE = EmbeddingProfile(
    provider="ollama",
    model="test-embedding-model",
    dimension=3,
    query_prefix="query: ",
    document_prefix="passage: ",
)

# Every ExtractionError document_parser can raise, and the ingestion category
# each one must resolve to. Row fields, in order:
#   0 id              short, bounded parametrize id
#   1 source_template text as it literally appears in document_parser source
#   2 rendered        the user_message the parser actually raises
#   3 http_status     the status the parser attaches
#   4 exception_class the class the parser instantiates
#   5 expected_category  the ErrorCategory _category_for_extraction_error yields
# For the ten non-f-string messages source_template and rendered are identical;
# only the page-limit row differs, keeping the literal {MAX_PDF_PAGES}
# placeholder so the source guard below can find it verbatim.
PARSER_ERROR_CONTRACT = (
    (
        "unsupported_file_type",
        "Unsupported file type. Please upload a PDF, DOCX, or plain text file.",
        "Unsupported file type. Please upload a PDF, DOCX, or plain text file.",
        415,
        UnsupportedFileTypeError,
        ErrorCategory.UNSUPPORTED_FILE_TYPE,
    ),
    (
        "zip_safety",
        "File is too large or complex to process.",
        "File is too large or complex to process.",
        422,
        ExtractionError,
        # The zip-safety message matches no keyword in
        # _category_for_extraction_error, so it classifies as
        # EXTRACTION_FAILED rather than FILE_TOO_LARGE. It stays an
        # input-scope 422, so per-file status and batch aggregation are
        # unaffected; the durable fix is a status or enum signal from the
        # held document_parser.py, not a keyword added here.
        ErrorCategory.EXTRACTION_FAILED,
    ),
    (
        "pdf_import_missing",
        "PDF processing is not available on the server.",
        "PDF processing is not available on the server.",
        500,
        ExtractionError,
        ErrorCategory.PARSER_UNAVAILABLE,
    ),
    (
        "page_limit",
        "PDF exceeds the {MAX_PDF_PAGES}-page limit. Please upload a shorter document.",
        f"PDF exceeds the {document_parser.MAX_PDF_PAGES}-page limit. "
        "Please upload a shorter document.",
        422,
        ExtractionError,
        ErrorCategory.PAGE_LIMIT_EXCEEDED,
    ),
    (
        "scanned_pdf",
        "This PDF appears to be scanned or image-based. Text extraction is not supported for image-only PDFs.",
        "This PDF appears to be scanned or image-based. Text extraction is not supported for image-only PDFs.",
        422,
        ExtractionError,
        ErrorCategory.IMAGE_ONLY_PDF,
    ),
    (
        "docx_import_missing",
        "DOCX processing is not available on the server.",
        "DOCX processing is not available on the server.",
        500,
        ExtractionError,
        ErrorCategory.PARSER_UNAVAILABLE,
    ),
    (
        "docx_empty",
        "This Word document appears to be empty or contains only images.",
        "This Word document appears to be empty or contains only images.",
        422,
        ExtractionError,
        ErrorCategory.EMPTY_DOCUMENT,
    ),
    (
        "text_empty",
        "Uploaded text file is empty.",
        "Uploaded text file is empty.",
        422,
        ExtractionError,
        ErrorCategory.EMPTY_DOCUMENT,
    ),
    (
        "file_empty",
        "Uploaded file is empty.",
        "Uploaded file is empty.",
        422,
        ExtractionError,
        ErrorCategory.EMPTY_DOCUMENT,
    ),
    (
        "file_too_large",
        "File exceeds the 50 MB size limit.",
        "File exceeds the 50 MB size limit.",
        413,
        ExtractionError,
        ErrorCategory.FILE_TOO_LARGE,
    ),
    (
        "corrupt_file",
        "Failed to extract text from this file. The file may be corrupted.",
        "Failed to extract text from this file. The file may be corrupted.",
        422,
        ExtractionError,
        ErrorCategory.EXTRACTION_FAILED,
    ),
)


class RecordingProvider:
    def __init__(self, events=None, response=None, error=None):
        self.events = events if events is not None else []
        self.response = response
        self.error = error

    def embed_documents(self, texts):
        self.events.append(("embed", list(texts)))
        if self.error is not None:
            raise self.error
        if callable(self.response):
            return self.response(texts)
        if self.response is not None:
            return self.response
        return [[1.0, 2.0, 3.0] for _ in texts]


class RecordingCursor:
    def __init__(self, connection):
        self.connection = connection
        self.calls = []
        self.chunk_inserts = 0
        self.closed = False

    def execute(self, sql, params):
        normalized = " ".join(sql.split())
        self.calls.append((normalized, params))
        if "INSERT INTO documents" in normalized:
            self.connection.events.append("document_insert")
            if self.connection.fail_phase == "document_insert":
                raise RuntimeError("private document insert detail")
        elif "INSERT INTO document_chunks" in normalized:
            self.chunk_inserts += 1
            self.connection.events.append("chunk_insert")
            if self.connection.fail_phase == "chunk_insert":
                raise RuntimeError("private chunk insert detail")

    def fetchone(self):
        self.connection.events.append("fetchone")
        if self.connection.fail_phase == "fetchone":
            return None
        return (self.connection.document_id,)

    def close(self):
        self.closed = True
        self.connection.events.append("cursor_close")
        if self.connection.cursor_close_fails:
            raise RuntimeError("private cursor close detail")


class RecordingConnection:
    def __init__(
        self,
        events=None,
        *,
        document_id=41,
        fail_phase=None,
        rollback_fails=False,
        cursor_close_fails=False,
        connection_close_fails=False,
        autocommit=False,
    ):
        self.events = events if events is not None else []
        self.document_id = document_id
        self.fail_phase = fail_phase
        self.rollback_fails = rollback_fails
        self.cursor_close_fails = cursor_close_fails
        self.connection_close_fails = connection_close_fails
        self.autocommit = autocommit
        self.cursor_object = RecordingCursor(self)
        self.commit_count = 0
        self.rollback_count = 0
        self.closed = False

    def cursor(self):
        self.events.append("cursor")
        if self.fail_phase == "cursor":
            raise RuntimeError("private cursor detail")
        return self.cursor_object

    def commit(self):
        self.events.append("commit")
        self.commit_count += 1
        if self.fail_phase == "commit":
            raise RuntimeError("private commit detail")

    def rollback(self):
        self.events.append("rollback")
        self.rollback_count += 1
        if self.rollback_fails:
            raise RuntimeError("private rollback detail")

    def close(self):
        self.events.append("connection_close")
        self.closed = True
        if self.connection_close_fails:
            raise RuntimeError("private connection close detail")


class RecordingFactory:
    def __init__(self, connections=None, events=None, *, returns_none=False, error=None):
        self.connections = list(connections or [])
        self.events = events if events is not None else []
        self.returns_none = returns_none
        self.error = error
        self.calls = 0

    def __call__(self):
        self.events.append("connection_factory")
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.returns_none:
            return None
        if self.connections:
            return self.connections.pop(0)
        return RecordingConnection(self.events, document_id=40 + self.calls)


def make_service(
    *,
    provider=None,
    connection=None,
    factory=None,
    events=None,
    dispatch=None,
    extractor=None,
    chunker=None,
):
    shared_events = events if events is not None else []
    provider = provider or RecordingProvider(shared_events)
    if factory is None:
        connection = connection or RecordingConnection(shared_events)
        factory = RecordingFactory([connection], shared_events)
    kwargs = {}
    if extractor is not None:
        kwargs["extractor"] = extractor
    if chunker is not None:
        kwargs["chunker"] = chunker
    return DocumentIngestionService(
        connection_factory=factory,
        embedding_provider=provider,
        embedding_profile=PROFILE,
        dispatch_after_commit=dispatch,
        clock=lambda: FIXED_NOW,
        json_adapter=lambda value: value,
        **kwargs,
    )


def test_original_txt_success_is_truthful_and_atomic():
    events = []
    connection = RecordingConnection(events, document_id=73)
    metadata = DocumentMetadata(
        author="Ada",
        title="Notes",
        publication_date="2026-09-17",
        source="Local",
        doi_url="https://example.test/notes",
    )
    service = make_service(connection=connection, events=events)

    result = service.ingest(
        UploadDocument("notes.txt", b"alpha beta gamma", metadata, uploader_id="user-1")
    )

    assert result.status is ResultStatus.INDEXED
    assert result.document_id == 73
    assert result.chunks_count == 1
    assert result.source_page_count is None
    assert result.indexed_page_count is None
    assert result.locator_coverage.to_dict() == {
        "locator_type": "text",
        "chunks_total": 1,
        "chunks_with_locator": 0,
        "source_pages_total": None,
        "source_pages_indexed": None,
        "page_numbers_indexed": [],
    }
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.closed is True
    assert connection.cursor_object.closed is True

    document_params = connection.cursor_object.calls[0][1]
    assert document_params == (
        "notes.txt",
        FIXED_NOW.isoformat(),
        "alpha beta gamma",
        "Ada",
        "Notes",
        "2026-09-17",
        "Local",
        "https://example.test/notes",
        "user-1",
    )
    chunk_params = connection.cursor_object.calls[1][1]
    assert chunk_params[0:6] == (73, "alpha beta gamma", 0, [1.0, 2.0, 3.0], None, None)
    assert chunk_params[6] == {"locator_type": "text"}


def test_multi_chunk_document_persists_every_chunk_in_insertion_order():
    # Comfortably past the chunker's 500-character default so the document
    # spans several chunks without pinning an exact chunk count.
    content = " ".join(f"word{index:04d}" for index in range(400)).encode("utf-8")
    connection = RecordingConnection()
    result = make_service(connection=connection).ingest(
        UploadDocument("long.txt", content)
    )

    assert result.status is ResultStatus.INDEXED
    assert result.chunks_count > 1
    assert connection.cursor_object.chunk_inserts == result.chunks_count

    calls = connection.cursor_object.calls
    document_positions = [
        position
        for position, (sql, _params) in enumerate(calls)
        if "INSERT INTO documents" in sql
    ]
    chunk_calls = [
        (position, params)
        for position, (sql, params) in enumerate(calls)
        if "INSERT INTO document_chunks" in sql
    ]

    assert len(document_positions) == 1
    assert len(chunk_calls) == result.chunks_count
    assert document_positions[0] < min(position for position, _ in chunk_calls)
    assert [params[2] for _, params in chunk_calls] == list(range(result.chunks_count))
    assert all(params[0] == result.document_id for _, params in chunk_calls)


def test_original_pdf_preserves_true_page_gap_and_coverage():
    pytest.importorskip("pdfplumber")
    connection = RecordingConnection()
    result = make_service(connection=connection).ingest(
        UploadDocument("source.pdf", make_pdf(["page one", None, "page three"]))
    )

    assert result.status is ResultStatus.INDEXED
    assert result.source_page_count == 3
    assert result.indexed_page_count == 2
    assert result.locator_coverage.page_numbers_indexed == (1, 3)
    assert result.locator_coverage.chunks_with_locator == result.chunks_count
    chunk_locator = connection.cursor_object.calls[1][1][6]
    assert chunk_locator == {"locator_type": "pdf_pages", "page_numbers": [1, 3]}


def test_original_docx_has_no_invented_pages():
    pytest.importorskip("docx")
    result = make_service().ingest(
        UploadDocument("source.docx", make_docx(["first paragraph", "second paragraph"]))
    )

    assert result.status is ResultStatus.INDEXED
    assert result.source_page_count is None
    assert result.indexed_page_count is None
    assert result.locator_coverage.locator_type == "docx"
    assert result.locator_coverage.page_numbers_indexed == ()


def test_pdf_without_extension_is_accepted_from_original_bytes():
    pytest.importorskip("pdfplumber")
    result = make_service().ingest(UploadDocument("report", make_pdf(["body"])))
    assert result.status is ResultStatus.INDEXED
    assert result.locator_coverage.locator_type == "pdf_pages"


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("wrong.txt", make_pdf(["this is a PDF"])),
        ("wrong.pdf", b"this is plain text"),
    ],
    ids=["pdf-bytes-named-txt", "text-bytes-named-pdf"],
)
def test_declared_extension_content_mismatch_is_rejected(filename, content):
    pytest.importorskip("pdfplumber")
    factory = RecordingFactory()
    result = make_service(factory=factory).ingest(UploadDocument(filename, content))
    assert result.status is ResultStatus.REJECTED
    assert result.error_category is ErrorCategory.EXTENSION_CONTENT_MISMATCH
    assert factory.calls == 0


def test_unknown_extension_is_rejected_even_for_textlike_content():
    factory = RecordingFactory()
    result = make_service(factory=factory).ingest(
        UploadDocument("notes.md", b"plain text")
    )
    assert result.error_category is ErrorCategory.UNSUPPORTED_FILE_TYPE
    assert factory.calls == 0


def test_unsupported_binary_and_empty_file_have_stable_categories():
    service = make_service(factory=RecordingFactory())
    binary = service.ingest(UploadDocument("image.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20))
    empty = service.ingest(UploadDocument("empty.txt", b""))
    assert binary.error_category is ErrorCategory.UNSUPPORTED_FILE_TYPE
    assert empty.error_category is ErrorCategory.EMPTY_FILE
    assert binary.status is ResultStatus.REJECTED
    assert empty.status is ResultStatus.REJECTED


def test_invalid_content_object_is_rejected_without_hashing_or_io():
    factory = RecordingFactory()
    result = make_service(factory=factory).ingest(
        UploadDocument("notes.txt", "not bytes")  # type: ignore[arg-type]
    )
    assert result.error_category is ErrorCategory.INVALID_UPLOAD
    assert result.content_sha256 is None
    assert factory.calls == 0


def test_corrupt_pdf_is_an_extraction_failure():
    pytest.importorskip("pdfplumber")
    result = make_service(factory=RecordingFactory()).ingest(
        UploadDocument("broken.pdf", b"%PDF-1.4\nnot a valid PDF")
    )
    assert result.error_category is ErrorCategory.EXTRACTION_FAILED
    assert result.status is ResultStatus.REJECTED


def test_image_only_pdf_is_rejected_visibly():
    pytest.importorskip("pdfplumber")
    result = make_service(factory=RecordingFactory()).ingest(
        UploadDocument("scan.pdf", make_pdf([None, None]))
    )
    assert result.error_category is ErrorCategory.IMAGE_ONLY_PDF
    assert "image-only" in result.error_message


@pytest.mark.parametrize(
    ("rendered", "http_status", "exception_class", "category"),
    [row[2:] for row in PARSER_ERROR_CONTRACT],
    ids=[row[0] for row in PARSER_ERROR_CONTRACT],
)
def test_extraction_limits_and_dependency_failure_are_stably_classified(
    rendered, http_status, exception_class, category
):
    if exception_class is UnsupportedFileTypeError:
        # UnsupportedFileTypeError builds its own 415 message and accepts only
        # an optional detail argument, so it is constructed without one rather
        # than as ExtractionError(message, 415).
        error = UnsupportedFileTypeError()
        assert error.user_message == rendered
        assert error.http_status == http_status
    else:
        error = exception_class(rendered, http_status)

    def rejecting_extractor(filename, content):
        raise error

    result = make_service(
        factory=RecordingFactory(), extractor=rejecting_extractor
    ).ingest(UploadDocument("source.pdf", b"nonempty"))
    assert result.error_category is category


def test_parser_error_contract_matches_document_parser_source():
    source = inspect.getsource(document_parser)
    assert document_parser.MAX_PDF_PAGES == 150
    for case_id, source_template, _rendered, _status, _cls, _cat in PARSER_ERROR_CONTRACT:
        assert source_template in source, case_id


def test_safe_diagnostics_name_category_and_type_without_private_detail(capsys):
    seed = "PRIVATE-SEED-7f3a"

    def rejecting_extractor(filename, content):
        raise ExtractionError(f"{seed}: uploaded text file is empty.", 422)

    rejected = make_service(
        factory=RecordingFactory(), extractor=rejecting_extractor
    ).ingest(UploadDocument("diagnostic.txt", b"alpha"))
    rejected_output = capsys.readouterr().out

    assert "ExtractionError" in rejected_output
    assert ErrorCategory.EMPTY_DOCUMENT.value in rejected_output
    assert "diagnostic.txt" in rejected_output
    assert seed not in rejected_output
    assert rejected.error_message == "The document contains no indexable text."
    assert seed not in rejected.error_message

    failed = make_service(
        factory=RecordingFactory(error=RuntimeError(f"{seed} connection detail"))
    ).ingest(UploadDocument("diagnostic.txt", b"alpha beta"))
    failed_output = capsys.readouterr().out

    assert "RuntimeError" in failed_output
    assert ErrorCategory.DATABASE_UNAVAILABLE.value in failed_output
    assert "diagnostic.txt" in failed_output
    assert seed not in failed_output
    assert failed.error_message == "Document storage is temporarily unavailable."
    assert seed not in failed.error_message


def test_text_truncation_is_visible_in_result_and_persisted_content():
    content = b"a" * 300_001
    connection = RecordingConnection()
    result = make_service(connection=connection).ingest(
        UploadDocument("large.txt", content)
    )
    assert result.status is ResultStatus.INDEXED
    assert result.truncated is True
    assert result.warnings == ("Document truncated to 300,000 characters.",)
    assert len(connection.cursor_object.calls[0][1][2]) == 300_000


def test_embedding_finishes_before_write_connection_opens():
    events = []
    service = make_service(events=events)
    result = service.ingest(UploadDocument("notes.txt", b"alpha beta"))
    assert result.status is ResultStatus.INDEXED
    assert events.index(("embed", ["alpha beta"])) < events.index("connection_factory")


def test_embedding_provider_failure_opens_no_database_and_hides_detail():
    factory = RecordingFactory()
    provider = RecordingProvider(error=EmbeddingProviderError("secret host detail"))
    result = make_service(provider=provider, factory=factory).ingest(
        UploadDocument("notes.txt", b"alpha")
    )
    assert result.error_category is ErrorCategory.EMBEDDING_UNAVAILABLE
    assert "secret" not in result.error_message
    assert factory.calls == 0


@pytest.mark.parametrize(
    "response",
    [
        [],
        [[1.0, 2.0]],
        [[1.0, "bad", 3.0]],
        [[1.0, float("nan"), 3.0]],
        [[0.0, 0.0, 0.0]],
    ],
)
def test_invalid_embedding_sets_never_open_database(response):
    factory = RecordingFactory()
    result = make_service(
        provider=RecordingProvider(response=response), factory=factory
    ).ingest(UploadDocument("notes.txt", b"alpha"))
    assert result.error_category is ErrorCategory.EMBEDDING_INVALID
    assert factory.calls == 0


def test_chunking_failure_and_zero_chunks_open_no_database():
    def exploding_chunker(document):
        raise RuntimeError("private chunker detail")

    factory_one = RecordingFactory()
    failed = make_service(factory=factory_one, chunker=exploding_chunker).ingest(
        UploadDocument("notes.txt", b"alpha")
    )
    factory_two = RecordingFactory()
    empty = make_service(factory=factory_two, chunker=lambda document: []).ingest(
        UploadDocument("notes.txt", b"alpha")
    )
    assert failed.error_category is ErrorCategory.CHUNKING_FAILED
    assert empty.error_category is ErrorCategory.EMPTY_DOCUMENT
    assert factory_one.calls == 0
    assert factory_two.calls == 0


@pytest.mark.parametrize(
    ("factory_mode", "fail_phase", "category", "expects_rollback"),
    [
        ("raise", None, ErrorCategory.DATABASE_UNAVAILABLE, False),
        ("none", None, ErrorCategory.DATABASE_UNAVAILABLE, False),
        ("connection", "cursor", ErrorCategory.DATABASE_WRITE_FAILED, True),
        ("connection", "document_insert", ErrorCategory.DATABASE_WRITE_FAILED, True),
        ("connection", "fetchone", ErrorCategory.DATABASE_WRITE_FAILED, True),
        ("connection", "chunk_insert", ErrorCategory.DATABASE_WRITE_FAILED, True),
        ("connection", "commit", ErrorCategory.DATABASE_COMMIT_FAILED, True),
    ],
)
def test_database_failures_are_rolled_back_closed_and_safely_categorized(
    factory_mode, fail_phase, category, expects_rollback
):
    connection = RecordingConnection(fail_phase=fail_phase)
    if factory_mode == "raise":
        factory = RecordingFactory(error=RuntimeError("private connection detail"))
    elif factory_mode == "none":
        factory = RecordingFactory(returns_none=True)
    else:
        factory = RecordingFactory([connection])

    result = make_service(factory=factory).ingest(
        UploadDocument("notes.txt", b"alpha beta")
    )

    assert result.status is ResultStatus.FAILED
    assert result.error_category is category
    assert "private" not in result.error_message
    if factory_mode == "connection":
        assert connection.rollback_count == (1 if expects_rollback else 0)
        assert connection.closed is True
        if fail_phase != "cursor":
            assert connection.cursor_object.closed is True


def test_rollback_uncertainty_is_reported_without_leaking_exception():
    connection = RecordingConnection(fail_phase="chunk_insert", rollback_fails=True)
    result = make_service(connection=connection).ingest(
        UploadDocument("notes.txt", b"alpha")
    )
    assert result.warnings == ("Rollback could not be confirmed.",)
    assert connection.closed is True


def test_autocommit_connection_is_rejected_before_any_insert():
    connection = RecordingConnection(autocommit=True)
    result = make_service(connection=connection).ingest(
        UploadDocument("notes.txt", b"alpha")
    )
    assert result.error_category is ErrorCategory.DATABASE_WRITE_FAILED
    assert connection.rollback_count == 1
    assert connection.cursor_object.calls == []
    assert connection.closed is True


@pytest.mark.parametrize(
    ("cursor_close_fails", "connection_close_fails", "expected_warnings"),
    [
        (True, False, ("Cursor close could not be confirmed.",)),
        (False, True, ("Connection close could not be confirmed.",)),
        (
            True,
            True,
            (
                "Cursor close could not be confirmed.",
                "Connection close could not be confirmed.",
            ),
        ),
    ],
)
def test_committed_document_reports_cleanup_uncertainty_as_warnings(
    cursor_close_fails, connection_close_fails, expected_warnings
):
    connection = RecordingConnection(
        cursor_close_fails=cursor_close_fails,
        connection_close_fails=connection_close_fails,
    )
    result = make_service(connection=connection).ingest(
        UploadDocument("notes.txt", b"alpha")
    )
    assert result.status is ResultStatus.INDEXED
    assert result.document_id == connection.document_id
    assert connection.commit_count == 1
    assert result.warnings == expected_warnings


def test_dispatch_runs_after_commit_and_close_with_committed_identity():
    events = []
    captured = []

    def dispatch(document):
        events.append("dispatch")
        captured.append(document)

    connection = RecordingConnection(events, document_id=99)
    result = make_service(
        connection=connection, events=events, dispatch=dispatch
    ).ingest(UploadDocument("notes.txt", b"alpha"))

    assert result.status is ResultStatus.INDEXED
    assert events.index("commit") < events.index("connection_close") < events.index("dispatch")
    assert captured[0].document_id == 99
    assert captured[0].content_sha256 == hashlib.sha256(b"alpha").hexdigest()
    assert captured[0].embedding_profile_id == PROFILE.profile_id


def test_dispatch_failure_warns_but_does_not_falsify_committed_state():
    def dispatch(document):
        raise RuntimeError("private dispatcher detail")

    connection = RecordingConnection()
    result = make_service(connection=connection, dispatch=dispatch).ingest(
        UploadDocument("notes.txt", b"alpha")
    )
    assert result.status is ResultStatus.INDEXED
    assert result.document_id == connection.document_id
    assert result.warnings == ("Post-commit follow-up dispatch failed.",)
    assert connection.commit_count == 1


def test_dispatch_never_runs_for_failed_transaction():
    dispatched = []
    connection = RecordingConnection(fail_phase="commit")
    result = make_service(
        connection=connection, dispatch=lambda document: dispatched.append(document)
    ).ingest(UploadDocument("notes.txt", b"alpha"))
    assert result.status is ResultStatus.FAILED
    assert dispatched == []


def test_result_contains_hash_profile_versions_and_sanitized_filename():
    content = b"alpha beta"
    result = make_service().ingest(UploadDocument("../unsafe?.txt", content))
    payload = result.to_dict()
    assert payload["filename"] == "unsafe.txt"
    assert payload["content_sha256"] == hashlib.sha256(content).hexdigest()
    assert payload["embedding_profile_id"] == PROFILE.profile_id
    assert payload["ingestion_contract_version"] == INGESTION_CONTRACT_VERSION
    assert payload["parser_version"] == PARSER_CONTRACT_VERSION
    assert payload["chunker_version"] == CHUNKER_CONTRACT_VERSION
    assert payload["error_category"] is None


def test_batch_all_success_is_200_and_preserves_order():
    events = []
    connections = [
        RecordingConnection(events, document_id=1),
        RecordingConnection(events, document_id=2),
    ]
    service = make_service(factory=RecordingFactory(connections, events), events=events)
    batch = service.ingest_batch(
        [
            UploadDocument("first.txt", b"one"),
            UploadDocument("second.txt", b"two"),
        ]
    )
    assert isinstance(batch, BatchIngestionResult)
    assert batch.http_status == 200
    assert [result.filename for result in batch.results] == ["first.txt", "second.txt"]
    assert batch.to_dict()["summary"] == {
        "total": 2,
        "indexed": 2,
        "rejected": 0,
        "failed": 0,
    }


def test_batch_mixed_success_and_input_rejection_is_207():
    service = make_service(factory=RecordingFactory([RecordingConnection()]))
    batch = service.ingest_batch(
        [UploadDocument("good.txt", b"one"), UploadDocument("empty.txt", b"")]
    )
    assert batch.http_status == 207
    assert [result.status for result in batch.results] == [
        ResultStatus.INDEXED,
        ResultStatus.REJECTED,
    ]


def test_batch_all_input_failures_is_422():
    batch = make_service(factory=RecordingFactory()).ingest_batch(
        [UploadDocument("empty.txt", b""), UploadDocument("bad.png", b"\x00\x01")]
    )
    assert batch.http_status == 422
    assert all(result.status is ResultStatus.REJECTED for result in batch.results)


def test_batch_with_no_success_and_infrastructure_failure_is_503():
    provider = RecordingProvider(error=EmbeddingProviderError("offline"))
    batch = make_service(provider=provider, factory=RecordingFactory()).ingest_batch(
        [UploadDocument("service.txt", b"one"), UploadDocument("empty.txt", b"")]
    )
    assert batch.http_status == 503
    assert [result.status for result in batch.results] == [
        ResultStatus.FAILED,
        ResultStatus.REJECTED,
    ]


def test_batch_dictionary_reports_every_aggregated_http_status():
    all_indexed = make_service(factory=RecordingFactory()).ingest_batch(
        [UploadDocument("first.txt", b"one"), UploadDocument("second.txt", b"two")]
    )
    mixed = make_service(factory=RecordingFactory([RecordingConnection()])).ingest_batch(
        [UploadDocument("good.txt", b"one"), UploadDocument("empty.txt", b"")]
    )
    all_input_failures = make_service(factory=RecordingFactory()).ingest_batch(
        [UploadDocument("empty.txt", b""), UploadDocument("bad.png", b"\x00\x01")]
    )
    infrastructure_failure = make_service(
        provider=RecordingProvider(error=EmbeddingProviderError("offline")),
        factory=RecordingFactory(),
    ).ingest_batch(
        [UploadDocument("service.txt", b"one"), UploadDocument("empty.txt", b"")]
    )

    assert all_indexed.to_dict()["http_status"] == 200
    assert mixed.to_dict()["http_status"] == 207
    assert all_input_failures.to_dict()["http_status"] == 422
    assert infrastructure_failure.to_dict()["http_status"] == 503


def test_empty_batch_fails_closed_without_opening_a_connection():
    factory = RecordingFactory()
    batch = make_service(factory=factory).ingest_batch([])

    assert batch.results == ()
    assert batch.http_status == 503
    assert factory.calls == 0


def test_custom_parser_page_data_still_produces_exact_locator_contract():
    parsed = ParsedDocument(
        pages=[ParsedPage(2, "alpha"), ParsedPage(7, "beta")],
        document_type="pdf",
        page_count=9,
    )
    chunks = [DocumentChunk("alpha beta", 0, (2, 7))]
    service = make_service(
        extractor=lambda filename, content: parsed,
        chunker=lambda document: chunks,
    )
    result = service.ingest(UploadDocument("source.pdf", b"original bytes"))
    assert result.locator_coverage.to_dict() == {
        "locator_type": "pdf_pages",
        "chunks_total": 1,
        "chunks_with_locator": 1,
        "source_pages_total": 9,
        "source_pages_indexed": 2,
        "page_numbers_indexed": [2, 7],
    }
