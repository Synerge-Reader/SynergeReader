"""Atomic, route-independent document ingestion for E1a-2.

This module owns the deterministic parse -> chunk -> embed -> persist flow.
It deliberately does not import ``main.py``, start background threads, open a
connection at import time, or change the database schema.  A future route
adapter can construct ``DocumentIngestionService`` with the application's
already-resolved embedding profile, provider, and connection factory.

The service establishes one transaction boundary per document.  Parsing,
chunking, and embedding finish before the write connection is opened.  Every
database-path failure explicitly attempts a rollback and always closes the
cursor/connection.  Post-commit work is exposed as an injected callback and
can never turn a committed document into a reported ingestion failure.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence

from psycopg2.extras import Json

from document_chunker import DocumentChunk, build_chunk_locator, chunk_document
from document_parser import (
    ExtractionError,
    ParsedDocument,
    UnsupportedFileTypeError,
    extract_text_from_upload,
    sanitize_filename,
)
from ollama_embedding_provider import EmbeddingProvider, EmbeddingProviderError
from rag_model_profiles import EmbeddingProfile


INGESTION_CONTRACT_VERSION = "e1a2.ingestion.v1"
PARSER_CONTRACT_VERSION = "e1a.parser.v1"
CHUNKER_CONTRACT_VERSION = "e1a.chunker.v1"


class ResultStatus(str, Enum):
    INDEXED = "indexed"
    REJECTED = "rejected"
    FAILED = "failed"


class FailureScope(str, Enum):
    INPUT = "input"
    INFRASTRUCTURE = "infrastructure"


class ErrorCategory(str, Enum):
    INVALID_UPLOAD = "invalid_upload"
    EMPTY_FILE = "empty_file"
    FILE_TOO_LARGE = "file_too_large"
    UNSUPPORTED_FILE_TYPE = "unsupported_file_type"
    EXTENSION_CONTENT_MISMATCH = "extension_content_mismatch"
    PAGE_LIMIT_EXCEEDED = "page_limit_exceeded"
    IMAGE_ONLY_PDF = "image_only_pdf"
    EMPTY_DOCUMENT = "empty_document"
    EXTRACTION_FAILED = "extraction_failed"
    PARSER_UNAVAILABLE = "parser_unavailable"
    CHUNKING_FAILED = "chunking_failed"
    EMBEDDING_UNAVAILABLE = "embedding_unavailable"
    EMBEDDING_INVALID = "embedding_invalid"
    DATABASE_UNAVAILABLE = "database_unavailable"
    DATABASE_WRITE_FAILED = "database_write_failed"
    DATABASE_COMMIT_FAILED = "database_commit_failed"


_INPUT_ERROR_CATEGORIES = frozenset(
    {
        ErrorCategory.INVALID_UPLOAD,
        ErrorCategory.EMPTY_FILE,
        ErrorCategory.FILE_TOO_LARGE,
        ErrorCategory.UNSUPPORTED_FILE_TYPE,
        ErrorCategory.EXTENSION_CONTENT_MISMATCH,
        ErrorCategory.PAGE_LIMIT_EXCEEDED,
        ErrorCategory.IMAGE_ONLY_PDF,
        ErrorCategory.EMPTY_DOCUMENT,
        ErrorCategory.EXTRACTION_FAILED,
    }
)

_EXTENSION_TYPES = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "text",
}

_SAFE_ERROR_MESSAGES = {
    ErrorCategory.INVALID_UPLOAD: "The uploaded document is invalid.",
    ErrorCategory.EMPTY_FILE: "The uploaded file is empty.",
    ErrorCategory.FILE_TOO_LARGE: "The uploaded file exceeds the size limit.",
    ErrorCategory.UNSUPPORTED_FILE_TYPE: "Upload a PDF, DOCX, or plain text file.",
    ErrorCategory.EXTENSION_CONTENT_MISMATCH: (
        "The filename extension does not match the file contents."
    ),
    ErrorCategory.PAGE_LIMIT_EXCEEDED: "The PDF exceeds the supported page limit.",
    ErrorCategory.IMAGE_ONLY_PDF: (
        "The PDF has no extractable text; image-only PDFs are not supported."
    ),
    ErrorCategory.EMPTY_DOCUMENT: "The document contains no indexable text.",
    ErrorCategory.EXTRACTION_FAILED: "The document could not be read.",
    ErrorCategory.PARSER_UNAVAILABLE: "Document parsing is temporarily unavailable.",
    ErrorCategory.CHUNKING_FAILED: "Document chunking is temporarily unavailable.",
    ErrorCategory.EMBEDDING_UNAVAILABLE: "Embedding is temporarily unavailable.",
    ErrorCategory.EMBEDDING_INVALID: "Embedding returned an invalid result.",
    ErrorCategory.DATABASE_UNAVAILABLE: "Document storage is temporarily unavailable.",
    ErrorCategory.DATABASE_WRITE_FAILED: "The document could not be stored.",
    ErrorCategory.DATABASE_COMMIT_FAILED: "The document could not be committed.",
}


class ConnectionLike(Protocol):
    """A DB-API connection handed back by ``connection_factory``.

    Precondition: the connection must already have
    ``pgvector.psycopg2.register_vector()`` applied to it. Chunk inserts pass
    each embedding as a plain ``list[float]``, and psycopg2 can adapt that
    list to the PostgreSQL ``vector`` type only after vector registration has
    run on the connection. An unregistered connection therefore fails at
    chunk insertion at runtime, not at connect time. The unit-test doubles
    record parameters instead of adapting them, so they cannot detect this
    precondition; enforcing it belongs to the future route unit that owns the
    real connection factory.
    """

    def cursor(self) -> Any: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class DocumentMetadata:
    author: Optional[str] = None
    title: Optional[str] = None
    publication_date: Optional[str] = None
    source: Optional[str] = None
    doi_url: Optional[str] = None


@dataclass(frozen=True)
class UploadDocument:
    filename: str
    content: bytes
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    uploader_id: Any = None


@dataclass(frozen=True)
class LocatorCoverage:
    locator_type: Optional[str]
    chunks_total: int
    chunks_with_locator: int
    source_pages_total: Optional[int]
    source_pages_indexed: Optional[int]
    page_numbers_indexed: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "locator_type": self.locator_type,
            "chunks_total": self.chunks_total,
            "chunks_with_locator": self.chunks_with_locator,
            "source_pages_total": self.source_pages_total,
            "source_pages_indexed": self.source_pages_indexed,
            "page_numbers_indexed": list(self.page_numbers_indexed),
        }


@dataclass(frozen=True)
class IngestionResult:
    filename: str
    status: ResultStatus
    document_id: Any = None
    chunks_count: int = 0
    truncated: bool = False
    warnings: tuple[str, ...] = ()
    source_page_count: Optional[int] = None
    indexed_page_count: Optional[int] = None
    locator_coverage: Optional[LocatorCoverage] = None
    error_category: Optional[ErrorCategory] = None
    error_message: Optional[str] = None
    content_sha256: Optional[str] = None
    embedding_profile_id: Optional[str] = None
    ingestion_contract_version: str = INGESTION_CONTRACT_VERSION
    parser_version: str = PARSER_CONTRACT_VERSION
    chunker_version: str = CHUNKER_CONTRACT_VERSION

    @property
    def failure_scope(self) -> Optional[FailureScope]:
        if self.error_category is None:
            return None
        if self.error_category in _INPUT_ERROR_CATEGORIES:
            return FailureScope.INPUT
        return FailureScope.INFRASTRUCTURE

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "document_id": self.document_id,
            "status": self.status.value,
            "chunks_count": self.chunks_count,
            "truncated": self.truncated,
            "warnings": list(self.warnings),
            "source_page_count": self.source_page_count,
            "indexed_page_count": self.indexed_page_count,
            "locator_coverage": (
                self.locator_coverage.to_dict() if self.locator_coverage else None
            ),
            "error_category": (
                self.error_category.value if self.error_category else None
            ),
            "error_message": self.error_message,
            "content_sha256": self.content_sha256,
            "embedding_profile_id": self.embedding_profile_id,
            "ingestion_contract_version": self.ingestion_contract_version,
            "parser_version": self.parser_version,
            "chunker_version": self.chunker_version,
        }


@dataclass(frozen=True)
class BatchIngestionResult:
    results: tuple[IngestionResult, ...]
    http_status: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [result.to_dict() for result in self.results],
            "summary": {
                "total": len(self.results),
                "indexed": sum(
                    result.status is ResultStatus.INDEXED for result in self.results
                ),
                "rejected": sum(
                    result.status is ResultStatus.REJECTED for result in self.results
                ),
                "failed": sum(
                    result.status is ResultStatus.FAILED for result in self.results
                ),
            },
            "http_status": self.http_status,
        }


@dataclass(frozen=True)
class CommittedDocument:
    document_id: Any
    filename: str
    text: str
    content_sha256: str
    embedding_profile_id: str


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _safe_filename(value: object) -> str:
    if not isinstance(value, str):
        return "untitled"
    return sanitize_filename(value)


def _emit_diagnostic(
    category: ErrorCategory,
    filename: str,
    exc: BaseException,
) -> None:
    """Emit one fixed-shape, content-free ingestion diagnostic.

    The line carries only the stable category value, the already-sanitized
    filename, and the exception's class name. ``str(exc)``, ``repr(exc)``,
    tracebacks, SQL, connection details, and document or chunk contents are
    deliberately never included, so the operator gets a failure signal that
    cannot leak private legal-document content or infrastructure detail.
    """
    print(f"[Ingestion] {category.value} for {filename}: {type(exc).__name__}")


def _failure(
    *,
    filename: str,
    category: ErrorCategory,
    content_sha256: Optional[str] = None,
    profile_id: Optional[str] = None,
    parsed: Optional[ParsedDocument] = None,
    warnings: Iterable[str] = (),
) -> IngestionResult:
    combined_warnings = list(parsed.warnings if parsed else [])
    combined_warnings.extend(warnings)
    source_pages = parsed.page_count if parsed and parsed.document_type == "pdf" else None
    status = (
        ResultStatus.REJECTED
        if category in _INPUT_ERROR_CATEGORIES
        else ResultStatus.FAILED
    )
    return IngestionResult(
        filename=filename,
        status=status,
        truncated=bool(parsed and parsed.truncated),
        warnings=tuple(combined_warnings),
        source_page_count=source_pages,
        error_category=category,
        error_message=_SAFE_ERROR_MESSAGES[category],
        content_sha256=content_sha256,
        embedding_profile_id=profile_id,
    )


def _category_for_extraction_error(exc: ExtractionError) -> ErrorCategory:
    message = exc.user_message.lower()
    if isinstance(exc, UnsupportedFileTypeError):
        return ErrorCategory.UNSUPPORTED_FILE_TYPE
    if exc.http_status >= 500:
        return ErrorCategory.PARSER_UNAVAILABLE
    if exc.http_status == 413:
        return ErrorCategory.FILE_TOO_LARGE
    if "scanned" in message or "image-only" in message:
        return ErrorCategory.IMAGE_ONLY_PDF
    if "page limit" in message:
        return ErrorCategory.PAGE_LIMIT_EXCEEDED
    if "empty" in message or "only images" in message:
        return ErrorCategory.EMPTY_DOCUMENT
    return ErrorCategory.EXTRACTION_FAILED


def _extension_matches(filename: str, document_type: str) -> bool:
    suffix = Path(filename).suffix.lower()
    if not suffix:
        return True
    expected = _EXTENSION_TYPES.get(suffix)
    return expected == document_type


def _locator_coverage(
    parsed: ParsedDocument,
    chunks: Sequence[DocumentChunk],
) -> LocatorCoverage:
    if parsed.document_type == "pdf":
        page_numbers = tuple(
            sorted({number for chunk in chunks for number in chunk.page_numbers})
        )
        return LocatorCoverage(
            locator_type="pdf_pages",
            chunks_total=len(chunks),
            chunks_with_locator=sum(bool(chunk.page_numbers) for chunk in chunks),
            source_pages_total=parsed.page_count,
            source_pages_indexed=len(page_numbers),
            page_numbers_indexed=page_numbers,
        )
    return LocatorCoverage(
        locator_type=parsed.document_type,
        chunks_total=len(chunks),
        chunks_with_locator=0,
        source_pages_total=None,
        source_pages_indexed=None,
    )


def _validate_embeddings(
    embeddings: object,
    *,
    expected_count: int,
    expected_dimension: int,
) -> list[list[float]]:
    if not isinstance(embeddings, list) or len(embeddings) != expected_count:
        raise ValueError("embedding count mismatch")

    validated: list[list[float]] = []
    for vector in embeddings:
        if not isinstance(vector, list) or len(vector) != expected_dimension:
            raise ValueError("embedding dimension mismatch")
        values: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("embedding value is not a real number")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("embedding value is not finite")
            values.append(number)
        if all(value == 0.0 for value in values):
            raise ValueError("embedding vector is all-zero")
        validated.append(values)
    return validated


def _batch_http_status(results: Sequence[IngestionResult]) -> int:
    if results and all(result.status is ResultStatus.INDEXED for result in results):
        return 200
    if any(result.status is ResultStatus.INDEXED for result in results):
        return 207
    if results and all(result.failure_scope is FailureScope.INPUT for result in results):
        return 422
    return 503


@dataclass
class DocumentIngestionService:
    connection_factory: Callable[[], Optional[ConnectionLike]]
    embedding_provider: EmbeddingProvider
    embedding_profile: EmbeddingProfile
    dispatch_after_commit: Optional[Callable[[CommittedDocument], None]] = None
    clock: Callable[[], dt.datetime] = _utc_now
    extractor: Callable[[str, bytes], ParsedDocument] = extract_text_from_upload
    chunker: Callable[[ParsedDocument], list[DocumentChunk]] = chunk_document
    locator_builder: Callable[[DocumentChunk, str], dict[str, Any]] = build_chunk_locator
    json_adapter: Callable[[dict[str, Any]], Any] = Json

    def ingest(self, upload: UploadDocument) -> IngestionResult:
        filename = _safe_filename(getattr(upload, "filename", None))
        raw_content = getattr(upload, "content", None)
        profile_id = self.embedding_profile.profile_id

        if not isinstance(raw_content, bytes):
            return _failure(
                filename=filename,
                category=ErrorCategory.INVALID_UPLOAD,
                profile_id=profile_id,
            )

        content_sha256 = hashlib.sha256(raw_content).hexdigest()
        if not raw_content:
            return _failure(
                filename=filename,
                category=ErrorCategory.EMPTY_FILE,
                content_sha256=content_sha256,
                profile_id=profile_id,
            )

        try:
            parsed = self.extractor(filename, raw_content)
        except ExtractionError as exc:
            category = _category_for_extraction_error(exc)
            _emit_diagnostic(category, filename, exc)
            return _failure(
                filename=filename,
                category=category,
                content_sha256=content_sha256,
                profile_id=profile_id,
            )
        except Exception as exc:
            _emit_diagnostic(ErrorCategory.PARSER_UNAVAILABLE, filename, exc)
            return _failure(
                filename=filename,
                category=ErrorCategory.PARSER_UNAVAILABLE,
                content_sha256=content_sha256,
                profile_id=profile_id,
            )

        if not _extension_matches(filename, parsed.document_type):
            return _failure(
                filename=filename,
                category=(
                    ErrorCategory.EXTENSION_CONTENT_MISMATCH
                    if Path(filename).suffix.lower() in _EXTENSION_TYPES
                    else ErrorCategory.UNSUPPORTED_FILE_TYPE
                ),
                content_sha256=content_sha256,
                profile_id=profile_id,
                parsed=parsed,
            )

        try:
            chunks = self.chunker(parsed)
        except Exception as exc:
            _emit_diagnostic(ErrorCategory.CHUNKING_FAILED, filename, exc)
            return _failure(
                filename=filename,
                category=ErrorCategory.CHUNKING_FAILED,
                content_sha256=content_sha256,
                profile_id=profile_id,
                parsed=parsed,
            )

        if not chunks or any(not chunk.text.strip() for chunk in chunks):
            return _failure(
                filename=filename,
                category=ErrorCategory.EMPTY_DOCUMENT,
                content_sha256=content_sha256,
                profile_id=profile_id,
                parsed=parsed,
            )

        chunk_texts = [chunk.text for chunk in chunks]
        try:
            raw_embeddings = self.embedding_provider.embed_documents(chunk_texts)
        except EmbeddingProviderError as exc:
            _emit_diagnostic(ErrorCategory.EMBEDDING_UNAVAILABLE, filename, exc)
            return _failure(
                filename=filename,
                category=ErrorCategory.EMBEDDING_UNAVAILABLE,
                content_sha256=content_sha256,
                profile_id=profile_id,
                parsed=parsed,
            )
        except Exception as exc:
            _emit_diagnostic(ErrorCategory.EMBEDDING_UNAVAILABLE, filename, exc)
            return _failure(
                filename=filename,
                category=ErrorCategory.EMBEDDING_UNAVAILABLE,
                content_sha256=content_sha256,
                profile_id=profile_id,
                parsed=parsed,
            )

        try:
            embeddings = _validate_embeddings(
                raw_embeddings,
                expected_count=len(chunks),
                expected_dimension=self.embedding_profile.dimension,
            )
        except (TypeError, ValueError) as exc:
            _emit_diagnostic(ErrorCategory.EMBEDDING_INVALID, filename, exc)
            return _failure(
                filename=filename,
                category=ErrorCategory.EMBEDDING_INVALID,
                content_sha256=content_sha256,
                profile_id=profile_id,
                parsed=parsed,
            )

        connection: Optional[ConnectionLike] = None
        cursor: Any = None
        phase = "connect"
        database_warnings: list[str] = []
        document_id: Any = None
        failure_category: Optional[ErrorCategory] = None

        try:
            connection = self.connection_factory()
            if connection is None:
                raise RuntimeError("connection factory returned None")

            phase = "write"
            if getattr(connection, "autocommit", False):
                raise RuntimeError("autocommit connections are not permitted")
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO documents
                (filename, upload_timestamp, content, author, title,
                 publication_date, source, doi_url, user_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    filename,
                    self.clock().isoformat(),
                    parsed.text,
                    upload.metadata.author,
                    upload.metadata.title,
                    upload.metadata.publication_date,
                    upload.metadata.source,
                    upload.metadata.doi_url,
                    upload.uploader_id,
                ),
            )
            row = cursor.fetchone()
            if not row or row[0] is None:
                raise RuntimeError("document insert returned no id")
            document_id = row[0]

            for chunk, embedding in zip(chunks, embeddings):
                locator = self.locator_builder(chunk, parsed.document_type)
                cursor.execute(
                    """
                    INSERT INTO document_chunks
                    (document_id, chunk_text, chunk_index, embedding,
                     page_start, page_end, locator_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        document_id,
                        chunk.text,
                        chunk.chunk_index,
                        embedding,
                        chunk.page_start,
                        chunk.page_end,
                        self.json_adapter(locator),
                    ),
                )

            phase = "commit"
            connection.commit()
        except Exception as exc:
            if connection is not None:
                try:
                    connection.rollback()
                except Exception:
                    database_warnings.append("Rollback could not be confirmed.")
            failure_category = {
                "connect": ErrorCategory.DATABASE_UNAVAILABLE,
                "write": ErrorCategory.DATABASE_WRITE_FAILED,
                "commit": ErrorCategory.DATABASE_COMMIT_FAILED,
            }[phase]
            _emit_diagnostic(failure_category, filename, exc)
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    database_warnings.append("Cursor close could not be confirmed.")
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    database_warnings.append("Connection close could not be confirmed.")

        if failure_category is not None:
            return _failure(
                filename=filename,
                category=failure_category,
                content_sha256=content_sha256,
                profile_id=profile_id,
                parsed=parsed,
                warnings=database_warnings,
            )

        warnings = list(parsed.warnings) + database_warnings
        committed = CommittedDocument(
            document_id=document_id,
            filename=filename,
            text=parsed.text,
            content_sha256=content_sha256,
            embedding_profile_id=profile_id,
        )
        if self.dispatch_after_commit is not None:
            try:
                self.dispatch_after_commit(committed)
            except Exception:
                warnings.append("Post-commit follow-up dispatch failed.")

        coverage = _locator_coverage(parsed, chunks)
        return IngestionResult(
            filename=filename,
            status=ResultStatus.INDEXED,
            document_id=document_id,
            chunks_count=len(chunks),
            truncated=parsed.truncated,
            warnings=tuple(warnings),
            source_page_count=coverage.source_pages_total,
            indexed_page_count=coverage.source_pages_indexed,
            locator_coverage=coverage,
            content_sha256=content_sha256,
            embedding_profile_id=profile_id,
        )

    def ingest_batch(self, uploads: Iterable[UploadDocument]) -> BatchIngestionResult:
        results = tuple(self.ingest(upload) for upload in uploads)
        return BatchIngestionResult(
            results=results,
            http_status=_batch_http_status(results),
        )
