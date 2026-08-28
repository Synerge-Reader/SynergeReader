import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dbSetup
from dbSetup import (
    EMBEDDING_VECTOR_DIMENSION,
    DocumentChunkSchemaError,
    KnowledgeBaseSchemaError,
    VectorColumnSchemaError,
    validate_document_chunks_embedding_schema,
    validate_knowledge_base_embedding_schema,
    validate_vector_column_schema,
)

DESTRUCTIVE_SQL_PATTERN = re.compile(
    r"\b(DROP\s+TABLE|DELETE\s+FROM|TRUNCATE|ALTER\s+TABLE)\b", re.IGNORECASE
)


def assert_no_destructive_sql(executed):
    for sql, _params in executed:
        assert not DESTRUCTIVE_SQL_PATTERN.search(sql), f"destructive SQL executed: {sql!r}"


class FakeCursor:
    """Fake cursor supporting both execute(sql, params) and execute(sql)."""

    def __init__(self, fetch_results=None, raise_on_execute=None):
        self.fetch_results = list(fetch_results or [])
        self.executed = []
        self.raise_on_execute = raise_on_execute
        self.closed = False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if self.raise_on_execute is not None:
            raise self.raise_on_execute

    def fetchone(self):
        if self.fetch_results:
            return self.fetch_results.pop(0)
        return None

    def close(self):
        self.closed = True


class FakeConnection:
    """Fake connection used only for the init_db() propagation test."""

    def __init__(self, cursor_factory=None):
        self._cursor_factory = cursor_factory or (lambda: FakeCursor(fetch_results=[]))
        self.commit_called = False
        self.rollback_called = False
        self.close_called = False

    def cursor(self):
        return self._cursor_factory()

    def commit(self):
        self.commit_called = True

    def rollback(self):
        self.rollback_called = True

    def close(self):
        self.close_called = True


def _valid_schema_cursor():
    return FakeCursor(
        fetch_results=[
            (12345,),  # to_regclass(...)::oid
            (True,),  # embedding column exists
            (EMBEDDING_VECTOR_DIMENSION,),  # atttypmod
        ]
    )


# --- 1. Fresh database ---


def test_fresh_database_returns_normally_without_further_queries():
    cursor = FakeCursor(fetch_results=[(None,)])
    validate_document_chunks_embedding_schema(cursor)
    assert len(cursor.executed) == 1
    assert_no_destructive_sql(cursor.executed)


# --- 2. Valid existing schema ---


def test_valid_existing_schema_returns_normally():
    cursor = _valid_schema_cursor()
    validate_document_chunks_embedding_schema(cursor)
    assert_no_destructive_sql(cursor.executed)


# --- 3. Missing embedding column ---


def test_missing_embedding_column_raises_with_explanation():
    cursor = FakeCursor(fetch_results=[(12345,), (False,)])
    with pytest.raises(DocumentChunkSchemaError) as exc_info:
        validate_document_chunks_embedding_schema(cursor)
    assert "embedding" in str(exc_info.value).lower()
    assert "column" in str(exc_info.value).lower()
    assert_no_destructive_sql(cursor.executed)


# --- 4. Dimension mismatch ---


def test_dimension_mismatch_raises_naming_both_values():
    wrong_dimension = EMBEDDING_VECTOR_DIMENSION + 1
    cursor = FakeCursor(fetch_results=[(12345,), (True,), (wrong_dimension,)])
    with pytest.raises(DocumentChunkSchemaError) as exc_info:
        validate_document_chunks_embedding_schema(cursor)
    message = str(exc_info.value)
    assert str(wrong_dimension) in message
    assert str(EMBEDDING_VECTOR_DIMENSION) in message
    assert_no_destructive_sql(cursor.executed)


# --- 5. Missing dimension metadata ---


def test_missing_dimension_metadata_raises():
    cursor = FakeCursor(fetch_results=[(12345,), (True,)])  # atttypmod fetchone() -> None
    with pytest.raises(DocumentChunkSchemaError) as exc_info:
        validate_document_chunks_embedding_schema(cursor)
    assert "missing" in str(exc_info.value).lower() or "malformed" in str(exc_info.value).lower()
    assert_no_destructive_sql(cursor.executed)


# --- 6. Malformed dimension metadata, including bool ---


@pytest.mark.parametrize("bad_dimension", [True, False, "768", 3.5, None])
def test_malformed_dimension_metadata_raises(bad_dimension):
    cursor = FakeCursor(fetch_results=[(12345,), (True,), (bad_dimension,)])
    with pytest.raises(DocumentChunkSchemaError) as exc_info:
        validate_document_chunks_embedding_schema(cursor)
    assert "malformed" in str(exc_info.value).lower() or "missing" in str(exc_info.value).lower()
    assert_no_destructive_sql(cursor.executed)


# --- 7. Inspection query failure ---


def test_inspection_failure_wraps_original_exception_as_cause():
    original = ValueError("boom")
    cursor = FakeCursor(raise_on_execute=original)
    with pytest.raises(DocumentChunkSchemaError) as exc_info:
        validate_document_chunks_embedding_schema(cursor)
    # Two-layer wrapping: validate_vector_column_schema wraps the original
    # failure as VectorColumnSchemaError, then validate_document_chunks_
    # embedding_schema wraps that as DocumentChunkSchemaError — the original
    # exception is preserved two levels down the __cause__ chain.
    assert exc_info.value.__cause__.__cause__ is original
    assert_no_destructive_sql(cursor.executed)


# --- 8. Read-only repeatability ---


def test_validation_is_repeatable_and_read_only():
    all_executed = []

    for _ in range(2):
        cursor = _valid_schema_cursor()
        validate_document_chunks_embedding_schema(cursor)
        all_executed.extend(cursor.executed)

    assert_no_destructive_sql(all_executed)


# --- 9. init_db() propagation ---


def test_init_db_propagates_document_chunk_schema_error(monkeypatch):
    connection = FakeConnection()
    error = DocumentChunkSchemaError("unsafe schema")

    monkeypatch.setattr(dbSetup, "connect_to_postgres", lambda: connection)
    monkeypatch.setattr(
        dbSetup,
        "validate_document_chunks_embedding_schema",
        lambda cursor: (_ for _ in ()).throw(error),
    )

    with pytest.raises(DocumentChunkSchemaError) as exc_info:
        dbSetup.init_db()

    assert exc_info.value is error
    assert connection.commit_called is False


# --- table_oid is bound as a parameter, never re-interpolated as text ---


def test_table_oid_passed_as_bound_parameter_not_interpolated():
    cursor = _valid_schema_cursor()
    validate_document_chunks_embedding_schema(cursor)

    # First call resolves the oid via a bound table_name parameter, not
    # interpolated SQL text.
    first_sql, first_params = cursor.executed[0]
    assert first_params == ("public.document_chunks",)
    assert "document_chunks" not in first_sql

    # Every subsequent call binds the already-resolved table_oid + column_name
    # — never re-interpolates the table/column name into SQL text.
    for sql, params in cursor.executed[1:]:
        assert params == (12345, "embedding")
        assert "'document_chunks'::regclass" not in sql
        assert "document_chunks" not in sql


# --- generalized validate_vector_column_schema() + per-table exception types ---


def test_validate_vector_column_schema_fresh_table_returns_normally():
    cursor = FakeCursor(fetch_results=[(None,)])
    validate_vector_column_schema(
        cursor,
        table_name="public.some_table",
        column_name="embedding",
        expected_dimension=EMBEDDING_VECTOR_DIMENSION,
    )
    assert len(cursor.executed) == 1
    assert_no_destructive_sql(cursor.executed)


def test_document_chunks_failure_raises_specific_subclass_not_generic_base():
    cursor = FakeCursor(fetch_results=[(12345,), (False,)])
    with pytest.raises(DocumentChunkSchemaError) as exc_info:
        validate_document_chunks_embedding_schema(cursor)
    assert type(exc_info.value) is DocumentChunkSchemaError
    assert isinstance(exc_info.value, VectorColumnSchemaError)
    assert not isinstance(exc_info.value, KnowledgeBaseSchemaError)
    assert_no_destructive_sql(cursor.executed)


def test_knowledge_base_failure_raises_knowledge_base_schema_error_not_document_chunk():
    cursor = FakeCursor(fetch_results=[(6789,), (False,)])
    with pytest.raises(KnowledgeBaseSchemaError) as exc_info:
        validate_knowledge_base_embedding_schema(cursor)
    assert type(exc_info.value) is KnowledgeBaseSchemaError
    assert isinstance(exc_info.value, VectorColumnSchemaError)
    assert not isinstance(exc_info.value, DocumentChunkSchemaError)
    assert_no_destructive_sql(cursor.executed)


def test_knowledge_base_dimension_mismatch_raises_with_both_values():
    wrong_dimension = EMBEDDING_VECTOR_DIMENSION + 1
    cursor = FakeCursor(fetch_results=[(6789,), (True,), (wrong_dimension,)])
    with pytest.raises(KnowledgeBaseSchemaError) as exc_info:
        validate_knowledge_base_embedding_schema(cursor)
    message = str(exc_info.value)
    assert str(wrong_dimension) in message
    assert str(EMBEDDING_VECTOR_DIMENSION) in message
    assert_no_destructive_sql(cursor.executed)
