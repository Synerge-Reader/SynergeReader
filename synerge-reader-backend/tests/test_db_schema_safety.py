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
    validate_document_chunks_embedding_schema,
    validate_knowledge_base_embedding_schema,
)


VALIDATORS = [
    (
        "document_chunks",
        validate_document_chunks_embedding_schema,
        DocumentChunkSchemaError,
    ),
    (
        "knowledge_base",
        validate_knowledge_base_embedding_schema,
        KnowledgeBaseSchemaError,
    ),
]

INVALID_DIMENSIONS = [True, False, 0, -1, "768", 768.0, None]

DATA_LOSS_SQL_PATTERN = re.compile(
    r"\b(DROP\s+TABLE|DROP\s+COLUMN|DELETE\s+FROM|TRUNCATE)\b",
    re.IGNORECASE,
)

DDL_SQL_PATTERN = re.compile(
    r"\b(CREATE|ALTER|DROP|DELETE|TRUNCATE)\b",
    re.IGNORECASE,
)


def _normalized(sql):
    return " ".join(sql.split())


def assert_no_data_loss_sql(executed):
    for sql, _params in executed:
        assert not DATA_LOSS_SQL_PATTERN.search(sql), (
            f"data-loss SQL executed: {sql!r}"
        )


def assert_read_only_sql(executed):
    for sql, _params in executed:
        assert not DDL_SQL_PATTERN.search(sql), (
            f"non-read-only SQL executed during validation: {sql!r}"
        )


class FakeCursor:
    """Queue-backed cursor for direct validator tests."""

    def __init__(
        self,
        fetch_results=None,
        raise_on_execute=None,
        close_error=None,
    ):
        self.fetch_results = list(fetch_results or [])
        self.executed = []
        self.raise_on_execute = raise_on_execute
        self.close_error = close_error
        self.close_count = 0
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
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error
        self.closed = True


class InitDbCursor(FakeCursor):
    """Simulates catalog answers after CREATE TABLE for init_db()."""

    def __init__(self, expected_dimension=EMBEDDING_VECTOR_DIMENSION, **kwargs):
        super().__init__(**kwargs)
        self.expected_dimension = expected_dimension
        self._next_fetch = None

    def execute(self, sql, params=None):
        super().execute(sql, params)
        normalized = _normalized(sql)
        if normalized.startswith("SELECT to_regclass"):
            table_name = params[0]
            table_oid = {
                "document_chunks": 101,
                "knowledge_base": 202,
            }[table_name]
            self._next_fetch = (table_oid,)
        elif "SELECT EXISTS" in normalized and "pg_attribute" in normalized:
            self._next_fetch = (True,)
        elif normalized.startswith("SELECT atttypmod"):
            self._next_fetch = (self.expected_dimension,)

    def fetchone(self):
        result = self._next_fetch
        self._next_fetch = None
        return result


class FakeConnection:
    def __init__(
        self,
        cursor=None,
        cursor_error=None,
        rollback_error=None,
        close_error=None,
    ):
        self._cursor = cursor if cursor is not None else FakeCursor()
        self.cursor_error = cursor_error
        self.rollback_error = rollback_error
        self.close_error = close_error
        self.cursor_count = 0
        self.commit_count = 0
        self.rollback_count = 0
        self.close_count = 0

    def cursor(self):
        self.cursor_count += 1
        if self.cursor_error is not None:
            raise self.cursor_error
        return self._cursor

    def commit(self):
        self.commit_count += 1

    def rollback(self):
        self.rollback_count += 1
        if self.rollback_error is not None:
            raise self.rollback_error

    def close(self):
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error


def _valid_schema_cursor(table_oid=12345):
    return FakeCursor(
        fetch_results=[
            (table_oid,),
            (True,),
            (EMBEDDING_VECTOR_DIMENSION,),
        ]
    )


def _raise(error):
    raise error


def _cleanup_notes(error):
    return getattr(error, "__notes__", [])


@pytest.mark.parametrize("valid_dimension", [1, 768, 1024])
def test_dimension_guard_accepts_positive_non_boolean_integers(valid_dimension):
    assert (
        dbSetup._require_positive_int_dimension(
            valid_dimension,
            "expected_dimension",
        )
        == valid_dimension
    )


@pytest.mark.parametrize("invalid_dimension", INVALID_DIMENSIONS)
def test_dimension_guard_rejects_invalid_values(invalid_dimension):
    with pytest.raises(ValueError, match="expected_dimension"):
        dbSetup._require_positive_int_dimension(
            invalid_dimension,
            "expected_dimension",
        )


@pytest.mark.parametrize("table_name,validator,error_type", VALIDATORS)
@pytest.mark.parametrize("invalid_dimension", INVALID_DIMENSIONS)
def test_public_validators_reject_invalid_dimensions_before_sql(
    table_name,
    validator,
    error_type,
    invalid_dimension,
):
    cursor = FakeCursor(
        raise_on_execute=AssertionError("invalid dimension reached SQL")
    )

    with pytest.raises(ValueError, match="expected_dimension"):
        validator(cursor, expected_dimension=invalid_dimension)

    assert cursor.executed == []


@pytest.mark.parametrize("invalid_dimension", INVALID_DIMENSIONS)
def test_init_db_rejects_invalid_dimension_before_connect(
    monkeypatch,
    invalid_dimension,
):
    connect_calls = []

    def unexpected_connect():
        connect_calls.append(True)
        raise AssertionError("invalid dimension reached connect_to_postgres")

    monkeypatch.setattr(dbSetup, "connect_to_postgres", unexpected_connect)

    with pytest.raises(ValueError, match="expected_dimension"):
        dbSetup.init_db(expected_dimension=invalid_dimension)

    assert connect_calls == []


@pytest.mark.parametrize("table_name,validator,error_type", VALIDATORS)
def test_absent_table_returns_normally_without_mutation(
    table_name,
    validator,
    error_type,
):
    cursor = FakeCursor(fetch_results=[(None,)])

    validator(cursor)

    assert cursor.executed == [
        ("SELECT to_regclass(%s)::oid", (table_name,))
    ]
    assert_read_only_sql(cursor.executed)


@pytest.mark.parametrize("table_name,validator,error_type", VALIDATORS)
def test_valid_existing_schema_returns_normally(
    table_name,
    validator,
    error_type,
):
    cursor = _valid_schema_cursor()

    validator(cursor)

    assert_read_only_sql(cursor.executed)


@pytest.mark.parametrize("table_name,validator,error_type", VALIDATORS)
def test_missing_embedding_column_raises_named_error(
    table_name,
    validator,
    error_type,
):
    cursor = FakeCursor(fetch_results=[(12345,), (False,)])

    with pytest.raises(error_type) as exc_info:
        validator(cursor)

    message = str(exc_info.value).lower()
    assert table_name in message
    assert "embedding" in message
    assert "column" in message
    assert_read_only_sql(cursor.executed)


@pytest.mark.parametrize("table_name,validator,error_type", VALIDATORS)
def test_dimension_mismatch_raises_naming_table_and_both_values(
    table_name,
    validator,
    error_type,
):
    cursor = FakeCursor(fetch_results=[(12345,), (True,), (1024,)])

    with pytest.raises(error_type) as exc_info:
        validator(cursor)

    message = str(exc_info.value)
    assert table_name in message
    assert "1024" in message
    assert str(EMBEDDING_VECTOR_DIMENSION) in message
    assert_read_only_sql(cursor.executed)


@pytest.mark.parametrize("table_name,validator,error_type", VALIDATORS)
def test_missing_dimension_metadata_raises(
    table_name,
    validator,
    error_type,
):
    cursor = FakeCursor(fetch_results=[(12345,), (True,)])

    with pytest.raises(error_type) as exc_info:
        validator(cursor)

    message = str(exc_info.value).lower()
    assert "missing" in message or "malformed" in message
    assert_read_only_sql(cursor.executed)


@pytest.mark.parametrize("table_name,validator,error_type", VALIDATORS)
@pytest.mark.parametrize("bad_dimension", [True, False, "768", 3.5, None])
def test_malformed_dimension_metadata_raises(
    table_name,
    validator,
    error_type,
    bad_dimension,
):
    cursor = FakeCursor(
        fetch_results=[(12345,), (True,), (bad_dimension,)]
    )

    with pytest.raises(error_type) as exc_info:
        validator(cursor)

    message = str(exc_info.value).lower()
    assert "malformed" in message or "missing" in message
    assert_read_only_sql(cursor.executed)


@pytest.mark.parametrize("table_name,validator,error_type", VALIDATORS)
def test_inspection_failure_wraps_original_exception_as_cause(
    table_name,
    validator,
    error_type,
):
    original = ValueError("boom")
    cursor = FakeCursor(raise_on_execute=original)

    with pytest.raises(error_type) as exc_info:
        validator(cursor)

    assert exc_info.value.__cause__ is original
    assert table_name in str(exc_info.value)
    assert_read_only_sql(cursor.executed)


@pytest.mark.parametrize("table_name,validator,error_type", VALIDATORS)
def test_validation_is_repeatable_and_read_only(
    table_name,
    validator,
    error_type,
):
    all_executed = []

    for _ in range(2):
        cursor = _valid_schema_cursor()
        validator(cursor)
        all_executed.extend(cursor.executed)

    assert_read_only_sql(all_executed)


@pytest.mark.parametrize("table_name,validator,error_type", VALIDATORS)
def test_catalog_identifiers_are_bound_parameters(
    table_name,
    validator,
    error_type,
):
    cursor = _valid_schema_cursor(table_oid=12345)

    validator(cursor)

    assert cursor.executed[0][1] == (table_name,)
    assert cursor.executed[1][1] == (12345, "embedding")
    assert cursor.executed[2][1] == (12345, "embedding")
    for sql, _params in cursor.executed:
        assert f"'{table_name}'::regclass" not in sql


def test_fresh_database_path_creates_dependencies_then_validates(monkeypatch):
    cursor = InitDbCursor()
    connection = FakeConnection(cursor)
    monkeypatch.setattr(dbSetup, "connect_to_postgres", lambda: connection)

    dbSetup.init_db()

    normalized = [_normalized(sql) for sql, _params in cursor.executed]

    create_positions = {
        table: next(
            index
            for index, sql in enumerate(normalized)
            if f"CREATE TABLE IF NOT EXISTS {table}" in sql
        )
        for table in [
            "users",
            "documents",
            "document_chunks",
            "chat_history",
            "knowledge_base",
            "admin_audit_log",
            "document_insights",
        ]
    }
    assert list(create_positions.values()) == sorted(create_positions.values())

    for table in ["document_chunks", "knowledge_base"]:
        validation_position = next(
            index
            for index, (_sql, params) in enumerate(cursor.executed)
            if params == (table,)
        )
        assert create_positions[table] < validation_position

    all_sql = " ".join(normalized)
    for preserved_fragment in [
        "email_verification_token",
        "password_reset_token",
        "user_id UUID REFERENCES users(id)",
        "source_type TEXT DEFAULT 'document'",
        "CREATE TABLE IF NOT EXISTS admin_audit_log",
        "CREATE TABLE IF NOT EXISTS document_insights",
        "page_start INTEGER",
        "page_end INTEGER",
        "locator_json JSONB",
    ]:
        assert preserved_fragment in all_sql

    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.close_count == 1
    assert cursor.closed is True
    assert_no_data_loss_sql(cursor.executed)


@pytest.mark.parametrize(
    "validator_name,error",
    [
        (
            "validate_document_chunks_embedding_schema",
            DocumentChunkSchemaError("unsafe document_chunks schema"),
        ),
        (
            "validate_knowledge_base_embedding_schema",
            KnowledgeBaseSchemaError("unsafe knowledge_base schema"),
        ),
    ],
)
def test_init_db_rolls_back_closes_and_propagates_schema_errors(
    monkeypatch,
    validator_name,
    error,
):
    cursor = InitDbCursor()
    connection = FakeConnection(cursor)
    monkeypatch.setattr(dbSetup, "connect_to_postgres", lambda: connection)
    monkeypatch.setattr(
        dbSetup,
        validator_name,
        lambda cursor, expected_dimension: (_ for _ in ()).throw(error),
    )

    with pytest.raises(type(error)) as exc_info:
        dbSetup.init_db()

    assert exc_info.value is error
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.close_count == 1
    assert cursor.closed is True
    assert_no_data_loss_sql(cursor.executed)


def test_init_db_never_alters_or_replaces_existing_embedding_columns(monkeypatch):
    cursor = InitDbCursor()
    connection = FakeConnection(cursor)
    monkeypatch.setattr(dbSetup, "connect_to_postgres", lambda: connection)

    dbSetup.init_db()

    normalized = [_normalized(sql) for sql, _params in cursor.executed]
    embedding_alters = [
        sql
        for sql in normalized
        if sql.startswith("ALTER TABLE") and "embedding" in sql.lower()
    ]
    assert embedding_alters == []
    assert_no_data_loss_sql(cursor.executed)


def test_non_default_dimension_is_used_in_both_create_table_declarations(
    monkeypatch,
):
    expected_dimension = 1024
    cursor = InitDbCursor(expected_dimension=expected_dimension)
    connection = FakeConnection(cursor)
    monkeypatch.setattr(dbSetup, "connect_to_postgres", lambda: connection)

    dbSetup.init_db(expected_dimension=expected_dimension)

    normalized = [_normalized(sql) for sql, _params in cursor.executed]
    document_chunks_sql = next(
        sql
        for sql in normalized
        if "CREATE TABLE IF NOT EXISTS document_chunks" in sql
    )
    knowledge_base_sql = next(
        sql
        for sql in normalized
        if "CREATE TABLE IF NOT EXISTS knowledge_base" in sql
    )
    assert "embedding vector(1024)" in document_chunks_sql
    assert "embedding vector(1024)" in knowledge_base_sql
    assert_no_data_loss_sql(cursor.executed)


def test_non_default_dimension_reaches_both_schema_validators(monkeypatch):
    expected_dimension = 1024
    cursor = InitDbCursor(expected_dimension=expected_dimension)
    connection = FakeConnection(cursor)
    received = []
    monkeypatch.setattr(dbSetup, "connect_to_postgres", lambda: connection)

    def record_document_chunks(cursor, expected_dimension):
        received.append(("document_chunks", cursor, expected_dimension))

    def record_knowledge_base(cursor, expected_dimension):
        received.append(("knowledge_base", cursor, expected_dimension))

    monkeypatch.setattr(
        dbSetup,
        "validate_document_chunks_embedding_schema",
        record_document_chunks,
    )
    monkeypatch.setattr(
        dbSetup,
        "validate_knowledge_base_embedding_schema",
        record_knowledge_base,
    )

    dbSetup.init_db(expected_dimension=expected_dimension)

    assert received == [
        ("document_chunks", cursor, 1024),
        ("knowledge_base", cursor, 1024),
    ]


def test_cursor_acquisition_failure_preserves_primary_and_closes_connection(
    monkeypatch,
):
    primary_error = RuntimeError("cursor acquisition failed")
    cursor = FakeCursor()
    connection = FakeConnection(
        cursor=cursor,
        cursor_error=primary_error,
    )
    monkeypatch.setattr(dbSetup, "connect_to_postgres", lambda: connection)

    with pytest.raises(RuntimeError) as exc_info:
        dbSetup.init_db()

    assert exc_info.value is primary_error
    assert connection.cursor_count == 1
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert cursor.close_count == 0
    assert connection.close_count == 1


def test_rollback_failure_is_noted_without_masking_primary(monkeypatch):
    primary_error = DocumentChunkSchemaError("primary schema failure")
    rollback_error = RuntimeError("rollback cleanup failure")
    cursor = InitDbCursor()
    connection = FakeConnection(
        cursor=cursor,
        rollback_error=rollback_error,
    )
    monkeypatch.setattr(dbSetup, "connect_to_postgres", lambda: connection)
    monkeypatch.setattr(
        dbSetup,
        "validate_document_chunks_embedding_schema",
        lambda cursor, expected_dimension: _raise(primary_error),
    )

    with pytest.raises(DocumentChunkSchemaError) as exc_info:
        dbSetup.init_db()

    assert exc_info.value is primary_error
    assert connection.rollback_count == 1
    assert cursor.close_count == 1
    assert connection.close_count == 1
    assert any(
        "Rollback failed" in note and "rollback cleanup failure" in note
        for note in _cleanup_notes(primary_error)
    )


def test_cursor_close_failure_is_noted_without_masking_primary(monkeypatch):
    primary_error = DocumentChunkSchemaError("primary schema failure")
    cursor_close_error = RuntimeError("cursor cleanup failure")
    cursor = InitDbCursor(close_error=cursor_close_error)
    connection = FakeConnection(cursor=cursor)
    monkeypatch.setattr(dbSetup, "connect_to_postgres", lambda: connection)
    monkeypatch.setattr(
        dbSetup,
        "validate_document_chunks_embedding_schema",
        lambda cursor, expected_dimension: _raise(primary_error),
    )

    with pytest.raises(DocumentChunkSchemaError) as exc_info:
        dbSetup.init_db()

    assert exc_info.value is primary_error
    assert connection.rollback_count == 1
    assert cursor.close_count == 1
    assert connection.close_count == 1
    assert any(
        "Cursor close failed" in note and "cursor cleanup failure" in note
        for note in _cleanup_notes(primary_error)
    )


def test_connection_close_failure_is_noted_without_masking_primary(
    monkeypatch,
):
    primary_error = DocumentChunkSchemaError("primary schema failure")
    connection_close_error = RuntimeError("connection cleanup failure")
    cursor = InitDbCursor()
    connection = FakeConnection(
        cursor=cursor,
        close_error=connection_close_error,
    )
    monkeypatch.setattr(dbSetup, "connect_to_postgres", lambda: connection)
    monkeypatch.setattr(
        dbSetup,
        "validate_document_chunks_embedding_schema",
        lambda cursor, expected_dimension: _raise(primary_error),
    )

    with pytest.raises(DocumentChunkSchemaError) as exc_info:
        dbSetup.init_db()

    assert exc_info.value is primary_error
    assert connection.rollback_count == 1
    assert cursor.close_count == 1
    assert connection.close_count == 1
    assert any(
        "Connection close failed" in note and "connection cleanup failure" in note
        for note in _cleanup_notes(primary_error)
    )


def test_success_cursor_close_failure_propagates_and_connection_close_runs(
    monkeypatch,
):
    cursor_close_error = RuntimeError("cursor cleanup failure")
    cursor = InitDbCursor(close_error=cursor_close_error)
    connection = FakeConnection(cursor=cursor)
    monkeypatch.setattr(dbSetup, "connect_to_postgres", lambda: connection)

    with pytest.raises(RuntimeError) as exc_info:
        dbSetup.init_db()

    assert exc_info.value is cursor_close_error
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert cursor.close_count == 1
    assert connection.close_count == 1


def test_success_connection_close_failure_propagates(monkeypatch):
    connection_close_error = RuntimeError("connection cleanup failure")
    cursor = InitDbCursor()
    connection = FakeConnection(
        cursor=cursor,
        close_error=connection_close_error,
    )
    monkeypatch.setattr(dbSetup, "connect_to_postgres", lambda: connection)

    with pytest.raises(RuntimeError) as exc_info:
        dbSetup.init_db()

    assert exc_info.value is connection_close_error
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert cursor.close_count == 1
    assert connection.close_count == 1


def test_success_both_close_failures_raise_first_and_note_second(monkeypatch):
    cursor_close_error = RuntimeError("cursor cleanup failure")
    connection_close_error = RuntimeError("connection cleanup failure")
    cursor = InitDbCursor(close_error=cursor_close_error)
    connection = FakeConnection(
        cursor=cursor,
        close_error=connection_close_error,
    )
    monkeypatch.setattr(dbSetup, "connect_to_postgres", lambda: connection)

    with pytest.raises(RuntimeError) as exc_info:
        dbSetup.init_db()

    assert exc_info.value is cursor_close_error
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert cursor.close_count == 1
    assert connection.close_count == 1
    assert any(
        "Connection close failed" in note and "connection cleanup failure" in note
        for note in _cleanup_notes(cursor_close_error)
    )
