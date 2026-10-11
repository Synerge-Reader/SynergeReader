"""Contracts for the additive chat_history provenance columns (CP2A).

Two layers, following tests/test_migrate_embeddings_script.py:

* Statement tests (always run): the exact SQL ``init_db`` and
  ``ensure_chat_history_provenance_columns`` issue, against a recording fake
  cursor. No database, network, or subprocess.

* Integration tests (run only when SYNERGE_MIGRATION_TEST_DSN is set): the
  real statements against a real PostgreSQL, inside a throw-away schema that
  is dropped afterwards -- nothing outside it is read or written. They skip
  otherwise. Example:

      SYNERGE_MIGRATION_TEST_DSN=postgresql://user@localhost:5432/somedb \\
          python -m pytest tests/test_chat_history_provenance_migration.py

What these prove: the change only ADDS two nullable columns with no default
(so no table rewrite and no backfill); it is idempotent; nothing in startup
updates, deletes, or recreates chat_history; a legacy row keeps every value
and reads back with NULL provenance; and a new row's JSONB provenance round
trips.
"""

import os
import re
import sys
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dbSetup
from dbSetup import EMBEDDING_VECTOR_DIMENSION, ensure_chat_history_provenance_columns


_EXPECTED_ALTERS = [
    "ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS trace_id UUID",
    "ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS answer_provenance JSONB",
]

# ALTER TABLE is this change's purpose, so it is not forbidden here; anything
# that removes, rewrites, or backfills data is.
_DESTRUCTIVE_OR_BACKFILL_SQL = re.compile(
    r"\b(DROP\s+TABLE|DROP\s+COLUMN|DELETE\s+FROM|TRUNCATE|UPDATE\s+chat_history|"
    r"ALTER\s+COLUMN|SET\s+DEFAULT|RENAME)\b",
    re.IGNORECASE,
)


def _normalized(sql):
    return " ".join(sql.split())


class _RecordingCursor:
    def __init__(self):
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))


# --- statement layer ----------------------------------------------------------


def test_the_provenance_columns_are_exactly_two_nullable_additions():
    cursor = _RecordingCursor()

    ensure_chat_history_provenance_columns(cursor)

    assert [_normalized(sql) for sql, _ in cursor.executed] == _EXPECTED_ALTERS
    for sql, params in cursor.executed:
        assert params is None
        assert "NOT NULL" not in sql.upper() and "DEFAULT" not in sql.upper(), (
            "a nullable column with no default is a catalog-only change: no rewrite"
        )
        assert not _DESTRUCTIVE_OR_BACKFILL_SQL.search(sql)


def test_the_provenance_migration_is_repeatable_by_construction():
    cursor = _RecordingCursor()

    ensure_chat_history_provenance_columns(cursor)
    ensure_chat_history_provenance_columns(cursor)

    assert [_normalized(sql) for sql, _ in cursor.executed] == _EXPECTED_ALTERS * 2


class _InitDbCursor(_RecordingCursor):
    """Answers init_db's catalog checks with a valid existing schema."""

    def __init__(self):
        super().__init__()
        self._next = None

    def execute(self, sql, params=None):
        super().execute(sql, params)
        normalized = _normalized(sql)
        if normalized.startswith("SELECT to_regclass"):
            self._next = (101,)
        elif "SELECT EXISTS" in normalized:
            self._next = (True,)
        elif normalized.startswith("SELECT atttypmod"):
            self._next = (EMBEDDING_VECTOR_DIMENSION,)

    def fetchone(self):
        result, self._next = self._next, None
        return result

    def close(self):
        pass


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        pass


def test_init_db_adds_the_columns_after_creating_chat_history_in_one_transaction(monkeypatch):
    cursor = _InitDbCursor()
    connection = _Connection(cursor)
    monkeypatch.setattr(dbSetup, "connect_to_postgres", lambda: connection)

    dbSetup.init_db()

    statements = [_normalized(sql) for sql, _ in cursor.executed]
    create = next(i for i, sql in enumerate(statements) if "CREATE TABLE IF NOT EXISTS chat_history" in sql)
    alters = [i for i, sql in enumerate(statements) if sql in _EXPECTED_ALTERS]
    knowledge_base = next(i for i, sql in enumerate(statements) if "CREATE TABLE IF NOT EXISTS knowledge_base" in sql)

    assert [statements[i] for i in alters] == _EXPECTED_ALTERS
    assert create < alters[0] < alters[1] < knowledge_base
    # A fresh database gets the same nullable columns from CREATE TABLE.
    assert "trace_id UUID, answer_provenance JSONB" in statements[create]
    assert connection.commits == 1 and connection.rollbacks == 0
    for sql in statements:
        if "chat_history" in sql:
            assert not _DESTRUCTIVE_OR_BACKFILL_SQL.search(sql), sql
            assert not sql.startswith(("UPDATE", "DELETE", "INSERT")), "startup never rewrites a history row"


# --- integration layer (real PostgreSQL, opt-in) --------------------------------

_DSN = os.environ.get("SYNERGE_MIGRATION_TEST_DSN")
needs_db = pytest.mark.skipif(not _DSN, reason="set SYNERGE_MIGRATION_TEST_DSN to run the database tests")

_LEGACY_ROW = ("2026-09-01T10:00:00", "selected", "What is the term?", "Two years [C1].", 4, "useful")


@pytest.fixture()
def legacy_db():
    """A throw-away schema holding chat_history exactly as it was before CP2A."""
    import psycopg2

    conn = psycopg2.connect(_DSN)
    schema = f"prov_test_{uuid.uuid4().hex[:10]}"
    cur = conn.cursor()
    cur.execute(f"CREATE SCHEMA {schema}")
    cur.execute(f"SET search_path TO {schema}")
    cur.execute(
        """
        CREATE TABLE chat_history (
            id SERIAL PRIMARY KEY,
            user_id UUID,
            ts TEXT NOT NULL,
            selected_text TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            rating INTEGER,
            comment TEXT
        )
        """
    )
    cur.execute(
        "INSERT INTO chat_history (ts, selected_text, question, answer, rating, comment) "
        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
        _LEGACY_ROW,
    )
    legacy_id = cur.fetchone()[0]
    conn.commit()
    try:
        yield conn, cur, legacy_id
    finally:
        conn.rollback()
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
        conn.commit()
        conn.close()


@needs_db
def test_the_columns_are_added_nullable_without_defaults_and_twice_is_harmless(legacy_db):
    conn, cur, _ = legacy_db

    ensure_chat_history_provenance_columns(cur)
    conn.commit()
    ensure_chat_history_provenance_columns(cur)
    conn.commit()

    cur.execute(
        """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = 'chat_history'
          AND column_name IN ('trace_id', 'answer_provenance')
        ORDER BY column_name
        """
    )
    assert cur.fetchall() == [
        ("answer_provenance", "jsonb", "YES", None),
        ("trace_id", "uuid", "YES", None),
    ]


@needs_db
def test_a_legacy_row_is_unchanged_and_reads_back_as_not_recorded(legacy_db):
    conn, cur, legacy_id = legacy_db
    columns = "ts, selected_text, question, answer, rating, comment"
    cur.execute(f"SELECT {columns}, xmin::text FROM chat_history WHERE id = %s", (legacy_id,))
    before = cur.fetchone()

    ensure_chat_history_provenance_columns(cur)
    conn.commit()

    cur.execute(
        f"SELECT {columns}, xmin::text, trace_id, answer_provenance FROM chat_history WHERE id = %s",
        (legacy_id,),
    )
    after = cur.fetchone()
    assert after[:6] == _LEGACY_ROW
    assert after[6] == before[6], "the row was not rewritten (same tuple version)"
    assert after[7:] == (None, None), "not recorded -- never backfilled with invented values"


@needs_db
def test_a_new_rows_provenance_round_trips_as_jsonb(legacy_db):
    conn, cur, _ = legacy_db
    ensure_chat_history_provenance_columns(cur)
    trace_id = str(uuid.uuid4())

    cur.execute(
        "INSERT INTO chat_history (ts, selected_text, question, answer, trace_id, answer_provenance) "
        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
        ("2026-10-04T12:00:00", "", "q", "a", trace_id, '{"schema_version":1,"trace_id":"%s"}' % trace_id),
    )
    new_id = cur.fetchone()[0]
    conn.commit()

    cur.execute("SELECT trace_id::text, answer_provenance FROM chat_history WHERE id = %s", (new_id,))
    stored_trace, provenance = cur.fetchone()
    assert stored_trace == trace_id
    assert provenance == {"schema_version": 1, "trace_id": trace_id}
