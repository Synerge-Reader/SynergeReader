"""Tests for migrate_embeddings_to_profile.py.

Two layers:

* Pure-logic tests (always run): vector validation, the cutover gate, naming,
  and the operator-facing instructions. No database, network or subprocess.

* Integration tests (run only when SYNERGE_MIGRATION_TEST_DSN is set): the
  real fill / cutover code against a real Postgres with pgvector, inside a
  throw-away schema that is dropped afterwards -- nothing outside that schema
  is read or written. They skip otherwise, so the guarded, no-database test
  runs used elsewhere in this directory are unaffected. Example:

      SYNERGE_MIGRATION_TEST_DSN=postgresql://user@localhost:5432/somedb \
          python -m pytest tests/test_migrate_embeddings_script.py

  The integration tests are what prove the properties that matter for a
  destructive migration: the live column is untouched until cutover, an
  interrupted or failed fill changes nothing the app uses, the swap is
  all-or-nothing across both tables, and it refuses to swap when any embeddable
  row is missing a vector.
"""

import math
import os
import uuid
from pathlib import Path

import pytest

import migrate_embeddings_to_profile as mig
from ollama_embedding_provider import EmbeddingProviderError

_SCRIPT = Path(mig.__file__)

# ------------------------------------------------------------ pure logic ----


def test_vector_ok_accepts_only_right_sized_finite_nonzero_vectors():
    assert mig.vector_ok([0.1, 0.2, 0.3], 3)
    assert not mig.vector_ok([0.1, 0.2], 3)  # wrong size
    assert not mig.vector_ok([0.0, 0.0, 0.0], 3)  # all zero "looks valid", matches nothing
    assert not mig.vector_ok([0.1, math.nan, 0.3], 3)
    assert not mig.vector_ok([0.1, math.inf, 0.3], 3)
    assert not mig.vector_ok([0.1, "x", 0.3], 3)
    assert not mig.vector_ok([True, 0.1, 0.2], 3)
    assert not mig.vector_ok(None, 3)


@pytest.mark.parametrize(
    "failed,missing,allow,expected",
    [
        (0, 0, False, True),  # everything embedded
        (0, 0, True, True),
        (2, 2, False, False),  # failures block by default
        (2, 2, True, True),  # ...unless the operator explicitly accepts them
        (0, 1, True, False),  # a row with no vector that is NOT a known failure (written mid-run) always blocks
        (1, 3, True, False),  # more missing than failed: some are unexplained -> block
    ],
)
def test_cutover_gate(failed, missing, allow, expected):
    assert mig.cutover_allowed(failed, missing, allow) is expected


def test_needs_migration_and_legacy_naming():
    assert mig.needs_migration(768, 1024)
    assert not mig.needs_migration(1024, 1024)
    assert mig.needs_migration(None, 1024)
    assert mig.legacy_column_name(768) == "embedding_legacy_768"
    assert mig.legacy_column_name(None) == "embedding_legacy_unknown"


def test_instructions_use_a_one_off_container_not_exec():
    doc = mig.__doc__
    assert "docker-compose run --rm backend python migrate_embeddings_to_profile.py" in doc
    # `exec` can't attach to a crash-looping container, and the server has the v1 binary.
    assert "docker compose exec backend python" not in doc
    assert "docker-compose exec backend python" not in doc
    assert "pg_dump" in doc, "the docs must tell the operator to take a backup first"


def test_script_exits_nonzero_on_failure_paths():
    src = _SCRIPT.read_text(encoding="utf-8")
    assert "sys.exit(main())" in src
    # every early-out that means "did not complete" returns 1 from main()
    assert src.count("return 1") >= 6


# ------------------------------------------------------ integration (DB) ----

_DSN = os.environ.get("SYNERGE_MIGRATION_TEST_DSN")
needs_db = pytest.mark.skipif(not _DSN, reason="set SYNERGE_MIGRATION_TEST_DSN to run the database tests")

OLD_DIM, NEW_DIM = 3, 4


class FakeProvider:
    """Deterministic embed_documents(); can be told to fail on chosen texts or
    to return a bad vector for them."""

    def __init__(self, fail_on=(), zero_on=()):
        self.fail_on = set(fail_on)
        self.zero_on = set(zero_on)
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            if t in self.fail_on:
                raise EmbeddingProviderError(f"cannot embed {t!r}")
            out.append([0.0] * NEW_DIM if t in self.zero_on else [float(len(t) + i + 1) for i in range(NEW_DIM)])
        return out


@pytest.fixture()
def db():
    import psycopg2
    from pgvector.psycopg2 import register_vector

    conn = psycopg2.connect(_DSN)
    schema = f"mig_test_{uuid.uuid4().hex[:10]}"
    cur = conn.cursor()
    cur.execute(f"CREATE SCHEMA {schema}")
    cur.execute(f"SET search_path TO {schema}, public")
    register_vector(conn)
    cur.execute(f"CREATE TABLE document_chunks (id serial PRIMARY KEY, chunk_text text, embedding vector({OLD_DIM}))")
    cur.execute(f"CREATE TABLE knowledge_base (id serial PRIMARY KEY, question text, embedding vector({OLD_DIM}))")
    for t in ("a", "bb", "   ", "cccc"):  # one blank row: not embeddable, must not count as a failure
        cur.execute("INSERT INTO document_chunks (chunk_text, embedding) VALUES (%s, %s)", (t, [1.0, 2.0, 3.0]))
    for q in ("q1", "q22"):
        cur.execute("INSERT INTO knowledge_base (question, embedding) VALUES (%s, %s)", (q, [1.0, 2.0, 3.0]))
    conn.commit()
    try:
        yield conn, cur
    finally:
        conn.rollback()
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
        conn.commit()
        conn.close()


def _add_shadow(cur, conn, *tables):
    for t in tables:
        cur.execute(f"ALTER TABLE {t} ADD COLUMN {mig.SHADOW_COLUMN} vector({NEW_DIM})")
    conn.commit()


def _fill(cur, conn, provider, table, id_col, text_col):
    return mig.fill_shadow(cur, conn, provider, table=table, id_col=id_col, text_col=text_col,
                           dimension=NEW_DIM, batch_size=2, log=lambda *_: None)


@needs_db
def test_fill_writes_only_the_shadow_column_and_skips_blank_rows(db):
    conn, cur = db
    _add_shadow(cur, conn, "document_chunks")
    written, failures = _fill(cur, conn, FakeProvider(), "document_chunks", "id", "chunk_text")
    assert (written, failures) == (3, [])
    # live column untouched: still the old dimension, still the old values
    assert mig.column_dimension(cur, "document_chunks", "embedding") == OLD_DIM
    cur.execute("SELECT count(*) FROM document_chunks WHERE embedding IS NOT NULL")
    assert cur.fetchone()[0] == 4
    # blank row has no new vector and is not "missing" (nothing to embed)
    assert mig.null_rows(cur, "document_chunks", "chunk_text", mig.SHADOW_COLUMN) == 0
    assert mig.embeddable_rows(cur, "document_chunks", "chunk_text") == 3


@needs_db
def test_fill_is_resumable_and_does_not_redo_finished_rows(db):
    conn, cur = db
    _add_shadow(cur, conn, "document_chunks")
    _fill(cur, conn, FakeProvider(), "document_chunks", "id", "chunk_text")
    again = FakeProvider()
    written, failures = _fill(cur, conn, again, "document_chunks", "id", "chunk_text")
    assert (written, failures) == (0, []) and again.calls == []


@needs_db
def test_failed_and_unusable_rows_are_reported_and_left_null(db):
    conn, cur = db
    _add_shadow(cur, conn, "document_chunks")
    provider = FakeProvider(fail_on={"bb"}, zero_on={"cccc"})
    written, failures = _fill(cur, conn, provider, "document_chunks", "id", "chunk_text")
    assert written == 1 and len(failures) == 2
    assert mig.null_rows(cur, "document_chunks", "chunk_text", mig.SHADOW_COLUMN) == 2


@needs_db
def test_cutover_swaps_both_tables_and_keeps_the_old_vectors(db):
    conn, cur = db
    _add_shadow(cur, conn, "document_chunks", "knowledge_base")
    _fill(cur, conn, FakeProvider(), "document_chunks", "id", "chunk_text")
    _fill(cur, conn, FakeProvider(), "knowledge_base", "id", "question")
    mig.cutover(conn, cur, [("document_chunks", "chunk_text", OLD_DIM), ("knowledge_base", "question", OLD_DIM)],
                allow_failed_rows=False, failed_by_table={})
    for table in ("document_chunks", "knowledge_base"):
        assert mig.column_dimension(cur, table, "embedding") == NEW_DIM
        assert mig.column_dimension(cur, table, "embedding_legacy_3") == OLD_DIM, "rollback column must survive"
        assert not mig.column_exists(cur, table, mig.SHADOW_COLUMN)
    cur.execute("SELECT count(*) FROM document_chunks WHERE embedding_legacy_3 IS NOT NULL")
    assert cur.fetchone()[0] == 4


@needs_db
def test_cutover_refuses_when_failures_exist_and_changes_nothing(db):
    conn, cur = db
    _add_shadow(cur, conn, "document_chunks")
    _, failures = _fill(cur, conn, FakeProvider(fail_on={"bb"}), "document_chunks", "id", "chunk_text")
    with pytest.raises(mig.MigrationBlocked):
        mig.cutover(conn, cur, [("document_chunks", "chunk_text", OLD_DIM)],
                    allow_failed_rows=False, failed_by_table={"document_chunks": len(failures)})
    assert mig.column_dimension(cur, "document_chunks", "embedding") == OLD_DIM
    assert mig.column_exists(cur, "document_chunks", mig.SHADOW_COLUMN), "shadow progress is kept for the re-run"


@needs_db
def test_allow_failed_rows_accepts_only_the_known_failures(db):
    conn, cur = db
    _add_shadow(cur, conn, "document_chunks")
    _, failures = _fill(cur, conn, FakeProvider(fail_on={"bb"}), "document_chunks", "id", "chunk_text")
    mig.cutover(conn, cur, [("document_chunks", "chunk_text", OLD_DIM)],
                allow_failed_rows=True, failed_by_table={"document_chunks": len(failures)})
    assert mig.column_dimension(cur, "document_chunks", "embedding") == NEW_DIM
    assert mig.null_rows(cur, "document_chunks", "chunk_text", "embedding") == 1  # the failed row stays NULL


@needs_db
def test_row_written_after_the_fill_blocks_even_with_allow_failed_rows(db):
    conn, cur = db
    _add_shadow(cur, conn, "document_chunks")
    _fill(cur, conn, FakeProvider(), "document_chunks", "id", "chunk_text")
    # the live app (or anything else) writes a new row after the fill finished
    cur.execute("INSERT INTO document_chunks (chunk_text, embedding) VALUES ('late', %s)", ([9.0, 9.0, 9.0],))
    conn.commit()
    with pytest.raises(mig.MigrationBlocked):
        mig.cutover(conn, cur, [("document_chunks", "chunk_text", OLD_DIM)],
                    allow_failed_rows=True, failed_by_table={"document_chunks": 0})
    assert mig.column_dimension(cur, "document_chunks", "embedding") == OLD_DIM


@needs_db
def test_cutover_is_all_or_nothing_across_tables(db):
    conn, cur = db
    _add_shadow(cur, conn, "document_chunks", "knowledge_base")
    _fill(cur, conn, FakeProvider(), "document_chunks", "id", "chunk_text")  # chunks fully ready
    # knowledge_base deliberately NOT filled -> its check fails
    with pytest.raises(mig.MigrationBlocked):
        mig.cutover(conn, cur, [("document_chunks", "chunk_text", OLD_DIM), ("knowledge_base", "question", OLD_DIM)],
                    allow_failed_rows=False, failed_by_table={})
    # document_chunks must NOT have been swapped on its own
    assert mig.column_dimension(cur, "document_chunks", "embedding") == OLD_DIM
    assert mig.column_exists(cur, "document_chunks", mig.SHADOW_COLUMN)
    assert not mig.column_exists(cur, "document_chunks", "embedding_legacy_3")


class _FailOnSecondTableSwap:
    """Cursor proxy that lets everything through except renaming the second
    table's shadow column -- i.e. a failure *during* the swap, after the first
    table has already been renamed inside the same transaction."""

    def __init__(self, real):
        self._real = real

    def execute(self, sql, params=None):
        if sql.startswith("ALTER TABLE knowledge_base RENAME COLUMN embedding_new"):
            raise RuntimeError("simulated failure mid-swap")
        return self._real.execute(sql, params)

    def __getattr__(self, name):
        return getattr(self._real, name)


@needs_db
def test_failure_during_the_swap_rolls_back_the_tables_already_renamed(db):
    conn, cur = db
    _add_shadow(cur, conn, "document_chunks", "knowledge_base")
    _fill(cur, conn, FakeProvider(), "document_chunks", "id", "chunk_text")
    _fill(cur, conn, FakeProvider(), "knowledge_base", "id", "question")
    with pytest.raises(RuntimeError, match="simulated"):
        mig.cutover(conn, _FailOnSecondTableSwap(cur),
                    [("document_chunks", "chunk_text", OLD_DIM), ("knowledge_base", "question", OLD_DIM)],
                    allow_failed_rows=False, failed_by_table={})
    # document_chunks was renamed first inside the transaction; the rollback must have undone it
    assert mig.column_dimension(cur, "document_chunks", "embedding") == OLD_DIM
    assert mig.column_exists(cur, "document_chunks", mig.SHADOW_COLUMN)
    assert not mig.column_exists(cur, "document_chunks", "embedding_legacy_3")
    assert mig.column_dimension(cur, "knowledge_base", "embedding") == OLD_DIM
