import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dbSetup import ensure_document_chunks_locator_columns


FORBIDDEN_DATA_LOSS_SQL = re.compile(
    r"\b(DROP\s+TABLE|DELETE\s+FROM|TRUNCATE)\b", re.IGNORECASE
)
# Note: ALTER TABLE is deliberately NOT forbidden here — it is this
# migration's entire purpose. Do not reuse test_db_schema_safety.py's
# stricter helper, which does forbid it; define this pattern locally and
# do not import anything from that other test module.


class FakeCursor:
    def __init__(self):
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))


def test_locator_migration_uses_exact_additive_columns():
    cursor = FakeCursor()

    ensure_document_chunks_locator_columns(cursor)

    normalized = [" ".join(sql.split()) for sql, _params in cursor.executed]

    assert normalized == [
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS page_start INTEGER",
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS page_end INTEGER",
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS locator_json JSONB",
    ]

    for sql in normalized:
        assert not FORBIDDEN_DATA_LOSS_SQL.search(sql)


def test_locator_migration_is_repeatable_by_construction():
    cursor = FakeCursor()

    ensure_document_chunks_locator_columns(cursor)
    ensure_document_chunks_locator_columns(cursor)

    normalized = [" ".join(sql.split()) for sql, _params in cursor.executed]

    expected = [
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS page_start INTEGER",
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS page_end INTEGER",
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS locator_json JSONB",
    ]

    assert normalized == expected * 2

    for sql in normalized:
        assert not FORBIDDEN_DATA_LOSS_SQL.search(sql)
