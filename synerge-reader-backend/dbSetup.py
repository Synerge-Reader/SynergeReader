import os
import psycopg2
import sys
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector


EMBEDDING_VECTOR_DIMENSION = 768


class VectorSchemaError(RuntimeError):
    """Raised when an existing vector column cannot be validated safely."""


class DocumentChunkSchemaError(VectorSchemaError):
    """Raised when document_chunks cannot be validated safely."""


class KnowledgeBaseSchemaError(VectorSchemaError):
    """Raised when knowledge_base cannot be validated safely."""


def _require_positive_int_dimension(value, name: str) -> int:
    """Return a valid vector dimension or raise a named ValueError."""
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"{name} must be a positive integer; received {value!r}."
        )
    return value


# Embedding-column migration policy: a missing or incompatible embedding
# column requires a separately reviewed migration. Startup must never add,
# drop, or replace an embedding column automatically. Adding one to a
# populated table would leave existing rows with NULL vectors while falsely
# suggesting that those rows are indexed. Nullable locator fields are
# different: they are additive provenance metadata and do not change vector
# identity or the existing retrievability of a row.
def _validate_embedding_schema(
    cursor,
    table_name: str,
    error_type: type[VectorSchemaError],
    expected_dimension: int,
) -> None:
    """Validate one existing pgvector column without changing its schema."""
    try:
        cursor.execute("SELECT to_regclass(%s)::oid", (table_name,))
        row = cursor.fetchone()
        table_oid = row[0] if row else None

        if table_oid is None:
            return

        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_attribute
                WHERE attrelid = %s
                  AND attname = %s
                  AND NOT attisdropped
            )
            """,
            (table_oid, "embedding"),
        )
        row = cursor.fetchone()
        embedding_exists = bool(row and row[0] is True)

        if not embedding_exists:
            raise error_type(
                f"{table_name} exists but is missing its 'embedding' column. "
                "Refusing to modify or recreate it automatically; run an "
                "explicit reviewed migration."
            )

        cursor.execute(
            """
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = %s
              AND attname = %s
              AND NOT attisdropped
            """,
            (table_oid, "embedding"),
        )
        row = cursor.fetchone()
        actual_dimension = row[0] if row else None

        if type(actual_dimension) is not int or actual_dimension <= 0:
            raise error_type(
                f"{table_name} embedding dimension metadata is missing or "
                "malformed. Refusing to modify or recreate it automatically."
            )

        if actual_dimension != expected_dimension:
            raise error_type(
                f"{table_name} embedding dimension is {actual_dimension}, "
                f"expected {expected_dimension}. Refusing to modify or recreate "
                "it automatically; run an explicit reviewed migration and reindex."
            )
    except error_type:
        raise
    except Exception as exc:
        raise error_type(
            f"Failed to validate {table_name} embedding schema safely."
        ) from exc


def validate_document_chunks_embedding_schema(
    cursor,
    expected_dimension: int = EMBEDDING_VECTOR_DIMENSION,
) -> None:
    expected_dimension = _require_positive_int_dimension(
        expected_dimension,
        "expected_dimension",
    )
    _validate_embedding_schema(
        cursor,
        "document_chunks",
        DocumentChunkSchemaError,
        expected_dimension,
    )


def validate_knowledge_base_embedding_schema(
    cursor,
    expected_dimension: int = EMBEDDING_VECTOR_DIMENSION,
) -> None:
    expected_dimension = _require_positive_int_dimension(
        expected_dimension,
        "expected_dimension",
    )
    _validate_embedding_schema(
        cursor,
        "knowledge_base",
        KnowledgeBaseSchemaError,
        expected_dimension,
    )


def ensure_document_chunks_locator_columns(cursor) -> None:
    """Add nullable locator columns without deleting or rewriting row data."""
    for col, definition in [
        ("page_start", "INTEGER"),
        ("page_end", "INTEGER"),
        ("locator_json", "JSONB"),
    ]:
        cursor.execute(
            f"ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS {col} {definition}"
        )


def connect_to_postgres():
    load_dotenv()
    connection = None
    cursor = None
    try:
        connection_string = os.getenv("DB_CONNECTION_STRING")
        print('Connecting to the PostgreSQL database...')
        connection = psycopg2.connect(connection_string)
        cursor = connection.cursor()
        cursor.execute('CREATE EXTENSION IF NOT EXISTS vector;')
        connection.commit()
        register_vector(connection)
        return connection

    except psycopg2.DatabaseError as error:
        print(f"Database error: {error}")
        return None
    finally:
        if cursor is not None and not cursor.closed:
            cursor.close()


def test_postgres_connection():
    load_dotenv()
    connection = None
    cursor = None
    try:
        connection = psycopg2.connect(os.getenv("DB_CONNECTION_STRING"))
        cursor = connection.cursor()
        cursor.execute('CREATE EXTENSION IF NOT EXISTS vector;')
        connection.commit()
        register_vector(connection)
        cursor.execute("SELECT version()")
        print(cursor.fetchone())
    finally:
        if cursor is not None and not cursor.closed:
            cursor.close()
        if connection is not None:
            connection.close()




def init_db(
    expected_dimension: int = EMBEDDING_VECTOR_DIMENSION,
):
    expected_dimension = _require_positive_int_dimension(
        expected_dimension,
        "expected_dimension",
    )
    conn = connect_to_postgres()
    if conn is None:
        print(" Failed to connect to PostgreSQL. Exiting")
        sys.exit(1)

    cursor = None
    try:
        cursor = conn.cursor()

        # Create dependency tables before the tables that reference them.
        cursor.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto";')

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            token TEXT,
            is_admin INTEGER DEFAULT 0
        )
        """)

        # Existing accounts remain verified by default; new registrations set
        # email_verified explicitly when they are created.
        for col, definition in [
            ("email", "TEXT"),
            ("is_active", "INTEGER DEFAULT 1"),
            ("email_verified", "INTEGER DEFAULT 1"),
            ("email_verification_token", "TEXT"),
            ("email_verification_expires", "TEXT"),
            ("password_reset_token", "TEXT"),
            ("password_reset_expires", "TEXT"),
        ]:
            cursor.execute(
                f"ALTER TABLE users ADD COLUMN IF NOT EXISTS {col} {definition}"
            )

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            upload_timestamp TEXT NOT NULL,
            content TEXT NOT NULL,
            author TEXT,
            title TEXT,
            publication_date TEXT,
            source TEXT,
            doi_url TEXT
        )
        """)
        cursor.execute(
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS "
            "user_id UUID REFERENCES users(id)"
        )

        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            document_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_index INTEGER,
            embedding vector({expected_dimension}),
            page_start INTEGER,
            page_end INTEGER,
            locator_json JSONB,
            FOREIGN KEY (document_id)
                REFERENCES documents (id)
                ON DELETE CASCADE
        )
        """)
        validate_document_chunks_embedding_schema(
            cursor,
            expected_dimension=expected_dimension,
        )
        ensure_document_chunks_locator_columns(cursor)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            user_id UUID,
            ts TEXT NOT NULL,
            selected_text TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            rating INTEGER,
            comment TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """)

        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id SERIAL PRIMARY KEY,
            question TEXT NOT NULL,
            original_answer TEXT,
            corrected_answer TEXT NOT NULL,
            created_at TEXT,
            chat_history_id INTEGER,
            context_text TEXT,
            corrected_by TEXT,
            usage_count INTEGER DEFAULT 0,
            embedding vector({expected_dimension}),
            source_type TEXT DEFAULT 'document',
            FOREIGN KEY (chat_history_id) REFERENCES chat_history (id)
        )
        """)
        validate_knowledge_base_embedding_schema(
            cursor,
            expected_dimension=expected_dimension,
        )

        for col, definition in [
            ("corrected_by", "TEXT"),
            ("usage_count", "INTEGER DEFAULT 0"),
            ("source_type", "TEXT DEFAULT 'document'"),
        ]:
            cursor.execute(
                f"ALTER TABLE knowledge_base "
                f"ADD COLUMN IF NOT EXISTS {col} {definition}"
            )

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS admin_audit_log (
            id SERIAL PRIMARY KEY,
            ts TEXT NOT NULL,
            actor_id UUID,
            actor_username TEXT,
            action TEXT NOT NULL,
            target_id UUID,
            target_username TEXT,
            detail TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_insights (
            id SERIAL PRIMARY KEY,
            document_id INTEGER NOT NULL UNIQUE
                REFERENCES documents(id)
                ON DELETE CASCADE,
            doc_type TEXT,
            keywords JSONB DEFAULT '[]',
            facts JSONB DEFAULT '[]',
            entities JSONB DEFAULT '[]',
            created_at TEXT
        )
        """)

        conn.commit()
    except BaseException as primary_error:
        try:
            conn.rollback()
        except BaseException as cleanup_error:
            primary_error.add_note(
                f"Rollback failed during init_db cleanup: {cleanup_error!r}"
            )

        if cursor is not None:
            try:
                cursor.close()
            except BaseException as cleanup_error:
                primary_error.add_note(
                    f"Cursor close failed during init_db cleanup: {cleanup_error!r}"
                )

        try:
            conn.close()
        except BaseException as cleanup_error:
            primary_error.add_note(
                f"Connection close failed during init_db cleanup: {cleanup_error!r}"
            )

        raise

    cleanup_error = None
    if cursor is not None:
        try:
            cursor.close()
        except BaseException as error:
            cleanup_error = error

    try:
        conn.close()
    except BaseException as error:
        if cleanup_error is None:
            cleanup_error = error
        else:
            cleanup_error.add_note(
                f"Connection close failed during init_db cleanup: {error!r}"
            )

    if cleanup_error is not None:
        raise cleanup_error

