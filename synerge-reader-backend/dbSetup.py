import os
import psycopg2
import sys
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

from embedding_config import EMBEDDING_VECTOR_DIMENSION


class VectorColumnSchemaError(RuntimeError):
    """Base class for any vector-column schema validation failure."""


class DocumentChunkSchemaError(VectorColumnSchemaError):
    pass


class KnowledgeBaseSchemaError(VectorColumnSchemaError):
    pass


def validate_vector_column_schema(
    cursor, *, table_name: str, column_name: str, expected_dimension: int
) -> None:
    """
    Generalized version of the existing document-chunks-only validator.
    table_name/column_name are always bound as query parameters, not
    interpolated into SQL text — this works because to_regclass() and the
    pg_attribute lookup both accept these as ordinary data values.
    Never call this with a table_name/column_name that originates from
    user input; in this codebase they are always fixed internal literals
    ("public.document_chunks", "public.knowledge_base", "embedding").
    """
    try:
        cursor.execute("SELECT to_regclass(%s)::oid", (table_name,))
        row = cursor.fetchone()
        table_oid = row[0] if row else None

        if table_oid is None:
            return  # table doesn't exist yet — let CREATE TABLE proceed normally, nothing to validate

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
            (table_oid, column_name),
        )
        column_exists = cursor.fetchone()[0]

        if not column_exists:
            raise VectorColumnSchemaError(
                f"{table_name} exists but is missing its '{column_name}' column. "
                "Refusing to recreate or drop the table automatically; run an "
                "explicit migration or reindex operation."
            )

        cursor.execute(
            """
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = %s
              AND attname = %s
              AND NOT attisdropped
            """,
            (table_oid, column_name),
        )
        row = cursor.fetchone()
        actual_dimension = row[0] if row else None

        if type(actual_dimension) is not int or actual_dimension <= 0:
            raise VectorColumnSchemaError(
                f"{table_name}.{column_name} dimension metadata is missing or "
                "malformed. Refusing to modify or recreate the table automatically."
            )

        if actual_dimension != expected_dimension:
            raise VectorColumnSchemaError(
                f"{table_name}.{column_name} dimension is {actual_dimension}, "
                f"expected {expected_dimension}. Refusing to modify or recreate "
                "the table automatically; run an explicit migration to reindex "
                "the column."
            )
    except VectorColumnSchemaError:
        raise
    except Exception as exc:
        raise VectorColumnSchemaError(
            f"Failed to validate {table_name}.{column_name} schema safely."
        ) from exc


def validate_document_chunks_embedding_schema(
    cursor, expected_dimension: int = EMBEDDING_VECTOR_DIMENSION
) -> None:
    """Thin wrapper preserving the existing public API and existing test imports."""
    try:
        validate_vector_column_schema(
            cursor,
            table_name="public.document_chunks",
            column_name="embedding",
            expected_dimension=expected_dimension,
        )
    except VectorColumnSchemaError as exc:
        raise DocumentChunkSchemaError(str(exc)) from exc


def validate_knowledge_base_embedding_schema(
    cursor, expected_dimension: int = EMBEDDING_VECTOR_DIMENSION
) -> None:
    try:
        validate_vector_column_schema(
            cursor,
            table_name="public.knowledge_base",
            column_name="embedding",
            expected_dimension=expected_dimension,
        )
    except VectorColumnSchemaError as exc:
        raise KnowledgeBaseSchemaError(str(exc)) from exc


def ensure_document_chunks_locator_columns(cursor) -> None:
    """Add nullable locator columns when absent.

    This migration never drops tables or columns and never rewrites
    or deletes existing rows.
    """
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




def init_db():
    conn = connect_to_postgres()
    if conn is None:
        print(" Failed to connect to PostgreSQL. Exiting")
        sys.exit(1)

    cursor = conn.cursor()



    
    # Users 
    cursor.execute("""
    CREATE EXTENSION IF NOT EXISTS "pgcrypto";
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    token TEXT,
    is_admin INTEGER DEFAULT 0
    )
    """)




    # Documents
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

    # Document chunks - validate schema without destructive mutation
    validate_document_chunks_embedding_schema(cursor)

    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            document_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_index INTEGER,
            embedding vector({EMBEDDING_VECTOR_DIMENSION}),
            page_start INTEGER,
            page_end INTEGER,
            locator_json JSONB,
            FOREIGN KEY (document_id)
                REFERENCES documents (id)
                ON DELETE CASCADE
        )
        """
    )

    ensure_document_chunks_locator_columns(cursor)

    # Chat history
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

    # Knowledge base — with semantic matching, source attribution, usage tracking
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
        embedding vector({EMBEDDING_VECTOR_DIMENSION}),
        FOREIGN KEY (chat_history_id) REFERENCES chat_history (id)
    )
    """)

    # Add new columns to existing knowledge_base table if they don't exist.
    # Every statement here is ADD COLUMN IF NOT EXISTS, which is inherently
    # idempotent/safe to retry — no try/except/rollback here. A genuine
    # failure propagates to init_db()'s caller instead of being silently
    # swallowed (the old rollback() here used to roll back the entire
    # transaction, including the document_chunks/chat_history tables
    # already created earlier in this same call).
    for col, definition in [
        ("corrected_by", "TEXT"),
        ("usage_count", "INTEGER DEFAULT 0"),
        ("embedding", f"vector({EMBEDDING_VECTOR_DIMENSION})"),
    ]:
        cursor.execute(f"""
            ALTER TABLE knowledge_base ADD COLUMN IF NOT EXISTS {col} {definition}
        """)

    validate_knowledge_base_embedding_schema(cursor)



  



    conn.commit()
    conn.close()

