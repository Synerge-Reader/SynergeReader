import os
import psycopg2
import sys
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector


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

    # Add columns that were introduced after the original table creation.
    # email_verified defaults to 1 (verified) so every account that already
    # existed before this feature stays usable — only new registrations get
    # created with it explicitly set to 0, gating them until they click the
    # verification link.
    for col, definition in [
        ("email", "TEXT"),
        ("is_active", "INTEGER DEFAULT 1"),
        ("email_verified", "INTEGER DEFAULT 1"),
        ("email_verification_token", "TEXT"),
        ("email_verification_expires", "TEXT"),
        ("password_reset_token", "TEXT"),
        ("password_reset_expires", "TEXT"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE users ADD COLUMN IF NOT EXISTS {col} {definition}")
        except Exception as e:
            print(f"Column {col} may already exist: {e}")
            conn.rollback()




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

    try:
        cursor.execute(
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id)"
        )
    except Exception as e:
        print(f"Column user_id may already exist: {e}")
        conn.rollback()

    # Document chunks - ensure correct schema
    try:
        cursor.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'document_chunks' AND column_name = 'embedding'
        )
        """)
        embedding_exists = cursor.fetchone()[0]
        
        # If table exists but embedding column doesn't, drop and recreate
        if not embedding_exists:
            cursor.execute("DROP TABLE IF EXISTS document_chunks CASCADE;")
        # If embedding exists with wrong dimension, drop and recreate
        elif embedding_exists:
            cursor.execute("""
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'document_chunks'::regclass
            AND attname = 'embedding';
            """)
            row = cursor.fetchone()
            if row and row[0] != 768:
                print(f"Dropping document_chunks because dimension {row[0]} != 768")
                cursor.execute("DROP TABLE IF EXISTS document_chunks CASCADE;")
    except Exception as e:
        print(f"Error checking document_chunks schema: {e}")
        cursor.execute("DROP TABLE IF EXISTS document_chunks CASCADE;")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_chunks (
        id SERIAL PRIMARY KEY,
        document_id INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        chunk_index INTEGER,
        embedding vector(768),
        FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
    )
    """)

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

    # If knowledge_base already exists with an embedding column of the wrong
    # dimension (e.g. a leftover from an older embedding model), fix just that
    # column rather than the document_chunks approach of dropping the whole
    # table — knowledge_base holds real, hand-verified Q&A content and manual
    # entries that aren't regenerable the way document chunks are. Existing
    # rows lose their embedding (so won't surface via similarity search until
    # re-saved) but keep their question/answer/usage data intact.
    try:
        cursor.execute("""
        SELECT atttypmod
        FROM pg_attribute
        WHERE attrelid = 'knowledge_base'::regclass
        AND attname = 'embedding'
        AND NOT attisdropped
        """)
        row = cursor.fetchone()
        # row is None when the column doesn't exist yet — the ADD COLUMN
        # IF NOT EXISTS loop below already creates it with the right
        # dimension in that case, so there's nothing to fix here.
        if row and row[0] != 768:
            print(f"Fixing knowledge_base.embedding: dimension {row[0]} != 768")
            cursor.execute("ALTER TABLE knowledge_base DROP COLUMN embedding")
            cursor.execute("ALTER TABLE knowledge_base ADD COLUMN embedding vector(768)")
    except Exception as e:
        # Most likely the table doesn't exist yet at all — the CREATE TABLE
        # below handles that case with the right dimension from the start.
        print(f"knowledge_base.embedding dimension check skipped: {e}")
        conn.rollback()

    # Knowledge base — with semantic matching, source attribution, usage tracking
    cursor.execute("""
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
        embedding vector(768),
        FOREIGN KEY (chat_history_id) REFERENCES chat_history (id)
    )
    """)

    # Add new columns to existing knowledge_base table if they don't exist
    for col, definition in [
        ("corrected_by", "TEXT"),
        ("usage_count", "INTEGER DEFAULT 0"),
        ("embedding", "vector(768)"),
        ("source_type", "TEXT DEFAULT 'document'"),
    ]:
        try:
            cursor.execute(f"""
                ALTER TABLE knowledge_base ADD COLUMN IF NOT EXISTS {col} {definition}
            """)
        except Exception as e:
            print(f"Column {col} may already exist: {e}")
            conn.rollback()

    # Admin audit log — who changed what, for the admin dashboard's audit feed
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

    # Document insights — LLM-extracted facts, keywords, entities and a document
    # type classification, one row per document, feeding the admin dashboard's
    # Insights tab. UNIQUE on document_id so re-analysis is an upsert.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_insights (
        id SERIAL PRIMARY KEY,
        document_id INTEGER NOT NULL UNIQUE REFERENCES documents(id) ON DELETE CASCADE,
        doc_type TEXT,
        keywords JSONB DEFAULT '[]',
        facts JSONB DEFAULT '[]',
        entities JSONB DEFAULT '[]',
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()

