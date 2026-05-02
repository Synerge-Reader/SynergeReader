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

    # Document chunks
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_chunks (
        id SERIAL PRIMARY KEY,
        document_id INTEGER,
        chunk_text TEXT NOT NULL,
        chunk_index INTEGER,
        embedding vector(384),
        FOREIGN KEY (document_id) REFERENCES documents (id)
    )
    """)

    cursor.execute("""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'document_chunks'
      AND column_name IN ('embedding_json', 'embedding')
    """)
    columns = {row[0] for row in cursor.fetchall()}

    if "embedding_json" in columns:
        if "embedding" not in columns:
            cursor.execute("""
            ALTER TABLE document_chunks
            ADD COLUMN embedding vector(384)
            """)

        cursor.execute("""
        UPDATE document_chunks
        SET embedding = embedding_json::vector
        WHERE embedding_json IS NOT NULL
          AND embedding IS NULL
        """)

        cursor.execute("""
        ALTER TABLE document_chunks
        DROP COLUMN embedding_json
        """)
    elif "embedding" not in columns:
        cursor.execute("""
        ALTER TABLE document_chunks
        ADD COLUMN embedding vector(384)
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

    # Knowledge base
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS knowledge_base (
        id SERIAL PRIMARY KEY,
        question TEXT NOT NULL,
        original_answer TEXT,
        corrected_answer TEXT NOT NULL,
        created_at TEXT,
        chat_history_id INTEGER,
        context_text TEXT,
        FOREIGN KEY (chat_history_id) REFERENCES chat_history (id)
    )
    """)
    


  



    conn.commit()
    conn.close()

