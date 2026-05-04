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
            if row and row[0] != 384:
                print(f"Dropping document_chunks because dimension {row[0]} != 384")
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
        embedding vector(384),
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

