import sqlite3
import os
import psycopg2
import sys
from dotenv import load_dotenv
DB_PATH = os.path.join(os.path.dirname(__file__), "synerge_reader.db")



def connect_to_postgres():
    
    load_dotenv()
    try:
        connection_string = os.getenv("DB_CONNECTION_STRING")
        print('Connecting to the PostgreSQL database...')
        connection = psycopg2.connect(connection_string)
        return connection

    except psycopg2.DatabaseError as error:
        print(f"Database error: {error}")
        return None


def test_postgres_connection():
    load_dotenv()
    try:
        connection = psycopg2.connect(os.getenv("DB_CONNECTION_STRING"))
        cursor = connection.cursor()
        cursor.execute("SELECT version()")
        print(cursor.fetchone())
    finally:
        connection.close()




def init_db():
    conn = connect_to_postgres()
    cursor = conn.cursor()

   # Users (must be first)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE,
        password TEXT,
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
        embedding_json TEXT,
        FOREIGN KEY (document_id) REFERENCES documents (id)
    )
    """)

    # Chat history
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
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

    # Insert anonymous user
    cursor.execute(
        "INSERT INTO users (id, username) VALUES (0, 'anonymous') ON CONFLICT (id) DO NOTHING"
    )

    cursor.execute(
        "ALTER TABLE users ADD COLUMN email TEXT UNIQUE"
    )
    
    conn.commit()
    conn.close()

