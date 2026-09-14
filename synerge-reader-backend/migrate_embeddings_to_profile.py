"""Re-embed document_chunks and knowledge_base onto the currently-resolved
embedding profile (documented default: mxbai-embed-large:335m, 1024 dims).

Why this exists
----------------
main.py's startup validation (dbSetup.validate_document_chunks_embedding_schema
/ validate_knowledge_base_embedding_schema) deliberately refuses to
auto-migrate a mismatched embedding column dimension -- see dbSetup.py's own
comment on that code path: "Startup must never add, drop, or replace an
embedding column automatically ... run an explicit reviewed migration." This
script is that migration, meant to be read and run by a person, not invoked
automatically by the application.

It does NOT re-chunk documents or touch any document/KB text. It only:

  1. Reports the current embedding column dimension on both tables vs. the
     target profile's dimension, and how many rows would be affected.
  2. (only with --apply) drops and recreates the `embedding` column on both
     tables at the target dimension, then re-embeds every existing row's
     text through the resolved provider and writes the new vector back:
       - document_chunks.chunk_text -> embed_documents(...)  (matches how
         main.py's /upload embeds a chunk at write time)
       - knowledge_base.question    -> embed_documents(...)  (matches every
         knowledge_base write call site in main.py)

Usage
-----
    # Dry run (default) -- reports what would happen, touches nothing:
    python migrate_embeddings_to_profile.py

    # Actually run it (asks for interactive "yes" confirmation first):
    python migrate_embeddings_to_profile.py --apply

    # Actually run it without the interactive prompt:
    python migrate_embeddings_to_profile.py --apply --yes

Typical production usage, from the repo root, if the backend runs via
docker-compose with a service named `backend`:

    docker compose exec backend python migrate_embeddings_to_profile.py
    docker compose exec backend python migrate_embeddings_to_profile.py --apply

Safe to re-run: if interrupted partway, running it again just re-embeds
everything from scratch (wasted compute, not unsafe -- it never reads its
own previous output as input, and every write is a plain UPDATE by primary
key).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import requests
from dotenv import load_dotenv

from dbSetup import (
    connect_to_postgres,
    validate_document_chunks_embedding_schema,
    validate_knowledge_base_embedding_schema,
)
from ollama_embedding_provider import EmbeddingProviderError, OllamaEmbeddingProvider
from rag_model_profiles import resolve_embedding_profile

BATCH_SIZE = 20
EMBED_TIMEOUT_S = 30


def _resolve_ollama_base_url() -> str:
    """Not the full multi-host fallback dance main.py does at request time —
    this is a one-off operator-run script, so it just honors OLLAMA_BASE_URL
    if set, else falls back to OLLAMA_HOST:OLLAMA_PORT (default localhost).
    If this can't reach Ollama, set OLLAMA_BASE_URL explicitly for this run."""
    base = os.getenv("OLLAMA_BASE_URL", "").strip()
    if base:
        return base.rstrip("/")
    host = os.getenv("OLLAMA_HOST", "127.0.0.1")
    port = os.getenv("OLLAMA_PORT", "11434")
    return f"http://{host}:{port}"


def _get_current_dimension(cursor, table_name: str):
    cursor.execute("SELECT to_regclass(%s)::oid", (table_name,))
    row = cursor.fetchone()
    table_oid = row[0] if row else None
    if table_oid is None:
        return None
    cursor.execute(
        """
        SELECT atttypmod FROM pg_attribute
        WHERE attrelid = %s AND attname = 'embedding' AND NOT attisdropped
        """,
        (table_oid,),
    )
    row = cursor.fetchone()
    return row[0] if row and isinstance(row[0], int) and row[0] > 0 else None


def _reembed_table(cur, conn, provider, *, table: str, id_col: str, text_col: str, rows: list, label: str):
    done = 0
    failed = []
    t0 = time.time()
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        ids = [r[0] for r in batch]
        texts = [r[1] or "" for r in batch]
        try:
            vectors = provider.embed_documents(texts)
        except EmbeddingProviderError as exc:
            print(f"  batch {i}-{i + len(batch)}: embedding failed ({exc}); retrying items individually")
            for row_id, text in zip(ids, texts):
                try:
                    vec = provider.embed_documents([text])[0]
                    cur.execute(f"UPDATE {table} SET embedding = %s WHERE {id_col} = %s", (vec, row_id))
                    done += 1
                except EmbeddingProviderError as e2:
                    failed.append((row_id, str(e2)))
            conn.commit()
            continue
        for row_id, vec in zip(ids, vectors):
            cur.execute(f"UPDATE {table} SET embedding = %s WHERE {id_col} = %s", (vec, row_id))
        conn.commit()
        done += len(batch)
        print(f"  {done}/{len(rows)} {label} re-embedded ({time.time() - t0:.1f}s elapsed)")
    if failed:
        print(f"  {len(failed)} {label.rstrip('s')}(s) FAILED and are left with NULL embedding: {failed}")
    return done, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Actually write changes (default: dry run only)")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive confirmation prompt")
    args = parser.parse_args()

    load_dotenv()
    profile = resolve_embedding_profile(os.environ)
    print(f"Target profile: provider={profile.provider} model={profile.model} dimension={profile.dimension}")
    print(f"  document_prefix={profile.document_prefix!r}  query_prefix={profile.query_prefix!r}")

    base_url = _resolve_ollama_base_url()
    print(f"Ollama endpoint: {base_url}")

    conn = connect_to_postgres()
    if conn is None:
        print("ERROR: could not connect to Postgres. Check DB_CONNECTION_STRING.")
        sys.exit(1)
    cur = conn.cursor()

    chunk_dim = _get_current_dimension(cur, "document_chunks")
    kb_dim = _get_current_dimension(cur, "knowledge_base")
    print(f"Current document_chunks.embedding dimension: {chunk_dim}")
    print(f"Current knowledge_base.embedding dimension:  {kb_dim}")

    cur.execute("SELECT COUNT(*) FROM document_chunks")
    chunk_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM knowledge_base WHERE question IS NOT NULL AND btrim(question) <> ''")
    kb_count = cur.fetchone()[0]
    print(f"document_chunks rows that would be re-embedded: {chunk_count}")
    print(f"knowledge_base rows that would be re-embedded:  {kb_count}")

    if chunk_dim == profile.dimension and kb_dim == profile.dimension:
        print("\nBoth tables are already at the target dimension. Nothing to do.")
        conn.close()
        return

    if not args.apply:
        print("\nDry run only -- no changes made. Re-run with --apply to execute.")
        conn.close()
        return

    if not args.yes:
        answer = input(
            f"\nThis will DROP and rebuild the embedding column on document_chunks "
            f"({chunk_count} rows) and knowledge_base ({kb_count} rows), then "
            f"re-embed every row through {profile.model}. Existing vectors are "
            f"permanently overwritten. Type 'yes' to continue: "
        )
        if answer.strip().lower() != "yes":
            print("Aborted -- no changes made.")
            conn.close()
            return

    provider = OllamaEmbeddingProvider(
        profile=profile,
        http_post=lambda endpoint, payload: requests.post(
            f"{base_url}{endpoint}", json=payload, timeout=EMBED_TIMEOUT_S
        ),
        keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
    )

    print("\n=== document_chunks ===")
    cur.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS embedding")
    cur.execute(f"ALTER TABLE document_chunks ADD COLUMN embedding vector({profile.dimension})")
    conn.commit()
    print(f"Column rebuilt at vector({profile.dimension}).")
    cur.execute("SELECT id, chunk_text FROM document_chunks ORDER BY id")
    chunk_done, chunk_failed = _reembed_table(
        cur, conn, provider, table="document_chunks", id_col="id", text_col="chunk_text",
        rows=cur.fetchall(), label="chunks",
    )

    print("\n=== knowledge_base ===")
    cur.execute("ALTER TABLE knowledge_base DROP COLUMN IF EXISTS embedding")
    cur.execute(f"ALTER TABLE knowledge_base ADD COLUMN embedding vector({profile.dimension})")
    conn.commit()
    print(f"Column rebuilt at vector({profile.dimension}).")
    cur.execute(
        "SELECT id, question FROM knowledge_base WHERE question IS NOT NULL AND btrim(question) <> '' ORDER BY id"
    )
    kb_done, kb_failed = _reembed_table(
        cur, conn, provider, table="knowledge_base", id_col="id", text_col="question",
        rows=cur.fetchall(), label="KB rows",
    )

    print("\n=== Verifying against the same check main.py runs at startup ===")
    try:
        validate_document_chunks_embedding_schema(cur, expected_dimension=profile.dimension)
        validate_knowledge_base_embedding_schema(cur, expected_dimension=profile.dimension)
        print("PASSED -- the backend will start cleanly against this database now.")
    except Exception as exc:
        print(f"FAILED -- {exc}")
        conn.close()
        sys.exit(1)

    conn.close()
    print(f"\nDone. {chunk_done} chunk(s) and {kb_done} KB row(s) re-embedded.")
    if chunk_failed or kb_failed:
        print("Some rows failed and were left with a NULL embedding (see above) -- re-run this script to retry them.")


if __name__ == "__main__":
    main()
