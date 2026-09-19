"""Re-embed document_chunks and knowledge_base onto the currently-resolved
embedding profile (documented default: mxbai-embed-large:335m, 1024 dims).

Why this exists
----------------
main.py's startup validation (dbSetup.validate_document_chunks_embedding_schema
/ validate_knowledge_base_embedding_schema) deliberately refuses to auto-migrate
a mismatched embedding column dimension -- see dbSetup.py's own comment: "Startup
must never add, drop, or replace an embedding column automatically ... run an
explicit reviewed migration." This script is that migration. It is meant to be
read and run by a person, never invoked by the application.

How it stays safe
------------------
The live `embedding` columns are NOT touched until every replacement vector
exists and has been checked:

  1. Preflight (no writes): reports each table's current dimension, row counts,
     NULL counts and any indexes on the column, and proves Ollama can produce a
     vector of the target dimension with the resolved profile.
  2. Fill: adds a shadow column `embedding_new vector(<target>)` and embeds every
     row into it, committing per batch. Resumable -- a re-run only embeds rows
     whose shadow value is still NULL. The live column is untouched throughout,
     so an interruption or failure at this stage changes nothing the app uses.
  3. Gate: any row that could not be embedded blocks the cutover (exit 1) unless
     you pass --allow-failed-rows. Vectors are checked before they are written
     (right dimension, finite, not all zero).
  4. Cutover: one transaction, both tables, under an exclusive lock: re-checks
     that no embeddable row is missing a new vector (catches rows written while
     the script ran), renames `embedding` -> `embedding_legacy_<olddim>` and
     `embedding_new` -> `embedding`, and commits. All or nothing.
  5. Verify: runs the same schema check main.py runs at startup and exits
     non-zero if it fails.

The previous vectors are kept as `embedding_legacy_<olddim>` (rollback:
rename the two columns back). Pass --drop-legacy to remove them once you are
satisfied. Indexes on the old column, if any exist, follow that column to its
new name and are NOT recreated for the new one -- the preflight lists them so
you can recreate them deliberately (this repository creates none).

It does not re-chunk documents or touch document/KB text. Rows are re-embedded
the way main.py writes them: document_chunks.chunk_text and
knowledge_base.question, both through embed_documents().

Usage
-----
    # Dry run (default): report only, changes nothing.
    python migrate_embeddings_to_profile.py

    # Do it (asks you to type MIGRATE; --yes skips the prompt):
    python migrate_embeddings_to_profile.py --apply

Take a backup first, e.g.  pg_dump -Fc -d <database> -f pre-embedding-migration.dump

On the production host, run it in a one-off container -- NOT `exec`. After a
deploy of a build that expects 1024 dimensions the backend container exits at
startup and restarts in a loop, and `exec` cannot attach to a restarting
container. The repository is bind-mounted, so the script is already there:

    docker-compose run --rm backend python migrate_embeddings_to_profile.py
    docker-compose run --rm backend python migrate_embeddings_to_profile.py --apply

Stop or avoid writes while it runs (a crash-looping backend does none).
Exit status: 0 = migrated (or nothing to do / dry run); 1 = blocked or failed,
with the live columns unchanged unless the message says the cutover ran.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Iterable, List, Optional, Sequence, Tuple

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
SHADOW_COLUMN = "embedding_new"
LIVE_COLUMN = "embedding"

# (table, primary-key column, text column that gets embedded)
TABLES: Tuple[Tuple[str, str, str], ...] = (
    ("document_chunks", "id", "chunk_text"),
    ("knowledge_base", "id", "question"),
)


class MigrationBlocked(RuntimeError):
    """The migration must stop; the message says what state the database is in."""


# --------------------------------------------------------------------------
# Small pure helpers (unit-tested)
# --------------------------------------------------------------------------


def vector_ok(vec: Sequence[float], dimension: int) -> bool:
    """A vector is only written if it has the target dimension, every component
    is finite, and it is not all zeros (an all-zero vector 'looks valid' but
    matches nothing -- the silent-failure mode this migration must not create)."""
    if vec is None or len(vec) != dimension:
        return False
    nonzero = False
    for x in vec:
        if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x):
            return False
        if x != 0:
            nonzero = True
    return nonzero


def needs_migration(current_dimension: Optional[int], target_dimension: int) -> bool:
    return current_dimension != target_dimension


def legacy_column_name(old_dimension: Optional[int]) -> str:
    return f"embedding_legacy_{old_dimension if old_dimension else 'unknown'}"


def cutover_allowed(failed_rows: int, missing_rows: int, allow_failed_rows: bool) -> bool:
    """Cutover needs every embeddable row to have a new vector, unless the
    operator explicitly accepted leaving the failed ones NULL. Missing rows that
    are not failures (e.g. written after the fill) always block."""
    if missing_rows == 0:
        return True
    return allow_failed_rows and missing_rows == failed_rows


def _embeddable(text_col: str) -> str:
    return f"{text_col} IS NOT NULL AND btrim({text_col}) <> ''"


# --------------------------------------------------------------------------
# Database inspection
# --------------------------------------------------------------------------


def column_dimension(cur, table: str, column: str) -> Optional[int]:
    cur.execute("SELECT to_regclass(%s)::oid", (table,))
    row = cur.fetchone()
    table_oid = row[0] if row else None
    if table_oid is None:
        return None
    cur.execute(
        "SELECT atttypmod FROM pg_attribute WHERE attrelid = %s AND attname = %s AND NOT attisdropped",
        (table_oid, column),
    )
    row = cur.fetchone()
    return row[0] if row and isinstance(row[0], int) and row[0] > 0 else None


def column_exists(cur, table: str, column: str) -> bool:
    cur.execute("SELECT to_regclass(%s)::oid", (table,))
    row = cur.fetchone()
    if not row or row[0] is None:
        return False
    cur.execute(
        "SELECT 1 FROM pg_attribute WHERE attrelid = %s AND attname = %s AND NOT attisdropped",
        (row[0], column),
    )
    return cur.fetchone() is not None


def embeddable_rows(cur, table: str, text_col: str) -> int:
    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {_embeddable(text_col)}")
    return cur.fetchone()[0]


def null_rows(cur, table: str, text_col: str, column: str) -> int:
    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {_embeddable(text_col)} AND {column} IS NULL")
    return cur.fetchone()[0]


def embedding_indexes(cur, table: str) -> List[Tuple[str, str]]:
    cur.execute(
        "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = %s AND indexdef ILIKE %s",
        (table, "%embedding%"),
    )
    return list(cur.fetchall())


# --------------------------------------------------------------------------
# Fill / cutover
# --------------------------------------------------------------------------


def fill_shadow(cur, conn, provider, *, table: str, id_col: str, text_col: str,
                dimension: int, batch_size: int = BATCH_SIZE, log=print) -> Tuple[int, List[Tuple[int, str]]]:
    """Embed every embeddable row whose shadow value is still NULL. Returns
    (rows_written, failures). Never touches the live column."""
    cur.execute(
        f"SELECT {id_col}, {text_col} FROM {table} "
        f"WHERE {_embeddable(text_col)} AND {SHADOW_COLUMN} IS NULL ORDER BY {id_col}"
    )
    rows = cur.fetchall()
    written = 0
    failures: List[Tuple[int, str]] = []
    started = time.time()

    def write(row_id, vec) -> bool:
        if not vector_ok(vec, dimension):
            failures.append((row_id, "provider returned an unusable vector (wrong size, non-finite or all zero)"))
            return False
        cur.execute(f"UPDATE {table} SET {SHADOW_COLUMN} = %s WHERE {id_col} = %s", (vec, row_id))
        return True

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]
        try:
            vectors = provider.embed_documents(texts)
            for row_id, vec in zip(ids, vectors):
                written += 1 if write(row_id, vec) else 0
        except EmbeddingProviderError as exc:
            log(f"  batch {i}-{i + len(batch)}: {exc}; retrying its rows one at a time")
            for row_id, text in zip(ids, texts):
                try:
                    written += 1 if write(row_id, provider.embed_documents([text])[0]) else 0
                except EmbeddingProviderError as e2:
                    failures.append((row_id, str(e2)))
        conn.commit()
        done = min(i + batch_size, len(rows))
        if done == len(rows) or (i // batch_size) % 5 == 4:
            log(f"  {done}/{len(rows)} rows processed ({time.time() - started:.1f}s)")
    return written, failures


def cutover(conn, cur, plans: Iterable[Tuple[str, str, Optional[int]]], *, allow_failed_rows: bool,
            failed_by_table: dict) -> None:
    """Swap shadow -> live for every table in ONE transaction. `plans` is
    (table, text_col, old_dimension). Raises MigrationBlocked (after rolling
    back, changing nothing) if any embeddable row lacks a new vector."""
    plans = list(plans)
    try:
        for table, text_col, _old in plans:
            cur.execute(f"LOCK TABLE {table} IN ACCESS EXCLUSIVE MODE")
        for table, text_col, _old in plans:
            missing = null_rows(cur, table, text_col, SHADOW_COLUMN)
            failed = failed_by_table.get(table, 0)
            if not cutover_allowed(failed, missing, allow_failed_rows):
                raise MigrationBlocked(
                    f"{table}: {missing} embeddable row(s) have no new vector "
                    f"({failed} failed to embed this run). Nothing was changed."
                )
        for table, text_col, old_dim in plans:
            if column_exists(cur, table, LIVE_COLUMN):
                legacy = legacy_column_name(old_dim)
                if column_exists(cur, table, legacy):
                    legacy = f"{legacy}_{int(time.time())}"
                cur.execute(f"ALTER TABLE {table} RENAME COLUMN {LIVE_COLUMN} TO {legacy}")
            cur.execute(f"ALTER TABLE {table} RENAME COLUMN {SHADOW_COLUMN} TO {LIVE_COLUMN}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _resolve_ollama_base_url() -> str:
    """Honors OLLAMA_BASE_URL, else OLLAMA_HOST:OLLAMA_PORT (default localhost).
    Not main.py's multi-host fallback: this is a one-off, operator-run script --
    set OLLAMA_BASE_URL explicitly if it can't reach Ollama."""
    base = os.getenv("OLLAMA_BASE_URL", "").strip()
    if base:
        return base.rstrip("/")
    return f"http://{os.getenv('OLLAMA_HOST', '127.0.0.1')}:{os.getenv('OLLAMA_PORT', '11434')}"


def _build_provider(profile, base_url: str) -> OllamaEmbeddingProvider:
    return OllamaEmbeddingProvider(
        profile=profile,
        http_post=lambda endpoint, payload: requests.post(f"{base_url}{endpoint}", json=payload, timeout=EMBED_TIMEOUT_S),
        keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Actually migrate (default: dry run, changes nothing)")
    parser.add_argument("--yes", action="store_true", help="Skip the typed confirmation prompt")
    parser.add_argument("--allow-failed-rows", action="store_true",
                        help="Proceed with the cutover even if some rows could not be embedded; they are left NULL")
    parser.add_argument("--drop-legacy", action="store_true",
                        help="After a verified cutover, drop the preserved old-dimension columns (removes the rollback)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)

    load_dotenv()
    profile = resolve_embedding_profile(os.environ)
    target = profile.dimension
    print(f"Target profile: provider={profile.provider} model={profile.model} dimension={target}")
    print(f"  document_prefix={profile.document_prefix!r}  query_prefix={profile.query_prefix!r}")
    base_url = _resolve_ollama_base_url()
    print(f"Ollama endpoint: {base_url}")

    conn = connect_to_postgres()
    if conn is None:
        print("ERROR: could not connect to Postgres. Check DB_CONNECTION_STRING.")
        return 1
    cur = conn.cursor()

    print("\n=== Preflight ===")
    todo = []
    for table, id_col, text_col in TABLES:
        dim = column_dimension(cur, table, LIVE_COLUMN)
        total = embeddable_rows(cur, table, text_col)
        nulls = null_rows(cur, table, text_col, LIVE_COLUMN) if column_exists(cur, table, LIVE_COLUMN) else total
        shadow_dim = column_dimension(cur, table, SHADOW_COLUMN)
        print(f"{table}: live embedding dimension={dim}, embeddable rows={total}, "
              f"already NULL={nulls}, leftover shadow column={'dim ' + str(shadow_dim) if shadow_dim else 'none'}")
        for name, definition in embedding_indexes(cur, table):
            print(f"  index on embedding: {name}: {definition}")
        if shadow_dim and shadow_dim != target:
            print(f"ERROR: {table}.{SHADOW_COLUMN} exists with dimension {shadow_dim}, expected {target}. "
                  "Inspect it and drop it by hand if it is a leftover; refusing to guess.")
            conn.close()
            return 1
        if not needs_migration(dim, target) and shadow_dim:
            print(f"ERROR: {table}.embedding is already dimension {dim} but a shadow column {SHADOW_COLUMN} also exists. "
                  "That state should not occur; refusing to swap it in over a live column. Inspect it and drop the "
                  "shadow column by hand if it is a leftover.")
            conn.close()
            return 1
        if needs_migration(dim, target):
            todo.append((table, id_col, text_col, dim))

    if not todo:
        print("\nBoth tables are already at the target dimension. Nothing to do.")
        conn.close()
        return 0

    if not args.apply:
        print("\nDry run only -- no changes made. Re-run with --apply to migrate.")
        print("Take a backup first:  pg_dump -Fc -d <database> -f pre-embedding-migration.dump")
        conn.close()
        return 0

    provider = _build_provider(profile, base_url)
    print("\nProving the embedding provider works before touching anything ...")
    try:
        probe = provider.embed_documents(["migration preflight probe"])[0]
    except EmbeddingProviderError as exc:
        print(f"ERROR: could not get a vector from Ollama: {exc}")
        conn.close()
        return 1
    if not vector_ok(probe, target):
        print(f"ERROR: the provider returned an unusable vector (expected {target} finite, non-zero components).")
        conn.close()
        return 1
    print(f"  OK ({len(probe)} dimensions).")

    if not args.yes:
        answer = input(
            "\nThis adds a shadow column, re-embeds every row, then swaps it in for the live "
            "embedding column (the old one is kept as embedding_legacy_*).\n"
            "Have you taken a backup? Type MIGRATE to continue: "
        )
        if answer.strip() != "MIGRATE":
            print("Aborted -- no changes made.")
            conn.close()
            return 1

    failed_by_table = {}
    plans = []
    for table, id_col, text_col, old_dim in todo:
        print(f"\n=== {table} ===")
        if column_dimension(cur, table, SHADOW_COLUMN) is None:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {SHADOW_COLUMN} vector({target})")
            conn.commit()
            print(f"Added shadow column {SHADOW_COLUMN} vector({target}). Live column untouched.")
        else:
            print(f"Resuming: shadow column {SHADOW_COLUMN} already exists; only rows still NULL are embedded.")
        written, failures = fill_shadow(cur, conn, provider, table=table, id_col=id_col, text_col=text_col,
                                        dimension=target, batch_size=args.batch_size)
        failed_by_table[table] = len(failures)
        print(f"{written} row(s) embedded, {len(failures)} failed.")
        for row_id, why in failures[:20]:
            print(f"  FAILED id={row_id}: {why}")
        plans.append((table, text_col, old_dim))

    print("\n=== Cutover ===")
    try:
        cutover(conn, cur, plans, allow_failed_rows=args.allow_failed_rows, failed_by_table=failed_by_table)
    except MigrationBlocked as exc:
        print(f"BLOCKED: {exc}")
        print("The live embedding columns are unchanged. The shadow columns keep their progress: fix the cause "
              "(or pass --allow-failed-rows to accept NULLs for the failed rows) and re-run.")
        conn.close()
        return 1
    print("Swapped: the new vectors are now the live `embedding` columns.")

    print("\n=== Verifying against the same check main.py runs at startup ===")
    try:
        validate_document_chunks_embedding_schema(cur, expected_dimension=target)
        validate_knowledge_base_embedding_schema(cur, expected_dimension=target)
    except Exception as exc:
        print(f"FAILED -- {exc}")
        conn.close()
        return 1
    print("PASSED -- the backend will pass its startup schema check against this database.")

    if args.drop_legacy:
        for table, _text_col, old_dim in plans:
            legacy = legacy_column_name(old_dim)
            if column_exists(cur, table, legacy):
                cur.execute(f"ALTER TABLE {table} DROP COLUMN {legacy}")
                print(f"Dropped {table}.{legacy}")
        conn.commit()
    else:
        print("Previous vectors kept as embedding_legacy_* (rollback = rename back). Re-run with "
              "--drop-legacy to remove them once you are satisfied.")

    conn.close()
    if any(failed_by_table.values()):
        print("\nDone WITH WARNINGS: some rows were left with a NULL embedding (they will not appear in search).")
    else:
        print("\nDone. Every embeddable row has a new vector.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
