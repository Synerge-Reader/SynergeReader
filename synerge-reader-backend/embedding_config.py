import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))
# Explicit path, not bare load_dotenv() — bare form depends on process cwd,
# which is not guaranteed to be synerge-reader-backend/ under pytest or other
# invocation contexts. This also fixes the load-order problem: main.py's
# load_dotenv() call happens after its `from dbSetup import ...` line, so any
# config module reachable through that import chain must load dotenv itself.

def read_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer; received {raw!r}.") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero; received {value}.")
    return value

EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large:335m").strip()
if not EMBED_MODEL:
    raise RuntimeError("EMBED_MODEL cannot be empty.")

EMBEDDING_VECTOR_DIMENSION = read_positive_int_env("EMBEDDING_VECTOR_DIMENSION", 1024)

# This prefix is part of the mxbai-embed-large model profile, not a universal
# constant — it is required for query-side embeddings per the model's
# documented usage, and NOT applied to document/passage-side embeddings.
# If EMBED_MODEL is ever changed, this prefix's correctness must be
# re-reviewed against the new model's own documentation before reuse.
EMBEDDING_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
