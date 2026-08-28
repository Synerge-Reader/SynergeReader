import math
from numbers import Real
from typing import Callable, Sequence, Literal

from embedding_config import EMBED_MODEL, EMBEDDING_VECTOR_DIMENSION, EMBEDDING_QUERY_PREFIX


class EmbeddingServiceError(RuntimeError):
    """Transport-level failure: connection, HTTP error, timeout, missing model."""


class EmbeddingResponseError(EmbeddingServiceError):
    """Response was received but is malformed, incomplete, or not valid JSON."""


class EmbeddingDimensionError(EmbeddingServiceError):
    """Response was well-formed but returned the wrong vector dimension."""


def prepare_embedding_input(text: str, input_type: Literal["document", "query"]) -> str:
    """
    Apply the mxbai query prefix ONLY for query-side embeddings, and only once.
    Document/passage text is returned unmodified. This function's output is
    used ONLY for the embedding API call — callers must continue to use the
    original, unprefixed text everywhere else (LLM prompt, chat history,
    logging, UI display) — see main.py call-site notes below.
    """
    if input_type == "query" and not text.startswith(EMBEDDING_QUERY_PREFIX):
        return EMBEDDING_QUERY_PREFIX + text
    return text


def _validate_vector(vector: object) -> list[float]:
    if not isinstance(vector, list):
        raise EmbeddingResponseError("Ollama returned a non-list embedding vector.")
    out = []
    for value in vector:
        # isinstance(value, bool) check guards against bool being a Real subclass
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise EmbeddingResponseError("Ollama returned an invalid embedding value.")
        out.append(float(value))
    if len(out) != EMBEDDING_VECTOR_DIMENSION:
        raise EmbeddingDimensionError(
            f"Model {EMBED_MODEL} returned {len(out)} dimensions; "
            f"expected {EMBEDDING_VECTOR_DIMENSION}."
        )
    return out


def _embed_texts(
    texts: Sequence[str],
    *,
    input_type: Literal["document", "query"],
    post_fn: Callable,
    keep_alive: str,
    timeout: int = 30,
) -> list[list[float]]:
    if isinstance(texts, (str, bytes)):
        # str satisfies Sequence[str] — guard against embedding one string
        # character-by-character if a caller passes a bare string by mistake.
        raise TypeError("texts must be a sequence of complete strings, not one string.")
    if not texts:
        return []
    for t in texts:
        if not isinstance(t, str):
            raise TypeError(f"All items in texts must be strings; got {type(t)}.")

    prepared = [prepare_embedding_input(t, input_type) for t in texts]

    try:
        resp = post_fn(
            "/api/embed",
            {
                "model": EMBED_MODEL,
                "input": prepared,
                "truncate": False,  # fail loud on oversized input rather than
                                     # silently truncating and returning a
                                     # partial/misleading embedding
                "keep_alive": keep_alive,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
    except Exception as exc:
        raise EmbeddingServiceError(f"Ollama embedding request failed: {exc}") from exc

    try:
        data = resp.json()
    except Exception as exc:
        raise EmbeddingResponseError("Ollama returned an invalid JSON response.") from exc

    if not isinstance(data, dict) or "embeddings" not in data:
        raise EmbeddingResponseError("Ollama response missing 'embeddings' field.")

    vectors = data["embeddings"]
    if not isinstance(vectors, list) or len(vectors) != len(prepared):
        raise EmbeddingResponseError(
            f"Expected {len(prepared)} vectors, received "
            f"{len(vectors) if isinstance(vectors, list) else 'non-list'}."
        )

    return [_validate_vector(v) for v in vectors]


def embed_documents(
    texts: Sequence[str], *, post_fn: Callable, keep_alive: str, timeout: int = 30
) -> list[list[float]]:
    """Use for anything being stored/indexed: document chunks, KB entries."""
    return _embed_texts(texts, input_type="document", post_fn=post_fn, keep_alive=keep_alive, timeout=timeout)


def embed_queries(
    texts: Sequence[str], *, post_fn: Callable, keep_alive: str, timeout: int = 30
) -> list[list[float]]:
    """Use for anything searching: incoming questions against documents or KB."""
    return _embed_texts(texts, input_type="query", post_fn=post_fn, keep_alive=keep_alive, timeout=timeout)


def embed_query(text: str, *, post_fn: Callable, keep_alive: str, timeout: int = 30) -> list[float]:
    return embed_queries([text], post_fn=post_fn, keep_alive=keep_alive, timeout=timeout)[0]
