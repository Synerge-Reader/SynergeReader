"""Strict Ollama embedding provider adapter.

Construction performs no I/O: the HTTP POST callable is injected by the
caller, so building an ``OllamaEmbeddingProvider`` never opens a socket,
loads a model, or reads the environment. All validation described below runs
in-process against whatever the injected callable returns.

Endpoint availability on the installed DGX Ollama instance is unverified —
that question is out of scope for this no-I/O module and blocks only actual
network use in a later unit, not the construction or validation logic here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Callable, List, Optional, Protocol, Sequence, Tuple

from rag_model_profiles import EmbeddingProfile

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class EmbeddingProviderError(RuntimeError):
    """Base class for all embedding provider failures."""


class InvalidEmbeddingInputError(EmbeddingProviderError):
    """A text input is the wrong type or shape (before any request is made)."""

    def __init__(self, message: str, *, index: Optional[int] = None) -> None:
        super().__init__(message)
        self.index = index


class EmbeddingTransportError(EmbeddingProviderError):
    """The HTTP call to Ollama failed (connection, timeout, HTTP status)."""

    def __init__(self, message: str, *, batch_range: Optional[Tuple[int, int]] = None) -> None:
        super().__init__(message)
        self.batch_range = batch_range


class EmbeddingResponseError(EmbeddingProviderError):
    """Ollama's response was received but is malformed or fails validation."""

    def __init__(
        self,
        message: str,
        *,
        index: Optional[int] = None,
        batch_range: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(message)
        self.index = index
        self.batch_range = batch_range


# --------------------------------------------------------------------------
# Provider abstractions
# --------------------------------------------------------------------------


class HttpResponseLike(Protocol):
    """The minimal response shape this module depends on (e.g. a
    ``requests.Response``): a status check and a JSON body parser."""

    def raise_for_status(self) -> None: ...

    def json(self) -> object: ...


# Mirrors main.py's existing post_ollama(endpoint, payload, ...) shape: the
# caller (composition boundary) owns base-URL resolution, headers, and
# timeouts, and hands this module a plain (endpoint, payload) -> response
# callable.
HttpPostCallable = Callable[[str, dict], HttpResponseLike]


class EmbeddingProvider(Protocol):
    """Provider-agnostic embedding interface."""

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


# --------------------------------------------------------------------------
# Ollama /api/embed request shape
# --------------------------------------------------------------------------


def _build_embed_request_payload(
    profile: EmbeddingProfile, prepared_texts: Sequence[str], *, keep_alive: str
) -> dict:
    """The modern Ollama ``/api/embed`` request body.

    ``prepared_texts`` must already be prepared (prefixed) — this function
    does not apply any prefix itself. Query/document prefixing is owned
    exclusively by ``OllamaEmbeddingProvider.embed_query()`` and
    ``embed_documents()``; this is purely a request-shape formatter.

    ``truncate=False`` makes Ollama fail loudly on oversized input rather
    than silently truncating it and returning a partial/misleading
    embedding.
    """
    return {
        "model": profile.model,
        "input": list(prepared_texts),
        "truncate": False,
        "keep_alive": keep_alive,
    }


def _apply_prefix(text: str, prefix: str) -> str:
    """Apply a profile prefix to text exactly once."""
    return f"{prefix}{text}" if prefix else text


def _validate_text_item(text: object, index: int) -> str:
    if not isinstance(text, str):
        raise InvalidEmbeddingInputError(
            f"item {index} must be a string; got {type(text).__name__}", index=index
        )
    if not text.strip():
        raise InvalidEmbeddingInputError(
            f"item {index} is empty or whitespace-only", index=index
        )
    return text


def _validate_vector(vector: object, *, index: int, expected_dimension: int) -> List[float]:
    if not isinstance(vector, list):
        raise EmbeddingResponseError(
            f"item {index}: embedding vector is not a list (got "
            f"{type(vector).__name__})",
            index=index,
        )

    values: List[float] = []
    for position, raw_value in enumerate(vector):
        # isinstance(x, bool) check first: bool is a Real subclass in Python,
        # so True/False would otherwise silently pass as 1.0/0.0.
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise EmbeddingResponseError(
                f"item {index}: embedding value at position {position} is "
                f"not a real number (got {type(raw_value).__name__})",
                index=index,
            )
        as_float = float(raw_value)
        if not math.isfinite(as_float):
            raise EmbeddingResponseError(
                f"item {index}: embedding value at position {position} is "
                "not finite (NaN/Inf)",
                index=index,
            )
        values.append(as_float)

    if len(values) != expected_dimension:
        raise EmbeddingResponseError(
            f"item {index}: expected {expected_dimension} dimensions, got "
            f"{len(values)}",
            index=index,
        )

    if all(value == 0.0 for value in values):
        raise EmbeddingResponseError(
            f"item {index}: embedding vector is an all-zero vector (exact "
            "zero L2 norm)",
            index=index,
        )

    return values


# --------------------------------------------------------------------------
# Ollama adapter
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OllamaEmbeddingProvider:
    """Ollama ``/api/embed`` adapter. Construction performs no I/O."""

    profile: EmbeddingProfile
    http_post: HttpPostCallable
    keep_alive: str = "30m"

    def __post_init__(self) -> None:
        if self.profile.provider != "ollama":
            raise ValueError(
                "OllamaEmbeddingProvider requires an embedding profile with "
                f"provider='ollama'; got provider={self.profile.provider!r}"
            )
        if not callable(self.http_post):
            raise TypeError("http_post must be a callable of (endpoint, payload)")

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a sequence of document/passage texts (for storage/indexing)."""
        if isinstance(texts, (str, bytes)):
            raise InvalidEmbeddingInputError(
                "embed_documents requires a sequence of texts, not a single "
                "str/bytes value"
            )
        texts_list = list(texts)
        if not texts_list:
            return []

        validated = [_validate_text_item(t, i) for i, t in enumerate(texts_list)]
        prepared = [_apply_prefix(t, self.profile.document_prefix) for t in validated]
        return self._request_embeddings(prepared)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single search query."""
        if not isinstance(text, str):
            raise InvalidEmbeddingInputError(
                f"embed_query requires a single string; got {type(text).__name__}"
            )
        validated = _validate_text_item(text, 0)
        prepared = _apply_prefix(validated, self.profile.query_prefix)
        return self._request_embeddings([prepared])[0]

    def _request_embeddings(self, prepared: List[str]) -> List[List[float]]:
        payload = _build_embed_request_payload(self.profile, prepared, keep_alive=self.keep_alive)
        batch_range = (0, len(prepared) - 1)

        try:
            response = self.http_post("/api/embed", payload)
            response.raise_for_status()
        except Exception as exc:
            raise EmbeddingTransportError(
                f"Ollama embedding request failed for batch range "
                f"{batch_range}: {exc}",
                batch_range=batch_range,
            ) from exc

        try:
            data = response.json()
        except Exception as exc:
            raise EmbeddingResponseError(
                "Ollama returned a response that is not valid JSON (batch "
                f"range {batch_range})",
                batch_range=batch_range,
            ) from exc

        if not isinstance(data, dict) or "embeddings" not in data:
            raise EmbeddingResponseError(
                "Ollama response is missing the 'embeddings' field (batch "
                f"range {batch_range})",
                batch_range=batch_range,
            )

        vectors = data["embeddings"]
        if not isinstance(vectors, list) or len(vectors) != len(prepared):
            received = len(vectors) if isinstance(vectors, list) else type(vectors).__name__
            raise EmbeddingResponseError(
                f"Expected {len(prepared)} embedding vector(s), received "
                f"{received} (batch range {batch_range}); Ollama did not "
                "identify which item(s) failed.",
                batch_range=batch_range,
            )

        return [
            _validate_vector(vector, index=i, expected_dimension=self.profile.dimension)
            for i, vector in enumerate(vectors)
        ]
