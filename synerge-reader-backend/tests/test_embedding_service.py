import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from embedding_config import EMBED_MODEL, EMBEDDING_QUERY_PREFIX, EMBEDDING_VECTOR_DIMENSION
from embedding_service import (
    EmbeddingDimensionError,
    EmbeddingResponseError,
    EmbeddingServiceError,
    embed_documents,
    embed_queries,
    embed_query,
    prepare_embedding_input,
)


class FakeResponse:
    def __init__(self, json_data=None, ok=True, json_exc=None):
        self._json_data = json_data
        self._ok = ok
        self._json_exc = json_exc

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("simulated HTTP error status")

    def json(self):
        if self._json_exc is not None:
            raise self._json_exc
        return self._json_data


def make_post_fn(response=None, raise_exc=None):
    calls = []

    def post_fn(endpoint, payload, timeout=None):
        calls.append({"endpoint": endpoint, "payload": payload, "timeout": timeout})
        if raise_exc is not None:
            raise raise_exc
        return response

    post_fn.calls = calls
    return post_fn


def _vector(value=0.1, dim=None):
    dim = dim if dim is not None else EMBEDDING_VECTOR_DIMENSION
    return [value] * dim


# --- prepare_embedding_input ---


def test_query_prefix_applied_to_query_input():
    result = prepare_embedding_input("what is this document about", "query")
    assert result == EMBEDDING_QUERY_PREFIX + "what is this document about"


def test_query_prefix_not_applied_to_document_input():
    result = prepare_embedding_input("this is passage text", "document")
    assert result == "this is passage text"


def test_query_prefix_not_double_applied():
    already_prefixed = EMBEDDING_QUERY_PREFIX + "some text"
    result = prepare_embedding_input(already_prefixed, "query")
    assert result == already_prefixed
    assert result.count(EMBEDDING_QUERY_PREFIX) == 1


# --- empty input short-circuits before any network call ---


def test_embed_documents_empty_list_returns_empty_and_skips_post_fn():
    post_fn = make_post_fn()
    result = embed_documents([], post_fn=post_fn, keep_alive="30m")
    assert result == []
    assert post_fn.calls == []


def test_embed_queries_empty_list_returns_empty_and_skips_post_fn():
    post_fn = make_post_fn()
    result = embed_queries([], post_fn=post_fn, keep_alive="30m")
    assert result == []
    assert post_fn.calls == []


# --- input type validation ---


def test_embed_documents_raw_string_raises_type_error():
    post_fn = make_post_fn()
    with pytest.raises(TypeError):
        embed_documents("a raw string", post_fn=post_fn, keep_alive="30m")
    assert post_fn.calls == []


def test_non_string_item_in_list_raises_type_error():
    post_fn = make_post_fn()
    with pytest.raises(TypeError):
        embed_documents(["ok", 123], post_fn=post_fn, keep_alive="30m")
    assert post_fn.calls == []


# --- transport failure never degrades to a zero vector ---


def test_post_fn_exception_raises_embedding_service_error():
    post_fn = make_post_fn(raise_exc=ConnectionError("simulated network failure"))
    with pytest.raises(EmbeddingServiceError):
        embed_documents(["hello"], post_fn=post_fn, keep_alive="30m")


def test_post_fn_http_error_status_raises_embedding_service_error():
    post_fn = make_post_fn(response=FakeResponse(ok=False))
    with pytest.raises(EmbeddingServiceError):
        embed_documents(["hello"], post_fn=post_fn, keep_alive="30m")


# --- request body shape ---


def test_request_body_uses_input_field_truncate_false_and_model():
    response = FakeResponse(json_data={"embeddings": [_vector()]})
    post_fn = make_post_fn(response=response)
    embed_documents(["hello"], post_fn=post_fn, keep_alive="30m")

    assert len(post_fn.calls) == 1
    payload = post_fn.calls[0]["payload"]
    assert payload["input"] == ["hello"]
    assert "prompt" not in payload
    assert payload["truncate"] is False
    assert payload["model"] == EMBED_MODEL


def test_request_body_includes_keep_alive_passthrough():
    response = FakeResponse(json_data={"embeddings": [_vector()]})
    post_fn = make_post_fn(response=response)
    embed_documents(["hello"], post_fn=post_fn, keep_alive="45m")

    payload = post_fn.calls[0]["payload"]
    assert payload["keep_alive"] == "45m"


# --- response validation ---


def test_response_missing_embeddings_key_raises_response_error():
    response = FakeResponse(json_data={"not_embeddings": []})
    post_fn = make_post_fn(response=response)
    with pytest.raises(EmbeddingResponseError):
        embed_documents(["hello"], post_fn=post_fn, keep_alive="30m")


def test_response_vector_count_mismatch_raises_response_error():
    response = FakeResponse(json_data={"embeddings": [_vector()]})
    post_fn = make_post_fn(response=response)
    with pytest.raises(EmbeddingResponseError):
        embed_documents(["hello", "world"], post_fn=post_fn, keep_alive="30m")


def test_response_non_numeric_vector_element_raises_response_error():
    bad_vector = [0.1] * (EMBEDDING_VECTOR_DIMENSION - 1) + ["not-a-number"]
    response = FakeResponse(json_data={"embeddings": [bad_vector]})
    post_fn = make_post_fn(response=response)
    with pytest.raises(EmbeddingResponseError):
        embed_documents(["hello"], post_fn=post_fn, keep_alive="30m")


def test_response_non_json_body_raises_response_error_not_raw_exception():
    class BoomJSONError(ValueError):
        pass

    response = FakeResponse(json_exc=BoomJSONError("invalid json"))
    post_fn = make_post_fn(response=response)
    with pytest.raises(EmbeddingResponseError):
        embed_documents(["hello"], post_fn=post_fn, keep_alive="30m")


def test_response_wrong_dimension_raises_dimension_error():
    wrong_length_vector = [0.1] * (EMBEDDING_VECTOR_DIMENSION - 1)
    response = FakeResponse(json_data={"embeddings": [wrong_length_vector]})
    post_fn = make_post_fn(response=response)
    with pytest.raises(EmbeddingDimensionError):
        embed_documents(["hello"], post_fn=post_fn, keep_alive="30m")


def test_embed_query_returns_single_vector_with_prefix_applied():
    response = FakeResponse(json_data={"embeddings": [_vector()]})
    post_fn = make_post_fn(response=response)
    result = embed_query("what is this", post_fn=post_fn, keep_alive="30m")
    assert result == _vector()
    assert post_fn.calls[0]["payload"]["input"] == [EMBEDDING_QUERY_PREFIX + "what is this"]
