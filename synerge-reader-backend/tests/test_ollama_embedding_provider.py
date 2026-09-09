import importlib.util
import math
import os
import sys
import types
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ollama_embedding_provider as provider_module
from ollama_embedding_provider import (
    EmbeddingResponseError,
    EmbeddingTransportError,
    InvalidEmbeddingInputError,
    OllamaEmbeddingProvider,
    _build_embed_request_payload,
)
from rag_model_profiles import EmbeddingProfile, default_embedding_profile


def _load_isolated_module(module_file: str, real_module_name: str):
    """Execute a fresh copy of a module's source under a private probe name.

    See the identical helper in test_rag_model_profiles.py for the full
    rationale: this proves import-time behavior (e.g. "never calls
    load_dotenv") without ever reloading (and thereby replacing the class
    objects of) the real, already-imported ``real_module_name`` module that
    this file's own top-level imports still reference.

    Because the target module imports EmbeddingProfile, a dataclass, the
    probe module is registered under its unique probe name in
    ``sys.modules`` before ``exec_module`` runs, then removed (or restored)
    in a ``finally`` block.
    """
    probe_name = f"_isolated_probe__{real_module_name}__{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(probe_name, module_file)
    module = importlib.util.module_from_spec(spec)

    previous = sys.modules.get(probe_name)
    sys.modules[probe_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(probe_name, None)
        else:
            sys.modules[probe_name] = previous
    return module


PROFILE = EmbeddingProfile(
    provider="ollama",
    model="test-embedding-model:latest",
    dimension=4,
    query_prefix="query: ",
    document_prefix="doc: ",
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


def make_http_post(response=None, raise_exc=None):
    calls = []

    def http_post(endpoint, payload):
        calls.append({"endpoint": endpoint, "payload": payload})
        if raise_exc is not None:
            raise raise_exc
        return response

    http_post.calls = calls
    return http_post


def vector(value=0.1, dim=4):
    return [value] * dim


def unreachable_http_post(endpoint, payload):
    raise AssertionError("http_post must not be called for this test")


# --- no I/O at import or construction time ---


def test_import_never_calls_load_dotenv(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("ollama_embedding_provider must never call load_dotenv")

    trap = types.ModuleType("dotenv")
    trap.load_dotenv = _boom
    monkeypatch.setitem(sys.modules, "dotenv", trap)

    _load_isolated_module(provider_module.__file__, "ollama_embedding_provider")


def test_construction_performs_no_network_call():
    # If __post_init__ or the dataclass machinery ever called http_post,
    # this would raise.
    OllamaEmbeddingProvider(profile=default_embedding_profile(), http_post=unreachable_http_post)


def test_construction_rejects_non_ollama_profile():
    profile = EmbeddingProfile(
        provider="not-ollama", model="m", dimension=4, query_prefix="", document_prefix=""
    )
    with pytest.raises(ValueError):
        OllamaEmbeddingProvider(profile=profile, http_post=unreachable_http_post)


def test_construction_rejects_non_callable_http_post():
    with pytest.raises(TypeError):
        OllamaEmbeddingProvider(profile=default_embedding_profile(), http_post="not callable")


# --- request payload shape ---


def test_request_payload_uses_input_truncate_false_and_model():
    response = FakeResponse(json_data={"embeddings": [vector()]})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post, keep_alive="45m")

    ollama.embed_documents(["hello"])

    assert len(http_post.calls) == 1
    call = http_post.calls[0]
    assert call["endpoint"] == "/api/embed"
    payload = call["payload"]
    assert payload["model"] == PROFILE.model
    assert payload["truncate"] is False
    assert payload["keep_alive"] == "45m"
    assert payload["input"] == ["doc: hello"]


def test_build_embed_request_payload_helper_takes_already_prepared_text():
    # _build_embed_request_payload does no prefixing of its own -- whatever
    # strings it is given are exactly what ends up in "input".
    payload = _build_embed_request_payload(PROFILE, ["a", "b"], keep_alive="30m")
    assert payload == {
        "model": PROFILE.model,
        "input": ["a", "b"],
        "truncate": False,
        "keep_alive": "30m",
    }


# --- prefix application: exactly once, per profile field ---


def test_embed_documents_applies_document_prefix_once():
    response = FakeResponse(json_data={"embeddings": [vector()]})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    ollama.embed_documents(["passage text"])

    assert http_post.calls[0]["payload"]["input"] == ["doc: passage text"]


def test_embed_query_applies_query_prefix_once():
    response = FakeResponse(json_data={"embeddings": [vector()]})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    ollama.embed_query("what is this")

    assert http_post.calls[0]["payload"]["input"] == ["query: what is this"]


def test_embed_query_does_not_detect_an_existing_prefix_and_double_prepends():
    # Deliberate non-idempotence: embed_query() always prepends the
    # configured prefix exactly once, with no detection of whether the raw
    # text already starts with it. This is intentional -- prefix-detection/
    # idempotence is explicitly not implemented in C1.
    response = FakeResponse(json_data={"embeddings": [vector()]})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    already_prefixed = "query: already has the prefix"
    ollama.embed_query(already_prefixed)

    assert http_post.calls[0]["payload"]["input"] == ["query: " + already_prefixed]
    assert http_post.calls[0]["payload"]["input"] == ["query: query: already has the prefix"]


def test_empty_document_prefix_leaves_text_unchanged():
    profile = EmbeddingProfile(
        provider="ollama", model="m", dimension=4, query_prefix="q: ", document_prefix=""
    )
    response = FakeResponse(json_data={"embeddings": [vector()]})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=profile, http_post=http_post)

    ollama.embed_documents(["plain text"])

    assert http_post.calls[0]["payload"]["input"] == ["plain text"]


def test_original_text_list_is_not_mutated():
    response = FakeResponse(json_data={"embeddings": [vector(), vector()]})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    original = ["first", "second"]
    snapshot = list(original)
    ollama.embed_documents(original)

    assert original == snapshot


def test_error_message_for_whitespace_item_shows_original_text_not_prefixed():
    http_post = make_http_post()
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(InvalidEmbeddingInputError) as excinfo:
        ollama.embed_documents(["   "])

    assert "doc: " not in str(excinfo.value)
    assert http_post.calls == []


# --- bare str/bytes and empty-text rejection ---


def test_embed_documents_rejects_bare_string():
    http_post = make_http_post()
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(InvalidEmbeddingInputError):
        ollama.embed_documents("a raw string")
    assert http_post.calls == []


def test_embed_documents_rejects_bare_bytes():
    http_post = make_http_post()
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(InvalidEmbeddingInputError):
        ollama.embed_documents(b"raw bytes")
    assert http_post.calls == []


def test_embed_documents_empty_list_returns_empty_and_skips_http_post():
    http_post = make_http_post()
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    assert ollama.embed_documents([]) == []
    assert http_post.calls == []


def test_embed_documents_rejects_empty_string_item_with_index():
    http_post = make_http_post()
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(InvalidEmbeddingInputError) as excinfo:
        ollama.embed_documents(["ok", ""])
    assert excinfo.value.index == 1
    assert http_post.calls == []


def test_embed_documents_rejects_whitespace_only_item_with_index():
    http_post = make_http_post()
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(InvalidEmbeddingInputError) as excinfo:
        ollama.embed_documents(["  \t\n  "])
    assert excinfo.value.index == 0


def test_embed_documents_rejects_non_string_item_with_index():
    http_post = make_http_post()
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(InvalidEmbeddingInputError) as excinfo:
        ollama.embed_documents(["ok", 123])
    assert excinfo.value.index == 1


def test_embed_query_rejects_non_string():
    http_post = make_http_post()
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(InvalidEmbeddingInputError):
        ollama.embed_query(["not", "a", "string"])
    assert http_post.calls == []


def test_embed_query_rejects_empty_string():
    http_post = make_http_post()
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(InvalidEmbeddingInputError):
        ollama.embed_query("   ")


# --- transport failures: __cause__ preserved, batch range reported ---


def test_transport_exception_wraps_as_embedding_transport_error_with_cause():
    original = ConnectionError("simulated network failure")
    http_post = make_http_post(raise_exc=original)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(EmbeddingTransportError) as excinfo:
        ollama.embed_documents(["a", "b", "c"])

    assert excinfo.value.__cause__ is original
    assert excinfo.value.batch_range == (0, 2)


def test_http_error_status_wraps_as_embedding_transport_error():
    response = FakeResponse(ok=False)
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(EmbeddingTransportError) as excinfo:
        ollama.embed_documents(["hello"])
    assert excinfo.value.__cause__ is not None
    assert excinfo.value.batch_range == (0, 0)


# --- malformed / missing JSON response ---


def test_non_json_response_raises_embedding_response_error_with_cause():
    class BoomJSONError(ValueError):
        pass

    boom = BoomJSONError("invalid json")
    response = FakeResponse(json_exc=boom)
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(EmbeddingResponseError) as excinfo:
        ollama.embed_documents(["hello"])
    assert excinfo.value.__cause__ is boom


def test_response_missing_embeddings_key_raises_response_error():
    response = FakeResponse(json_data={"not_embeddings": []})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(EmbeddingResponseError):
        ollama.embed_documents(["hello"])


def test_response_non_dict_json_raises_response_error():
    response = FakeResponse(json_data=["not", "a", "dict"])
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(EmbeddingResponseError):
        ollama.embed_documents(["hello"])


# --- exact vector count ---


def test_vector_count_mismatch_reports_batch_range_not_fabricated_index():
    response = FakeResponse(json_data={"embeddings": [vector()]})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(EmbeddingResponseError) as excinfo:
        ollama.embed_documents(["hello", "world"])

    assert excinfo.value.index is None
    assert excinfo.value.batch_range == (0, 1)


def test_vectors_not_a_list_raises_response_error():
    response = FakeResponse(json_data={"embeddings": "not-a-list"})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(EmbeddingResponseError):
        ollama.embed_documents(["hello"])


# --- per-vector validation: dimension, bool/nonnumeric/NaN/Inf, all-zero ---
# (exact-zero rejection and offending-item-index behavior are unchanged from
# the original C1 implementation.)


def test_wrong_dimension_vector_raises_with_index():
    wrong = [0.1, 0.2, 0.3]  # profile.dimension is 4
    response = FakeResponse(json_data={"embeddings": [wrong]})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(EmbeddingResponseError) as excinfo:
        ollama.embed_documents(["hello"])
    assert excinfo.value.index == 0


def test_second_item_wrong_dimension_reports_index_one():
    good = vector()
    bad = [0.1, 0.2]
    response = FakeResponse(json_data={"embeddings": [good, bad]})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(EmbeddingResponseError) as excinfo:
        ollama.embed_documents(["hello", "world"])
    assert excinfo.value.index == 1


def test_vector_element_bool_is_rejected():
    bad_vector = [True, 0.1, 0.2, 0.3]
    response = FakeResponse(json_data={"embeddings": [bad_vector]})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(EmbeddingResponseError) as excinfo:
        ollama.embed_documents(["hello"])
    assert excinfo.value.index == 0


def test_vector_element_non_numeric_is_rejected():
    bad_vector = [0.1, "not-a-number", 0.2, 0.3]
    response = FakeResponse(json_data={"embeddings": [bad_vector]})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(EmbeddingResponseError):
        ollama.embed_documents(["hello"])


def test_vector_element_nan_is_rejected():
    bad_vector = [math.nan, 0.1, 0.2, 0.3]
    response = FakeResponse(json_data={"embeddings": [bad_vector]})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(EmbeddingResponseError):
        ollama.embed_documents(["hello"])


def test_vector_element_inf_is_rejected():
    bad_vector = [math.inf, 0.1, 0.2, 0.3]
    response = FakeResponse(json_data={"embeddings": [bad_vector]})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(EmbeddingResponseError):
        ollama.embed_documents(["hello"])


def test_all_zero_vector_is_rejected():
    zero_vector = [0.0, 0.0, 0.0, 0.0]
    response = FakeResponse(json_data={"embeddings": [zero_vector]})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(EmbeddingResponseError) as excinfo:
        ollama.embed_documents(["hello"])
    assert excinfo.value.index == 0


def test_single_all_zero_vector_fails_the_whole_batch():
    response = FakeResponse(json_data={"embeddings": [vector(), [0.0, 0.0, 0.0, 0.0]]})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    with pytest.raises(EmbeddingResponseError) as excinfo:
        ollama.embed_documents(["hello", "world"])
    assert excinfo.value.index == 1


def test_near_zero_but_not_exact_zero_vector_is_accepted():
    tiny_vector = [1e-12, 0.0, 0.0, 0.0]
    response = FakeResponse(json_data={"embeddings": [tiny_vector]})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    result = ollama.embed_documents(["hello"])
    assert result == [tiny_vector]


# --- successful round trip ---


def test_embed_documents_returns_vectors_in_order():
    v1, v2 = vector(0.1), vector(0.2)
    response = FakeResponse(json_data={"embeddings": [v1, v2]})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    result = ollama.embed_documents(["a", "b"])
    assert result == [v1, v2]


def test_embed_query_returns_single_vector():
    v = vector(0.3)
    response = FakeResponse(json_data={"embeddings": [v]})
    http_post = make_http_post(response=response)
    ollama = OllamaEmbeddingProvider(profile=PROFILE, http_post=http_post)

    result = ollama.embed_query("what is this")
    assert result == v
