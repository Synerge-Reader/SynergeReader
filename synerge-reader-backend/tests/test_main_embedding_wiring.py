"""AST-only structural contract tests for main.py's C2a embedding wiring.

These are source-shape/AST contract tests, not runtime integration tests.
main.py is parsed with the stdlib ``ast`` module and is never imported or
executed anywhere in this file -- there is no network, database, Ollama,
container, or production access here, and no ``sys.path`` manipulation is
needed since main.py is located as a plain file relative to ``__file__``.

What these tests prove: the legacy ``embed_chunks`` implementation and its
fabricated zero-vector/``"/api/embeddings"`` fallbacks are gone; the eight
call sites named in the C2a spec route through the exact
``_EMBEDDING_PROVIDER.embed_query``/``embed_documents`` receiver for their
semantic direction (query vs. document), never some other object's
same-named method; the composition boundary assigns ``_EMBEDDING_PROFILE``
exactly once from a bare ``resolve_embedding_profile()`` call (no manually
supplied profile id/version) and ``_EMBEDDING_PROVIDER`` exactly once from
an ``OllamaEmbeddingProvider(profile=_EMBEDDING_PROFILE,
http_post=_post_embedding_request, keep_alive=OLLAMA_KEEP_ALIVE)`` call,
whose adapter forwards ``(endpoint, payload)`` to ``post_ollama`` with a
pinned ``timeout=30``; ``EmbeddingProviderError`` propagates past (rather
than being silently swallowed by) the specific broad-exception/fallback
handlers named in the C2a spec; ``/ask``'s context-building path has a
safe, non-leaking ``EmbeddingProviderError`` handler; the two background
functions log an ``EmbeddingProviderError`` observably, and
``generate_kb_from_document`` aborts its whole uncommitted batch atomically
(rollback, no partial inserts, no post-failure row) rather than continuing
with a null vector; and the upload route's embedding call happens before it
ever opens a database connection.

What these tests do NOT prove: they do not prove runtime wiring (main.py is
never executed by this file), they do not validate Ollama's successful
``/api/embed`` response contract, and they do not validate the mxbai model,
1024 dimensions, prefix behavior, schema compatibility, database
persistence, or any other production behavior. C2a introduces no OpenRouter
changes; this file intentionally carries no OpenRouter-presence check, since
that would only assert the absence of a C2a diff rather than a durable
invariant, and C2b is expected to remove ``stream_openrouter_chat`` entirely.
"""

import ast
import os

import pytest

_MAIN_PY_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "main.py"))

_EMBEDDING_PROVIDER_NAME = "_EMBEDDING_PROVIDER"

_QUERY_SIDE_FUNCTIONS = ["get_relevant_chunks", "get_relevant_knowledge_base"]
_DOCUMENT_SIDE_FUNCTIONS = [
    "auto_save_to_kb",
    "generate_kb_from_document",
    "upload_documents",
    "submit_correction",
    "add_knowledge",
    "update_knowledge",
]


@pytest.fixture(scope="module")
def main_tree():
    with open(_MAIN_PY_PATH, "r", encoding="utf-8") as f:
        source = f.read()
    return ast.parse(source, filename=_MAIN_PY_PATH)


# --- AST helpers (the primary verification mechanism throughout this file) ---


def _iter_functions_by_name(node, name):
    for candidate in ast.walk(node):
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)) and candidate.name == name:
            yield candidate


def _find_function(node, name):
    for fn in _iter_functions_by_name(node, name):
        return fn
    raise AssertionError(f"function {name!r} not found")


def _calls_to_name(node, func_name):
    """All ast.Call nodes within `node` whose func is a bare name `func_name(...)`."""
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == func_name
    ]


def _calls_to_embedding_provider_method(node, method_name):
    """All ast.Call nodes within `node` that are exactly
    `_EMBEDDING_PROVIDER.<method_name>(...)` -- not `anything_else.<method_name>(...)`.

    This is deliberately stricter than matching on attribute name alone: a
    call must have an Attribute func whose value is the bare Name
    `_EMBEDDING_PROVIDER` for it to count.
    """
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == method_name
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == _EMBEDDING_PROVIDER_NAME
    ]


def _module_level_assignments(tree, target_name):
    """Top-level (module-body) `target_name = value` assignments only."""
    return [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == target_name
    ]


def _handler_type_name(handler):
    t = handler.type
    return t.id if isinstance(t, ast.Name) else None


def _is_bare_reraise(handler):
    return (
        len(handler.body) == 1
        and isinstance(handler.body[0], ast.Raise)
        and handler.body[0].exc is None
    )


def _attr_calls_on_name(node, receiver_name, attr_name):
    """All ast.Call nodes within `node` that are exactly `receiver_name.attr_name(...)`."""
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == attr_name
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == receiver_name
    ]


# --- 1/2: embed_chunks is completely gone ---


def test_no_embed_chunks_function_definition(main_tree):
    names = [
        n.name for n in ast.walk(main_tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert "embed_chunks" not in names


def test_no_embed_chunks_call(main_tree):
    for node in ast.walk(main_tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            assert func.id != "embed_chunks"
        elif isinstance(func, ast.Attribute):
            assert func.attr != "embed_chunks"


# --- 3: no legacy /api/embeddings string ---


def test_no_legacy_api_embeddings_string(main_tree):
    for node in ast.walk(main_tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "/api/embeddings" not in node.value


# --- 4: no list-multiplication zero-vector fallback pattern (e.g. [0.0] * 768) ---


def _is_single_element_zero_list(node):
    return (
        isinstance(node, ast.List)
        and len(node.elts) == 1
        and isinstance(node.elts[0], ast.Constant)
        and isinstance(node.elts[0].value, (int, float))
        and not isinstance(node.elts[0].value, bool)
        and node.elts[0].value == 0
    )


def test_no_list_multiplication_zero_vector_fallback_pattern(main_tree):
    for node in ast.walk(main_tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            assert not _is_single_element_zero_list(node.left), (
                "found a `[0.0] * N`-style fabricated zero-vector fallback"
            )
            assert not _is_single_element_zero_list(node.right), (
                "found a `N * [0.0]`-style fabricated zero-vector fallback"
            )


# --- 5/6/7: profile/provider composition boundary ---


def test_no_direct_embeddingprofile_construction(main_tree):
    assert not _calls_to_name(main_tree, "EmbeddingProfile"), (
        "main.py must never construct EmbeddingProfile directly"
    )
    for node in ast.walk(main_tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert alias.name != "EmbeddingProfile", (
                    "main.py must not even import EmbeddingProfile"
                )


def test_imports_come_from_the_c1_modules(main_tree):
    imported_from = {}
    for node in ast.walk(main_tree):
        if isinstance(node, ast.ImportFrom) and node.module in (
            "ollama_embedding_provider",
            "rag_model_profiles",
        ):
            for alias in node.names:
                imported_from[alias.name] = node.module
    assert imported_from.get("resolve_embedding_profile") == "rag_model_profiles"
    assert imported_from.get("OllamaEmbeddingProvider") == "ollama_embedding_provider"
    assert imported_from.get("EmbeddingProviderError") == "ollama_embedding_provider"


def test_embedding_profile_assigned_once_from_bare_resolver_call(main_tree):
    assignments = _module_level_assignments(main_tree, "_EMBEDDING_PROFILE")
    assert len(assignments) == 1, (
        f"expected exactly one module-level `_EMBEDDING_PROFILE = ...` assignment, "
        f"found {len(assignments)}"
    )
    value = assignments[0].value
    assert isinstance(value, ast.Call) and isinstance(value.func, ast.Name), (
        "_EMBEDDING_PROFILE must be assigned directly from a single function call"
    )
    assert value.func.id == "resolve_embedding_profile", (
        "_EMBEDDING_PROFILE must be assigned from resolve_embedding_profile()"
    )
    assert not value.args and not value.keywords, (
        "resolve_embedding_profile() must be called bare here -- no manually "
        "supplied profile id/version or any other argument"
    )


def test_embedding_provider_assigned_once_with_expected_kwargs(main_tree):
    assignments = _module_level_assignments(main_tree, "_EMBEDDING_PROVIDER")
    assert len(assignments) == 1, (
        f"expected exactly one module-level `_EMBEDDING_PROVIDER = ...` assignment, "
        f"found {len(assignments)}"
    )
    value = assignments[0].value
    assert isinstance(value, ast.Call) and isinstance(value.func, ast.Name), (
        "_EMBEDDING_PROVIDER must be assigned directly from a single constructor call"
    )
    assert value.func.id == "OllamaEmbeddingProvider", (
        "_EMBEDDING_PROVIDER must be constructed via OllamaEmbeddingProvider(...)"
    )

    kwargs = {kw.arg: kw.value for kw in value.keywords}
    assert set(kwargs) == {"profile", "http_post", "keep_alive"}, (
        f"unexpected keyword arguments to OllamaEmbeddingProvider(...): {sorted(kwargs)}"
    )
    assert isinstance(kwargs["profile"], ast.Name) and kwargs["profile"].id == "_EMBEDDING_PROFILE"
    assert (
        isinstance(kwargs["http_post"], ast.Name) and kwargs["http_post"].id == "_post_embedding_request"
    )
    assert isinstance(kwargs["keep_alive"], ast.Name) and kwargs["keep_alive"].id == "OLLAMA_KEEP_ALIVE"


def test_post_embedding_request_adapter_forwards_and_pins_timeout(main_tree):
    fn = _find_function(main_tree, "_post_embedding_request")

    param_names = [a.arg for a in fn.args.args]
    assert param_names[:2] == ["endpoint", "payload"], (
        "_post_embedding_request must accept (endpoint, payload, ...)"
    )

    calls = _calls_to_name(fn, "post_ollama")
    assert len(calls) == 1, (
        f"_post_embedding_request must contain exactly one call to post_ollama, found {len(calls)}"
    )
    call = calls[0]

    assert len(call.args) == 2, "post_ollama must be called with exactly the forwarded (endpoint, payload)"
    assert isinstance(call.args[0], ast.Name) and call.args[0].id == "endpoint"
    assert isinstance(call.args[1], ast.Name) and call.args[1].id == "payload"

    kwargs = {kw.arg: kw.value for kw in call.keywords}
    assert "timeout" in kwargs, "the post_ollama call must supply timeout=30"
    assert isinstance(kwargs["timeout"], ast.Constant) and kwargs["timeout"].value == 30, (
        "the embedding request timeout must be pinned to exactly 30 seconds"
    )


# --- 8/9/10: the eight call sites use the exact _EMBEDDING_PROVIDER receiver, exclusively ---


@pytest.mark.parametrize("name", _QUERY_SIDE_FUNCTIONS)
def test_query_side_function_uses_embed_query_exactly_once(main_tree, name):
    fn = _find_function(main_tree, name)
    query_calls = _calls_to_embedding_provider_method(fn, "embed_query")
    document_calls = _calls_to_embedding_provider_method(fn, "embed_documents")
    assert len(query_calls) == 1, (
        f"{name} must call _EMBEDDING_PROVIDER.embed_query(...) exactly once, found {len(query_calls)}"
    )
    assert len(document_calls) == 0, (
        f"{name} is query-side and must not call _EMBEDDING_PROVIDER.embed_documents(...)"
    )


@pytest.mark.parametrize("name", _DOCUMENT_SIDE_FUNCTIONS)
def test_document_side_function_uses_embed_documents_exactly_once(main_tree, name):
    fn = _find_function(main_tree, name)
    document_calls = _calls_to_embedding_provider_method(fn, "embed_documents")
    query_calls = _calls_to_embedding_provider_method(fn, "embed_query")
    assert len(document_calls) == 1, (
        f"{name} must call _EMBEDDING_PROVIDER.embed_documents(...) exactly once, "
        f"found {len(document_calls)}"
    )
    assert len(query_calls) == 0, (
        f"{name} is document/corpus-side and must not call _EMBEDDING_PROVIDER.embed_query(...)"
    )


# --- 11: get_relevant_chunks re-raises EmbeddingProviderError before the broad except ---


def test_get_relevant_chunks_reraises_before_broad_except(main_tree):
    fn = _find_function(main_tree, "get_relevant_chunks")
    found = False
    for try_node in ast.walk(fn):
        if not isinstance(try_node, ast.Try):
            continue
        type_names = [_handler_type_name(h) for h in try_node.handlers]
        if "EmbeddingProviderError" in type_names and "Exception" in type_names:
            found = True
            embed_idx = type_names.index("EmbeddingProviderError")
            exc_idx = type_names.index("Exception")
            assert embed_idx < exc_idx, (
                "except EmbeddingProviderError must appear before the broad except Exception"
            )
            assert _is_bare_reraise(try_node.handlers[embed_idx]), (
                "the EmbeddingProviderError handler must be a bare re-raise"
            )
    assert found, "expected a try/except in get_relevant_chunks with both handler types"


# --- 12: get_relevant_knowledge_base guards both its generic fallbacks ---
#
# Note (recorded, not repaired): this test requires exactly two guarded broad
# handlers. If a future change removes one of get_relevant_knowledge_base's
# two broad `except Exception` fallbacks, this test's `== 2` assertion will
# need to be revisited alongside that change.


def test_get_relevant_knowledge_base_guards_both_generic_fallbacks(main_tree):
    fn = _find_function(main_tree, "get_relevant_knowledge_base")
    guarded_broad_excepts = 0
    for try_node in ast.walk(fn):
        if not isinstance(try_node, ast.Try):
            continue
        type_names = [_handler_type_name(h) for h in try_node.handlers]
        if "Exception" not in type_names:
            continue
        assert "EmbeddingProviderError" in type_names, (
            "a broad `except Exception` in get_relevant_knowledge_base is missing a "
            "preceding `except EmbeddingProviderError: raise` guard"
        )
        embed_idx = type_names.index("EmbeddingProviderError")
        exc_idx = type_names.index("Exception")
        assert embed_idx < exc_idx
        assert _is_bare_reraise(try_node.handlers[embed_idx])
        guarded_broad_excepts += 1
    # The semantic-search-to-keyword fallback's except and the outer except.
    assert guarded_broad_excepts == 2, (
        "expected both the semantic->keyword fallback and the outer handler to be guarded, "
        f"found {guarded_broad_excepts}"
    )


# --- 13: /ask has a specific, safe context-building EmbeddingProviderError path ---


def test_ask_stream_generate_has_safe_embedding_provider_error_path(main_tree):
    ask_fn = _find_function(main_tree, "ask_question")
    stream_generate = _find_function(ask_fn, "stream_generate")

    found = False
    for try_node in ast.walk(stream_generate):
        if not isinstance(try_node, ast.Try):
            continue
        type_names = [_handler_type_name(h) for h in try_node.handlers]
        if "EmbeddingProviderError" not in type_names:
            continue
        found = True
        handler = try_node.handlers[type_names.index("EmbeddingProviderError")]
        # Safe: the exception object is never bound, so it cannot be leaked
        # into the yielded message even by accident.
        assert handler.name is None, (
            "the /ask EmbeddingProviderError handler must not bind the exception "
            "(as e) -- that would risk leaking raw internal text to the client"
        )
        yields = [n for n in ast.walk(handler) if isinstance(n, ast.Yield)]
        assert yields, "the EmbeddingProviderError handler must yield a client-visible message"
        for y in yields:
            assert isinstance(y.value, ast.Constant) and isinstance(y.value.value, str), (
                "the yielded embedding-unavailable message must be a plain string literal, "
                "not an f-string or any other expression that could interpolate raw internals"
            )
    assert found, "expected an except EmbeddingProviderError clause inside stream_generate"


# --- 14: background functions log EmbeddingProviderError observably ---
#
# Note (recorded, not repaired): this test requires a `print(...)` call in the
# handler. If logging is later replaced with a proper logger, this test will
# need rewriting alongside that change.


@pytest.mark.parametrize("name", ["auto_save_to_kb", "generate_kb_from_document"])
def test_background_function_logs_embedding_provider_error(main_tree, name):
    fn = _find_function(main_tree, name)
    logged = False
    for try_node in ast.walk(fn):
        if not isinstance(try_node, ast.Try):
            continue
        type_names = [_handler_type_name(h) for h in try_node.handlers]
        if "EmbeddingProviderError" not in type_names:
            continue
        handler = try_node.handlers[type_names.index("EmbeddingProviderError")]
        print_calls = [
            n
            for n in ast.walk(handler)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"
        ]
        if print_calls:
            logged = True
    assert logged, f"{name} must observably log an EmbeddingProviderError, not swallow it silently"


# --- background abort invariant: generate_kb_from_document aborts atomically ---


def test_generate_kb_from_document_aborts_batch_atomically_on_embedding_failure(main_tree):
    """Structural only: this proves the AST shape of the abort path (rollback,
    return, no None-vector assignment, no insert after failure, and a
    try/finally that always closes the connection). It does not run
    generate_kb_from_document, and it does not prove runtime behavior against
    a real database or embedding provider.
    """
    fn = _find_function(main_tree, "generate_kb_from_document")

    embedding_provider_handler = None
    for try_node in ast.walk(fn):
        if not isinstance(try_node, ast.Try):
            continue
        type_names = [_handler_type_name(h) for h in try_node.handlers]
        if "EmbeddingProviderError" in type_names:
            embedding_provider_handler = try_node.handlers[type_names.index("EmbeddingProviderError")]
            break
    assert embedding_provider_handler is not None, (
        "expected an except EmbeddingProviderError clause in generate_kb_from_document"
    )

    assert _attr_calls_on_name(embedding_provider_handler, "conn", "rollback"), (
        "the EmbeddingProviderError handler must call conn.rollback()"
    )

    assert any(isinstance(n, ast.Return) for n in ast.walk(embedding_provider_handler)), (
        "the EmbeddingProviderError handler must return immediately"
    )

    for n in ast.walk(embedding_provider_handler):
        if not isinstance(n, ast.Assign):
            continue
        assigns_q_vec = any(isinstance(t, ast.Name) and t.id == "q_vec" for t in n.targets)
        if assigns_q_vec:
            assert not (isinstance(n.value, ast.Constant) and n.value.value is None), (
                "the EmbeddingProviderError handler must not assign None to q_vec"
            )

    assert not _attr_calls_on_name(embedding_provider_handler, "c", "execute"), (
        "the EmbeddingProviderError handler must not perform a c.execute call"
    )

    finally_closes_connection = False
    for try_node in ast.walk(fn):
        if not isinstance(try_node, ast.Try) or not try_node.finalbody:
            continue
        for stmt in try_node.finalbody:
            if _attr_calls_on_name(stmt, "conn", "close"):
                finally_closes_connection = True
    assert finally_closes_connection, (
        "expected a try/finally in the connection-owned region of "
        "generate_kb_from_document whose finally body calls conn.close()"
    )


# --- 15: upload embeds before it ever opens a database connection ---


def test_upload_embedding_call_precedes_database_connection(main_tree):
    fn = _find_function(main_tree, "upload_documents")
    embed_calls = _calls_to_embedding_provider_method(fn, "embed_documents")
    connect_calls = _calls_to_name(fn, "connect_to_postgres")
    assert embed_calls, "upload_documents must call _EMBEDDING_PROVIDER.embed_documents(...)"
    assert connect_calls, "upload_documents must call connect_to_postgres"
    assert min(c.lineno for c in embed_calls) < min(c.lineno for c in connect_calls), (
        "the embedding call must occur before the database connection/insertion "
        "in upload_documents, so an embedding failure never touches the database"
    )
