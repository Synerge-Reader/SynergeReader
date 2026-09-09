"""AST/source-shape structural contract tests for main.py's I3 embedding wiring.

These are source-shape/AST/text contract tests, not runtime integration
tests. main.py and docker-compose.yml are parsed as plain text (and, for
main.py, with the stdlib ``ast`` module); neither is ever imported, executed,
or run through Docker/Compose anywhere in this file. There is no network,
database, Ollama, container, or production access here.

main.py is never imported because it unconditionally calls ``init_db(...)``
at module scope -- importing it would attempt real database initialization
during test collection.

What these tests prove: the legacy ``embed_chunks``/``chunk_text`` helpers
and their fabricated zero-vector/``"/api/embeddings"`` fallbacks are gone
(while the unrelated ``chunk_text`` *column* name is untouched); the eight
call sites named in the I3 spec route through the exact
``_EMBEDDING_PROVIDER.embed_query``/``embed_documents`` receiver for their
semantic direction (query vs. document); the composition boundary assigns
``_EMBEDDING_PROFILE`` exactly once from ``resolve_embedding_profile(os.environ)``
and ``_EMBEDDING_PROVIDER`` exactly once from an appropriately-wired
``OllamaEmbeddingProvider(...)`` call, whose adapter forwards
``(endpoint, payload)`` to ``post_ollama`` with a pinned ``timeout=30``;
``init_db(...)`` is called with the exact keyword binding
``expected_dimension=_EMBEDDING_PROFILE.dimension``; ``EmbeddingProviderError``
propagates past (rather than being silently swallowed by) the specific
broad-exception/fallback handlers named in the I3 spec; ``_save_kb_pairs``
aborts its whole uncommitted batch atomically (rollback, no partial inserts,
no post-failure row, connection always closed) rather than continuing with a
null vector, and both of its callers (``generate_kb_from_document``,
``import_knowledge_from_url``) delegate to it rather than embedding directly;
the upload route embeds and validates the embedding count before it ever
opens the ingestion write connection (while its separate, earlier
``lookup_conn`` auth-token lookup is permitted and unrelated to that
ordering rule); page-aware chunking, locator persistence, and the
page-aware retrieval helpers are wired into ``upload_documents`` and
``get_relevant_chunks`` rather than reimplemented by hand; and the six
embedding-profile environment keys are present, bare, unduped, and
explained by a comment in ``docker-compose.yml``.

What these tests do NOT prove: they do not prove runtime wiring (main.py is
never executed by this file), they do not validate Ollama's successful
``/api/embed`` response contract, and they do not validate the mxbai/Nomic
models, dimensions, prefix behavior, schema compatibility, database
persistence, legacy Compose ``.env`` propagation, or any other production
behavior. This file carries no OpenRouter-presence check (that is
``test_main_local_generation_policy.py``'s responsibility) and does not
assert that the frontend displays citations -- locator metadata is proven
to be persisted and reconstructed internally, nothing about client-visible
rendering.
"""

import ast
from pathlib import Path

import pytest
import yaml

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_MAIN_PY_PATH = _REPOSITORY_ROOT / "synerge-reader-backend" / "main.py"
_COMPOSE_PATH = _REPOSITORY_ROOT / "docker-compose.yml"

_EMBEDDING_PROVIDER_NAME = "_EMBEDDING_PROVIDER"

_QUERY_SIDE_FUNCTIONS = ["get_relevant_chunks", "get_relevant_knowledge_base"]
_DOCUMENT_SIDE_FUNCTIONS = [
    "auto_save_to_kb",
    "_save_kb_pairs",
    "upload_documents",
    "submit_correction",
    "add_knowledge",
    "update_knowledge",
]

_COMPOSE_EMBEDDING_KEYS = [
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_QUERY_PREFIX",
    "EMBEDDING_DOCUMENT_PREFIX",
    "EMBEDDING_PROFILE_UNVERIFIED_ACK",
]


@pytest.fixture(scope="module")
def main_source():
    return _MAIN_PY_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def main_tree(main_source):
    return ast.parse(main_source, filename=str(_MAIN_PY_PATH))


@pytest.fixture(scope="module")
def compose_source():
    return _COMPOSE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def compose_doc(compose_source):
    return yaml.safe_load(compose_source)


# --- AST helpers (the primary verification mechanism throughout this file) ---


def _iter_functions_by_name(node, name):
    for candidate in ast.walk(node):
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)) and candidate.name == name:
            yield candidate


def _find_function(node, name):
    for fn in _iter_functions_by_name(node, name):
        return fn
    raise AssertionError(f"function {name!r} not found")


def _function_source_segment(source, fn_node):
    """The exact source lines spanning one function definition, including its
    full nested body -- used for simple, robust substring checks that are
    easier to hand-verify than an equivalent deep AST predicate."""
    lines = source.splitlines()
    return "\n".join(lines[fn_node.lineno - 1 : fn_node.end_lineno])


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


def _assignments_to_name(node, target_name):
    """Assignments `target_name = value` anywhere within node (any nesting depth)."""
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == target_name
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


# --- 1/2: embed_chunks is completely gone; the chunk_text *function* is gone, ---
# --- but the chunk_text *column* name must remain untouched everywhere else. ---


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


def test_no_legacy_chunk_text_function_definition(main_tree):
    # The Python helper is gone -- but this must not be confused with the
    # `chunk_text` SQL column name, which is expected to remain (see the
    # companion test below).
    names = [
        n.name for n in ast.walk(main_tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert "chunk_text" not in names


def test_no_legacy_chunk_text_call(main_tree):
    for node in ast.walk(main_tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            assert func.id != "chunk_text"


def test_chunk_text_column_name_still_used_in_sql(main_tree):
    # A plain `"chunk_text" in main_source` substring check would also be
    # satisfied by the unrelated Python variable `chunk_texts`, so this
    # proves the real thing: the `chunk_text` *SQL column* is still present,
    # in order, in the `document_chunks` INSERT statement's column list.
    matches = [
        node
        for node in ast.walk(main_tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "INSERT INTO document_chunks" in node.value
    ]
    assert len(matches) == 1
    normalized = " ".join(matches[0].value.split())
    assert (
        "INSERT INTO document_chunks "
        "(document_id, chunk_text, chunk_index, embedding, page_start, page_end, locator_json)"
        in normalized
    )


def test_no_legacy_api_embeddings_string(main_tree):
    for node in ast.walk(main_tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "/api/embeddings" not in node.value


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


# --- 3/4/5: profile/provider composition boundary ---


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


def test_imports_come_from_the_expected_modules(main_tree):
    imported_from = {}
    for node in ast.walk(main_tree):
        if isinstance(node, ast.ImportFrom) and node.module in (
            "ollama_embedding_provider",
            "rag_model_profiles",
            "document_chunker",
            "document_retrieval",
        ):
            for alias in node.names:
                imported_from[alias.name] = node.module
    assert imported_from.get("resolve_embedding_profile") == "rag_model_profiles"
    assert imported_from.get("OllamaEmbeddingProvider") == "ollama_embedding_provider"
    assert imported_from.get("EmbeddingProviderError") == "ollama_embedding_provider"
    assert imported_from.get("chunk_document") == "document_chunker"
    assert imported_from.get("build_chunk_locator") == "document_chunker"
    assert imported_from.get("build_relevant_chunks_query") == "document_retrieval"
    assert imported_from.get("retrieved_chunk_from_row") == "document_retrieval"


def test_json_imported_from_psycopg2_extras(main_tree):
    for node in ast.walk(main_tree):
        if isinstance(node, ast.ImportFrom) and node.module == "psycopg2.extras":
            if any(alias.name == "Json" for alias in node.names):
                return
    raise AssertionError("expected `from psycopg2.extras import Json`")


def test_embedding_profile_assigned_once_from_resolve_call_with_os_environ(main_tree):
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
    assert not value.keywords, (
        "resolve_embedding_profile() must be called with a positional argument, not keywords"
    )
    assert len(value.args) == 1, (
        "resolve_embedding_profile() must be called with exactly one argument: os.environ"
    )
    arg = value.args[0]
    assert isinstance(arg, ast.Attribute) and arg.attr == "environ", (
        "resolve_embedding_profile() must be called with os.environ, not a bare call "
        "or a hand-built mapping"
    )
    assert isinstance(arg.value, ast.Name) and arg.value.id == "os"


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


def test_init_db_called_with_expected_dimension_from_embedding_profile(main_tree):
    module_level_calls = [
        node.value
        for node in main_tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "init_db"
    ]
    assert len(module_level_calls) == 1, (
        f"expected exactly one module-level init_db(...) call, found {len(module_level_calls)}"
    )
    call = module_level_calls[0]
    assert not call.args, "init_db must be called with no positional arguments"

    kwargs = {kw.arg: kw.value for kw in call.keywords}
    assert set(kwargs) == {"expected_dimension"}, (
        f"init_db(...) must be called with exactly expected_dimension=...; got keywords {sorted(kwargs)}"
    )
    value = kwargs["expected_dimension"]
    assert isinstance(value, ast.Attribute) and value.attr == "dimension", (
        "expected_dimension must come from an attribute access ending in `.dimension`, "
        "not a literal, so a future edit can't silently reintroduce the 768 default"
    )
    assert isinstance(value.value, ast.Name) and value.value.id == "_EMBEDDING_PROFILE"


# --- 6/7: the eight call sites use the exact _EMBEDDING_PROVIDER receiver, exclusively ---


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


# --- 8: generate_kb_from_document and import_knowledge_from_url both delegate to ---
# --- _save_kb_pairs, rather than embedding directly ---


def test_generate_kb_from_document_delegates_to_save_kb_pairs(main_tree):
    fn = _find_function(main_tree, "generate_kb_from_document")
    calls = _calls_to_name(fn, "_save_kb_pairs")
    assert len(calls) == 1, "generate_kb_from_document must delegate to _save_kb_pairs exactly once"


def test_generate_kb_from_document_has_no_direct_embedding_provider_call(main_tree):
    fn = _find_function(main_tree, "generate_kb_from_document")
    assert not _calls_to_embedding_provider_method(fn, "embed_documents"), (
        "generate_kb_from_document must not gain its own duplicate embedding call; "
        "embedding is _save_kb_pairs's responsibility"
    )
    assert not _calls_to_embedding_provider_method(fn, "embed_query")


def test_import_knowledge_from_url_delegates_to_save_kb_pairs(main_tree):
    fn = _find_function(main_tree, "import_knowledge_from_url")
    calls = _calls_to_name(fn, "_save_kb_pairs")
    assert len(calls) == 1, "import_knowledge_from_url must delegate to _save_kb_pairs exactly once"


# --- 9/10/11: EmbeddingProviderError is never accidentally swallowed by a ---
# --- pre-existing broad except/fallback ---


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


@pytest.mark.parametrize("name", ["auto_save_to_kb", "_save_kb_pairs"])
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


@pytest.mark.parametrize("name", ["submit_correction", "add_knowledge", "update_knowledge"])
def test_sync_endpoint_embedding_failure_raises_sanitized_service_unavailable(main_tree, name):
    fn = _find_function(main_tree, name)
    handler = None
    for try_node in ast.walk(fn):
        if not isinstance(try_node, ast.Try):
            continue
        type_names = [_handler_type_name(h) for h in try_node.handlers]
        if "EmbeddingProviderError" in type_names:
            handler = try_node.handlers[type_names.index("EmbeddingProviderError")]
            break
    assert handler is not None, f"expected an except EmbeddingProviderError clause in {name}"

    raises = [n for n in ast.walk(handler) if isinstance(n, ast.Raise)]
    http_raises = [
        r
        for r in raises
        if isinstance(r.exc, ast.Call)
        and isinstance(r.exc.func, ast.Name)
        and r.exc.func.id == "HTTPException"
    ]
    assert http_raises, f"{name}'s EmbeddingProviderError handler must raise HTTPException(...)"
    for r in http_raises:
        call = r.exc
        assert len(call.args) == 2, "HTTPException(status, detail) must have exactly two positional args"
        status, detail = call.args
        assert isinstance(status, ast.Constant) and status.value == 503, (
            f"{name} must report 503 (service unavailable) for an embedding failure"
        )
        assert isinstance(detail, ast.Constant) and isinstance(detail.value, str), (
            "the HTTPException detail must be a plain string literal, not an f-string or "
            "anything else that could interpolate the raw exception"
        )

    if name == "submit_correction":
        # submit_correction's outer `except HTTPException: raise` (unlike
        # add_knowledge/update_knowledge's) does not close conn on this path,
        # so the inner EmbeddingProviderError handler itself must close it --
        # proven here structurally, in rollback -> close -> raise line order,
        # rather than by a substring/count check that a reorder could still
        # satisfy.
        rollback_calls = _attr_calls_on_name(handler, "conn", "rollback")
        close_calls = _attr_calls_on_name(handler, "conn", "close")
        assert rollback_calls, "submit_correction's EmbeddingProviderError handler must call conn.rollback()"
        assert close_calls, (
            "submit_correction's EmbeddingProviderError handler must call conn.close() -- "
            "the outer `except HTTPException: raise` in submit_correction does not close it"
        )
        rollback_line = min(c.lineno for c in rollback_calls)
        close_line = min(c.lineno for c in close_calls)
        raise_line = min(r.lineno for r in http_raises)
        assert rollback_line < close_line < raise_line, (
            "submit_correction must rollback, then close, then raise, in that order, "
            "on embedding failure -- got rollback@%d close@%d raise@%d"
            % (rollback_line, close_line, raise_line)
        )


def test_upload_embedding_failure_appends_sanitized_error(main_tree):
    fn = _find_function(main_tree, "upload_documents")
    handler = None
    for try_node in ast.walk(fn):
        if not isinstance(try_node, ast.Try):
            continue
        type_names = [_handler_type_name(h) for h in try_node.handlers]
        if "EmbeddingProviderError" in type_names:
            handler = try_node.handlers[type_names.index("EmbeddingProviderError")]
            break
    assert handler is not None, "expected an except EmbeddingProviderError clause in upload_documents"

    append_calls = _attr_calls_on_name(handler, "results", "append")
    assert append_calls, "the handler must append a per-file error entry to results"
    for call in append_calls:
        assert len(call.args) == 1 and isinstance(call.args[0], ast.Dict)
        error_dict = call.args[0]
        for key_node, value_node in zip(error_dict.keys, error_dict.values):
            if isinstance(key_node, ast.Constant) and key_node.value == "error":
                assert isinstance(value_node, ast.Constant) and isinstance(value_node.value, str), (
                    "the per-file 'error' value must be a plain string literal, not an "
                    "f-string referencing the raw exception"
                )

    assert any(isinstance(n, ast.Continue) for n in ast.walk(handler)), (
        "upload_documents must continue to the next file after an embedding failure, "
        "not abort the whole multi-file request"
    )


# --- 12/13: _save_kb_pairs fails closed: rollback, abort, no NULL vector, always closes ---


def test_save_kb_pairs_aborts_batch_atomically_on_embedding_failure(main_tree):
    """Structural only: this proves the AST shape of the abort path (rollback,
    return, no None-vector assignment, no insert after failure, and a
    try/finally that always closes the connection). It does not run
    _save_kb_pairs, and it does not prove runtime behavior against a real
    database or embedding provider.
    """
    fn = _find_function(main_tree, "_save_kb_pairs")

    embedding_provider_handler = None
    for try_node in ast.walk(fn):
        if not isinstance(try_node, ast.Try):
            continue
        type_names = [_handler_type_name(h) for h in try_node.handlers]
        if "EmbeddingProviderError" in type_names:
            embedding_provider_handler = try_node.handlers[type_names.index("EmbeddingProviderError")]
            break
    assert embedding_provider_handler is not None, (
        "expected an except EmbeddingProviderError clause in _save_kb_pairs"
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
        "expected a try/finally in _save_kb_pairs whose finally body calls conn.close()"
    )


# --- 14/15: upload embeds and validates the count before opening the ingestion ---
# --- write connection; the earlier auth lookup connection is a separate, permitted case ---


def test_upload_has_permitted_auth_lookup_connection(main_tree):
    fn = _find_function(main_tree, "upload_documents")
    lookup_assignments = _assignments_to_name(fn, "lookup_conn")
    assert lookup_assignments, "upload_documents must retain its lookup_conn auth-lookup connection"
    value = lookup_assignments[0].value
    assert (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "connect_to_postgres"
    ), "lookup_conn must be opened via connect_to_postgres()"


def test_upload_embeds_then_validates_count_then_opens_ingestion_write_connection(main_tree, main_source):
    fn = _find_function(main_tree, "upload_documents")
    segment = _function_source_segment(main_source, fn)

    embed_pos = segment.find("_EMBEDDING_PROVIDER.embed_documents(")
    count_check_pos = segment.find("len(embeddings) != len(chunks)")
    # Leading space is deliberate: it anchors on the bare `conn` identifier
    # and excludes the earlier, unrelated `lookup_conn = connect_to_postgres()`
    # auth lookup, whose own assignment would otherwise also match a plain
    # "conn = connect_to_postgres()" substring search (since "lookup_conn"
    # ends in "conn").
    write_conn_pos = segment.find(" conn = connect_to_postgres()")

    assert embed_pos != -1, "expected the embed_documents call in upload_documents"
    assert count_check_pos != -1, "expected an explicit len(embeddings) != len(chunks) guard"
    assert write_conn_pos != -1, "expected the ingestion write connection assignment"
    assert embed_pos < count_check_pos < write_conn_pos, (
        "upload_documents must embed, then validate the count, then open the ingestion "
        "write connection -- strictly in that order -- so neither an embedding failure "
        "nor a count mismatch ever touches the database"
    )


def test_upload_ingestion_write_connection_is_distinct_from_lookup_connection(main_tree):
    fn = _find_function(main_tree, "upload_documents")
    embed_calls = _calls_to_embedding_provider_method(fn, "embed_documents")
    assert len(embed_calls) == 1
    embed_lineno = embed_calls[0].lineno

    write_conn_assignments = [
        a
        for a in _assignments_to_name(fn, "conn")
        if isinstance(a.value, ast.Call)
        and isinstance(a.value.func, ast.Name)
        and a.value.func.id == "connect_to_postgres"
    ]
    assert write_conn_assignments, "upload_documents must open an ingestion write connection assigned to `conn`"
    assert all(a.lineno > embed_lineno for a in write_conn_assignments), (
        "the ingestion write connection (`conn`) must be opened only after the embedding "
        "call -- unlike the separate, earlier `lookup_conn` auth lookup, which is permitted "
        "to precede it"
    )


# --- 16/17/18: page-aware chunking, locator persistence, and preserved ownership ---


def test_upload_uses_chunk_document_and_page_aware_chunk_fields(main_tree, main_source):
    fn = _find_function(main_tree, "upload_documents")
    segment = _function_source_segment(main_source, fn)
    assert "chunk_document(result)" in segment, "upload_documents must call chunk_document(result)"
    for attr in ("chunk.text", "chunk.chunk_index", "chunk.page_start", "chunk.page_end"):
        assert attr in segment, f"expected {attr} to be used when inserting a chunk row"


def test_upload_builds_and_wraps_locator_json(main_tree, main_source):
    fn = _find_function(main_tree, "upload_documents")
    segment = _function_source_segment(main_source, fn)
    assert "build_chunk_locator(chunk, result.document_type)" in segment, (
        "upload_documents must build each chunk's locator via build_chunk_locator(chunk, result.document_type)"
    )
    assert "Json(locator_json)" in segment, (
        "the locator dict must be wrapped in psycopg2.extras.Json before insertion"
    )


def test_upload_chunk_insert_names_required_columns(main_tree, main_source):
    fn = _find_function(main_tree, "upload_documents")
    segment = _function_source_segment(main_source, fn)
    assert "INSERT INTO document_chunks" in segment
    for column in (
        "document_id",
        "chunk_text",
        "chunk_index",
        "embedding",
        "page_start",
        "page_end",
        "locator_json",
    ):
        assert column in segment, f"expected the document_chunks insert to name column {column!r}"


def test_upload_preserves_uploader_id_in_document_insert(main_tree, main_source):
    fn = _find_function(main_tree, "upload_documents")
    segment = _function_source_segment(main_source, fn)
    assert "INSERT INTO documents" in segment
    assert "user_id" in segment, "the documents insert must still name the user_id column"
    assert "uploader_id" in segment, "the documents insert must still pass uploader_id as its value"


def test_get_relevant_chunks_delegates_to_page_aware_retrieval_helpers(main_tree, main_source):
    fn = _find_function(main_tree, "get_relevant_chunks")
    segment = _function_source_segment(main_source, fn)
    assert "build_relevant_chunks_query(" in segment
    assert "retrieved_chunk_from_row(" in segment
    # It must delegate query construction, not hand-roll a second SELECT.
    assert "SELECT" not in segment, (
        "get_relevant_chunks must not contain its own hand-written SQL SELECT -- "
        "query construction belongs to build_relevant_chunks_query()"
    )


# --- 19: docker-compose.yml embedding-profile forwarding surface ---


def test_compose_backend_environment_has_six_bare_embedding_keys_exactly_once(compose_doc):
    env_list = compose_doc["services"]["backend"]["environment"]
    assert isinstance(env_list, list)
    for key in _COMPOSE_EMBEDDING_KEYS:
        # A bare key parses from YAML as the literal string "KEY"; a
        # KEY=value or ${KEY} entry is a different string and must not count
        # as satisfying "bare".
        bare_matches = [entry for entry in env_list if entry == key]
        assert len(bare_matches) == 1, (
            f"expected exactly one bare '{key}' entry under services.backend.environment, "
            f"found {len(bare_matches)}"
        )
        value_style_matches = [
            entry
            for entry in env_list
            if isinstance(entry, str) and entry != key and entry.split("=")[0] == key
        ]
        assert not value_style_matches, (
            f"{key} must not also appear as a KEY=value entry: {value_style_matches}"
        )


def test_compose_explanatory_comment_present(compose_source):
    lowered = compose_source.lower()
    assert "keep these embedding-profile entries bare deliberately" in lowered
    assert "fail-closed explicit-override path" in lowered
    assert "docker-compose 1.29.2" in lowered


def test_compose_existing_backend_environment_entries_preserved(compose_doc):
    env_list = compose_doc["services"]["backend"]["environment"]
    for existing in (
        "PYTHONUNBUFFERED=1",
        "OLLAMA_BASE_URL=${OLLAMA_BASE_URL}",
        "OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE}",
        "GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID}",
        "DB_CONNECTION_STRING=${DB_CONNECTION_STRING}",
        "EMAIL_KEY=${EMAIL_KEY}",
        "ADMIN_EMAILS=${ADMIN_EMAILS}",
        "FRONTEND_URL=${FRONTEND_URL}",
    ):
        assert existing in env_list, f"expected pre-existing entry {existing!r} to be preserved"


def test_compose_contract_tests_do_not_execute_docker():
    this_path = Path(__file__).resolve()
    this_source = this_path.read_text(encoding="utf-8")
    this_tree = ast.parse(this_source, filename=str(this_path))

    forbidden_import_roots = {"subprocess", "docker"}
    imported_roots = set()
    for node in ast.walk(this_tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint(forbidden_import_roots)

    forbidden_calls = {"system", "run", "Popen", "call", "check_call", "check_output"}
    for node in ast.walk(this_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden_calls, (
                f"unexpected subprocess/os-execution-like call in this test file: {node.func.attr}"
            )
