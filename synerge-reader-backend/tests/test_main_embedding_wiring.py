"""AST/source-shape structural contract tests for main.py's I3 embedding wiring.

These are source-shape/AST/text contract tests, not runtime integration
tests. main.py and docker-compose.yml are parsed as plain text (and, for
main.py, with the stdlib ``ast`` module); neither is ever imported, executed,
or run through Docker/Compose anywhere in this file. There is no network,
database, Ollama, container, or production access here.

This file deliberately parses rather than imports ``main.py`` so it remains
a source-only contract suite. Runtime import and lifespan behavior are covered
separately by ``test_main_lifecycle.py``.

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
``init_db(...)`` is called exactly once through the lifespan-wired initializer
with the exact keyword binding ``expected_dimension=_EMBEDDING_PROFILE.dimension``;
``EmbeddingProviderError``
propagates past (rather than being silently swallowed by) the specific
broad-exception/fallback handlers named in the I3 spec; ``_save_kb_pairs``
aborts its whole uncommitted batch atomically (rollback, no partial inserts,
no post-failure row, connection always closed) rather than continuing with a
null vector, and both of its callers (``generate_kb_from_document``,
``import_knowledge_from_url``) delegate to it rather than embedding directly;
the ingestion service embeds and validates the embeddings before it ever
opens the write connection (while the upload route's separate auth-token
lookup is permitted and unrelated to that ordering rule); page-aware
chunking and locator persistence are wired into the ingestion service, and
the page-aware retrieval helpers into ``get_relevant_chunks``, rather than
reimplemented by hand; and the six
embedding-profile environment keys are present, bare, unduped, and
explained by a comment in ``docker-compose.yml``.

E1b UPDATE. POST /upload no longer parses, chunks, embeds, or writes anything
itself: it is a thin adapter over ``DocumentIngestionService``. The guarantees
that used to be asserted inside ``upload_documents`` -- embed-and-validate
before the write connection opens, page-aware chunk fields, locator
persistence, the two INSERT column lists, and preserved uploader ownership --
are therefore asserted here against ``document_ingestion.py``, which now owns
them, plus the route's composition of that service. Nothing was dropped; the
assertions moved to the module that holds the behavior. The route's separate,
permitted auth-token lookup also moved, into ``_resolve_uploader_id``.

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
_INGESTION_PY_PATH = (
    _REPOSITORY_ROOT / "synerge-reader-backend" / "document_ingestion.py"
)
_COMPOSE_PATH = _REPOSITORY_ROOT / "docker-compose.yml"

_EMBEDDING_PROVIDER_NAME = "_EMBEDDING_PROVIDER"

_QUERY_SIDE_FUNCTIONS = ["get_relevant_chunks", "get_relevant_knowledge_base"]
# upload_documents is deliberately absent: since E1b it embeds nothing itself
# and delegates to DocumentIngestionService, which receives _EMBEDDING_PROVIDER
# from the route's composition (see the E1b section at the end of this file).
_DOCUMENT_SIDE_FUNCTIONS = [
    "auto_save_to_kb",
    "_save_kb_pairs",
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
def ingestion_source():
    return _INGESTION_PY_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ingestion_tree(ingestion_source):
    return ast.parse(ingestion_source, filename=str(_INGESTION_PY_PATH))


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


def _find_class(node, name):
    """Locate one class definition by name.

    _find_function above walks only FunctionDef/AsyncFunctionDef, so it can
    never find DocumentIngestionService, which is an ast.ClassDef. This is the
    narrow counterpart, used to read the ingestion service's
    injected-dependency defaults.
    """
    for candidate in ast.walk(node):
        if isinstance(candidate, ast.ClassDef) and candidate.name == name:
            return candidate
    raise AssertionError(f"class {name!r} not found")


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


def test_chunk_text_column_name_still_used_in_sql(ingestion_tree):
    # A plain `"chunk_text" in source` substring check would also be satisfied
    # by the unrelated Python variable `chunk_texts`, so this proves the real
    # thing: the `chunk_text` *SQL column* is still present, in order, in the
    # `document_chunks` INSERT statement's column list. Since E1b that INSERT
    # lives in document_ingestion.py, not in the route.
    matches = [
        node
        for node in ast.walk(ingestion_tree)
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


def test_upload_route_holds_no_document_chunk_sql(main_tree):
    """The companion to the test above: the INSERT moved, it was not copied."""
    for node in ast.walk(main_tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "INSERT INTO document_chunks" not in node.value, (
                "main.py must not keep a second copy of the chunk INSERT -- "
                "persistence belongs to DocumentIngestionService"
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


def _imports_by_name(tree, modules):
    imported_from = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in modules:
            for alias in node.names:
                imported_from[alias.name] = node.module
    return imported_from


def test_imports_come_from_the_expected_modules(main_tree):
    imported_from = _imports_by_name(
        main_tree,
        (
            "ollama_embedding_provider",
            "rag_model_profiles",
            "document_ingestion",
            "document_retrieval",
        ),
    )
    assert imported_from.get("resolve_embedding_profile") == "rag_model_profiles"
    assert imported_from.get("OllamaEmbeddingProvider") == "ollama_embedding_provider"
    assert imported_from.get("EmbeddingProviderError") == "ollama_embedding_provider"
    assert imported_from.get("build_relevant_chunks_query") == "document_retrieval"
    assert imported_from.get("retrieved_chunk_from_row") == "document_retrieval"
    # E1b: the chunking/persistence imports moved with the behavior. main.py now
    # imports the ingestion contract instead of the chunker primitives.
    for name in ("DocumentIngestionService", "UploadDocument", "DocumentMetadata", "CommittedDocument"):
        assert imported_from.get(name) == "document_ingestion", (
            f"main.py must import {name} from document_ingestion"
        )


def test_main_no_longer_imports_the_ingestion_primitives(main_tree):
    """Delegation, not duplication: the route cannot reach the low-level parse,
    chunk, locator, or JSON-adaptation primitives at all any more."""
    leaked = _imports_by_name(main_tree, ("document_chunker", "psycopg2.extras"))
    assert leaked == {}, (
        f"main.py must no longer import ingestion primitives: {sorted(leaked)}"
    )
    parser_imports = _imports_by_name(main_tree, ("document_parser",))
    assert set(parser_imports) == {"sanitize_filename"}, (
        "main.py should keep only sanitize_filename from document_parser (used by "
        f"the DOCX conversion route); found {sorted(parser_imports)}"
    )


def test_ingestion_module_owns_the_chunker_and_json_imports(ingestion_tree):
    imported_from = _imports_by_name(
        ingestion_tree, ("document_chunker", "document_parser", "psycopg2.extras")
    )
    assert imported_from.get("chunk_document") == "document_chunker"
    assert imported_from.get("build_chunk_locator") == "document_chunker"
    assert imported_from.get("extract_text_from_upload") == "document_parser"
    assert imported_from.get("sanitize_filename") == "document_parser"
    assert imported_from.get("Json") == "psycopg2.extras"


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
    # Every module-body statement that is not a function or class definition
    # runs at import time, so each one is walked in full -- not just bare
    # `init_db(...)` expression statements, which would miss an assignment,
    # or a call nested inside a module-level if/try/with.
    for statement in main_tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        assert not _calls_to_name(statement, "init_db"), (
            "init_db(...) must not run at module import; found a call in the "
            f"module-level statement starting at line {statement.lineno}"
        )
        assert not _calls_to_name(statement, "_initialize_application"), (
            "_initialize_application() must not run at module import; found a "
            f"call in the module-level statement starting at line {statement.lineno}"
        )

    initializer = _find_function(main_tree, "_initialize_application")
    initializer_calls = _calls_to_name(initializer, "init_db")
    assert len(initializer_calls) == 1, (
        "_initialize_application() must call init_db(...) exactly once; "
        f"found {len(initializer_calls)} call(s)"
    )

    call = initializer_calls[0]
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

    fastapi_calls = _calls_to_name(main_tree, "FastAPI")
    assert len(fastapi_calls) == 1, (
        f"expected exactly one FastAPI(...) application construction, found {len(fastapi_calls)}"
    )
    app_kwargs = {kw.arg: kw.value for kw in fastapi_calls[0].keywords}
    assert "lifespan" in app_kwargs, "FastAPI(...) must receive lifespan=..."
    lifespan_value = app_kwargs["lifespan"]
    assert isinstance(lifespan_value, ast.Name), "lifespan= must reference a named function"

    lifespan_fn = _find_function(main_tree, lifespan_value.id)
    assert isinstance(lifespan_fn, ast.AsyncFunctionDef), (
        f"the lifespan {lifespan_value.id!r} must be an async function"
    )
    assert len(lifespan_fn.decorator_list) == 1, (
        "the lifespan must carry exactly one decorator, @asynccontextmanager; "
        f"found {len(lifespan_fn.decorator_list)}"
    )
    decorator = lifespan_fn.decorator_list[0]
    assert (
        isinstance(decorator, ast.Name) and decorator.id == "asynccontextmanager"
    ) or (
        isinstance(decorator, ast.Attribute) and decorator.attr == "asynccontextmanager"
    ), (
        "the lifespan must be decorated with @asynccontextmanager (bare or "
        "attribute-qualified); without it the async generator is never turned "
        "into the context manager the ASGI server enters at startup"
    )

    lifecycle_calls = _calls_to_name(lifespan_fn, "_initialize_application")
    assert len(lifecycle_calls) == 1, (
        "the FastAPI lifespan must call _initialize_application() exactly once; "
        f"found {len(lifecycle_calls)} call(s)"
    )
    yields = [node for node in ast.walk(lifespan_fn) if isinstance(node, ast.Yield)]
    assert len(yields) == 1, f"the FastAPI lifespan must yield exactly once; found {len(yields)}"
    # A source-order check, kept for readability of the contract. It is not
    # treated as proof of control flow: test_main_lifecycle.py's
    # test_configured_lifespan_initializes_once_before_yield actually drives
    # the configured lifespan and asserts initialization has already happened
    # while it sits suspended at this yield.
    assert lifecycle_calls[0].lineno < yields[0].lineno, (
        "_initialize_application() must appear before the lifespan yields control"
    )


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


def test_upload_route_reports_embedding_failures_through_the_service(main_tree):
    """E1b: the route no longer catches EmbeddingProviderError per file.

    An embedding failure is now classified by DocumentIngestionService as the
    infrastructure-scope ``embedding_unavailable`` category, returned as that
    file's own result with a fixed safe message, and aggregated into the batch
    status -- so a sanitized per-file error still reaches the client without
    the route holding a second copy of the handling. This asserts the route
    does not reintroduce its own handler or its own error string.
    """
    fn = _find_function(main_tree, "upload_documents")
    for try_node in ast.walk(fn):
        if not isinstance(try_node, ast.Try):
            continue
        assert "EmbeddingProviderError" not in [
            _handler_type_name(h) for h in try_node.handlers
        ], (
            "upload_documents must not re-handle embedding failures; the "
            "ingestion service owns that classification"
        )
    assert not _calls_to_embedding_provider_method(fn, "embed_documents"), (
        "upload_documents must not embed anything itself"
    )
    assert not _calls_to_embedding_provider_method(fn, "embed_query")


def test_ingestion_service_returns_a_safe_embedding_failure_message(ingestion_tree):
    """The sanitized message the route used to build now lives in the service's
    fixed message table, as plain string literals that cannot interpolate a raw
    exception."""
    table = None
    for node in ast.walk(ingestion_tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_SAFE_ERROR_MESSAGES"
        ):
            table = node.value
    assert isinstance(table, ast.Dict), "expected a _SAFE_ERROR_MESSAGES dict"
    for value_node in table.values:
        assert isinstance(value_node, ast.Constant) and isinstance(value_node.value, str), (
            "every safe error message must be a plain string literal, never an "
            "f-string that could interpolate the raw exception"
        )
    categories = {key.attr for key in table.keys if isinstance(key, ast.Attribute)}
    assert "EMBEDDING_UNAVAILABLE" in categories
    assert "EMBEDDING_INVALID" in categories


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


# --- 14/15: the embed-before-connect ordering, now owned by the ingestion ---
# --- service; the route keeps only its separate, permitted auth lookup     ---


def test_upload_has_permitted_auth_lookup_connection(main_tree):
    # E1b: the lookup moved out of the route body into _resolve_uploader_id,
    # which upload_documents calls. It is still the only connection the route
    # side opens, and it is still opened via connect_to_postgres().
    fn = _find_function(main_tree, "_resolve_uploader_id")
    lookup_assignments = _assignments_to_name(fn, "lookup_conn")
    assert lookup_assignments, "the auth lookup must retain its lookup_conn connection"
    value = lookup_assignments[0].value
    assert (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "connect_to_postgres"
    ), "lookup_conn must be opened via connect_to_postgres()"

    route = _find_function(main_tree, "upload_documents")
    assert len(_calls_to_name(route, "_resolve_uploader_id")) == 1, (
        "upload_documents must resolve the uploader through that one helper"
    )


def test_ingestion_embeds_then_validates_then_opens_the_write_connection(
    ingestion_source, ingestion_tree
):
    # The ordering rule is unchanged; it is asserted where the code now lives.
    fn = _find_function(ingestion_tree, "ingest")
    segment = _function_source_segment(ingestion_source, fn)

    embed_pos = segment.find("self.embedding_provider.embed_documents(")
    validate_pos = segment.find("_validate_embeddings(")
    connect_pos = segment.find("self.connection_factory()")

    assert embed_pos != -1, "expected the embed_documents call in ingest()"
    assert validate_pos != -1, "expected an explicit embedding validation step"
    assert connect_pos != -1, "expected the write connection to come from the factory"
    assert embed_pos < validate_pos < connect_pos, (
        "ingest() must embed, then validate the embeddings, then open the write "
        "connection -- strictly in that order -- so neither an embedding failure "
        "nor an invalid embedding set ever touches the database"
    )


def test_upload_route_opens_no_write_connection_of_its_own(main_tree):
    fn = _find_function(main_tree, "upload_documents")
    assert not _calls_to_name(fn, "connect_to_postgres"), (
        "the ingestion write connection must come from the service's connection "
        "factory, not from a second connection opened inside the route"
    )


def test_route_passes_connect_to_postgres_as_the_connection_factory(main_tree):
    """pgvector precondition: connect_to_postgres is the only factory in this
    codebase that applies register_vector() to the connection it returns, and
    the service inserts each embedding as a plain list[float]. Passing anything
    else would fail at chunk insertion at runtime."""
    fn = _find_function(main_tree, "_build_ingestion_service")
    calls = _calls_to_name(fn, "DocumentIngestionService")
    assert len(calls) == 1
    kwargs = {kw.arg: kw.value for kw in calls[0].keywords}
    factory = kwargs.get("connection_factory")
    assert isinstance(factory, ast.Name) and factory.id == "connect_to_postgres", (
        "connection_factory= must be the bare name connect_to_postgres"
    )
    assert not _calls_to_name(main_tree, "register_vector"), (
        "main.py must not build a second, unregistered connection path"
    )


def test_route_passes_the_resolved_provider_and_profile_to_the_service(main_tree):
    fn = _find_function(main_tree, "_build_ingestion_service")
    calls = _calls_to_name(fn, "DocumentIngestionService")
    kwargs = {kw.arg: kw.value for kw in calls[0].keywords}
    provider = kwargs.get("embedding_provider")
    profile = kwargs.get("embedding_profile")
    assert isinstance(provider, ast.Name) and provider.id == _EMBEDDING_PROVIDER_NAME, (
        "the service must embed through the application's already-resolved provider"
    )
    assert isinstance(profile, ast.Name) and profile.id == "_EMBEDDING_PROFILE", (
        "the service must validate against the already-resolved profile"
    )


def test_route_returns_the_services_batch_status_as_the_http_status(
    main_tree, main_source
):
    fn = _find_function(main_tree, "upload_documents")
    segment = _function_source_segment(main_source, fn)
    assert "status_code=batch.http_status" in segment, (
        "the service's aggregated 200/207/422/503 must become the real HTTP "
        "status, not a field buried inside a 200 body"
    )
    assert "batch.to_dict()" in segment, (
        "the complete batch envelope must be returned as-is"
    )


# --- 16/17/18: page-aware chunking, locator persistence, and preserved ---
# --- ownership -- all now asserted against document_ingestion.py         ---


def _service_header(ingestion_source, ingestion_tree):
    """The DocumentIngestionService class body down to its first method, where
    the injected parse/chunk/locator/JSON defaults are bound."""
    service = _find_class(ingestion_tree, "DocumentIngestionService")
    return _function_source_segment(ingestion_source, service).split("def ingest")[0]


def test_ingestion_uses_chunk_document_and_page_aware_chunk_fields(
    ingestion_source, ingestion_tree
):
    fn = _find_function(ingestion_tree, "ingest")
    segment = _function_source_segment(ingestion_source, fn)
    assert "self.chunker(parsed)" in segment, "ingest() must chunk through its chunker"
    assert "= chunk_document" in _service_header(ingestion_source, ingestion_tree), (
        "the chunker must default to document_chunker.chunk_document"
    )
    for attr in ("chunk.text", "chunk.chunk_index", "chunk.page_start", "chunk.page_end"):
        assert attr in segment, f"expected {attr} to be used when inserting a chunk row"


def test_ingestion_builds_and_wraps_locator_json(ingestion_source, ingestion_tree):
    fn = _find_function(ingestion_tree, "ingest")
    segment = _function_source_segment(ingestion_source, fn)
    assert "self.locator_builder(chunk, parsed.document_type)" in segment, (
        "each chunk's locator must be built from the chunk and its document type"
    )
    assert "self.json_adapter(locator)" in segment, (
        "the locator dict must be wrapped by the JSON adapter before insertion"
    )
    header = _service_header(ingestion_source, ingestion_tree)
    assert "= build_chunk_locator" in header, (
        "locator_builder must default to document_chunker.build_chunk_locator"
    )
    assert "= Json" in header, "json_adapter must default to psycopg2.extras.Json"


def test_ingestion_chunk_insert_names_required_columns(ingestion_source, ingestion_tree):
    fn = _find_function(ingestion_tree, "ingest")
    segment = _function_source_segment(ingestion_source, fn)
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
        assert column in segment, (
            f"expected the document_chunks insert to name column {column!r}"
        )


def test_ingestion_preserves_uploader_id_in_document_insert(
    ingestion_source, ingestion_tree
):
    fn = _find_function(ingestion_tree, "ingest")
    segment = _function_source_segment(ingestion_source, fn)
    assert "INSERT INTO documents" in segment
    assert "user_id" in segment, "the documents insert must still name the user_id column"
    assert "upload.uploader_id" in segment, (
        "the documents insert must still pass the upload's uploader_id as its value"
    )


def test_route_forwards_the_resolved_uploader_id_into_the_upload_contract(
    main_tree, main_source
):
    fn = _find_function(main_tree, "upload_documents")
    segment = _function_source_segment(main_source, fn)
    assert "uploader_id=uploader_id" in segment, (
        "ownership is preserved by handing the resolved uploader_id to "
        "UploadDocument; the service writes it to documents.user_id"
    )


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
