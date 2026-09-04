"""Source/AST contract tests for the I3 local-generation policy.

These tests read and parse source files only. They never import or execute
``main.py`` (which unconditionally calls ``init_db(...)`` at module scope) or
``GridApp.jsx``, and make no network, database, subprocess, or filesystem-write
calls. Negative checks for the removed external provider are paired with
positive checks that ``ask_question`` retains its local Ollama
``/api/generate`` streaming structure, failed output is not persisted, and
the admin-status UI/backend still report the fields that were not removed.

These contracts do not validate successful Ollama protocol behavior, model
availability, mxbai/Nomic, dimensions, schema compatibility, production
deployment, or the unrelated Resend or Google egress categories.
"""

import ast
from pathlib import Path

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_MAIN_PATH = _REPOSITORY_ROOT / "synerge-reader-backend" / "main.py"
_README_PATH = _REPOSITORY_ROOT / "README.md"
_GRIDAPP_PATH = _REPOSITORY_ROOT / "synerge-reader-frontend" / "src" / "GridApp.jsx"
_THIS_TEST_PATH = Path(__file__).resolve()


@pytest.fixture(scope="module")
def main_source():
    return _MAIN_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def main_tree(main_source):
    return ast.parse(main_source, filename=str(_MAIN_PATH))


@pytest.fixture(scope="module")
def readme_source():
    return _README_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def gridapp_source():
    return _GRIDAPP_PATH.read_text(encoding="utf-8")


def _functions_named(node, name):
    return [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
        and candidate.name == name
    ]


def _one_function_named(node, name):
    matches = _functions_named(node, name)
    assert len(matches) == 1, f"expected exactly one function named {name!r}"
    return matches[0]


def _bare_name_calls(node, name):
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == name
    ]


def _is_requests_post(call):
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "post"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "requests"
    )


def _generation_call(stream_generate):
    calls = _bare_name_calls(stream_generate, "post_ollama")
    assert len(calls) == 1, "stream_generate must contain one local generation call"
    return calls[0]


def _generation_try(stream_generate):
    generation_call = _generation_call(stream_generate)
    matches = []
    for candidate in ast.walk(stream_generate):
        if not isinstance(candidate, ast.Try):
            continue
        body_nodes = [nested for statement in candidate.body for nested in ast.walk(statement)]
        if generation_call in body_nodes:
            matches.append(candidate)
    assert len(matches) == 1, "local generation call must be inside one try statement"
    return matches[0]


def _literal_text(node):
    return "".join(
        candidate.value
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str)
    )


# --- absence of the external generation provider ---


def test_main_has_no_external_generation_provider_tokens(main_source):
    lowered = main_source.lower()
    assert "openrouter" not in lowered
    assert "chat/completions" not in lowered


def test_external_streaming_function_is_absent(main_tree):
    assert not _functions_named(main_tree, "stream_openrouter_chat")


def test_no_external_provider_identifiers_remain(main_tree):
    identifiers = []
    for node in ast.walk(main_tree):
        for attribute in ("id", "arg", "name", "attr"):
            value = getattr(node, attribute, None)
            if isinstance(value, str):
                identifiers.append(value)
    assert not [name for name in identifiers if name.upper().startswith("OPENROUTER_")]


# --- positive checks: local Ollama streaming structure is intact ---


def test_ask_retains_nested_local_generation_function(main_tree):
    ask_question = _one_function_named(main_tree, "ask_question")
    stream_generate = _one_function_named(ask_question, "stream_generate")
    assert stream_generate is not ask_question


def test_local_generation_call_retains_required_structure(main_tree):
    ask_question = _one_function_named(main_tree, "ask_question")
    stream_generate = _one_function_named(ask_question, "stream_generate")
    call = _generation_call(stream_generate)

    assert len(call.args) >= 1
    assert isinstance(call.args[0], ast.Constant)
    assert call.args[0].value == "/api/generate"

    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert isinstance(keywords.get("stream"), ast.Constant)
    assert keywords["stream"].value is True
    assert isinstance(keywords.get("timeout"), ast.Constant)
    assert keywords["timeout"].value == 60

    assert any(
        isinstance(candidate, ast.With)
        and any(item.context_expr is call for item in candidate.items)
        for candidate in ast.walk(stream_generate)
    ), "post_ollama must remain a context-managed streaming call"


def test_ask_has_no_direct_requests_post_or_fallback_message_state(main_tree):
    ask_question = _one_function_named(main_tree, "ask_question")
    assert not [call for call in ast.walk(ask_question) if _is_requests_post(call)]
    assert not [
        name
        for name in ast.walk(ask_question)
        if isinstance(name, ast.Name)
        and isinstance(name.ctx, ast.Store)
        and name.id == "fallback_messages"
    ]


def test_local_generation_exception_is_safe_and_has_no_fallback_call(main_tree):
    ask_question = _one_function_named(main_tree, "ask_question")
    stream_generate = _one_function_named(ask_question, "stream_generate")
    generation_try = _generation_try(stream_generate)
    handlers = [
        handler
        for handler in generation_try.handlers
        if isinstance(handler.type, ast.Name) and handler.type.id == "Exception"
    ]
    assert len(handlers) == 1
    handler = handlers[0]

    stream_error_assignments = [
        assignment
        for assignment in ast.walk(handler)
        if isinstance(assignment, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "stream_error"
            for target in assignment.targets
        )
    ]
    assert len(stream_error_assignments) == 1
    assigned_value = stream_error_assignments[0].value
    assert isinstance(assigned_value, ast.Constant) and assigned_value.value is True

    yields = [node for node in ast.walk(handler) if isinstance(node, ast.Yield)]
    assert len(yields) == 2
    yielded_text = [_literal_text(node.value).lower() for node in yields]
    assert all(text.startswith("__error__") and "ollama" in text for text in yielded_text)
    assert any("model" in text and "installed" in text for text in yielded_text)
    assert any(
        "local llm server is not reachable" in text and "ollama_base_url" in text
        for text in yielded_text
    )
    assert all(
        not any(isinstance(node, ast.Name) and node.id == handler.name for node in ast.walk(yielded.value))
        for yielded in yields
    ), "raw generation exceptions must not be yielded"

    calls = [node for node in ast.walk(handler) if isinstance(node, ast.Call)]
    assert all(
        isinstance(call.func, ast.Name) and call.func.id == "getattr" for call in calls
    ), "the local-generation exception handler must not invoke a fallback generator"


def test_failed_or_empty_generation_guard_precedes_chat_history_persistence(main_tree):
    ask_question = _one_function_named(main_tree, "ask_question")
    stream_generate = _one_function_named(ask_question, "stream_generate")

    guard_indexes = []
    persistence_indexes = []
    for index, statement in enumerate(stream_generate.body):
        if isinstance(statement, ast.If) and isinstance(statement.test, ast.BoolOp):
            values = statement.test.values
            is_required_guard = (
                isinstance(statement.test.op, ast.Or)
                and len(values) == 2
                and isinstance(values[0], ast.Name)
                and values[0].id == "stream_error"
                and isinstance(values[1], ast.UnaryOp)
                and isinstance(values[1].op, ast.Not)
                and isinstance(values[1].operand, ast.Name)
                and values[1].operand.id == "answer_parts"
                and any(isinstance(node, ast.Return) for node in statement.body)
            )
            if is_required_guard:
                guard_indexes.append(index)

        if any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "insert into chat_history" in node.value.lower()
            for node in ast.walk(statement)
        ):
            persistence_indexes.append(index)

    assert len(guard_indexes) == 1
    assert len(persistence_indexes) == 1
    assert guard_indexes[0] < persistence_indexes[0]


# --- backend admin-status field removal (paired with what must remain) ---


def test_admin_system_status_return_dict_has_no_openrouter_key(main_tree):
    fn = _one_function_named(main_tree, "admin_system_status")
    return_dicts = [
        node.value
        for node in ast.walk(fn)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    ]
    assert return_dicts, "expected admin_system_status to return a dict literal"
    for return_dict in return_dicts:
        keys = [k.value for k in return_dict.keys if isinstance(k, ast.Constant)]
        assert "openrouter_fallback_configured" not in keys, (
            "admin_system_status must no longer report an OpenRouter fallback status field"
        )
        assert "database" in keys, "admin_system_status must still report database status"
        assert "llm_backend" in keys, "admin_system_status must still report llm_backend status"


# --- frontend admin-status row removal (paired with what must remain) ---


def test_gridapp_has_no_openrouter_references(gridapp_source):
    assert "openrouter" not in gridapp_source.lower()


def test_gridapp_system_status_card_sibling_rows_preserved(gridapp_source):
    # The removed row was one specific row inside the "System status" card --
    # this proves that removal didn't collaterally take the card's other
    # rows with it (a wholesale deletion of the whole card would also
    # satisfy the plain absence check above, but must not pass this one).
    lowered = gridapp_source.lower()
    assert "adminsystemstatus.database.ok" in lowered
    assert "adminsystemstatus.llm_backend.ok" in lowered


# --- README policy ---


def test_readme_documents_local_ollama_generation_only(readme_source):
    lowered = readme_source.lower()
    assert "openrouter" not in lowered
    assert "local ollama" in lowered
    assert "generation" in lowered
    assert "no external generation fallback" in lowered


# --- this file makes no side-effecting calls of its own ---


def test_contract_suite_has_no_side_effect_imports_or_filesystem_writes():
    test_source = _THIS_TEST_PATH.read_text(encoding="utf-8")
    test_tree = ast.parse(test_source, filename=str(_THIS_TEST_PATH))

    forbidden_import_roots = {
        "asyncpg",
        "httpx",
        "psycopg2",
        "requests",
        "socket",
        "sqlalchemy",
        "subprocess",
        "docker",
    }
    imported_roots = set()
    for node in ast.walk(test_tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint(forbidden_import_roots)

    write_methods = {
        "chmod",
        "mkdir",
        "rename",
        "replace",
        "rmdir",
        "touch",
        "unlink",
        "write",
        "write_bytes",
        "write_text",
        "writelines",
    }
    assert not [
        call
        for call in ast.walk(test_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in write_methods
    ]
