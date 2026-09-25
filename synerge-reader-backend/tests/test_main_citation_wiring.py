"""Transport and route-composition contracts for the E2 /ask stream.

Three kinds of test live here, and none imports the application:

* NDJSON transport tests exercise ``ask_stream.py`` in memory. That module is
  pure encode/decode -- no route, no connection, no model -- which is why the
  wire format can be tested at all without starting anything.
* Route-composition tests parse ``main.py`` with the stdlib ``ast`` module.
* Route-execution tests compile only ``ask_question`` and the claim verifier
  out of that same parse tree into an isolated namespace in which every
  collaborator -- Ollama, the database, evidence planning, the knowledge base
  -- is an in-memory stub, then drive the real stream generator and decode
  what it emits.

main.py is never imported, so none of its module-level setup runs, and no
database, Ollama, network, subprocess, or filesystem write occurs.

What these prove: every event is an independently valid JSON line, arbitrary
network chunk boundaries reconstruct exactly, old sentinel text inside an
answer is data rather than control, the entry-id/history behaviour is intact
and still guarded, a generation or history-persistence failure terminates with
a safe error followed by an explicit non-success done, an uncommitted history
write is rolled back and every opened connection closed, the local-only
generation policy survives the transport change, the claim verifier sends its
temperature as an Ollama option and accepts only a reply that is exactly one
status, and the route composes the evidence planner and citation registry
instead of re-deciding evidence priority itself.

What these do NOT prove: Starlette/ASGI streaming, Ollama protocol
behaviour, SQL validity, or anything about the frontend.
"""

import ast
import datetime
import json
import os
from pathlib import Path
import re
import sys
import threading
from types import SimpleNamespace
from typing import Optional

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from answer_evidence import (
    EvidenceBundle,
    EvidenceItem,
    EvidenceMode,
    EvidencePlanningResult,
)
from ask_stream import (
    EVENT_TYPES,
    EventStreamDecoder,
    STREAM_MEDIA_TYPE,
    delta_event,
    done_event,
    encode_event,
    entry_id_event,
    error_event,
    evidence_event,
    iter_events,
    verification_event,
)
from citation_generation import (
    CitationGenerationResult,
    CitationLimits,
    CitationRegistry,
    build_generation_prompt,
    generate_citations,
    safe_error,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_MAIN_PATH = _REPOSITORY_ROOT / "synerge-reader-backend" / "main.py"


@pytest.fixture(scope="module")
def main_source():
    return _MAIN_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def main_tree(main_source):
    return ast.parse(main_source, filename=str(_MAIN_PATH))


def _find_function(node, name):
    for candidate in ast.walk(node):
        if (
            isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
            and candidate.name == name
        ):
            return candidate
    raise AssertionError(f"function {name!r} not found")


def _calls_to_name(node, name):
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == name
    ]


def _segment(source, node):
    return "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])


def _sample_stream():
    return (
        evidence_event(
            [{"citation_id": "C1", "filename": "agreement.pdf", "display_label": "agreement.pdf · page 4"}],
            mode="hybrid_retrieval",
            truncated=False,
            warnings=["evidence_truncated"],
        )
        + delta_event("The term is three years ")
        + delta_event("[C1].")
        + verification_event(
            [{"text": "The term is three years [C1].", "citation_ids": ["C1"], "status": "supported"}],
            invalid_citation_ids=["C9"],
        )
        + entry_id_event(4242)
        + done_event(ok=True)
    )


# --- route-execution harness ------------------------------------------------
#
# Only these definitions are compiled out of main.py; every other name they
# read is supplied by _route_harness below. Nothing else in main.py runs.

_ROUTE_DEFINITIONS = (
    "_CLAIM_VERIFIER_PROMPT",
    "_CLAIM_VERIFIER_WORDS",
    "_claim_verifier_status",
    "_ollama_claim_verifier",
    "_resolve_citations",
    "ask_question",
)

_ANSWER_TOKENS = ("The term is three years ", "[C1].")


def _defined_name(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id
    return None


def _compile_route_definitions(main_tree, namespace):
    selected = [node for node in main_tree.body if _defined_name(node) in _ROUTE_DEFINITIONS]
    assert sorted(_defined_name(node) for node in selected) == sorted(_ROUTE_DEFINITIONS)
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(_MAIN_PATH), "exec"), namespace)


class _SimulatedDatabaseError(Exception):
    pass


class _FakeCursor:
    def __init__(self, connection):
        self._connection = connection
        self._row = None

    def execute(self, sql, params=()):
        if "INSERT INTO chat_history" in sql:
            self._connection._step("insert")
            self._row = (self._connection.entry_id,)
        else:
            self._connection._step("select")
            self._row = (7,)

    def fetchone(self):
        return self._row


class _FakeConnection:
    """Records each transaction step in order; ``failures`` makes a step raise.

    Steps: ``select`` (user lookup), ``insert``, ``commit``, ``rollback``,
    ``close``.
    """

    def __init__(self, *, entry_id=4242, **failures):
        self.entry_id = entry_id
        self.failures = failures
        self.steps = []

    def _step(self, name):
        self.steps.append(name)
        if name in self.failures:
            raise self.failures[name]

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        self._step("commit")

    def rollback(self):
        self._step("rollback")

    def close(self):
        self._step("close")


class _FakeGenerationStream:
    """A streamed /api/generate response: NDJSON lines cut into small chunks."""

    def __init__(self, tokens):
        self._payload = "".join(json.dumps({"response": token}) + "\n" for token in tokens)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, decode_unicode=True, chunk_size=32):
        for index in range(0, len(self._payload), chunk_size):
            yield self._payload[index : index + chunk_size]


class _FakeVerifierResponse:
    def __init__(self, word):
        self._word = word

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": self._word}


class _CapturedStreamingResponse:
    def __init__(self, content, media_type=None, headers=None):
        self.content = content
        self.media_type = media_type
        self.headers = headers


def _evidence_bundle():
    return EvidenceBundle(
        mode=EvidenceMode.HYBRID_RETRIEVAL,
        items=(
            EvidenceItem(
                text="The agreement term is three years from signature.",
                source_type="document_chunk",
                document_id=1,
                filename="agreement.pdf",
                chunk_id="1-0",
                chunk_index=0,
            ),
        ),
    )


def _route_harness(main_tree, connection, verifier_reply="supported"):
    """The route's real code, with every collaborator an in-memory stub."""
    calls = SimpleNamespace(ollama=[], auto_saved=threading.Event())
    planning = EvidencePlanningResult(bundle=_evidence_bundle())

    def post_ollama(endpoint, payload, *, stream=False, timeout=60):
        calls.ollama.append(
            SimpleNamespace(endpoint=endpoint, payload=payload, stream=stream, timeout=timeout)
        )
        if stream:
            return _FakeGenerationStream(_ANSWER_TOKENS)
        return _FakeVerifierResponse(verifier_reply)

    namespace = {
        "app": SimpleNamespace(post=lambda path: (lambda function: function)),
        "AskRequest": object,
        "Optional": Optional,
        "re": re,
        "StreamingResponse": _CapturedStreamingResponse,
        "STREAM_MEDIA_TYPE": STREAM_MEDIA_TYPE,
        "EmbeddingProviderError": type("EmbeddingProviderError", (Exception,), {}),
        "OLLAMA_KEEP_ALIVE": "30m",
        "json": json,
        "datetime": datetime,
        "post_ollama": post_ollama,
        "connect_to_postgres": lambda: connection,
        "_resolve_authorized_scope": lambda auth_token: "authorized-scope",
        "_build_answer_evidence_planner": lambda scope: SimpleNamespace(
            plan=lambda request, scope: planning
        ),
        "_evidence_request_from_ask": lambda request, scope: request,
        "get_relevant_knowledge_base": lambda question, limit=3: [],
        "increment_kb_usage": lambda ids: None,
        "auto_save_to_kb": lambda *args: calls.auto_saved.set(),
        "CitationRegistry": CitationRegistry,
        "CitationGenerationResult": CitationGenerationResult,
        "_CITATION_LIMITS": CitationLimits(max_claims_verified=2),
        "build_generation_prompt": build_generation_prompt,
        "generate_citations": generate_citations,
        "safe_error": safe_error,
        "evidence_event": evidence_event,
        "delta_event": delta_event,
        "verification_event": verification_event,
        "entry_id_event": entry_id_event,
        "error_event": error_event,
        "done_event": done_event,
    }
    _compile_route_definitions(main_tree, namespace)
    return namespace, calls


def _call_without_event_loop(coroutine):
    """ask_question awaits nothing, so one send() runs it to its return."""
    try:
        coroutine.send(None)
    except StopIteration as finished:
        return finished.value
    coroutine.close()
    raise AssertionError("ask_question suspended; this harness runs no event loop")


def _stream_ask(main_tree, connection, *, auth_token=None, verifier_reply="supported"):
    namespace, calls = _route_harness(main_tree, connection, verifier_reply)
    request = SimpleNamespace(
        question="How long is the agreement term?",
        selected_text="",
        auth_token=auth_token,
        model="local-test-model",
    )
    response = _call_without_event_loop(namespace["ask_question"](request))
    assert response.media_type == STREAM_MEDIA_TYPE

    payload = "".join(response.content)
    decoder = EventStreamDecoder()
    events = decoder.feed(payload) + decoder.close()
    assert decoder.malformed_lines == 0, "every emitted line must parse as an event"
    return payload, events, calls


# --- 25: every event is an independently valid JSON line -------------------


def test_every_event_line_parses_independently():
    payload = _sample_stream()
    lines = [line for line in payload.split("\n") if line]

    assert len(lines) == 6
    for line in lines:
        event = json.loads(line)
        assert isinstance(event, dict)
        assert event["type"] in EVENT_TYPES
    assert [json.loads(line)["type"] for line in lines] == [
        "evidence",
        "delta",
        "delta",
        "verification",
        "entry_id",
        "done",
    ]


def test_each_event_carries_only_its_own_payload():
    evidence = json.loads(evidence_event([{"citation_id": "C1"}], mode="selected_text").strip())
    delta = json.loads(delta_event("answer text").strip())
    verification = json.loads(verification_event([{"status": "unverified"}]).strip())

    assert set(evidence) == {"type", "citations", "mode", "truncated", "warnings"}
    assert set(delta) == {"type", "text"}, "a delta event carries answer text and nothing else"
    assert delta["text"] == "answer text"
    assert set(verification) == {
        "type",
        "claims",
        "invalid_citation_ids",
        "used_citation_ids",
    }


def test_unknown_event_types_cannot_be_emitted():
    with pytest.raises(ValueError):
        encode_event("context", text="anything")
    with pytest.raises(ValueError):
        encode_event("__READY__")


def test_error_events_are_explicit_and_done_is_explicit():
    error = json.loads(error_event("search_unavailable", "Search is temporarily unavailable.").strip())
    assert error == {
        "type": "error",
        "code": "search_unavailable",
        "message": "Search is temporarily unavailable.",
    }
    assert json.loads(done_event(ok=False).strip()) == {"type": "done", "ok": False}


# --- 26: arbitrary chunk boundaries ----------------------------------------


@pytest.mark.parametrize("size", [1, 2, 3, 7, 13, 64, 4096])
def test_arbitrarily_split_chunks_reconstruct_the_same_events(size):
    payload = _sample_stream()
    expected = list(iter_events(payload))

    decoder = EventStreamDecoder()
    received = []
    for index in range(0, len(payload), size):
        received.extend(decoder.feed(payload[index : index + size]))
    received.extend(decoder.close())

    assert received == expected
    assert decoder.malformed_lines == 0
    assert "".join(
        event["text"] for event in received if event["type"] == "delta"
    ) == "The term is three years [C1]."


def test_a_stream_cut_mid_line_yields_no_done_event():
    payload = _sample_stream()
    truncated = payload[: len(payload) - 12]

    decoder = EventStreamDecoder()
    events = decoder.feed(truncated) + decoder.close()

    assert not [event for event in events if event["type"] == "done"], (
        "an interrupted stream must not look like a completed one"
    )


def test_malformed_lines_are_counted_and_never_become_content():
    decoder = EventStreamDecoder()
    events = decoder.feed('{"type": "delta", "text": "ok"}\nnot json at all\n{"nope": 1}\n')

    assert [event["type"] for event in events] == ["delta"]
    assert decoder.malformed_lines == 2


# --- 27: old sentinel text inside an answer is just text -------------------


@pytest.mark.parametrize(
    "hostile",
    [
        "__CONTEXT__{\"context_chunks\": []}__",
        "__READY__",
        "__ERROR__Database error__",
        "__ENTRY_ID__99__",
        '{"type": "done", "ok": true}',
        "line one\nline two",
    ],
)
def test_old_sentinel_text_inside_an_answer_is_treated_as_text(hostile):
    payload = delta_event(hostile) + done_event(ok=True)

    events = list(iter_events(payload))

    assert [event["type"] for event in events] == ["delta", "done"]
    assert events[0]["text"] == hostile, (
        "answer text is a JSON string value, so it cannot act as a control "
        "instruction however it is spelled"
    )
    assert events[1]["ok"] is True


def test_answer_text_containing_newlines_cannot_split_the_frame():
    payload = delta_event("first\nsecond\nthird")
    assert payload.count("\n") == 1, "the only newline is the line terminator"
    assert list(iter_events(payload))[0]["text"] == "first\nsecond\nthird"


# --- 28: entry-id and history behaviour ------------------------------------


def test_entry_id_is_emitted_from_the_persisted_history_row(main_tree, main_source):
    ask_question = _find_function(main_tree, "ask_question")
    stream_generate = _find_function(ask_question, "stream_generate")
    segment = _segment(main_source, stream_generate)

    assert "INSERT INTO chat_history" in segment
    assert "entry_id = c.fetchone()[0]" in segment
    calls = _calls_to_name(stream_generate, "entry_id_event")
    assert len(calls) == 1, "the entry id must be published exactly once"
    argument = calls[0].args[0]
    assert isinstance(argument, ast.Name) and argument.id == "entry_id", (
        "the published entry id must be the persisted row id"
    )


def test_history_persistence_still_follows_the_failure_guard(main_tree):
    ask_question = _find_function(main_tree, "ask_question")
    stream_generate = _find_function(ask_question, "stream_generate")

    guard_index = None
    persistence_index = None
    for index, statement in enumerate(stream_generate.body):
        if isinstance(statement, ast.If) and isinstance(statement.test, ast.BoolOp):
            values = statement.test.values
            if (
                len(values) == 2
                and isinstance(values[0], ast.Name)
                and values[0].id == "stream_error"
                and any(isinstance(node, ast.Return) for node in statement.body)
            ):
                guard_index = index
        if any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "insert into chat_history" in node.value.lower()
            for node in ast.walk(statement)
        ):
            persistence_index = index

    assert guard_index is not None and persistence_index is not None
    assert guard_index < persistence_index, (
        "a failed or empty generation must still never be written to history"
    )


def test_history_failure_reports_a_safe_error_not_a_raw_exception(main_tree, main_source):
    ask_question = _find_function(main_tree, "ask_question")
    stream_generate = _find_function(ask_question, "stream_generate")
    segment = _segment(main_source, stream_generate)

    assert "safe_error(" in segment, "history failure must use the fixed safe-error table"
    assert "Database error" not in segment
    for handler in ast.walk(stream_generate):
        if not isinstance(handler, ast.ExceptHandler):
            continue
        for node in ast.walk(handler):
            if isinstance(node, ast.Yield) and handler.name:
                assert not any(
                    isinstance(inner, ast.Name) and inner.id == handler.name
                    for inner in ast.walk(node)
                ), "no handler may yield its bound exception"


# --- 29: generation failure terminates safely ------------------------------


def test_generation_failure_ends_with_a_safe_error_then_an_explicit_done(main_tree, main_source):
    ask_question = _find_function(main_tree, "ask_question")
    stream_generate = _find_function(ask_question, "stream_generate")

    generation_call = _calls_to_name(stream_generate, "post_ollama")[0]
    generation_try = None
    for candidate in ast.walk(stream_generate):
        if isinstance(candidate, ast.Try) and any(
            generation_call is node
            for statement in candidate.body
            for node in ast.walk(statement)
        ):
            generation_try = candidate
    assert generation_try is not None

    handler = [
        h for h in generation_try.handlers
        if isinstance(h.type, ast.Name) and h.type.id == "Exception"
    ][0]
    yields = [node for node in ast.walk(handler) if isinstance(node, ast.Yield)]
    assert len(yields) == 2
    for node in yields:
        assert isinstance(node.value, ast.Constant), (
            "a generation-failure line must be a literal, so no exception text "
            "can be interpolated into it"
        )
        event = json.loads(node.value.value)
        assert event["type"] == "error"
        assert isinstance(event["message"], str) and event["message"]
        rendered = json.dumps(event).lower()
        for leak in ("traceback", "psycopg2", "select ", "password", "db_connection_string"):
            assert leak not in rendered

    segment = _segment(main_source, stream_generate)
    assert "done_event(ok=False)" in segment, (
        "an interrupted or empty generation must terminate with an explicit "
        "unsuccessful done, never silently"
    )
    assert "done_event(ok=True)" in segment


def test_every_error_path_closes_the_stream_unsuccessfully(main_tree, main_source):
    """No error path may leave the stream open or end it as a success.

    A handler either emits its own ``done(ok=false)`` and returns, or it sets
    ``stream_error`` so the shared failure guard emits that unsuccessful done
    on its behalf. Nothing else is accepted.
    """
    ask_question = _find_function(main_tree, "ask_question")
    stream_generate = _find_function(ask_question, "stream_generate")

    checked = 0
    for handler in ast.walk(stream_generate):
        if not isinstance(handler, ast.ExceptHandler):
            continue
        literal_yields = [
            node
            for node in ast.walk(handler)
            if isinstance(node, ast.Yield)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ]
        events = []
        for node in literal_yields:
            try:
                events.append(json.loads(node.value.value))
            except Exception:
                raise AssertionError("a literal yield in an error path is not valid JSON")
        if not any(event.get("type") == "error" for event in events):
            continue
        checked += 1

        ends_itself = events[-1].get("type") == "done" and events[-1].get("ok") is False
        defers_to_guard = any(
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "stream_error" for t in node.targets)
            for node in ast.walk(handler)
        )
        assert ends_itself or defers_to_guard, (
            "an error path must either emit done(ok=false) itself or set "
            "stream_error so the shared guard does"
        )
        if defers_to_guard:
            assert "done_event(ok=False)" in _segment(main_source, stream_generate)

    assert checked >= 3, (
        "expected the search-unavailable, evidence-unavailable and generation "
        f"error paths to be checked; checked {checked}"
    )


_SUCCESS_EVENT_TYPES = ["delta", "evidence", "delta", "delta", "verification", "entry_id", "done"]
_FAILURE_EVENT_TYPES = ["delta", "evidence", "delta", "delta", "verification", "error", "done"]
_DATABASE_SECRETS = ("hunter2", "db.internal", "psycopg2", "INSERT")


def _database_error(step):
    return _SimulatedDatabaseError(f"psycopg2 {step}: password=hunter2 host=db.internal")


def _assert_no_database_detail(payload, logged):
    for leak in _DATABASE_SECRETS:
        assert leak not in payload, f"{leak!r} reached the client"
        assert leak not in logged, f"{leak!r} reached the server log"


def _assert_safe_history_failure(payload, events, calls, logged):
    assert [event["type"] for event in events] == _FAILURE_EVENT_TYPES
    assert events[-2] == {"type": "error", **safe_error("internal_error")}
    assert events[-1] == {"type": "done", "ok": False}, (
        "a history failure must not terminate the stream as a success"
    )
    assert not calls.auto_saved.is_set(), "an unpersisted answer must not reach the KB"
    _assert_no_database_detail(payload, logged)


def test_a_persisted_answer_streams_its_entry_id_then_a_successful_done(main_tree):
    connection = _FakeConnection(entry_id=4242)

    _, events, calls = _stream_ask(main_tree, connection)

    assert [event["type"] for event in events] == _SUCCESS_EVENT_TYPES
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "".join(_ANSWER_TOKENS)
    assert events[4]["used_citation_ids"] == ["C1"]
    assert events[5]["entry_id"] == 4242
    assert events[-1] == {"type": "done", "ok": True}

    assert connection.steps == ["insert", "commit", "close"], (
        "a persisted answer commits and closes, and is never rolled back"
    )
    assert calls.auto_saved.wait(timeout=5), "the persisted answer must still reach the KB"


@pytest.mark.parametrize(
    ("failing_step", "auth_token", "expected_steps"),
    [
        ("select", "user-token", ["select", "rollback", "close"]),
        ("insert", None, ["insert", "rollback", "close"]),
        ("commit", None, ["insert", "commit", "rollback", "close"]),
    ],
    ids=["user-lookup-execute", "history-insert-execute", "commit"],
)
def test_a_pre_commit_history_failure_rolls_back_closes_and_ends_not_ok(
    main_tree, capsys, failing_step, auth_token, expected_steps
):
    connection = _FakeConnection(**{failing_step: _database_error(failing_step)})

    payload, events, calls = _stream_ask(main_tree, connection, auth_token=auth_token)

    assert connection.steps == expected_steps, (
        "an uncommitted write must be rolled back and then closed"
    )
    _assert_safe_history_failure(payload, events, calls, capsys.readouterr().out)


@pytest.mark.parametrize("failing_step", ["insert", "commit"])
def test_a_failing_cleanup_neither_masks_the_safe_error_nor_skips_the_close(
    main_tree, capsys, failing_step
):
    connection = _FakeConnection(
        **{
            failing_step: _database_error(failing_step),
            "rollback": _database_error("rollback"),
            "close": _database_error("close"),
        }
    )

    payload, events, calls = _stream_ask(main_tree, connection)

    assert connection.steps[-2:] == ["rollback", "close"], (
        "close must still be attempted after the rollback itself failed"
    )
    _assert_safe_history_failure(payload, events, calls, capsys.readouterr().out)


def test_a_close_failure_after_commit_keeps_the_committed_answer_successful(
    main_tree, capsys
):
    connection = _FakeConnection(entry_id=4242, close=_database_error("close"))

    payload, events, calls = _stream_ask(main_tree, connection)

    assert connection.steps == ["insert", "commit", "close"], (
        "a committed write must never be rolled back"
    )
    assert [event["type"] for event in events] == _SUCCESS_EVENT_TYPES, (
        "the row is durable, so a failed close must not report the answer as failed"
    )
    assert events[-2] == {"type": "entry_id", "entry_id": 4242}
    assert events[-1] == {"type": "done", "ok": True}
    assert calls.auto_saved.wait(timeout=5)
    _assert_no_database_detail(payload, capsys.readouterr().out)


def test_an_unavailable_history_connection_ends_not_ok_with_nothing_to_clean_up(
    main_tree, capsys
):
    payload, events, calls = _stream_ask(main_tree, None)

    _assert_safe_history_failure(payload, events, calls, capsys.readouterr().out)


def test_stream_declares_the_ndjson_media_type(main_tree, main_source):
    ask_question = _find_function(main_tree, "ask_question")
    segment = _segment(main_source, ask_question)
    assert "media_type=STREAM_MEDIA_TYPE" in segment
    assert STREAM_MEDIA_TYPE == "application/x-ndjson"
    assert "text/plain" not in segment


# --- 30: local-only generation policy --------------------------------------


def test_local_only_generation_policy_remains_enforced(main_tree, main_source):
    ask_question = _find_function(main_tree, "ask_question")
    stream_generate = _find_function(ask_question, "stream_generate")

    calls = _calls_to_name(stream_generate, "post_ollama")
    assert len(calls) == 1
    assert isinstance(calls[0].args[0], ast.Constant)
    assert calls[0].args[0].value == "/api/generate"

    lowered = main_source.lower()
    assert "openrouter" not in lowered
    assert "chat/completions" not in lowered
    assert "api.openai" not in lowered
    assert not [
        call for call in ast.walk(ask_question)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "post"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "requests"
    ]


def test_the_runtime_claim_verifier_is_local_only(main_tree):
    verifier = _find_function(main_tree, "_ollama_claim_verifier")
    calls = _calls_to_name(verifier, "post_ollama")
    assert len(calls) == 1, "verification must reuse the one local transport"
    assert calls[0].args[0].value == "/api/generate"
    assert not [
        call for call in ast.walk(verifier)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in ("post", "get")
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "requests"
    ]


def test_the_claim_verifier_sends_temperature_as_an_ollama_option(main_tree):
    """Ollama reads runtime generation settings from ``options`` and ignores an
    unknown top-level ``temperature``, so the deterministic verifier setting
    must travel there to take effect."""
    namespace, calls = _route_harness(main_tree, _FakeConnection())
    records = CitationRegistry.from_bundle(_evidence_bundle()).records

    status = namespace["_ollama_claim_verifier"](
        "The term is three years [C1].", records, "local-verifier-model"
    )

    assert status == "supported"
    assert len(calls.ollama) == 1
    request = calls.ollama[0]
    assert request.endpoint == "/api/generate"
    assert request.stream is False and request.timeout == 30
    assert set(request.payload) == {"model", "prompt", "stream", "options", "keep_alive"}
    assert request.payload["options"] == {"temperature": 0.0}
    assert request.payload["model"] == "local-verifier-model"
    assert request.payload["stream"] is False
    assert request.payload["keep_alive"] == "30m"
    assert "[C1] The agreement term is three years" in request.payload["prompt"]


# --- the verifier's reply must BE a status, not merely contain one ----------

_VERIFIED_CLAIM = "The term is three years [C1]."


def _verifier_result(main_tree, reply):
    namespace, _ = _route_harness(main_tree, _FakeConnection(), reply)
    records = CitationRegistry.from_bundle(_evidence_bundle()).records
    return namespace["_ollama_claim_verifier"](_VERIFIED_CLAIM, records, "local-verifier-model")


def _streamed_claims(main_tree, reply):
    _, events, _ = _stream_ask(main_tree, _FakeConnection(), verifier_reply=reply)
    [verification] = [event for event in events if event["type"] == "verification"]
    assert events[-1] == {"type": "done", "ok": True}, (
        "a verifier verdict never decides whether the answer itself succeeded"
    )
    return verification["claims"]


@pytest.mark.parametrize(
    ("reply", "status"),
    [
        ("supported", "supported"),
        ("partially_supported", "partially_supported"),
        ("unsupported", "unsupported"),
        ("  Supported.\n", "supported"),
        ("**UNSUPPORTED**", "unsupported"),
        ('"partially_supported"', "partially_supported"),
    ],
    ids=repr,
)
def test_an_exact_verifier_status_is_accepted(main_tree, reply, status):
    assert _verifier_result(main_tree, reply) == status
    claims = _streamed_claims(main_tree, reply)
    assert [(claim["status"], claim["reason"]) for claim in claims] == [(status, "verified")]


@pytest.mark.parametrize("reply", ["partially supported", "Partially-Supported."], ids=repr)
def test_partially_supported_spelled_with_a_separator_is_partial_never_supported(
    main_tree, reply
):
    assert _verifier_result(main_tree, reply) == "partially_supported"
    claims = _streamed_claims(main_tree, reply)
    assert [claim["status"] for claim in claims] == ["partially_supported"]


@pytest.mark.parametrize(
    "reply",
    [
        # negative
        "not supported",
        "Not supported.",
        "This claim is not supported by the evidence.",
        # mixed-status prose
        "supported or unsupported",
        "unsupported supported",
        "Supported, but the evidence only partially_supported it.",
        "Supported.\nOn reflection, the evidence does not state this.",
        # malformed
        "supportedly",
        "yes",
        '{"status": "supported"}',
        "",
        None,
    ],
    ids=repr,
)
def test_a_negative_ambiguous_or_malformed_reply_never_yields_a_supported_claim(
    main_tree, reply
):
    assert _verifier_result(main_tree, reply) is None, (
        "only a reply that is exactly one status may be classified"
    )
    claims = _streamed_claims(main_tree, reply)
    assert claims, "the cited claim must still be reported"
    assert all(claim["status"] != "supported" for claim in claims)
    assert [(claim["status"], claim["reason"]) for claim in claims] == [
        ("unverified", "verifier_malformed")
    ]


# --- route composition: priority belongs to the planner --------------------


def test_route_delegates_evidence_priority_to_the_planner(main_tree, main_source):
    ask_question = _find_function(main_tree, "ask_question")
    segment = _segment(main_source, ask_question)

    assert "planner.plan(" in segment, "the route must delegate to the planner"
    for banned in (
        "is_summary_question",
        "get_documents_by_filenames",
        "context_source",
        "build_context",
    ):
        assert banned not in segment, (
            f"the route must not re-decide evidence selection; found {banned!r}"
        )


def test_route_builds_its_prompt_and_registry_from_the_bundle(main_tree, main_source):
    ask_question = _find_function(main_tree, "ask_question")
    segment = _segment(main_source, ask_question)

    assert "CitationRegistry.from_bundle(" in segment
    assert "build_generation_prompt(" in segment
    assert "registry.to_dict()" in segment


def test_planner_is_composed_with_the_authorized_scope(main_tree):
    builder = _find_function(main_tree, "_build_answer_evidence_planner")
    calls = _calls_to_name(builder, "AnswerEvidencePlanner")
    assert len(calls) == 1
    kwargs = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert set(kwargs) == {
        "load_document_text",
        "retrieve",
        "limits",
        "propagate_exceptions",
    }
    assert isinstance(kwargs["retrieve"], ast.Name)
    assert kwargs["retrieve"].id == "_hybrid_retrieve_evidence"


# --- model evidence vs public excerpt, and the verification budget ---------


def test_the_verifier_reads_internal_evidence_text_not_the_public_excerpt(
    main_tree, main_source
):
    """A claim supported past character 480 must still be judgeable.

    The public excerpt is bounded for the UI; judging a claim against it would
    mark anything supported by later text as unsupported, which is exactly what
    a long selection or a complete short document looks like.
    """
    verifier = _find_function(main_tree, "_ollama_claim_verifier")
    segment = _segment(main_source, verifier)

    assert "record.evidence_text" in segment
    assert "record.excerpt" not in segment, (
        "the verifier must not judge claims against the bounded UI excerpt"
    )


def test_the_prompt_is_built_from_the_registry_not_from_excerpts(main_tree, main_source):
    ask_question = _find_function(main_tree, "ask_question")
    segment = _segment(main_source, ask_question)

    assert "build_generation_prompt(" in segment
    assert "excerpt" not in segment, (
        "the route must not reach into excerpts to assemble prompt evidence"
    )


def test_the_evidence_event_carries_no_internal_evidence_text():
    """The wire-level half of the same seam, built from real objects."""
    marker = "ZZ-WIRE-LEAK-MARKER-4d21"
    body = ("clause text. " * 200) + marker
    bundle = EvidenceBundle(
        mode=EvidenceMode.COMPLETE_DOCUMENT,
        items=(
            EvidenceItem(
                text=body,
                source_type="complete_document",
                document_id=1,
                filename="agreement.pdf",
            ),
        ),
    )
    registry = CitationRegistry.from_bundle(bundle)

    line = evidence_event(registry.to_dict()["citations"], mode=bundle.mode.value)
    event = json.loads(line.strip())

    assert marker not in line, (
        "the complete document leaked into the client-facing evidence event"
    )
    assert marker in registry.records[0].evidence_text, "the model evidence is intact"
    for citation in event["citations"]:
        assert "evidence_text" not in citation
        assert len(citation["excerpt"]) <= 480


def test_route_configures_a_verification_budget_of_exactly_two(main_tree):
    """Verification is one serialised local model call per claim, so the route
    caps it. Claims past the cap are reported unverified, never upgraded."""
    assignments = [
        node
        for node in main_tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_CITATION_LIMITS"
    ]
    assert len(assignments) == 1, "expected one module-level _CITATION_LIMITS"

    value = assignments[0].value
    assert isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
    assert value.func.id == "CitationLimits"
    kwargs = {keyword.arg: keyword.value for keyword in value.keywords}
    assert "max_claims_verified" in kwargs, (
        "the route must set its own verification budget explicitly"
    )
    budget = kwargs["max_claims_verified"]
    assert isinstance(budget, ast.Constant) and budget.value == 2, (
        f"the /ask route's verification budget must be exactly 2; got {ast.dump(budget)}"
    )
    assert CitationLimits().max_claims_verified != 2, (
        "the module default is deliberately left alone; only this route is capped"
    )


def test_the_route_limits_object_is_the_one_handed_to_verification(main_tree, main_source):
    ask_question = _find_function(main_tree, "ask_question")
    resolver = _find_function(main_tree, "_resolve_citations")
    for function in (ask_question, resolver):
        segment = _segment(main_source, function)
        assert "_CITATION_LIMITS" in segment, (
            "the configured limits must reach both registration and verification"
        )


# --- authorization wiring ---------------------------------------------------


def test_scope_lookup_fails_closed(main_tree, main_source):
    scope = _find_function(main_tree, "_resolve_authorized_scope")
    segment = _segment(main_source, scope)

    assert segment.count("AuthorizedScope.unresolved()") >= 3, (
        "no connection, a failed lookup, and an unknown token must each fail "
        "closed rather than degrade into a wider scope"
    )
    assert "WHERE user_id IS NULL" in segment
    assert "WHERE user_id = %s" in segment


def test_every_documents_query_in_the_ask_flow_is_owner_scoped(main_tree):
    """Scoped to the /ask evidence path on purpose.

    The admin endpoints read across users deliberately and are gated by
    _require_admin; this contract is about the answer path, where an unscoped
    read would put another user's file into someone's answer.
    """
    checked = 0
    for name in (
        "_resolve_authorized_scope",
        "_load_authorized_document_text",
        "_hybrid_retrieve_evidence",
    ):
        function = _find_function(main_tree, name)
        for node in ast.walk(function):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            normalized = " ".join(node.value.split())
            if "FROM documents" not in normalized and "JOIN documents" not in normalized:
                continue
            checked += 1
            assert "user_id" in normalized or "d.id = ANY(%s)" in normalized, (
                "an unscoped documents read would expose another user's file: "
                f"{normalized[:120]}"
            )
    assert checked >= 3, f"expected several scoped document reads, saw {checked}"


def test_document_text_loading_is_scoped_twice(main_tree, main_source):
    loader = _find_function(main_tree, "_load_authorized_document_text")
    segment = _segment(main_source, loader)

    assert "scope.authorizes(document_id)" in segment, "the in-memory scope is checked first"
    assert "AND user_id IS NULL" in segment and "AND user_id = %s" in segment, (
        "the SELECT must be owner-scoped again, so a caller bug cannot become a "
        "cross-user read"
    )


def test_hybrid_retrieval_uses_only_authorized_builders(main_tree, main_source):
    retrieve = _find_function(main_tree, "_hybrid_retrieve_evidence")
    segment = _segment(main_source, retrieve)

    assert "build_authorized_semantic_query(" in segment
    assert "build_authorized_lexical_query(" in segment
    assert "build_relevant_chunks_query(" not in segment, (
        "the filename-scoped builder must not be used for authorized retrieval"
    )
    assert "reciprocal_rank_fusion(" in segment
    assert "_EMBEDDING_PROVIDER.embed_query(" in segment

    embed_position = segment.find("_EMBEDDING_PROVIDER.embed_query(")
    connect_position = segment.find("connect_to_postgres()")
    assert embed_position < connect_position, (
        "the query embedding must be produced before the database is touched, "
        "so an embedding outage propagates instead of looking like no results"
    )


def test_no_global_retrieval_path_remains_in_the_ask_flow(main_tree, main_source):
    ask_question = _find_function(main_tree, "ask_question")
    segment = _segment(main_source, ask_question)
    assert "get_relevant_chunks(" not in segment, (
        "the unscoped global retrieval helper must not be reachable from /ask"
    )


def test_this_file_contacts_no_external_service():
    this_path = Path(__file__).resolve()
    tree = ast.parse(this_path.read_text(encoding="utf-8"), filename=str(this_path))

    forbidden = {"subprocess", "docker", "requests", "psycopg2", "socket", "httpx"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert imported.isdisjoint(forbidden)
    assert "main" not in imported, "this file must not import the application"


# --- E2b: only the citations the answer used are transmitted ---------------


def test_verification_event_carries_the_used_citation_ids():
    line = verification_event(
        [{"text": "A claim [C3].", "citation_ids": ["C3"], "status": "supported", "reason": "verified"}],
        invalid_citation_ids=["C9"],
        used_citation_ids=["C3", "C1"],
    )
    event = json.loads(line.strip())

    assert event["type"] == "verification"
    assert event["used_citation_ids"] == ["C3", "C1"], (
        "first-use order must survive the wire, so the client can number "
        "sources by use rather than by candidate position"
    )
    assert event["invalid_citation_ids"] == ["C9"]
    assert set(event) == {"type", "claims", "invalid_citation_ids", "used_citation_ids"}


def test_used_citation_ids_default_to_empty_rather_than_absent():
    event = json.loads(verification_event([]).strip())
    assert event["used_citation_ids"] == [], (
        "the key must always be present, so the client never has to guess"
    )


def test_the_route_publishes_the_used_citation_ids(main_tree, main_source):
    ask_question = _find_function(main_tree, "ask_question")
    stream_generate = _find_function(ask_question, "stream_generate")
    calls = _calls_to_name(stream_generate, "verification_event")
    assert len(calls) == 1, "verification is published exactly once"

    kwargs = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert "used_citation_ids" in kwargs
    value = kwargs["used_citation_ids"]
    assert isinstance(value, ast.Attribute) and value.attr == "used_citation_ids"
    assert isinstance(value.value, ast.Name) and value.value.id == "citation_result", (
        "the used ids must come from the parsed answer, not from the candidate "
        "evidence bundle"
    )
    assert "invalid_citation_ids=citation_result.invalid_citation_ids" in _segment(
        main_source, stream_generate
    )


def test_the_evidence_event_is_still_candidates_only(main_tree, main_source):
    """The early evidence event stays candidates; it never claims to be sources."""
    ask_question = _find_function(main_tree, "ask_question")
    stream_generate = _find_function(ask_question, "stream_generate")
    calls = _calls_to_name(stream_generate, "evidence_event")
    assert len(calls) == 1
    assert "used_citation_ids" not in _segment(main_source, stream_generate).split(
        "verification_event"
    )[0], "the evidence event must not pretend to know what the answer will cite"


def test_public_citation_payloads_never_label_a_chunk():
    bundle = EvidenceBundle(
        mode=EvidenceMode.HYBRID_RETRIEVAL,
        items=(
            EvidenceItem(
                text="a retrieved passage",
                source_type="document_chunk",
                document_id=1,
                filename="notes.txt",
                chunk_id="1-4",
                chunk_index=4,
            ),
        ),
    )
    registry = CitationRegistry.from_bundle(bundle)
    line = evidence_event(registry.to_dict()["citations"], mode=bundle.mode.value)
    event = json.loads(line.strip())

    citation = event["citations"][0]
    assert citation["locator"]["label"] == "Relevant passage"
    assert "chunk 4" not in json.dumps(citation)
    assert "chunk" not in citation["display_label"].lower()
