"""The /api/generate completion rules shared by /ask, the claim verifier, the
background helpers and the diagnostic harness: the NDJSON stream reader and
the non-streamed reply validator.

Pure, in memory: every stream and body below is a hand-built payload.
"""

import ast
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ollama_generation import (
    FAILURE_INCOMPLETE,
    FAILURE_MALFORMED,
    FAILURE_MODEL_ERROR,
    FAILURE_OUTPUT_LIMIT,
    MODEL_ERROR_MESSAGE,
    OllamaGenerateStream,
    completed_generate_response,
)

_FINAL = {
    "response": "", "done": True, "done_reason": "stop",
    "total_duration": 900, "load_duration": 100, "prompt_eval_count": 1500,
    "prompt_eval_duration": 300, "eval_count": 42, "eval_duration": 500,
}


def _ndjson(*records):
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def _read(payload, size=None):
    reader = OllamaGenerateStream()
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    size = size or len(data) or 1
    tokens = []
    for index in range(0, len(data), size):
        tokens.extend(reader.feed(data[index:index + size]))
    tokens.extend(reader.close())
    return reader, "".join(tokens)


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 32, 4096])
def test_split_lines_and_split_multibyte_characters_reassemble_exactly(size):
    answer = ["The term ", "is “three” years — renewable ", "[C1]. 期間は3年。"]
    payload = _ndjson(*({"response": token} for token in answer), _FINAL)

    reader, text = _read(payload, size)

    assert text == "".join(answer)
    assert reader.failure is None
    assert reader.done_reason == "stop"
    assert reader.metrics == {k: v for k, v in _FINAL.items() if k.endswith(("duration", "count"))}


def test_text_chunks_are_accepted_as_well_as_bytes():
    reader = OllamaGenerateStream()
    payload = _ndjson({"response": "ok"}, _FINAL)
    tokens = reader.feed(payload[:5]) + reader.feed(payload[5:]) + reader.close()
    assert tokens == ["ok"] and reader.failure is None


def test_thinking_never_becomes_answer_text():
    payload = _ndjson(
        {"thinking": "The user wants [C1] cited. ", "response": ""},
        {"thinking": "Answer: eight.", "response": ""},
        {"response": "The cap is 8 [C1]."},
        _FINAL,
    )
    reader, text = _read(payload, 3)
    assert text == "The cap is 8 [C1]."
    assert reader.thinking_chars == len("The user wants [C1] cited. ") + len("Answer: eight.")
    assert reader.failure is None


def test_a_thinking_only_reply_has_no_answer_text():
    reader, text = _read(_ndjson({"thinking": "hmm", "response": ""}, _FINAL))
    assert text == "" and reader.thinking_chars == 3
    assert reader.failure is None, "the caller decides that an empty answer is not a success"


def test_an_empty_reply_has_no_answer_text():
    reader, text = _read(_ndjson(_FINAL))
    assert text == "" and reader.failure is None


def test_document_text_that_looks_like_markup_is_not_stripped():
    literal = "Section <think>2</think> of the contract says so [C1]."
    reader, text = _read(_ndjson({"response": literal}, _FINAL))
    assert text == literal


def test_an_error_line_fails_the_stream_and_keeps_no_server_text():
    payload = _ndjson({"response": "Partial "}, {"error": "model runner crashed at /srv/x"}, {"response": "more"})
    reader, text = _read(payload)
    assert text == "Partial "
    assert reader.failure == FAILURE_MODEL_ERROR
    assert "crashed" not in json.dumps(vars(reader), default=str)


@pytest.mark.parametrize(
    "bad_line",
    ['{"response": "trunc', "not json", "[1, 2]", '{"response": 7}'],
)
def test_a_malformed_line_fails_the_stream_and_stops_further_text(bad_line):
    payload = _ndjson({"response": "Before "}) + bad_line + "\n" + _ndjson({"response": "after"}, _FINAL)
    reader, text = _read(payload, 4)
    assert text == "Before "
    assert reader.failure == FAILURE_MALFORMED


def test_invalid_utf8_is_malformed_not_replaced():
    payload = _ndjson({"response": "ok "}).encode() + b'{"response": "\xff"}\n' + _ndjson(_FINAL).encode()
    reader, text = _read(payload)
    assert reader.failure == FAILURE_MALFORMED
    assert "�" not in text


def test_a_stream_without_its_final_record_is_incomplete():
    reader, text = _read(_ndjson({"response": "The term is "}, {"response": "three"}))
    assert text == "The term is three"
    assert reader.failure == FAILURE_INCOMPLETE


def test_a_stream_cut_mid_line_is_incomplete_or_malformed_never_finished():
    full = _ndjson({"response": "The term is three years."}, _FINAL)
    reader, _ = _read(full[: len(full) - 10])
    assert reader.failure in (FAILURE_INCOMPLETE, FAILURE_MALFORMED)


def test_the_output_limit_is_reported():
    final = dict(_FINAL, done_reason="length", eval_count=1200)
    reader, text = _read(_ndjson({"response": "A long answer that was cut"}, final))
    assert text == "A long answer that was cut"
    assert reader.failure == FAILURE_OUTPUT_LIMIT
    assert reader.metrics["eval_count"] == 1200


def test_a_final_line_without_a_newline_is_still_read():
    payload = _ndjson({"response": "x"}) + json.dumps(_FINAL)
    reader, text = _read(payload)
    assert text == "x" and reader.done and reader.failure is None


def test_nothing_may_follow_the_final_record():
    reader, text = _read(_ndjson({"response": "x"}, _FINAL, {"response": "late"}))
    assert text == "x"
    assert reader.failure == FAILURE_MALFORMED


def test_blank_keepalive_lines_are_ignored():
    reader, text = _read("\n\n" + _ndjson({"response": "x"}) + "\r\n" + _ndjson(_FINAL))
    assert text == "x" and reader.failure is None


# --- a finished stream says "stop" ------------------------------------------

_NOT_STOP_FINALS = {
    "reason-missing": {k: v for k, v in _FINAL.items() if k != "done_reason"},
    "reason-null": dict(_FINAL, done_reason=None),
    "reason-number": dict(_FINAL, done_reason=7),
    "reason-list": dict(_FINAL, done_reason=["stop"]),
    "reason-bool": dict(_FINAL, done_reason=True),
    "reason-load": dict(_FINAL, done_reason="load"),
    "reason-unload": dict(_FINAL, done_reason="unload"),
    "reason-unknown": dict(_FINAL, done_reason="something_new"),
    "reason-empty": dict(_FINAL, done_reason=""),
    "reason-uppercase": dict(_FINAL, done_reason="STOP"),
    "reason-padded": dict(_FINAL, done_reason=" stop"),
}


@pytest.mark.parametrize("case", sorted(_NOT_STOP_FINALS))
def test_a_final_record_without_a_stop_reason_is_incomplete(case):
    reader, text = _read(_ndjson({"response": "The term is three years [C1]."}, _NOT_STOP_FINALS[case]), 5)
    assert reader.done is True, "the final record did arrive"
    assert text == "The term is three years [C1].", "streamed text is still reported to the caller"
    assert reader.failure == FAILURE_INCOMPLETE, "but it is never a finished generation"


def test_a_non_string_reason_is_not_kept_as_the_reported_reason():
    reader, _ = _read(_ndjson({"response": "x"}, dict(_FINAL, done_reason=7)))
    assert reader.done_reason is None and reader.failure == FAILURE_INCOMPLETE


@pytest.mark.parametrize(
    "done",
    [False, "true", 1, None],
    ids=["false", "string-true", "one", "null"],
)
def test_only_done_true_is_a_final_record(done):
    reader, text = _read(_ndjson({"response": "x"}, dict(_FINAL, done=done)))
    assert text == "x"
    assert reader.done is False
    assert reader.failure == FAILURE_INCOMPLETE


def test_the_output_limit_keeps_its_own_classification():
    reader, _ = _read(_ndjson({"response": "cut"}, dict(_FINAL, done_reason="length")))
    assert reader.failure == FAILURE_OUTPUT_LIMIT


# --- completed_generate_response: one non-streamed reply --------------------

_STOP = {"response": "supported", "done": True, "done_reason": "stop"}


def test_a_finished_reply_returns_its_original_response_text():
    body = dict(_STOP, response="  Supported.\n", eval_count=3)
    assert completed_generate_response(body) == "  Supported.\n", "returned unchanged, untrimmed"


def test_separate_thinking_is_ignored():
    body = dict(_STOP, thinking="SECRET-THOUGHT: unsupported", response="supported")
    assert completed_generate_response(body) == "supported"


def test_thinking_alone_is_not_a_reply():
    assert completed_generate_response(dict(_STOP, thinking="supported", response="")) is None


@pytest.mark.parametrize(
    "done",
    ["missing", False, "true", 1, None],
    ids=["missing", "false", "string-true", "one", "null"],
)
def test_a_reply_that_is_not_done_is_none(done):
    body = dict(_STOP)
    if done == "missing":
        del body["done"]
    else:
        body["done"] = done
    assert completed_generate_response(body) is None


@pytest.mark.parametrize(
    "reason",
    ["missing", None, 7, True, ["stop"], "load", "unload", "length", "something_new", "", "STOP", "stop "],
    ids=repr,
)
def test_a_reply_without_a_stop_reason_is_none(reason):
    body = dict(_STOP)
    if reason == "missing":
        del body["done_reason"]
    else:
        body["done_reason"] = reason
    assert completed_generate_response(body) is None


@pytest.mark.parametrize(
    "response",
    ["missing", "", "   ", "\n\t\r ", None, 7, ["supported"], {"text": "supported"}],
    ids=repr,
)
def test_a_blank_or_non_string_response_is_none(response):
    body = dict(_STOP)
    if response == "missing":
        del body["response"]
    else:
        body["response"] = response
    assert completed_generate_response(body) is None


@pytest.mark.parametrize(
    "body",
    [None, [], ["supported"], "supported", 42, b'{"response": "supported"}'],
    ids=repr,
)
def test_a_body_that_is_not_an_object_is_none(body):
    assert completed_generate_response(body) is None


@pytest.mark.parametrize(
    "body",
    [
        {"error": "SECRET-SERVER-ERROR at /srv/models/blob"},
        {"error": "SECRET-SERVER-ERROR", "response": "supported", "done": True, "done_reason": "stop"},
        {"error": None},
        {"error": "SECRET-SERVER-ERROR", "done": False},
    ],
    ids=["error-only", "error-with-a-finished-reply", "error-null", "error-not-done"],
)
def test_an_error_body_raises_fixed_text_before_completion_is_checked(body):
    with pytest.raises(RuntimeError) as excinfo:
        completed_generate_response(body)
    assert str(excinfo.value) == MODEL_ERROR_MESSAGE
    assert "SECRET-SERVER-ERROR" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None and excinfo.value.__context__ is None


def test_the_body_is_not_modified():
    body = dict(_STOP, thinking="hmm")
    before = json.dumps(body, sort_keys=True)
    completed_generate_response(body)
    assert json.dumps(body, sort_keys=True) == before


def test_the_completion_module_imports_nothing_that_performs_io():
    source = (Path(__file__).resolve().parents[1] / "ollama_generation.py").read_text(encoding="utf-8")
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots == {"__future__", "codecs", "json", "typing"}
