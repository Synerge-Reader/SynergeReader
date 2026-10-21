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
    completion_diagnostic,
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


# --- completion_diagnostic: counts and fixed words, never text ---------------------

_SECRET = "SECRET-PROMPT-OR-REASONING-7f3a"


def _result(line):
    return dict(field.split("=", 1) for field in line.split()[2:])["result"]


def test_a_reply_cut_off_by_its_allowance_is_described_without_its_text():
    """The shape of Nemotron's verifier reply in the candidate check, inferred
    (its body was not kept): the whole allowance spent, no final reply."""
    body = {"thinking": _SECRET * 10, "response": "", "done": True, "done_reason": "length",
            "prompt_eval_count": 1259, "eval_count": 512}
    line = completion_diagnostic("verifier", "nemotron35_lightning_30b_experimental", 512, body)
    assert line == (
        "[Generation] completion role=verifier profile=nemotron35_lightning_30b_experimental "
        "result=output_limit done_reason=length prompt_eval_count=1259 eval_count=512 num_predict=512 "
        f"at_output_limit=yes thinking_chars={len(_SECRET) * 10} response_chars=0"
    )


def test_a_finished_reply_is_described_by_its_counts():
    body = {"thinking": "abc", "response": "supported", "done": True, "done_reason": "stop",
            "prompt_eval_count": 1200, "eval_count": 300}
    assert completion_diagnostic("insights", "qwen38_27b_experimental", 4096, body) == (
        "[Generation] completion role=insights profile=qwen38_27b_experimental result=completed "
        "done_reason=stop prompt_eval_count=1200 eval_count=300 num_predict=4096 at_output_limit=no "
        "thinking_chars=3 response_chars=9"
    )


_DIAGNOSED_BODIES = {
    "finished": (_STOP, "completed"),
    "thinking-and-reply": (dict(_STOP, thinking="x"), "completed"),
    "output-limit": ({"response": "supp", "done": True, "done_reason": "length"}, "output_limit"),
    "load": ({"response": "supported", "done": True, "done_reason": "load"}, "unfinished_reason"),
    "unknown-reason": ({"response": "supported", "done": True, "done_reason": _SECRET}, "unfinished_reason"),
    "reason-missing": ({"response": "supported", "done": True}, "unfinished_reason"),
    "reason-non-string": ({"response": "supported", "done": True, "done_reason": 7}, "unfinished_reason"),
    "not-done": ({"response": "supported", "done": False, "done_reason": "stop"}, "not_done"),
    "done-string": ({"response": "supported", "done": "true", "done_reason": "stop"}, "not_done"),
    "blank": ({"response": "  ", "done": True, "done_reason": "stop"}, "empty_response"),
    "thinking-only": ({"thinking": _SECRET, "response": "", "done": True, "done_reason": "stop"}, "empty_response"),
    "non-string-response": ({"response": 5, "done": True, "done_reason": "stop"}, "empty_response"),
    "error": ({"error": _SECRET, "response": "supported", "done": True, "done_reason": "stop"}, "model_error"),
    "not-an-object": ([_SECRET], "not_an_object"),
    "none": (None, "not_an_object"),
}


@pytest.mark.parametrize("case", sorted(_DIAGNOSED_BODIES))
def test_the_diagnostic_result_agrees_with_the_completion_rule(case):
    body, expected = _DIAGNOSED_BODIES[case]
    line = completion_diagnostic("qa_pairs", "llama_control", 1200, body)
    assert _result(line) == expected
    if expected == "model_error":
        with pytest.raises(RuntimeError):
            completed_generate_response(body)
    else:
        assert (completed_generate_response(body) is not None) == (expected == "completed")
    assert _SECRET not in line, "no text from the body reaches the line"


@pytest.mark.parametrize(
    "reason, logged",
    [("stop", "stop"), ("length", "length"), ("load", "load"), ("unload", "unload"),
     (_SECRET, "other"), ("STOP", "other"), (7, "missing"), (None, "missing")],
    ids=repr,
)
def test_only_known_completion_reasons_are_logged(reason, logged):
    body = {"response": "", "done": True}
    if reason is not None:
        body["done_reason"] = reason
    line = completion_diagnostic("verifier", "llama_control", 16, body)
    assert f" done_reason={logged} " in line and _SECRET not in line


@pytest.mark.parametrize(
    "eval_count, at_limit, shown",
    [(2048, "yes", "2048"), (4096, "yes", "4096"), (2047, "no", "2047"), (0, "no", "0"),
     (None, "unknown", "unknown"), (True, "unknown", "unknown"), (-1, "unknown", "unknown"),
     ("2048", "unknown", "unknown"), (2048.0, "unknown", "unknown")],
    ids=repr,
)
def test_the_output_limit_flag_compares_integer_counts_only(eval_count, at_limit, shown):
    body = {"response": "", "done": True, "done_reason": "length", "prompt_eval_count": "many"}
    if eval_count is not None:
        body["eval_count"] = eval_count
    line = completion_diagnostic("insights", "llama_control", 2048, body)
    assert f" eval_count={shown} " in line and f" at_output_limit={at_limit} " in line
    assert " prompt_eval_count=unknown " in line


def test_absent_or_non_string_text_fields_are_reported_as_absent():
    line = completion_diagnostic("verifier", "llama_control", 16, {"done": True, "thinking": ["x"]})
    assert line.endswith(" thinking_chars=absent response_chars=absent")


def test_the_diagnostic_does_not_modify_the_body():
    body = dict(_STOP, thinking="hmm", eval_count=3)
    before = json.dumps(body, sort_keys=True)
    completion_diagnostic("verifier", "llama_control", 16, body)
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
