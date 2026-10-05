"""The /api/generate NDJSON reader shared by /ask and the diagnostic harness.

Pure, in memory: every stream below is a hand-built byte or text payload.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ollama_generation import (
    FAILURE_INCOMPLETE,
    FAILURE_MALFORMED,
    FAILURE_MODEL_ERROR,
    FAILURE_OUTPUT_LIMIT,
    OllamaGenerateStream,
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
