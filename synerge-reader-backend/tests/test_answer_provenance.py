"""Unit contracts for answer_provenance.py.

Pure: no ``main`` import, no database, no Ollama, no network or subprocess.
Snapshots come from the real registry resolver with synthetic digests; model
streams are fed to the real reader as NDJSON text.

What these prove: a record has exactly the versioned schema; every value is
copied from its own trusted server source (expected and observed digests
never stand in for each other); graded and ungraded modes are told apart;
Ollama metrics keep nanoseconds separate from server milliseconds and are
null -- never 0 -- when not reported; timing is monotonic and first-mark-wins;
request-shaped stand-ins, malformed trace ids and unfinished streams are
refused; and the stored JSON is deterministic and never contains thinking
text.
"""

import dataclasses
import json
import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from answer_evidence import EvidenceMode
from answer_provenance import (
    PROVENANCE_SCHEMA_VERSION,
    TIMING_MARKS,
    AnswerProvenanceError,
    AnswerTimer,
    answer_provenance_json,
    build_answer_provenance,
    new_trace_id,
)
from citation_generation import AnswerMode
from generation_profiles import (
    GEMMA_CANDIDATE,
    LLAMA_CONTROL,
    LLAMA_CONTROL_DIGEST,
    ModelRegistry,
    Qualification,
    default_model_registry,
    resolve_answer_generation,
)
from ollama_generation import OllamaGenerateStream
from rag_model_profiles import default_embedding_profile
from release_metadata import release_metadata_from_env

SYNTHETIC_SHA = "0123456789abcdef0123456789abcdef01234567"
SYNTHETIC_DIGEST = "5e" * 32
TRACE_ID = "6f9619ff-8b86-4d11-b42d-00c04fc964ff"


class _Clock:
    def __init__(self, *readings):
        self._readings = list(readings)

    def __call__(self):
        return self._readings.pop(0)


def _stream(*records):
    stream = OllamaGenerateStream()
    stream.feed("".join(json.dumps(record) + "\n" for record in records))
    stream.close()
    return stream


_DONE = {
    "response": "", "done": True, "done_reason": "stop",
    "total_duration": 4_000_000_000, "load_duration": 1_000_000,
    "prompt_eval_count": 900, "prompt_eval_duration": 700_000_000,
    "eval_count": 12, "eval_duration": 300_000_000,
}


def _llama_default():
    return resolve_answer_generation(
        default_model_registry(),
        requested_profile_id=None,
        caller_is_admin=lambda: pytest.fail("no admin lookup for the default"),
        installed_digest=lambda tag: pytest.fail("no inventory lookup for the unqualified control"),
    )


def _qualified_gemma_override(observed=SYNTHETIC_DIGEST):
    base = default_model_registry()
    gemma = dataclasses.replace(
        base.get(GEMMA_CANDIDATE),
        expected_digest=SYNTHETIC_DIGEST,
        qualification=Qualification(
            think_false_verified=True, combined_capacity_verified=True,
            quality_verified=True, evidence_ref="synthetic test identity",
        ),
    )
    registry = ModelRegistry(
        profiles=(base.get(LLAMA_CONTROL), gemma),
        default_profile_id=LLAMA_CONTROL,
        verifier_profile_id=LLAMA_CONTROL,
    )
    return resolve_answer_generation(
        registry,
        requested_profile_id=GEMMA_CANDIDATE,
        caller_is_admin=lambda: True,
        installed_digest=lambda tag: observed,
    )


def _timer():
    # origin 100.0 s; then the four milestones in order.
    timer = AnswerTimer(clock=_Clock(100.0, 100.012, 100.640, 103.020, 103.550))
    for name in TIMING_MARKS:
        timer.mark(name)
    return timer


def _build(**overrides):
    fields = dict(
        trace_id=TRACE_ID,
        release=release_metadata_from_env({"SYNERGE_BUILD_SHA": SYNTHETIC_SHA}),
        generation=_llama_default(),
        answer_mode=AnswerMode.DOCUMENT_QA,
        evidence_mode=EvidenceMode.COMPLETE_DOCUMENT,
        embedding_profile=default_embedding_profile(),
        timer=_timer(),
        stream=_stream({"response": "The term is two years [C1]."}, _DONE),
        verifier_calls=1,
    )
    fields.update(overrides)
    return build_answer_provenance(**fields)


# --- the schema ---------------------------------------------------------------


def test_a_default_llama_record_has_exactly_the_versioned_schema():
    embedding = default_embedding_profile()
    record = _build()

    assert record == {
        "schema_version": 1,
        "trace_id": TRACE_ID,
        "release": {"app_version": "0.1.0", "build_sha": SYNTHETIC_SHA},
        "generation": {
            "profile_id": "llama_control",
            "model_tag": "llama3.1:8b",
            "selection": "default",
            "expected_digest": LLAMA_CONTROL_DIGEST,
            "observed_digest": None,
            "requested_options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1200},
            "think": None,
            "keep_alive": "5m",
            "answer_read_timeout_s": 60.0,
        },
        "prompt": {
            "answer_prompt_version": "rag-answer-v1",
            "answer_mode": "document_qa",
            "claim_support_prompt_version": "claim-support-v1",
        },
        "verification": {
            "profile_id": "llama_control",
            "model_tag": "llama3.1:8b",
            "requested_options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 16},
            "keep_alive": "5m",
            "calls_attempted": 1,
        },
        "retrieval": {
            "evidence_mode": "complete_document",
            "configured_embedding_profile": {
                "profile_id": embedding.profile_id,
                "provider": "ollama",
                "model": "mxbai-embed-large:335m",
                "dimension": 1024,
            },
        },
        "timing": {
            "origin": "ask_request_received",
            "generation_request_sent_ms": 12,
            "first_answer_token_ms": 640,
            "generation_complete_ms": 3020,
            "verification_complete_ms": 3550,
        },
        "ollama_metrics": {
            "done_reason": "stop",
            "prompt_eval_count": 900,
            "eval_count": 12,
            "load_duration_ns": 1_000_000,
            "prompt_eval_duration_ns": 700_000_000,
            "eval_duration_ns": 300_000_000,
            "total_duration_ns": 4_000_000_000,
        },
    }
    assert PROVENANCE_SCHEMA_VERSION == 1


def test_an_unknown_build_is_recorded_as_null_not_guessed():
    record = _build(release=release_metadata_from_env({}))
    assert record["release"] == {"app_version": "0.1.0", "build_sha": None}


# --- generation identity ------------------------------------------------------


def test_the_default_llama_path_has_no_observed_digest():
    record = _build()
    assert record["generation"]["expected_digest"] == LLAMA_CONTROL_DIGEST
    assert record["generation"]["observed_digest"] is None, (
        "no inventory lookup happened, so nothing was observed"
    )


def test_an_admin_override_records_its_selection_and_the_observed_digest():
    record = _build(generation=_qualified_gemma_override(observed="SHA256:" + SYNTHETIC_DIGEST.upper()))

    generation = record["generation"]
    assert generation["profile_id"] == "gemma_candidate"
    assert generation["model_tag"] == "gemma4:31b"
    assert generation["selection"] == "admin_override"
    assert generation["expected_digest"] == SYNTHETIC_DIGEST
    assert generation["observed_digest"] == SYNTHETIC_DIGEST, "the lookup's value, normalized"
    assert generation["think"] is False, "think: false was sent"
    assert generation["answer_read_timeout_s"] == 150.0
    # Verification stays the fixed Llama call whichever profile answered.
    assert record["verification"]["model_tag"] == "llama3.1:8b"


def test_expected_and_observed_digests_are_each_copied_from_their_own_field():
    """Unreachable through selection (a mismatch is refused), so built
    directly: proves the record never fills one digest from the other."""
    snapshot = dataclasses.replace(_llama_default(), expected_digest="aa" * 32, observed_digest="bb" * 32)
    generation = _build(generation=snapshot)["generation"]
    assert (generation["expected_digest"], generation["observed_digest"]) == ("aa" * 32, "bb" * 32)

    no_pin = dataclasses.replace(_llama_default(), expected_digest=None, observed_digest=None)
    generation = _build(generation=no_pin)["generation"]
    assert (generation["expected_digest"], generation["observed_digest"]) == (None, None)


@pytest.mark.parametrize("field", ["expected_digest", "observed_digest"])
def test_a_malformed_digest_is_refused(field):
    snapshot = dataclasses.replace(_llama_default(), **{field: "sha256:" + "AA" * 32})
    with pytest.raises(AnswerProvenanceError):
        _build(generation=snapshot)


# --- graded and ungraded modes ------------------------------------------------


@pytest.mark.parametrize("mode", [AnswerMode.STRUCTURED_JSON, AnswerMode.MODEL_REASONING])
def test_ungraded_modes_record_no_verifier_prompt_or_call(mode):
    record = _build(answer_mode=mode, verifier_calls=0)
    assert record["prompt"] == {
        "answer_prompt_version": "rag-answer-v1",
        "answer_mode": mode.value,
        "claim_support_prompt_version": None,
    }
    assert record["verification"] is None


def test_a_graded_answer_with_no_cited_claims_records_zero_verifier_calls():
    assert _build(verifier_calls=0)["verification"]["calls_attempted"] == 0


@pytest.mark.parametrize("mode", list(EvidenceMode))
def test_the_evidence_mode_is_recorded_and_the_embedding_profile_is_labelled_configured(mode):
    retrieval = _build(evidence_mode=mode)["retrieval"]
    assert retrieval["evidence_mode"] == mode.value
    assert set(retrieval) == {"evidence_mode", "configured_embedding_profile"}, (
        "no field may claim the embedding model actually ran"
    )


# --- metrics and timing -------------------------------------------------------


def test_metrics_ollama_did_not_report_are_null_never_zero():
    # A finished final record (only "stop" finishes a generation) that reports
    # no counters or durations at all.
    stream = _stream({"response": "An answer."}, {"response": "", "done": True, "done_reason": "stop"})
    metrics = _build(stream=stream)["ollama_metrics"]
    assert metrics == {
        "done_reason": "stop",
        "prompt_eval_count": None,
        "eval_count": None,
        "load_duration_ns": None,
        "prompt_eval_duration_ns": None,
        "eval_duration_ns": None,
        "total_duration_ns": None,
    }


def test_a_reported_zero_stays_zero_and_malformed_metrics_become_null():
    stream = _stream(
        {"response": "An answer."},
        {"response": "", "done": True, "done_reason": "stop", "load_duration": 0,
         "eval_count": "12", "prompt_eval_count": True, "total_duration": -5},
    )
    metrics = _build(stream=stream)["ollama_metrics"]
    assert metrics["load_duration_ns"] == 0, "a reported zero is real"
    assert metrics["eval_count"] is None and metrics["prompt_eval_count"] is None
    assert metrics["total_duration_ns"] is None


def test_timing_is_monotonic_milliseconds_and_the_first_mark_wins():
    timer = AnswerTimer(clock=_Clock(10.0, 10.5, 10.75))
    timer.mark("first_answer_token")
    timer.mark("first_answer_token")  # a later token must not move it
    timer.mark("generation_complete")
    assert timer.elapsed_ms() == {
        "origin": "ask_request_received",
        "generation_request_sent_ms": None,
        "first_answer_token_ms": 500,
        "generation_complete_ms": 750,
        "verification_complete_ms": None,
    }


def test_an_unknown_timing_mark_is_refused():
    with pytest.raises(AnswerProvenanceError):
        AnswerTimer().mark("client_reported")


# --- refusals: only trusted server objects ------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        ("release", {"app_version": "9.9.9", "build_sha": "f" * 40}),
        ("generation", {"profile_id": "evil", "model_tag": "evil:latest"}),
        ("answer_mode", "document_qa"),
        ("evidence_mode", "hybrid_retrieval"),
        ("embedding_profile", {"model": "spoofed-embedder"}),
        ("timer", {"first_answer_token_ms": 1}),
        ("stream", {"metrics": {"eval_count": 1}}),
    ],
)
def test_a_request_shaped_stand_in_is_refused(field, value):
    with pytest.raises(AnswerProvenanceError):
        _build(**{field: value})


@pytest.mark.parametrize(
    "trace_id",
    [
        "client-chosen-trace",
        TRACE_ID.upper(),
        "{" + TRACE_ID + "}",
        str(uuid.uuid1()),
        "",
        None,
    ],
    ids=["free-text", "uppercase", "braced", "uuid1", "empty", "none"],
)
def test_a_trace_id_that_is_not_a_canonical_uuid4_is_refused(trace_id):
    with pytest.raises(AnswerProvenanceError):
        _build(trace_id=trace_id)


def test_new_trace_ids_are_canonical_uuid4_and_unique():
    ids = {new_trace_id() for _ in range(200)}
    assert len(ids) == 200
    for value in ids:
        assert str(uuid.UUID(value)) == value and uuid.UUID(value).version == 4


@pytest.mark.parametrize(
    "records",
    [
        [{"response": "Cut off"}],                                          # no final record
        [{"response": "The term"}, dict(_DONE, done_reason="length")],      # output limit
        [{"response": "The term"}, {"error": "runner crashed"}],            # model error
    ],
    ids=["incomplete", "output-limit", "model-error"],
)
def test_an_unfinished_stream_has_no_provenance(records):
    with pytest.raises(AnswerProvenanceError):
        _build(stream=_stream(*records))


@pytest.mark.parametrize(
    "final",
    [
        {k: v for k, v in _DONE.items() if k != "done_reason"},
        dict(_DONE, done_reason=None),
        dict(_DONE, done_reason=7),
        dict(_DONE, done_reason="load"),
        dict(_DONE, done_reason="something_new"),
        dict(_DONE, done="true"),
    ],
    ids=["reason-missing", "reason-null", "reason-non-string", "reason-load", "reason-unknown", "done-not-true"],
)
def test_a_final_record_without_a_stop_completion_has_no_provenance(final):
    with pytest.raises(AnswerProvenanceError):
        _build(stream=_stream({"response": "The term is two years [C1]."}, final))


@pytest.mark.parametrize("calls", [-1, True, 1.0, "1"])
def test_verifier_calls_must_be_a_non_negative_integer(calls):
    with pytest.raises(AnswerProvenanceError):
        _build(verifier_calls=calls)


# --- stored JSON --------------------------------------------------------------


def test_the_stored_json_is_deterministic_and_round_trips():
    record = _build()
    text = answer_provenance_json(record)
    assert text == answer_provenance_json(json.loads(text))
    assert json.loads(text) == record


def test_thinking_text_never_reaches_the_record():
    stream = _stream(
        {"thinking": "SECRET-THOUGHT about the lease", "response": ""},
        {"response": "The term is two years [C1]."},
        _DONE,
    )
    text = answer_provenance_json(_build(stream=stream))
    assert "SECRET-THOUGHT" not in text and "thinking" not in text


@pytest.mark.parametrize("value", [{"schema_version": 2}, [], "text", None])
def test_only_a_provenance_record_is_serialized(value):
    with pytest.raises(AnswerProvenanceError):
        answer_provenance_json(value)


def test_non_finite_numbers_are_refused_rather_than_stored():
    record = _build()
    record["timing"]["first_answer_token_ms"] = float("nan")
    with pytest.raises(ValueError):
        answer_provenance_json(record)
