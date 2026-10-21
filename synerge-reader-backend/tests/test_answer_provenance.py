"""Unit contracts for answer_provenance.py.

Pure: no ``main`` import, no database, no Ollama, no network or subprocess.
Snapshots come from the real shared-selection functions with synthetic
digests; model streams are fed to the real reader as NDJSON text.

What these prove: a new record has exactly the version 3 schema and records
the shared selection and its epoch -- and nothing else may be written as a new
record; stored version 1 and 2 records stay readable with their own
vocabulary and are never modified, while mixed or unknown vocabularies are
refused; every value is
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
    SELECTIONS_BY_SCHEMA_VERSION,
    TIMING_MARKS,
    AnswerProvenanceError,
    AnswerTimer,
    answer_provenance_json,
    build_answer_provenance,
    new_trace_id,
    read_answer_provenance,
)
from citation_generation import AnswerMode
from generation_profiles import (
    GEMMA_CANDIDATE,
    LLAMA_CONTROL,
    LLAMA_CONTROL_DIGEST,
    GenerationProfileError,
    ModelRegistry,
    Qualification,
    _snapshot,
    check_installed_identity,
    default_model_registry,
    shared_generation_snapshot,
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


def _llama_default(epoch=1):
    """The control as the shared selection at ``epoch``: no inventory lookup."""
    return shared_generation_snapshot(default_model_registry().get(LLAMA_CONTROL), epoch=epoch, observed_digest=None)


def _qualified_gemma_shared(observed=SYNTHETIC_DIGEST, epoch=3):
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
    )
    profile = registry.get(GEMMA_CANDIDATE)
    return shared_generation_snapshot(
        profile, epoch=epoch, observed_digest=check_installed_identity(profile, lambda tag: observed),
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


def test_a_shared_llama_record_has_exactly_the_versioned_schema():
    embedding = default_embedding_profile()
    record = _build()

    assert record == {
        "schema_version": 4,
        "trace_id": TRACE_ID,
        "release": {"app_version": "0.1.0", "build_sha": SYNTHETIC_SHA},
        "generation": {
            "profile_id": "llama_control",
            "model_tag": "llama3.1:8b",
            "selection": "shared_active",
            "selection_epoch": 1,
            "expected_digest": LLAMA_CONTROL_DIGEST,
            "observed_digest": None,
            "requested_options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1200},
            "think": None,
            "keep_alive": "5m",
            "answer_read_timeout_s": 60.0,
            "runtime_key": "primary",
            "experimental": False,
        },
        "prompt": {
            "answer_prompt_version": "rag-answer-v1",
            "answer_mode": "document_qa",
            "claim_support_prompt_version": "claim-support-v1",
        },
        "verification": {
            "profile_id": "llama_control",
            "model_tag": "llama3.1:8b",
            "expected_digest": LLAMA_CONTROL_DIGEST,
            "observed_digest": None,
            "requested_options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 16},
            "think": None,
            "keep_alive": "5m",
            "read_timeout_s": 30.0,
            "runtime_key": "primary",
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
    assert PROVENANCE_SCHEMA_VERSION == 4


def test_an_unknown_build_is_recorded_as_null_not_guessed():
    record = _build(release=release_metadata_from_env({}))
    assert record["release"] == {"app_version": "0.1.0", "build_sha": None}


# --- generation identity ------------------------------------------------------


def test_the_shared_llama_path_has_no_observed_digest():
    record = _build()
    assert record["generation"]["expected_digest"] == LLAMA_CONTROL_DIGEST
    assert record["generation"]["observed_digest"] is None, (
        "no inventory lookup happened, so nothing was observed"
    )


def test_a_shared_gemma_selection_records_its_epoch_and_the_observed_digest():
    record = _build(generation=_qualified_gemma_shared(observed="SHA256:" + SYNTHETIC_DIGEST.upper()))

    generation = record["generation"]
    assert generation["profile_id"] == "gemma_candidate"
    assert generation["model_tag"] == "gemma4:31b"
    assert (generation["selection"], generation["selection_epoch"]) == ("shared_active", 3)
    assert generation["expected_digest"] == SYNTHETIC_DIGEST
    assert generation["observed_digest"] == SYNTHETIC_DIGEST, "the lookup's value, normalized"
    assert generation["think"] is False, "think: false was sent"
    assert generation["answer_read_timeout_s"] == 150.0
    # The answer was verified by the model that wrote it, with the same
    # identity: the pin, and the observation made when the answer resolved.
    assert record["verification"] == {
        "profile_id": "gemma_candidate",
        "model_tag": "gemma4:31b",
        "expected_digest": SYNTHETIC_DIGEST,
        "observed_digest": SYNTHETIC_DIGEST,
        "requested_options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 16},
        "think": False,
        "keep_alive": "5m",
        "read_timeout_s": 150.0,
        "runtime_key": "primary",
        "calls_attempted": 1,
    }


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


# --- verification identity ------------------------------------------------------


def test_verification_is_copied_only_from_the_captured_verifier_call():
    """Built directly with values selection could not produce, so a record
    that took any of them from the answer fields instead would show it."""
    snapshot = _llama_default()
    verifier = dataclasses.replace(
        snapshot.verifier, expected_digest="aa" * 32, observed_digest="bb" * 32, read_timeout_s=42.5,
    )
    record = _build(generation=dataclasses.replace(snapshot, verifier=verifier))

    verification = record["verification"]
    assert (verification["expected_digest"], verification["observed_digest"]) == ("aa" * 32, "bb" * 32)
    assert verification["read_timeout_s"] == 42.5
    assert (record["generation"]["expected_digest"], record["generation"]["observed_digest"]) == (
        LLAMA_CONTROL_DIGEST, None,
    ), "the answer's own identity is recorded separately and unchanged"


def test_the_verifier_think_policy_is_recorded_as_sent():
    assert _build()["verification"]["think"] is None, "None: no think field was sent"
    assert _build(generation=_qualified_gemma_shared())["verification"]["think"] is False


def test_a_stand_in_verifier_is_refused():
    # The snapshot itself refuses a verifier that is not its own model call.
    with pytest.raises(GenerationProfileError):
        dataclasses.replace(_llama_default(), verifier={"model_tag": "spoofed:latest", "read_timeout_s": 1})
    # And no record is built from one smuggled into a frozen snapshot.
    snapshot = _llama_default()
    object.__setattr__(snapshot, "verifier", {"model_tag": "spoofed:latest", "read_timeout_s": 1})
    with pytest.raises(AnswerProvenanceError):
        _build(generation=snapshot)


def test_a_verifier_on_another_runtime_is_refused():
    snapshot = _llama_default()
    object.__setattr__(snapshot, "verifier", dataclasses.replace(snapshot.verifier, runtime_key="compat_cuda121"))
    with pytest.raises(AnswerProvenanceError):
        _build(generation=snapshot)


def test_the_verification_and_selection_fields_survive_json_storage():
    record = _build(generation=_qualified_gemma_shared())
    text = answer_provenance_json(record)
    stored = json.loads(text)
    assert stored["schema_version"] == 4
    assert (stored["generation"]["selection"], stored["generation"]["selection_epoch"]) == ("shared_active", 3)
    assert stored["verification"] == record["verification"]
    assert stored["verification"]["think"] is False, "false is stored, not dropped"
    assert stored["verification"]["read_timeout_s"] == 150.0
    assert "://" not in text, "no endpoint is recorded"


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


# A version 1 or 2 dict is not what this writer produces: stored rows are
# returned as they are and never re-serialized (see the admin-history tests).
@pytest.mark.parametrize("value", [{"schema_version": 1}, {"schema_version": 2}, [], "text", None])
def test_only_a_provenance_record_is_serialized(value):
    with pytest.raises(AnswerProvenanceError):
        answer_provenance_json(value)


def test_non_finite_numbers_are_refused_rather_than_stored():
    record = _build()
    record["timing"]["first_answer_token_ms"] = float("nan")
    with pytest.raises(ValueError):
        answer_provenance_json(record)


# --- a new record describes only the shared selection -----------------------------


@pytest.mark.parametrize("selection", ["default", "admin_override"])
def test_a_historical_selection_is_never_written_as_a_new_record(selection):
    historical = _snapshot(default_model_registry().get(LLAMA_CONTROL), selection, None)
    with pytest.raises(AnswerProvenanceError):
        _build(generation=historical)


# --- reading stored records with their own vocabulary -------------------------------


def _stored_v1():
    return {
        "schema_version": 1, "trace_id": TRACE_ID,
        "generation": {"profile_id": "llama_control", "model_tag": "llama3.1:8b", "selection": "default",
                       "expected_digest": LLAMA_CONTROL_DIGEST, "observed_digest": None},
        "verification": {"profile_id": "llama_control", "model_tag": "llama3.1:8b", "calls_attempted": 1},
    }


def _stored_v2():
    record = _stored_v1()
    record["schema_version"] = 2
    record["generation"].update(selection="admin_override", profile_id="gemma_candidate", model_tag="gemma4:31b",
                                expected_digest=SYNTHETIC_DIGEST, observed_digest=SYNTHETIC_DIGEST)
    return record


def test_the_reader_vocabulary_is_explicit_per_schema_version():
    assert SELECTIONS_BY_SCHEMA_VERSION == {
        1: ("default", "admin_override"), 2: ("default", "admin_override"), 3: ("shared_active",),
        4: ("shared_active",),
    }


def test_stored_v1_and_v2_records_read_with_their_own_vocabulary_and_stay_unchanged():
    for stored, expected in (
        (_stored_v1(), {"schema_version": 1, "selection": "default", "selection_epoch": None,
                        "profile_id": "llama_control", "model_tag": "llama3.1:8b",
                        "expected_digest": LLAMA_CONTROL_DIGEST, "observed_digest": None,
                        "runtime_key": None}),
        (_stored_v2(), {"schema_version": 2, "selection": "admin_override", "selection_epoch": None,
                        "profile_id": "gemma_candidate", "model_tag": "gemma4:31b",
                        "expected_digest": SYNTHETIC_DIGEST, "observed_digest": SYNTHETIC_DIGEST,
                        "runtime_key": None}),
    ):
        before = json.dumps(stored, sort_keys=True)
        assert read_answer_provenance(stored) == expected
        assert json.dumps(stored, sort_keys=True) == before, "reading never rewrites or fills in a record"


def _stored_v3(epoch=5):
    """A version 3 record as the previous release stored it: no runtime key."""
    stored = json.loads(answer_provenance_json(_build(generation=_qualified_gemma_shared(epoch=epoch))))
    stored["schema_version"] = 3
    del stored["generation"]["runtime_key"]
    del stored["generation"]["experimental"]
    del stored["verification"]["runtime_key"]
    return stored


def test_a_v3_record_reads_back_its_shared_selection_and_epoch():
    stored = _stored_v3(epoch=5)
    before = json.dumps(stored, sort_keys=True)
    view = read_answer_provenance(stored)
    assert (view["schema_version"], view["selection"], view["selection_epoch"]) == (3, "shared_active", 5)
    assert (view["expected_digest"], view["observed_digest"]) == (SYNTHETIC_DIGEST, SYNTHETIC_DIGEST)
    assert view["runtime_key"] is None, "a version 3 record predates runtime keys: not recorded, never guessed"
    assert json.dumps(stored, sort_keys=True) == before, "reading never rewrites or fills in a record"


def test_a_v4_record_reads_back_its_runtime_key():
    stored = json.loads(answer_provenance_json(_build(generation=_qualified_gemma_shared(epoch=5))))
    view = read_answer_provenance(stored)
    assert (view["schema_version"], view["selection_epoch"], view["runtime_key"]) == (4, 5, "primary")


@pytest.mark.parametrize(
    "change",
    ["v1-with-shared", "v2-with-epoch", "v3-with-default", "v3-without-epoch", "v3-bool-epoch", "v4", "v0",
     "string-version", "no-generation", "bad-digest", "v3-with-runtime", "v4-without-runtime",
     "v4-unknown-runtime", "v5"],
)
def test_a_mixed_or_unknown_vocabulary_is_refused_not_guessed(change):
    if change.startswith("v3"):
        stored = _stored_v3()
        if change == "v3-with-runtime":
            stored["generation"]["runtime_key"] = "primary"
    elif change in ("v4-without-runtime", "v4-unknown-runtime", "v5"):
        stored = json.loads(answer_provenance_json(_build()))
    else:
        stored = _stored_v2() if change == "v2-with-epoch" else _stored_v1()
    generation = stored["generation"]
    if change == "v1-with-shared":
        generation["selection"] = "shared_active"
    elif change == "v2-with-epoch":
        generation["selection_epoch"] = 2
    elif change == "v3-with-default":
        generation["selection"] = "default"
    elif change == "v3-without-epoch":
        del generation["selection_epoch"]
    elif change == "v3-bool-epoch":
        generation["selection_epoch"] = True
    elif change == "v4":
        stored["schema_version"] = 4
    elif change == "v4-without-runtime":
        del generation["runtime_key"]
    elif change == "v4-unknown-runtime":
        generation["runtime_key"] = "http://ollama.invalid:11434"
    elif change == "v5":
        stored["schema_version"] = 5
    elif change == "v3-with-runtime":
        pass
    elif change == "v0":
        stored["schema_version"] = 0
    elif change == "string-version":
        stored["schema_version"] = "1"
    elif change == "no-generation":
        del stored["generation"]
    else:
        generation["expected_digest"] = "SHA256:" + "AA" * 32
    with pytest.raises(AnswerProvenanceError):
        read_answer_provenance(stored)
