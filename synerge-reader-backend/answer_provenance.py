"""Server-created provenance for one successfully saved answer.

Pure value construction, like ``generation_profiles.py``: no network,
database, filesystem, or environment access. ``main.py`` builds one record per
answer, immediately before the chat-history INSERT, and stores it in the same
row (``chat_history.answer_provenance``) so it commits or rolls back with the
answer itself.

Where each value comes from -- never from the request:

* identity: a trace id this module generates;
* release: the server's ``ReleaseMetadata`` captured when /ask received the
  request (a missing build commit stays ``null``, never a guess);
* generation: the ``AnswerGenerationSnapshot`` fixed before anything streamed
  -- profile, exact tag, the registry's expected digest and, separately, the
  digest the installed-model lookup observed for THIS answer (``null`` when no
  lookup was made), and the settings this backend requested;
* prompt: the template versions in ``release_metadata.py`` and the validated
  answer mode the server answered under. The raw prompt is never stored;
* retrieval: the evidence mode the planner reported, and the CONFIGURED
  embedding profile. Whether that model actually ran for this answer is not
  recorded: an empty retrieval is reported as mode ``none`` whether or not the
  query was embedded, so actual use cannot be derived reliably;
* timing: monotonic milliseconds measured by this server, from when /ask
  received the request;
* ollama_metrics: counters and nanosecond durations Ollama reported in its
  final stream record. A value Ollama did not report is ``null``, never 0.

Thinking text, prompt text, evidence text, credentials, endpoints, and paths
are never part of a record: no input here carries them.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Callable, Optional

from answer_evidence import EvidenceMode
from citation_generation import AnswerMode
from generation_profiles import AnswerGenerationSnapshot, normalize_digest
from ollama_generation import OllamaGenerateStream
from rag_model_profiles import EmbeddingProfile
from release_metadata import ReleaseMetadata

PROVENANCE_SCHEMA_VERSION = 1

TIMING_ORIGIN = "ask_request_received"
TIMING_MARKS = (
    "generation_request_sent",
    "first_answer_token",
    "generation_complete",
    "verification_complete",
)

_OLLAMA_COUNTS = ("prompt_eval_count", "eval_count")
# Ollama reports these in nanoseconds; stored with an explicit ``_ns`` suffix
# so they can never be confused with this server's millisecond timings.
_OLLAMA_DURATIONS = ("load_duration", "prompt_eval_duration", "eval_duration", "total_duration")

_SELECTIONS = ("default", "admin_override")


class AnswerProvenanceError(ValueError):
    """Provenance could not be built from trusted server values."""


def new_trace_id() -> str:
    """A fresh server-generated trace id (canonical UUID4 text)."""
    return str(uuid.uuid4())


def _require_trace_id(value: object) -> str:
    try:
        parsed = uuid.UUID(value) if isinstance(value, str) else None
    except ValueError:
        parsed = None
    if parsed is None or parsed.version != 4 or str(parsed) != value:
        raise AnswerProvenanceError("trace_id must be a server-generated UUID4")
    return value


def _require_digest(value: object) -> Optional[str]:
    if value is not None and normalize_digest(value) != value:
        raise AnswerProvenanceError("a model digest must be None or 64 lowercase hex characters")
    return value


def _non_negative_int(value: object) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


class AnswerTimer:
    """Monotonic milestones for one answer.

    The origin is construction time, which /ask does first. Only the first
    mark of each milestone counts, so marking every streamed token records
    the first one.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._origin = clock()
        self._marks: dict = {}

    def mark(self, name: str) -> None:
        if name not in TIMING_MARKS:
            raise AnswerProvenanceError(f"unknown timing mark {name!r}")
        if name not in self._marks:
            self._marks[name] = self._clock()

    def elapsed_ms(self) -> dict:
        timing = {"origin": TIMING_ORIGIN}
        for name in TIMING_MARKS:
            at = self._marks.get(name)
            timing[f"{name}_ms"] = None if at is None else max(0, round((at - self._origin) * 1000))
        return timing


def _ollama_metrics(stream: OllamaGenerateStream) -> dict:
    reported = stream.metrics
    metrics = {"done_reason": stream.done_reason if isinstance(stream.done_reason, str) else None}
    for name in _OLLAMA_COUNTS:
        metrics[name] = _non_negative_int(reported.get(name))
    for name in _OLLAMA_DURATIONS:
        metrics[f"{name}_ns"] = _non_negative_int(reported.get(name))
    return metrics


def build_answer_provenance(
    *,
    trace_id: str,
    release: ReleaseMetadata,
    generation: AnswerGenerationSnapshot,
    answer_mode: AnswerMode,
    evidence_mode: EvidenceMode,
    embedding_profile: EmbeddingProfile,
    timer: AnswerTimer,
    stream: OllamaGenerateStream,
    verifier_calls: int,
) -> dict:
    """The provenance record for one finished answer, or AnswerProvenanceError.

    Every argument must be the server's own typed object; anything else --
    including a request-shaped dict -- is refused, so nothing a client sends
    can stand in for a trusted value.
    """
    _require_trace_id(trace_id)
    for value, expected in (
        (release, ReleaseMetadata),
        (generation, AnswerGenerationSnapshot),
        (answer_mode, AnswerMode),
        (evidence_mode, EvidenceMode),
        (embedding_profile, EmbeddingProfile),
        (timer, AnswerTimer),
        (stream, OllamaGenerateStream),
    ):
        if not isinstance(value, expected):
            raise AnswerProvenanceError(f"expected a server {expected.__name__}")
    if stream.failure is not None:
        raise AnswerProvenanceError("an unfinished generation has no answer provenance")
    if generation.selection not in _SELECTIONS:
        raise AnswerProvenanceError("unknown model selection")
    if isinstance(verifier_calls, bool) or not isinstance(verifier_calls, int) or verifier_calls < 0:
        raise AnswerProvenanceError("verifier_calls must be a non-negative integer")

    # Only document_qa answers are graded claim by claim; the other modes
    # never call the verifier, so they record no verifier prompt or call.
    graded = answer_mode is AnswerMode.DOCUMENT_QA
    verifier = generation.verifier

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "trace_id": trace_id,
        "release": {
            "app_version": release.app_version,
            "build_sha": release.build_sha,
        },
        "generation": {
            "profile_id": generation.profile_id,
            "model_tag": generation.model_tag,
            "selection": generation.selection,
            "expected_digest": _require_digest(generation.expected_digest),
            "observed_digest": _require_digest(generation.observed_digest),
            "requested_options": generation.options.as_ollama_options(),
            # None: no think field was sent. False: ``think: false`` was sent.
            "think": generation.think,
            "keep_alive": generation.keep_alive,
            "answer_read_timeout_s": generation.answer_read_timeout_s,
        },
        "prompt": {
            "answer_prompt_version": release.answer_prompt_version,
            "answer_mode": answer_mode.value,
            "claim_support_prompt_version": release.claim_support_prompt_version if graded else None,
        },
        "verification": {
            "profile_id": verifier.profile_id,
            "model_tag": verifier.model_tag,
            "requested_options": verifier.options.as_ollama_options(),
            "keep_alive": verifier.keep_alive,
            "calls_attempted": verifier_calls,
        } if graded else None,
        "retrieval": {
            "evidence_mode": evidence_mode.value,
            "configured_embedding_profile": {
                "profile_id": embedding_profile.profile_id,
                "provider": embedding_profile.provider,
                "model": embedding_profile.model,
                "dimension": embedding_profile.dimension,
            },
        },
        "timing": timer.elapsed_ms(),
        "ollama_metrics": _ollama_metrics(stream),
    }


def answer_provenance_json(provenance: dict) -> str:
    """Deterministic JSON text for the JSONB column. NaN/Infinity are refused."""
    if not isinstance(provenance, dict) or provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise AnswerProvenanceError("not an answer provenance record")
    return json.dumps(provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
