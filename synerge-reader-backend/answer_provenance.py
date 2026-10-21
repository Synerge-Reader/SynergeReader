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
* selection: which shared selection the answer was admitted under --
  ``selection`` is ``"shared_active"`` and ``selection_epoch`` the epoch of
  that persisted selection, captured with the admission ticket;
* verification (graded answers only): the verifier call captured in that same
  snapshot -- its profile, exact tag, requested options, ``keep_alive``, think
  policy (``null`` means no think field was sent), per-read timeout, expected
  digest and the observed digest copied from the answer's resolution -- plus
  how many verifier calls were attempted;
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
are never part of a record: no input here carries them. A runtime is recorded
only by its backend-owned key, never by its URL.

Schema versions. New records are version 4. Stored records are never
rewritten, filled in or upgraded; ``read_answer_provenance`` reads every
version with its own vocabulary:

* version 1: ``generation.selection`` is ``"default"`` or ``"admin_override"``
  (the retired per-request override); no verifier think, timeout or digest
  fields;
* version 2: the same selection vocabulary; the verifier think, timeout and
  digest fields were added;
* version 3: ``generation.selection`` is ``"shared_active"`` and
  ``generation.selection_epoch`` is the shared selection's epoch;
* version 4: version 3 plus ``generation.runtime_key`` and
  ``generation.experimental``, and ``verification.runtime_key`` for graded
  answers. ``think`` may also be ``true`` or an effort level (for an
  experimental profile whose runtime smoke used it). A version 1-3 record
  has no runtime key: it is read as not recorded, never as ``primary``.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Callable, Optional

from answer_evidence import EvidenceMode
from citation_generation import AnswerMode
from generation_profiles import (
    RUNTIME_KEYS,
    SHARED_SELECTION,
    AnswerGenerationSnapshot,
    FixedModelCall,
    GenerationProfileError,
    normalize_digest,
    require_think_value,
)
from ollama_generation import OllamaGenerateStream
from rag_model_profiles import EmbeddingProfile
from release_metadata import ReleaseMetadata

PROVENANCE_SCHEMA_VERSION = 4

# The selection values each stored schema version may carry.
SELECTIONS_BY_SCHEMA_VERSION = {
    1: ("default", "admin_override"),
    2: ("default", "admin_override"),
    3: (SHARED_SELECTION,),
    4: (SHARED_SELECTION,),
}
# The first version that records which runtime generated the answer.
RUNTIME_KEY_SCHEMA_VERSION = 4

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


def _require_runtime(value: object) -> str:
    if not isinstance(value, str) or value not in RUNTIME_KEYS:
        raise AnswerProvenanceError("a runtime is recorded by its backend-owned key")
    return value


def _require_think(value: object):
    try:
        return require_think_value("think", value)
    except GenerationProfileError:
        raise AnswerProvenanceError("think must be a think setting this backend can send") from None


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
    if not isinstance(generation.verifier, FixedModelCall):
        raise AnswerProvenanceError("expected the snapshot's own verifier call")
    if stream.failure is not None:
        raise AnswerProvenanceError("an unfinished generation has no answer provenance")
    # A new record describes only the shared selection it was admitted under;
    # the historical selection values belong to stored version 1/2 records.
    if generation.selection != SHARED_SELECTION or not _is_epoch(generation.selection_epoch):
        raise AnswerProvenanceError("a new answer records its shared selection and epoch")
    if isinstance(verifier_calls, bool) or not isinstance(verifier_calls, int) or verifier_calls < 0:
        raise AnswerProvenanceError("verifier_calls must be a non-negative integer")

    # Only document_qa answers are graded claim by claim; the other modes
    # never call the verifier, so they record no verifier prompt or call.
    graded = answer_mode is AnswerMode.DOCUMENT_QA
    verifier = generation.verifier
    runtime_key = _require_runtime(generation.runtime_key)
    if _require_runtime(verifier.runtime_key) != runtime_key:
        raise AnswerProvenanceError("the verifier runs on the answer's own runtime")

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
            "selection_epoch": generation.selection_epoch,
            "expected_digest": _require_digest(generation.expected_digest),
            "observed_digest": _require_digest(generation.observed_digest),
            "requested_options": generation.options.as_ollama_options(),
            # None: no think field was sent. Otherwise the exact value sent
            # (false, true or an effort level); thinking text is never kept.
            "think": _require_think(generation.think),
            "keep_alive": generation.keep_alive,
            "answer_read_timeout_s": generation.answer_read_timeout_s,
            # Backend-owned key, never an endpoint.
            "runtime_key": runtime_key,
            "experimental": bool(generation.experimental),
        },
        "prompt": {
            "answer_prompt_version": release.answer_prompt_version,
            "answer_mode": answer_mode.value,
            "claim_support_prompt_version": release.claim_support_prompt_version if graded else None,
        },
        "verification": {
            "profile_id": verifier.profile_id,
            "model_tag": verifier.model_tag,
            "expected_digest": _require_digest(verifier.expected_digest),
            "observed_digest": _require_digest(verifier.observed_digest),
            "requested_options": verifier.options.as_ollama_options(),
            # None: no think field was sent. Otherwise the exact value sent.
            "think": _require_think(verifier.think),
            "keep_alive": verifier.keep_alive,
            "read_timeout_s": verifier.read_timeout_s,
            "runtime_key": runtime_key,
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
    """Deterministic JSON text for the JSONB column. NaN/Infinity are refused.
    Only a current (version 4) record is ever serialized for storage."""
    if not isinstance(provenance, dict) or provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise AnswerProvenanceError("not an answer provenance record")
    return json.dumps(provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _is_epoch(value: object) -> bool:
    return type(value) is int and value >= 1


def read_answer_provenance(record: object) -> dict:
    """The model-selection facts of one STORED record, read with the
    vocabulary of its own schema version; AnswerProvenanceError otherwise.

    Pure and read-only: the record is never modified, upgraded or filled in.
    A version 1 or 2 record reports its historical ``selection`` and a
    ``selection_epoch`` of None (those versions had no shared selection); a
    version 3 or 4 record must carry ``"shared_active"`` and a positive epoch.
    A version 4 record must also name a known ``runtime_key``; earlier
    versions report ``runtime_key`` None (not recorded). Any other version or
    value is refused rather than guessed. A NULL column ("not recorded") is
    the caller's case and never reaches this reader.
    """
    if not isinstance(record, dict):
        raise AnswerProvenanceError("not an answer provenance record")
    version = record.get("schema_version")
    if type(version) is not int or version not in SELECTIONS_BY_SCHEMA_VERSION:
        raise AnswerProvenanceError("unknown answer provenance schema version")
    generation = record.get("generation")
    if not isinstance(generation, dict):
        raise AnswerProvenanceError("answer provenance has no generation record")
    selection = generation.get("selection")
    if selection not in SELECTIONS_BY_SCHEMA_VERSION[version]:
        raise AnswerProvenanceError("selection value does not belong to this schema version")
    if version >= 3:
        epoch = generation.get("selection_epoch")
        if not _is_epoch(epoch):
            raise AnswerProvenanceError("a shared selection record needs its epoch")
    else:
        if "selection_epoch" in generation:
            raise AnswerProvenanceError("a historical record carries no selection epoch")
        epoch = None
    for key in ("expected_digest", "observed_digest"):
        value = generation.get(key)
        if value is not None and normalize_digest(value) != value:
            raise AnswerProvenanceError("a model digest must be None or 64 lowercase hex characters")
    if version >= RUNTIME_KEY_SCHEMA_VERSION:
        runtime_key = generation.get("runtime_key")
        if not isinstance(runtime_key, str) or runtime_key not in RUNTIME_KEYS:
            raise AnswerProvenanceError("a version 4 record names its runtime by a known key")
    else:
        if "runtime_key" in generation:
            raise AnswerProvenanceError("a record before version 4 carries no runtime key")
        runtime_key = None
    return {
        "schema_version": version,
        "selection": selection,
        "selection_epoch": epoch,
        "profile_id": generation.get("profile_id"),
        "model_tag": generation.get("model_tag"),
        "expected_digest": generation.get("expected_digest"),
        "observed_digest": generation.get("observed_digest"),
        "runtime_key": runtime_key,
    }
