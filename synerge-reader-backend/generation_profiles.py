"""Backend-owned registry of local answer-generation model profiles.

This module is a pure configuration/value-object layer, like
``rag_model_profiles.py``: no network, filesystem, database, or environment
access happens at import time or in any function here. The application's
composition boundary (``main.py``) supplies the one runtime fact a snapshot
may need -- which digest an installed model tag currently has -- as a
callable.

What the registry decides, and what a client can never override:

* which exact local model tags may answer a question at all;
* the native Ollama ``options`` each call sends (sampling, context window,
  output limit), its ``keep_alive``, and how long each role's transport may
  wait on a read;
* whether a ``think`` field is sent, which only happens as ``think: false``
  for a profile whose no-thinking behaviour has been evidenced;
* the bounded claim-verification and background-helper calls. There is no
  separate verifier or helper model: each role call is derived from one
  resolved profile -- its exact tag, context window, ``keep_alive`` and think
  policy -- with the role's own temperature, output limit and read timeout.

Which registered profile is in use is NOT decided here or by any request: it
is the one shared active selection an administrator changes through the
switch coordinator (``generation_selection.py`` / ``generation_coordinator.py``).
Every answer, its verifier calls and every helper job capture that selection
-- profile id and epoch -- once, when admission grants them, and build their
snapshot from it with ``shared_generation_snapshot``. There is no per-request
override and no fallback: a selection naming an unknown, unqualified or
mismatched profile refuses instead of using another model.

The option values are the settings this backend REQUESTS. Ollama may still
apply its own limits (for example a server-wide ``OLLAMA_NUM_PARALLEL`` or a
memory-driven context cap), so they are not proof of the effective runtime
settings; the diagnostic harness records what the server reports instead.

Two separate kinds of reviewed data live on a profile:

* ``expected_digest`` -- the manifest digest of the artifact this profile is
  meant to run. It is a pin, not an observation: an answer's snapshot carries
  an ``observed_digest`` only when the installed identity was actually read
  for that answer.
* ``qualification`` -- which release gates have evidence. A profile with
  ``requires_qualification`` is selectable (as the shared selection, or as the
  registry default that first seeds it) only when its combined-capacity and
  quality evidence are both recorded, and every answer or helper job it
  produces is preceded by an installed-digest check.

The unqualified control (``llama_control``) is the existing production model:
it is explicitly marked as not requiring qualification, so its answers cost no
per-job inventory call. Whether a profile needs qualification never depends on
which profile happens to be selected.

Experimental profiles (19 October 2026) are a third, separate tier:

* ``experimental`` -- the profile has only operator-reported runtime-smoke
  evidence (``RuntimeSmokeEvidence``): one bounded synthetic request on the
  runtime that will serve it. No quality, capacity or vision gate is recorded
  or implied, and ``selectable`` (the qualified tier) stays False for it.
  Whether it may be selected is a DEPLOYMENT decision the registry holds:
  ``experimental_enabled`` (server-side, default off) and the profile's
  runtime being configured. Every answer and helper job such a profile
  produces, and every switch to it, still checks the installed identity on
  the profile's own runtime first: ``requires_identity_check`` is decoupled
  from ``requires_qualification``.
* ``runtime_key`` -- a backend-owned key naming the generation runtime that
  serves the profile. Server configuration maps keys to endpoints; no request
  can supply or override an endpoint, a runtime key or a tag, and a runtime
  that fails never falls back to another runtime or model.
* ``think`` may additionally be ``True`` or one of ``THINK_LEVELS`` for an
  experimental profile whose runtime smoke used exactly that value. Thinking
  text is never returned, persisted or shown; only ``response`` is the answer.
* the verifier and helper roles have explicit ``RolePolicy`` values (think
  setting and output limit). A thinking model's short roles get a bounded
  thinking-plus-final-output allowance instead of inheriting a 16-token limit
  they would exhaust while thinking. The roles always use the job's own model
  and runtime; nothing silently switches the verifier to another model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, FrozenSet, Literal, Mapping, Optional, Tuple, Union

from ollama_generation import completion_diagnostic

# --------------------------------------------------------------------------
# Profile ids, installed tags and pinned artifact identities
# --------------------------------------------------------------------------

LLAMA_CONTROL = "llama_control"
GEMMA_CANDIDATE = "gemma_candidate"
QWEN38_EXPERIMENTAL = "qwen38_27b_experimental"
NEMOTRON35_EXPERIMENTAL = "nemotron35_lightning_30b_experimental"
MISTRAL35_EXPERIMENTAL = "mistral_medium35_128b_experimental"

LLAMA_CONTROL_TAG = "llama3.1:8b"
GEMMA_CANDIDATE_TAG = "gemma4:31b"
QWEN38_TAG = "qwen3.8:27b"
NEMOTRON35_TAG = "nemotron-3.5-lightning:30b"
MISTRAL35_TAG = "mistral-medium-3.5:128b"

# Operator-verified full manifest digests of the installed artifacts. These are
# expected identities (pins), never evidence about a particular answer.
LLAMA_CONTROL_DIGEST = "42182419e9508c30c4b1fe55015f06b65f4ca4b9e28a744be55008d21998a093"
GEMMA_CANDIDATE_DIGEST = "6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7"
QWEN38_DIGEST = "aaee06c39dcf2437cde036998d960e1fc1494b8191be7cc9657d01e509097813"
NEMOTRON35_DIGEST = "e6905fffe2e2d9bbe325a9a2d2e074e3bd32683721bc6dfba409dede2f17f4dc"
MISTRAL35_DIGEST = "0341632adb051badc332bd814779ab931ab78619750bebe3b7c7b1401e9d70df"

# --------------------------------------------------------------------------
# Generation runtimes
# --------------------------------------------------------------------------
#
# Backend-owned keys. Which endpoint a key reaches is server configuration
# (main.py); the keys themselves are a closed set, so a profile can never name
# an endpoint, and nothing a client sends can choose one.
#
# ``primary``: the existing host Ollama that serves Llama, Gemma and the
# configured embedder. ``compat_cuda121``: the separate, private CUDA 12.1 /
# SM70 compatibility runtime (Ollama 0.32.12) for the experimental profiles.
PRIMARY_RUNTIME = "primary"
COMPAT_RUNTIME = "compat_cuda121"
RUNTIME_KEYS = (PRIMARY_RUNTIME, COMPAT_RUNTIME)

# The image the experimental profiles' runtime smoke ran on (operator-reported).
COMPAT_RUNTIME_TESTED_IMAGE_ID = "sha256:96e3cc4e4155ef96d11578415f1ac36892d8bdc3a83c40623cc8ffdfbf4392ed"
COMPAT_RUNTIME_OLLAMA_VERSION = "0.32.12"

# The think values Ollama's native API defines: a bool, or an effort level.
# A profile may send one only with evidence for exactly that value.
THINK_LEVELS = ("low", "medium", "high")
ThinkValue = Optional[Union[bool, str]]

# --------------------------------------------------------------------------
# Requested settings
# --------------------------------------------------------------------------

ANSWER_TEMPERATURE = 0.2
ANSWER_NUM_CTX = 8192
ANSWER_NUM_PREDICT = 1200
ANSWER_KEEP_ALIVE = "5m"

# How long each role's transport may wait for one read, including the wait for
# the first byte, which covers a cold model load. These are per-read timeouts,
# not caps on a whole answer or job.
#
# Llama keeps the values its call sites already used: 60 s for an answer, 30 s
# for a verifier call and 120 s for a background helper call.
#
# Gemma's bounded smoke took 102.527 s in total, 91.783 s of it loading, so
# 60 s would time out a cold Gemma answer before its first token. 150 s leaves
# margin for that load plus prompt evaluation. Any Gemma role call can be the
# one that finds the model cold, so its verifier and helpers get the same
# provisional 150 s. Gemma's later qualification left these values unchanged:
# they only remove a guaranteed premature timeout and say nothing about
# acceptable latency.
LLAMA_ANSWER_READ_TIMEOUT_S = 60.0
LLAMA_VERIFIER_READ_TIMEOUT_S = 30.0
LLAMA_HELPER_READ_TIMEOUT_S = 120.0
GEMMA_ANSWER_READ_TIMEOUT_S = 150.0
GEMMA_VERIFIER_READ_TIMEOUT_S = 150.0
GEMMA_HELPER_READ_TIMEOUT_S = 150.0
_MIN_READ_TIMEOUT_S = 1.0
_MAX_READ_TIMEOUT_S = 600.0

# Experimental profiles: PROVISIONAL bounds, not measured at the application
# context. The runtime smoke used a 2,048-token context; the application asks
# for ANSWER_NUM_CTX (8,192), which the bounded candidate check must exercise.
#
# A streamed answer's read timeout covers the wait for the first record: the
# load plus prompt evaluation (thinking records then arrive as bytes). Qwen
# and Nemotron loaded in 9.28 s and 12.13 s at 2,048 tokens; 180 s leaves room
# for a larger context and the full evidence prompt. Mistral's cold load alone
# took 180.38 s at 2,048 tokens -- more than Gemma's 150 s -- so it gets 420 s.
# Verifier and helper calls are NOT streamed: their read timeout covers the
# whole bounded generation, thinking included, and is still clamped to the
# job's remaining budget. Mistral's job budgets are raised to 900 s (the
# shared default is 420 s) because one answer can include a cold load, a
# thinking-plus-answer generation and two verifier calls.
EXPERIMENTAL_READ_TIMEOUT_S = 180.0
EXPERIMENTAL_HELPER_READ_TIMEOUT_S = 300.0
MISTRAL35_READ_TIMEOUT_S = 420.0
MISTRAL35_HELPER_READ_TIMEOUT_S = 600.0
MISTRAL35_JOB_BUDGET_S = 900.0
# Thinking and the final answer share one output limit (thinking tokens are
# generated tokens). 2,048 keeps prompt plus output inside the 8,192-token
# context for the largest evidence prompt; a reply that reaches it is reported
# as cut off, never as an answer.
EXPERIMENTAL_ANSWER_NUM_PREDICT = 2048
# The short roles inherit the answer's tested think setting (no no-thinking
# behaviour is evidenced for these models) with a bounded allowance for
# thinking plus the status word / structured reply.
EXPERIMENTAL_VERIFIER_NUM_PREDICT = 512
EXPERIMENTAL_HELPER_NUM_PREDICT = 2048
# Nemotron's own role allowances. In the bounded candidate check
# (xm-20261020T030140Z-dc2e7a43) its answer finished at 1,530 of 2,048 tokens,
# but its verifier generated exactly 512 and its insights helper exactly 2,048
# tokens, and neither reply was usable; Qwen and Mistral used 166-706 tokens
# for the same calls. Thinking stays on (think=False and effort levels are
# accepted by the runtime but untested for this model). The verifier gets the
# answer's own allowance: its prompt holds a subset of the answer's evidence, so
# it fits the 8,192-token context by the answer's bound. The helper gets the
# registry maximum: its excerpt is at most 6,000 characters (the Q&A helper's
# 4,000); the check's insights prompt was 1,218 tokens. At the observed 110-118
# generated tokens/s, 2,048 and 4,096 tokens take about 19 s and 37 s, inside
# the unchanged 180 s and 300 s read timeouts and the 420 s job budget. Still
# bounded: a reply that reaches either limit is cut off and is never a
# judgment, nor stored.
NEMOTRON35_VERIFIER_NUM_PREDICT = 2048
NEMOTRON35_HELPER_NUM_PREDICT = 4096
# One combined answer-or-helper job at a time while an experimental profile is
# the shared selection: a conservative initial limit, not a capacity claim.
EXPERIMENTAL_MAX_CONCURRENT_JOBS = 1
_MAX_CONCURRENT_JOBS = 64
_MIN_JOB_BUDGET_S = 1.0
_MAX_JOB_BUDGET_S = 3600.0

# A readiness probe: load the model and emit a few tokens. Kept apart from the
# answer settings so a smoke run can never be mistaken for an answer profile.
SMOKE_NUM_CTX = 4096
SMOKE_NUM_PREDICT = 32

# The verifier must answer with one status word. A small output cap keeps a
# chatty reply from running on; a truncated reply is treated as malformed.
VERIFIER_TEMPERATURE = 0.0
VERIFIER_NUM_PREDICT = 16

# Background helpers (insight extraction, knowledge-base Q&A from an admin URL
# import) ask for a short structured reply from a bounded excerpt.
HELPER_NUM_PREDICT = 1200

_MIN_NUM_CTX = 512
_MAX_NUM_CTX = 32768
_MAX_NUM_PREDICT = 4096
_MAX_TOP_K = 1000

_PROFILE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_MODEL_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*:[A-Za-z0-9._-]+$")
_KEEP_ALIVE_PATTERN = re.compile(r"^[1-9][0-9]{0,3}[smh]$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class GenerationProfileError(ValueError):
    """A profile, option, or registry definition is invalid."""


class ModelSelectionError(Exception):
    """A model selection or snapshot was refused. Messages are fixed and safe."""


class UnknownModelProfileError(ModelSelectionError):
    """The selected profile id is not in the registry."""


class ProfileNotQualifiedError(ModelSelectionError):
    """The profile requires qualification and its evidence is incomplete."""


class ModelIdentityMismatchError(ModelSelectionError):
    """The expected tag is absent, or installed with a different digest."""


class ModelIdentityUnavailableError(ModelSelectionError):
    """The installed model identity could not be read."""


class ExperimentalProfileDisabledError(ProfileNotQualifiedError):
    """An experimental profile while experimental profiles are disabled."""


class ModelRuntimeUnavailableError(ModelSelectionError):
    """The profile's generation runtime is not configured on this server."""


# --------------------------------------------------------------------------
# Validators
# --------------------------------------------------------------------------


def _is_real_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require_int(name: str, value: object, low: int, high: int) -> int:
    # bool is an int subclass; True must never pass as 1.
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise GenerationProfileError(f"{name} must be an integer in [{low}, {high}]; got {value!r}")
    return value


def _require_match(name: str, pattern: re.Pattern, value: object) -> str:
    if not isinstance(value, str) or not pattern.match(value):
        raise GenerationProfileError(f"{name} is not valid: {value!r}")
    return value


def _require_read_timeout(name: str, value: object) -> float:
    # _is_real_number rejects bool, so True can never pass as 1 second.
    if not _is_real_number(value) or not _MIN_READ_TIMEOUT_S <= value <= _MAX_READ_TIMEOUT_S:
        raise GenerationProfileError(
            f"{name} must be a finite number in [{_MIN_READ_TIMEOUT_S}, {_MAX_READ_TIMEOUT_S}]"
        )
    return value


def require_think_value(name: str, value: object) -> ThinkValue:
    """``value`` if it is a think setting this backend can send: None (no
    field), the bools False/True, or one of THINK_LEVELS. Anything else --
    another string, a number, ``1`` or ``0`` posing as a bool -- is refused."""
    if value is None or value is False or value is True:
        return value
    if type(value) is str and value in THINK_LEVELS:
        return value
    raise GenerationProfileError(f"{name} must be None, a bool or one of {THINK_LEVELS}; got {value!r}")


def same_think(a: object, b: object) -> bool:
    """Exact think equality: True is not "high", and 1 is not True."""
    return type(a) is type(b) and a == b


def require_runtime_key(name: str, value: object) -> str:
    if not isinstance(value, str) or value not in RUNTIME_KEYS:
        raise GenerationProfileError(f"{name} must be one of {RUNTIME_KEYS}; got {value!r}")
    return value


def _require_job_budget(name: str, value: object) -> Optional[float]:
    if value is None:
        return None
    if not _is_real_number(value) or not _MIN_JOB_BUDGET_S <= value <= _MAX_JOB_BUDGET_S:
        raise GenerationProfileError(
            f"{name} must be None or a finite number in [{_MIN_JOB_BUDGET_S}, {_MAX_JOB_BUDGET_S}]"
        )
    return float(value)


def normalize_digest(value: object) -> Optional[str]:
    """A manifest digest as 64 lowercase hex characters, or None if malformed.

    Ollama reports ``/api/tags`` digests without a prefix; a pinned digest may
    be written with ``sha256:``. Both compare equal after this.
    """
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    if candidate.startswith("sha256:"):
        candidate = candidate[len("sha256:"):]
    return candidate if _DIGEST_PATTERN.match(candidate) else None


# --------------------------------------------------------------------------
# Value objects
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationOptions:
    """Native Ollama ``options`` for one call. Validated on construction.

    ``top_k``/``top_p`` default to None, which means "not sent": each model
    keeps its own Modelfile defaults, so per-model sampling differences are
    preserved rather than flattened to one shared value.
    """

    temperature: float
    num_ctx: int
    num_predict: int
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    def __post_init__(self) -> None:
        if not _is_real_number(self.temperature) or not 0.0 <= self.temperature <= 2.0:
            raise GenerationProfileError(
                f"temperature must be a number in [0, 2]; got {self.temperature!r}"
            )
        _require_int("num_ctx", self.num_ctx, _MIN_NUM_CTX, _MAX_NUM_CTX)
        _require_int("num_predict", self.num_predict, 1, _MAX_NUM_PREDICT)
        if self.num_predict >= self.num_ctx:
            raise GenerationProfileError("num_predict must be smaller than num_ctx")
        if self.top_k is not None:
            _require_int("top_k", self.top_k, 1, _MAX_TOP_K)
        if self.top_p is not None and (
            not _is_real_number(self.top_p) or not 0.0 < self.top_p <= 1.0
        ):
            raise GenerationProfileError(f"top_p must be a number in (0, 1]; got {self.top_p!r}")

    def as_ollama_options(self) -> dict:
        """A fresh dict on every call, so no caller can mutate the profile."""
        options = {
            "temperature": float(self.temperature),
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }
        if self.top_k is not None:
            options["top_k"] = self.top_k
        if self.top_p is not None:
            options["top_p"] = float(self.top_p)
        return options


ANSWER_OPTIONS = GenerationOptions(
    temperature=ANSWER_TEMPERATURE,
    num_ctx=ANSWER_NUM_CTX,
    num_predict=ANSWER_NUM_PREDICT,
)
SMOKE_OPTIONS = GenerationOptions(
    temperature=ANSWER_TEMPERATURE,
    num_ctx=SMOKE_NUM_CTX,
    num_predict=SMOKE_NUM_PREDICT,
)


@dataclass(frozen=True)
class Qualification:
    """Which release gates have evidence for a profile that requires it.

    ``think_false_verified`` means ``think: false`` was accepted by the
    installed model and its thinking stayed separate from the answer in a
    bounded run. It permits sending ``think: false``; it does not qualify the
    model. Selection additionally needs combined generator/verifier/embedder
    capacity and answer-quality evidence. ``evidence_ref`` names where the
    evidence is recorded, so every recorded gate is reviewable.
    """

    think_false_verified: bool
    combined_capacity_verified: bool
    quality_verified: bool
    evidence_ref: str

    def __post_init__(self) -> None:
        for name in ("think_false_verified", "combined_capacity_verified", "quality_verified"):
            if not isinstance(getattr(self, name), bool):
                raise GenerationProfileError(f"{name} must be a bool")
        if not isinstance(self.evidence_ref, str) or not self.evidence_ref.strip():
            raise GenerationProfileError("qualification evidence_ref must be a non-empty string")


RUNTIME_SMOKE_VERDICTS = ("PASS", "REVIEW_REQUIRED")


@dataclass(frozen=True)
class RuntimeSmokeEvidence:
    """Operator-reported, bounded runtime smoke for an experimental profile.

    What it records: on the runtime that will serve the profile, one
    synthetic request with exactly ``tested_think`` and a ``tested_num_ctx``
    context returned the expected final answer in the native ``response``
    field, emitted a separate thinking field and ended with
    ``done_reason == "stop"``. ``verdict`` is the original script verdict,
    preserved as reported (REVIEW_REQUIRED keeps its ``review_note``).

    What it does NOT record: answer quality, combined capacity, vision, the
    application's own context or output budgets, or latency. Those gates are
    deferred and are never implied by this evidence. The evidence was
    supplied by the operator; offline tests do not reproduce inference.
    """

    tested_think: Tuple[ThinkValue, ...]
    tested_num_ctx: int
    runtime_version: str
    runtime_image_id: str
    verdict: str
    evidence_ref: str
    review_note: Optional[str] = None

    def __post_init__(self) -> None:
        tested = tuple(self.tested_think)
        if not tested:
            raise GenerationProfileError("runtime smoke must name the think value(s) it tested")
        for value in tested:
            require_think_value("tested_think", value)
        object.__setattr__(self, "tested_think", tested)
        _require_int("tested_num_ctx", self.tested_num_ctx, _MIN_NUM_CTX, _MAX_NUM_CTX)
        if not isinstance(self.runtime_version, str) or not re.match(r"^[0-9]+\.[0-9]+\.[0-9]+$", self.runtime_version):
            raise GenerationProfileError("runtime_version must be a version like 0.32.12")
        if not isinstance(self.runtime_image_id, str) or normalize_digest(self.runtime_image_id) is None \
                or not self.runtime_image_id.startswith("sha256:"):
            raise GenerationProfileError("runtime_image_id must be a sha256: image id")
        if self.verdict not in RUNTIME_SMOKE_VERDICTS:
            raise GenerationProfileError(f"verdict must be one of {RUNTIME_SMOKE_VERDICTS}")
        if self.verdict == "REVIEW_REQUIRED" and (not isinstance(self.review_note, str) or not self.review_note.strip()):
            raise GenerationProfileError("a REVIEW_REQUIRED verdict keeps its review note")
        if self.review_note is not None and not isinstance(self.review_note, str):
            raise GenerationProfileError("review_note must be a string or None")
        if not isinstance(self.evidence_ref, str) or not self.evidence_ref.strip():
            raise GenerationProfileError("runtime smoke evidence_ref must be a non-empty string")

    def tested(self, think: object) -> bool:
        return any(same_think(value, think) for value in self.tested_think)


@dataclass(frozen=True)
class RolePolicy:
    """The think setting and output limit of one bounded role call (the claim
    verifier, or a background helper). Format only; whether the profile may
    send that think value is checked against the profile's own evidence."""

    think: ThinkValue
    num_predict: int

    def __post_init__(self) -> None:
        require_think_value("role think", self.think)
        _require_int("role num_predict", self.num_predict, 1, _MAX_NUM_PREDICT)


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    model_tag: str
    display_name: str
    answer_options: GenerationOptions
    keep_alive: str
    # Per-read transport timeouts for each role this profile serves.
    answer_read_timeout_s: float
    verifier_read_timeout_s: float
    helper_read_timeout_s: float
    # Explicit policy, never inferred from default status: True means the
    # profile may answer only with complete qualification evidence and only
    # after its installed digest matches ``expected_digest``.
    requires_qualification: bool
    expected_digest: Optional[str] = None
    # None: no think field is sent. False is sent as ``think: false`` and is
    # allowed only once that behaviour is evidenced (Qualification). True or
    # an effort level is allowed only for an experimental profile whose
    # runtime smoke used exactly that value; its thinking text is never kept.
    think: ThinkValue = None
    qualification: Optional[Qualification] = None
    # The generation runtime that serves this profile (a backend-owned key).
    runtime_key: str = PRIMARY_RUNTIME
    # Experimental tier: runtime-smoke evidence only. Never qualified.
    experimental: bool = False
    runtime_smoke: Optional[RuntimeSmokeEvidence] = None
    # Explicit role policies; None keeps the historical derivation (the
    # profile's think setting with the shared verifier/helper output limits).
    verifier_policy: Optional[RolePolicy] = None
    helper_policy: Optional[RolePolicy] = None
    # While this profile is the shared selection: at most this many admitted
    # or unresolved jobs of either kind (None: the shared admission settings),
    # and per-kind whole-job budgets (None: the shared settings).
    max_concurrent_jobs: Optional[int] = None
    answer_job_budget_s: Optional[float] = None
    helper_job_budget_s: Optional[float] = None

    def __post_init__(self) -> None:
        _require_match("profile_id", _PROFILE_ID_PATTERN, self.profile_id)
        _require_match("model_tag", _MODEL_TAG_PATTERN, self.model_tag)
        _require_match("keep_alive", _KEEP_ALIVE_PATTERN, self.keep_alive)
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise GenerationProfileError("display_name must be a non-empty string")
        if not isinstance(self.answer_options, GenerationOptions):
            raise GenerationProfileError("answer_options must be GenerationOptions")
        for name in ("answer_read_timeout_s", "verifier_read_timeout_s", "helper_read_timeout_s"):
            _require_read_timeout(name, getattr(self, name))
        if not isinstance(self.requires_qualification, bool):
            raise GenerationProfileError("requires_qualification must be a bool")
        if self.qualification is not None and not isinstance(self.qualification, Qualification):
            raise GenerationProfileError("qualification must be a Qualification")
        require_runtime_key("runtime_key", self.runtime_key)
        if not isinstance(self.experimental, bool):
            raise GenerationProfileError("experimental must be a bool")
        if self.runtime_smoke is not None and not isinstance(self.runtime_smoke, RuntimeSmokeEvidence):
            raise GenerationProfileError("runtime_smoke must be RuntimeSmokeEvidence")
        expected = None
        if self.expected_digest is not None:
            expected = normalize_digest(self.expected_digest)
            if expected is None:
                raise GenerationProfileError("expected_digest must be a sha256 manifest digest")
        if self.requires_qualification and expected is None:
            raise GenerationProfileError(
                "a profile that requires qualification needs an expected_digest to verify"
            )
        if self.experimental:
            # A separate tier, never a shortcut into the qualified one.
            if self.requires_qualification or self.qualification is not None:
                raise GenerationProfileError("an experimental profile carries no qualification tier")
            if self.runtime_smoke is None:
                raise GenerationProfileError("an experimental profile needs its runtime smoke evidence")
            if expected is None:
                raise GenerationProfileError("an experimental profile needs an expected_digest to verify")
            if self.max_concurrent_jobs is None:
                raise GenerationProfileError("an experimental profile needs an explicit job limit")
        elif self.runtime_smoke is not None:
            raise GenerationProfileError("runtime smoke evidence belongs to an experimental profile")
        self._require_sendable_think("think", self.think)
        for name in ("verifier_policy", "helper_policy"):
            policy = getattr(self, name)
            if policy is None:
                continue
            if not isinstance(policy, RolePolicy):
                raise GenerationProfileError(f"{name} must be a RolePolicy")
            self._require_sendable_think(f"{name}.think", policy.think)
            if policy.num_predict >= self.answer_options.num_ctx:
                raise GenerationProfileError(f"{name}.num_predict must be smaller than num_ctx")
        if self.max_concurrent_jobs is not None:
            _require_int("max_concurrent_jobs", self.max_concurrent_jobs, 1, _MAX_CONCURRENT_JOBS)
        object.__setattr__(self, "answer_job_budget_s", _require_job_budget("answer_job_budget_s", self.answer_job_budget_s))
        object.__setattr__(self, "helper_job_budget_s", _require_job_budget("helper_job_budget_s", self.helper_job_budget_s))
        object.__setattr__(self, "expected_digest", expected)

    def _require_sendable_think(self, name: str, value: object) -> None:
        require_think_value(name, value)
        if value is None:
            return
        if value is False:
            if self.qualification is None or not self.qualification.think_false_verified:
                raise GenerationProfileError(
                    "think: false may only be sent once that behaviour is evidenced"
                )
            return
        # True or an effort level: only what this profile's runtime smoke used.
        if not self.experimental or self.runtime_smoke is None or not self.runtime_smoke.tested(value):
            raise GenerationProfileError(
                f"{name} {value!r} may only be sent by an experimental profile whose runtime smoke used it"
            )

    @property
    def selectable(self) -> bool:
        """Whether this profile is in the control or qualified tier: it may
        answer, explicitly or as default, wherever its runtime is available.

        False for every experimental profile: whether an experimental profile
        may be selected is the deployment's decision (``ModelRegistry``)."""
        if self.experimental:
            return False
        if not self.requires_qualification:
            return True
        evidence = self.qualification
        return (
            evidence is not None
            and evidence.combined_capacity_verified
            and evidence.quality_verified
        )

    @property
    def requires_identity_check(self) -> bool:
        """Whether every answer, helper job and switch to this profile first
        reads its installed identity on its own runtime. Decoupled from the
        qualification tier: an experimental profile is checked too."""
        return self.requires_qualification or self.experimental

    @property
    def evidence_tier(self) -> str:
        if self.experimental:
            return "experimental"
        return "qualified" if self.requires_qualification else "control"

    def verifier_role(self) -> RolePolicy:
        if self.verifier_policy is not None:
            return self.verifier_policy
        return RolePolicy(think=self.think, num_predict=VERIFIER_NUM_PREDICT)

    def helper_role(self) -> RolePolicy:
        if self.helper_policy is not None:
            return self.helper_policy
        return RolePolicy(think=self.think, num_predict=HELPER_NUM_PREDICT)


def build_generate_payload(
    model_tag: str,
    prompt: str,
    options: GenerationOptions,
    keep_alive: str,
    *,
    stream: bool,
    think: ThinkValue = None,
) -> dict:
    """One native ``/api/generate`` request body.

    Sampling, context and output limits travel inside ``options``, where Ollama
    reads them; ``keep_alive`` and ``think`` are top-level request fields.
    Nothing else is sent, so no stale top-level setting can masquerade as
    policy. No endpoint, runtime or credential is ever part of a payload.
    """
    if not isinstance(prompt, str):
        raise GenerationProfileError("prompt must be a string")
    require_think_value("think", think)
    payload = {
        "model": model_tag,
        "prompt": prompt,
        "stream": bool(stream),
        "options": options.as_ollama_options(),
        "keep_alive": keep_alive,
    }
    if think is not None:
        payload["think"] = think
    return payload


@dataclass(frozen=True)
class FixedModelCall:
    """A trusted role call that no request can redirect: one answer's claim
    verifier, or one background helper call.

    It is derived from a single resolved profile -- the exact tag, context
    window, ``keep_alive`` and think policy the snapshot holds -- with the
    role's own bounded options and per-read timeout. It holds values, never a
    reference to the registry, so nothing that happens after resolution can
    change it.

    ``expected_digest`` is the profile's pin. ``observed_digest`` is copied
    from the snapshot it was derived from: the installed identity read when
    that answer or helper job was resolved, or None when no lookup was made
    (the unqualified control). It is not a separate per-call measurement and
    does not prove the shared tag was unchanged afterwards.
    """

    profile_id: str
    model_tag: str
    options: GenerationOptions
    keep_alive: str
    # None: no think field is sent. Otherwise the exact value sent (False,
    # True or an effort level), already checked against the profile.
    think: ThinkValue
    read_timeout_s: float
    expected_digest: Optional[str]
    observed_digest: Optional[str]
    # The runtime this call is sent to: always the job snapshot's own.
    runtime_key: str = PRIMARY_RUNTIME

    def __post_init__(self) -> None:
        _require_match("profile_id", _PROFILE_ID_PATTERN, self.profile_id)
        _require_match("model_tag", _MODEL_TAG_PATTERN, self.model_tag)
        _require_match("keep_alive", _KEEP_ALIVE_PATTERN, self.keep_alive)
        if not isinstance(self.options, GenerationOptions):
            raise GenerationProfileError("options must be GenerationOptions")
        require_think_value("think", self.think)
        require_runtime_key("runtime_key", self.runtime_key)
        _require_read_timeout("read_timeout_s", self.read_timeout_s)
        for name in ("expected_digest", "observed_digest"):
            value = getattr(self, name)
            if value is not None and normalize_digest(value) != value:
                raise GenerationProfileError(f"{name} must be None or 64 lowercase hex characters")

    def ollama_payload(self, prompt: str, *, stream: bool = False) -> dict:
        return build_generate_payload(
            self.model_tag, prompt, self.options, self.keep_alive, stream=stream, think=self.think
        )

    def describe_completion(self, role: str, body) -> str:
        """One log line for a reply to this call: its profile, the output
        limit it requested, and the reply's counts and fixed words -- never
        its text (see ollama_generation.completion_diagnostic)."""
        return completion_diagnostic(role, self.profile_id, self.options.num_predict, body)


def _role_call(
    *,
    profile_id: str,
    model_tag: str,
    num_ctx: int,
    keep_alive: str,
    think: ThinkValue,
    expected_digest: Optional[str],
    observed_digest: Optional[str],
    temperature: float,
    num_predict: int,
    read_timeout_s: float,
    runtime_key: str = PRIMARY_RUNTIME,
) -> FixedModelCall:
    """One bounded role call. Only the role's temperature, output limit,
    think policy and read timeout differ from the profile the identity and
    runtime fields come from."""
    return FixedModelCall(
        profile_id=profile_id,
        model_tag=model_tag,
        options=GenerationOptions(temperature=temperature, num_ctx=num_ctx, num_predict=num_predict),
        keep_alive=keep_alive,
        think=think,
        read_timeout_s=float(read_timeout_s),
        expected_digest=expected_digest,
        observed_digest=observed_digest,
        runtime_key=runtime_key,
    )


# "shared_active": the one shared selection an administrator switches, captured
# with its epoch at admission. That is the only selection the application
# makes. "default" and "admin_override" are the historical values of answers
# saved before the shared selection existed (provenance schema 1 and 2); they
# remain valid snapshot values only so that historical records and the
# operator diagnostic, which builds its own fixed snapshots, keep their meaning.
SHARED_SELECTION = "shared_active"
Selection = Literal["default", "admin_override", "shared_active"]
_SELECTIONS = ("default", "admin_override", SHARED_SELECTION)
MAX_SELECTION_EPOCH = 2**62


def require_selection_epoch(value: object) -> int:
    """A shared-selection epoch: a real int in [1, 2**62]; never a bool."""
    if type(value) is not int or not 1 <= value <= MAX_SELECTION_EPOCH:
        raise GenerationProfileError("a selection epoch must be an integer of at least 1")
    return value


@dataclass(frozen=True)
class AnswerGenerationSnapshot:
    """Everything one answer or helper job is generated and verified with,
    fixed up front.

    Built once per admitted /ask answer, before anything streams, and once per
    admitted helper job, from the shared selection captured at admission. It
    holds values, not a reference to the registry or the selection, so a later
    switch or registry change cannot alter an answer or job already in
    progress.

    ``expected_digest`` is the registry's pin for this profile.
    ``observed_digest`` is what the installed-model lookup reported while
    building THIS snapshot, and is None whenever no lookup was made (the
    unqualified control). The two are never interchangeable.

    ``selection_epoch`` is the shared selection's epoch the job was admitted
    under (``selection`` "shared_active"); it is None only for the historical
    selection values, which carry no shared epoch.

    ``verifier`` is derived from this same profile and identity when the
    snapshot is built; ``helper_call`` derives a helper call from the values
    held here. Neither consults the registry.

    ``runtime_key`` is the runtime every call of this job is sent to: the
    answer, its verifier calls and every helper call. It is bound here, with
    the model, and never re-resolved from a request or a later selection.
    """

    profile_id: str
    model_tag: str
    display_name: str
    options: GenerationOptions
    keep_alive: str
    think: ThinkValue
    answer_read_timeout_s: float
    helper_read_timeout_s: float
    selection: Selection
    expected_digest: Optional[str]
    observed_digest: Optional[str]
    verifier: FixedModelCall
    selection_epoch: Optional[int] = None
    runtime_key: str = PRIMARY_RUNTIME
    experimental: bool = False
    # None: the historical helper derivation (this snapshot's think setting
    # with HELPER_NUM_PREDICT).
    helper_policy: Optional[RolePolicy] = None

    def __post_init__(self) -> None:
        if self.selection not in _SELECTIONS:
            raise GenerationProfileError("unknown model selection")
        if self.selection == SHARED_SELECTION:
            require_selection_epoch(self.selection_epoch)
        elif self.selection_epoch is not None:
            raise GenerationProfileError("only a shared selection carries an epoch")
        require_think_value("think", self.think)
        require_runtime_key("runtime_key", self.runtime_key)
        if not isinstance(self.experimental, bool):
            raise GenerationProfileError("experimental must be a bool")
        if self.helper_policy is not None and not isinstance(self.helper_policy, RolePolicy):
            raise GenerationProfileError("helper_policy must be a RolePolicy")
        if not isinstance(self.verifier, FixedModelCall) or (
            self.verifier.profile_id != self.profile_id
            or self.verifier.model_tag != self.model_tag
            or self.verifier.runtime_key != self.runtime_key
        ):
            # The verifier is this job's own model on this job's own runtime.
            raise GenerationProfileError("the verifier must be derived from this snapshot's profile")

    def ollama_payload(self, prompt: str, *, stream: bool = True) -> dict:
        return build_generate_payload(
            self.model_tag,
            prompt,
            self.options,
            self.keep_alive,
            stream=stream,
            think=self.think,
        )

    def public_metadata(self) -> dict:
        """What the client is told produced this answer. An experimental
        profile says so; no runtime, endpoint or digest is ever included."""
        metadata = {"profile_id": self.profile_id, "model": self.model_tag}
        if self.experimental:
            metadata["experimental"] = True
        return metadata

    def requested_settings(self) -> dict:
        settings = {"options": self.options.as_ollama_options(), "keep_alive": self.keep_alive}
        if self.think is not None:
            settings["think"] = self.think
        return settings

    def helper_role(self) -> RolePolicy:
        if self.helper_policy is not None:
            return self.helper_policy
        return RolePolicy(think=self.think, num_predict=HELPER_NUM_PREDICT)

    def helper_call(self, temperature: float) -> FixedModelCall:
        """One bounded background helper call (insights, KB Q&A import) from
        this snapshot's profile, identity and runtime."""
        role = self.helper_role()
        return _role_call(
            profile_id=self.profile_id,
            model_tag=self.model_tag,
            num_ctx=self.options.num_ctx,
            keep_alive=self.keep_alive,
            think=role.think,
            expected_digest=self.expected_digest,
            observed_digest=self.observed_digest,
            temperature=temperature,
            num_predict=role.num_predict,
            read_timeout_s=self.helper_read_timeout_s,
            runtime_key=self.runtime_key,
        )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


INELIGIBLE_NOT_QUALIFIED = "not_qualified"
INELIGIBLE_EXPERIMENTAL_DISABLED = "experimental_disabled"
INELIGIBLE_RUNTIME_NOT_CONFIGURED = "runtime_not_configured"


@dataclass(frozen=True)
class ModelRegistry:
    """The registered profiles, the default, and the deployment facts that
    decide which of them may be selected. There is no separate verifier or
    helper profile: those role calls are derived from whichever profile a
    snapshot resolved.

    ``experimental_enabled`` (server-side, default off) and
    ``configured_runtimes`` (the runtime keys this server has an endpoint
    for) are fixed when the registry is built, at import, from server
    configuration -- never from a request. A profile whose runtime is not
    configured is refused, but stays registered, so a persisted selection
    naming it is reported (and can be switched away from) rather than
    silently replaced."""

    profiles: Tuple[ModelProfile, ...]
    default_profile_id: str
    experimental_enabled: bool = False
    configured_runtimes: FrozenSet[str] = frozenset({PRIMARY_RUNTIME})
    _by_id: Mapping[str, ModelProfile] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        profiles = tuple(self.profiles)
        if not profiles or not all(isinstance(p, ModelProfile) for p in profiles):
            raise GenerationProfileError("a registry needs at least one ModelProfile")
        if not isinstance(self.experimental_enabled, bool):
            raise GenerationProfileError("experimental_enabled must be a bool")
        runtimes = frozenset(self.configured_runtimes)
        for key in runtimes:
            require_runtime_key("configured runtime", key)
        if PRIMARY_RUNTIME not in runtimes:
            raise GenerationProfileError("the primary runtime is always configured")
        by_id = {}
        for profile in profiles:
            if profile.profile_id in by_id:
                raise GenerationProfileError(f"duplicate profile id {profile.profile_id!r}")
            by_id[profile.profile_id] = profile
        default = by_id.get(self.default_profile_id)
        if default is None:
            raise GenerationProfileError("default_profile_id is not a registered profile")
        if not default.selectable:
            raise GenerationProfileError(
                "the default profile requires qualification and its evidence is incomplete"
            )
        if default.experimental or default.runtime_key != PRIMARY_RUNTIME:
            raise GenerationProfileError("the default profile is a primary-runtime, non-experimental profile")
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "configured_runtimes", runtimes)
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))

    def get(self, profile_id: str) -> Optional[ModelProfile]:
        return self._by_id.get(profile_id)

    @property
    def default(self) -> ModelProfile:
        return self._by_id[self.default_profile_id]

    def runtime_configured(self, runtime_key: object) -> bool:
        return isinstance(runtime_key, str) and runtime_key in self.configured_runtimes

    def ineligible_reason(self, profile: ModelProfile) -> Optional[str]:
        """None when ``profile`` may be the shared selection on this server;
        otherwise a fixed reason category."""
        if not isinstance(profile, ModelProfile) or self._by_id.get(profile.profile_id) is not profile:
            return INELIGIBLE_NOT_QUALIFIED
        if not self.runtime_configured(profile.runtime_key):
            return INELIGIBLE_RUNTIME_NOT_CONFIGURED
        if profile.experimental:
            return None if self.experimental_enabled else INELIGIBLE_EXPERIMENTAL_DISABLED
        return None if profile.selectable else INELIGIBLE_NOT_QUALIFIED

    def is_selectable(self, profile: ModelProfile) -> bool:
        return self.ineligible_reason(profile) is None

    def admission_policy(self, profile_id: object) -> Optional[Tuple[Optional[int], Optional[float], Optional[float]]]:
        """(max_concurrent_jobs, answer_job_budget_s, helper_job_budget_s)
        for a registered profile, or None for an unknown id. Pure: the ledger
        applies it while that profile is the shared selection."""
        profile = self._by_id.get(profile_id) if isinstance(profile_id, str) else None
        if profile is None:
            return None
        return (profile.max_concurrent_jobs, profile.answer_job_budget_s, profile.helper_job_budget_s)

    def generation_identities(self) -> Tuple[Tuple[str, str, Optional[str]], ...]:
        """(profile_id, model_tag, expected_digest) for every registered
        generation profile: what the switch residency barrier treats as a
        generation model on any runtime (an embedder is never listed)."""
        return tuple((p.profile_id, p.model_tag, p.expected_digest) for p in self.profiles)


# Every Gemma gate is recorded and reviewed: think:false accepted with thinking
# contained in a bounded smoke; the pinned model resident with the embedder
# while two overlapping answer-plus-verifier jobs ran inside the application
# job budget; and six-fixture answer quality, checked against the frozen
# source excerpts beside a Llama control. Selectable means an
# administrator MAY switch to it through the coordinator. It changes neither
# the registry default nor any persisted selection.
GEMMA_CANDIDATE_EVIDENCE = Qualification(
    think_false_verified=True,
    combined_capacity_verified=True,
    quality_verified=True,
    evidence_ref=(
        "Reviewed release record diagnostics/GEMMA_RELEASE_QUALIFICATION_20261010.md "
        "for installed gemma4:31b (digest 6316f0629137b426...ff47e7) on Ollama "
        "0.30.8: capacity probe gcp-40002734529445dea22bc7c110a88d4d (resident "
        "with mxbai-embed-large:335m; two overlapping C04/C05 answer-plus-verifier "
        "jobs); quality runs gq-eaec840f08924dd38218e87dedcbea1c (C01-C06) and "
        "gq-21958b73c99146198dc397fc406af74f (C01/C05 repeat) beside llama_control "
        "runs gq-0d6f1de739934be0bbc32ff99124b63b and "
        "gq-db4735babd264d02a1eedfd50ec4f55b; earlier think:false smoke 102.527 s "
        "total, 91.783 s load. Short frozen-evidence fixtures only: not full-length "
        "answers, retrieval, upload throughput or website latency."
    ),
)


_DGX_MODEL_TESTS = "/home/haihua/synergereader-model-tests"
_SMOKE_SCOPE = (
    "Operator-reported DGX runtime smoke, 19 October 2026: one synthetic invoice calculation "
    "(BALANCE=200.00 USD returned in the native response field, separate thinking field emitted, "
    "done_reason=stop) on the custom Ollama 0.32.12 / CUDA 12.1 / SM70 image "
    f"{COMPAT_RUNTIME_TESTED_IMAGE_ID} (runtime build {_DGX_MODEL_TESTS}/"
    "cuda121_build_20261019T155454Z_2Sxha4) at a 2,048-token context. Not reproduced by offline "
    "tests. No quality, capacity, vision, application-context or latency gate."
)

QWEN38_RUNTIME_SMOKE = RuntimeSmokeEvidence(
    tested_think=(True,),
    tested_num_ctx=2048,
    runtime_version=COMPAT_RUNTIME_OLLAMA_VERSION,
    runtime_image_id=COMPAT_RUNTIME_TESTED_IMAGE_ID,
    verdict="PASS",
    evidence_ref=(
        f"{_DGX_MODEL_TESTS}/qwen38_cuda121_generation_20261019T161846Z_zpCMYz for qwen3.8:27b "
        "(digest aaee06c39dcf2437...e509097813): think=true, 100% GPU residency (20.42 GiB), "
        "11.69 s request including a 9.28 s load. " + _SMOKE_SCOPE
    ),
)
NEMOTRON35_RUNTIME_SMOKE = RuntimeSmokeEvidence(
    tested_think=(True,),
    tested_num_ctx=2048,
    runtime_version=COMPAT_RUNTIME_OLLAMA_VERSION,
    runtime_image_id=COMPAT_RUNTIME_TESTED_IMAGE_ID,
    verdict="PASS",
    evidence_ref=(
        f"{_DGX_MODEL_TESTS}/nemotron35_cuda121_generation_20261019T164233Z_tMMcgC for "
        "nemotron-3.5-lightning:30b (digest e6905fffe2e2d9bb...e2f17f4dc): think=true, 100% GPU "
        "residency (23.65 GiB), 16.65 s request including a 12.13 s load (the separate 2,257.28 s "
        "download is cached). " + _SMOKE_SCOPE
    ),
)
MISTRAL35_RUNTIME_SMOKE = RuntimeSmokeEvidence(
    tested_think=("high",),
    tested_num_ctx=2048,
    runtime_version=COMPAT_RUNTIME_OLLAMA_VERSION,
    runtime_image_id=COMPAT_RUNTIME_TESTED_IMAGE_ID,
    verdict="REVIEW_REQUIRED",
    review_note=(
        "Original verdict preserved: every functional check passed; only the original 99% "
        "GPU-residency target failed (91.27%, 70.35 of 77.08 GiB). The model ran from GPU and CPU "
        "memory; its 180.38 s cold load exceeds Gemma's 150 s per-read timeout."
    ),
    evidence_ref=(
        f"{_DGX_MODEL_TESTS}/mistral35_cuda121_generation_20261019T163014Z_s56gYq for "
        "mistral-medium-3.5:128b (digest 0341632adb051bad...931ab78619): think=\"high\", "
        "OLLAMA_SCHED_SPREAD=1 over GPUs 0, 1 and 3, 200.72 s request including a 180.38 s load. "
        + _SMOKE_SCOPE
    ),
)

EXPERIMENTAL_ANSWER_OPTIONS = GenerationOptions(
    temperature=ANSWER_TEMPERATURE,
    num_ctx=ANSWER_NUM_CTX,
    num_predict=EXPERIMENTAL_ANSWER_NUM_PREDICT,
)


def _experimental_profile(
    *,
    profile_id: str,
    model_tag: str,
    display_name: str,
    digest: str,
    think: ThinkValue,
    evidence: RuntimeSmokeEvidence,
    read_timeout_s: float,
    helper_read_timeout_s: float,
    job_budget_s: Optional[float] = None,
    verifier_num_predict: int = EXPERIMENTAL_VERIFIER_NUM_PREDICT,
    helper_num_predict: int = EXPERIMENTAL_HELPER_NUM_PREDICT,
) -> ModelProfile:
    return ModelProfile(
        profile_id=profile_id,
        model_tag=model_tag,
        display_name=display_name,
        answer_options=EXPERIMENTAL_ANSWER_OPTIONS,
        keep_alive=ANSWER_KEEP_ALIVE,
        answer_read_timeout_s=read_timeout_s,
        verifier_read_timeout_s=read_timeout_s,
        helper_read_timeout_s=helper_read_timeout_s,
        requires_qualification=False,
        expected_digest=digest,
        think=think,
        runtime_key=COMPAT_RUNTIME,
        experimental=True,
        runtime_smoke=evidence,
        verifier_policy=RolePolicy(think=think, num_predict=verifier_num_predict),
        helper_policy=RolePolicy(think=think, num_predict=helper_num_predict),
        max_concurrent_jobs=EXPERIMENTAL_MAX_CONCURRENT_JOBS,
        answer_job_budget_s=job_budget_s,
        helper_job_budget_s=job_budget_s,
    )


def experimental_profiles() -> Tuple[ModelProfile, ...]:
    """The three experimental profiles, served by the compatibility runtime.
    Registered always; selectable only where the deployment enables them."""
    return (
        _experimental_profile(
            profile_id=QWEN38_EXPERIMENTAL,
            model_tag=QWEN38_TAG,
            display_name="Qwen3.8 27B",
            digest=QWEN38_DIGEST,
            think=True,
            evidence=QWEN38_RUNTIME_SMOKE,
            read_timeout_s=EXPERIMENTAL_READ_TIMEOUT_S,
            helper_read_timeout_s=EXPERIMENTAL_HELPER_READ_TIMEOUT_S,
        ),
        _experimental_profile(
            profile_id=NEMOTRON35_EXPERIMENTAL,
            model_tag=NEMOTRON35_TAG,
            display_name="Nemotron 3.5 Lightning 30B",
            digest=NEMOTRON35_DIGEST,
            think=True,
            evidence=NEMOTRON35_RUNTIME_SMOKE,
            read_timeout_s=EXPERIMENTAL_READ_TIMEOUT_S,
            helper_read_timeout_s=EXPERIMENTAL_HELPER_READ_TIMEOUT_S,
            verifier_num_predict=NEMOTRON35_VERIFIER_NUM_PREDICT,
            helper_num_predict=NEMOTRON35_HELPER_NUM_PREDICT,
        ),
        _experimental_profile(
            profile_id=MISTRAL35_EXPERIMENTAL,
            model_tag=MISTRAL35_TAG,
            display_name="Mistral Medium 3.5 128B",
            digest=MISTRAL35_DIGEST,
            think="high",
            evidence=MISTRAL35_RUNTIME_SMOKE,
            read_timeout_s=MISTRAL35_READ_TIMEOUT_S,
            helper_read_timeout_s=MISTRAL35_HELPER_READ_TIMEOUT_S,
            job_budget_s=MISTRAL35_JOB_BUDGET_S,
        ),
    )


def default_model_registry(
    *,
    experimental_enabled: bool = False,
    configured_runtimes: FrozenSet[str] = frozenset({PRIMARY_RUNTIME}),
) -> ModelRegistry:
    """The production registry: Llama is the default; Gemma is qualified; the
    three experimental profiles are registered on the compatibility runtime.

    The default only seeds the shared selection the first time ownership
    creates it; after that the persisted selection decides which profile
    answers, and changing it is an administrator's coordinated switch.
    ``experimental_enabled`` and ``configured_runtimes`` are the deployment's
    server-side facts (main.py reads them once, at import); with the defaults
    no experimental profile can be selected."""
    return ModelRegistry(
        experimental_enabled=experimental_enabled,
        configured_runtimes=configured_runtimes,
        profiles=(
            ModelProfile(
                profile_id=LLAMA_CONTROL,
                model_tag=LLAMA_CONTROL_TAG,
                display_name="Llama 3.1 8B",
                answer_options=ANSWER_OPTIONS,
                keep_alive=ANSWER_KEEP_ALIVE,
                answer_read_timeout_s=LLAMA_ANSWER_READ_TIMEOUT_S,
                verifier_read_timeout_s=LLAMA_VERIFIER_READ_TIMEOUT_S,
                helper_read_timeout_s=LLAMA_HELPER_READ_TIMEOUT_S,
                requires_qualification=False,
                expected_digest=LLAMA_CONTROL_DIGEST,
            ),
            ModelProfile(
                profile_id=GEMMA_CANDIDATE,
                model_tag=GEMMA_CANDIDATE_TAG,
                display_name="Gemma 4 31B",
                answer_options=ANSWER_OPTIONS,
                keep_alive=ANSWER_KEEP_ALIVE,
                answer_read_timeout_s=GEMMA_ANSWER_READ_TIMEOUT_S,
                verifier_read_timeout_s=GEMMA_VERIFIER_READ_TIMEOUT_S,
                helper_read_timeout_s=GEMMA_HELPER_READ_TIMEOUT_S,
                requires_qualification=True,
                expected_digest=GEMMA_CANDIDATE_DIGEST,
                think=False,
                qualification=GEMMA_CANDIDATE_EVIDENCE,
            ),
            *experimental_profiles(),
        ),
        default_profile_id=LLAMA_CONTROL,
    )


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def _snapshot(
    profile: ModelProfile,
    selection: Selection,
    observed_digest: Optional[str],
    selection_epoch: Optional[int] = None,
) -> AnswerGenerationSnapshot:
    # The verifier is this exact profile, identity and runtime with the
    # bounded verifier role policy, fixed here with the answer: verification
    # never rebuilds it from whatever the registry holds later.
    role = profile.verifier_role()
    verifier = _role_call(
        profile_id=profile.profile_id,
        model_tag=profile.model_tag,
        num_ctx=profile.answer_options.num_ctx,
        keep_alive=profile.keep_alive,
        think=role.think,
        expected_digest=profile.expected_digest,
        observed_digest=observed_digest,
        temperature=VERIFIER_TEMPERATURE,
        num_predict=role.num_predict,
        read_timeout_s=profile.verifier_read_timeout_s,
        runtime_key=profile.runtime_key,
    )
    return AnswerGenerationSnapshot(
        profile_id=profile.profile_id,
        model_tag=profile.model_tag,
        display_name=profile.display_name,
        options=profile.answer_options,
        keep_alive=profile.keep_alive,
        think=profile.think,
        answer_read_timeout_s=float(profile.answer_read_timeout_s),
        helper_read_timeout_s=float(profile.helper_read_timeout_s),
        selection=selection,
        expected_digest=profile.expected_digest,
        observed_digest=observed_digest,
        verifier=verifier,
        selection_epoch=selection_epoch,
        runtime_key=profile.runtime_key,
        experimental=profile.experimental,
        helper_policy=profile.helper_policy,
    )


def _refuse_ineligible(registry: ModelRegistry, profile: ModelProfile) -> None:
    reason = registry.ineligible_reason(profile)
    if reason is None:
        return
    if reason == INELIGIBLE_RUNTIME_NOT_CONFIGURED:
        raise ModelRuntimeUnavailableError("the model profile's runtime is not configured on this server")
    if reason == INELIGIBLE_EXPERIMENTAL_DISABLED:
        raise ExperimentalProfileDisabledError("experimental model profiles are disabled on this server")
    raise ProfileNotQualifiedError("model profile is not qualified for selection")


def select_shared_profile(registry: ModelRegistry, profile_id: object) -> ModelProfile:
    """The registered, selectable profile a shared selection names, or a
    ModelSelectionError. Pure: no lookup, no fallback to any other profile or
    runtime -- an unknown, unqualified, disabled experimental or unconfigured
    runtime selection refuses."""
    profile = registry.get(profile_id) if isinstance(profile_id, str) else None
    if profile is None:
        raise UnknownModelProfileError("unknown model profile")
    _refuse_ineligible(registry, profile)
    return profile


def check_installed_identity(
    profile: ModelProfile, installed_digest: Callable[[str], Optional[str]]
) -> str:
    """Read the installed digest of the profile's exact tag and require it to
    equal the profile's pinned digest; return what was observed.

    Read-only: ``installed_digest`` is an inventory lookup, and the caller
    binds it to the profile's OWN runtime (``profile.runtime_key``): a tag
    present on another runtime proves nothing about this one. An absent tag
    or a different digest is a mismatch; a failed or malformed lookup is
    unavailable. A profile without a pin cannot be verified."""
    if profile.expected_digest is None:
        raise ModelIdentityUnavailableError("model profile has no pinned identity")
    try:
        observed = installed_digest(profile.model_tag)
    except Exception:
        raise ModelIdentityUnavailableError("installed model identity unavailable") from None
    if observed is None:
        raise ModelIdentityMismatchError("expected model is not installed")
    observed = normalize_digest(observed)
    if observed is None:
        raise ModelIdentityUnavailableError("installed model identity is unreadable")
    if observed != profile.expected_digest:
        raise ModelIdentityMismatchError("installed model digest differs from the expected digest")
    return observed


def shared_generation_snapshot(
    profile: ModelProfile,
    *,
    epoch: int,
    observed_digest: Optional[str],
    registry: Optional[ModelRegistry] = None,
) -> AnswerGenerationSnapshot:
    """The snapshot for one job admitted under the shared selection
    ``(profile, epoch)``.

    ``observed_digest`` must be what ``check_installed_identity`` returned for
    THIS job, on the profile's own runtime, when the profile requires an
    identity check (qualified or experimental), and None otherwise: the
    unqualified control makes no per-job inventory call, and its pinned digest
    is never copied into the observed one. The verifier and every helper call
    are derived from this same snapshot, on its runtime.

    ``registry`` is the deployment registry; when given, the profile must be
    eligible there. An experimental profile is refused without it, so no
    caller can build an experimental snapshot while the deployment has not
    enabled experimental profiles."""
    if not isinstance(profile, ModelProfile):
        raise GenerationProfileError("expected a registered ModelProfile")
    require_selection_epoch(epoch)
    if registry is not None:
        if not isinstance(registry, ModelRegistry):
            raise GenerationProfileError("registry must be a ModelRegistry")
        _refuse_ineligible(registry, profile)
    elif profile.experimental:
        raise ExperimentalProfileDisabledError("an experimental profile needs the deployment registry")
    elif not profile.selectable:
        raise ProfileNotQualifiedError("model profile is not qualified for selection")
    if profile.requires_identity_check:
        if observed_digest is None or observed_digest != profile.expected_digest:
            raise ModelIdentityUnavailableError("this profile needs its observed identity")
    elif observed_digest is not None:
        raise GenerationProfileError("the unqualified control records no observed identity")
    return _snapshot(profile, SHARED_SELECTION, observed_digest, epoch)
