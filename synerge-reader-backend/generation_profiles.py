"""Backend-owned registry of local answer-generation model profiles.

This module is a pure configuration/value-object layer, like
``rag_model_profiles.py``: no network, filesystem, database, or environment
access happens at import time or in any function here. The application's
composition boundary (``main.py``) supplies the two runtime facts a selection
may need -- whether the caller is a trusted administrator, and which digest an
installed model tag currently has -- as callables.

What the registry decides, and what a client can never override:

* which exact local model tags may answer a question at all;
* the native Ollama ``options`` each call sends (sampling, context window,
  output limit), its ``keep_alive``, and how long the answer transport may
  wait on a read;
* whether a ``think`` field is sent, which only happens as ``think: false``
  for a profile whose no-thinking behaviour has been evidenced;
* the fixed, bounded Llama call used for claim verification and for the
  background helper prompts, whichever profile wrote the answer.

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
  ``requires_qualification`` is selectable (explicitly, or as the default)
  only when its combined-capacity and quality evidence are both recorded, and
  every answer it produces is preceded by an installed-digest check.

The unqualified control (``llama_control``) is the existing production model:
it is explicitly marked as not requiring qualification, so ordinary answers
cost no admin lookup and no per-request inventory call. Whether a profile needs
qualification never depends on which profile happens to be the default.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Optional, Tuple

# --------------------------------------------------------------------------
# Profile ids, installed tags and pinned artifact identities
# --------------------------------------------------------------------------

LLAMA_CONTROL = "llama_control"
GEMMA_CANDIDATE = "gemma_candidate"

LLAMA_CONTROL_TAG = "llama3.1:8b"
GEMMA_CANDIDATE_TAG = "gemma4:31b"

# Operator-verified full manifest digests of the installed artifacts. These are
# expected identities (pins), never evidence about a particular answer.
LLAMA_CONTROL_DIGEST = "42182419e9508c30c4b1fe55015f06b65f4ca4b9e28a744be55008d21998a093"
GEMMA_CANDIDATE_DIGEST = "6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7"

# --------------------------------------------------------------------------
# Requested settings
# --------------------------------------------------------------------------

ANSWER_TEMPERATURE = 0.2
ANSWER_NUM_CTX = 8192
ANSWER_NUM_PREDICT = 1200
ANSWER_KEEP_ALIVE = "5m"

# How long the answer transport may wait for one read, including the wait for
# the first byte, which covers a cold model load. Llama keeps the previous 60 s.
# Gemma's bounded smoke took 102.527 s in total, 91.783 s of it loading, so
# 60 s would time out a cold Gemma answer before its first token. 150 s leaves
# margin for that load plus prompt evaluation. This only removes a guaranteed
# premature timeout; it says nothing about acceptable website latency.
LLAMA_ANSWER_READ_TIMEOUT_S = 60.0
GEMMA_ANSWER_READ_TIMEOUT_S = 150.0
_MIN_READ_TIMEOUT_S = 1.0
_MAX_READ_TIMEOUT_S = 600.0

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
    """An answer-model selection was refused. Messages are fixed and safe."""


class OverrideNotPermittedError(ModelSelectionError):
    """An explicit profile was requested by a caller without admin authority."""


class UnknownModelProfileError(ModelSelectionError):
    """The requested profile id is not in the registry."""


class ProfileNotQualifiedError(ModelSelectionError):
    """The profile requires qualification and its evidence is incomplete."""


class ModelIdentityMismatchError(ModelSelectionError):
    """The expected tag is absent, or installed with a different digest."""


class SelectionAuthorityUnavailableError(ModelSelectionError):
    """Admin authority could not be established (the lookup failed)."""


class ModelIdentityUnavailableError(ModelSelectionError):
    """The installed model identity could not be read."""


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


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    model_tag: str
    display_name: str
    answer_options: GenerationOptions
    keep_alive: str
    answer_read_timeout_s: float
    # Explicit policy, never inferred from default status: True means the
    # profile may answer only with complete qualification evidence and only
    # after its installed digest matches ``expected_digest``.
    requires_qualification: bool
    expected_digest: Optional[str] = None
    # None: no think field is sent. False is sent as ``think: false`` and is
    # allowed only once that behaviour is evidenced. Thinking output is never
    # requested, so True is not a valid value.
    think: Optional[bool] = None
    qualification: Optional[Qualification] = None

    def __post_init__(self) -> None:
        _require_match("profile_id", _PROFILE_ID_PATTERN, self.profile_id)
        _require_match("model_tag", _MODEL_TAG_PATTERN, self.model_tag)
        _require_match("keep_alive", _KEEP_ALIVE_PATTERN, self.keep_alive)
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise GenerationProfileError("display_name must be a non-empty string")
        if not isinstance(self.answer_options, GenerationOptions):
            raise GenerationProfileError("answer_options must be GenerationOptions")
        if not _is_real_number(self.answer_read_timeout_s) or not (
            _MIN_READ_TIMEOUT_S <= self.answer_read_timeout_s <= _MAX_READ_TIMEOUT_S
        ):
            raise GenerationProfileError(
                f"answer_read_timeout_s must be a finite number in "
                f"[{_MIN_READ_TIMEOUT_S}, {_MAX_READ_TIMEOUT_S}]"
            )
        if not isinstance(self.requires_qualification, bool):
            raise GenerationProfileError("requires_qualification must be a bool")
        if self.qualification is not None and not isinstance(self.qualification, Qualification):
            raise GenerationProfileError("qualification must be a Qualification")
        expected = None
        if self.expected_digest is not None:
            expected = normalize_digest(self.expected_digest)
            if expected is None:
                raise GenerationProfileError("expected_digest must be a sha256 manifest digest")
        if self.requires_qualification and expected is None:
            raise GenerationProfileError(
                "a profile that requires qualification needs an expected_digest to verify"
            )
        if self.think is not None:
            if self.think is not False:
                raise GenerationProfileError("think may only be None or False")
            if self.qualification is None or not self.qualification.think_false_verified:
                raise GenerationProfileError(
                    "think: false may only be sent once that behaviour is evidenced"
                )
        object.__setattr__(self, "expected_digest", expected)

    @property
    def selectable(self) -> bool:
        """Whether this profile may answer at all, explicitly or as default."""
        if not self.requires_qualification:
            return True
        evidence = self.qualification
        return (
            evidence is not None
            and evidence.combined_capacity_verified
            and evidence.quality_verified
        )


def build_generate_payload(
    model_tag: str,
    prompt: str,
    options: GenerationOptions,
    keep_alive: str,
    *,
    stream: bool,
    think: Optional[bool] = None,
) -> dict:
    """One native ``/api/generate`` request body.

    Sampling, context and output limits travel inside ``options``, where Ollama
    reads them; ``keep_alive`` and ``think`` are top-level request fields.
    Nothing else is sent, so no stale top-level setting can masquerade as
    policy.
    """
    if not isinstance(prompt, str):
        raise GenerationProfileError("prompt must be a string")
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
    """A backend-fixed call that no request can redirect: the verifier and the
    background helpers. It never sends a think field."""

    profile_id: str
    model_tag: str
    options: GenerationOptions
    keep_alive: str

    def ollama_payload(self, prompt: str, *, stream: bool = False) -> dict:
        return build_generate_payload(
            self.model_tag, prompt, self.options, self.keep_alive, stream=stream
        )


Selection = Literal["default", "admin_override"]


@dataclass(frozen=True)
class AnswerGenerationSnapshot:
    """Everything one answer is generated and verified with, fixed up front.

    Built once per /ask request, before anything streams. It holds values, not
    a reference to the registry, so a later selection or registry change
    cannot alter an answer already in progress.

    ``expected_digest`` is the registry's pin for this profile.
    ``observed_digest`` is what the installed-model lookup reported while
    resolving THIS answer, and is None whenever no lookup was made (the
    unqualified control). The two are never interchangeable.
    """

    profile_id: str
    model_tag: str
    display_name: str
    options: GenerationOptions
    keep_alive: str
    think: Optional[bool]
    answer_read_timeout_s: float
    selection: Selection
    expected_digest: Optional[str]
    observed_digest: Optional[str]
    verifier: FixedModelCall

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
        """What the client is told produced this answer."""
        return {"profile_id": self.profile_id, "model": self.model_tag}

    def requested_settings(self) -> dict:
        settings = {"options": self.options.as_ollama_options(), "keep_alive": self.keep_alive}
        if self.think is not None:
            settings["think"] = self.think
        return settings


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRegistry:
    profiles: Tuple[ModelProfile, ...]
    default_profile_id: str
    # The profile whose tag serves verification and helpers. It must be an
    # unqualified-control profile without a think field: those calls are made
    # without a per-call identity check and never send a thinking field.
    verifier_profile_id: str
    _by_id: Mapping[str, ModelProfile] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        profiles = tuple(self.profiles)
        if not profiles or not all(isinstance(p, ModelProfile) for p in profiles):
            raise GenerationProfileError("a registry needs at least one ModelProfile")
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
        verifier = by_id.get(self.verifier_profile_id)
        if verifier is None:
            raise GenerationProfileError("verifier_profile_id is not a registered profile")
        if verifier.think is not None or verifier.requires_qualification:
            raise GenerationProfileError(
                "the verifier profile must be an unqualified control without a think field"
            )
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))

    def get(self, profile_id: str) -> Optional[ModelProfile]:
        return self._by_id.get(profile_id)

    @property
    def default(self) -> ModelProfile:
        return self._by_id[self.default_profile_id]

    def _fixed_call(self, temperature: float, num_predict: int) -> FixedModelCall:
        profile = self._by_id[self.verifier_profile_id]
        return FixedModelCall(
            profile_id=profile.profile_id,
            model_tag=profile.model_tag,
            options=GenerationOptions(
                temperature=temperature,
                num_ctx=profile.answer_options.num_ctx,
                num_predict=num_predict,
            ),
            keep_alive=profile.keep_alive,
        )

    def verifier_call(self) -> FixedModelCall:
        """Fixed Llama verification, for answers from every profile."""
        return self._fixed_call(VERIFIER_TEMPERATURE, VERIFIER_NUM_PREDICT)

    def helper_call(self, temperature: float) -> FixedModelCall:
        """Bounded background helper call (insights, KB Q&A import)."""
        return self._fixed_call(temperature, HELPER_NUM_PREDICT)

    def is_selectable(self, profile: ModelProfile) -> bool:
        return profile.selectable


# Only the evidenced Gemma result is recorded: think:false accepted, thinking
# contained, in one bounded smoke. Capacity and quality remain unverified, so
# the profile stays registered but unselectable.
GEMMA_CANDIDATE_EVIDENCE = Qualification(
    think_false_verified=True,
    combined_capacity_verified=False,
    quality_verified=False,
    evidence_ref=(
        "Operator-reported bounded smoke on installed gemma4:31b (digest "
        "6316f0629137b426...ff47e7): native /api/generate, num_ctx 4096, "
        "num_predict 32, think:false accepted and thinking contained; 102.527 s "
        "total, 91.783 s load. Combined generator/verifier/embedder capacity, "
        "answer quality and website latency are not evidenced."
    ),
)


def default_model_registry() -> ModelRegistry:
    """The production registry: Llama answers by default; Gemma unqualified."""
    return ModelRegistry(
        profiles=(
            ModelProfile(
                profile_id=LLAMA_CONTROL,
                model_tag=LLAMA_CONTROL_TAG,
                display_name="Llama 3.1 8B",
                answer_options=ANSWER_OPTIONS,
                keep_alive=ANSWER_KEEP_ALIVE,
                answer_read_timeout_s=LLAMA_ANSWER_READ_TIMEOUT_S,
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
                requires_qualification=True,
                expected_digest=GEMMA_CANDIDATE_DIGEST,
                think=False,
                qualification=GEMMA_CANDIDATE_EVIDENCE,
            ),
        ),
        default_profile_id=LLAMA_CONTROL,
        verifier_profile_id=LLAMA_CONTROL,
    )


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def _snapshot(
    registry: ModelRegistry,
    profile: ModelProfile,
    selection: Selection,
    observed_digest: Optional[str],
) -> AnswerGenerationSnapshot:
    return AnswerGenerationSnapshot(
        profile_id=profile.profile_id,
        model_tag=profile.model_tag,
        display_name=profile.display_name,
        options=profile.answer_options,
        keep_alive=profile.keep_alive,
        think=profile.think,
        answer_read_timeout_s=float(profile.answer_read_timeout_s),
        selection=selection,
        expected_digest=profile.expected_digest,
        observed_digest=observed_digest,
        verifier=registry.verifier_call(),
    )


def resolve_answer_generation(
    registry: ModelRegistry,
    *,
    requested_profile_id: Optional[str],
    caller_is_admin: Callable[[], bool],
    installed_digest: Callable[[str], Optional[str]],
) -> AnswerGenerationSnapshot:
    """The one answer model a request may use, or a ModelSelectionError.

    * No requested profile: the registry default, with no admin lookup. A
      legacy client model string is not an input here at all -- it can never
      pick a model.
    * A requested profile: admin authority is established first, so a
      non-admin learns nothing about the registry; then the id must exist.
    * Either way the profile must be selectable, and a profile that requires
      qualification answers only after its installed digest equals its
      expected digest. The unqualified control makes no inventory call.
    """
    if requested_profile_id is None:
        profile, selection = registry.default, "default"
    else:
        try:
            is_admin = caller_is_admin()
        except Exception:
            raise SelectionAuthorityUnavailableError("admin authority could not be established") from None
        if is_admin is not True:
            raise OverrideNotPermittedError("model selection requires an administrator")
        profile = registry.get(requested_profile_id)
        if profile is None:
            raise UnknownModelProfileError("unknown model profile")
        selection = "admin_override"

    if not profile.selectable:
        raise ProfileNotQualifiedError("model profile is not qualified for selection")
    if not profile.requires_qualification:
        return _snapshot(registry, profile, selection, None)

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
    return _snapshot(registry, profile, selection, observed)
