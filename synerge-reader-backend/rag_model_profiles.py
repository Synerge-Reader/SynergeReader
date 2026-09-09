"""Immutable, validated embedding/generation profiles for SynergeReader RAG.

This module is a pure configuration/value-object layer:

- No network I/O, no filesystem I/O, and no environment access happen at
  import time.
- ``load_dotenv`` is never called here, and no backend ``.env`` file is ever
  read by this module. Configuration functions accept an explicit
  ``Mapping[str, str]``; only when the caller omits it do they fall back to
  ``os.environ`` at *call* time (the application's composition boundary is
  responsible for populating that environment however it sees fit, e.g. via
  Compose or its own ``main.py``-level ``load_dotenv()``).
- Nothing in main.py, dbSetup.py, or routing behavior is touched or wired up
  by this module.

Profiles are looked up/derived through one of three entry points:

- ``default_embedding_profile()`` — the one documented default proposal,
  itself just ``known_embedding_profile()`` for a pinned registry key.
- ``known_embedding_profile(provider, model)`` — a registry-backed profile
  for a model whose preprocessing is fully specified in
  ``KNOWN_EMBEDDING_MODELS``.
- ``explicit_unverified_embedding_profile(...)`` — a fully-specified,
  caller-supplied profile for a model whose preprocessing the registry does
  not (yet) know, or for a model outside the registry entirely. A known
  model's ``dimension`` must still match the registry or this fails closed.

``resolve_embedding_profile(config)`` ties these together for a
``Mapping[str, str]``-shaped configuration source (see its docstring).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Literal, Mapping, Optional, Tuple

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ProfileError(ValueError):
    """Base class for all profile configuration/validation failures."""


class InvalidProfileFieldError(ProfileError):
    """A profile field is missing, the wrong type, or out of range."""


class UnknownEmbeddingModelError(ProfileError):
    """(provider, model) is not present in the known-model registry."""


class UnverifiedPreprocessingError(ProfileError):
    """The registry has no known-good query/document prefixes for this model."""


class EmbeddingDimensionMismatchError(ProfileError):
    """A configured dimension does not match the known model's registry entry."""


class PartialEmbeddingProfileOverrideError(ProfileError):
    """Some, but not all, of the five override fields were supplied, or the
    unverified acknowledgement was supplied without them."""


class UnacknowledgedUnverifiedProfileError(ProfileError):
    """All five override fields were supplied, but the unverified-pending-
    Phase-H acknowledgement is absent or incorrect."""


class UnsupportedGenerationProviderError(ProfileError):
    """The requested generation provider is not a supported, local provider."""


# --------------------------------------------------------------------------
# Verification status
# --------------------------------------------------------------------------

VerificationStatus = Literal["verified", "unverified_pending_phase_h"]

# Defined ahead of the value objects below because EmbeddingProfile's
# verification_status field default references UNVERIFIED_PENDING_PHASE_H at
# class-definition time.
VERIFIED: VerificationStatus = "verified"
UNVERIFIED_PENDING_PHASE_H: VerificationStatus = "unverified_pending_phase_h"


# --------------------------------------------------------------------------
# Profile id derivation
# --------------------------------------------------------------------------

# A namespace for the id scheme itself, so a future incompatible change to
# the derivation algorithm (field set, serialization, hash) can mint a new
# scheme string rather than silently colliding with v1 ids.
_PROFILE_ID_SCHEME = "synerge.profile.v1"


def _derive_profile_id(domain: str, *operative_fields: object) -> str:
    """Derive a full, domain-separated, deterministic profile id.

    Serialization: the list ``[_PROFILE_ID_SCHEME, domain, *operative_fields]``
    (in that exact order) is JSON-encoded with ``separators=(",", ":")`` (no
    incidental whitespace) and ``ensure_ascii=True`` (non-ASCII text is
    escaped rather than emitted as raw bytes, so the same logical value
    always serializes identically regardless of platform text encoding). The
    UTF-8 bytes of that JSON string are hashed with SHA-256; the full hex
    digest (64 characters) is the profile id — it is not truncated.

    ``domain`` (e.g. ``"embedding"`` vs. ``"generation"``) separates id
    spaces so two different kinds of profile can never collide even if they
    happen to share the same provider/model and no other operative fields.
    ``operative_fields`` must contain only fields that are part of the
    profile's identity — verification metadata and other non-operative
    fields must never be passed here.

    Because the encoding is a JSON list (not string concatenation), field
    values are unambiguously delimited even if they contain punctuation or
    control characters — there is no separator-collision risk.
    """
    canonical = json.dumps(
        [_PROFILE_ID_SCHEME, domain, *operative_fields],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Field validators
# --------------------------------------------------------------------------


def _validate_nonempty_str(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidProfileFieldError(f"{name} must be a non-empty string; got {value!r}")
    return value


def _validate_str(name: str, value: object) -> str:
    # Allows "" (e.g. a verified-empty document prefix) but not non-str values.
    if not isinstance(value, str):
        raise InvalidProfileFieldError(
            f"{name} must be a string; got {type(value).__name__}"
        )
    return value


def _validate_positive_int(name: str, value: object) -> int:
    # bool is a subclass of int in Python; exclude it explicitly so
    # True/False can never silently pass as 1/0.
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidProfileFieldError(
            f"{name} must be a positive integer; got {value!r}"
        )
    if value <= 0:
        raise InvalidProfileFieldError(
            f"{name} must be a positive integer; got {value!r}"
        )
    return value


# --------------------------------------------------------------------------
# Frozen-dataclass construction helper
# --------------------------------------------------------------------------


def _freeze_set(instance: object, **fields: object) -> None:
    """The single place this module reaches into a frozen dataclass.

    Centralizes ``object.__setattr__`` so it is never scattered across
    ``__post_init__`` methods or profile-construction functions. Used both
    to finish constructing a profile after validation (inside
    ``__post_init__``) and to let a registry-backed constructor (e.g.
    ``known_embedding_profile``) populate metadata — such as
    ``verification_status`` — that the public constructor does not accept.
    """
    for name, value in fields.items():
        object.__setattr__(instance, name, value)


# --------------------------------------------------------------------------
# Value objects
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingProfile:
    """An atomic, validated embedding configuration.

    ``profile_id`` is always derived from ``provider``/``model``/``dimension``/
    ``query_prefix``/``document_prefix`` — it cannot be passed to the
    constructor. ``verification_status`` is visible metadata (defaults to
    ``UNVERIFIED_PENDING_PHASE_H``) that is likewise not constructor-settable
    and is deliberately excluded from both the profile id and dataclass
    equality/hashing (``compare=False``): it describes our knowledge about a
    profile, not the operative, vector-producing profile itself. Two
    profiles with identical operative fields but different verification
    status are the *same* profile, just described with different confidence
    — they compare equal and hash equal.
    """

    provider: str
    model: str
    dimension: int
    query_prefix: str
    document_prefix: str
    verification_status: VerificationStatus = field(
        init=False, default=UNVERIFIED_PENDING_PHASE_H, compare=False
    )
    profile_id: str = field(init=False)

    def __post_init__(self) -> None:
        provider = _validate_nonempty_str("provider", self.provider)
        model = _validate_nonempty_str("model", self.model)
        dimension = _validate_positive_int("dimension", self.dimension)
        query_prefix = _validate_str("query_prefix", self.query_prefix)
        document_prefix = _validate_str("document_prefix", self.document_prefix)

        # Validation must complete before any field is (re)written, so a
        # rejected profile never partially exists. verification_status is
        # deliberately left untouched here (it already carries its dataclass
        # default) and is deliberately excluded from the id below.
        _freeze_set(
            self,
            provider=provider,
            model=model,
            dimension=dimension,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            profile_id=_derive_profile_id(
                "embedding", provider, model, dimension, query_prefix, document_prefix
            ),
        )


SUPPORTED_GENERATION_PROVIDERS = frozenset({"ollama"})


@dataclass(frozen=True)
class GenerationProfile:
    """An atomic, validated, server-controlled, local-only generation
    configuration. ``profile_id`` is derived from ``provider`` and ``model``.
    """

    provider: str
    model: str
    profile_id: str = field(init=False)

    def __post_init__(self) -> None:
        provider = _validate_nonempty_str("provider", self.provider)
        if provider not in SUPPORTED_GENERATION_PROVIDERS:
            raise UnsupportedGenerationProviderError(
                f"Unsupported generation provider {provider!r}; supported "
                f"providers are {sorted(SUPPORTED_GENERATION_PROVIDERS)}. "
                "External providers (e.g. OpenRouter) are not permitted for "
                "the server-controlled generation profile."
            )
        model = _validate_nonempty_str("model", self.model)

        _freeze_set(
            self,
            provider=provider,
            model=model,
            profile_id=_derive_profile_id("generation", provider, model),
        )


# --------------------------------------------------------------------------
# Known-model registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KnownEmbeddingModel:
    """A registry entry describing what is known about one (provider, model).

    ``query_prefix``/``document_prefix`` are ``None`` when the correct
    preprocessing recipe for this model is not yet determined — that is
    distinct from a verified/proposed empty string, and callers must not
    treat ``None`` as if it meant "no prefix". A model with ``None`` prefixes
    can only be used via ``explicit_unverified_embedding_profile``, where the
    caller supplies both prefixes itself.
    """

    provider: str
    model: str
    expected_dimension: int
    query_prefix: Optional[str]
    document_prefix: Optional[str]
    verification_status: VerificationStatus
    notes: str = ""


_KNOWN_EMBEDDING_MODELS: Dict[Tuple[str, str], KnownEmbeddingModel] = {
    ("ollama", "mxbai-embed-large:335m"): KnownEmbeddingModel(
        provider="ollama",
        model="mxbai-embed-large:335m",
        expected_dimension=1024,
        query_prefix="Represent this sentence for searching relevant passages: ",
        document_prefix="",
        verification_status=UNVERIFIED_PENDING_PHASE_H,
        notes=(
            "Query prefix and empty document prefix are a proposed "
            "configuration (per the model's documented usage), not yet "
            "empirically verified against production retrieval quality."
        ),
    ),
    ("ollama", "bge-m3:567m"): KnownEmbeddingModel(
        provider="ollama",
        model="bge-m3:567m",
        expected_dimension=1024,
        query_prefix=None,
        document_prefix=None,
        verification_status=UNVERIFIED_PENDING_PHASE_H,
        notes=(
            "Query/document preprocessing recipe is unverified. Callers must "
            "use an explicit unverified profile supplying both prefixes "
            "rather than relying on an invented default."
        ),
    ),
    ("ollama", "embeddinggemma:latest"): KnownEmbeddingModel(
        provider="ollama",
        model="embeddinggemma:latest",
        expected_dimension=768,
        query_prefix=None,
        document_prefix=None,
        verification_status=UNVERIFIED_PENDING_PHASE_H,
        notes=(
            "This entry is the full-precision (768-d) output only. "
            "EmbeddingGemma's legitimate 512/256/128 Matryoshka variants are "
            "intentionally unsupported in Phase C1: a configured dimension "
            "of 512, 256, or 128 for this model is rejected as a dimension "
            "mismatch against this registry entry until Phase H adds "
            "explicit variant entries for them. Query/document preprocessing "
            "is also unverified."
        ),
    ),
    ("ollama", "nomic-embed-text:v1.5"): KnownEmbeddingModel(
        provider="ollama",
        model="nomic-embed-text:v1.5",
        expected_dimension=768,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        verification_status=UNVERIFIED_PENDING_PHASE_H,
        notes=(
            "Query prefix \"search_query: \" and document prefix "
            "\"search_document: \" come from Nomic Embed Text v1.5's "
            "documented usage recipe. They have not yet been empirically "
            "verified on this project's DGX/Ollama installation, and "
            "retrieval quality remains pending Phase H validation."
        ),
    ),
}

# Read-only view: KNOWN_EMBEDDING_MODELS[...] = ... raises TypeError. Callers
# must go through known_embedding_profile()/explicit_unverified_embedding_profile()
# rather than mutating registry entries in place.
KNOWN_EMBEDDING_MODELS: Mapping[Tuple[str, str], KnownEmbeddingModel] = MappingProxyType(
    _KNOWN_EMBEDDING_MODELS
)


# --------------------------------------------------------------------------
# Registry-backed and explicit profile construction
# --------------------------------------------------------------------------


def known_embedding_profile(provider: str, model: str) -> EmbeddingProfile:
    """Build a profile for a registry model with fully-specified preprocessing.

    The returned profile's ``verification_status`` is populated from the
    registry entry (not left at the constructor's default), so a future
    Phase H upgrade of a registry entry to ``VERIFIED`` is reflected here
    automatically.

    Raises ``UnknownEmbeddingModelError`` if (provider, model) is not
    registered, and ``UnverifiedPreprocessingError`` if the registry has no
    known-good prefixes for it (use ``explicit_unverified_embedding_profile``
    instead, supplying prefixes yourself).
    """
    entry = KNOWN_EMBEDDING_MODELS.get((provider, model))
    if entry is None:
        raise UnknownEmbeddingModelError(
            f"{provider}/{model} is not in the known embedding model "
            "registry; use explicit_unverified_embedding_profile() with "
            "every field supplied instead."
        )
    if entry.query_prefix is None or entry.document_prefix is None:
        raise UnverifiedPreprocessingError(
            f"{provider}/{model} has unverified preprocessing in the "
            "registry (query_prefix/document_prefix are not yet "
            "determined); use explicit_unverified_embedding_profile() and "
            "supply both prefixes explicitly rather than relying on an "
            "invented default."
        )
    profile = EmbeddingProfile(
        provider=entry.provider,
        model=entry.model,
        dimension=entry.expected_dimension,
        query_prefix=entry.query_prefix,
        document_prefix=entry.document_prefix,
    )
    _freeze_set(profile, verification_status=entry.verification_status)
    return profile


def explicit_unverified_embedding_profile(
    *,
    provider: str,
    model: str,
    dimension: int,
    query_prefix: str,
    document_prefix: str,
) -> EmbeddingProfile:
    """Build a fully caller-specified, unverified embedding profile.

    Every atomic field must be supplied explicitly; there are no partial
    defaults. If (provider, model) is a known registry model, the supplied
    ``dimension`` must match the registry's ``expected_dimension`` exactly or
    this fails closed with ``EmbeddingDimensionMismatchError`` — this is what
    blocks e.g. EmbeddingGemma's unsupported Matryoshka dimensions in C1.
    Prefixes may still be overridden even for a known model, since this path
    is explicitly an unverified, caller-owned profile. The returned profile's
    ``verification_status`` is always ``UNVERIFIED_PENDING_PHASE_H`` (the
    ``EmbeddingProfile`` constructor default), regardless of what the
    registry entry (if any) says.
    """
    entry = KNOWN_EMBEDDING_MODELS.get((provider, model))
    if entry is not None and dimension != entry.expected_dimension:
        raise EmbeddingDimensionMismatchError(
            f"{provider}/{model} is a known model with expected dimension "
            f"{entry.expected_dimension}; refusing configured dimension "
            f"{dimension}."
        )
    return EmbeddingProfile(
        provider=provider,
        model=model,
        dimension=dimension,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
    )


# The registry is the sole source of the documented default's model,
# dimension, and prefix values — default_embedding_profile() carries no
# field values of its own, only this key.
_DEFAULT_EMBEDDING_MODEL_KEY: Tuple[str, str] = ("ollama", "mxbai-embed-large:335m")


def default_embedding_profile() -> EmbeddingProfile:
    """The documented default embedding profile proposal."""
    return known_embedding_profile(*_DEFAULT_EMBEDDING_MODEL_KEY)


# --------------------------------------------------------------------------
# Configuration-driven resolution
# --------------------------------------------------------------------------

_EMBEDDING_OVERRIDE_KEYS: Tuple[str, ...] = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_QUERY_PREFIX",
    "EMBEDDING_DOCUMENT_PREFIX",
)
_UNVERIFIED_ACK_KEY = "EMBEDDING_PROFILE_UNVERIFIED_ACK"
_UNVERIFIED_ACK_VALUE = "pending_phase_h"


def _parse_positive_int(name: str, raw: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise InvalidProfileFieldError(
            f"{name} must be a positive integer; got {raw!r}"
        ) from exc
    return _validate_positive_int(name, value)


def resolve_embedding_profile(config: Optional[Mapping[str, str]] = None) -> EmbeddingProfile:
    """Resolve the active embedding profile from a string-keyed configuration.

    ``config`` should be an explicit ``Mapping[str, str]`` (e.g. a plain dict
    built by the application's composition boundary). When omitted, this
    reads ``os.environ`` directly at call time — this module never calls
    ``load_dotenv`` and never reads a backend ``.env`` file itself; whatever
    populated ``os.environ`` (Compose, the shell, or main.py's own
    ``load_dotenv()``) is the composition boundary's responsibility.

    Behavior:
    - If none of ``EMBEDDING_PROVIDER``/``EMBEDDING_MODEL``/
      ``EMBEDDING_DIMENSION``/``EMBEDDING_QUERY_PREFIX``/
      ``EMBEDDING_DOCUMENT_PREFIX`` are present as keys in the mapping, the
      documented default embedding profile is returned — *unless*
      ``EMBEDDING_PROFILE_UNVERIFIED_ACK`` is present on its own, which is
      rejected: the acknowledgement only means something alongside the
      override it acknowledges.
    - If all five keys are present AND ``EMBEDDING_PROFILE_UNVERIFIED_ACK``
      equals exactly ``"pending_phase_h"``, an explicit unverified profile is
      built from them via ``explicit_unverified_embedding_profile`` (which
      still enforces the known-model dimension check). A key may be present
      with an empty string value (e.g. an intentional empty document
      prefix) — presence is judged by key membership, not truthiness, so an
      explicit empty override is never confused with an absent one.
    - If all five keys are present but the acknowledgement is absent or
      incorrect, ``UnacknowledgedUnverifiedProfileError`` is raised.
    - Any other combination — some but not all of the five keys present — is
      rejected as a partial override. This is deliberate: a caller cannot
      silently drift onto an unverified model/preprocessing combination by
      setting only one or two environment variables.
    """
    if config is None:
        config = os.environ

    present_keys = [key for key in _EMBEDDING_OVERRIDE_KEYS if key in config]

    if not present_keys:
        if _UNVERIFIED_ACK_KEY in config:
            raise PartialEmbeddingProfileOverrideError(
                f"{_UNVERIFIED_ACK_KEY} cannot appear without the complete "
                f"five-field override it is meant to acknowledge "
                f"({', '.join(_EMBEDDING_OVERRIDE_KEYS)}). Set all five "
                "together with the acknowledgement, or remove the "
                "acknowledgement to use the default profile."
            )
        return default_embedding_profile()

    if len(present_keys) < len(_EMBEDDING_OVERRIDE_KEYS):
        missing = [key for key in _EMBEDDING_OVERRIDE_KEYS if key not in present_keys]
        raise PartialEmbeddingProfileOverrideError(
            "Partial embedding profile overrides are not allowed. Set all "
            f"of {_EMBEDDING_OVERRIDE_KEYS} together (missing {missing}), or "
            "none of them to use the default profile."
        )

    if config.get(_UNVERIFIED_ACK_KEY, "") != _UNVERIFIED_ACK_VALUE:
        raise UnacknowledgedUnverifiedProfileError(
            "A fully explicit embedding profile override requires "
            f"{_UNVERIFIED_ACK_KEY}={_UNVERIFIED_ACK_VALUE!r} to visibly "
            "acknowledge that it is unverified pending Phase H."
        )

    return explicit_unverified_embedding_profile(
        provider=config["EMBEDDING_PROVIDER"],
        model=config["EMBEDDING_MODEL"],
        dimension=_parse_positive_int("EMBEDDING_DIMENSION", config["EMBEDDING_DIMENSION"]),
        query_prefix=config["EMBEDDING_QUERY_PREFIX"],
        document_prefix=config["EMBEDDING_DOCUMENT_PREFIX"],
    )


def resolve_generation_profile(config: Optional[Mapping[str, str]] = None) -> GenerationProfile:
    """Resolve the active generation profile from a string-keyed configuration.

    Same ``Mapping``/``os.environ`` contract as ``resolve_embedding_profile``.
    ``GENERATION_PROVIDER`` defaults to ``"ollama"`` only when the key is
    absent entirely; if it is present with an empty string, that empty value
    is passed through to validation (which rejects it) rather than silently
    substituted — a caller that explicitly sets an empty provider has made a
    configuration mistake that should fail closed, not fall back quietly.
    ``GENERATION_MODEL`` has no safe universal default and must be set.
    """
    if config is None:
        config = os.environ

    provider = config["GENERATION_PROVIDER"] if "GENERATION_PROVIDER" in config else "ollama"
    model = config.get("GENERATION_MODEL", "")
    if not model:
        raise InvalidProfileFieldError(
            "GENERATION_MODEL must be set to resolve a generation profile; "
            "there is no safe universal default local model name."
        )
    return GenerationProfile(provider=provider, model=model)
