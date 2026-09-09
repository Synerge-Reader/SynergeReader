import importlib.util
import os
import sys
import types
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import rag_model_profiles as profiles
from rag_model_profiles import (
    KNOWN_EMBEDDING_MODELS,
    UNVERIFIED_PENDING_PHASE_H,
    VERIFIED,
    EmbeddingDimensionMismatchError,
    EmbeddingProfile,
    GenerationProfile,
    InvalidProfileFieldError,
    PartialEmbeddingProfileOverrideError,
    UnacknowledgedUnverifiedProfileError,
    UnknownEmbeddingModelError,
    UnsupportedGenerationProviderError,
    UnverifiedPreprocessingError,
    default_embedding_profile,
    explicit_unverified_embedding_profile,
    known_embedding_profile,
    resolve_embedding_profile,
    resolve_generation_profile,
)


def _load_isolated_module(module_file: str, real_module_name: str):
    """Execute a fresh copy of a module's source under a private probe name.

    This proves import-time behavior (e.g. "never calls load_dotenv")
    without ever touching the real, already-imported ``real_module_name``
    module object — unlike ``importlib.reload()``, which re-executes a
    module's class/def statements *in place*, minting brand-new class
    objects while every name this test file imported via
    ``from real_module_name import SomeClass`` keeps pointing at the old
    ones. A later ``pytest.raises(SomeClass)`` would then silently fail to
    match an instance of the new, reloaded SomeClass, masking real
    regressions. Loading into an isolated probe module avoids that entirely:
    the production module and this file's existing imported names are never
    touched.

    Because the target module defines dataclasses, the probe module is
    registered under its unique probe name in ``sys.modules`` before
    ``exec_module`` runs, then removed (or restored to whatever was there
    before) in a ``finally`` block.
    """
    probe_name = f"_isolated_probe__{real_module_name}__{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(probe_name, module_file)
    module = importlib.util.module_from_spec(spec)

    previous = sys.modules.get(probe_name)
    sys.modules[probe_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(probe_name, None)
        else:
            sys.modules[probe_name] = previous
    return module


class _EnvironTrap:
    """A hostile stand-in for os.environ: any read raises."""

    def __getitem__(self, key):
        raise AssertionError(f"must not read os.environ[{key!r}]")

    def get(self, key, default=None):
        raise AssertionError(f"must not read os.environ.get({key!r})")

    def __contains__(self, key):
        raise AssertionError(f"must not check {key!r} in os.environ")


# --- no I/O / no dotenv at import or construction time ---


def test_import_never_calls_load_dotenv(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("rag_model_profiles must never call load_dotenv")

    trap = types.ModuleType("dotenv")
    trap.load_dotenv = _boom
    monkeypatch.setitem(sys.modules, "dotenv", trap)

    # Executing a fresh copy of the module's source while dotenv is trapped
    # proves no module-level code path calls load_dotenv, without reloading
    # (and thereby replacing the class objects of) the real, already-imported
    # rag_model_profiles module that the rest of this file depends on.
    _load_isolated_module(profiles.__file__, "rag_model_profiles")


def test_constructing_default_profile_does_not_touch_os_environ(monkeypatch):
    # _EnvironTrap is deliberately nonconforming: its __getitem__/get/
    # __contains__ all raise AssertionError, whereas a real environment
    # mapping's .get() returns a value/default and a missing key normally
    # surfaces as KeyError (or is simply absent, never an assertion). Code
    # outside this test's control depends on that normal-mapping contract --
    # e.g. shutil.get_terminal_size() reads os.environ["COLUMNS"] expecting
    # KeyError/ValueError on a miss (which it catches), and pytest's own
    # terminal writer calls os.environ.get("PY_COLORS") expecting a plain
    # return value, not an exception. Patching the real, process-wide
    # `os.environ` (via monkeypatch.setattr(os, "environ", ...)) would hand
    # this hostile trap to pytest and the stdlib as well, and an
    # AssertionError escaping from inside their machinery is not something
    # they catch -- that previously produced 215 cascading pytest setup/
    # teardown errors instead of a clean pass/fail for this one test.
    #
    # Patching only the module-under-test's own `os` binding keeps the trap
    # entirely local: rag_model_profiles.py does `import os` at its top, so
    # `profiles.os` is the name this test needs to control, not the real
    # `os` module every other piece of the process shares.
    module_os = types.SimpleNamespace(environ=_EnvironTrap())
    monkeypatch.setattr(profiles, "os", module_os)
    # Should not raise: EmbeddingProfile construction is pure, in-memory
    # validation and never reads os.environ. If default_embedding_profile()
    # ever read profiles.os.environ, _EnvironTrap would raise; if it reached
    # for another environment API (e.g. profiles.os.getenv), this bare
    # SimpleNamespace lacks that attribute and the test would fail with
    # AttributeError instead. Either way, the real stdlib os module (and its
    # real os.environ) is untouched for pytest, shutil, and everything else
    # in the process, and monkeypatch restores profiles.os automatically
    # when this test ends -- leaving the later resolve_embedding_profile()
    # tests, which exercise real os.environ, unaffected.
    default_embedding_profile()


# --- default profile is one atomic profile, sourced solely from the registry ---


def test_default_embedding_profile_fields():
    p = default_embedding_profile()
    assert p.provider == "ollama"
    assert p.model == "mxbai-embed-large:335m"
    assert p.dimension == 1024
    assert p.query_prefix == "Represent this sentence for searching relevant passages: "
    assert p.document_prefix == ""
    assert isinstance(p.profile_id, str) and p.profile_id


def test_default_query_prefix_ends_with_a_space():
    # The trailing space is load-bearing: it separates the prefix from the
    # embedded text. This is exactly the value that must survive quoting in
    # .env.example.
    p = default_embedding_profile()
    assert p.query_prefix.endswith(" ")


def test_default_embedding_model_key_is_pinned():
    assert profiles._DEFAULT_EMBEDDING_MODEL_KEY == ("ollama", "mxbai-embed-large:335m")


def test_default_embedding_profile_golden_id():
    # Literal, pinned expected value for the documented default profile's
    # derived id -- a regression here means the id derivation, the scheme
    # string, the domain, or the default profile's fields changed.
    p = default_embedding_profile()
    assert p.profile_id == (
        "3a5949f5b1ff824767b09a0ccd8e297fef05aaca166e32a939a3e0afe2aa85fe"
    )


# --- registry behavior ---


def test_known_embedding_profile_matches_default_for_mxbai():
    p = known_embedding_profile("ollama", "mxbai-embed-large:335m")
    assert p == default_embedding_profile()


def test_known_embedding_profile_unknown_model_raises():
    with pytest.raises(UnknownEmbeddingModelError):
        known_embedding_profile("ollama", "does-not-exist:latest")


@pytest.mark.parametrize("model", ["bge-m3:567m", "embeddinggemma:latest"])
def test_known_embedding_profile_unverified_preprocessing_raises(model):
    with pytest.raises(UnverifiedPreprocessingError):
        known_embedding_profile("ollama", model)


def test_registry_entries_are_marked_pending_phase_h():
    for entry in KNOWN_EMBEDDING_MODELS.values():
        assert entry.verification_status == UNVERIFIED_PENDING_PHASE_H


def test_bge_m3_registry_dimension():
    entry = KNOWN_EMBEDDING_MODELS[("ollama", "bge-m3:567m")]
    assert entry.expected_dimension == 1024
    assert entry.query_prefix is None
    assert entry.document_prefix is None


def test_embeddinggemma_registry_full_dimension_and_matryoshka_note():
    entry = KNOWN_EMBEDDING_MODELS[("ollama", "embeddinggemma:latest")]
    assert entry.expected_dimension == 768
    assert "matryoshka" in entry.notes.lower() or "512" in entry.notes


def test_known_embedding_models_registry_is_immutable():
    with pytest.raises(TypeError):
        KNOWN_EMBEDDING_MODELS[("ollama", "new-model:latest")] = None


# --- known-model dimension mismatch failure ---


def test_explicit_profile_known_model_dimension_mismatch_fails_closed():
    with pytest.raises(EmbeddingDimensionMismatchError):
        explicit_unverified_embedding_profile(
            provider="ollama",
            model="mxbai-embed-large:335m",
            dimension=1536,
            query_prefix="whatever: ",
            document_prefix="",
        )


@pytest.mark.parametrize("bad_dimension", [512, 256, 128])
def test_embeddinggemma_matryoshka_dimensions_are_rejected_in_c1(bad_dimension):
    with pytest.raises(EmbeddingDimensionMismatchError):
        explicit_unverified_embedding_profile(
            provider="ollama",
            model="embeddinggemma:latest",
            dimension=bad_dimension,
            query_prefix="q: ",
            document_prefix="",
        )


def test_explicit_profile_known_model_matching_dimension_succeeds():
    p = explicit_unverified_embedding_profile(
        provider="ollama",
        model="bge-m3:567m",
        dimension=1024,
        query_prefix="q: ",
        document_prefix="d: ",
    )
    assert p.dimension == 1024
    assert p.query_prefix == "q: "
    assert p.document_prefix == "d: "


def test_explicit_profile_unknown_model_has_no_dimension_check():
    p = explicit_unverified_embedding_profile(
        provider="ollama",
        model="some-new-model:latest",
        dimension=384,
        query_prefix="",
        document_prefix="",
    )
    assert p.dimension == 384


# --- verification status: visible, metadata-only, excluded from identity ---


def test_default_profile_reports_unverified_pending_phase_h():
    p = default_embedding_profile()
    assert p.verification_status == UNVERIFIED_PENDING_PHASE_H


def test_verification_status_is_visible_on_the_profile():
    p = default_embedding_profile()
    assert hasattr(p, "verification_status")
    assert p.verification_status in ("verified", "unverified_pending_phase_h")


def test_explicit_unverified_profile_always_reports_unverified_pending_phase_h():
    p = explicit_unverified_embedding_profile(
        provider="ollama",
        model="bge-m3:567m",
        dimension=1024,
        query_prefix="q: ",
        document_prefix="",
    )
    assert p.verification_status == UNVERIFIED_PENDING_PHASE_H


def test_changing_only_verification_status_does_not_change_identity_or_hash():
    fields = dict(
        provider="ollama", model="m", dimension=8, query_prefix="q", document_prefix="d"
    )
    profile_a = EmbeddingProfile(**fields)
    profile_b = EmbeddingProfile(**fields)

    # Change only profile_b's verification status, via the same internal
    # mechanism known_embedding_profile() uses. Neither object is hashed
    # before this point, so equality/hash equality below can't be an
    # artifact of a hash computed (and possibly cached) beforehand.
    profiles._freeze_set(profile_b, verification_status=VERIFIED)

    assert profile_a.verification_status != profile_b.verification_status
    assert profile_a.profile_id == profile_b.profile_id
    assert profile_a == profile_b
    assert hash(profile_a) == hash(profile_b)


def test_verification_status_cannot_be_passed_to_constructor():
    with pytest.raises(TypeError):
        EmbeddingProfile(
            provider="ollama",
            model="m",
            dimension=8,
            query_prefix="",
            document_prefix="",
            verification_status=VERIFIED,
        )


# --- deterministic, domain-separated derived profile ids ---


def test_embedding_profile_id_is_deterministic():
    a = EmbeddingProfile(
        provider="ollama", model="m", dimension=8, query_prefix="q", document_prefix="d"
    )
    b = EmbeddingProfile(
        provider="ollama", model="m", dimension=8, query_prefix="q", document_prefix="d"
    )
    assert a.profile_id == b.profile_id
    assert len(a.profile_id) == 64  # full SHA-256 hex digest, not truncated


@pytest.mark.parametrize(
    "field_name,override",
    [
        ("provider", "not-ollama"),
        ("model", "different-model"),
        ("dimension", 9),
        ("query_prefix", "different query"),
        ("document_prefix", "different document"),
    ],
)
def test_embedding_profile_id_changes_when_any_field_changes(field_name, override):
    base_fields = dict(
        provider="ollama", model="m", dimension=8, query_prefix="q", document_prefix="d"
    )
    base = EmbeddingProfile(**base_fields)
    changed_fields = dict(base_fields)
    changed_fields[field_name] = override
    changed = EmbeddingProfile(**changed_fields)
    assert base.profile_id != changed.profile_id


def test_generation_profile_id_is_deterministic_and_changes_per_field():
    a = GenerationProfile(provider="ollama", model="llama3.1:latest")
    b = GenerationProfile(provider="ollama", model="llama3.1:latest")
    c = GenerationProfile(provider="ollama", model="mistral:latest")
    assert a.profile_id == b.profile_id
    assert a.profile_id != c.profile_id


def test_embedding_and_generation_profile_ids_differ_for_same_provider_model():
    embedding = explicit_unverified_embedding_profile(
        provider="ollama",
        model="llama3.1:latest",
        dimension=8,
        query_prefix="",
        document_prefix="",
    )
    generation = GenerationProfile(provider="ollama", model="llama3.1:latest")

    assert embedding.provider == generation.provider == "ollama"
    assert embedding.model == generation.model == "llama3.1:latest"
    assert embedding.profile_id != generation.profile_id


# --- atomic field validation ---


@pytest.mark.parametrize("bad_dimension", [0, -1, -1024, 1.5, "1024", True, False, None])
def test_embedding_profile_rejects_non_positive_or_non_int_dimension(bad_dimension):
    with pytest.raises(InvalidProfileFieldError):
        EmbeddingProfile(
            provider="ollama",
            model="m",
            dimension=bad_dimension,
            query_prefix="",
            document_prefix="",
        )


@pytest.mark.parametrize("bad_str", ["", "   ", None, 123])
def test_embedding_profile_rejects_empty_or_non_string_provider(bad_str):
    with pytest.raises(InvalidProfileFieldError):
        EmbeddingProfile(
            provider=bad_str, model="m", dimension=8, query_prefix="", document_prefix=""
        )


@pytest.mark.parametrize("bad_str", ["", "   ", None, 123])
def test_embedding_profile_rejects_empty_or_non_string_model(bad_str):
    with pytest.raises(InvalidProfileFieldError):
        EmbeddingProfile(
            provider="ollama", model=bad_str, dimension=8, query_prefix="", document_prefix=""
        )


def test_embedding_profile_allows_empty_prefixes_but_not_non_string_prefixes():
    # Empty string prefixes are fine (verified-empty document prefix case).
    p = EmbeddingProfile(
        provider="ollama", model="m", dimension=8, query_prefix="", document_prefix=""
    )
    assert p.query_prefix == "" and p.document_prefix == ""

    with pytest.raises(InvalidProfileFieldError):
        EmbeddingProfile(
            provider="ollama", model="m", dimension=8, query_prefix=None, document_prefix=""
        )


def test_embedding_profile_id_cannot_be_passed_to_constructor():
    with pytest.raises(TypeError):
        EmbeddingProfile(
            provider="ollama",
            model="m",
            dimension=8,
            query_prefix="",
            document_prefix="",
            profile_id="hand-rolled-id",
        )


# --- generation profile: local-only, reject external providers ---


@pytest.mark.parametrize("bad_provider", ["openrouter", "openai", "anthropic", ""])
def test_generation_profile_rejects_unsupported_or_external_providers(bad_provider):
    expected = InvalidProfileFieldError if not bad_provider else UnsupportedGenerationProviderError
    with pytest.raises(expected):
        GenerationProfile(provider=bad_provider, model="llama3.1:latest")


def test_generation_profile_rejects_empty_model():
    with pytest.raises(InvalidProfileFieldError):
        GenerationProfile(provider="ollama", model="")


def test_generation_profile_accepts_ollama():
    p = GenerationProfile(provider="ollama", model="llama3.1:latest")
    assert p.provider == "ollama"
    assert p.model == "llama3.1:latest"
    assert p.profile_id


# --- resolve_embedding_profile: config-driven, all-or-nothing overrides ---


def test_resolve_embedding_profile_empty_mapping_returns_default():
    assert resolve_embedding_profile({}) == default_embedding_profile()


@pytest.mark.parametrize(
    "partial",
    [
        {"EMBEDDING_MODEL": "bge-m3:567m"},
        {"EMBEDDING_PROVIDER": "ollama", "EMBEDDING_MODEL": "bge-m3:567m"},
        {
            "EMBEDDING_PROVIDER": "ollama",
            "EMBEDDING_MODEL": "bge-m3:567m",
            "EMBEDDING_DIMENSION": "1024",
        },
    ],
)
def test_resolve_embedding_profile_rejects_partial_overrides(partial):
    with pytest.raises(PartialEmbeddingProfileOverrideError):
        resolve_embedding_profile(partial)


def test_resolve_embedding_profile_rejects_orphaned_acknowledgement():
    # The acknowledgement alone, with none of the five override fields, is
    # meaningless -- and must be rejected with a message that says so.
    with pytest.raises(PartialEmbeddingProfileOverrideError, match="cannot appear without"):
        resolve_embedding_profile({"EMBEDDING_PROFILE_UNVERIFIED_ACK": "pending_phase_h"})


def test_resolve_embedding_profile_full_override_without_ack_is_rejected():
    full = {
        "EMBEDDING_PROVIDER": "ollama",
        "EMBEDDING_MODEL": "bge-m3:567m",
        "EMBEDDING_DIMENSION": "1024",
        "EMBEDDING_QUERY_PREFIX": "q: ",
        "EMBEDDING_DOCUMENT_PREFIX": "",
    }
    with pytest.raises(UnacknowledgedUnverifiedProfileError):
        resolve_embedding_profile(full)


def test_resolve_embedding_profile_full_override_with_ack_succeeds():
    full = {
        "EMBEDDING_PROVIDER": "ollama",
        "EMBEDDING_MODEL": "bge-m3:567m",
        "EMBEDDING_DIMENSION": "1024",
        "EMBEDDING_QUERY_PREFIX": "q: ",
        "EMBEDDING_DOCUMENT_PREFIX": "",
        "EMBEDDING_PROFILE_UNVERIFIED_ACK": "pending_phase_h",
    }
    p = resolve_embedding_profile(full)
    assert p.provider == "ollama"
    assert p.model == "bge-m3:567m"
    assert p.dimension == 1024
    assert p.query_prefix == "q: "


def test_resolve_embedding_profile_full_override_still_enforces_known_dimension():
    full = {
        "EMBEDDING_PROVIDER": "ollama",
        "EMBEDDING_MODEL": "mxbai-embed-large:335m",
        "EMBEDDING_DIMENSION": "9999",
        "EMBEDDING_QUERY_PREFIX": "q: ",
        "EMBEDDING_DOCUMENT_PREFIX": "",
        "EMBEDDING_PROFILE_UNVERIFIED_ACK": "pending_phase_h",
    }
    with pytest.raises(EmbeddingDimensionMismatchError):
        resolve_embedding_profile(full)


def test_resolve_embedding_profile_invalid_dimension_string_raises():
    full = {
        "EMBEDDING_PROVIDER": "ollama",
        "EMBEDDING_MODEL": "some-model:latest",
        "EMBEDDING_DIMENSION": "not-an-int",
        "EMBEDDING_QUERY_PREFIX": "",
        "EMBEDDING_DOCUMENT_PREFIX": "",
        "EMBEDDING_PROFILE_UNVERIFIED_ACK": "pending_phase_h",
    }
    with pytest.raises(InvalidProfileFieldError):
        resolve_embedding_profile(full)


def test_resolve_embedding_profile_defaults_to_os_environ_when_omitted(monkeypatch):
    for key in profiles._EMBEDDING_OVERRIDE_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv(profiles._UNVERIFIED_ACK_KEY, raising=False)
    assert resolve_embedding_profile() == default_embedding_profile()


# --- resolve_generation_profile ---


def test_resolve_generation_profile_requires_model():
    with pytest.raises(InvalidProfileFieldError):
        resolve_generation_profile({})


def test_resolve_generation_profile_defaults_provider_to_ollama_when_absent():
    p = resolve_generation_profile({"GENERATION_MODEL": "llama3.1:latest"})
    assert p.provider == "ollama"
    assert p.model == "llama3.1:latest"


def test_resolve_generation_profile_rejects_external_provider():
    with pytest.raises(UnsupportedGenerationProviderError):
        resolve_generation_profile(
            {"GENERATION_PROVIDER": "openrouter", "GENERATION_MODEL": "openrouter/auto"}
        )


def test_resolve_generation_profile_present_but_empty_provider_fails_closed():
    # An explicitly empty GENERATION_PROVIDER is a configuration mistake, not
    # "unset" -- it must be rejected by validation, not silently defaulted.
    with pytest.raises(InvalidProfileFieldError):
        resolve_generation_profile({"GENERATION_PROVIDER": "", "GENERATION_MODEL": "x"})


# --- Nomic Embed Text v1.5 (I2): additive registry entry, MxBAI stays default ---


def test_nomic_registry_entry_exists():
    assert ("ollama", "nomic-embed-text:v1.5") in KNOWN_EMBEDDING_MODELS


def test_nomic_registry_entry_fields():
    entry = KNOWN_EMBEDDING_MODELS[("ollama", "nomic-embed-text:v1.5")]
    assert entry.provider == "ollama"
    assert entry.model == "nomic-embed-text:v1.5"
    assert entry.expected_dimension == 768
    assert entry.query_prefix == "search_query: "
    assert entry.document_prefix == "search_document: "


def test_nomic_registry_entry_verification_status_is_pending_phase_h():
    # Asserted directly on the KnownEmbeddingModel registry entry, not on an
    # EmbeddingProfile -- EmbeddingProfile.verification_status already
    # defaults to UNVERIFIED_PENDING_PHASE_H regardless of what (if anything)
    # the registry says, so that field alone would not prove this entry is
    # actually marked pending in the registry itself.
    entry = KNOWN_EMBEDDING_MODELS[("ollama", "nomic-embed-text:v1.5")]
    assert entry.verification_status == UNVERIFIED_PENDING_PHASE_H


def test_nomic_registry_notes_state_phase_h_validation_is_pending():
    notes = KNOWN_EMBEDDING_MODELS[("ollama", "nomic-embed-text:v1.5")].notes.lower()
    assert "phase h" in notes
    assert "pending" in notes
    assert "verifi" in notes and "not" in notes


def test_known_embedding_profile_nomic_succeeds_without_acknowledgement():
    # Both prefixes are non-None in the registry, so the direct known-model
    # lookup succeeds on its own -- no EMBEDDING_PROFILE_UNVERIFIED_ACK is
    # involved at all, unlike the resolve_embedding_profile() config route
    # below, which always requires the full five-key override plus ack.
    p = known_embedding_profile("ollama", "nomic-embed-text:v1.5")
    assert p.provider == "ollama"
    assert p.model == "nomic-embed-text:v1.5"
    assert p.dimension == 768
    assert p.query_prefix == "search_query: "
    assert p.document_prefix == "search_document: "


def test_resolve_embedding_profile_nomic_full_override_with_ack_succeeds():
    full = {
        "EMBEDDING_PROVIDER": "ollama",
        "EMBEDDING_MODEL": "nomic-embed-text:v1.5",
        "EMBEDDING_DIMENSION": "768",
        "EMBEDDING_QUERY_PREFIX": "search_query: ",
        "EMBEDDING_DOCUMENT_PREFIX": "search_document: ",
        "EMBEDDING_PROFILE_UNVERIFIED_ACK": "pending_phase_h",
    }
    p = resolve_embedding_profile(full)
    assert p.provider == "ollama"
    assert p.model == "nomic-embed-text:v1.5"
    assert p.dimension == 768
    assert p.query_prefix == "search_query: "
    assert p.document_prefix == "search_document: "


def test_resolve_embedding_profile_nomic_wrong_dimension_fails_closed():
    full = {
        "EMBEDDING_PROVIDER": "ollama",
        "EMBEDDING_MODEL": "nomic-embed-text:v1.5",
        "EMBEDDING_DIMENSION": "1024",
        "EMBEDDING_QUERY_PREFIX": "search_query: ",
        "EMBEDDING_DOCUMENT_PREFIX": "search_document: ",
        "EMBEDDING_PROFILE_UNVERIFIED_ACK": "pending_phase_h",
    }
    with pytest.raises(EmbeddingDimensionMismatchError):
        resolve_embedding_profile(full)
