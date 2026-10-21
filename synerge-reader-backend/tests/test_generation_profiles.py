"""Registry, option validation and shared-selection snapshots for
answer-generation profiles.

Pure: generation_profiles.py does no I/O, and the one runtime fact a snapshot
needs (the installed digest) is injected here as a callable. Gate behaviour is
exercised with synthetic digests on constructed profiles; only the
production-constant tests compare against the operator-supplied pins.

Which profile answers is the shared selection an administrator switches (see
generation_coordinator.py); the per-request override and its admin-authority
resolution are retired, and these tests prove they no longer exist.
"""

import dataclasses
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import generation_profiles
from generation_profiles import (
    GEMMA_CANDIDATE,
    LLAMA_CONTROL,
    SHARED_SELECTION,
    SMOKE_OPTIONS,
    FixedModelCall,
    GenerationOptions,
    GenerationProfileError,
    ModelIdentityMismatchError,
    ModelIdentityUnavailableError,
    ModelProfile,
    ModelRegistry,
    ProfileNotQualifiedError,
    Qualification,
    UnknownModelProfileError,
    _snapshot,
    check_installed_identity,
    default_model_registry,
    select_shared_profile,
    shared_generation_snapshot,
)

# Operator-supplied pins, asserted only against the production constants.
OPERATOR_LLAMA_DIGEST = "42182419e9508c30c4b1fe55015f06b65f4ca4b9e28a744be55008d21998a093"
OPERATOR_GEMMA_DIGEST = "6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7"

# Synthetic identities for gate behaviour.
SYNTHETIC_DIGEST = "ab" * 32
OTHER_DIGEST = "cd" * 32


def _never(*_args):
    raise AssertionError("this lookup must not run")


def _evidence(**overrides):
    fields = dict(
        think_false_verified=True,
        combined_capacity_verified=True,
        quality_verified=True,
        evidence_ref="synthetic test evidence",
    )
    fields.update(overrides)
    return Qualification(**fields)


def _gemma(**overrides):
    base = default_model_registry().get(GEMMA_CANDIDATE)
    fields = dict(expected_digest=SYNTHETIC_DIGEST)
    fields.update(overrides)
    return dataclasses.replace(base, **fields)


def _unqualified_gemma(**overrides):
    """Gemma with only its think:false gate recorded: an explicitly unqualified
    candidate, independent of the released production qualification."""
    return _gemma(qualification=_evidence(combined_capacity_verified=False, quality_verified=False), **overrides)


def _registry(gemma, default=LLAMA_CONTROL):
    return ModelRegistry(
        profiles=(default_model_registry().get(LLAMA_CONTROL), gemma),
        default_profile_id=default,
    )


def _qualified_registry(default=LLAMA_CONTROL):
    return _registry(_gemma(qualification=_evidence()), default=default)


def _shared(registry, profile_id, *, epoch=1, lookup=_never):
    """A snapshot the way main builds one for an admitted job: the selected
    profile, its identity read only if it requires qualification."""
    profile = select_shared_profile(registry, profile_id)
    observed = check_installed_identity(profile, lookup) if profile.requires_qualification else None
    return shared_generation_snapshot(profile, epoch=epoch, observed_digest=observed)


# --- the production registry --------------------------------------------------


def test_production_registry_has_the_two_installed_profiles_and_llama_default():
    registry = default_model_registry()
    assert registry.default.profile_id == LLAMA_CONTROL
    assert registry.get(LLAMA_CONTROL).model_tag == "llama3.1:8b"
    assert registry.get(GEMMA_CANDIDATE).model_tag == "gemma4:31b"
    # The control and qualified profiles are unchanged; the three
    # experimental profiles are registered after them, on their own runtime.
    assert [p.profile_id for p in registry.profiles if not p.experimental] == [LLAMA_CONTROL, GEMMA_CANDIDATE]
    assert [p.profile_id for p in registry.profiles if p.experimental] == [
        "qwen38_27b_experimental", "nemotron35_lightning_30b_experimental", "mistral_medium35_128b_experimental",
    ]
    assert {p.runtime_key for p in registry.profiles if not p.experimental} == {"primary"}
    assert {p.runtime_key for p in registry.profiles if p.experimental} == {"compat_cuda121"}
    assert "qwen3:latest" not in {p.model_tag for p in registry.profiles}, "the unapproved legacy tag stays out"


def test_production_pins_equal_the_operator_supplied_digests():
    registry = default_model_registry()
    assert registry.get(LLAMA_CONTROL).expected_digest == OPERATOR_LLAMA_DIGEST
    assert registry.get(GEMMA_CANDIDATE).expected_digest == OPERATOR_GEMMA_DIGEST


def test_production_qualification_policy_is_explicit_and_records_only_evidenced_gates():
    registry = default_model_registry()
    llama, gemma = registry.get(LLAMA_CONTROL), registry.get(GEMMA_CANDIDATE)

    assert llama.requires_qualification is False and llama.qualification is None
    assert llama.think is None, "Llama never receives a think field"

    assert gemma.requires_qualification is True, "qualified, but every job still checks its identity"
    assert gemma.qualification.think_false_verified is True
    assert gemma.qualification.combined_capacity_verified is True
    assert gemma.qualification.quality_verified is True
    assert gemma.think is False, "think:false is evidenced, so it is the Gemma policy"
    assert gemma.selectable and registry.is_selectable(gemma)
    # The recorded gates name a reviewable release record kept in the repository.
    record = "diagnostics/GEMMA_RELEASE_QUALIFICATION_20261010.md"
    assert record in gemma.qualification.evidence_ref
    assert os.path.isfile(os.path.join(os.path.dirname(__file__), "..", *record.split("/")))


def test_answer_profiles_request_the_reviewed_settings_and_timeouts():
    registry = default_model_registry()
    for profile_id in (LLAMA_CONTROL, GEMMA_CANDIDATE):
        profile = registry.get(profile_id)
        assert profile.answer_options.as_ollama_options() == {
            "temperature": 0.2, "num_ctx": 8192, "num_predict": 1200,
        }
        assert profile.keep_alive == "5m"
    assert registry.get(LLAMA_CONTROL).answer_read_timeout_s == 60.0
    # The observed cold load was 91.783 s; the answer timeout must exceed it.
    assert registry.get(GEMMA_CANDIDATE).answer_read_timeout_s == 150.0 > 91.783
    assert SMOKE_OPTIONS.as_ollama_options() == {"temperature": 0.2, "num_ctx": 4096, "num_predict": 32}
    # Per-read role timeouts: Llama keeps its call sites' previous values; any
    # Gemma role call may find the model cold, so each gets the provisional 150 s.
    llama, gemma = registry.get(LLAMA_CONTROL), registry.get(GEMMA_CANDIDATE)
    assert (llama.verifier_read_timeout_s, llama.helper_read_timeout_s) == (30.0, 120.0)
    assert (gemma.verifier_read_timeout_s, gemma.helper_read_timeout_s) == (150.0, 150.0)


def test_production_gemma_is_selectable_only_with_its_pinned_identity():
    registry = default_model_registry()
    gemma = select_shared_profile(registry, GEMMA_CANDIDATE)
    lookups = []
    snapshot = _shared(registry, GEMMA_CANDIDATE, epoch=2,
                       lookup=lambda tag: lookups.append(tag) or OPERATOR_GEMMA_DIGEST)
    assert lookups == ["gemma4:31b"], "the exact tag, read once for this job"
    assert (snapshot.profile_id, snapshot.selection, snapshot.selection_epoch) == (GEMMA_CANDIDATE, SHARED_SELECTION, 2)
    assert snapshot.expected_digest == snapshot.observed_digest == OPERATOR_GEMMA_DIGEST
    for observed in (None, OTHER_DIGEST):
        with pytest.raises(ModelIdentityUnavailableError):
            shared_generation_snapshot(gemma, epoch=1, observed_digest=observed)


def test_production_gemma_answer_verifier_and_helpers_keep_the_reviewed_settings():
    snapshot = _shared(default_model_registry(), GEMMA_CANDIDATE, lookup=lambda tag: OPERATOR_GEMMA_DIGEST)
    answer_options = {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1200}
    assert snapshot.ollama_payload("p") == {
        "model": "gemma4:31b", "prompt": "p", "stream": True,
        "options": answer_options, "keep_alive": "5m", "think": False,
    }
    assert snapshot.requested_settings() == {"options": answer_options, "keep_alive": "5m", "think": False}
    assert snapshot.public_metadata() == {"profile_id": GEMMA_CANDIDATE, "model": "gemma4:31b"}
    assert snapshot.answer_read_timeout_s == 150.0
    # The answer's verifier is the same model and identity with bounded options.
    verifier = snapshot.verifier
    assert verifier.ollama_payload("p") == {
        "model": "gemma4:31b", "prompt": "p", "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 16},
        "keep_alive": "5m", "think": False,
    }
    assert (verifier.profile_id, verifier.read_timeout_s) == (GEMMA_CANDIDATE, 150.0)
    assert verifier.expected_digest == verifier.observed_digest == OPERATOR_GEMMA_DIGEST
    for temperature in (0.2, 0.1):
        helper = snapshot.helper_call(temperature)
        assert helper.ollama_payload("p") == {
            "model": "gemma4:31b", "prompt": "p", "stream": False,
            "options": {"temperature": temperature, "num_ctx": 8192, "num_predict": 1200},
            "keep_alive": "5m", "think": False,
        }
        assert (helper.profile_id, helper.read_timeout_s) == (GEMMA_CANDIDATE, 150.0)
        assert helper.expected_digest == helper.observed_digest == OPERATOR_GEMMA_DIGEST


def test_verifier_and_helpers_are_bounded_llama_calls_without_think():
    snapshot = _shared(default_model_registry(), LLAMA_CONTROL)
    verifier = snapshot.verifier.ollama_payload("p")
    assert verifier == {
        "model": "llama3.1:8b", "prompt": "p", "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 16},
        "keep_alive": "5m",
    }
    assert snapshot.verifier.read_timeout_s == 30.0
    for temperature in (0.2, 0.1):
        call = snapshot.helper_call(temperature)
        assert call.ollama_payload("p") == {
            "model": "llama3.1:8b", "prompt": "p", "stream": False,
            "options": {"temperature": temperature, "num_ctx": 8192, "num_predict": 1200},
            "keep_alive": "5m",
        }, "no think field and no top-level temperature"
        assert call.read_timeout_s == 120.0


# --- option and profile validation ------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        dict(temperature=-0.1),
        dict(temperature=2.5),
        dict(temperature=True),
        dict(temperature="0.2"),
        dict(temperature=float("nan")),
        dict(num_ctx=131072),
        dict(num_ctx=256),
        dict(num_ctx=8192.0),
        dict(num_predict=0),
        dict(num_predict=8192),
        dict(num_predict=True),
        dict(top_k=0),
        dict(top_k=True),
        dict(top_p=0.0),
        dict(top_p=1.5),
    ],
)
def test_invalid_options_are_rejected(fields):
    valid = dict(temperature=0.2, num_ctx=8192, num_predict=1200)
    valid.update(fields)
    with pytest.raises(GenerationProfileError):
        GenerationOptions(**valid)


def test_optional_sampling_is_sent_only_when_set():
    options = GenerationOptions(temperature=0.2, num_ctx=8192, num_predict=1200, top_k=64, top_p=0.95)
    assert options.as_ollama_options() == {
        "temperature": 0.2, "num_ctx": 8192, "num_predict": 1200, "top_k": 64, "top_p": 0.95,
    }


def test_options_dict_is_a_copy_the_caller_cannot_use_to_mutate_policy():
    profile = default_model_registry().default
    leaked = profile.answer_options.as_ollama_options()
    leaked["num_ctx"] = 131072
    assert profile.answer_options.as_ollama_options()["num_ctx"] == 8192
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.answer_options.num_ctx = 131072


_VALID_PROFILE = dict(
    profile_id="candidate", model_tag="model:tag", display_name="Model",
    answer_options=SMOKE_OPTIONS, keep_alive="5m", answer_read_timeout_s=60.0,
    verifier_read_timeout_s=30.0, helper_read_timeout_s=120.0,
    requires_qualification=False,
)


@pytest.mark.parametrize(
    "fields",
    [
        dict(profile_id="Bad Id"),
        dict(model_tag="llama3.1"),            # an exact tag, never a bare name
        dict(model_tag="llama3.1:8b; rm"),
        dict(keep_alive="-1"),
        dict(keep_alive="forever"),
        dict(display_name=" "),
        dict(answer_read_timeout_s=0),
        dict(answer_read_timeout_s=601),
        dict(answer_read_timeout_s=math.inf),
        dict(answer_read_timeout_s=None),
        dict(answer_read_timeout_s=True),
        dict(requires_qualification=1),
        dict(requires_qualification=True),     # nothing to verify identity against
        dict(expected_digest="not-a-digest"),
        dict(think=False),                     # unevidenced no-thinking control
        dict(think=True, qualification=_evidence()),  # thinking output is never requested
        dict(think="false", qualification=_evidence()),
        # The role read timeouts use the same finite bounds, and a bool never
        # passes as a number of seconds.
        dict(verifier_read_timeout_s=0),
        dict(verifier_read_timeout_s=601),
        dict(verifier_read_timeout_s=math.inf),
        dict(verifier_read_timeout_s=float("nan")),
        dict(verifier_read_timeout_s=None),
        dict(verifier_read_timeout_s=True),
        dict(verifier_read_timeout_s="30"),
        dict(helper_read_timeout_s=0.5),
        dict(helper_read_timeout_s=600.5),
        dict(helper_read_timeout_s=-math.inf),
        dict(helper_read_timeout_s=None),
        dict(helper_read_timeout_s=False),
        dict(helper_read_timeout_s="120"),
    ],
)
def test_invalid_profiles_are_rejected(fields):
    base = dict(_VALID_PROFILE)
    base.update(fields)
    with pytest.raises(GenerationProfileError):
        ModelProfile(**base)


def test_think_false_needs_its_specific_evidence():
    with pytest.raises(GenerationProfileError):
        _gemma(qualification=_evidence(think_false_verified=False), think=False)
    assert _gemma(qualification=_evidence(think_false_verified=False), think=None).think is None


def test_expected_digests_are_normalized():
    assert _gemma(expected_digest="sha256:" + SYNTHETIC_DIGEST.upper()).expected_digest == SYNTHETIC_DIGEST


def test_registry_rejects_a_missing_default_and_duplicate_ids():
    llama = default_model_registry().get(LLAMA_CONTROL)
    with pytest.raises(GenerationProfileError):
        ModelRegistry(profiles=(llama,), default_profile_id="absent")
    with pytest.raises(GenerationProfileError):
        ModelRegistry(profiles=(llama, llama), default_profile_id=LLAMA_CONTROL)
    with pytest.raises(GenerationProfileError):
        ModelRegistry(profiles=(), default_profile_id=LLAMA_CONTROL)


def test_the_registry_has_no_independent_verifier_or_helper_selector():
    """Verification and helpers are derived from a resolved profile; a
    registry can no longer name a separate model for them."""
    registry = default_model_registry()
    with pytest.raises(TypeError):
        ModelRegistry(
            profiles=registry.profiles,
            default_profile_id=LLAMA_CONTROL,
            verifier_profile_id=LLAMA_CONTROL,
        )
    for removed in ("verifier_profile_id", "verifier_call", "helper_call"):
        assert not hasattr(registry, removed), removed


# --- an incomplete candidate can never be the default ----------------------


@pytest.mark.parametrize(
    "qualification",
    [
        None,
        _evidence(combined_capacity_verified=False),
        _evidence(quality_verified=False),
        _evidence(combined_capacity_verified=False, quality_verified=False),
    ],
    ids=["no-evidence", "no-capacity", "no-quality", "pre-release-shape"],
)
def test_an_incomplete_candidate_cannot_be_constructed_as_default(qualification):
    think = False if qualification is not None else None
    with pytest.raises(GenerationProfileError):
        _registry(_gemma(qualification=qualification, think=think), default=GEMMA_CANDIDATE)


def test_the_production_default_stays_llama_and_an_unqualified_gemma_cannot_be_default():
    """Qualification makes Gemma eligible for the administrator's switch; it
    does not make Gemma the default that seeds a new shared selection."""
    registry = default_model_registry()
    assert registry.default_profile_id == LLAMA_CONTROL and registry.default.profile_id == LLAMA_CONTROL
    with pytest.raises(GenerationProfileError):
        ModelRegistry(
            profiles=(registry.get(LLAMA_CONTROL), _unqualified_gemma()),
            default_profile_id=GEMMA_CANDIDATE,
        )


def test_an_unqualified_selection_is_refused_with_no_fallback():
    registry = _registry(_unqualified_gemma())
    with pytest.raises(ProfileNotQualifiedError):
        select_shared_profile(registry, GEMMA_CANDIDATE)
    # Even handed the profile directly, an unqualified one yields no snapshot.
    with pytest.raises(ProfileNotQualifiedError):
        shared_generation_snapshot(registry.get(GEMMA_CANDIDATE), epoch=1, observed_digest=SYNTHETIC_DIGEST)


@pytest.mark.parametrize("profile_id", ["qwen3_latest", "Llama Control", None, 7])
def test_an_unknown_selection_is_refused_with_no_fallback(profile_id):
    with pytest.raises(UnknownModelProfileError):
        select_shared_profile(_qualified_registry(), profile_id)


def test_the_per_request_override_resolution_no_longer_exists():
    for retired in ("resolve_answer_generation", "resolve_default_generation", "OverrideNotPermittedError",
                    "SelectionAuthorityUnavailableError"):
        assert not hasattr(generation_profiles, retired), retired


# --- snapshots from the shared selection ---------------------------------------


def test_the_llama_control_needs_no_inventory_lookup():
    snapshot = _shared(default_model_registry(), LLAMA_CONTROL, epoch=4)
    assert snapshot.profile_id == LLAMA_CONTROL
    assert (snapshot.selection, snapshot.selection_epoch) == (SHARED_SELECTION, 4)
    assert snapshot.answer_read_timeout_s == 60.0
    assert snapshot.ollama_payload("p") == {
        "model": "llama3.1:8b", "prompt": "p", "stream": True,
        "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1200},
        "keep_alive": "5m",
    }
    assert snapshot.public_metadata() == {"profile_id": LLAMA_CONTROL, "model": "llama3.1:8b"}


def test_expected_and_observed_digests_are_never_conflated():
    llama = _shared(default_model_registry(), LLAMA_CONTROL)
    assert llama.expected_digest == OPERATOR_LLAMA_DIGEST
    assert llama.observed_digest is None, "a pin is not an observation for this answer"
    assert "digest" not in str(llama.public_metadata())
    with pytest.raises(GenerationProfileError):
        shared_generation_snapshot(default_model_registry().get(LLAMA_CONTROL), epoch=1,
                                   observed_digest=OPERATOR_LLAMA_DIGEST)

    gemma = _shared(_qualified_registry(), GEMMA_CANDIDATE, lookup=lambda tag: "sha256:" + SYNTHETIC_DIGEST.upper())
    assert gemma.expected_digest == SYNTHETIC_DIGEST
    assert gemma.observed_digest == SYNTHETIC_DIGEST, "the normalized value the lookup reported"
    for observed in (None, OTHER_DIGEST):
        with pytest.raises(ModelIdentityUnavailableError):
            shared_generation_snapshot(_qualified_registry().get(GEMMA_CANDIDATE), epoch=1, observed_digest=observed)


def test_a_qualified_candidate_needs_its_exact_digest_read_once():
    lookups = []

    def installed(tag):
        lookups.append(tag)
        return SYNTHETIC_DIGEST

    snapshot = _shared(_qualified_registry(), GEMMA_CANDIDATE, epoch=3, lookup=installed)
    assert lookups == ["gemma4:31b"]
    assert (snapshot.profile_id, snapshot.selection_epoch) == (GEMMA_CANDIDATE, 3)
    assert snapshot.answer_read_timeout_s == 150.0
    assert snapshot.ollama_payload("p")["think"] is False
    # The answer is verified by the model that wrote it, with no second lookup.
    assert snapshot.verifier.model_tag == "gemma4:31b"
    assert snapshot.verifier.observed_digest == SYNTHETIC_DIGEST


_WRONG_IDENTITY_LOOKUPS = pytest.mark.parametrize(
    "lookup, error",
    [
        (lambda tag: None, ModelIdentityMismatchError),
        (lambda tag: OTHER_DIGEST, ModelIdentityMismatchError),
        (lambda tag: "not-a-digest", ModelIdentityUnavailableError),
        (lambda tag: (_ for _ in ()).throw(ConnectionError("ollama down")), ModelIdentityUnavailableError),
    ],
    ids=["absent", "changed", "malformed", "unreadable"],
)


@_WRONG_IDENTITY_LOOKUPS
def test_a_qualified_selection_with_the_wrong_identity_fails_closed(lookup, error):
    with pytest.raises(error) as excinfo:
        _shared(_qualified_registry(), GEMMA_CANDIDATE, lookup=lookup)
    assert "ollama down" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None


@_WRONG_IDENTITY_LOOKUPS
def test_production_gemma_with_a_missing_or_mismatched_identity_fails_closed(lookup, error):
    with pytest.raises(error) as excinfo:
        _shared(default_model_registry(), GEMMA_CANDIDATE, lookup=lookup)
    assert "ollama down" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None


def test_an_identity_check_needs_a_pin_and_reads_only_the_exact_tag():
    unpinned = ModelProfile(**{**_VALID_PROFILE, "expected_digest": None})
    with pytest.raises(ModelIdentityUnavailableError):
        check_installed_identity(unpinned, _never)
    lookups = []
    assert check_installed_identity(
        default_model_registry().get(LLAMA_CONTROL), lambda tag: lookups.append(tag) or OPERATOR_LLAMA_DIGEST
    ) == OPERATOR_LLAMA_DIGEST
    assert lookups == ["llama3.1:8b"]


@pytest.mark.parametrize("epoch", [0, -1, True, 1.0, "1", None, 2**62 + 1])
def test_a_shared_snapshot_carries_a_valid_epoch(epoch):
    with pytest.raises(GenerationProfileError):
        shared_generation_snapshot(default_model_registry().get(LLAMA_CONTROL), epoch=epoch, observed_digest=None)


def test_historical_selections_carry_no_epoch_and_stay_constructible():
    llama = default_model_registry().get(LLAMA_CONTROL)
    for selection in ("default", "admin_override"):
        assert _snapshot(llama, selection, None).selection_epoch is None, "provenance v1/v2 and the diagnostic"
        with pytest.raises(GenerationProfileError):
            _snapshot(llama, selection, None, 3)
    with pytest.raises(GenerationProfileError):
        _snapshot(llama, SHARED_SELECTION, None)
    with pytest.raises(GenerationProfileError):
        _snapshot(llama, "per_request", None)


def test_a_snapshot_is_immutable_and_independent_of_a_later_switch():
    snapshot = _shared(_qualified_registry(), LLAMA_CONTROL, epoch=1)
    assert _shared(_qualified_registry(), GEMMA_CANDIDATE, epoch=3,
                   lookup=lambda tag: SYNTHETIC_DIGEST).model_tag == "gemma4:31b"
    assert snapshot.model_tag == "llama3.1:8b" and snapshot.selection_epoch == 1
    assert snapshot.ollama_payload("p")["model"] == "llama3.1:8b"
    assert snapshot.answer_read_timeout_s == 60.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.model_tag = "gemma4:31b"
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.selection_epoch = 3


# --- role calls: the verifier and helpers follow the snapshot's profile --------


def test_a_qualified_candidate_verifier_has_the_answers_identity_and_bounded_settings():
    snapshot = _shared(_qualified_registry(), GEMMA_CANDIDATE, lookup=lambda tag: "sha256:" + SYNTHETIC_DIGEST.upper())

    verifier = snapshot.verifier
    assert (verifier.profile_id, verifier.model_tag) == (GEMMA_CANDIDATE, "gemma4:31b")
    assert (verifier.expected_digest, verifier.observed_digest) == (SYNTHETIC_DIGEST, SYNTHETIC_DIGEST)
    assert verifier.think is False and verifier.keep_alive == "5m"
    assert verifier.read_timeout_s == 150.0
    assert verifier.ollama_payload("p") == {
        "model": "gemma4:31b", "prompt": "p", "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 16},
        "keep_alive": "5m", "think": False,
    }
    helper = snapshot.helper_call(0.1)
    assert helper.ollama_payload("p") == {
        "model": "gemma4:31b", "prompt": "p", "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": 1200},
        "keep_alive": "5m", "think": False,
    }
    assert (helper.read_timeout_s, helper.expected_digest, helper.observed_digest) == (
        150.0, SYNTHETIC_DIGEST, SYNTHETIC_DIGEST,
    )


def test_the_control_role_calls_carry_its_pin_and_no_observation():
    snapshot = _shared(default_model_registry(), LLAMA_CONTROL)
    for call in (snapshot.verifier, snapshot.helper_call(0.2)):
        assert call.profile_id == LLAMA_CONTROL and call.model_tag == "llama3.1:8b"
        assert call.expected_digest == OPERATOR_LLAMA_DIGEST
        assert call.observed_digest is None, "no inventory lookup was made for the control"
        assert call.think is None


def test_each_snapshot_routes_only_its_own_verifier_and_helpers():
    registry = _qualified_registry()
    gemma = _shared(registry, GEMMA_CANDIDATE, epoch=3, lookup=lambda tag: SYNTHETIC_DIGEST)
    llama = _shared(registry, LLAMA_CONTROL, epoch=1)
    assert gemma.verifier.model_tag == "gemma4:31b" and gemma.helper_call(0.2).model_tag == "gemma4:31b"
    assert llama.verifier.model_tag == "llama3.1:8b", "a job admitted under another epoch keeps its own model"


def test_role_calls_are_frozen_with_their_snapshot():
    snapshot = _shared(_qualified_registry(), LLAMA_CONTROL)
    verifier, helper = snapshot.verifier, snapshot.helper_call(0.2)
    # A later switch changes nothing already built.
    assert _shared(_qualified_registry(), GEMMA_CANDIDATE, epoch=3,
                   lookup=lambda tag: SYNTHETIC_DIGEST).verifier.model_tag == "gemma4:31b"
    assert snapshot.verifier is verifier and verifier.model_tag == "llama3.1:8b"
    assert snapshot.helper_call(0.2) == helper and helper.model_tag == "llama3.1:8b"
    with pytest.raises(dataclasses.FrozenInstanceError):
        verifier.model_tag = "gemma4:31b"
    with pytest.raises(dataclasses.FrozenInstanceError):
        verifier.read_timeout_s = 600.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.verifier = helper


_VALID_ROLE_CALL = dict(
    profile_id="candidate", model_tag="model:tag",
    options=GenerationOptions(temperature=0.0, num_ctx=8192, num_predict=16),
    keep_alive="5m", think=None, read_timeout_s=30.0,
    expected_digest=SYNTHETIC_DIGEST, observed_digest=None,
)


def test_a_valid_role_call_is_accepted_as_given():
    call = FixedModelCall(**_VALID_ROLE_CALL)
    assert call.ollama_payload("p") == {
        "model": "model:tag", "prompt": "p", "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 16}, "keep_alive": "5m",
    }


@pytest.mark.parametrize(
    "fields",
    [
        # True and the effort levels are sendable formats (a profile decides
        # with evidence whether it may send one); these are not.
        dict(think="false"),
        dict(think="HIGH"),
        dict(think="max"),
        dict(think=1),
        dict(think=0),
        dict(runtime_key="http://ollama.invalid:11434"),
        dict(runtime_key=""),
        dict(runtime_key=None),
        dict(read_timeout_s=0),
        dict(read_timeout_s=601),
        dict(read_timeout_s=math.inf),
        dict(read_timeout_s=True),
        dict(read_timeout_s=None),
        dict(expected_digest="sha256:" + SYNTHETIC_DIGEST),
        dict(observed_digest=SYNTHETIC_DIGEST.upper()),
        dict(observed_digest="not-a-digest"),
        dict(model_tag="llama3.1"),
        dict(keep_alive="-1"),
        dict(options={"temperature": 0.0}),
    ],
)
def test_an_invalid_role_call_is_rejected(fields):
    base = dict(_VALID_ROLE_CALL)
    base.update(fields)
    with pytest.raises(GenerationProfileError):
        FixedModelCall(**base)
