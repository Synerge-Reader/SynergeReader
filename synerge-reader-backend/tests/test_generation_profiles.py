"""Registry, option validation and selection for answer-generation profiles.

Pure: generation_profiles.py does no I/O, and every runtime fact a selection
needs (admin authority, installed digest) is injected here as a callable.
Gate behaviour is exercised with synthetic digests on constructed profiles;
only the production-constant tests compare against the operator-supplied pins.
"""

import dataclasses
import inspect
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation_profiles import (
    GEMMA_CANDIDATE,
    LLAMA_CONTROL,
    SMOKE_OPTIONS,
    FixedModelCall,
    GenerationOptions,
    GenerationProfileError,
    ModelIdentityMismatchError,
    ModelIdentityUnavailableError,
    ModelProfile,
    ModelRegistry,
    OverrideNotPermittedError,
    ProfileNotQualifiedError,
    Qualification,
    SelectionAuthorityUnavailableError,
    UnknownModelProfileError,
    default_model_registry,
    resolve_answer_generation,
    resolve_default_generation,
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


def _registry(gemma, default=LLAMA_CONTROL):
    return ModelRegistry(
        profiles=(default_model_registry().get(LLAMA_CONTROL), gemma),
        default_profile_id=default,
    )


def _qualified_registry(default=LLAMA_CONTROL):
    return _registry(_gemma(qualification=_evidence()), default=default)


def _resolve(registry, profile_id, *, admin=True, digest=None):
    return resolve_answer_generation(
        registry,
        requested_profile_id=profile_id,
        caller_is_admin=lambda: admin,
        installed_digest=lambda tag: digest,
    )


# --- the production registry --------------------------------------------------


def test_production_registry_has_the_two_installed_profiles_and_llama_default():
    registry = default_model_registry()
    assert registry.default.profile_id == LLAMA_CONTROL
    assert registry.get(LLAMA_CONTROL).model_tag == "llama3.1:8b"
    assert registry.get(GEMMA_CANDIDATE).model_tag == "gemma4:31b"
    assert [p.profile_id for p in registry.profiles] == [LLAMA_CONTROL, GEMMA_CANDIDATE]
    assert not any("qwen" in p.model_tag.lower() for p in registry.profiles)


def test_production_pins_equal_the_operator_supplied_digests():
    registry = default_model_registry()
    assert registry.get(LLAMA_CONTROL).expected_digest == OPERATOR_LLAMA_DIGEST
    assert registry.get(GEMMA_CANDIDATE).expected_digest == OPERATOR_GEMMA_DIGEST


def test_production_qualification_policy_is_explicit_and_records_only_evidenced_gates():
    registry = default_model_registry()
    llama, gemma = registry.get(LLAMA_CONTROL), registry.get(GEMMA_CANDIDATE)

    assert llama.requires_qualification is False and llama.qualification is None
    assert llama.think is None, "Llama never receives a think field"

    assert gemma.requires_qualification is True
    assert gemma.qualification.think_false_verified is True
    assert gemma.qualification.combined_capacity_verified is False
    assert gemma.qualification.quality_verified is False
    assert gemma.think is False, "think:false is evidenced, so it is the Gemma policy"
    assert not gemma.selectable and not registry.is_selectable(gemma)


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


def test_production_gemma_stays_disabled_for_an_admin_too():
    registry = default_model_registry()
    with pytest.raises(ProfileNotQualifiedError):
        resolve_answer_generation(
            registry, requested_profile_id=GEMMA_CANDIDATE,
            caller_is_admin=lambda: True, installed_digest=_never,
        )


def test_verifier_and_helpers_are_bounded_llama_calls_without_think():
    snapshot = resolve_answer_generation(
        default_model_registry(), requested_profile_id=None,
        caller_is_admin=_never, installed_digest=_never,
    )
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
    ids=["no-evidence", "no-capacity", "no-quality", "production-shape"],
)
def test_an_incomplete_candidate_cannot_be_constructed_as_default(qualification):
    think = False if qualification is not None else None
    with pytest.raises(GenerationProfileError):
        _registry(_gemma(qualification=qualification, think=think), default=GEMMA_CANDIDATE)


def test_the_production_gemma_cannot_be_made_default():
    registry = default_model_registry()
    with pytest.raises(GenerationProfileError):
        ModelRegistry(
            profiles=registry.profiles,
            default_profile_id=GEMMA_CANDIDATE,
        )


def test_an_incomplete_default_is_refused_at_resolution_even_if_construction_was_bypassed():
    registry = _registry(_gemma())
    # Simulate a registry whose invariant was broken after construction.
    object.__setattr__(registry, "default_profile_id", GEMMA_CANDIDATE)
    with pytest.raises(ProfileNotQualifiedError):
        resolve_answer_generation(
            registry, requested_profile_id=None,
            caller_is_admin=_never, installed_digest=_never,
        )


# --- selection: the omitted-profile default -----------------------------------


def test_the_llama_default_needs_no_admin_and_no_inventory_lookup():
    snapshot = resolve_answer_generation(
        default_model_registry(),
        requested_profile_id=None,
        caller_is_admin=_never,
        installed_digest=_never,
    )
    assert snapshot.profile_id == LLAMA_CONTROL and snapshot.selection == "default"
    assert snapshot.answer_read_timeout_s == 60.0
    assert snapshot.ollama_payload("p") == {
        "model": "llama3.1:8b", "prompt": "p", "stream": True,
        "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1200},
        "keep_alive": "5m",
    }
    assert snapshot.public_metadata() == {"profile_id": LLAMA_CONTROL, "model": "llama3.1:8b"}


def test_expected_and_observed_digests_are_never_conflated():
    llama = resolve_answer_generation(
        default_model_registry(), requested_profile_id=None,
        caller_is_admin=_never, installed_digest=_never,
    )
    assert llama.expected_digest == OPERATOR_LLAMA_DIGEST
    assert llama.observed_digest is None, "a pin is not an observation for this answer"
    assert "digest" not in str(llama.public_metadata())

    gemma = resolve_answer_generation(
        _qualified_registry(default=GEMMA_CANDIDATE), requested_profile_id=None,
        caller_is_admin=_never, installed_digest=lambda tag: "sha256:" + SYNTHETIC_DIGEST.upper(),
    )
    assert gemma.expected_digest == SYNTHETIC_DIGEST
    assert gemma.observed_digest == SYNTHETIC_DIGEST, "the normalized value the lookup reported"


def test_a_qualified_candidate_default_needs_no_admin_but_its_exact_digest():
    lookups = []

    def installed(tag):
        lookups.append(tag)
        return SYNTHETIC_DIGEST

    snapshot = resolve_answer_generation(
        _qualified_registry(default=GEMMA_CANDIDATE), requested_profile_id=None,
        caller_is_admin=_never, installed_digest=installed,
    )
    assert lookups == ["gemma4:31b"]
    assert snapshot.profile_id == GEMMA_CANDIDATE and snapshot.selection == "default"
    assert snapshot.answer_read_timeout_s == 150.0
    assert snapshot.ollama_payload("p")["think"] is False
    # The answer is verified by the model that wrote it, with no second lookup.
    assert snapshot.verifier.model_tag == "gemma4:31b"
    assert snapshot.verifier.observed_digest == SYNTHETIC_DIGEST


@pytest.mark.parametrize(
    "lookup, error",
    [
        (lambda tag: None, ModelIdentityMismatchError),
        (lambda tag: OTHER_DIGEST, ModelIdentityMismatchError),
        (lambda tag: "not-a-digest", ModelIdentityUnavailableError),
        (lambda tag: (_ for _ in ()).throw(ConnectionError("ollama down")), ModelIdentityUnavailableError),
    ],
    ids=["absent", "changed", "malformed", "unreadable"],
)
def test_a_qualified_default_with_the_wrong_identity_fails_closed(lookup, error):
    with pytest.raises(error) as excinfo:
        resolve_answer_generation(
            _qualified_registry(default=GEMMA_CANDIDATE), requested_profile_id=None,
            caller_is_admin=_never, installed_digest=lookup,
        )
    assert "ollama down" not in str(excinfo.value)


# --- selection: explicit profiles ------------------------------------------------


@pytest.mark.parametrize("profile_id", [LLAMA_CONTROL, GEMMA_CANDIDATE, "qwen_anything", "x"])
def test_a_non_admin_explicit_profile_is_refused_before_the_registry_is_consulted(profile_id):
    with pytest.raises(OverrideNotPermittedError):
        resolve_answer_generation(
            _qualified_registry(), requested_profile_id=profile_id,
            caller_is_admin=lambda: False, installed_digest=_never,
        )


@pytest.mark.parametrize("admin_value", [1, "true", None])
def test_only_a_real_true_admin_flag_counts(admin_value):
    with pytest.raises(OverrideNotPermittedError):
        _resolve(default_model_registry(), LLAMA_CONTROL, admin=admin_value)


def test_a_failed_admin_lookup_fails_closed():
    def broken():
        raise RuntimeError("db down")

    with pytest.raises(SelectionAuthorityUnavailableError) as excinfo:
        resolve_answer_generation(
            default_model_registry(), requested_profile_id=LLAMA_CONTROL,
            caller_is_admin=broken, installed_digest=_never,
        )
    assert "db down" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__


def test_admin_unknown_profile_is_refused():
    with pytest.raises(UnknownModelProfileError):
        _resolve(default_model_registry(), "qwen3_latest")


def test_admin_may_explicitly_choose_the_control_without_an_inventory_lookup():
    snapshot = resolve_answer_generation(
        default_model_registry(), requested_profile_id=LLAMA_CONTROL,
        caller_is_admin=lambda: True, installed_digest=_never,
    )
    assert snapshot.profile_id == LLAMA_CONTROL and snapshot.selection == "admin_override"
    assert snapshot.observed_digest is None


def test_an_explicit_qualified_candidate_needs_its_exact_installed_digest():
    registry = _qualified_registry()

    snapshot = _resolve(registry, GEMMA_CANDIDATE, digest=SYNTHETIC_DIGEST)
    assert snapshot.model_tag == "gemma4:31b" and snapshot.selection == "admin_override"
    assert snapshot.observed_digest == SYNTHETIC_DIGEST
    payload = snapshot.ollama_payload("p")
    assert payload["think"] is False
    assert payload["options"] == {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1200}
    verifier = snapshot.verifier.ollama_payload("p")
    assert verifier["model"] == "gemma4:31b" and verifier["think"] is False

    with pytest.raises(ModelIdentityMismatchError):
        _resolve(registry, GEMMA_CANDIDATE, digest=None)
    with pytest.raises(ModelIdentityMismatchError):
        _resolve(registry, GEMMA_CANDIDATE, digest=OTHER_DIGEST)


def test_a_snapshot_is_immutable_and_independent_of_later_registry_changes():
    snapshot = _resolve(_qualified_registry(), None)
    gemma_default = _qualified_registry(default=GEMMA_CANDIDATE)
    assert _resolve(gemma_default, None, digest=SYNTHETIC_DIGEST).model_tag == "gemma4:31b"
    assert snapshot.model_tag == "llama3.1:8b"
    assert snapshot.ollama_payload("p")["model"] == "llama3.1:8b"
    assert snapshot.answer_read_timeout_s == 60.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.model_tag = "gemma4:31b"
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.answer_read_timeout_s = 600.0


# --- role calls: the verifier and helpers follow the resolved profile ---------


def test_a_qualified_candidate_verifier_has_the_answers_identity_and_bounded_settings():
    snapshot = _resolve(_qualified_registry(), GEMMA_CANDIDATE, digest="sha256:" + SYNTHETIC_DIGEST.upper())

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
    snapshot = resolve_answer_generation(
        default_model_registry(), requested_profile_id=None,
        caller_is_admin=_never, installed_digest=_never,
    )
    for call in (snapshot.verifier, snapshot.helper_call(0.2)):
        assert call.profile_id == LLAMA_CONTROL and call.model_tag == "llama3.1:8b"
        assert call.expected_digest == OPERATOR_LLAMA_DIGEST
        assert call.observed_digest is None, "no inventory lookup was made for the control"
        assert call.think is None


def test_an_admin_override_routes_only_that_answers_verifier():
    registry = _qualified_registry()
    override = _resolve(registry, GEMMA_CANDIDATE, digest=SYNTHETIC_DIGEST)
    default = _resolve(registry, None)
    assert override.verifier.model_tag == "gemma4:31b"
    assert default.verifier.model_tag == "llama3.1:8b", "another answer keeps the default's verifier"
    helper_default = resolve_default_generation(registry, installed_digest=_never)
    assert helper_default.profile_id == LLAMA_CONTROL and helper_default.selection == "default"


def test_role_calls_are_frozen_with_their_snapshot():
    snapshot = _resolve(_qualified_registry(), None)
    verifier, helper = snapshot.verifier, snapshot.helper_call(0.2)
    # A later registry with a different default changes nothing already resolved.
    replaced = _qualified_registry(default=GEMMA_CANDIDATE)
    assert _resolve(replaced, None, digest=SYNTHETIC_DIGEST).verifier.model_tag == "gemma4:31b"
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
        dict(think=True),
        dict(think="false"),
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


# --- helper jobs: the trusted default, resolved once at job start ------------


def test_helper_resolution_takes_no_requested_profile_or_caller():
    assert list(inspect.signature(resolve_default_generation).parameters) == ["registry", "installed_digest"]


def test_the_control_default_for_helpers_needs_no_admin_and_no_inventory():
    snapshot = resolve_default_generation(default_model_registry(), installed_digest=_never)
    assert snapshot.profile_id == LLAMA_CONTROL and snapshot.selection == "default"
    assert snapshot.observed_digest is None
    assert snapshot.helper_call(0.1).read_timeout_s == 120.0


def test_a_qualified_candidate_default_for_helpers_needs_its_exact_digest():
    lookups = []

    def installed(tag):
        lookups.append(tag)
        return "sha256:" + SYNTHETIC_DIGEST.upper()

    snapshot = resolve_default_generation(_qualified_registry(default=GEMMA_CANDIDATE), installed_digest=installed)
    assert lookups == ["gemma4:31b"], "one inventory read, at job start"
    assert snapshot.observed_digest == SYNTHETIC_DIGEST
    helper = snapshot.helper_call(0.2).ollama_payload("p")
    assert helper["model"] == "gemma4:31b" and helper["think"] is False


@pytest.mark.parametrize(
    "lookup, error",
    [
        (lambda tag: None, ModelIdentityMismatchError),
        (lambda tag: OTHER_DIGEST, ModelIdentityMismatchError),
        (lambda tag: "not-a-digest", ModelIdentityUnavailableError),
        (lambda tag: (_ for _ in ()).throw(ConnectionError("ollama down")), ModelIdentityUnavailableError),
    ],
    ids=["absent", "changed", "malformed", "unreadable"],
)
def test_a_qualified_helper_default_with_the_wrong_identity_fails_closed(lookup, error):
    with pytest.raises(error) as excinfo:
        resolve_default_generation(_qualified_registry(default=GEMMA_CANDIDATE), installed_digest=lookup)
    assert "ollama down" not in str(excinfo.value)


def test_an_ineligible_helper_default_is_refused_before_any_lookup():
    registry = _registry(_gemma())
    # Simulate a registry whose invariant was broken after construction.
    object.__setattr__(registry, "default_profile_id", GEMMA_CANDIDATE)
    with pytest.raises(ProfileNotQualifiedError):
        resolve_default_generation(registry, installed_digest=_never)
