"""Contracts for the experimental generation tier (19 October 2026).

Pure units: generation_profiles.py and generation_admission.py only -- no
main import, database, model, network or subprocess.

What these prove:

* the three experimental profiles carry exactly the operator-supplied tags,
  digests and tested think settings on the compatibility runtime, and their
  runtime-smoke evidence (Mistral's REVIEW_REQUIRED verdict preserved) never
  sets a quality, capacity or vision gate;
* an experimental profile is a separate tier: it cannot carry a
  qualification, needs its smoke evidence, a pinned digest and an explicit job
  limit, and may send only the think value its smoke used;
* eligibility is the deployment's decision (default off; a missing runtime
  refuses), never a request's, and the existing Llama/Gemma eligibility,
  options and role calls are unchanged;
* every experimental answer, verifier and helper call is bound to the job's
  own model and runtime, needs an observed identity, and sends only the
  validated think value -- arbitrary strings, numbers and client controls are
  refused;
* Nemotron's document insights alone send think=false, and only with role
  evidence that exactly covers that profile, purpose, model, runtime build,
  call settings and the job's observed identity; missing or mismatched
  evidence refuses, and every other role and profile is unchanged;
* while an experimental profile is the shared selection the ledger admits one
  combined answer-or-helper job, applies that profile's budgets, and treats a
  failing policy resolver as one job at a time; Gemma's admission is
  unchanged.
"""

import dataclasses
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation_admission import (  # noqa: E402
    AdmissionRefusal,
    AdmissionRefused,
    AdmissionSettings,
    CallState,
    DispatchIntents,
    GenerationLedger,
    GenerationTicket,
    JobKind,
    SelectionPolicy,
    SharedSelection,
)
from generation_profiles import (  # noqa: E402
    COMPAT_RUNTIME,
    COMPAT_RUNTIME_TESTED_IMAGE_ID,
    GEMMA_CANDIDATE,
    GEMMA_CANDIDATE_DIGEST,
    INELIGIBLE_EXPERIMENTAL_DISABLED,
    INELIGIBLE_RUNTIME_NOT_CONFIGURED,
    LLAMA_CONTROL,
    MISTRAL35_DIGEST,
    MISTRAL35_EXPERIMENTAL,
    NEMOTRON35_DIGEST,
    NEMOTRON35_EXPERIMENTAL,
    PRIMARY_RUNTIME,
    QWEN38_DIGEST,
    QWEN38_EXPERIMENTAL,
    NEMOTRON35_INSIGHTS_THINK_FALSE_EVIDENCE,
    ExperimentalProfileDisabledError,
    FixedModelCall,
    GenerationOptions,
    GenerationProfileError,
    ModelIdentityMismatchError,
    ModelIdentityUnavailableError,
    ModelProfile,
    ModelRegistry,
    ModelRuntimeUnavailableError,
    Qualification,
    RolePolicy,
    RoleThinkEvidence,
    RuntimeSmokeEvidence,
    build_generate_payload,
    check_installed_identity,
    default_model_registry,
    require_think_value,
    select_shared_profile,
    shared_generation_snapshot,
)

BOTH_RUNTIMES = frozenset({PRIMARY_RUNTIME, COMPAT_RUNTIME})
EXPERIMENTAL_IDS = (QWEN38_EXPERIMENTAL, NEMOTRON35_EXPERIMENTAL, MISTRAL35_EXPERIMENTAL)
# (verifier, helper) output allowances, thinking included. Nemotron has its
# own after the bounded candidate check cut both of its role calls off at the
# shared 512 / 2,048; the other two keep the shared values.
_ROLE_BUDGETS = {
    QWEN38_EXPERIMENTAL: (512, 2048),
    NEMOTRON35_EXPERIMENTAL: (2048, 4096),
    MISTRAL35_EXPERIMENTAL: (512, 2048),
}


def _enabled():
    return default_model_registry(experimental_enabled=True, configured_runtimes=BOTH_RUNTIMES)


def _snapshot(registry, profile_id, epoch=7):
    profile = registry.get(profile_id)
    return shared_generation_snapshot(profile, epoch=epoch, observed_digest=profile.expected_digest,
                                      registry=registry)


# --- the registered experimental profiles ----------------------------------------


def test_the_three_profiles_carry_the_operator_supplied_identities_and_think_settings():
    registry = default_model_registry()
    expected = {
        QWEN38_EXPERIMENTAL: ("qwen3.8:27b", QWEN38_DIGEST, True, "PASS"),
        NEMOTRON35_EXPERIMENTAL: ("nemotron-3.5-lightning:30b", NEMOTRON35_DIGEST, True, "PASS"),
        MISTRAL35_EXPERIMENTAL: ("mistral-medium-3.5:128b", MISTRAL35_DIGEST, "high", "REVIEW_REQUIRED"),
    }
    assert QWEN38_DIGEST == "aaee06c39dcf2437cde036998d960e1fc1494b8191be7cc9657d01e509097813"
    assert NEMOTRON35_DIGEST == "e6905fffe2e2d9bbe325a9a2d2e074e3bd32683721bc6dfba409dede2f17f4dc"
    assert MISTRAL35_DIGEST == "0341632adb051badc332bd814779ab931ab78619750bebe3b7c7b1401e9d70df"
    for profile_id, (tag, digest, think, verdict) in expected.items():
        profile = registry.get(profile_id)
        assert (profile.model_tag, profile.expected_digest, profile.runtime_key) == (tag, digest, COMPAT_RUNTIME)
        assert profile.think == think and type(profile.think) is type(think)
        assert profile.experimental and profile.evidence_tier == "experimental"
        assert profile.runtime_smoke.verdict == verdict
        assert profile.runtime_smoke.tested_num_ctx == 2048
        assert profile.runtime_smoke.runtime_version == "0.32.12"
        assert profile.runtime_smoke.runtime_image_id == COMPAT_RUNTIME_TESTED_IMAGE_ID
        # No quality, capacity or vision gate is recorded or implied.
        assert profile.qualification is None and profile.requires_qualification is False
        assert profile.selectable is False, "never part of the qualified tier"
        assert profile.requires_identity_check is True, "identity is checked even though not qualified"
        assert profile.max_concurrent_jobs == 1
        assert profile.answer_options.num_ctx == 8192, "the application context, not the 2,048-token smoke"


def test_mistrals_original_verdict_and_review_note_are_preserved():
    smoke = default_model_registry().get(MISTRAL35_EXPERIMENTAL).runtime_smoke
    assert smoke.verdict == "REVIEW_REQUIRED"
    assert "91.27%" in smoke.review_note and "99%" in smoke.review_note
    assert "180.38" in smoke.review_note
    for text in (smoke.review_note, smoke.evidence_ref):
        assert "PASS" not in text.replace("passed", ""), "the evidence never claims a pass it did not have"


def test_nemotron_has_its_own_finite_role_allowances_and_keeps_thinking_on():
    """The candidate check (xm-20261020T030140Z-dc2e7a43) saw Nemotron's
    verifier generate exactly 512 tokens and its insights helper exactly 2,048
    with no usable reply. Only those two allowances change: thinking stays on
    for the answer, verifier and helper policy (the one value its smoke
    tested; document insights have their own evidenced policy, tested below),
    and the context, answer options, read timeouts and job budgets are the
    shared experimental ones."""
    nemotron = default_model_registry().get(NEMOTRON35_EXPERIMENTAL)
    assert nemotron.verifier_role() == RolePolicy(think=True, num_predict=2048)
    assert nemotron.helper_role() == RolePolicy(think=True, num_predict=4096)
    assert nemotron.think is True and nemotron.runtime_smoke.tested_think == (True,)
    for policy in (nemotron.verifier_role(), nemotron.helper_role()):
        assert policy.think is True, "no untested no-thinking or effort-level role call"
        assert policy.num_predict < nemotron.answer_options.num_ctx == 8192
        assert policy.num_predict <= 4096, "finite: within the registry's maximum output limit"
    assert (nemotron.answer_options.temperature, nemotron.answer_options.num_ctx,
            nemotron.answer_options.num_predict) == (0.2, 8192, 2048), "the answer keeps its settings"
    assert (nemotron.answer_read_timeout_s, nemotron.verifier_read_timeout_s,
            nemotron.helper_read_timeout_s) == (180.0, 180.0, 300.0), "deadlines unchanged"
    assert (nemotron.answer_job_budget_s, nemotron.helper_job_budget_s) == (None, None), (
        "the shared job budgets still bound each job"
    )


def test_a_role_call_describes_a_reply_with_its_own_profile_and_limit_never_its_text():
    snapshot = _snapshot(_enabled(), NEMOTRON35_EXPERIMENTAL)
    body = {"thinking": "SECRET-REASONING", "response": "", "done": True, "done_reason": "length",
            "prompt_eval_count": 1259, "eval_count": 2048}
    line = snapshot.verifier.describe_completion("verifier", body)
    assert line.startswith(
        f"[Generation] completion role=verifier profile={NEMOTRON35_EXPERIMENTAL} result=output_limit ")
    assert " num_predict=2048 at_output_limit=yes " in line and "SECRET" not in line
    helper_line = snapshot.helper_call(0.1).describe_completion("insights", body)
    assert " num_predict=4096 at_output_limit=no " in helper_line


@pytest.mark.parametrize("profile_id", [QWEN38_EXPERIMENTAL, MISTRAL35_EXPERIMENTAL])
def test_qwen_and_mistral_keep_the_shared_role_allowances(profile_id):
    profile = default_model_registry().get(profile_id)
    assert profile.verifier_role() == RolePolicy(think=profile.think, num_predict=512)
    assert profile.helper_role() == RolePolicy(think=profile.think, num_predict=2048)


@pytest.mark.parametrize("profile_id", [LLAMA_CONTROL, GEMMA_CANDIDATE])
def test_the_control_and_qualified_profiles_keep_their_role_calls(profile_id):
    profile = default_model_registry().get(profile_id)
    assert profile.verifier_policy is None and profile.helper_policy is None
    assert profile.verifier_role().num_predict == 16 and profile.helper_role().num_predict == 1200
    assert profile.verifier_role().think == profile.think == profile.helper_role().think


def test_mistral_gets_a_larger_bounded_read_timeout_and_job_budget():
    registry = default_model_registry()
    mistral, gemma = registry.get(MISTRAL35_EXPERIMENTAL), registry.get(GEMMA_CANDIDATE)
    assert mistral.answer_read_timeout_s > 180.38 > gemma.answer_read_timeout_s
    assert mistral.answer_read_timeout_s <= 600.0 and mistral.helper_read_timeout_s <= 600.0
    assert mistral.answer_job_budget_s == mistral.helper_job_budget_s == 900.0
    assert registry.get(QWEN38_EXPERIMENTAL).answer_job_budget_s is None, "the shared 420 s applies"


def test_the_existing_profiles_and_their_role_calls_are_unchanged():
    registry = _enabled()
    # The control makes no identity lookup, so it records no observed digest.
    llama = shared_generation_snapshot(registry.get(LLAMA_CONTROL), epoch=1, observed_digest=None, registry=registry)
    gemma = _snapshot(registry, GEMMA_CANDIDATE)
    assert (llama.think, llama.verifier.think, llama.helper_call(0.1).think) == (None, None, None)
    assert (gemma.think, gemma.verifier.think, gemma.helper_call(0.1).think) == (False, False, False)
    for snapshot in (llama, gemma):
        assert snapshot.runtime_key == snapshot.verifier.runtime_key == PRIMARY_RUNTIME
        assert snapshot.options.num_predict == 1200
        assert snapshot.verifier.options.num_predict == 16
        assert snapshot.helper_call(0.1).options.num_predict == 1200
        assert snapshot.public_metadata() == {"profile_id": snapshot.profile_id, "model": snapshot.model_tag}
        assert snapshot.experimental is False
    assert registry.admission_policy(GEMMA_CANDIDATE) == (None, None, None)
    assert registry.admission_policy(LLAMA_CONTROL) == (None, None, None)


# --- eligibility is the deployment's decision -----------------------------------------


def test_experimental_profiles_are_off_by_default_and_need_their_runtime():
    off = default_model_registry()
    disabled = default_model_registry(configured_runtimes=BOTH_RUNTIMES)
    no_runtime = default_model_registry(experimental_enabled=True)
    on = _enabled()
    for profile_id in EXPERIMENTAL_IDS:
        assert off.ineligible_reason(off.get(profile_id)) == INELIGIBLE_RUNTIME_NOT_CONFIGURED
        assert disabled.ineligible_reason(disabled.get(profile_id)) == INELIGIBLE_EXPERIMENTAL_DISABLED
        assert no_runtime.ineligible_reason(no_runtime.get(profile_id)) == INELIGIBLE_RUNTIME_NOT_CONFIGURED
        assert on.is_selectable(on.get(profile_id))
        with pytest.raises(ModelRuntimeUnavailableError):
            select_shared_profile(off, profile_id)
        with pytest.raises(ExperimentalProfileDisabledError):
            select_shared_profile(disabled, profile_id)
        assert select_shared_profile(on, profile_id) is on.get(profile_id)
    # The existing profiles stay eligible in every configuration.
    for registry in (off, disabled, no_runtime, on):
        assert registry.is_selectable(registry.get(LLAMA_CONTROL))
        assert registry.is_selectable(registry.get(GEMMA_CANDIDATE))


def test_a_profile_object_from_another_registry_is_not_eligible_here():
    on = _enabled()
    stranger = dataclasses.replace(on.get(QWEN38_EXPERIMENTAL))
    assert on.is_selectable(on.get(QWEN38_EXPERIMENTAL))
    assert not on.is_selectable(stranger), "only this registry's own registered profile is judged"


def test_the_default_profile_is_never_experimental_or_on_another_runtime():
    base = _enabled()
    with pytest.raises(GenerationProfileError):
        ModelRegistry(profiles=base.profiles, default_profile_id=QWEN38_EXPERIMENTAL,
                      experimental_enabled=True, configured_runtimes=BOTH_RUNTIMES)
    with pytest.raises(GenerationProfileError):
        ModelRegistry(profiles=base.profiles, default_profile_id=LLAMA_CONTROL,
                      configured_runtimes=frozenset({COMPAT_RUNTIME}))
    with pytest.raises(GenerationProfileError):
        ModelRegistry(profiles=base.profiles, default_profile_id=LLAMA_CONTROL,
                      configured_runtimes=frozenset({PRIMARY_RUNTIME, "http://ollama.invalid:11434"}))
    with pytest.raises(GenerationProfileError):
        ModelRegistry(profiles=base.profiles, default_profile_id=LLAMA_CONTROL, experimental_enabled="yes")


# --- the experimental tier is separate and evidence-bound --------------------------------


def _experimental(**changes):
    return dataclasses.replace(default_model_registry().get(QWEN38_EXPERIMENTAL), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        dict(qualification=Qualification(think_false_verified=True, combined_capacity_verified=True,
                                         quality_verified=True, evidence_ref="smuggled into the experimental tier")),
        dict(requires_qualification=True),
        dict(runtime_smoke=None),
        dict(expected_digest=None),
        dict(max_concurrent_jobs=None),
        dict(think="high"),  # Qwen's smoke used think=true, not an effort level
        dict(think=False),  # no no-thinking behaviour is evidenced
        dict(think="HIGH"),
        dict(think=1),
        dict(verifier_policy=RolePolicy(think="medium", num_predict=512)),
        dict(helper_policy=RolePolicy(think=False, num_predict=512)),
        dict(verifier_policy=RolePolicy(think="high", num_predict=512)),
        dict(runtime_key="http://ollama.invalid:11434"),
        dict(max_concurrent_jobs=0),
        dict(answer_job_budget_s=0.5),
        dict(answer_job_budget_s=True),
    ],
)
def test_an_invalid_experimental_profile_is_rejected(changes):
    with pytest.raises(GenerationProfileError):
        _experimental(**changes)


def test_a_non_experimental_profile_cannot_send_thinking_or_carry_smoke_evidence():
    gemma = default_model_registry().get(GEMMA_CANDIDATE)
    with pytest.raises(GenerationProfileError):
        dataclasses.replace(gemma, think=True)
    with pytest.raises(GenerationProfileError):
        dataclasses.replace(gemma, think="high")
    with pytest.raises(GenerationProfileError):
        dataclasses.replace(gemma, runtime_smoke=default_model_registry().get(QWEN38_EXPERIMENTAL).runtime_smoke)


def test_runtime_smoke_evidence_is_validated():
    smoke = default_model_registry().get(MISTRAL35_EXPERIMENTAL).runtime_smoke
    for changes in (dict(tested_think=()), dict(tested_think=("max",)), dict(verdict="FAIL"),
                    dict(review_note=None), dict(runtime_image_id="96e3cc4e"), dict(runtime_version="latest"),
                    dict(evidence_ref=" "), dict(tested_num_ctx=100)):
        with pytest.raises(GenerationProfileError):
            dataclasses.replace(smoke, **changes)
    assert smoke.tested("high") and not smoke.tested(True) and not smoke.tested("High")


@pytest.mark.parametrize("value", [None, False, True, "low", "medium", "high"])
def test_the_sendable_think_values(value):
    assert require_think_value("think", value) is value


@pytest.mark.parametrize("value", ["true", "false", "High", "max", "", 0, 1, 1.0, [], {"effort": "high"}])
def test_arbitrary_think_values_are_refused(value):
    with pytest.raises(GenerationProfileError):
        require_think_value("think", value)
    with pytest.raises(GenerationProfileError):
        build_generate_payload("qwen3.8:27b", "p", default_model_registry().get(QWEN38_EXPERIMENTAL).answer_options,
                               "5m", stream=True, think=value)


# --- snapshots bind the job's model and runtime ---------------------------------------


@pytest.mark.parametrize("profile_id", EXPERIMENTAL_IDS)
def test_every_role_of_an_experimental_job_uses_its_own_model_runtime_and_tested_think(profile_id):
    registry = _enabled()
    profile = registry.get(profile_id)
    snapshot = _snapshot(registry, profile_id)
    helper = snapshot.helper_call(0.1)
    assert snapshot.runtime_key == snapshot.verifier.runtime_key == helper.runtime_key == COMPAT_RUNTIME
    assert snapshot.model_tag == snapshot.verifier.model_tag == helper.model_tag == profile.model_tag
    assert snapshot.think is profile.think and snapshot.verifier.think is profile.think and helper.think is profile.think
    # A bounded thinking-plus-final allowance, never the 16-token status limit.
    assert (snapshot.verifier.options.num_predict, helper.options.num_predict) == _ROLE_BUDGETS[profile_id]
    assert snapshot.options.num_predict == 2048 and snapshot.options.num_ctx == 8192
    assert snapshot.observed_digest == snapshot.verifier.observed_digest == profile.expected_digest
    assert snapshot.public_metadata() == {"profile_id": profile_id, "model": profile.model_tag, "experimental": True}
    payloads = (snapshot.ollama_payload("q"), snapshot.verifier.ollama_payload("q"), helper.ollama_payload("q"))
    for payload in payloads:
        assert payload["think"] is profile.think
        assert payload["model"] == profile.model_tag
        assert set(payload) == {"model", "prompt", "stream", "options", "keep_alive", "think"}
        assert "://" not in repr(payload), "no endpoint ever travels in a payload"


def test_an_experimental_snapshot_needs_the_deployment_registry_and_its_observed_identity():
    registry = _enabled()
    profile = registry.get(NEMOTRON35_EXPERIMENTAL)
    with pytest.raises(ExperimentalProfileDisabledError):
        shared_generation_snapshot(profile, epoch=2, observed_digest=profile.expected_digest)
    with pytest.raises(ModelIdentityUnavailableError):
        shared_generation_snapshot(profile, epoch=2, observed_digest=None, registry=registry)
    with pytest.raises(ModelIdentityUnavailableError):
        shared_generation_snapshot(profile, epoch=2, observed_digest="77" * 32, registry=registry)
    disabled = default_model_registry(configured_runtimes=BOTH_RUNTIMES)
    with pytest.raises(ExperimentalProfileDisabledError):
        shared_generation_snapshot(disabled.get(NEMOTRON35_EXPERIMENTAL), epoch=2,
                                   observed_digest=profile.expected_digest, registry=disabled)


def test_the_identity_check_reads_the_lookup_it_is_given_and_never_another_profile():
    profile = _enabled().get(MISTRAL35_EXPERIMENTAL)
    asked = []
    assert check_installed_identity(profile, lambda tag: asked.append(tag) or "sha256:" + MISTRAL35_DIGEST.upper()) \
        == MISTRAL35_DIGEST
    assert asked == ["mistral-medium-3.5:128b"]
    with pytest.raises(ModelIdentityMismatchError):
        check_installed_identity(profile, lambda tag: None)
    with pytest.raises(ModelIdentityMismatchError):
        check_installed_identity(profile, lambda tag: GEMMA_CANDIDATE_DIGEST)


def test_a_snapshot_refuses_a_verifier_from_another_model_or_runtime():
    snapshot = _snapshot(_enabled(), QWEN38_EXPERIMENTAL)
    for verifier in (dataclasses.replace(snapshot.verifier, runtime_key=PRIMARY_RUNTIME),
                     dataclasses.replace(snapshot.verifier, model_tag="gemma4:31b"),
                     dataclasses.replace(snapshot.verifier, profile_id=GEMMA_CANDIDATE)):
        with pytest.raises(GenerationProfileError):
            dataclasses.replace(snapshot, verifier=verifier)
    with pytest.raises(GenerationProfileError):
        dataclasses.replace(snapshot, runtime_key="elsewhere")


def test_a_role_call_validates_its_runtime_key():
    call = _snapshot(_enabled(), QWEN38_EXPERIMENTAL).verifier
    assert isinstance(call, FixedModelCall)
    for bad in ("", None, "http://compat:11434", "PRIMARY"):
        with pytest.raises(GenerationProfileError):
            dataclasses.replace(call, runtime_key=bad)


# --- Nemotron's document insights: think=false on its own role evidence ------------------
#
# Diagnostic pr58-prod-nemotron-diag-20261022T071320Z (VALID, THINK_FALSE): with thinking on, its
# insights helper used 3,454-3,942 of 4,096 tokens on one document; with think=false (C1, C2) it
# stopped after 336 and 332 tokens with usable JSON. Only that purpose changes.

# What C1 and C2 sent, from their call records (the prompt left out).
_C_REQUEST = {"keep_alive": "5m", "model": "nemotron-3.5-lightning:30b",
              "options": {"num_ctx": 8192, "num_predict": 4096, "temperature": 0.1},
              "stream": False, "think": False}


def _nemotron():
    return default_model_registry().get(NEMOTRON35_EXPERIMENTAL)


def test_nemotron_insights_send_exactly_the_evidenced_request_from_the_captured_job():
    snapshot = _snapshot(_enabled(), NEMOTRON35_EXPERIMENTAL, epoch=7)
    call = snapshot.insights_call(0.1)
    payload = call.ollama_payload("PROMPT", stream=False)
    assert payload.pop("prompt") == "PROMPT"
    assert payload == _C_REQUEST
    assert (call.profile_id, call.runtime_key, call.read_timeout_s) == (NEMOTRON35_EXPERIMENTAL, COMPAT_RUNTIME, 300.0)
    assert call.expected_digest == call.observed_digest == NEMOTRON35_DIGEST, "the identity this job observed"
    assert (snapshot.selection, snapshot.selection_epoch) == ("shared_active", 7)


def test_only_nemotron_insights_change_and_nemotron_stays_experimental():
    nemotron = _nemotron()
    assert nemotron.insights_policy == RolePolicy(think=False, num_predict=4096)
    assert nemotron.insights_evidence is NEMOTRON35_INSIGHTS_THINK_FALSE_EVIDENCE
    snapshot = _snapshot(_enabled(), NEMOTRON35_EXPERIMENTAL)
    kb = snapshot.helper_call(0.2)
    assert snapshot.ollama_payload("q")["think"] is True, "the answer keeps thinking on"
    assert (snapshot.verifier.think, snapshot.verifier.options.num_predict) == (True, 2048)
    assert (kb.think, kb.options.num_predict, kb.read_timeout_s) == (True, 4096, 300.0), "the KB helper too"
    # No qualification, never selectable as qualified; the runtime smoke record is untouched.
    assert nemotron.experimental and nemotron.qualification is None and nemotron.selectable is False
    assert nemotron.think is True and nemotron.runtime_smoke.tested_think == (True,)
    assert "think=true" in nemotron.runtime_smoke.evidence_ref
    evidence = nemotron.insights_evidence
    for fact in ("7246da7d235f18e17e37a0b89c993d10bbf315a794055e3bc7f54780fe35f3f8", "VALID", "THINK_FALSE",
                 "336 and 332", "One document only", "not a quality, capacity or acceptance result"):
        assert fact in evidence.evidence_ref
    # The evidence names a reviewable record kept in the repository.
    record = "diagnostics/NEMOTRON_INSIGHTS_THINK_FALSE_20261022.md"
    assert record in evidence.evidence_ref
    assert os.path.isfile(os.path.join(os.path.dirname(__file__), "..", *record.split("/")))


@pytest.mark.parametrize("profile_id", [LLAMA_CONTROL, GEMMA_CANDIDATE, QWEN38_EXPERIMENTAL, MISTRAL35_EXPERIMENTAL])
def test_every_other_profiles_insights_call_is_its_helper_call(profile_id):
    registry = _enabled()
    profile = registry.get(profile_id)
    assert profile.insights_policy is None and profile.insights_evidence is None
    observed = profile.expected_digest if profile.requires_identity_check else None
    snapshot = shared_generation_snapshot(profile, epoch=2, observed_digest=observed, registry=registry)
    for temperature in (0.1, 0.2):
        assert snapshot.insights_call(temperature) == snapshot.helper_call(temperature)


def test_think_false_for_insights_needs_its_role_evidence_and_relaxes_nothing_else():
    nemotron = _nemotron()
    with pytest.raises(GenerationProfileError, match="go together"):
        dataclasses.replace(nemotron, insights_evidence=None)
    with pytest.raises(GenerationProfileError, match="go together"):
        dataclasses.replace(nemotron, insights_policy=None)
    # The answer, verifier and helper keep the profile's own think rules.
    for changes in (dict(think=False), dict(verifier_policy=RolePolicy(think=False, num_predict=2048)),
                    dict(helper_policy=RolePolicy(think=False, num_predict=4096))):
        with pytest.raises(GenerationProfileError, match="think: false may only be sent"):
            dataclasses.replace(nemotron, **changes)
    # No other profile can borrow it.
    override = dict(insights_policy=nemotron.insights_policy, insights_evidence=nemotron.insights_evidence)
    for profile_id in (QWEN38_EXPERIMENTAL, MISTRAL35_EXPERIMENTAL):
        with pytest.raises(GenerationProfileError, match="does not cover its profile_id, model_tag, digest"):
            dataclasses.replace(default_model_registry().get(profile_id), **override)
    for profile_id in (LLAMA_CONTROL, GEMMA_CANDIDATE):
        with pytest.raises(GenerationProfileError, match="belongs to an experimental profile"):
            dataclasses.replace(default_model_registry().get(profile_id), **override)


def test_role_evidence_is_validated():
    evidence = NEMOTRON35_INSIGHTS_THINK_FALSE_EVIDENCE
    for changes in (dict(purpose="kb_qa"), dict(purpose="answer"), dict(think=None), dict(think="max"),
                    dict(digest="e6905fff"), dict(model_tag="nemotron"), dict(runtime_key="http://compat:11434"),
                    dict(options={"temperature": 0.1}), dict(runtime_version=" "), dict(evidence_ref="")):
        with pytest.raises(GenerationProfileError):
            dataclasses.replace(evidence, **changes)
    assert isinstance(evidence, RoleThinkEvidence) and evidence.digest == NEMOTRON35_DIGEST


@pytest.mark.parametrize(
    "changes, uncovered",
    [
        (dict(profile_id=QWEN38_EXPERIMENTAL), "profile_id"),
        (dict(model_tag="qwen3.8:27b"), "model_tag"),
        (dict(digest=QWEN38_DIGEST), "digest"),
        (dict(runtime_key=PRIMARY_RUNTIME), "runtime_key"),
        (dict(runtime_version="0.32.13"), "runtime_version"),
        (dict(runtime_image_id="sha256:" + "ab" * 32), "runtime_image_id"),
        (dict(think=True), "think"),
        (dict(think="high"), "think"),
        (dict(options=GenerationOptions(temperature=0.1, num_ctx=4096, num_predict=2048)), "num_ctx"),
        (dict(options=GenerationOptions(temperature=0.1, num_ctx=8192, num_predict=2048)), "num_predict"),
    ],
)
def test_mismatched_insights_evidence_refuses_the_profile(changes, uncovered):
    nemotron = _nemotron()
    evidence = dataclasses.replace(nemotron.insights_evidence, **changes)
    with pytest.raises(GenerationProfileError, match=f"does not cover its .*{uncovered}"):
        dataclasses.replace(nemotron, insights_evidence=evidence)


@pytest.mark.parametrize(
    "changes, uncovered",
    [
        (dict(expected_digest=QWEN38_DIGEST), "digest"),
        (dict(model_tag="nemotron-3.5-lightning:30b-q8"), "model_tag"),
        (dict(runtime_key=PRIMARY_RUNTIME), "runtime_key"),
        (dict(answer_options=GenerationOptions(temperature=0.2, num_ctx=16384, num_predict=2048)), "num_ctx"),
        (dict(insights_policy=RolePolicy(think=False, num_predict=2048)), "num_predict"),
        (dict(insights_policy=RolePolicy(think=True, num_predict=4096)), "think"),
    ],
)
def test_a_profile_its_insights_evidence_no_longer_covers_is_refused(changes, uncovered):
    with pytest.raises(GenerationProfileError, match=f"does not cover its .*{uncovered}"):
        dataclasses.replace(_nemotron(), **changes)


@pytest.mark.parametrize("build", [dict(runtime_version="0.33.0"), dict(runtime_image_id="sha256:" + "cd" * 32)])
def test_insights_evidence_from_another_runtime_build_is_refused(build):
    nemotron = _nemotron()
    smoke = dataclasses.replace(nemotron.runtime_smoke, **build)
    with pytest.raises(GenerationProfileError, match=f"does not cover its .*{next(iter(build))}"):
        dataclasses.replace(nemotron, runtime_smoke=smoke)


def test_a_snapshot_carries_the_insights_policy_only_with_evidence_that_covers_it():
    snapshot = _snapshot(_enabled(), NEMOTRON35_EXPERIMENTAL)
    assert snapshot.insights_policy == RolePolicy(think=False, num_predict=4096)
    assert snapshot.insights_evidence is NEMOTRON35_INSIGHTS_THINK_FALSE_EVIDENCE
    for changes in (dict(insights_evidence=None), dict(insights_policy=None), dict(experimental=False),
                    dict(insights_evidence=dataclasses.replace(snapshot.insights_evidence, digest=QWEN38_DIGEST)),
                    dict(insights_policy=RolePolicy(think=False, num_predict=2048))):
        with pytest.raises(GenerationProfileError):
            dataclasses.replace(snapshot, **changes)


def test_an_insights_call_outside_its_evidence_is_refused_never_sent_another_way():
    snapshot = _snapshot(_enabled(), NEMOTRON35_EXPERIMENTAL)
    for temperature in (0.0, 0.2, 1.0):
        with pytest.raises(GenerationProfileError, match="options or identity"):
            snapshot.insights_call(temperature)
    for observed in (None, QWEN38_DIGEST):
        with pytest.raises(GenerationProfileError, match="options or identity"):
            dataclasses.replace(snapshot, observed_digest=observed).insights_call(0.1)
    assert snapshot.helper_call(0.2).think is True, "the KB helper is unaffected"


# --- admission while an experimental profile is the shared selection --------------------


class _Intents(DispatchIntents):
    def record_dispatch(self, intent):
        pass

    def record_outcome(self, intent, state):
        pass


def _ledger(profile_id, *, settings=None, policy=None, recovered=None, now=None):
    registry = _enabled()

    def from_registry(selection):
        values = registry.admission_policy(selection.profile_id)
        if values is None or values == (None, None, None):
            return None
        return SelectionPolicy(*values)

    clock = (lambda: now[0]) if now is not None else time.monotonic
    ledger = GenerationLedger(settings or AdmissionSettings(), clock=clock,
                              selection_policy=policy or from_registry)
    assert ledger.install_owner(intents=_Intents(), owner_seq=1, selection=SharedSelection(profile_id, 3),
                                paused_by=None, recovered=recovered or {})
    return ledger


def _start_answer(ledger):
    ticket = ledger.enqueue_answer(lambda: None)
    assert isinstance(ticket, GenerationTicket) and ledger.start(ticket)
    return ticket


@pytest.mark.parametrize("profile_id", EXPERIMENTAL_IDS)
def test_one_combined_answer_or_helper_job_at_a_time(profile_id):
    ledger = _ledger(profile_id)
    assert ledger.snapshot().selection_job_limit == 1
    answer = _start_answer(ledger)
    assert ledger.try_acquire_helper() is None, "a helper waits while the one answer runs"
    waiter = ledger.enqueue_answer(lambda: None)
    assert not isinstance(waiter, GenerationTicket), "a second answer waits (bounded), it is not admitted"
    assert ledger.settle_answer_waiter(waiter) is None
    assert ledger.release(answer) is CallState.NEVER_DISPATCHED
    helper = ledger.try_acquire_helper()
    assert helper is not None and ledger.helper_free() == 0
    waiter = ledger.enqueue_answer(lambda: None)
    assert not isinstance(waiter, GenerationTicket), "an answer waits while the one helper runs"
    ledger.settle_answer_waiter(waiter)
    ledger.release(helper)


def test_gemma_keeps_the_shared_two_answer_admission():
    ledger = _ledger(GEMMA_CANDIDATE)
    assert ledger.snapshot().selection_job_limit is None
    first, second = _start_answer(ledger), _start_answer(ledger)
    assert first.kind is second.kind is JobKind.ANSWER
    for ticket in (first, second):
        ledger.release(ticket)


def test_an_unobserved_experimental_call_holds_the_only_slot():
    ledger = _ledger(QWEN38_EXPERIMENTAL)
    answer = _start_answer(ledger)
    answer.bind_snapshot(_snapshot(_enabled(), QWEN38_EXPERIMENTAL, epoch=3))
    answer.dispatching("answer")
    assert ledger.release(answer) is CallState.UNKNOWN
    with pytest.raises(AdmissionRefused) as refused:
        ledger.enqueue_answer(lambda: None)
    assert refused.value.reason is AdmissionRefusal.UNRESOLVED
    assert ledger.try_acquire_helper() is None


def test_recovered_unresolved_work_blocks_an_experimental_selection_after_a_restart():
    ledger = _ledger(MISTRAL35_EXPERIMENTAL, recovered={JobKind.HELPER: 1})
    with pytest.raises(AdmissionRefused) as refused:
        ledger.enqueue_answer(lambda: None)
    assert refused.value.reason is AdmissionRefusal.UNRESOLVED


def test_the_profiles_job_budgets_set_the_deadline():
    now = [1000.0]
    ledger = _ledger(MISTRAL35_EXPERIMENTAL, now=now)
    answer = _start_answer(ledger)
    assert answer.deadline == pytest.approx(1900.0), "Mistral's 900 s answer budget"
    ledger.release(answer)
    helper = ledger.try_acquire_helper()
    assert helper.deadline == pytest.approx(1900.0)
    ledger.release(helper)
    qwen = _ledger(QWEN38_EXPERIMENTAL, now=now)
    answer = _start_answer(qwen)
    assert answer.deadline == pytest.approx(1420.0), "no override: the shared 420 s"
    qwen.release(answer)


def test_a_failing_or_malformed_policy_resolver_admits_one_job_at_a_time():
    def broken(selection):
        raise RuntimeError("SECRET-DETAIL")

    for policy in (broken, lambda selection: {"max_jobs": 99}):
        ledger = _ledger(GEMMA_CANDIDATE, policy=policy)
        assert ledger.snapshot().selection_job_limit == 1
        answer = _start_answer(ledger)
        assert ledger.try_acquire_helper() is None
        ledger.release(answer)


def test_a_policy_never_widens_the_shared_settings():
    ledger = _ledger(GEMMA_CANDIDATE, settings=AdmissionSettings(total_slots=1, answer_slots=1, helper_slots=1),
                     policy=lambda selection: SelectionPolicy(max_jobs=8))
    answer = _start_answer(ledger)
    assert ledger.try_acquire_helper() is None
    ledger.release(answer)


def test_resuming_with_another_selection_switches_the_policy():
    ledger = _ledger(GEMMA_CANDIDATE)
    assert ledger.pause("switch-1") == SharedSelection(GEMMA_CANDIDATE, 3)
    assert ledger.resume("switch-1", SharedSelection(QWEN38_EXPERIMENTAL, 5))
    assert ledger.snapshot().selection_job_limit == 1
    assert ledger.pause("switch-2") is not None
    assert ledger.resume("switch-2", SharedSelection(GEMMA_CANDIDATE, 7))
    assert ledger.snapshot().selection_job_limit is None


@pytest.mark.parametrize("values", [dict(max_jobs=0), dict(max_jobs=True), dict(answer_job_budget_s=0.1),
                                    dict(helper_job_budget_s=float("inf"))])
def test_an_invalid_selection_policy_is_rejected(values):
    with pytest.raises(ValueError):
        SelectionPolicy(**values)
