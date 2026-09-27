"""The deterministic half of claim verification: a green badge fails closed.

Offline and model-free. The verifier is injected as an in-memory fake, so
nothing here reaches Ollama, a database, the network, or the filesystem.

What these prove: a claim is judged only against the evidence IT cited; a
compound claim is not fully supported when only part of it is evidenced; a
topically similar but factually different passage can never produce a green
result; and correct, directly grounded claims are still allowed through. The
floor only ever lowers a status, so it cannot manufacture support.

What these do NOT prove: anything about the local model's own judgement. The
floor is what remains true regardless of what Llama 3.1 8B answers.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from answer_evidence import EvidenceBundle, EvidenceItem, EvidenceMode
from citation_generation import (
    CitationLimits,
    CitationRegistry,
    ClaimReason,
    ClaimVerificationStatus,
    apply_support_floor,
    generate_citations,
    salient_terms,
    verify_claims,
)


def _item(text, **kwargs):
    defaults = dict(
        source_type="document_chunk",
        document_id=1,
        filename="attention.pdf",
        chunk_id="1-0",
        chunk_index=0,
    )
    defaults.update(kwargs)
    return EvidenceItem(text=text, **defaults)


def _registry(*texts):
    items = tuple(
        _item(text, chunk_id=f"1-{i}", chunk_index=i) for i, text in enumerate(texts)
    )
    return CitationRegistry.from_bundle(
        EvidenceBundle(mode=EvidenceMode.HYBRID_RETRIEVAL, items=items)
    )


def _always_supported(claim, records):
    """The failure mode being defended against: a verifier that says yes."""
    return "supported"


# --- what counts as a checkable term ---------------------------------------


def test_numbers_and_distinctive_names_are_checkable():
    terms = salient_terms("PaLM-2L improved by 27% on STEM using 8 GPUs [C1].")

    assert "#27" in terms
    assert "#8" in terms
    assert "@palm-2l" in terms, "a hyphenated model name is one checkable name"
    assert "@stem" in terms
    assert "@gpu" in terms, "a pluralised acronym normalises to its singular"


def test_ordinary_capitalised_words_are_not_treated_as_evidence():
    """Otherwise every sentence-initial word would reject a correct claim."""
    terms = salient_terms("The model was trained on a large corpus.")

    assert terms == set(), (
        "plain prose asserts nothing checkable; the verifier decides it"
    )


def test_written_numerals_match_printed_digits():
    assert "#8" in salient_terms("eight attention heads")
    assert "#8" in salient_terms("h = 8")


def test_citation_markers_are_not_mistaken_for_names():
    assert salient_terms("A plain claim [C1].") == set()


def test_grouping_commas_do_not_change_a_number_but_decimals_do():
    assert salient_terms("100,000 steps") == salient_terms("100000 steps")
    assert salient_terms("3.5 days") != salient_terms("35 days")


# --- 7: topical similarity can never be green ------------------------------


def test_a_topically_similar_passage_cannot_produce_a_green_result():
    """The live IRCoT false positive: right topic, wrong page, marked green."""
    registry = _registry(
        "IRCoT is evaluated on four datasets and compared against one-step "
        "retrieval baselines across several model sizes."
    )

    result = generate_citations(
        "IRCoT requires a retriever, a reasoner, and a maximum of 8 steps [C1].",
        registry,
        verifier=_always_supported,
    )

    assert result.claims[0].status is not ClaimVerificationStatus.SUPPORTED
    assert result.claims[0].reason is ClaimReason.EVIDENCE_TERMS_MISSING


def test_a_claim_whose_numbers_are_absent_is_unsupported_not_partly():
    registry = _registry("The system alternates retrieval and reasoning steps.")

    claims = verify_claims(
        ["The limit is 8 reasoning steps and 15 paragraphs [C1]."],
        registry,
        verifier=_always_supported,
    )

    assert claims[0].status is ClaimVerificationStatus.UNSUPPORTED


def test_a_wrong_number_is_never_green():
    """A near miss is a miss: 15 paragraphs is not 50 paragraphs.

    The floor reports this as partly supported rather than unsupported: the
    subject matched and only the value did not, and this round deliberately
    does not try to tell a contradicted number from an absent one. What it
    guarantees is the part that matters -- the claim cannot be green.
    """
    registry = _registry("IRCoT collects at most 50 paragraphs in total.")

    claims = verify_claims(
        ["IRCoT collects at most 15 paragraphs [C1]."],
        registry,
        verifier=_always_supported,
    )

    assert claims[0].status is not ClaimVerificationStatus.SUPPORTED
    assert claims[0].status is ClaimVerificationStatus.PARTIALLY_SUPPORTED
    assert claims[0].reason is ClaimReason.EVIDENCE_TERMS_MISSING


# --- 6: a compound claim needs every component -----------------------------


def test_a_compound_claim_is_only_partly_supported_when_half_is_evidenced():
    registry = _registry(
        "We trained the base model for 100,000 steps on the described hardware."
    )

    claims = verify_claims(
        ["The base model trained for 100,000 steps on 8 P100 GPUs [C1]."],
        registry,
        verifier=_always_supported,
    )

    assert claims[0].status is ClaimVerificationStatus.PARTIALLY_SUPPORTED
    assert claims[0].reason is ClaimReason.EVIDENCE_TERMS_MISSING


def test_every_component_present_stays_supported():
    registry = _registry(
        "The base model was trained for 100,000 steps, about 12 hours, on "
        "8 NVIDIA P100 GPUs."
    )

    claims = verify_claims(
        ["The base model trained for 100,000 steps on 8 P100 GPUs [C1]."],
        registry,
        verifier=_always_supported,
    )

    assert claims[0].status is ClaimVerificationStatus.SUPPORTED
    assert claims[0].reason is ClaimReason.VERIFIED


# --- 8: correct, grounded claims are not falsely rejected ------------------


def test_the_eight_head_claim_is_not_falsely_rejected():
    """The reported false negative. 'eight' and 'h = 8' are the same fact."""
    registry = _registry(
        "In this work we employ h = 8 parallel attention layers, or heads."
    )

    claims = verify_claims(
        ["The model uses eight attention heads [C1]."],
        registry,
        verifier=_always_supported,
    )

    assert claims[0].status is ClaimVerificationStatus.SUPPORTED
    assert claims[0].reason is ClaimReason.VERIFIED


def test_the_dimension_claim_is_not_falsely_rejected():
    registry = _registry(
        "For each of these we use dmodel/h = 64 as the key and value dimension."
    )

    claims = verify_claims(
        ["Each head uses a dimension of 64 [C1]."],
        registry,
        verifier=_always_supported,
    )

    assert claims[0].status is ClaimVerificationStatus.SUPPORTED


def test_the_training_cost_claim_is_not_falsely_rejected():
    registry = _registry(
        "The big model was trained for 300,000 steps, about 3.5 days on "
        "8 P100 GPUs."
    )

    claims = verify_claims(
        ["The big model took about 3.5 days over 300,000 steps [C1]."],
        registry,
        verifier=_always_supported,
    )

    assert claims[0].status is ClaimVerificationStatus.SUPPORTED


# --- the floor only ever lowers -------------------------------------------


def test_the_floor_never_raises_a_status():
    registry = _registry("The base model trained for 100,000 steps on 8 P100 GPUs.")
    records = registry.records

    for start in (
        ClaimVerificationStatus.UNSUPPORTED,
        ClaimVerificationStatus.PARTIALLY_SUPPORTED,
    ):
        status, lowered = apply_support_floor(
            "The base model trained for 100,000 steps on 8 P100 GPUs [C1].",
            records,
            start,
        )
        assert status is start
        assert lowered is False


def test_an_unverified_status_is_left_alone():
    registry = _registry("Unrelated text.")
    status, lowered = apply_support_floor(
        "A claim about 8 GPUs [C1].", registry.records, ClaimVerificationStatus.UNVERIFIED
    )

    assert status is ClaimVerificationStatus.UNVERIFIED
    assert lowered is False


def test_generic_framing_is_left_to_the_verifier():
    """Framing asserts nothing checkable, so the floor must not judge it."""
    registry = _registry("IRCoT interleaves retrieval with chain-of-thought.")

    claims = verify_claims(
        [
            "IRCoT and Step-Back Prompting differ in their approach to "
            "improving multi-step reasoning [C1]."
        ],
        registry,
        verifier=_always_supported,
    )

    assert claims[0].status is ClaimVerificationStatus.SUPPORTED


# --- the claim is judged only against what it cited ------------------------


def test_evidence_the_claim_did_not_cite_cannot_rescue_it():
    registry = _registry(
        "Unrelated discussion of checkpoint averaging.",
        "The base model trained for 100,000 steps on 8 P100 GPUs.",
    )

    claims = verify_claims(
        ["The base model trained for 100,000 steps on 8 P100 GPUs [C1]."],
        registry,
        verifier=_always_supported,
    )

    assert claims[0].citation_ids == ("C1",)
    assert claims[0].status is ClaimVerificationStatus.UNSUPPORTED, (
        "C2 holds the support, but the claim cited C1; uncited evidence must "
        "not certify it"
    )


def test_citing_the_right_block_of_several_is_supported():
    registry = _registry(
        "Unrelated discussion of checkpoint averaging.",
        "The base model trained for 100,000 steps on 8 P100 GPUs.",
    )

    claims = verify_claims(
        ["The base model trained for 100,000 steps on 8 P100 GPUs [C2]."],
        registry,
        verifier=_always_supported,
    )

    assert claims[0].status is ClaimVerificationStatus.SUPPORTED


# --- 9: a missing marker stays visibly unsupported -------------------------


def test_a_claim_with_no_marker_never_silently_acquires_a_source():
    registry = _registry("The base model trained for 100,000 steps on 8 P100 GPUs.")

    result = generate_citations(
        "The base model trained for 100,000 steps on 8 P100 GPUs.",
        registry,
        verifier=_always_supported,
    )

    assert result.claims[0].citation_ids == ()
    assert result.claims[0].status is ClaimVerificationStatus.UNSUPPORTED
    assert result.claims[0].reason is ClaimReason.NO_CITATION
    assert result.used_citation_ids == (), (
        "an uncited answer presents no sources, however well the evidence "
        "would have supported it"
    )


def test_the_floor_does_not_disturb_the_verification_budget():
    registry = _registry("The term is three years.")

    result = generate_citations(
        "The term is three years [C1]. The term is three years [C1]. "
        "The term is three years [C1].",
        registry,
        verifier=_always_supported,
        limits=CitationLimits(max_claims_verified=2),
    )

    statuses = [claim.status for claim in result.claims]
    assert statuses[:2] == [ClaimVerificationStatus.SUPPORTED] * 2
    assert statuses[2] is ClaimVerificationStatus.UNVERIFIED
    assert result.claims[2].reason is ClaimReason.NOT_VERIFIED
