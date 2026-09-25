"""Unit contracts for E2 citation generation.

Route-independent: ``citation_generation.py`` imports no route, opens no
connection, and contacts no model. The claim verifier is injected here as an
in-memory fake, so nothing in this file reaches Ollama, a database, the
network, a subprocess, or the filesystem.

What these prove: identifiers are deterministic, filename and locator metadata
survive registration, a missing page never becomes a fabricated page, excerpts
are bounded, duplicate evidence registers once, only registry ids resolve,
claims without a valid citation are never "supported", verifier failure yields
"unverified", and safe errors carry no exception, SQL, credential, or document
text.

What these do NOT prove: how the evidence was chosen (test_answer_evidence.py),
how it was retrieved (test_hybrid_retrieval.py), or how it is transported
(test_main_citation_wiring.py).
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from answer_evidence import EvidenceBundle, EvidenceItem, EvidenceMode
from citation_generation import (
    ABSENCE_RULE_COMPLETE_DOCUMENT,
    ABSENCE_RULE_PARTIAL_EVIDENCE,
    CITATION_MARKER_PATTERN,
    CITATION_RULES,
    MODEL_REASONING_NO_EVIDENCE_INSTRUCTION,
    PASSAGE_LOCATION_LABEL,
    STRUCTURED_NO_EVIDENCE_INSTRUCTION,
    AnswerMode,
    CitationLimits,
    CitationRegistry,
    ClaimReason,
    ClaimVerificationStatus,
    EvidenceBudgetExceeded,
    LOCATION_UNAVAILABLE_LABEL,
    SAFE_ERROR_MESSAGES,
    absence_rule,
    bounded_excerpt,
    build_evidence_blocks,
    build_generation_prompt,
    build_mode_prompt,
    citations_without_claims,
    evidence_scope_statement,
    extract_claims,
    generate_citations,
    normalize_locator,
    parse_citation_markers,
    prompt_evidence_chars,
    safe_error,
    verify_claims,
)


def _item(text="evidence text", **kwargs):
    defaults = dict(
        source_type="document_chunk",
        document_id=1,
        filename="agreement.pdf",
        chunk_id="1-0",
        chunk_index=0,
    )
    defaults.update(kwargs)
    return EvidenceItem(text=text, **defaults)


def _bundle(*items, mode=EvidenceMode.HYBRID_RETRIEVAL):
    return EvidenceBundle(mode=mode, items=tuple(items))


# --- 11: deterministic identifiers -----------------------------------------


def test_stable_evidence_produces_stable_citation_identifiers():
    bundle = _bundle(
        _item("first", chunk_id="1-0", chunk_index=0),
        _item("second", chunk_id="1-4", chunk_index=4),
        _item("third", document_id=2, filename="statute.pdf", chunk_id="2-1", chunk_index=1),
    )

    first = CitationRegistry.from_bundle(bundle)
    second = CitationRegistry.from_bundle(bundle)

    assert first.citation_ids == ("C1", "C2", "C3")
    assert first.citation_ids == second.citation_ids
    assert [record.excerpt for record in first.records] == [
        record.excerpt for record in second.records
    ]
    assert first.get("C2").excerpt == "second"


# --- 12: metadata survives registration ------------------------------------


def test_filename_and_locator_metadata_survive_registration():
    bundle = _bundle(
        _item(
            "the passage",
            page_start=4,
            page_end=4,
            locator_json={"type": "pdf_pages", "pages": [4]},
            semantic_score=0.81,
            lexical_score=0.22,
            combined_score=0.44,
        )
    )

    record = CitationRegistry.from_bundle(bundle).records[0]

    assert record.filename == "agreement.pdf"
    assert record.document_id == 1
    assert record.chunk_id == "1-0"
    assert record.chunk_index == 0
    assert record.page_start == 4 and record.page_end == 4
    assert record.locator.kind == "page" and record.locator.label == "page 4"
    assert record.display_label == "agreement.pdf · page 4"
    payload = record.to_dict()
    assert payload["scores"] == {"semantic": 0.81, "lexical": 0.22, "combined": 0.44}
    assert payload["filename"] == "agreement.pdf"


def test_page_range_is_reported_as_a_range():
    record = CitationRegistry.from_bundle(
        _bundle(_item("x", page_start=4, page_end=6))
    ).records[0]
    assert record.locator.kind == "page_range"
    assert record.locator.label == "pages 4-6"


def test_display_source_never_falls_back_to_a_bare_document_id():
    record = CitationRegistry.from_bundle(
        _bundle(_item("x", filename=None, source_type="selection", document_id=7))
    ).records[0]

    assert record.display_source == "Your highlighted text"
    assert "7" not in record.display_source
    assert record.to_dict()["display_source"] == "Your highlighted text"


# --- 13: no fabricated pages -----------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected_kind", "expected_label"),
    [
        ({"locator_json": {"paragraph": 12}}, "paragraph", "paragraph 12"),
        ({"locator_json": {"line_start": 3, "line_end": 9}}, "line_range", "lines 3-9"),
        ({"locator_json": {"line_start": 3}}, "line", "line 3"),
        ({"chunk_index": 5}, "passage", "Relevant passage"),
        ({"source_type": "selection", "chunk_index": None, "chunk_id": None}, "selection", "highlighted selection"),
        (
            {"source_type": "complete_document", "chunk_index": None, "chunk_id": None},
            "document",
            "complete document",
        ),
        ({"chunk_index": None, "chunk_id": None}, "unavailable", LOCATION_UNAVAILABLE_LABEL),
    ],
)
def test_missing_page_information_never_creates_a_fake_page(kwargs, expected_kind, expected_label):
    locator = normalize_locator(_item("x", **kwargs), EvidenceMode.HYBRID_RETRIEVAL)

    assert locator.kind == expected_kind
    assert locator.label == expected_label
    assert locator.page_start is None and locator.page_end is None
    assert "page" not in locator.label or expected_kind.startswith("page")


def test_zero_and_negative_page_numbers_are_not_treated_as_pages():
    record = CitationRegistry.from_bundle(_bundle(_item("x", page_start=0, chunk_index=2))).records[0]
    assert record.page_start is None
    assert record.locator.kind == "passage"


# --- 14: bounded excerpts ---------------------------------------------------


def test_citation_excerpts_obey_the_configured_limit():
    limits = CitationLimits(max_excerpt_chars=40)
    record = CitationRegistry.from_bundle(_bundle(_item("word " * 200)), limits).records[0]

    assert len(record.excerpt) <= 40, (
        "the limit is a hard ceiling: the ellipsis counts against it"
    )
    assert record.excerpt_truncated is True
    assert record.excerpt.endswith("…")


@pytest.mark.parametrize("limit", [0, 1, 2, 3, 17, 40, 480])
def test_bounded_excerpt_never_exceeds_a_non_negative_limit(limit):
    for text in ("", "   ", "short", "word " * 200, "x" * limit if limit else "x"):
        excerpt, _ = bounded_excerpt(text, limit)
        assert len(excerpt) <= limit, (
            f"limit {limit} produced {len(excerpt)} characters for {text[:20]!r}"
        )


def test_bounded_excerpt_zero_limit_yields_nothing_but_reports_truncation():
    assert bounded_excerpt("some text", 0) == ("", True)
    assert bounded_excerpt("", 0) == ("", False)
    assert bounded_excerpt("   ", 0) == ("", False), (
        "whitespace-only text normalises to empty, so nothing was truncated"
    )
    assert bounded_excerpt("text", -5) == ("", True), "a negative limit yields nothing"


def test_bounded_excerpt_limit_of_one_is_just_the_ellipsis():
    assert bounded_excerpt("a longer sentence", 1) == ("…", True)
    assert bounded_excerpt("a", 1) == ("a", False), (
        "text that already fits is returned untouched, ellipsis or not"
    )


def test_bounded_excerpt_at_exactly_the_limit_is_untouched():
    text = "x" * 40
    assert bounded_excerpt(text, 40) == (text, False)
    excerpt, truncated = bounded_excerpt("x" * 41, 40)
    assert truncated is True
    assert len(excerpt) == 40
    assert excerpt == "x" * 39 + "…"


def test_bounded_excerpt_strips_whitespace_at_the_truncation_boundary():
    # The cut lands on the space before "beta", which must not be left dangling
    # in front of the ellipsis.
    excerpt, truncated = bounded_excerpt("alpha beta gamma", 7)

    assert truncated is True
    assert excerpt == "alpha…", "trailing whitespace is stripped before the ellipsis"
    assert len(excerpt) <= 7


def test_short_excerpts_are_normalised_but_not_marked_truncated():
    record = CitationRegistry.from_bundle(_bundle(_item("  spaced   out\n text "))).records[0]
    assert record.excerpt == "spaced out text"
    assert record.excerpt_truncated is False


# --- 15: duplicate evidence registers once ---------------------------------


def test_duplicate_evidence_does_not_create_duplicate_citations():
    bundle = _bundle(
        _item("same passage", chunk_id="1-2", chunk_index=2),
        _item("same passage", chunk_id="1-2", chunk_index=2),
        _item("other", chunk_id="1-3", chunk_index=3),
    )

    registry = CitationRegistry.from_bundle(bundle)

    assert registry.citation_ids == ("C1", "C2")
    assert registry.get("C2").excerpt == "other"


# --- prompt construction ----------------------------------------------------


def test_prompt_contains_one_labelled_block_per_citation():
    registry = CitationRegistry.from_bundle(
        _bundle(
            _item("first passage", page_start=4),
            _item("second passage", chunk_id="1-9", chunk_index=9, page_start=None),
        )
    )

    blocks = build_evidence_blocks(registry)
    prompt = build_generation_prompt("What is the term?", registry)

    assert "[C1]\nDocument: agreement.pdf\nLocation: page 4\nEvidence:\nfirst passage" in blocks
    assert "[C2]\nDocument: agreement.pdf\nLocation: Relevant passage" in blocks
    assert blocks in prompt
    assert "What is the term?" in prompt
    for rule in (
        "Use ONLY the evidence blocks",
        "immediately after each material claim",
        "Never invent a citation id",
        "say what is missing",
        "Do not overstate",
        "not definitive legal advice",
    ):
        assert rule.lower() in prompt.lower(), f"the prompt must state: {rule}"


def test_prompt_without_evidence_forbids_citations_and_admits_the_gap():
    prompt = build_generation_prompt("q", CitationRegistry())

    assert "No document evidence is available" in prompt
    assert "Do not output any citation ids" in prompt
    assert "[C1]" not in prompt


def test_prompt_is_pure_and_carries_the_task_prefix_and_extra_context():
    registry = CitationRegistry.from_bundle(_bundle(_item("passage")))
    first = build_generation_prompt("q", registry, extra_context="KB", task_prefix="Summarize.")
    second = build_generation_prompt("q", registry, extra_context="KB", task_prefix="Summarize.")

    assert first == second
    assert first.startswith("Summarize.")
    assert "KB" in first


# --- 16/17: marker resolution ----------------------------------------------


def test_valid_model_citation_markers_resolve():
    registry = CitationRegistry.from_bundle(_bundle(_item("a"), _item("b", chunk_id="1-1", chunk_index=1)))

    parsed = parse_citation_markers("The term is three years [C1]. Notice is 30 days [C2].", registry)

    assert parsed.valid_ids == ("C1", "C2")
    assert parsed.invalid_ids == ()
    assert registry.get("C1") is not None


def test_unknown_citation_markers_are_rejected_and_never_resolve():
    registry = CitationRegistry.from_bundle(_bundle(_item("a")))

    parsed = parse_citation_markers("Invented support [C7] and [C2].", registry)

    assert parsed.valid_ids == ()
    assert sorted(parsed.invalid_ids) == ["C2", "C7"]
    assert registry.get("C7") is None
    assert registry.is_valid("C7") is False


def test_repeated_markers_are_reported_once_and_in_order():
    registry = CitationRegistry.from_bundle(_bundle(_item("a"), _item("b", chunk_id="1-1", chunk_index=1)))
    parsed = parse_citation_markers("[C2] then [C1] then [C2] again", registry)
    assert parsed.valid_ids == ("C2", "C1")


def test_marker_pattern_does_not_match_ordinary_bracketed_text():
    assert not CITATION_MARKER_PATTERN.findall("[Clause 4] and [see page 3]")


# --- 18/19: claim status ----------------------------------------------------


def test_claims_without_citations_are_not_supported():
    registry = CitationRegistry.from_bundle(_bundle(_item("a")))

    claims = verify_claims(
        ["The agreement is governed by Delaware law."],
        registry,
        verifier=lambda claim, records: "supported",
    )

    assert claims[0].status is ClaimVerificationStatus.UNSUPPORTED
    assert claims[0].status is not ClaimVerificationStatus.SUPPORTED
    assert claims[0].reason is ClaimReason.NO_CITATION
    assert claims[0].citation_ids == ()


def test_a_claim_citing_only_an_unknown_id_is_not_supported():
    registry = CitationRegistry.from_bundle(_bundle(_item("a")))

    claims = verify_claims(["Fabricated [C9]."], registry, verifier=lambda c, r: "supported")

    assert claims[0].status is ClaimVerificationStatus.UNSUPPORTED
    assert claims[0].reason is ClaimReason.INVALID_CITATION
    assert claims[0].invalid_citation_ids == ("C9",)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("supported", ClaimVerificationStatus.SUPPORTED),
        ("Partially Supported", ClaimVerificationStatus.PARTIALLY_SUPPORTED),
        ("unsupported", ClaimVerificationStatus.UNSUPPORTED),
        ({"status": "partially_supported"}, ClaimVerificationStatus.PARTIALLY_SUPPORTED),
    ],
)
def test_verifier_statuses_remain_distinguishable(response, expected):
    registry = CitationRegistry.from_bundle(_bundle(_item("a")))
    claims = verify_claims(["A cited claim [C1]."], registry, verifier=lambda c, r: response)
    assert claims[0].status is expected


@pytest.mark.parametrize(
    "response",
    ["", "yes", "true", None, {"status": "definitely"}, 42, {"verdict": "supported"}, "unverified"],
)
def test_malformed_verifier_responses_produce_unverified(response):
    registry = CitationRegistry.from_bundle(_bundle(_item("a")))

    claims = verify_claims(["A cited claim [C1]."], registry, verifier=lambda c, r: response)

    assert claims[0].status is ClaimVerificationStatus.UNVERIFIED
    assert claims[0].reason is ClaimReason.VERIFIER_MALFORMED


def test_verifier_failure_produces_unverified_without_exposing_the_exception():
    registry = CitationRegistry.from_bundle(_bundle(_item("a")))

    def exploding(claim, records):
        raise RuntimeError("ollama at 10.0.0.5 refused: token=hunter2")

    claims = verify_claims(["A cited claim [C1]."], registry, verifier=exploding)

    assert claims[0].status is ClaimVerificationStatus.UNVERIFIED
    assert claims[0].reason is ClaimReason.VERIFIER_UNAVAILABLE
    rendered = str(claims[0].to_dict())
    assert "hunter2" not in rendered and "10.0.0.5" not in rendered


def test_absent_verifier_leaves_cited_claims_unverified_never_supported():
    registry = CitationRegistry.from_bundle(_bundle(_item("a")))
    claims = verify_claims(["A cited claim [C1]."], registry, verifier=None)
    assert claims[0].status is ClaimVerificationStatus.UNVERIFIED
    assert claims[0].reason is ClaimReason.NOT_VERIFIED


def test_verification_budget_marks_the_remainder_unverified_not_supported():
    registry = CitationRegistry.from_bundle(_bundle(_item("a")))
    limits = CitationLimits(max_claims_verified=1)

    claims = verify_claims(
        ["First [C1].", "Second [C1].", "Third [C1]."],
        registry,
        verifier=lambda c, r: "supported",
        limits=limits,
    )

    assert claims[0].status is ClaimVerificationStatus.SUPPORTED
    assert [claim.status for claim in claims[1:]] == [ClaimVerificationStatus.UNVERIFIED] * 2


def test_extract_claims_splits_sentences_and_ignores_bare_markers():
    claims = extract_claims("The term is three years [C1]. Notice is 30 days [C2].\n[C1]\n- A bullet [C1]")

    assert claims[0].startswith("The term is three years")
    assert any("Notice is 30 days" in claim for claim in claims)
    assert any("A bullet" in claim for claim in claims)
    assert "[C1]" not in [claim.strip() for claim in claims]


def test_generate_citations_reports_used_and_invalid_ids_together():
    registry = CitationRegistry.from_bundle(_bundle(_item("a")))

    result = generate_citations(
        "Supported point [C1]. Invented point [C4].",
        registry,
        verifier=lambda c, r: "supported",
    )

    assert result.used_citation_ids == ("C1",)
    assert result.invalid_citation_ids == ("C4",)
    statuses = [claim.status for claim in result.claims]
    assert ClaimVerificationStatus.SUPPORTED in statuses
    assert ClaimVerificationStatus.UNSUPPORTED in statuses
    payload = result.to_dict()
    assert payload["citations"][0]["citation_id"] == "C1"
    assert payload["invalid_citation_ids"] == ["C4"]


# --- 20: safe errors --------------------------------------------------------


def test_safe_errors_contain_no_raw_exception_sql_or_credentials():
    for code in SAFE_ERROR_MESSAGES:
        error = safe_error(code)
        assert set(error) == {"code", "message"}
        rendered = error["message"].lower()
        for leak in ("traceback", "select ", "insert ", "psycopg2", "password", "token=", "://"):
            assert leak not in rendered
        assert len(error["message"]) < 200


def test_unknown_error_codes_cannot_smuggle_text_into_a_response():
    error = safe_error("SELECT * FROM users; password=hunter2")

    assert error["code"] == "internal_error"
    assert "hunter2" not in error["message"]
    assert error["message"] == SAFE_ERROR_MESSAGES["internal_error"]


def test_registry_payload_carries_no_unbounded_document_text():
    limits = CitationLimits(max_excerpt_chars=60)
    registry = CitationRegistry.from_bundle(_bundle(_item("secret clause " * 500)), limits)

    payload = registry.to_dict()

    assert len(payload["citations"][0]["excerpt"]) <= 60
    assert payload["contract_version"].startswith("e2.citation")


# --- Correction 1: internal model evidence vs the bounded public excerpt ----
#
# The registry carries two representations of the same finalized evidence: the
# public ``excerpt`` the UI shows, and the internal ``evidence_text`` the model
# and the verifier read. These tests pin the seam in both directions -- the
# full text must reach the prompt, and it must never reach the client.


_TAIL_MARKER = "ZZ-UNIQUE-TAIL-MARKER-7f3a"
_MID_MARKER = "ZZ-UNIQUE-MID-MARKER-91c4"


def _long_document_text(total_chars=11_800):
    """A complete-document body near the 12,000-character planner threshold,
    with unique markers in its middle and final thirds."""
    filler = "clause text. "
    body = (filler * ((total_chars // len(filler)) + 1))[:total_chars]
    third = total_chars // 3
    body = body[:third] + _MID_MARKER + body[third + len(_MID_MARKER):]
    body = body[: total_chars - len(_TAIL_MARKER) - 1] + " " + _TAIL_MARKER
    return body


def test_complete_document_evidence_reaches_the_prompt_in_full():
    text = _long_document_text()
    bundle = _bundle(
        _item(text, source_type="complete_document", chunk_id=None, chunk_index=None),
        mode=EvidenceMode.COMPLETE_DOCUMENT,
    )

    registry = CitationRegistry.from_bundle(bundle)
    prompt = build_generation_prompt("What does the agreement say?", registry)

    assert _MID_MARKER in prompt, "evidence from the middle of the document is missing"
    assert _TAIL_MARKER in prompt, (
        "evidence from the final third never reached the model -- the prompt was "
        "built from the bounded UI excerpt instead of the finalized evidence"
    )
    assert registry.records[0].evidence_text == text
    assert _TAIL_MARKER not in registry.records[0].excerpt


def test_long_selected_text_reaches_the_prompt_without_an_ellipsis_tail():
    selection = ("The parties agree as follows. " * 40) + _TAIL_MARKER
    assert len(selection) > 480
    bundle = _bundle(
        _item(selection, source_type="selection", filename=None, chunk_id=None, chunk_index=None),
        mode=EvidenceMode.SELECTED_TEXT,
    )

    registry = CitationRegistry.from_bundle(bundle)
    blocks = build_evidence_blocks(registry)

    assert selection in blocks, "the complete finalized selection must reach the model"
    assert _TAIL_MARKER in blocks
    assert "…" not in blocks, (
        "an excerpt ellipsis must never stand in for the tail of the evidence "
        "the model is asked to reason over"
    )
    assert registry.records[0].excerpt.endswith("…"), (
        "the public excerpt is still bounded, unlike the model evidence"
    )


def test_prompt_evidence_total_stays_within_the_configured_ceiling():
    limits = CitationLimits()
    items = [
        _item("x" * 2000, chunk_id=f"1-{index}", chunk_index=index)
        for index in range(8)
    ]
    registry = CitationRegistry.from_bundle(_bundle(*items), limits)

    assert prompt_evidence_chars(registry.records) == 16_000
    assert prompt_evidence_chars(registry.records) <= limits.max_prompt_evidence_chars
    assert limits.max_prompt_evidence_chars == 16_000


def test_an_over_budget_bundle_is_rejected_not_quietly_trimmed():
    limits = CitationLimits(max_prompt_evidence_chars=1000)
    bundle = _bundle(_item("x" * 1001))

    with pytest.raises(EvidenceBudgetExceeded):
        CitationRegistry.from_bundle(bundle, limits)

    # And a registry assembled directly is still refused at prompt time, rather
    # than being silently reduced to fit.
    loose = CitationRegistry.from_bundle(bundle, CitationLimits())
    over_budget = CitationRegistry(records=loose.records, limits=limits)
    with pytest.raises(EvidenceBudgetExceeded):
        build_evidence_blocks(over_budget)


def test_full_evidence_never_appears_in_any_frontend_facing_payload():
    text = _long_document_text()
    bundle = _bundle(
        _item(text, source_type="complete_document", chunk_id=None, chunk_index=None),
        mode=EvidenceMode.COMPLETE_DOCUMENT,
    )
    registry = CitationRegistry.from_bundle(bundle)
    result = generate_citations("A point [C1].", registry, verifier=lambda c, r: "supported")

    record_payload = registry.records[0].to_dict()
    assert "evidence_text" not in record_payload, (
        "the internal model evidence must not be serialized to the client"
    )
    assert set(record_payload) == {
        "citation_id", "document_id", "filename", "display_source", "display_label",
        "chunk_id", "chunk_index", "page_start", "page_end", "locator",
        "evidence_mode", "source_type", "excerpt", "excerpt_truncated", "scores",
    }

    for payload in (record_payload, registry.to_dict(), result.to_dict()):
        rendered = str(payload)
        assert _TAIL_MARKER not in rendered, (
            "text past the excerpt bound leaked into a client-facing payload"
        )
        assert _MID_MARKER not in rendered
        assert len(rendered) < 4000, "a client payload must not carry document-scale text"


def test_the_verifier_receives_evidence_beyond_the_excerpt_bound():
    text = _long_document_text()
    bundle = _bundle(
        _item(text, source_type="complete_document", chunk_id=None, chunk_index=None),
        mode=EvidenceMode.COMPLETE_DOCUMENT,
    )
    registry = CitationRegistry.from_bundle(bundle)
    seen = []

    def recording_verifier(claim, records):
        seen.append("\n".join(record.evidence_text for record in records))
        return "supported"

    claims = verify_claims(["A cited claim [C1]."], registry, verifier=recording_verifier)

    assert claims[0].status is ClaimVerificationStatus.SUPPORTED
    assert _TAIL_MARKER in seen[0], (
        "a claim supported by text past character 480 would be misjudged if the "
        "verifier only ever saw the bounded excerpt"
    )
    # The verifier's public result stays bounded citation data only.
    assert _TAIL_MARKER not in str(claims[0].to_dict())


# --- E2b: truthful TXT locator, clean generation, clean claim extraction ----


def test_chunk_evidence_is_labelled_as_a_passage_never_as_a_chunk():
    """A chunk index is an ingestion detail, not a place a reader can turn to."""
    record = CitationRegistry.from_bundle(
        _bundle(_item("passage text", chunk_id="1-7", chunk_index=7))
    ).records[0]

    assert record.locator.kind == "passage"
    assert record.locator.label == PASSAGE_LOCATION_LABEL == "Relevant passage"
    assert "chunk" not in record.locator.label.lower()
    assert record.locator.page_start is None and record.locator.page_end is None
    payload = record.to_dict()
    assert "chunk" not in payload["display_label"].lower()
    assert "chunk" not in payload["locator"]["label"].lower()
    # The index itself may stay on the record for internal use; it is the
    # *label* that must never expose it.
    assert payload["chunk_index"] == 7


def test_no_public_locator_label_ever_contains_a_chunk_number():
    bundle = _bundle(
        _item("a", chunk_id="1-0", chunk_index=0),
        _item("b", chunk_id="1-1", chunk_index=1, page_start=3),
        _item("c", source_type="selection", chunk_id=None, chunk_index=None),
    )
    for record in CitationRegistry.from_bundle(bundle).records:
        assert not re.search(r"chunk\s*\d", record.display_label, re.IGNORECASE)


@pytest.mark.parametrize(
    "rule",
    [
        "Answer the question directly",
        "Do not repeat or restate the question",
        "Keep the answer concise",
        "Do not offer further help",
        "Paraphrase the evidence in your own words",
        "Do not present your own wording as a direct quotation",
        "Attach a citation id only to the claim",
        "Do not describe or explain the citation markers themselves",
        "Never invent a citation id",
    ],
)
def test_generation_prompt_states_the_required_answer_rule(rule):
    registry = CitationRegistry.from_bundle(_bundle(_item("passage")))
    prompt = build_generation_prompt("What is the term?", registry)
    assert rule in prompt, f"the prompt must instruct the model: {rule}"


def test_absence_language_matches_the_evidence_scope():
    """Only a complete document can be said to be silent on a subject."""
    complete = CitationRegistry.from_bundle(
        _bundle(
            _item("whole doc", source_type="complete_document", chunk_id=None, chunk_index=None),
            mode=EvidenceMode.COMPLETE_DOCUMENT,
        )
    )
    retrieved = CitationRegistry.from_bundle(_bundle(_item("a passage")))

    complete_prompt = build_generation_prompt("q", complete)
    retrieved_prompt = build_generation_prompt("q", retrieved)

    assert "the document does not address it" in complete_prompt
    assert "not found in the provided passages" not in complete_prompt

    assert "not found in the provided passages" in retrieved_prompt
    assert "Do not claim the document as a whole is silent" in retrieved_prompt
    assert "the document does not address it" not in retrieved_prompt


def test_absence_rule_helper_is_scope_addressable():
    assert absence_rule(EvidenceMode.COMPLETE_DOCUMENT) == ABSENCE_RULE_COMPLETE_DOCUMENT
    for mode in (EvidenceMode.HYBRID_RETRIEVAL, EvidenceMode.SELECTED_TEXT, EvidenceMode.NONE):
        assert absence_rule(mode) == ABSENCE_RULE_PARTIAL_EVIDENCE


def test_the_trusted_quotation_surface_is_the_registry_excerpt_not_model_prose():
    """The displayed quotation comes from the server's bounded excerpt.

    The prompt forbids the model from passing its own wording off as a
    quotation, and the record carries the quotable text itself, so the UI never
    has to trust generated prose for what a document says.
    """
    registry = CitationRegistry.from_bundle(_bundle(_item("The term is three years.")))
    prompt = build_generation_prompt("q", registry)

    assert "Do not present your own wording as a direct quotation" in prompt
    assert "Paraphrase the evidence in your own words" in prompt
    record = registry.records[0].to_dict()
    assert record["excerpt"] == "The term is three years.", (
        "the quotable text is server-supplied, bounded, and independent of the answer"
    )
    assert record["excerpt_truncated"] is False


def test_question_echoes_are_not_claims():
    claims = extract_claims(
        "What is the term of the agreement? The term is three years [C1]."
    )

    assert claims == ["The term is three years [C1]."]
    assert not any(claim.strip().endswith("?") for claim in claims)


def test_citation_mechanics_commentary_is_not_a_claim():
    claims = extract_claims(
        "The term is three years [C1]. [C1] is cited as evidence for this information."
    )

    assert claims == ["The term is three years [C1]."]


@pytest.mark.parametrize(
    "sentence",
    [
        "[C1] is cited as evidence for this information.",
        "[C2] is cited as evidence.",
        "This is provided as the source for this claim.",
        "[C1] is referenced as support for the above.",
    ],
)
def test_citation_mechanics_variants_are_excluded(sentence):
    assert extract_claims(sentence) == []


@pytest.mark.parametrize(
    "sentence",
    [
        "The contract is cited in Exhibit A [C1].",
        "The statute cited by the parties governs renewal [C1].",
        "Evidence of delivery was provided to the buyer [C1].",
    ],
)
def test_ordinary_factual_sentences_with_markers_are_kept(sentence):
    assert extract_claims(sentence) == [sentence], (
        "a factual sentence must not be discarded merely for carrying a marker"
    )


def test_citation_mechanics_commentary_cannot_consume_a_verification_slot():
    """With a budget of one, the real claim must be the one that gets checked."""
    # Evidence that genuinely states the claim: the deterministic support
    # floor checks the claim's numbers against the cited evidence, so a
    # placeholder string would (correctly) refuse to certify "three years".
    registry = CitationRegistry.from_bundle(_bundle(_item("The term is three years.")))
    answer = "[C1] is cited as evidence for this information. The term is three years [C1]."
    checked = []

    def verifier(claim, records):
        checked.append(claim)
        return "supported"

    result = generate_citations(
        answer, registry, verifier=verifier, limits=CitationLimits(max_claims_verified=1)
    )

    assert checked == ["The term is three years [C1]."]
    assert [claim.text for claim in result.claims] == ["The term is three years [C1]."]
    assert result.claims[0].status is ClaimVerificationStatus.SUPPORTED


def test_used_citation_ids_are_valid_deduplicated_and_in_first_use_order():
    registry = CitationRegistry.from_bundle(
        _bundle(
            _item("first", chunk_id="1-0", chunk_index=0),
            _item("second", chunk_id="1-1", chunk_index=1),
            _item("third", chunk_id="1-2", chunk_index=2),
        )
    )

    result = generate_citations(
        "Third point [C3]. First point [C1]. Third again [C3]. Unknown [C9].",
        registry,
        verifier=lambda c, r: "supported",
    )

    assert result.used_citation_ids == ("C3", "C1"), (
        "used ids follow first use in the answer, not registry position"
    )
    assert len(set(result.used_citation_ids)) == len(result.used_citation_ids)
    assert all(registry.is_valid(cid) for cid in result.used_citation_ids)
    assert "C2" not in result.used_citation_ids, "an unused candidate is not a source"
    assert result.invalid_citation_ids == ("C9",)
    assert registry.get("C9") is None


@pytest.mark.parametrize(
    ("answer", "verifier", "limits", "expected_status", "expected_reason"),
    [
        ("A plain claim.", lambda c, r: "supported", None,
         ClaimVerificationStatus.UNSUPPORTED, ClaimReason.NO_CITATION),
        ("A claim [C9].", lambda c, r: "supported", None,
         ClaimVerificationStatus.UNSUPPORTED, ClaimReason.INVALID_CITATION),
        ("A claim [C1].", None, None,
         ClaimVerificationStatus.UNVERIFIED, ClaimReason.NOT_VERIFIED),
        ("A claim [C1].", "boom", None,
         ClaimVerificationStatus.UNVERIFIED, ClaimReason.VERIFIER_UNAVAILABLE),
    ],
    ids=["no_citation", "invalid_citation", "limit_or_not_run", "verifier_error"],
)
def test_claim_reasons_stay_distinguishable(answer, verifier, limits, expected_status, expected_reason):
    registry = CitationRegistry.from_bundle(_bundle(_item("evidence")))

    def exploding(claim, records):
        raise RuntimeError("verifier down")

    resolved = exploding if verifier == "boom" else verifier
    result = generate_citations(
        answer, registry, verifier=resolved, limits=limits or CitationLimits()
    )

    assert result.claims[0].status is expected_status
    assert result.claims[0].reason is expected_reason
    payload = result.claims[0].to_dict()
    assert payload["status"] == expected_status.value
    assert payload["reason"] == expected_reason.value


def test_a_claim_skipped_by_the_budget_is_unverified_not_unsupported():
    registry = CitationRegistry.from_bundle(_bundle(_item("evidence")))

    result = generate_citations(
        "First [C1]. Second [C1]. Third [C1].",
        registry,
        verifier=lambda c, r: "supported",
        limits=CitationLimits(max_claims_verified=2),
    )

    statuses = [claim.status for claim in result.claims]
    reasons = [claim.reason for claim in result.claims]
    assert statuses[:2] == [ClaimVerificationStatus.SUPPORTED] * 2
    assert statuses[2] is ClaimVerificationStatus.UNVERIFIED
    assert reasons[2] is ClaimReason.NOT_VERIFIED, (
        "a claim the budget skipped must be reported as not checked, never as "
        "unsupported"
    )


# --- Defect A: a stray marker belongs to the claim it was written for -------
#
# The model sometimes writes the citation after the full stop. The splitter
# then hands the marker over as its own fragment; discarding it stripped the
# citation off a real, correctly cited claim and reported that claim as
# uncited and unsupported.


def test_a_marker_written_after_the_full_stop_stays_with_its_claim():
    claims = extract_claims("No monetary penalty is stated. [C1]")

    assert len(claims) == 1, "the marker must not become a claim of its own"
    assert "No monetary penalty is stated" in claims[0]
    assert "[C1]" in claims[0], (
        "the citation belongs to the sentence it followed, not to nothing"
    )


def test_the_repaired_claim_is_cited_rather_than_reported_as_uncited():
    """The live Meridian defect, end to end."""
    registry = CitationRegistry.from_bundle(
        _bundle(
            _item(
                "whole doc",
                source_type="complete_document",
                chunk_id=None,
                chunk_index=None,
            ),
            mode=EvidenceMode.COMPLETE_DOCUMENT,
        )
    )

    result = generate_citations(
        "The document does not mention a monetary penalty. [C1]",
        registry,
        verifier=lambda c, r: "supported",
    )

    assert len(result.claims) == 1
    assert result.claims[0].citation_ids == ("C1",)
    assert result.claims[0].reason is not ClaimReason.NO_CITATION
    assert result.claims[0].status is ClaimVerificationStatus.SUPPORTED


def test_several_stray_markers_all_attach_to_the_preceding_claim():
    claims = extract_claims("The fee is fixed. [C1] [C2]")

    assert len(claims) == 1
    assert "[C1]" in claims[0] and "[C2]" in claims[0]


def test_consecutive_marker_only_fragments_all_fold_into_one_claim():
    claims = extract_claims("The fee is fixed. [C1]\n[C2]\n[C3]")

    assert len(claims) == 1
    for token in ("[C1]", "[C2]", "[C3]"):
        assert token in claims[0]


def test_a_stray_invalid_marker_attaches_and_stays_available_for_validation():
    """Re-attaching must not hide a fabricated id from invalid-id handling."""
    registry = CitationRegistry.from_bundle(_bundle(_item("evidence")))
    answer = "The penalty is unstated. [C99]"

    claims = extract_claims(answer)
    assert len(claims) == 1
    assert "[C99]" in claims[0]
    assert parse_citation_markers(claims[0], registry).invalid_ids == ("C99",)

    result = generate_citations(answer, registry, verifier=lambda c, r: "supported")
    assert result.claims[0].invalid_citation_ids == ("C99",)
    assert result.claims[0].status is ClaimVerificationStatus.UNSUPPORTED
    assert result.claims[0].reason is ClaimReason.INVALID_CITATION
    assert result.invalid_citation_ids == ("C99",)


def test_a_leading_marker_only_fragment_is_discarded():
    claims = extract_claims("[C1]\nThe term is three years [C2].")

    assert claims == ["The term is three years [C2]."], (
        "a marker with no claim before it has nothing to attach to"
    )


def test_a_marker_after_a_skipped_question_neither_cites_nor_restores_it():
    claims = extract_claims("What is the monetary penalty? [C1]")

    assert claims == [], (
        "a skipped question is not a retained claim, so the marker is dropped "
        "rather than turning the question into a cited assertion"
    )


def test_a_question_marker_pair_does_not_block_the_real_claim():
    claims = extract_claims("Is a penalty stated? [C1]\nNo penalty is stated [C2].")

    assert claims == ["No penalty is stated [C2]."]


def test_a_marker_only_fragment_does_not_resurrect_suppressed_commentary():
    """The suppressed sentence still stands between the marker and the claim.

    Dropping the commentary must not hand its marker to whatever claim came
    before it: the term claim cited C1 and nothing else.
    """
    claims = extract_claims(
        "The term is three years [C1]. "
        "[C2] is cited as evidence for this information. [C2]"
    )

    assert claims == ["The term is three years [C1]."], (
        "the suppressed commentary closed the marker target, so the trailing "
        "[C2] belongs to no claim and is discarded"
    )
    assert "[C2]" not in claims[0]


def test_a_marker_after_a_skipped_question_does_not_attach_to_an_earlier_claim():
    """A skipped question closes the marker target just as commentary does."""
    claims = extract_claims("The term is three years [C1]. Is a penalty stated? [C2]")

    assert claims == ["The term is three years [C1]."]
    assert "[C2]" not in claims[0], (
        "the question intervened, so the marker is not the term claim's"
    )


@pytest.mark.parametrize(
    "sentence",
    [
        "The Supreme Court decision cited by the seller controls this dispute [C1].",
        "Evidence of insurance must be provided within ten days [C1].",
        "The indemnity is supported by the escrow provision [C1].",
        "Notice was provided to the tenant on 1 March [C1].",
    ],
)
def test_legal_sentences_about_evidence_and_support_remain_claims(sentence):
    assert extract_claims(sentence) == [sentence], (
        "the repair must not widen into a regex that deletes legal sentences "
        "for using the words cited, evidence, support or provided"
    )


def test_the_marker_repair_does_not_change_the_claim_budget():
    registry = CitationRegistry.from_bundle(_bundle(_item("evidence")))

    result = generate_citations(
        "First point. [C1]\nSecond point. [C1]\nThird point. [C1]",
        registry,
        verifier=lambda c, r: "supported",
        limits=CitationLimits(max_claims_verified=2),
    )

    assert len(result.claims) == 3
    statuses = [claim.status for claim in result.claims]
    assert statuses[:2] == [ClaimVerificationStatus.SUPPORTED] * 2
    assert statuses[2] is ClaimVerificationStatus.UNVERIFIED
    assert result.claims[2].reason is ClaimReason.NOT_VERIFIED


# --- Defect C: exact evidence-block citation discipline, in the prompt ------


@pytest.mark.parametrize(
    "rule",
    [
        "Place the citation id immediately before the sentence's terminal punctuation",
        "A citation is valid only when that specific evidence block contains the factual material",
        "A matching document name, heading, or topic is not enough",
        "Do not cite an evidence block merely because it belongs to the right document",
        "When a claim combines facts drawn from more than one evidence block",
        "cite every evidence block that claim needs",
    ],
)
def test_generation_prompt_states_the_exact_block_citation_rules(rule):
    registry = CitationRegistry.from_bundle(_bundle(_item("passage")))
    prompt = build_generation_prompt("What is the term?", registry)
    assert rule in prompt, f"the prompt must instruct the model: {rule}"


def test_a_complete_document_negative_finding_must_carry_its_citation():
    registry = CitationRegistry.from_bundle(
        _bundle(
            _item(
                "whole doc",
                source_type="complete_document",
                chunk_id=None,
                chunk_index=None,
            ),
            mode=EvidenceMode.COMPLETE_DOCUMENT,
        )
    )
    prompt = build_generation_prompt("Is a penalty stated?", registry)

    assert "the document does not address it" in prompt
    assert "cite the complete-document evidence block it rests on" in prompt
    assert "immediately before the terminal punctuation" in prompt


def test_a_hybrid_negative_finding_is_scoped_to_the_retrieved_passages():
    registry = CitationRegistry.from_bundle(_bundle(_item("a passage")))
    prompt = build_generation_prompt("Is a penalty stated?", registry)

    assert "not found in the retrieved passages" in prompt
    assert "Do not state or imply that the entire document is silent" in prompt
    assert "the document does not address it" not in prompt, (
        "only a complete document can be said to be silent as a whole"
    )


# --- answer modes: prompts for ungraded output ------------------------------


_UNGRADED_MODES = (AnswerMode.STRUCTURED_JSON, AnswerMode.MODEL_REASONING)


@pytest.mark.parametrize(
    ("mode", "expected", "forbidden"),
    [
        (EvidenceMode.COMPLETE_DOCUMENT, "complete text of the document", None),
        (EvidenceMode.HYBRID_RETRIEVAL, "not their complete text", "complete text of the document"),
        (EvidenceMode.SELECTED_TEXT, "text the user selected", "complete text of the document"),
    ],
)
def test_the_scope_statement_never_overstates_coverage(mode, expected, forbidden):
    statement = evidence_scope_statement(mode)
    assert expected in statement
    if forbidden:
        assert forbidden not in statement


@pytest.mark.parametrize("mode", _UNGRADED_MODES)
def test_ungraded_prompts_carry_the_evidence_and_its_true_scope(mode):
    registry = CitationRegistry.from_bundle(_bundle(_item("The term is three years.")))
    prompt = build_mode_prompt(mode, "Summarize the term.", registry, extra_context="<kb>note</kb>")

    assert "The term is three years." in prompt, "the same evidence the planner chose"
    assert "not their complete text" in prompt, "retrieved passages are never called complete"
    assert "<kb>note</kb>" in prompt
    assert CITATION_RULES not in prompt


@pytest.mark.parametrize(
    ("mode", "instruction"),
    [
        (AnswerMode.STRUCTURED_JSON, STRUCTURED_NO_EVIDENCE_INSTRUCTION),
        (AnswerMode.MODEL_REASONING, MODEL_REASONING_NO_EVIDENCE_INSTRUCTION),
    ],
)
def test_ungraded_prompts_without_evidence_forbid_inventing_document_content(mode, instruction):
    prompt = build_mode_prompt(mode, "Summarize the term.", CitationRegistry())
    assert instruction in prompt
    assert "<evidence>" not in prompt


def test_document_qa_prompts_are_not_built_by_the_mode_builder():
    registry = CitationRegistry.from_bundle(_bundle(_item("text")))
    with pytest.raises(ValueError):
        build_mode_prompt(AnswerMode.DOCUMENT_QA, "q", registry)


def test_citations_without_claims_reports_usage_but_grades_nothing():
    registry = CitationRegistry.from_bundle(_bundle(_item("first"), _item("second", chunk_id="1-1", chunk_index=1)))
    result = citations_without_claims("Uses the second [C2] and a stray [C9].", registry)

    assert result.claims == ()
    assert result.used_citation_ids == ("C2",)
    assert result.invalid_citation_ids == ("C9",)
