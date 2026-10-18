"""Table-row completion between retrieval ranking and evidence budgeting.

The saved candidate run (evidence bundle live_extraction_candidate_20261018T011649Z) indexed D12's
page-4 statement of financial position with one serialized line per row, and the unchanged
500-character chunker cut the "Other reserves" row between chunks 26 and 27: chunk 26 ends
"... RM'000: (23,326) | Audited As" and chunk 27 starts "at 31.05.2025 RM'000: (6,373)". An answer
built from chunk 26 alone cannot state the 2025 amount, and chunk 27 alone gives (6,373) without
its row label.

The D12 constants below are copied verbatim from that bundle:
* chunks 26-28 of document 8 from dgx_evidence/candidate_tables_run/candidate_chunks.csv
  (document 10's chunks are identical);
* an excerpt of document 8's stored content from acceptance_tables/T6_D12_enhanced_tables_content.txt
  (sha256 3c50b4ec9fddd0e9056d6476f0eff1b2e498f6a2221c3165d14fc58ab202f5c9), where the excerpt
  occurs exactly once.

No database, model, network or filesystem access: the loader and retrieval are in-memory fakes. The
last section imports ``main`` and replaces only its connection factory with a recording fake, to show
the completion runs through main's own owner-scoped loader.
"""

import contextlib
import dataclasses
import importlib
import os
import sys

import dotenv
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from answer_evidence import (
    AnswerEvidencePlanner,
    AuthorizedDocument,
    AuthorizedScope,
    EvidenceItem,
    EvidenceLimits,
    EvidenceMode,
    EvidenceRequest,
    EvidenceWarning,
    _DocumentTexts,
)
from citation_generation import CitationLimits, CitationRegistry, build_generation_prompt
from table_row_completion import (
    LEADING,
    TRAILING,
    CanonicalText,
    CompletionStatus,
    RefusalReason,
    RowCompletionLimits,
    complete_fragment,
    may_hold_table_text,
)


# --- saved D12 evidence (verbatim) --------------------------------------------

D12_FILENAME = "D12_Infomina_Q4_FY2026_Interim_Report.pdf"

D12_STORED_EXCERPT = "\n".join([
    "[Table 1, page 4] TOTAL ASSETS | Unaudited As at 31.05.2026 RM'000: 389,861 | Audited As at 31.05.2025 RM'000: 281,916",
    "[Table 1, page 4] Section: EQUITY AND LIABILITIES",
    "[Table 1, page 4] Section: Equity attributable to owners of the Company",
    "[Table 1, page 4] Share capital | Unaudited As at 31.05.2026 RM'000: 42,048 | Audited As at 31.05.2025 RM'000: 42,048",
    "[Table 1, page 4] Irredeemable preference shares | Unaudited As at 31.05.2026 RM'000: 202 | Audited As at 31.05.2025 RM'000: 65",
    "[Table 1, page 4] Retained earnings | Unaudited As at 31.05.2026 RM'000: 148,775 | Audited As at 31.05.2025 RM'000: 121,753",
    "[Table 1, page 4] Reorganisation reserve | Unaudited As at 31.05.2026 RM'000: (5,700) | Audited As at 31.05.2025 RM'000: (5,700)",
    "[Table 1, page 4] Other reserves | Unaudited As at 31.05.2026 RM'000: (23,326) | Audited As at 31.05.2025 RM'000: (6,373)",
    "[Table 1, page 4] | Unaudited As at 31.05.2026 RM'000: 161,999 | Audited As at 31.05.2025 RM'000: 151,793",
    "[Table 1, page 4] Non-controlling interests | Unaudited As at 31.05.2026 RM'000: 174 | Audited As at 31.05.2025 RM'000: (9)",
    "[Table 1, page 4] TOTAL EQUITY | Unaudited As at 31.05.2026 RM'000: 162,173 | Audited As at 31.05.2025 RM'000: 151,784",
    "[Table 1, page 4] Section: Non-current liabilities",
    "[Table 1, page 4] Borrowings | Unaudited As at 31.05.2026 RM'000: 2,675 | Audited As at 31.05.2025 RM'000: 2,083",
    "[Table 1, page 4] Deferred tax liabilities | Unaudited As at 31.05.2026 RM'000: 4,130 | Audited As at 31.05.2025 RM'000: 4,612",
    "[Table 1, page 4] Total non-current liabilities | Unaudited As at 31.05.2026 RM'000: 6,805 | Audited As at 31.05.2025 RM'000: 6,695",
    "",
    "INFOMINA BERHAD",
    "Registration No.: 200701018579 (776590-U) (Incorporated in Malaysia)",
    "INTERIM FINANCIAL REPORT FOR THE FOURTH QUARTER ENDED 31 MAY 2026",
    "UNAUDITED CONDENSED CONSOLIDATED STATEMENTS OF FINANCIAL POSITION AS AT 31 MAY 2026 (CONTINUED) (1)",
])

D12_CHUNK_26 = (
    "31.05.2025 RM'000: 42,048 [Table 1, page 4] Irredeemable preference shares | Unaudited As at "
    "31.05.2026 RM'000: 202 | Audited As at 31.05.2025 RM'000: 65 [Table 1, page 4] Retained earnings | "
    "Unaudited As at 31.05.2026 RM'000: 148,775 | Audited As at 31.05.2025 RM'000: 121,753 [Table 1, page 4] "
    "Reorganisation reserve | Unaudited As at 31.05.2026 RM'000: (5,700) | Audited As at 31.05.2025 RM'000: "
    "(5,700) [Table 1, page 4] Other reserves | Unaudited As at 31.05.2026 RM'000: (23,326) | Audited As"
)
D12_CHUNK_27 = (
    "at 31.05.2025 RM'000: (6,373) [Table 1, page 4] | Unaudited As at 31.05.2026 RM'000: 161,999 | "
    "Audited As at 31.05.2025 RM'000: 151,793 [Table 1, page 4] Non-controlling interests | Unaudited As at "
    "31.05.2026 RM'000: 174 | Audited As at 31.05.2025 RM'000: (9) [Table 1, page 4] TOTAL EQUITY | "
    "Unaudited As at 31.05.2026 RM'000: 162,173 | Audited As at 31.05.2025 RM'000: 151,784 [Table 1, page 4] "
    "Section: Non-current liabilities [Table 1, page 4] Borrowings | Unaudited As at 31.05.2026 RM'000:"
)
D12_CHUNK_28 = (
    "2,675 | Audited As at 31.05.2025 RM'000: 2,083 [Table 1, page 4] Deferred tax liabilities | Unaudited "
    "As at 31.05.2026 RM'000: 4,130 | Audited As at 31.05.2025 RM'000: 4,612 [Table 1, page 4] Total "
    "non-current liabilities | Unaudited As at 31.05.2026 RM'000: 6,805 | Audited As at 31.05.2025 RM'000: "
    "6,695 INFOMINA BERHAD Registration No.: 200701018579 (776590-U) (Incorporated in Malaysia) INTERIM "
    "FINANCIAL REPORT FOR THE FOURTH QUARTER ENDED 31 MAY 2026 UNAUDITED CONDENSED CONSOLIDATED STATEMENTS"
)

OTHER_RESERVES_ROW = (
    "[Table 1, page 4] Other reserves | Unaudited As at 31.05.2026 RM'000: (23,326) | "
    "Audited As at 31.05.2025 RM'000: (6,373)"
)
QUESTION = "What were other reserves as at 31.05.2026 and as at 31.05.2025? Include the units."
# The route's own citation limits (main._CITATION_LIMITS).
ROUTE_CITATION_LIMITS = CitationLimits(max_claims_verified=2)
# The saved document (61,201 characters) is far above the 12,000-character complete-document limit, so
# questions about it are answered by hybrid retrieval. The excerpt is shorter, so these tests lower that
# one limit to stay on the same path; every other limit keeps its default unless a test says otherwise.
HYBRID_PATH_COMPLETE_DOCUMENT_CHARS = 1_000


# --- helpers -----------------------------------------------------------------


def _limits(**overrides):
    return EvidenceLimits(complete_document_chars=HYBRID_PATH_COMPLETE_DOCUMENT_CHARS, **overrides)


def _scope(*documents, user_id="owner-a"):
    return AuthorizedScope(documents=tuple(documents), established=True, user_id=user_id, anonymous=False)


def _doc(document_id, filename=D12_FILENAME):
    return AuthorizedDocument(document_id=document_id, filename=filename)


def _chunk(document_id, chunk_index, text, page_start=4, page_end=4, filename=D12_FILENAME):
    pages = list(range(page_start, page_end + 1)) if page_start is not None else []
    return EvidenceItem(
        text=text,
        source_type="document_chunk",
        document_id=document_id,
        filename=filename,
        chunk_id=f"{document_id}-{chunk_index}",
        chunk_index=chunk_index,
        page_start=page_start,
        page_end=page_end,
        locator_json={"locator_type": "pdf_pages", "page_numbers": pages},
    )


def _d12(document_id, chunk_index):
    text, pages = {26: (D12_CHUNK_26, (4, 4)), 27: (D12_CHUNK_27, (4, 4)), 28: (D12_CHUNK_28, (4, 5))}[chunk_index]
    return _chunk(document_id, chunk_index, text, *pages)


class _Recorder:
    def __init__(self, texts=None, chunks=None, fail=False):
        self.texts = texts or {}
        self.chunks = chunks or []
        self.fail = fail
        self.loaded = []

    def load(self, document_id):
        self.loaded.append(document_id)
        if self.fail:
            raise RuntimeError("stored text unavailable")
        return self.texts.get(document_id)

    def retrieve(self, question, document_ids, top_k):
        return list(self.chunks)


def _plan(recorder, documents, requested, limits=None, row_limits=dataclasses.MISSING, scope=None):
    kwargs = {} if row_limits is dataclasses.MISSING else {"row_completion_limits": row_limits}
    planner = AnswerEvidencePlanner(
        load_document_text=recorder.load,
        retrieve=recorder.retrieve,
        limits=limits or _limits(),
        **kwargs,
    )
    request = EvidenceRequest(question=QUESTION, requested_document_ids=tuple(requested))
    return planner.plan(request, scope or _scope(*documents))


def _prompt(result):
    registry = CitationRegistry.from_bundle(result.bundle, ROUTE_CITATION_LIMITS)
    return registry, build_generation_prompt(QUESTION, registry)


def _sides(item):
    return [(line.side, line.table, line.page, len(line.text)) for line in item.row_completion]


# --- 1. the saved D12 boundary -------------------------------------------------


def test_saved_d12_chunks_are_adjacent_windows_of_the_saved_stored_text():
    assert (len(D12_CHUNK_26), len(D12_CHUNK_27), len(D12_CHUNK_28)) == (498, 495, 500), (
        "the chunk constants must be the saved chunk texts, character for character"
    )
    words = " ".join(D12_STORED_EXCERPT.split())
    assert f"{D12_CHUNK_26} {D12_CHUNK_27} {D12_CHUNK_28}" in words
    assert D12_STORED_EXCERPT.count(OTHER_RESERVES_ROW) == 1
    assert "(6,373)" not in D12_CHUNK_26 and "Other reserves" not in D12_CHUNK_27


def test_without_completion_the_saved_chunk_26_cannot_state_the_2025_amount():
    recorder = _Recorder(texts={8: D12_STORED_EXCERPT}, chunks=[_d12(8, 26)])
    result = _plan(recorder, [_doc(8)], [8], row_limits=None)

    _, prompt = _prompt(result)
    assert "(23,326)" in prompt
    assert "(6,373)" not in prompt, "the confirmed defect: the 2025 amount is in the next chunk"


def test_saved_chunk_cut_at_its_end_reaches_the_prompt_with_both_amounts():
    recorder = _Recorder(texts={8: D12_STORED_EXCERPT}, chunks=[_d12(8, 26)])
    result = _plan(recorder, [_doc(8)], [8])

    assert result.mode is EvidenceMode.HYBRID_RETRIEVAL
    item = result.bundle.items[0]
    assert item.text.startswith(
        "[Table 1, page 4] Share capital | Unaudited As at 31.05.2026 RM'000: 42,048 | Audited As at "
        + D12_CHUNK_26[:40]
    )
    assert item.text.endswith("(23,326) | Audited As at 31.05.2025 RM'000: (6,373)")
    assert _sides(item) == [(LEADING, 1, 4, 91), (TRAILING, 1, 4, 29)]

    registry, prompt = _prompt(result)
    # Each amount stays attached to its own year, audit status and unit, in one row.
    assert OTHER_RESERVES_ROW in prompt
    assert "Unaudited As at 31.05.2026 RM'000: (23,326)" in prompt
    assert "Audited As at 31.05.2025 RM'000: (6,373)" in prompt
    assert [record.locator.label for record in registry.records] == ["page 4"]
    assert registry.records[0].evidence_text == item.text, "the verifier reads what the model read"


def test_saved_chunk_cut_at_its_start_regains_its_row_label():
    recorder = _Recorder(texts={8: D12_STORED_EXCERPT}, chunks=[_d12(8, 27)])
    result = _plan(recorder, [_doc(8)], [8])

    item = result.bundle.items[0]
    assert item.text.startswith(OTHER_RESERVES_ROW + " [Table 1, page 4] | Unaudited")
    assert item.text.endswith(
        "Borrowings | Unaudited As at 31.05.2026 RM'000: 2,675 | Audited As at 31.05.2025 RM'000: 2,083"
    )
    assert _sides(item) == [(LEADING, 1, 4, 91), (TRAILING, 1, 4, 46)]
    registry, prompt = _prompt(result)
    assert OTHER_RESERVES_ROW in prompt
    assert registry.records[0].locator.label == "page 4"


def test_saved_chunks_retrieved_together_are_each_self_contained():
    recorder = _Recorder(texts={8: D12_STORED_EXCERPT}, chunks=[_d12(8, 26), _d12(8, 27)])
    result = _plan(recorder, [_doc(8)], [8])

    registry, prompt = _prompt(result)
    assert [record.chunk_index for record in registry.records] == [26, 27], "rank order is kept"
    for record in registry.records:
        assert OTHER_RESERVES_ROW in record.evidence_text
        assert record.locator.label == "page 4"
    assert recorder.loaded == [8], "the stored text is fetched once per document per answer"


def test_saved_chunk_spanning_two_pages_completes_its_row_and_not_the_prose():
    recorder = _Recorder(texts={8: D12_STORED_EXCERPT}, chunks=[_d12(8, 28)])
    result = _plan(recorder, [_doc(8)], [8])

    item = result.bundle.items[0]
    assert item.text == (
        "[Table 1, page 4] Borrowings | Unaudited As at 31.05.2026 RM'000: " + D12_CHUNK_28
    )
    assert _sides(item) == [(LEADING, 1, 4, 65)]
    assert (item.page_start, item.page_end) == (4, 5)
    registry, _ = _prompt(result)
    assert registry.records[0].locator.label == "pages 4-5"


ROW_TAIL_THEN_PROSE = "6,695 INFOMINA BERHAD Registration No.: 200701018579 (776590-U)"
TOTAL_NON_CURRENT_HEAD = (
    "[Table 1, page 4] Total non-current liabilities | Unaudited As at 31.05.2026 RM'000: 6,805 | "
    "Audited As at 31.05.2025 RM'000:"
)


def test_a_chunk_opening_with_a_bare_row_value_is_completed_when_its_text_is_already_held():
    # No tag and no field separator: nothing in the chunk itself shows a table. With one document in scope
    # the plan already holds its stored text, so the chunk is still checked and regains its row.
    chunk = _chunk(8, 90, ROW_TAIL_THEN_PROSE, page_start=4, page_end=5)
    assert not may_hold_table_text(chunk.text)
    recorder = _Recorder(texts={8: D12_STORED_EXCERPT}, chunks=[chunk])

    item = _plan(recorder, [_doc(8)], [8]).bundle.items[0]

    assert item.text == f"{TOTAL_NON_CURRENT_HEAD} {ROW_TAIL_THEN_PROSE}"
    assert _sides(item) == [(LEADING, 1, 4, len(TOTAL_NON_CURRENT_HEAD))]


def test_in_a_multi_document_scope_a_bare_row_value_needs_a_table_chunk_from_its_document():
    bare = _chunk(8, 90, ROW_TAIL_THEN_PROSE, page_start=4, page_end=5)
    other = _chunk(1, 0, "An ordinary passage.", filename="agreement.pdf")
    scope = ([_doc(8), _doc(1, "agreement.pdf")], [8, 1])

    alone = _Recorder(texts={8: D12_STORED_EXCERPT}, chunks=[other, bare])
    assert list(_plan(alone, *scope).bundle.items) == [other, bare]
    assert alone.loaded == [], "ordinary-looking chunks alone never cause a read"

    with_table = _Recorder(texts={8: D12_STORED_EXCERPT}, chunks=[other, bare, _d12(8, 27)])
    items = _plan(with_table, *scope).bundle.items
    assert items[1].text == f"{TOTAL_NON_CURRENT_HEAD} {ROW_TAIL_THEN_PROSE}"
    assert with_table.loaded == [8]


# --- 2. the pure matcher ---------------------------------------------------------


def _row(page, label, *fields, table=1):
    return f"[Table {table}, page {page}] {label} | " + " | ".join(fields)


LONG_ROW = _row(6, "At 1 June 2025", *[f"Column {n} heading path / RM'000: {n},000" for n in range(1, 13)])
SYNTHETIC_STORED = "\n".join([
    "Statement of changes in equity",
    _row(6, "Opening", "Total / RM'000: 10"),
    LONG_ROW,
    _row(6, "Closing", "Total / RM'000: 12"),
    "",
    "Notes follow in prose on the next page.",
])


def test_a_chunk_inside_one_long_row_is_completed_on_both_sides():
    source = CanonicalText.with_table_lines(SYNTHETIC_STORED)
    words = LONG_ROW.split()
    fragment = " ".join(words[8:30])

    result = complete_fragment(source, fragment, 6, 6)

    assert result.status is CompletionStatus.COMPLETED
    leading, trailing = result.lines
    assert (leading.side, trailing.side) == (LEADING, TRAILING)
    assert f"{leading.text} {fragment} {trailing.text}" == LONG_ROW


def test_a_chunk_on_exact_row_boundaries_and_prose_are_not_cut():
    source = CanonicalText.with_table_lines(SYNTHETIC_STORED)
    whole_rows = " ".join([_row(6, "Opening", "Total / RM'000: 10"), LONG_ROW])
    assert complete_fragment(source, whole_rows, 6, 6).status is CompletionStatus.NOT_CUT

    prose = complete_fragment(source, "follow in prose on the", 6, 7)
    assert prose.status is CompletionStatus.NOT_CUT and prose.lines == ()


def test_only_word_aligned_occurrences_count():
    source = CanonicalText.with_table_lines(_row(4, "Item", "Flat 31.05.2025 RM'000: 5"))
    assert source.locate("at 31.05.2025")[1] is CompletionStatus.SOURCE_MISSING
    assert source.locate("Flat 31.05.2025")[0] is not None


def test_ambiguous_and_missing_fragments_are_refused():
    repeated = "\n".join([_row(4, "Fees", "2026 RM'000: 1"), _row(5, "Fees", "2026 RM'000: 1")])
    source = CanonicalText.with_table_lines(repeated)

    assert complete_fragment(source, "Fees | 2026", 4, 5).status is CompletionStatus.AMBIGUOUS
    assert complete_fragment(source, "Fees | 2027", 4, 5).status is CompletionStatus.SOURCE_MISSING
    assert complete_fragment(source, "   ", 4, 5).status is CompletionStatus.SOURCE_MISSING


def test_a_completion_from_a_page_outside_the_chunk_is_refused():
    source = CanonicalText.with_table_lines(_row(5, "Fees", "2026 RM'000: 1", "2025 RM'000: 2"))

    outside = complete_fragment(source, "Fees | 2026", 4, 4)
    assert outside.status is CompletionStatus.REFUSED
    assert set(outside.refusals) == {(LEADING, RefusalReason.PAGE_OUTSIDE_CHUNK),
                                     (TRAILING, RefusalReason.PAGE_OUTSIDE_CHUNK)}
    no_pages = complete_fragment(source, "Fees | 2026", None, None)
    assert {reason for _, reason in no_pages.refusals} == {RefusalReason.PAGE_UNKNOWN}

    unknown = CanonicalText.with_table_lines("[Table 1, page ?] Fees | 2026 RM'000: 1 | 2025 RM'000: 2")
    assert {reason for _, reason in complete_fragment(unknown, "Fees | 2026", 4, 4).refusals} == {
        RefusalReason.PAGE_UNKNOWN
    }


def test_a_side_over_its_bound_is_refused_whole():
    source = CanonicalText.with_table_lines(SYNTHETIC_STORED)
    words = LONG_ROW.split()
    fragment = " ".join(words[20:40])
    assert len(" ".join(words[:20])) > 40 and len(" ".join(words[40:])) > 40

    result = complete_fragment(source, fragment, 6, 6, max_side_chars=40)

    assert result.status is CompletionStatus.REFUSED and result.lines == ()
    assert set(result.refusals) == {(LEADING, RefusalReason.TOO_LONG), (TRAILING, RefusalReason.TOO_LONG)}


def test_a_line_holding_several_rows_is_not_one_table_line():
    flattened = CanonicalText.with_table_lines(" ".join(D12_STORED_EXCERPT.split()))

    result = complete_fragment(flattened, D12_CHUNK_26, 4, 4)

    assert result.status is CompletionStatus.REFUSED and result.lines == ()
    assert {reason for _, reason in result.refusals} == {RefusalReason.NOT_ONE_TABLE_LINE}


def test_only_chunks_with_table_structure_are_looked_up():
    assert may_hold_table_text(D12_CHUNK_26)
    assert may_hold_table_text("Column 3 heading path / RM'000: 3,000 | Column 4 heading")
    assert not may_hold_table_text("The term of this agreement is three years.")
    assert not may_hold_table_text("")
    assert CanonicalText.with_table_lines("Plain prose only.\nNo table lines.") is None


# --- 3. unchanged behaviour ----------------------------------------------------------


def test_ordinary_text_is_unchanged_and_loads_nothing():
    chunks = [
        _chunk(1, 0, "The term of this agreement is three years.", filename="agreement.pdf"),
        _chunk(2, 0, "Section 12 governs notice.", filename="statute.pdf"),
    ]
    recorder = _Recorder(texts={1: "x", 2: "y"}, chunks=chunks)
    result = _plan(recorder, [_doc(1, "agreement.pdf"), _doc(2, "statute.pdf")], [1, 2])

    assert list(result.bundle.items) == chunks
    assert recorder.loaded == []


def test_already_complete_table_rows_are_unchanged():
    complete_rows = (
        "[Table 1, page 4] Other reserves | Unaudited As at 31.05.2026 RM'000: (23,326) | Audited As at "
        "31.05.2025 RM'000: (6,373) [Table 1, page 4] | Unaudited As at 31.05.2026 RM'000: 161,999 | "
        "Audited As at 31.05.2025 RM'000: 151,793"
    )
    chunk = _chunk(8, 40, complete_rows)
    recorder = _Recorder(texts={8: D12_STORED_EXCERPT}, chunks=[chunk])

    result = _plan(recorder, [_doc(8)], [8])

    assert list(result.bundle.items) == [chunk]


@pytest.mark.parametrize(
    "texts, fail",
    [
        ({}, False),  # no stored text
        ({8: D12_STORED_EXCERPT.replace("(23,326)", "(23,999)")}, False),  # stored text no longer matches
        ({8: D12_STORED_EXCERPT + "\n" + D12_STORED_EXCERPT}, False),  # ambiguous: the chunk occurs twice
        ({8: " ".join(D12_STORED_EXCERPT.split())}, False),  # rows flattened onto one line
        ({8: "Plain prose only.\nNo table lines."}, False),  # no table line at all
        ({8: D12_STORED_EXCERPT}, True),  # the loader fails
    ],
    ids=["missing", "stale", "ambiguous", "flattened-rows", "no-table-lines", "load-error"],
)
def test_unmatched_sources_leave_the_chunk_as_retrieved(texts, fail):
    chunk = _d12(8, 26)
    recorder = _Recorder(texts=texts, chunks=[chunk], fail=fail)

    # Two documents in scope, so no complete-document load runs before the completion's own.
    result = _plan(recorder, [_doc(8), _doc(1, "agreement.pdf")], [8, 1])

    assert result.mode is EvidenceMode.HYBRID_RETRIEVAL
    assert list(result.bundle.items) == [chunk]
    assert recorder.loaded == [8]


def test_completion_can_be_switched_off():
    chunk = _d12(8, 26)
    recorder = _Recorder(texts={8: D12_STORED_EXCERPT}, chunks=[chunk])

    result = _plan(recorder, [_doc(8), _doc(1, "agreement.pdf")], [8, 1], row_limits=None)

    assert list(result.bundle.items) == [chunk] and recorder.loaded == []


def test_complete_document_mode_is_untouched():
    short = "[Table 1, page 1] Fees | 2026 RM'000: 1 | 2025 RM'000: 2"
    recorder = _Recorder(texts={8: short}, chunks=[_d12(8, 26)])
    result = _plan(recorder, [_doc(8)], [8])
    assert result.mode is EvidenceMode.COMPLETE_DOCUMENT
    assert result.bundle.items[0].text == short and result.bundle.items[0].row_completion == ()


# --- 4. authorization and requested-document scope ------------------------------------


def test_each_chunk_is_completed_from_its_own_document_never_a_same_named_one():
    other_copy = D12_STORED_EXCERPT.replace("(6,373)", "(9,999)")
    recorder = _Recorder(texts={8: D12_STORED_EXCERPT, 10: other_copy},
                         chunks=[_d12(8, 26), _d12(10, 26)])

    result = _plan(recorder, [_doc(8), _doc(10)], [8, 10])

    from_8, from_10 = result.bundle.items
    assert (from_8.document_id, from_10.document_id) == (8, 10)
    assert from_8.filename == from_10.filename == D12_FILENAME
    assert from_8.text.endswith("RM'000: (6,373)") and "(9,999)" not in from_8.text
    assert from_10.text.endswith("RM'000: (9,999)") and "(6,373)" not in from_10.text
    assert recorder.loaded == [8, 10]


def test_an_authorized_but_unrequested_same_named_document_is_never_loaded():
    recorder = _Recorder(texts={8: D12_STORED_EXCERPT, 10: D12_STORED_EXCERPT},
                         chunks=[_d12(10, 26), _d12(8, 27)])

    result = _plan(recorder, [_doc(8), _doc(10)], [8])

    assert [item.document_id for item in result.bundle.items] == [8]
    assert EvidenceWarning.DOCUMENT_NOT_AUTHORIZED.value in result.warnings
    assert recorder.loaded == [8], "only the requested document's stored text is read"


def test_the_completion_step_itself_refuses_a_document_outside_the_scope():
    # Defence in depth: plan() already drops such chunks before this step runs, but the step must not
    # load a document it was not given in scope even if handed one of its chunks.
    recorder = _Recorder(texts={8: D12_STORED_EXCERPT, 99: D12_STORED_EXCERPT})
    planner = AnswerEvidencePlanner(load_document_text=recorder.load, retrieve=recorder.retrieve, limits=_limits())
    foreign = _d12(99, 26)

    completed = planner._complete_table_rows([foreign, _d12(8, 27)], [_doc(8)], _DocumentTexts(recorder.load))

    assert completed[0] == foreign
    assert recorder.loaded == [8]


def test_another_users_same_named_document_is_never_loaded():
    recorder = _Recorder(texts={8: D12_STORED_EXCERPT, 99: D12_STORED_EXCERPT},
                         chunks=[_d12(99, 26), _d12(8, 27)])

    result = _plan(recorder, [_doc(8)], [8, 99])

    assert [item.document_id for item in result.bundle.items] == [8]
    assert 99 not in recorder.loaded


# --- 5. evidence limits and page metadata ------------------------------------------------


def test_completion_only_uses_room_the_ranked_evidence_leaves():
    chunk = _d12(8, 26)
    # Room for the 29-character trailing completion (30 with its joining space), not the 91-character head.
    limits = _limits(total_evidence_chars=len(D12_CHUNK_26) + 30)
    recorder = _Recorder(texts={8: D12_STORED_EXCERPT}, chunks=[chunk])

    result = _plan(recorder, [_doc(8)], [8], limits=limits)

    item = result.bundle.items[0]
    assert _sides(item) == [(TRAILING, 1, 4, 29)]
    assert item.text == D12_CHUNK_26 + " at 31.05.2025 RM'000: (6,373)"
    assert result.bundle.total_chars == limits.total_evidence_chars
    assert result.bundle.truncated is False and item.truncated is False


def test_no_room_means_no_completion_and_an_identical_bundle():
    chunks = [_d12(8, 26), _d12(8, 27)]
    limits = _limits(total_evidence_chars=len(D12_CHUNK_26) + len(D12_CHUNK_27))
    with_completion = _plan(_Recorder(texts={8: D12_STORED_EXCERPT}, chunks=chunks), [_doc(8)], [8], limits=limits)
    without = _plan(_Recorder(texts={8: D12_STORED_EXCERPT}, chunks=chunks), [_doc(8)], [8],
                    limits=limits, row_limits=None)

    assert with_completion.bundle == without.bundle


def test_the_per_answer_cap_and_the_item_limit_hold():
    first = _chunk(1, 0, "An ordinary passage ranked first.", filename="agreement.pdf")
    recorder = _Recorder(texts={8: D12_STORED_EXCERPT}, chunks=[first, _d12(8, 26), _d12(8, 27)])
    limits = _limits(max_evidence_items=2)

    result = _plan(recorder, [_doc(1, "agreement.pdf"), _doc(8)], [1, 8], limits=limits,
                   row_limits=RowCompletionLimits(max_total_chars=100))

    assert [item.chunk_index for item in result.bundle.items] == [0, 26]
    assert EvidenceWarning.EVIDENCE_ITEM_LIMIT_REACHED.value in result.warnings
    # 92 (head) fits the 100-character cap; the 30-character tail no longer does.
    assert _sides(result.bundle.items[1]) == [(LEADING, 1, 4, 91)]
    assert sum(line.cost for item in result.bundle.items for line in item.row_completion) <= 100
    assert len(result.bundle.items) <= limits.max_evidence_items
    assert result.bundle.total_chars <= limits.total_evidence_chars


def test_chunks_past_the_item_limit_are_neither_loaded_nor_completed():
    first = _chunk(1, 0, "An ordinary passage ranked first.", filename="agreement.pdf")
    recorder = _Recorder(texts={8: D12_STORED_EXCERPT}, chunks=[first, _d12(8, 26)])

    result = _plan(recorder, [_doc(1, "agreement.pdf"), _doc(8)], [1, 8], limits=_limits(max_evidence_items=1))

    assert list(result.bundle.items) == [first]
    assert recorder.loaded == [], "no stored text is read for evidence the item limit drops"


def test_a_side_over_the_per_side_bound_is_skipped_and_the_other_kept():
    recorder = _Recorder(texts={8: D12_STORED_EXCERPT}, chunks=[_d12(8, 26)])

    result = _plan(recorder, [_doc(8)], [8], row_limits=RowCompletionLimits(max_side_chars=50))

    assert _sides(result.bundle.items[0]) == [(TRAILING, 1, 4, 29)]


def test_page_metadata_is_kept_and_a_row_from_another_page_is_refused():
    chunk = _d12(8, 26)
    recorder = _Recorder(texts={8: D12_STORED_EXCERPT}, chunks=[chunk])
    item = _plan(recorder, [_doc(8)], [8]).bundle.items[0]
    assert (item.page_start, item.page_end, item.locator_json) == (
        chunk.page_start, chunk.page_end, chunk.locator_json
    )

    # The cut "Other reserves" line now names page 5, while the chunk's metadata covers page 4 only:
    # its continuation is refused, and the "Share capital" head on page 4 is still supplied.
    retag = ("[Table 1, page 4] Other reserves", "[Table 1, page 5] Other reserves")
    moved_chunk = dataclasses.replace(chunk, text=D12_CHUNK_26.replace(*retag))
    recorder = _Recorder(texts={8: D12_STORED_EXCERPT.replace(*retag)}, chunks=[moved_chunk])
    item = _plan(recorder, [_doc(8)], [8]).bundle.items[0]
    assert _sides(item) == [(LEADING, 1, 4, 91)]
    assert "(6,373)" not in item.text and (item.page_start, item.page_end) == (4, 4)


# --- 6. main's own composition: owner-scoped loader, duplicate filenames across users ------


_EMBEDDING_PROFILE_KEYS = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_QUERY_PREFIX",
    "EMBEDDING_DOCUMENT_PREFIX",
    "EMBEDDING_PROFILE_UNVERIFIED_ACK",
)


@contextlib.contextmanager
def _imported_main():
    """Import a fresh ``main`` and restore ``sys.modules`` exactly afterwards."""
    previous = sys.modules.pop("main", None)
    try:
        yield importlib.import_module("main")
    finally:
        sys.modules.pop("main", None)
        if previous is not None:
            sys.modules["main"] = previous


@pytest.fixture
def main_module(monkeypatch):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    for key in _EMBEDDING_PROFILE_KEYS:
        monkeypatch.delenv(key, raising=False)
    with _imported_main() as module:
        yield module


class _DocumentsTable:
    """Rows of (id, owner, stored text), served only to the exact owner-scoped SELECT."""

    QUERY = "SELECT content FROM documents WHERE id = %s AND user_id = %s"

    def __init__(self, rows):
        self.rows = rows
        self.executed = []

    def connect(self):
        table = self

        class _Cursor:
            def execute(self, sql, params):
                statement = " ".join(sql.split())
                table.executed.append((statement, tuple(params)))
                self._row = None
                if statement == table.QUERY:
                    self._row = next(
                        ((text,) for (row_id, owner, text) in table.rows if (row_id, owner) == tuple(params)),
                        None,
                    )

            def fetchone(self):
                return self._row

            def close(self):
                pass

        class _Connection:
            def cursor(self):
                return _Cursor()

            def close(self):
                pass

        return _Connection()


def test_main_completes_through_its_owner_scoped_loader(monkeypatch, main_module):
    table = _DocumentsTable([
        (8, "owner-a", D12_STORED_EXCERPT),
        (10, "owner-a", D12_STORED_EXCERPT.replace("(6,373)", "(1,111)")),
        (12, "owner-b", D12_STORED_EXCERPT.replace("(6,373)", "(2,222)")),
    ])
    monkeypatch.setattr(main_module, "connect_to_postgres", table.connect)
    scope = _scope(_doc(8), _doc(10), user_id="owner-a")
    planner = dataclasses.replace(
        main_module._build_answer_evidence_planner(scope),
        retrieve=lambda question, document_ids, top_k: [_d12(12, 26), _d12(10, 26), _d12(8, 26)],
        limits=dataclasses.replace(
            main_module._EVIDENCE_LIMITS, complete_document_chars=HYBRID_PATH_COMPLETE_DOCUMENT_CHARS
        ),
    )

    result = planner.plan(EvidenceRequest(question=QUESTION, requested_document_ids=(8,)), scope)

    assert [item.document_id for item in result.bundle.items] == [8]
    assert result.bundle.items[0].text.endswith("Audited As at 31.05.2025 RM'000: (6,373)")
    assert table.executed == [(_DocumentsTable.QUERY, (8, "owner-a"))], (
        "only the requested document is read, by id and owner"
    )
    registry = CitationRegistry.from_bundle(result.bundle, main_module._CITATION_LIMITS)
    assert OTHER_RESERVES_ROW in build_generation_prompt(QUESTION, registry)


def test_main_never_reads_another_users_same_named_document(monkeypatch, main_module):
    table = _DocumentsTable([(12, "owner-b", D12_STORED_EXCERPT)])
    monkeypatch.setattr(main_module, "connect_to_postgres", table.connect)
    # A scope that wrongly lists another user's id still cannot read it: the SELECT is owner-scoped.
    scope = _scope(_doc(12), user_id="owner-a")
    planner = dataclasses.replace(
        main_module._build_answer_evidence_planner(scope),
        retrieve=lambda question, document_ids, top_k: [_d12(12, 26), _d12(12, 27)],
    )

    result = planner.plan(EvidenceRequest(question=QUESTION, requested_document_ids=(12,)), scope)

    assert [item.text for item in result.bundle.items] == [D12_CHUNK_26, D12_CHUNK_27]
    assert table.executed == [(_DocumentsTable.QUERY, (12, "owner-a"))]
