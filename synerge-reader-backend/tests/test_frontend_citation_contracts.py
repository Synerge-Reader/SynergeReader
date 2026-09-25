"""Source-level contracts for the E2 frontend integration in GridApp.jsx.

Text/source-shape tests. GridApp.jsx is read as plain text; nothing here
bundles, transpiles, imports, renders, or executes frontend code, and no npm,
browser, network, database, or Ollama access occurs.

What these prove: All Documents mode no longer ships a concatenated dump of
every open document as if the user had highlighted it, an explicit highlight is
still sent as selected evidence, document scope travels as backend ids, the
stream is parsed as NDJSON across chunk boundaries rather than by sentinel
sniffing, citations render as filename plus the backend's truthful locator, an
unknown citation id is not interactive, the four claim states stay visually
distinct, and the E1b original-file upload path is untouched.

What these do NOT prove: that the UI renders correctly, that the click handler
scrolls to the right page, or anything about runtime behaviour.
"""

from pathlib import Path
import re

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_GRID_APP_PATH = _REPOSITORY_ROOT / "synerge-reader-frontend" / "src" / "GridApp.jsx"


@pytest.fixture(scope="module")
def grid_app_source():
    assert _GRID_APP_PATH.is_file()
    return _GRID_APP_PATH.read_text(encoding="utf-8")


def _slice(source, start_marker, end_marker, label):
    start = source.find(start_marker)
    assert start != -1, f"could not find {start_marker!r} in {label}"
    end = source.find(end_marker, start)
    assert end != -1, f"could not find {end_marker!r} after {start_marker!r} in {label}"
    return source[start:end]


@pytest.fixture(scope="module")
def send_message_block(grid_app_source):
    block = _slice(
        grid_app_source,
        "const sendMessage = useCallback",
        "// ── task-mode tools",
        "GridApp.jsx",
    )
    assert "${BACKEND}/ask" in block
    return block


@pytest.fixture(scope="module")
def tool_query_block(grid_app_source):
    block = _slice(
        grid_app_source,
        "const runToolQuery = useCallback",
        "const runArgumentTool",
        "GridApp.jsx",
    )
    assert "${BACKEND}/ask" in block
    return block


# --- 31: All Documents no longer sends combined document text --------------


def test_all_documents_mode_no_longer_sends_combined_document_text(send_message_block):
    for banned in ("combinedSelections", "perDocLimit", "(d.text || \"\").slice("):
        assert banned not in send_message_block, (
            f"{banned!r} rebuilt the client-side document dump that this "
            "milestone replaces with authorized server-side retrieval"
        )


def test_tool_queries_also_stopped_sending_the_document_dump(tool_query_block):
    for banned in ("combinedSelections", "perDocLimit", "(d.text || \"\").slice("):
        assert banned not in tool_query_block
    assert "selections:           []," in tool_query_block, (
        "a tool query has no user highlight, so it sends no selected evidence"
    )


def test_no_document_text_is_read_into_the_ask_request(send_message_block):
    assert "d.text" not in send_message_block, (
        "document text belongs to the local preview only; the server holds the "
        "documents it retrieves from"
    )


# --- 32: explicit selected text is still transmitted -----------------------


def test_explicit_selected_text_remains_transmitted(send_message_block):
    assert "selected_text:        askedContext?.text || \"\"," in send_message_block
    assert "const explicitSelections = askedContext?.text" in send_message_block
    assert "selections:           explicitSelections," in send_message_block
    assert "text:          askedContext.text," in send_message_block


def test_a_selection_carries_its_document_id_for_attribution(
    grid_app_source, send_message_block
):
    assert "document_id:   Number.isInteger(askedContext.docId) ? askedContext.docId : null," in send_message_block
    assert "docId: activeDoc?.id ?? null" in grid_app_source, (
        "the highlight must record which open document it came from"
    )


# --- 33: backend document ids define scope ---------------------------------


def test_backend_document_ids_define_scope(send_message_block):
    assert "const scopeDocumentIds = docs.map(d => d.id).filter(id => Number.isInteger(id));" in send_message_block
    assert "document_ids:         isAllScope ? scopeDocumentIds : []," in send_message_block
    assert "active_document_id:   isAllScope ? null : (Number.isInteger(activeDoc?.id) ? activeDoc.id : null)," in send_message_block


def test_tool_queries_use_the_same_id_scope(tool_query_block):
    assert "const scopeDocumentIds = docs.map(d => d.id).filter(id => Number.isInteger(id));" in tool_query_block
    assert "document_ids:" in tool_query_block
    assert "active_document_id:" in tool_query_block


# --- structured transport ---------------------------------------------------


def test_the_stream_is_parsed_as_ndjson_events(send_message_block):
    assert "createEventDecoder()" in send_message_block
    for event_type in ("evidence", "delta", "verification", "entry_id", "error", "done"):
        assert f'event.type === "{event_type}"' in send_message_block, (
            f"the {event_type} event must be handled explicitly"
        )


def test_chunk_boundaries_are_buffered_not_assumed(grid_app_source, send_message_block):
    assert "function createEventDecoder()" in grid_app_source
    assert 'buffer.indexOf("\\n")' in grid_app_source, (
        "lines must be reassembled from a buffer, since a chunk can split one"
    )
    assert "dec.decode(value, { stream: true })" in send_message_block, (
        "the text decoder must be told the stream continues, so a multi-byte "
        "character split across chunks is not corrupted"
    )
    assert "decoder.close()" in send_message_block


def test_sentinel_sniffing_is_gone(grid_app_source):
    for banned in ("__ENTRY_ID__", "__ERROR__", "__CONTEXT__", "__READY__", "__SEARCHING__"):
        assert banned not in grid_app_source, (
            f"{banned} parsing treats generated text as transport control"
        )


def test_an_unfinished_stream_is_not_shown_as_a_finished_answer(send_message_block):
    assert "incomplete: !finished || failed," in send_message_block
    assert "The answer stream ended before it finished." in send_message_block


def test_done_ok_false_is_honored_by_the_done_handler(send_message_block):
    """done(ok:false) must fail the answer, not merely end it.

    The backend closes an evidence or generation failure with an explicit
    unsuccessful done. A handler that only set finished=true would render a
    half-answer as complete, auto-navigate to a citation, and count it toward
    the knowledge base.
    """
    done_branch = _slice(
        send_message_block,
        'else if (event.type === "done")',
        "};",
        "sendMessage",
    )

    assert "finished = true;" in done_branch
    assert "if (event.ok === false) failed = true;" in done_branch, (
        "the done handler must evaluate event.ok and drive failed"
    )


def test_done_ok_false_is_evaluated_before_the_completion_decisions(send_message_block):
    """Source order: failed is set inside the event handler, which runs while
    the stream is consumed, strictly before the post-stream decisions read it."""
    ok_check = send_message_block.index("if (event.ok === false) failed = true;")
    incomplete = send_message_block.index("incomplete: !finished || failed,")
    navigation = send_message_block.index("if (!failed && citations.length) handleCitation(")
    kb_count = send_message_block.index("if (!failed) setKbCount(")

    assert ok_check < incomplete < navigation < kb_count, (
        "done(ok:false) must be evaluated before the incomplete flag, the "
        "citation auto-navigation, and the knowledge-base counter"
    )


def test_the_three_completion_decisions_all_key_off_failed(send_message_block):
    assert "incomplete: !finished || failed," in send_message_block, (
        "a failed stream is shown as incomplete"
    )
    assert "if (!failed && citations.length) handleCitation(citations[0]);" in send_message_block, (
        "a failed stream must not auto-navigate to a citation"
    )
    assert "if (!failed) setKbCount(k => k + 1);" in send_message_block, (
        "a failed stream must not increment the knowledge-base counter"
    )
    assert "msg.incomplete && !msg.streaming" in _GRID_APP_PATH.read_text(encoding="utf-8"), (
        "the existing incomplete banner is what surfaces the warning"
    )


# --- 34/35: citation rendering ---------------------------------------------


def test_citations_render_filename_and_locator(grid_app_source):
    # A source card now carries the display number, the document, the truthful
    # locator and the server excerpt -- the raw Cn id is no longer the label.
    assert "const source = citation.filename || citation.display_source" in grid_app_source
    assert 'const locator = citation.locator?.label || "Relevant passage";' in grid_app_source
    assert "{citation.displayNumber}" in grid_app_source
    assert "<CitationCard key={c.citation_id || i} citation={c} onOpen={handleCitation} />" in grid_app_source
    assert "page={c.page}" not in grid_app_source, (
        "citations no longer come from scraping page numbers out of the answer"
    )


def test_citation_pages_come_from_the_backend_locator_only(grid_app_source):
    assert "const page = citation.locator?.page_start ?? null;" in grid_app_source, (
        "a citation may only highlight a page the backend actually reported"
    )
    assert "setHlPage(page);" in grid_app_source
    # Every page the viewer is told to highlight comes from the backend
    # locator, an explicit numeric argument, a matched TXT word offset, or is
    # cleared -- never from parsing the answer.
    allowed = {
        "setHlPage(citation);",
        "setHlPage(page);",
        "setHlPage(Math.floor(range.start / TXT_WORDS_PER_PAGE) + 1);",
        "setHlPage(null);",
    }
    for call in re.findall(r"setHlPage\([^;]*\);", grid_app_source):
        assert call in allowed, f"unexpected page source: {call}"
    assert "matchAll(/\\b(?:page|p\\.)" not in grid_app_source, (
        "page numbers must not be regex-scraped out of the generated answer"
    )


def test_unknown_citation_ids_are_not_interactive(grid_app_source):
    inline = _slice(
        grid_app_source, "function InlineCitation(", "function AnswerText(", "GridApp.jsx"
    )
    unresolved = _slice(inline, "if (!citation) {", "return (\n    <button", "InlineCitation")

    assert "<span" in unresolved and "onClick" not in unresolved, (
        "a marker the model invented must render as inert text, never a button"
    )
    assert "{token}" in unresolved, (
        "an unresolved marker stays visibly what the model wrote"
    )
    assert "if (!citation || citation.invalid) return;" in grid_app_source, (
        "the click handler must also refuse an invalid citation"
    )


def test_citation_excerpts_are_shown_as_the_bounded_backend_excerpt(grid_app_source):
    card = _slice(grid_app_source, "function CitationCard(", "function InlineCitation(", "GridApp.jsx")
    assert "{citation.excerpt}" in card, (
        "the quotation shown is the backend's bounded excerpt, never generated prose "
        "and never local document text"
    )
    assert "doc.text" not in card


# --- 36: claim verification states stay distinguishable --------------------


def test_claim_verification_states_remain_distinguishable(grid_app_source):
    # Every state the backend can report has its own reader-facing wording, and
    # the reason is what separates "checked and unsupported" from "never
    # checked". Collapsing those is the defect this replaces.
    for key, label in (
        ("supported", "Supported"),
        ("partially_supported", "Partly supported"),
        ("no_citation", "Not supported by a cited document"),
        ("invalid_citation", "Citation could not be resolved"),
        ("not_checked", "Not checked"),
        ("check_unavailable", "Check unavailable"),
    ):
        assert f"{key}:" in grid_app_source, f"{key} must have its own presentation"
        assert f'label: "{label}"' in grid_app_source


def test_claim_states_are_derived_from_the_backend_status_and_reason(grid_app_source):
    assert "<ClaimSupportSummary claims={msg.claims} citations={msg.citations} onOpen={handleCitation} />" in grid_app_source
    assert "claims: event.claims || []" in grid_app_source, (
        "claim states come from the backend verification event, not from the UI"
    )
    mapper = _slice(grid_app_source, "function claimStateKey(", "const SUPPORT_HEADLINE", "GridApp.jsx")
    assert 'claim.reason === "invalid_citation"' in mapper
    assert 'claim.reason === "verifier_unavailable"' in mapper
    assert 'claim.reason === "verifier_malformed"' in mapper


def test_verified_is_not_used_as_a_user_facing_claim_label(grid_app_source):
    assert 'label: "verified"' not in grid_app_source
    assert 'label: "not verified"' not in grid_app_source, (
        "'not verified' hid the difference between unsupported and unchecked"
    )
    assert "2 verified" not in grid_app_source


# --- 37: E1b upload behaviour is untouched ---------------------------------


def test_original_file_upload_behaviour_is_unchanged(grid_app_source):
    block = _slice(
        grid_app_source,
        "const processFiles = useCallback",
        "// Generate suggested questions",
        "GridApp.jsx",
    )
    assert 'fd.append("files", file, file.name)' in block, (
        "E1b's original-file transport must survive this milestone"
    )
    assert "new Blob(" not in block
    assert 'result.status !== "indexed"' in block
    assert "result?.error_message" in block


def test_local_preview_parsing_is_retained(grid_app_source):
    for parser in ("await parsePDF(file)", "await parseDOCX(file)", "await parseTXT(file)"):
        assert parser in grid_app_source, (
            "local parsing still powers preview, page display and selection"
        )


# --- E2b: viewer, used-only sources, display numbering, claim language ------


def test_txt_page_card_does_not_shrink(grid_app_source):
    """The TXT card sits in a flex column; without this it collapses, clips its
    own text, and leaves the outer panel nothing to scroll."""
    card = _slice(
        grid_app_source,
        '{textPages.map((pageWords, i) => {',
        "<div style={{\n                display: \"flex\", justifyContent: \"space-between\"",
        "GridApp.jsx",
    )
    assert 'overflow: "hidden"' in card
    assert "flexShrink: 0," in card, (
        "the TXT page card must declare flexShrink: 0"
    )


def test_shared_viewer_containers_were_not_restyled(grid_app_source):
    # The fix belongs to the TXT card only; the scrolling container keeps its
    # flex: 1 / overflow: auto ownership.
    assert grid_app_source.count('flex: 1, overflow: "auto", background: "#e9ecef"') == 2


# --- sources appear only once the answer says which it used ----------------


def test_candidate_evidence_is_not_rendered_as_sources(send_message_block):
    evidence_branch = _slice(
        send_message_block, 'if (event.type === "evidence")', "} else if (event.type === \"delta\")", "sendMessage"
    )
    assert "candidates = Array.isArray(event.citations)" in evidence_branch, (
        "the evidence event carries candidates, not sources"
    )
    assert "citations: []" in evidence_branch, (
        "no source chip may render before verification says what was cited"
    )
    assert "sourcesPending: true" in evidence_branch


def test_sources_come_from_used_citation_ids(send_message_block):
    verification = _slice(
        send_message_block,
        'if (event.type === "verification")',
        'else if (event.type === "entry_id")',
        "sendMessage",
    )
    assert "(event.used_citation_ids || [])" in verification
    assert ".map(id => byId[id])" in verification
    assert ".filter(Boolean)" in verification
    assert "displayNumber: index + 1" in verification, (
        "display numbers follow first use, which is the order the backend sends"
    )
    assert "sourcesPending: false" in verification


def test_display_numbering_does_not_follow_candidate_position(send_message_block):
    verification = _slice(
        send_message_block,
        'if (event.type === "verification")',
        'else if (event.type === "entry_id")',
        "sendMessage",
    )
    # The map is keyed by id and the number comes from the position in
    # used_citation_ids, never from the candidate array index.
    assert "candidates.forEach(c => { byId[c.citation_id] = c; });" in verification
    assert "candidates[index]" not in verification
    assert "candidates.map((record, index)" not in verification


def test_a_stream_without_verification_presents_no_sources(send_message_block):
    assert "citations: verified ? msg.citations : []," in send_message_block, (
        "if generation ends before verification, candidates must not be shown "
        "as cited sources"
    )
    assert "let   verified = false;" in send_message_block


def test_a_neutral_pending_state_is_shown_while_checking(grid_app_source):
    assert "{msg.sourcesPending && (" in grid_app_source
    assert "Checking sources…" in grid_app_source


# --- inline marker rewriting ------------------------------------------------


def test_only_exact_citation_tokens_are_rewritten(grid_app_source):
    answer = _slice(grid_app_source, "function AnswerText(", "// What happened to one claim", "GridApp.jsx")

    assert "const pattern = /\\[C(\\d+)\\]/g;" in answer, (
        "only exact [Cn] tokens are candidates for rewriting"
    )
    assert "byId[id]" in answer, "a token is rewritten only if that id resolves"
    assert "nodes.push(body.slice(last, match.index))" in answer, (
        "text between markers is passed through untouched"
    )


def test_resolved_markers_render_the_display_number(grid_app_source):
    inline = _slice(grid_app_source, "function InlineCitation(", "function AnswerText(", "GridApp.jsx")
    assert ">{citation.displayNumber}</button>" in inline
    assert "citation.citation_id" not in inline, (
        "the internal id is never the visible marker"
    )
    assert "aria-label={`Open source ${citation.displayNumber}" in inline, (
        "the marker keeps an accessible name"
    )


def test_no_internal_identifier_or_score_is_rendered_as_a_label(grid_app_source):
    card = _slice(grid_app_source, "function CitationCard(", "function InlineCitation(", "GridApp.jsx")
    assert "citation.citation_id" not in card, "C1..Cn is not a user-facing label"
    assert "scores" not in card and "similarity" not in card
    assert not re.search(r"chunk\s*\$?\{?\d", grid_app_source, re.IGNORECASE)
    assert not re.search(r'"chunk ', grid_app_source)


# --- support summary language ----------------------------------------------


@pytest.mark.parametrize(
    "headline",
    [
        "Supported by your documents",
        "Parts of this answer are not supported by your documents",
        "Citation check incomplete",
        "Citation check unavailable",
    ],
)
def test_support_headline_language_is_exact(grid_app_source, headline):
    assert f'"{headline}"' in grid_app_source


def test_headline_precedence_is_explicit(grid_app_source):
    chooser = _slice(grid_app_source, "function supportHeadlineKey(", "function ClaimSupportSummary(", "GridApp.jsx")
    unsupported = chooser.index('return "unsupported"')
    unavailable = chooser.index('return "unavailable"')
    incomplete = chooser.index('return "incomplete"')
    supported = chooser.index('return "supported"')
    assert unsupported < unavailable < incomplete < supported, (
        "an unsupported claim is the most important thing to say; a fully "
        "checked answer is the last case left"
    )


def test_limit_skipped_and_failed_checks_read_differently(grid_app_source):
    assert 'not_checked:         { label: "Not checked"' in grid_app_source
    assert 'check_unavailable:   { label: "Check unavailable"' in grid_app_source
    mapper = _slice(grid_app_source, "function claimStateKey(", "const SUPPORT_HEADLINE", "GridApp.jsx")
    assert '"check_unavailable"' in mapper and '"not_checked"' in mapper


def test_the_answer_stays_visible_when_verification_is_incomplete(grid_app_source):
    # The summary is a sibling of the answer body, never a replacement for it.
    assert "<AnswerText text={msg.text} citations={msg.citations}" in grid_app_source
    assert "<ClaimSupportSummary claims={msg.claims} citations={msg.citations} onOpen={handleCitation} />" in grid_app_source


def test_markers_are_not_called_invalid_before_verification_arrives(grid_app_source):
    """A still-streaming answer must not accuse its own markers of being fake."""
    inline = _slice(grid_app_source, "function InlineCitation(", "function AnswerText(", "GridApp.jsx")
    assert "if (pending) {" in inline
    assert "pending={msg.streaming || msg.sourcesPending}" in grid_app_source
    pending_branch = _slice(inline, "if (pending) {", "return (\n      <span", "InlineCitation")
    assert "onClick" not in pending_branch, "a pending marker is not clickable either"
    assert "#9ca3af" in pending_branch, "it is neutral, not the invalid-marker red"


# --- TXT navigation and highlighting ---------------------------------------


def test_txt_navigation_matches_the_public_excerpt(grid_app_source):
    handler = _slice(grid_app_source, "const handleCitation = useCallback", "}, [docs]);", "GridApp.jsx")

    assert "findWordRange(normalizeWords(target.text), excerptNeedle(citation))" in handler
    assert "evidence_text" not in handler, (
        "the frontend never receives or uses the internal evidence text"
    )
    assert "setHlRange({ docId: target.id, start: range.start, end: range.end });" in handler
    assert "Math.floor(range.start / TXT_WORDS_PER_PAGE) + 1" in handler, (
        "the preview page is derived from the matched word offset"
    )


def test_excerpt_needle_drops_the_ellipsis_only_when_truncated(grid_app_source):
    needle = _slice(grid_app_source, "function excerptNeedle(", "function findWordRange(", "GridApp.jsx")
    assert "citation?.excerpt_truncated" in needle
    assert "citation?.excerpt" in needle
    assert "evidence_text" not in needle


def test_txt_highlighting_uses_react_nodes_not_raw_html(grid_app_source):
    assert "dangerouslySetInnerHTML" not in grid_app_source, (
        "highlighting must not inject raw HTML into the document preview"
    )
    renderer = _slice(grid_app_source, "function renderPageWords(", "// One cited source, as the reader sees it", "GridApp.jsx")
    assert "<mark style=" in renderer
    assert "pageWords.slice(0, from).join" in renderer, (
        "the surrounding text stays as plain text nodes, so selection still works"
    )


def test_a_citation_spanning_two_preview_pages_highlights_both(grid_app_source):
    viewer = _slice(grid_app_source, "const citedRange = highlightRange", "const hi = pg === highlightPage", "GridApp.jsx")
    assert "citedRange.end > offset" in viewer
    assert "citedRange.start < offset + pageWords.length" in viewer


def test_stale_highlighting_is_cleared_between_citations(grid_app_source):
    handler = _slice(grid_app_source, "const handleCitation = useCallback", "}, [docs]);", "GridApp.jsx")
    assert "setHlRange(null);" in handler
    assert "highlightRange.docId === doc.id" in grid_app_source, (
        "a highlight belongs to one document and must not bleed onto another"
    )


def test_pdf_page_navigation_is_preserved(grid_app_source):
    handler = _slice(grid_app_source, "const handleCitation = useCallback", "}, [docs]);", "GridApp.jsx")
    assert "const page = citation.locator?.page_start ?? null;" in handler
    assert "setHlPage(page);" in handler
    assert 'id={`vpg-${pg}`}' in grid_app_source, "page anchors still exist for scrolling"


def test_docx_and_unmatched_text_get_no_fabricated_page(grid_app_source):
    handler = _slice(grid_app_source, "const handleCitation = useCallback", "}, [docs]);", "GridApp.jsx")
    tail = handler[handler.index("// DOCX, an unmatched excerpt"):]
    assert "setHlPage(null);" in tail, (
        "with no real location the source opens without a fabricated page"
    )
    assert "setSourceOpen(true);" in tail
    assert "page_start" not in tail


def test_the_viewer_receives_the_highlight_range(grid_app_source):
    assert "function PdfViewer({ doc, highlightPage, highlightRange })" in grid_app_source
    assert "<PdfViewer doc={activeDoc} highlightPage={hlPage} highlightRange={hlRange} />" in grid_app_source


# --- Defect B: a complete document is not a passage ------------------------
#
# In complete-document mode the model receives the whole short document, but
# the public excerpt is only its bounded opening. Presenting that opening as
# the supporting quotation -- and matching it to place a highlight -- sent the
# reader to the first paragraph however far from it the real support lay.


@pytest.fixture(scope="module")
def citation_card_block(grid_app_source):
    return _slice(
        grid_app_source, "function CitationCard(", "function InlineCitation(", "GridApp.jsx"
    )


@pytest.fixture(scope="module")
def citation_handler_block(grid_app_source):
    return _slice(
        grid_app_source, "const handleCitation = useCallback", "}, [docs]);", "GridApp.jsx"
    )


@pytest.fixture(scope="module")
def complete_document_branch(citation_handler_block):
    return _slice(
        citation_handler_block,
        "if (isCompleteDocumentCitation(citation)) {",
        "// PDF:",
        "handleCitation",
    )


def test_the_complete_document_test_accepts_either_backend_field(grid_app_source):
    predicate = _slice(
        grid_app_source,
        "function isCompleteDocumentCitation(",
        "// Renders one preview page",
        "GridApp.jsx",
    )
    assert 'citation?.evidence_mode === "complete_document"' in predicate
    assert 'citation?.source_type === "complete_document"' in predicate


def test_a_complete_document_source_card_says_the_whole_document_was_reviewed(
    grid_app_source, citation_card_block
):
    assert 'const COMPLETE_DOCUMENT_LABEL = "Complete document reviewed";' in grid_app_source
    assert "const wholeDocument = isCompleteDocumentCitation(citation);" in citation_card_block
    assert (
        "const locatorLabel = wholeDocument ? COMPLETE_DOCUMENT_LABEL : locator;"
        in citation_card_block
    )
    assert "{locatorLabel}" in citation_card_block


def test_a_complete_document_card_does_not_quote_the_bounded_head_excerpt(
    citation_card_block,
):
    assert "{!wholeDocument && citation.excerpt && (" in citation_card_block, (
        "the excerpt of a complete-document citation is the document's opening, "
        "not the passage that supported the claim"
    )


def test_a_complete_document_card_opens_the_document_not_a_source(citation_card_block):
    assert '{wholeDocument ? "Open document" : "Open source"}' in citation_card_block


def test_an_ordinary_citation_keeps_its_passage_locator_and_excerpt(citation_card_block):
    assert 'const locator = citation.locator?.label || "Relevant passage";' in citation_card_block
    assert "{citation.excerpt}" in citation_card_block, (
        "every other citation still shows the backend's bounded excerpt"
    )


def test_a_complete_document_citation_clears_the_page_and_range(
    citation_handler_block, complete_document_branch
):
    assert "setHlPage(null);" in complete_document_branch
    assert "setSourceOpen(true);" in complete_document_branch
    assert citation_handler_block.index("setHlRange(null);") < citation_handler_block.index(
        "if (isCompleteDocumentCitation(citation)) {"
    ), "the stale TXT highlight range is cleared before this branch returns"


def test_a_complete_document_citation_still_selects_the_document(citation_handler_block):
    assert citation_handler_block.index(
        "if (target) setActiveDocId(target.id);"
    ) < citation_handler_block.index(
        "if (isCompleteDocumentCitation(citation)) {"
    ), "the cited document is opened before the branch returns"


def test_a_complete_document_citation_never_runs_txt_excerpt_matching(
    complete_document_branch,
):
    for banned in ("findWordRange", "excerptNeedle", "normalizeWords"):
        assert banned not in complete_document_branch, (
            f"{banned} would match the document's opening and call it the "
            "supporting passage"
        )


def test_a_complete_document_citation_fabricates_no_location(complete_document_branch):
    for banned in ("page_start", "setHlRange({", "TXT_WORDS_PER_PAGE", "paragraph", "line_start"):
        assert banned not in complete_document_branch


def test_a_complete_document_citation_shows_no_referenced_in_answer_banner(grid_app_source):
    # The banner has exactly two sites, both driven by the highlighted page or
    # range -- and the complete-document branch clears both, so it cannot show.
    assert "highlighted={pg === highlightPage}" in grid_app_source
    assert "const hi = pg === highlightPage || overlaps;" in grid_app_source
    assert grid_app_source.count("▲ Referenced in answer") == 2, (
        "the banner must stay driven only by a real highlighted page or range"
    )


def test_suggestions_no_longer_parse_the_retired_sentinels(grid_app_source):
    suggestions = _slice(grid_app_source, "async function fetchSuggestions(", "const sendMessage", "GridApp.jsx")
    assert "createEventDecoder()" in suggestions, (
        "every /ask consumer must read the NDJSON transport"
    )
    assert 'event.type === "delta"' in suggestions
    assert "SEARCHING" not in suggestions


# --- Final round: no internal identifier reaches the reader ----------------
#
# Registry ids stayed internal in the answer body but still leaked into the
# expanded verification rows, which rendered claim.text verbatim -- so a
# reader opening "details" saw "[C6][C1]". Two adjacent public markers also
# abutted, so [C2][C3] read as a single citation "23".


@pytest.fixture(scope="module")
def claim_summary_block(grid_app_source):
    return _slice(
        grid_app_source, "function ClaimSupportSummary(", "function DotsLoader(", "GridApp.jsx"
    )


@pytest.fixture(scope="module")
def answer_text_block(grid_app_source):
    return _slice(
        grid_app_source, "function AnswerText(", "// What happened to one claim", "GridApp.jsx"
    )


def test_expanded_claim_text_is_rewritten_not_printed_raw(claim_summary_block):
    assert ">{claim.text}<" not in claim_summary_block, (
        "printing the model's claim verbatim as a JSX child exposes its raw "
        "[Cn] registry ids"
    )
    assert "<AnswerText text={claim.text} citations={citations}" in claim_summary_block, (
        "claim rows must go through the same public-number rewriter as the "
        "answer body"
    )


def test_the_summary_receives_the_citations_it_needs_to_rewrite_with(
    grid_app_source, claim_summary_block
):
    assert "function ClaimSupportSummary({ claims, citations, onOpen })" in grid_app_source
    assert "onOpen={onOpen}" in claim_summary_block


def test_only_public_display_numbers_are_ever_rendered(
    grid_app_source, answer_text_block
):
    """The public marker is the display number; the registry id is a key only."""
    inline = _slice(
        grid_app_source,
        "function InlineCitation(",
        "function AnswerText(",
        "GridApp.jsx",
    )

    assert "{citation.displayNumber}" in inline, (
        "the marker a reader sees is the compact public number"
    )
    assert "citation.citation_id" not in inline, (
        "the internal registry id must never reach the rendered marker"
    )
    assert "citation.citation_id" not in answer_text_block, (
        "nor the surrounding answer text"
    )
    assert "byId[id]" in answer_text_block, (
        "the internal id survives only as a lookup key into the citations map"
    )


def test_adjacent_markers_render_as_separate_clickable_markers(answer_text_block):
    assert "} else if (nodes.length) {" in answer_text_block, (
        "two markers written back to back need something between them"
    )
    assert 'aria-hidden="true"' in answer_text_block
    assert "sep-${match.index}" in answer_text_block


def test_each_marker_is_its_own_button(grid_app_source):
    inline = _slice(
        grid_app_source, "function InlineCitation(", "function AnswerText(", "GridApp.jsx"
    )
    assert "<button" in inline
    assert "onClick={() => onOpen(citation)}" in inline, (
        "every public marker is independently clickable"
    )


def test_no_chunk_number_reaches_the_citation_ui(grid_app_source):
    assert not re.search(r"chunk\s*\$?\{?\d", grid_app_source, re.IGNORECASE)
    assert not re.search(r'"chunk ', grid_app_source)
    card = _slice(
        grid_app_source, "function CitationCard(", "function InlineCitation(", "GridApp.jsx"
    )
    assert "chunk_index" not in card and "chunk_id" not in card


def test_source_cards_keep_name_locator_excerpt_and_open_source(grid_app_source):
    card = _slice(
        grid_app_source, "function CitationCard(", "function InlineCitation(", "GridApp.jsx"
    )
    assert "const source = citation.filename || citation.display_source" in card
    assert 'const locator = citation.locator?.label || "Relevant passage";' in card
    assert "{citation.excerpt}" in card
    assert '"Open source"' in card


# --- Final round: generic framing does not turn a cited answer wholly red --


def test_a_cited_answer_is_not_made_wholly_red_by_one_framing_sentence(grid_app_source):
    chooser = _slice(
        grid_app_source, "function supportHeadlineKey(", "function ClaimSupportSummary(", "GridApp.jsx"
    )
    assert "if (unsupported > 0 && supported === 0) return \"unsupported\";" in chooser, (
        "red is reserved for an answer with no supported claim at all"
    )
    assert 'if (unsupported > 0) return "mixed";' in chooser, (
        "an answer mixing supported and unsupported claims says so, rather "
        "than presenting as wholly unsupported"
    )


def test_a_mixed_answer_says_some_claims_need_checking(grid_app_source):
    assert '"Some claims in this answer need checking"' in grid_app_source


def test_zero_supported_claims_stays_red(grid_app_source):
    chooser = _slice(
        grid_app_source, "function supportHeadlineKey(", "function ClaimSupportSummary(", "GridApp.jsx"
    )
    unsupported_rule = chooser.index('supported === 0) return "unsupported"')
    mixed_rule = chooser.index('if (unsupported > 0) return "mixed"')
    assert unsupported_rule < mixed_rule, (
        "the all-unsupported case must be decided before the mixed case, or "
        "a wholly unsupported answer would show as merely mixed"
    )


def test_headline_precedence_still_puts_the_worst_case_first(grid_app_source):
    chooser = _slice(
        grid_app_source, "function supportHeadlineKey(", "function ClaimSupportSummary(", "GridApp.jsx"
    )
    order = [
        chooser.index('return "unsupported"'),
        chooser.index('return "mixed"'),
        chooser.index('return "unavailable"'),
        chooser.index('return "incomplete"'),
        chooser.index('return "supported"'),
    ]
    assert order == sorted(order)


def test_detailed_claim_statuses_are_never_hidden(claim_summary_block):
    assert "claims.map((claim, i) =>" in claim_summary_block
    assert "CLAIM_STATE_LABEL[claimStateKey(claim)]" in claim_summary_block
    assert "{state.label}" in claim_summary_block


def test_an_evidence_mismatch_reads_differently_from_an_uncited_claim(grid_app_source):
    assert 'evidence_mismatch:   { label: "Cited source does not state this"' in grid_app_source
    mapper = _slice(grid_app_source, "function claimStateKey(", "const SUPPORT_HEADLINE", "GridApp.jsx")
    assert 'if (claim.reason === "evidence_terms_missing") return "evidence_mismatch";' in mapper, (
        "a claim whose cited source does not state it is not the same failure "
        "as a claim that cited nothing"
    )
