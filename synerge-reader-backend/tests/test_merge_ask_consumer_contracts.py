"""Source contracts for every /ask consumer after integrating PR #53 into E2.

GridApp.jsx is read as text; nothing is bundled, rendered, or executed, and no
npm, browser, network, database, or Ollama access occurs.

What these prove: there are exactly four /ask consumers and each one reads the
NDJSON transport with the shared decoder, decodes chunks as a continuing stream,
flushes the decoder, and declares its answer mode; failure is sticky (an error
or done(ok:false) cannot be undone by a later event) and a stream with no done
event is treated as incomplete; the Compare explanation no longer sniffs
sentinels or renders raw frames and keeps each passage's document id; the
selection actions main added keep document provenance; suggestions are scoped to
the uploaded document under the caller's token; AI-analysis answers never show
a document-support headline, a source card, or a citation chip, and chat and
Compare strip markers by one shared rule; chat failure stays sticky after an
error; and a newly indexed document carries its backend id and a persisted
flag.

What these do NOT prove: runtime rendering, network behaviour, or that the
model follows its prompt.
"""

from pathlib import Path
import re

import pytest

_GRIDAPP = Path(__file__).resolve().parents[2] / "synerge-reader-frontend" / "src" / "GridApp.jsx"


@pytest.fixture(scope="module")
def src():
    return _GRIDAPP.read_text(encoding="utf-8")


def _between(text, start, end):
    a = text.index(start)
    return text[a : text.index(end, a)]


def _send_message(src):
    return _between(src, "const sendMessage = useCallback", "  const handleSearchSelectionSources")


def _run_tool_query(src):
    return _between(src, "const runToolQuery = useCallback", "  // ── AI explanation of a diff")


def _explain_diff_pair(src):
    return _between(src, "const explainDiffPair = useCallback", "}, [authToken]);")


def _fetch_suggestions(src):
    return _between(src, "async function fetchSuggestions(", "  // ── ask / stream")


_CONSUMERS = {
    "sendMessage": _send_message,
    "runToolQuery": _run_tool_query,
    "explainDiffPair": _explain_diff_pair,
    "fetchSuggestions": _fetch_suggestions,
}

# The three non-chat consumers share one completion rule; sendMessage keeps its
# own richer message-state version, which test_frontend_citation_contracts pins.
_SHARED_RULE_CONSUMERS = ("runToolQuery", "explainDiffPair", "fetchSuggestions")


def test_there_are_exactly_four_ask_consumers(src):
    assert src.count("${BACKEND}/ask`") == 4
    for name, block in _CONSUMERS.items():
        assert "${BACKEND}/ask`" in block(src), f"{name} is not where its /ask call is"


@pytest.mark.parametrize("name", sorted(_CONSUMERS))
def test_every_consumer_reads_the_ndjson_stream_with_the_shared_decoder(src, name):
    block = _CONSUMERS[name](src)
    assert "createEventDecoder()" in block
    assert "dec.decode(value, { stream: true })" in block, (
        "a multi-byte character split across chunks must not be corrupted"
    )
    assert "decoder.close().forEach(applyEvent)" in block, "a final unterminated line is still read"
    assert 'event.type === "delta"' in block
    for sniffed in ("__ENTRY_ID__", "__ERROR__", "__READY__", "__CONTEXT__", "__SEARCHING__"):
        assert sniffed not in block


@pytest.mark.parametrize(
    ("name", "mode"),
    [
        ("runToolQuery", '"structured_json"'),
        ("fetchSuggestions", '"structured_json"'),
        ("explainDiffPair", '"model_reasoning"'),
    ],
)
def test_each_non_chat_consumer_declares_its_answer_mode(src, name, mode):
    assert f"mode: {mode}" in re.sub(r"mode:\s+", "mode: ", _CONSUMERS[name](src))


def test_chat_grades_documents_but_sends_ai_analysis_as_model_reasoning(src):
    send = _send_message(src)
    assert 'const answerMode = flags?.unverified ? "model_reasoning" : "document_qa";' in send
    assert "mode:                 answerMode," in send


@pytest.mark.parametrize("name", _SHARED_RULE_CONSUMERS)
def test_failure_is_sticky_and_a_missing_done_is_incomplete(src, name):
    block = _CONSUMERS[name](src)
    assert 'event.type === "error"' in block and "failed = true" in block
    assert "if (event.ok === false) failed = true;" in block
    assert "finished = true;" in block
    assert len(re.findall(r"\bfailed\s*=\s*false\b", block)) == 1, (
        "failed is only initialised to false: no later event may clear a recorded failure"
    )
    after_stream = block[block.index("decoder.close().forEach(applyEvent)"):]
    assert re.search(r"if \((failed \|\| !finished|failed)\)", after_stream), (
        "the result is checked for failure only after the whole stream was read"
    )
    assert "!finished" in after_stream


def test_chat_failure_is_sticky_once_an_error_arrives(src):
    """PR #54: pins the chat consumer's existing rule. An error event records
    failure, the done handler can only add failure, and nothing clears it -- so
    an error followed by done(ok:true) still ends failed."""
    send = _send_message(src)
    error_branch = _between(send, 'else if (event.type === "error")', 'else if (event.type === "done")')
    assert "failed = true;" in error_branch
    done_branch = _between(send, 'else if (event.type === "done")', "while (true)")
    assert "if (event.ok === false) failed = true;" in done_branch
    assert not re.search(r"\bfailed\s*=\s*(false|!|event\.ok)", done_branch), (
        "done(ok:true) must never clear a failure recorded earlier"
    )
    assert len(re.findall(r"\bfailed\s*=\s*false\b", send)) == 1, (
        "failed is only initialised to false"
    )
    after_stream = send[send.index("decoder.close().forEach(applyEvent)"):]
    assert "incomplete: !finished || failed," in after_stream
    assert "if (!failed && citations.length) handleCitation(citations[0]);" in after_stream


# --- AI analysis links to no source (PR #54) -----------------------------------


def test_ai_analysis_builds_no_source_citations_whatever_ids_arrive(src):
    verification = _between(
        _send_message(src), 'if (event.type === "verification")', 'else if (event.type === "entry_id")'
    )
    assert 'citations = answerMode === "document_qa"' in verification
    assert re.search(r"\.map\(\(record, index\) => \(\{ \.\.\.record, displayNumber: index \+ 1 \}\)\)\s*: \[\];", verification), (
        "anything other than a graded document answer gets no citations"
    )


def test_ai_analysis_renders_no_source_cards_or_citation_chips(src):
    assert (
        "<AnswerText text={msg.text} citations={msg.citations} pending={msg.streaming || msg.sourcesPending} "
        "onOpen={handleCitation} linkCitations={!msg.unverified} />"
    ) in src
    assert "{!msg.unverified && msg.citations?.length > 0 && (" in src
    answer = _between(src, "function AnswerText(", "const body = text")
    assert "linkCitations = true" in answer
    assert "if (!linkCitations) return <>{stripCitationMarkers(text)}</>;" in answer


def test_chat_and_compare_share_one_marker_stripping_rule(src):
    helper = _between(src, "function stripCitationMarkers(text)", "// ── citation text matching")
    assert 'return (text || "").replace(/\\s*\\[C\\d+\\]/g, "");' in helper
    assert "return stripCitationMarkers(full).trim();" in _explain_diff_pair(src)
    assert src.count("\\[C\\d+\\]/g") == 1, "one stripping rule, not a second copy drifting apart"


def test_the_compare_explanation_is_never_raw_stream_text(src):
    explain = _explain_diff_pair(src)
    assert "const raw = dec.decode(value)" not in explain
    assert "full += raw" not in explain
    assert "if (!res.ok) throw new Error(" in explain, "an HTTP error is an error, not an explanation"
    assert 'throw new Error(streamError || "The explanation did not complete.")' in explain
    assert 'throw new Error("The explanation ended before it finished.")' in explain


def test_the_compare_explanation_keeps_each_passages_document_id(src):
    explain = _explain_diff_pair(src)
    assert "document_id: Number.isInteger(baseline.docId) ? baseline.docId : null" in explain
    assert "document_id: Number.isInteger(candidate.docId) ? candidate.docId : null" in explain


# --- selection provenance -----------------------------------------------------


@pytest.mark.parametrize(
    "handler", ["handleSearchSelectionSources", "handleSummarizeSelection"]
)
def test_selection_actions_pass_the_document_id(src, handler):
    block = _between(src, f"const {handler} = useCallback", "[sendMessage]")
    assert f"const {handler} = useCallback((text, docName, docId) =>" in src
    assert "{ text, docName, docId }" in block
    assert f'{handler}(text, activeDoc?.name || "", activeDoc?.id ?? null);' in src


def test_ask_about_selection_still_records_the_document_id(src):
    assert 'setSelectedContext({ text: selPopover.text, docName: activeDoc?.name || "", docId: activeDoc?.id ?? null });' in src


# --- suggestions scope ------------------------------------------------------------


def test_suggestions_are_scoped_to_the_uploaded_document_under_the_callers_token(src):
    suggestions = _fetch_suggestions(src)
    assert "async function fetchSuggestions(snippet, filename, model, documentId, token)" in src
    assert "active_document_id: Number.isInteger(documentId) ? documentId : null," in suggestions
    assert "auth_token: token || null," in suggestions
    assert "authToken" not in suggestions, "the caller passes its current token in; none is captured"
    assert (
        'fetchSuggestions(parsed.text.slice(0, 2500), file.name, task?.model || "llama3.1:8b", docId, authToken);'
        in src
    )


# --- rendering and document state ---------------------------------------------


def test_ai_analysis_never_shows_a_document_support_headline(src):
    guarded = src.index("{!msg.unverified && (")
    summary = src.index("<ClaimSupportSummary claims={msg.claims}")
    assert guarded < summary < guarded + 400
    assert src.count("<ClaimSupportSummary") == 1


def test_all_three_answer_notices_survive_the_merge(src):
    assert "⚠ This answer did not finish streaming." in src
    assert "Checking sources…" in src
    assert "AI analysis only — any cases or citations above are not verified against a source." in src


def test_an_indexed_upload_carries_its_backend_id_and_is_persisted(src):
    upload = _between(src, "const processFiles = useCallback", "addedDocs.push(newDoc);")
    assert "id:                docId," in upload
    assert "persisted:         true," in upload
    assert "Date.now()" not in upload.split("const newDoc = {")[1], (
        "a client timestamp would pass Number.isInteger and be sent as a document id"
    )
