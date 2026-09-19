"""Source contracts for the document search / selection-tools / compare / delete
frontend, read as text from GridApp.jsx (the repo has no frontend test runner,
and, like the other tests here, these never execute the app).

Each test pins one review finding that was fixed, so it can't quietly come back:
what a button calls, what a prompt is allowed to demand, which context wins.
They check the code says the right thing; the behavior itself was verified by
running the app.
"""

import re
from pathlib import Path

import pytest

_GRIDAPP = Path(__file__).resolve().parents[2] / "synerge-reader-frontend" / "src" / "GridApp.jsx"


@pytest.fixture(scope="module")
def src():
    return _GRIDAPP.read_text(encoding="utf-8")


def _between(text, start, end):
    a = text.index(start)
    return text[a : text.index(end, a)]


# ---- delete: authenticated, by id, confirmed, scoped -------------------------


def test_delete_uses_the_authenticated_id_route_never_the_filename_route(src):
    assert "/documents/delete" not in src, "the unauthenticated filename-based route must not be called"
    handler = _between(src, "const handleDeleteDoc = useCallback", "async function fetchSuggestions")
    assert "`${BACKEND}/me/documents/${doc.id}`" in handler
    assert 'method: "DELETE"' in handler
    assert "Authorization: `Bearer ${authToken" in handler
    assert "filename" not in handler, "delete must identify the document by id, not by name"


def test_delete_needs_a_second_click_to_confirm(src):
    handler = _between(src, "const handleDeleteDoc = useCallback", "async function fetchSuggestions")
    # first click arms; only an already-armed click reaches the request
    assert "if (confirmDeleteId !== doc.id)" in handler
    assert handler.index("confirmDeleteId !== doc.id") < handler.index("fetch(")
    assert "Click again to permanently delete" in src


def test_delete_button_is_hidden_for_anonymous_sessions(src):
    assert "(authToken || doc.persisted === false) && (" in src


def test_deleting_a_document_also_clears_state_that_quotes_it(src):
    local = _between(src, "const removeDocLocally = useCallback", "const handleDeleteDoc = useCallback")
    assert "setCompareQueue(prev => prev.filter(item => item.docId !== doc.id))" in local
    assert "setSelectedContext(" in local


# ---- search / reopen: header token, stale responses ignored ------------------


def test_history_calls_send_the_token_in_a_header_not_the_url(src):
    # Scoped to the three document-history URLs this PR owns: older endpoints
    # (e.g. /me, the admin views) still use ?token= and are out of scope here.
    assert "/me/documents/search?q=${encodeURIComponent(docSearchQuery)}&token" not in src
    assert "/content?token=" not in src
    assert "/me/documents/${doc.id}?token" not in src
    search = _between(src, "const t = setTimeout(() => {\n      setDocSearchLoading(true);", "}, 250)")
    assert "/me/documents/search?q=" in search and "Authorization: `Bearer" in search
    reopen = _between(src, "const openSearchedDocument = useCallback", "const newDoc = {")
    assert "/content`, {" in reopen and "Authorization: `Bearer" in reopen


def test_search_ignores_responses_for_superseded_queries(src):
    effect = _between(src, "let cancelled = false;", "[docSearchQuery, docSearchOpen, authToken, currentUser]")
    assert "cancelled = true" in effect
    assert effect.count("if (!cancelled)") >= 3, "results, error and loading updates must all be guarded"


def test_reopened_documents_are_labelled_as_extracted_text(src):
    assert "fromHistory: true" in src
    assert "extracted text" in src and "Extracted text" in src


# ---- combined summary: consider every doc, invent nothing --------------------


def test_summary_prompt_no_longer_forces_an_entry_from_every_document(src):
    prompt = _between(src, "const scopeInstruction = isAllScope", "const prompt = `You are a legal document analyst")
    # The old wording demanded "at least one entry from each" document in every section while also
    # allowing empty sections -- contradictory, and it invited the model to invent entries.
    assert "at least one entry from each" not in prompt
    # Recall now comes from asking for everything each document STATES, one document at a time...
    assert "ONE AT A TIME" in prompt
    assert "every party, date, obligation and notable clause that its text explicitly states" in prompt
    assert "do not drop stated items" in prompt
    # ...while inventing stays forbidden and an empty contribution is explicitly fine.
    assert "never infer or invent" in prompt
    assert "adds nothing there" in prompt
    assert "file name in parentheses" in prompt  # every item is attributable
    assert "only the opening portion of each document is provided" in prompt


def test_summary_tells_the_user_how_much_of_each_document_was_read(src):
    assert "function combinedPerDocLimit" in src
    assert src.count("combinedPerDocLimit(docs.length)") >= 3  # chat, tool and summary agree
    assert "built from roughly the first" in src


# ---- context priority ---------------------------------------------------------


def test_an_explicit_selection_beats_the_combined_document_bundle(src):
    assert "const combinedSelections = isAllScope && !askedContext?.text" in src


# ---- "Sources" is analysis, not retrieval ------------------------------------


def test_sources_action_is_labelled_as_ai_analysis(src):
    assert 'title="Analyze related legal issues (AI analysis — not verified sources)"' in src
    handler = _between(src, "const handleSearchSelectionSources = useCallback", "[sendMessage]")
    assert "not a lookup of verified sources" in handler
    assert "if you are not certain a case" in handler
    # The caveat must not depend on the model remembering to write it: the action flags its answer
    # and the UI renders a fixed note under it.
    assert "{ unverified: true }" in handler
    assert "unverified: !!flags?.unverified" in src
    assert "AI analysis only — any cases or citations above are not verified against a source." in src


# ---- compare modal ------------------------------------------------------------


def test_explain_model_comes_from_config_and_diffs_are_memoized(src):
    assert 'const EXPLAIN_MODEL = (TASK_MODES.find(t => t.id === "summarize") || {}).model' in src
    explain = _between(src, "const explainDiffPair = useCallback", "[authToken]")
    assert "model: EXPLAIN_MODEL" in explain and "qwen3" not in explain
    assert "const diffs = useMemo(" in src
    assert "computeWordDiff(" not in _between(src, "{others.map((item, idx) => {", "const explanation = explanations[item.id]"), (
        "the diff must not be recomputed inline on every render"
    )
