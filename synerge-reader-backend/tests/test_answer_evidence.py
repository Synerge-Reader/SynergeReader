"""Unit contracts for the E2 answer-evidence planner.

Route-independent by construction: ``answer_evidence.py`` imports no route,
opens no connection, and contacts no model, and both of the planner's
dependencies (document loading and retrieval) are injected here as in-memory
fakes. No database, Ollama, network, subprocess, or filesystem write occurs.

What these prove: the binding priority (explicit selection, then a complete
short authorized document, then authorized hybrid retrieval), that
authorization is a hard boundary rather than a filter applied afterwards, that
a truncated document is never labelled complete, and that budgeting and
deduplication are deterministic.

What these do NOT prove: anything about SQL, pgvector, the embedding provider,
citation identifiers, or the wire format. Those belong to
test_hybrid_retrieval.py, test_citation_generation.py, and
test_main_citation_wiring.py respectively.
"""

import os
import sys

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
    INSUFFICIENT_EVIDENCE_MESSAGE,
    SelectedTextInput,
    apply_budget,
)


DOC_A = AuthorizedDocument(document_id=1, filename="agreement.pdf", title="Agreement")
DOC_B = AuthorizedDocument(document_id=2, filename="statute.pdf")
FOREIGN = AuthorizedDocument(document_id=99, filename="someone-elses.pdf")


def _scope(*documents):
    return AuthorizedScope(documents=tuple(documents), established=True, user_id="user-1", anonymous=False)


def _chunk(document_id, chunk_index, text, filename="agreement.pdf", **kwargs):
    return EvidenceItem(
        text=text,
        source_type="document_chunk",
        document_id=document_id,
        filename=filename,
        chunk_id=f"{document_id}-{chunk_index}",
        chunk_index=chunk_index,
        **kwargs,
    )


class _Recorder:
    """Records what the planner asked for, and returns canned results."""

    def __init__(self, texts=None, chunks=None):
        self.texts = texts or {}
        self.chunks = chunks or []
        self.loaded = []
        self.retrieved_with = []

    def load(self, document_id):
        self.loaded.append(document_id)
        return self.texts.get(document_id)

    def retrieve(self, question, document_ids, top_k):
        self.retrieved_with.append((question, tuple(document_ids), top_k))
        return list(self.chunks)


def _planner(recorder, limits=None, propagate=()):
    return AnswerEvidencePlanner(
        load_document_text=recorder.load,
        retrieve=recorder.retrieve,
        limits=limits or EvidenceLimits(),
        propagate_exceptions=propagate,
    )


# --- 1/2: explicit selection wins ------------------------------------------


def test_selected_text_wins_over_complete_document_and_retrieval():
    recorder = _Recorder(texts={1: "the whole short document"}, chunks=[_chunk(1, 0, "a chunk")])
    request = EvidenceRequest(
        question="What is the term?",
        selections=(SelectedTextInput(text="The term is three years.", document_id=1),),
        requested_document_ids=(1,),
    )

    result = _planner(recorder).plan(request, _scope(DOC_A))

    assert result.mode is EvidenceMode.SELECTED_TEXT
    assert [item.text for item in result.bundle.items] == ["The term is three years."]
    assert recorder.loaded == [], "a selection must not trigger a document load"
    assert recorder.retrieved_with == [], "a selection must not trigger retrieval"


def test_selected_text_wins_with_multiple_documents_in_scope():
    recorder = _Recorder(chunks=[_chunk(1, 0, "a chunk"), _chunk(2, 0, "another")])
    request = EvidenceRequest(
        question="Compare the clauses",
        selections=(
            SelectedTextInput(text="Clause 4 governs renewal.", document_id=1),
            SelectedTextInput(text="Section 12 governs notice.", document_id=2),
        ),
    )

    result = _planner(recorder).plan(request, _scope(DOC_A, DOC_B))

    assert result.mode is EvidenceMode.SELECTED_TEXT
    assert len(result.bundle.items) == 2
    assert recorder.retrieved_with == []
    assert [item.filename for item in result.bundle.items] == ["agreement.pdf", "statute.pdf"]


def test_loose_selected_text_without_structured_selections_is_still_priority_one():
    recorder = _Recorder(texts={1: "whole doc"}, chunks=[_chunk(1, 0, "chunk")])
    request = EvidenceRequest(question="q", selected_text="  A highlighted sentence.  ")

    result = _planner(recorder).plan(request, _scope(DOC_A))

    assert result.mode is EvidenceMode.SELECTED_TEXT
    assert result.bundle.items[0].text == "A highlighted sentence."
    assert result.bundle.items[0].filename is None
    assert recorder.loaded == []


# --- 3/4: complete short document, and the truncation honesty rule ----------


def test_single_short_authorized_document_is_used_whole_without_retrieval():
    recorder = _Recorder(texts={1: "A short but complete agreement."}, chunks=[_chunk(1, 0, "chunk")])
    request = EvidenceRequest(question="What does it say?", requested_document_ids=(1,))

    result = _planner(recorder).plan(request, _scope(DOC_A))

    assert result.mode is EvidenceMode.COMPLETE_DOCUMENT
    assert result.bundle.items[0].text == "A short but complete agreement."
    assert result.bundle.items[0].truncated is False
    assert result.bundle.truncated is False
    assert recorder.retrieved_with == [], "a complete short document must not also retrieve"


def test_oversized_document_is_never_labelled_complete():
    limits = EvidenceLimits(complete_document_chars=100)
    recorder = _Recorder(
        texts={1: "x" * 5000},
        chunks=[_chunk(1, 3, "the relevant passage")],
    )
    request = EvidenceRequest(question="What does it say?", requested_document_ids=(1,))

    result = _planner(recorder, limits).plan(request, _scope(DOC_A))

    assert result.mode is EvidenceMode.HYBRID_RETRIEVAL, (
        "a document above the complete-document limit must fall through to "
        "retrieval, never be cut and presented as the complete document"
    )
    assert EvidenceWarning.DOCUMENT_TOO_LONG_FOR_COMPLETE.value in result.warnings
    assert all(item.source_type != "complete_document" for item in result.bundle.items)


def test_empty_document_body_does_not_become_complete_evidence():
    recorder = _Recorder(texts={1: "   "}, chunks=[])
    request = EvidenceRequest(question="q", requested_document_ids=(1,))

    result = _planner(recorder).plan(request, _scope(DOC_A))

    assert result.mode is EvidenceMode.NONE
    assert result.insufficient


# --- 5: multiple documents use hybrid retrieval -----------------------------


def test_multiple_documents_use_hybrid_retrieval_over_the_authorized_ids():
    recorder = _Recorder(
        texts={1: "short", 2: "short"},
        chunks=[_chunk(1, 0, "from A"), _chunk(2, 0, "from B", filename="statute.pdf")],
    )
    request = EvidenceRequest(question="Compare", requested_document_ids=(1, 2))

    result = _planner(recorder).plan(request, _scope(DOC_A, DOC_B))

    assert result.mode is EvidenceMode.HYBRID_RETRIEVAL
    assert recorder.loaded == [], "multi-document scope must not load whole documents"
    question, document_ids, top_k = recorder.retrieved_with[0]
    assert question == "Compare"
    assert sorted(document_ids) == [1, 2]
    assert top_k > 0


# --- 6: authorization is a hard boundary ------------------------------------


def test_unauthorized_document_ids_never_enter_the_bundle():
    recorder = _Recorder(texts={99: "someone else's contract"}, chunks=[])
    request = EvidenceRequest(question="q", requested_document_ids=(99,))

    result = _planner(recorder).plan(request, _scope(DOC_A))

    assert result.mode is EvidenceMode.NONE
    assert recorder.loaded == [], "an unauthorized id must never reach the loader"
    assert EvidenceWarning.DOCUMENT_NOT_AUTHORIZED.value in result.warnings
    assert result.message == INSUFFICIENT_EVIDENCE_MESSAGE
    assert "someone" not in (result.message or "")


def test_retrieved_chunk_outside_the_authorized_scope_is_dropped():
    # Defence in depth: even if a retrieval implementation returned a foreign
    # chunk, the planner refuses it rather than citing it.
    recorder = _Recorder(chunks=[_chunk(99, 0, "foreign text", filename="someone-elses.pdf"),
                                 _chunk(1, 0, "mine")])
    request = EvidenceRequest(question="q", requested_document_ids=(1, 2))

    result = _planner(recorder).plan(request, _scope(DOC_A, DOC_B))

    assert [item.document_id for item in result.bundle.items] == [1]
    assert EvidenceWarning.DOCUMENT_NOT_AUTHORIZED.value in result.warnings


def test_selection_claiming_an_unauthorized_document_loses_its_attribution():
    recorder = _Recorder()
    request = EvidenceRequest(
        question="q",
        selections=(SelectedTextInput(text="highlighted", document_id=99, filename="someone-elses.pdf"),),
    )

    result = _planner(recorder).plan(request, _scope(DOC_A))

    item = result.bundle.items[0]
    assert item.text == "highlighted", "the user's own highlight is still usable"
    assert item.document_id is None and item.filename is None, (
        "a client-claimed document id cannot attribute a selection to a "
        "document this caller is not authorized for"
    )
    assert EvidenceWarning.SELECTION_SCOPE_UNVERIFIED.value in result.warnings


def test_unresolved_authorization_fails_closed():
    recorder = _Recorder(texts={1: "text"}, chunks=[_chunk(1, 0, "chunk")])
    request = EvidenceRequest(
        question="q",
        selected_text="a highlight",
        requested_document_ids=(1,),
    )

    result = _planner(recorder).plan(request, AuthorizedScope.unresolved())

    assert result.mode is EvidenceMode.NONE
    assert result.insufficient
    assert recorder.loaded == [] and recorder.retrieved_with == []


# --- 7: empty evidence is safe ---------------------------------------------


def test_empty_evidence_produces_a_safe_insufficient_result():
    recorder = _Recorder(chunks=[])
    result = _planner(recorder).plan(EvidenceRequest(question="q"), _scope())

    assert result.mode is EvidenceMode.NONE
    assert result.insufficient
    assert result.message == INSUFFICIENT_EVIDENCE_MESSAGE
    assert result.bundle.items == ()
    payload = result.to_dict()
    assert payload["insufficient"] is True and payload["bundle"]["items"] == []


def test_retrieval_failure_degrades_to_insufficient_without_leaking_detail():
    class _Boom(_Recorder):
        def retrieve(self, question, document_ids, top_k):
            raise RuntimeError("connection to 10.0.0.5 failed: password=hunter2")

    recorder = _Boom(texts={1: "x" * 999999}, chunks=[])
    limits = EvidenceLimits(complete_document_chars=10)
    result = _planner(recorder, limits).plan(
        EvidenceRequest(question="q", requested_document_ids=(1, 2)), _scope(DOC_A, DOC_B)
    )

    assert result.mode is EvidenceMode.NONE
    assert EvidenceWarning.RETRIEVAL_UNAVAILABLE.value in result.warnings
    assert "hunter2" not in str(result.to_dict())


def test_declared_exception_types_propagate_instead_of_degrading():
    class _Outage(Exception):
        pass

    class _Boom(_Recorder):
        def retrieve(self, question, document_ids, top_k):
            raise _Outage("embedding provider down")

    recorder = _Boom()
    planner = _planner(recorder, propagate=(_Outage,))

    with pytest.raises(_Outage):
        planner.plan(EvidenceRequest(question="q", requested_document_ids=(1, 2)), _scope(DOC_A, DOC_B))


# --- 8/10: deterministic, bounded budgeting --------------------------------


def test_context_budgeting_is_deterministic_and_bounded():
    limits = EvidenceLimits(total_evidence_chars=100, max_evidence_items=10)
    items = [_chunk(1, index, "x" * 40) for index in range(5)]

    first, truncated_first, warnings_first = apply_budget(items, limits)
    second, truncated_second, warnings_second = apply_budget(items, limits)

    assert [item.text for item in first] == [item.text for item in second]
    assert warnings_first == warnings_second
    assert truncated_first is truncated_second is True
    assert sum(len(item.text) for item in first) == 100
    assert first[-1].truncated is True
    assert EvidenceWarning.EVIDENCE_TRUNCATED.value in warnings_first


def test_item_count_limit_is_enforced():
    limits = EvidenceLimits(total_evidence_chars=10_000, max_evidence_items=3)
    items = [_chunk(1, index, "short") for index in range(9)]

    kept, _, warnings = apply_budget(items, limits)

    assert len(kept) == 3
    assert EvidenceWarning.EVIDENCE_ITEM_LIMIT_REACHED.value in warnings


def test_multi_document_budgeting_is_bounded_and_ordered():
    limits = EvidenceLimits(total_evidence_chars=90, max_evidence_items=8)
    recorder = _Recorder(
        chunks=[
            _chunk(1, 0, "a" * 50),
            _chunk(2, 0, "b" * 50, filename="statute.pdf"),
            _chunk(1, 1, "c" * 50),
        ]
    )
    request = EvidenceRequest(question="q", requested_document_ids=(1, 2))

    result = _planner(recorder, limits).plan(request, _scope(DOC_A, DOC_B))

    assert result.bundle.total_chars <= 90
    assert [item.document_id for item in result.bundle.items] == [1, 2], (
        "retrieval order must be preserved and the budget applied in that order"
    )
    assert result.bundle.truncated is True


# --- 9: deduplication -------------------------------------------------------


def test_duplicate_chunks_are_deduplicated_and_keep_the_best_scores():
    recorder = _Recorder(
        chunks=[
            _chunk(1, 2, "the same passage", semantic_score=0.4, combined_score=0.1),
            _chunk(1, 2, "the same passage", lexical_score=0.9, combined_score=0.3),
            _chunk(1, 3, "a different passage"),
        ]
    )
    request = EvidenceRequest(question="q", requested_document_ids=(1, 2))

    result = _planner(recorder).plan(request, _scope(DOC_A, DOC_B))

    assert len(result.bundle.items) == 2
    merged = result.bundle.items[0]
    assert merged.semantic_score == 0.4 and merged.lexical_score == 0.9
    assert merged.combined_score == 0.3


def test_identical_selection_text_is_deduplicated():
    recorder = _Recorder()
    request = EvidenceRequest(
        question="q",
        selections=(
            SelectedTextInput(text="The term is three years."),
            SelectedTextInput(text="the   term is THREE years."),
        ),
    )

    result = _planner(recorder).plan(request, _scope(DOC_A))

    assert len(result.bundle.items) == 1, (
        "the same passage highlighted twice is one piece of evidence"
    )


def test_bundle_reports_its_document_ids_and_contract_version():
    recorder = _Recorder(chunks=[_chunk(1, 0, "a"), _chunk(2, 0, "b", filename="statute.pdf")])
    result = _planner(recorder).plan(
        EvidenceRequest(question="q", requested_document_ids=(1, 2)), _scope(DOC_A, DOC_B)
    )

    assert result.bundle.document_ids == (1, 2)
    assert result.bundle.to_dict()["contract_version"].startswith("e2.evidence")


def test_authorized_document_requires_a_filename():
    # A citation must never render as a bare database id, so an unnamed
    # document cannot become citable evidence in the first place.
    with pytest.raises(ValueError):
        AuthorizedDocument(document_id=5, filename="   ")
