"""Unit contracts for E2 authorization-scoped hybrid retrieval.

``document_retrieval.py`` builds SQL strings and fuses ranked lists; it opens
no connection and executes nothing. These tests inspect the generated SQL as
text and exercise the fusion function in memory, so no database, Ollama,
network, subprocess, or filesystem access occurs.

What these prove: both rankers are scoped to explicit authorized document ids
with no global fallback, a filename can never widen that scope, the question is
bound as a parameter rather than interpolated, fusion is deterministic and
stably tie-broken, and every locator field a citation needs survives.

What these do NOT prove: that PostgreSQL accepts or plans these statements, or
anything about pgvector behaviour at runtime. That is for controlled execution.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from document_retrieval import (
    RRF_K,
    build_authorized_lexical_query,
    build_authorized_semantic_query,
    build_relevant_chunks_query,
    lexical_chunk_from_row,
    reciprocal_rank_fusion,
    retrieved_chunk_from_row,
)


EMBEDDING = [0.1, 0.2, 0.3]


def _normalize(sql):
    return " ".join(sql.split())


def _row(chunk_id, document_id, chunk_index, score, filename="agreement.pdf"):
    return (
        chunk_id,
        document_id,
        chunk_index,
        4,
        5,
        {"type": "pdf_pages", "pages": [4, 5]},
        filename,
        f"chunk text {chunk_id}",
        score,
    )


# --- 21: retrieval is restricted to authorized document ids ----------------


def test_semantic_retrieval_is_restricted_to_authorized_document_ids():
    sql, params = build_authorized_semantic_query(EMBEDDING, 5, [1, 2])

    assert "AND d.id = ANY(%s)" in sql
    assert params[1] == [1, 2]
    assert params[3] == 5
    assert "d.filename = ANY" not in sql


def test_lexical_retrieval_is_restricted_to_authorized_document_ids():
    sql, params = build_authorized_lexical_query("renewal term", 5, [7])

    assert "WHERE d.id = ANY(%s)" in _normalize(sql)
    assert params[1] == [7]
    assert params[3] == 5


@pytest.mark.parametrize(
    "builder",
    [
        lambda ids: build_authorized_semantic_query(EMBEDDING, 5, ids),
        lambda ids: build_authorized_lexical_query("q", 5, ids),
    ],
    ids=["semantic", "lexical"],
)
@pytest.mark.parametrize("empty", [[], (), None], ids=["list", "tuple", "none"])
def test_authorized_retrieval_refuses_an_empty_scope(builder, empty):
    # There is no unscoped path: an empty authorized scope is an error, not an
    # invitation to search every document in the database.
    with pytest.raises(ValueError):
        builder(empty)


# --- 22: filenames cannot bypass authorization -----------------------------


def test_client_filenames_never_appear_in_the_authorized_queries():
    semantic_sql, semantic_params = build_authorized_semantic_query(EMBEDDING, 5, [1])
    lexical_sql, lexical_params = build_authorized_lexical_query("agreement.pdf", 5, [1])

    for sql in (semantic_sql, lexical_sql):
        assert "filename = ANY" not in sql, (
            "a filename must never be a scoping predicate; only document ids are "
            "checkable against ownership"
        )
        assert "WHERE d.filename" not in sql
    # The filename-looking question travels as a bound parameter, never spliced
    # into SQL and never used to widen the scope.
    assert "agreement.pdf" not in lexical_sql
    assert lexical_params[0] == "agreement.pdf"
    assert lexical_params[1] == [1]
    assert semantic_params[1] == [1]


def test_lexical_question_is_bound_not_interpolated():
    hostile = "'; DROP TABLE documents; --"
    sql, params = build_authorized_lexical_query(hostile, 3, [1])

    assert hostile not in sql
    assert params[0] == hostile and params[2] == hostile
    assert "plainto_tsquery" in sql, (
        "plainto_tsquery treats the question as data, not tsquery syntax"
    )


def test_legacy_filename_scoped_builder_is_unchanged():
    # The pre-E2 builder is deliberately untouched, so existing callers and
    # their tests keep their exact behaviour.
    sql, params = build_relevant_chunks_query(EMBEDDING, 4, ["a.pdf"])
    assert "AND d.filename = ANY(%s)" in sql
    assert params[1] == ["a.pdf"]


# --- 23: deterministic fusion ----------------------------------------------


def test_semantic_and_lexical_rankings_combine_deterministically():
    semantic = [retrieved_chunk_from_row(_row(10, 1, 0, 0.90)),
                retrieved_chunk_from_row(_row(11, 1, 1, 0.80))]
    lexical = [lexical_chunk_from_row(_row(11, 1, 1, 0.55)),
               lexical_chunk_from_row(_row(12, 2, 0, 0.40, filename="statute.pdf"))]

    fused = reciprocal_rank_fusion(semantic, lexical)
    again = reciprocal_rank_fusion(semantic, lexical)

    assert [chunk["chunk_id"] for chunk in fused] == [chunk["chunk_id"] for chunk in again]
    assert fused[0]["chunk_id"] == 11, (
        "a chunk both rankers found must outrank one only a single ranker found"
    )
    assert fused[0]["fusion_score"] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert {chunk["chunk_id"] for chunk in fused} == {10, 11, 12}


def test_fusion_preserves_both_scores_on_a_doubly_ranked_chunk():
    semantic = [retrieved_chunk_from_row(_row(11, 1, 1, 0.80))]
    lexical = [lexical_chunk_from_row(_row(11, 1, 1, 0.55))]

    fused = reciprocal_rank_fusion(semantic, lexical)

    assert fused[0]["similarity"] == pytest.approx(0.80)
    assert fused[0]["lexical_rank"] == pytest.approx(0.55)
    assert fused[0]["semantic_rank"] == 1
    assert fused[0]["lexical_rank_position"] == 1


def test_fusion_ties_break_by_document_and_chunk_identity_not_input_order():
    first = [retrieved_chunk_from_row(_row(30, 2, 5, 0.5, filename="statute.pdf"))]
    second = [retrieved_chunk_from_row(_row(20, 1, 9, 0.5))]

    forward = reciprocal_rank_fusion(first + second, [])
    backward = reciprocal_rank_fusion(second + first, [])

    assert [chunk["chunk_id"] for chunk in forward] == [30, 20], (
        "rank 1 and rank 2 differ, so input order still decides here"
    )
    assert [chunk["chunk_id"] for chunk in backward] == [20, 30]

    # Same rank in different lists: the fused scores are equal, so identity
    # alone must decide, identically in both directions.
    tied_forward = reciprocal_rank_fusion(first, second)
    tied_backward = reciprocal_rank_fusion(second, first)
    assert [chunk["chunk_id"] for chunk in tied_forward] == [20, 30]
    assert [chunk["chunk_id"] for chunk in tied_backward] == [20, 30]


def test_fusion_handles_a_missing_ranker_and_respects_top_k():
    semantic = [retrieved_chunk_from_row(_row(index, 1, index, 0.5)) for index in range(5)]

    fused = reciprocal_rank_fusion(semantic, [], top_k=3)

    assert len(fused) == 3
    assert [chunk["chunk_id"] for chunk in fused] == [0, 1, 2]


def test_fusion_tolerates_legacy_rows_without_a_chunk_index():
    legacy = retrieved_chunk_from_row((5, 1, None, None, None, None, "a.pdf", "text", 0.5))
    fused = reciprocal_rank_fusion([legacy], [])
    assert fused[0]["chunk_index"] is None


# --- 24: metadata survives retrieval ---------------------------------------


def test_retrieval_retains_filename_page_chunk_and_locator_metadata():
    chunk = retrieved_chunk_from_row(_row(10, 1, 3, 0.9))

    assert chunk["document_name"] == "agreement.pdf"
    assert chunk["document_id"] == 1
    assert chunk["chunk_id"] == 10
    assert chunk["chunk_index"] == 3
    assert chunk["page_start"] == 4 and chunk["page_end"] == 5
    assert chunk["locator"] == {"type": "pdf_pages", "pages": [4, 5]}
    assert chunk["similarity"] == pytest.approx(0.9)


def test_lexical_rows_keep_the_same_metadata_with_a_lexical_score():
    chunk = lexical_chunk_from_row(_row(10, 1, 3, 0.42))

    assert chunk["document_name"] == "agreement.pdf"
    assert chunk["page_start"] == 4 and chunk["page_end"] == 5
    assert chunk["locator"] == {"type": "pdf_pages", "pages": [4, 5]}
    assert chunk["lexical_rank"] == pytest.approx(0.42)
    assert "similarity" not in chunk, "a lexical row carries no semantic similarity"


def test_fused_chunks_still_carry_every_citation_field():
    semantic = [retrieved_chunk_from_row(_row(10, 1, 3, 0.9))]
    lexical = [lexical_chunk_from_row(_row(10, 1, 3, 0.4))]

    fused = reciprocal_rank_fusion(semantic, lexical)[0]

    for key in ("document_id", "document_name", "chunk_id", "chunk_index",
                "page_start", "page_end", "locator", "text"):
        assert key in fused, f"fusion dropped {key!r}, which a citation needs"


def test_both_authorized_queries_select_every_citation_column():
    semantic_sql, _ = build_authorized_semantic_query(EMBEDDING, 5, [1])
    lexical_sql, _ = build_authorized_lexical_query("q", 5, [1])

    for sql in (semantic_sql, lexical_sql):
        normalized = _normalize(sql)
        for column in ("dc.id", "dc.document_id", "dc.chunk_index", "dc.page_start",
                       "dc.page_end", "dc.locator_json", "d.filename", "dc.chunk_text"):
            assert column in normalized, f"{column} must be selected for citations"


def test_authorized_queries_order_deterministically():
    semantic_sql, _ = build_authorized_semantic_query(EMBEDDING, 5, [1])
    lexical_sql, _ = build_authorized_lexical_query("q", 5, [1])

    assert "dc.document_id, dc.chunk_index, dc.id" in _normalize(semantic_sql)
    assert "dc.document_id, dc.chunk_index, dc.id" in _normalize(lexical_sql)
