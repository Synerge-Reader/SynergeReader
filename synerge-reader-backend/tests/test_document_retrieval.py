import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from document_retrieval import build_relevant_chunks_query, retrieved_chunk_from_row


def test_query_without_document_names_is_global():
    embedding = [0.1, 0.2]

    query, params = build_relevant_chunks_query(
        embedding,
        top_k=3,
        document_names=None,
    )

    normalized = " ".join(query.split())

    assert "JOIN documents d ON d.id = dc.document_id" in normalized
    assert "d.filename = ANY(%s)" not in normalized
    assert params == (embedding, embedding, 3)
    assert query.count("%s") == len(params)


def test_empty_document_names_preserves_existing_unscoped_semantics():
    embedding = [0.1, 0.2]

    query, params = build_relevant_chunks_query(
        embedding,
        top_k=3,
        document_names=[],
    )

    normalized = " ".join(query.split())

    assert "d.filename = ANY(%s)" not in normalized
    assert params == (embedding, embedding, 3)
    assert query.count("%s") == len(params)


def test_scoped_query_uses_expected_parameter_order():
    embedding = [0.1, 0.2]
    names = ["a.pdf", "b.pdf"]

    query, params = build_relevant_chunks_query(
        embedding,
        top_k=5,
        document_names=names,
    )

    normalized = " ".join(query.split())

    assert normalized.count("SELECT") == 1
    assert normalized.count("JOIN documents") == 1
    assert "d.filename = ANY(%s)" in normalized
    assert params == (embedding, names, embedding, 5)
    assert query.count("%s") == len(params)


def test_retrieved_chunk_from_row_maps_all_fields():
    row = (
        101,              # chunk_id
        42,               # document_id
        3,                # chunk_index
        1,                # page_start
        3,                # page_end
        {"locator_type": "pdf_pages", "page_numbers": [1, 3]},  # locator_json
        "Agreement.pdf",  # filename
        "chunk text here",  # chunk_text
        0.87,             # similarity
    )

    result = retrieved_chunk_from_row(row)

    assert result == {
        "source_type": "document_chunk",
        "chunk_id": 101,
        "document_id": 42,
        "document_name": "Agreement.pdf",
        "chunk_index": 3,
        "text": "chunk text here",
        "similarity": 0.87,
        "page_start": 1,
        "page_end": 3,
        "locator": {"locator_type": "pdf_pages", "page_numbers": [1, 3]},
    }
    assert isinstance(result["similarity"], float)


def test_retrieved_chunk_from_row_preserves_none_for_legacy_rows():
    row = (
        202, 7, 0,
        None, None, None,   # legacy row — no locator metadata
        "OldFile.txt", "legacy chunk", 0.5,
    )

    result = retrieved_chunk_from_row(row)

    assert result["page_start"] is None
    assert result["page_end"] is None
    assert result["locator"] is None
    # These two keys are what build_context() actually consumes — must
    # always be present regardless of locator availability.
    assert "text" in result
    assert "similarity" in result
