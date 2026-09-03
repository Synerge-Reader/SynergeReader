from typing import Any, Optional, Sequence


def build_relevant_chunks_query(
    question_embedding: Any,
    top_k: int,
    document_names: Optional[Sequence[str]] = None,
) -> tuple[str, tuple]:
    query = """
        SELECT
            dc.id,
            dc.document_id,
            dc.chunk_index,
            dc.page_start,
            dc.page_end,
            dc.locator_json,
            d.filename,
            dc.chunk_text,
            1 - (dc.embedding <=> %s::vector) AS similarity
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        WHERE dc.embedding IS NOT NULL
    """
    params = [question_embedding]

    # Preserve current truthy-check semantics: None and [] are unscoped.
    if document_names:
        query += "\nAND d.filename = ANY(%s)"
        params.append(list(document_names))

    query += """
        ORDER BY dc.embedding <=> %s::vector
        LIMIT %s
    """
    params.extend([question_embedding, top_k])

    return query, tuple(params)


def retrieved_chunk_from_row(row: tuple) -> dict:
    """Maps a raw document_chunks retrieval row into the evidence dict
    consumed by build_context(). Row order must exactly match the
    SELECT in build_relevant_chunks_query():
    id, document_id, chunk_index, page_start, page_end, locator_json,
    filename, chunk_text, similarity."""
    (
        chunk_id, document_id, chunk_index,
        page_start, page_end, locator,
        filename, text, similarity,
    ) = row
    return {
        "source_type": "document_chunk",
        "chunk_id": chunk_id,
        "document_id": document_id,
        "document_name": filename,
        "chunk_index": chunk_index,
        "text": text,
        "similarity": float(similarity),
        "page_start": page_start,
        "page_end": page_end,
        "locator": locator,
    }
