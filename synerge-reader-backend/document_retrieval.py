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


# ---------------------------------------------------------------------------
# E2 hybrid retrieval.
#
# The two builders below are the authorization-scoped counterparts of
# build_relevant_chunks_query above: they filter on documents.id, which is the
# only thing that can be checked against what a user owns. The original
# filename-scoped builder is left exactly as it was, because filenames are not
# authorization and callers that still use it are unchanged by this milestone.
#
# No schema migration is involved. Lexical ranking computes its tsvector on the
# fly with to_tsvector(); that is slower than a stored, indexed column, but it
# needs no ALTER TABLE, no reindex, and no backfill for this milestone.
# ---------------------------------------------------------------------------

RRF_K = 60


def build_authorized_semantic_query(
    question_embedding: Any,
    top_k: int,
    document_ids: Sequence[Any],
) -> tuple[str, tuple]:
    """Semantic ranking restricted to an explicit set of document ids.

    ``document_ids`` is required and must be non-empty: an unscoped semantic
    search over every document in the database is exactly what this milestone
    removes.
    """
    if not document_ids:
        raise ValueError("authorized semantic retrieval requires document ids")

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
        AND d.id = ANY(%s)
        ORDER BY dc.embedding <=> %s::vector, dc.document_id, dc.chunk_index, dc.id
        LIMIT %s
    """
    params = (
        question_embedding,
        list(document_ids),
        question_embedding,
        top_k,
    )
    return query, params


def build_authorized_lexical_query(
    question: str,
    top_k: int,
    document_ids: Sequence[Any],
) -> tuple[str, tuple]:
    """Lexical (full-text) ranking restricted to the same document ids.

    Uses plainto_tsquery so the user's question is treated as data, never as
    tsquery syntax, and ts_rank over an on-the-fly tsvector so no stored column
    or migration is required.
    """
    if not document_ids:
        raise ValueError("authorized lexical retrieval requires document ids")

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
            ts_rank(
                to_tsvector('english', dc.chunk_text),
                plainto_tsquery('english', %s)
            ) AS lexical_rank
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        WHERE d.id = ANY(%s)
        AND to_tsvector('english', dc.chunk_text) @@ plainto_tsquery('english', %s)
        ORDER BY lexical_rank DESC, dc.document_id, dc.chunk_index, dc.id
        LIMIT %s
    """
    params = (question, list(document_ids), question, top_k)
    return query, params


def lexical_chunk_from_row(row: tuple) -> dict:
    """Maps a lexical retrieval row. Same column order as the semantic query,
    with the trailing score being a lexical rank rather than a similarity."""
    chunk = retrieved_chunk_from_row(row)
    score = chunk.pop("similarity")
    chunk["source_type"] = "document_chunk"
    chunk["lexical_rank"] = float(score)
    return chunk


def _fusion_identity(chunk: dict) -> tuple:
    """Stable identity for one retrieved chunk, used for fusion and ties."""
    return (chunk.get("document_id"), chunk.get("chunk_id"))


def _tie_break_key(chunk: dict) -> tuple:
    """Deterministic ordering for chunks with identical fused scores.

    Sorting by document id, then chunk index, then chunk id means two runs over
    the same data always produce the same order, whatever order the database
    returned rows in. ``-1`` stands in for a missing chunk index so legacy rows
    sort before indexed ones instead of raising on a None comparison.
    """
    document_id = chunk.get("document_id")
    chunk_index = chunk.get("chunk_index")
    chunk_id = chunk.get("chunk_id")
    return (
        str(document_id),
        chunk_index if isinstance(chunk_index, int) else -1,
        str(chunk_id),
    )


def reciprocal_rank_fusion(
    semantic_chunks: Sequence[dict],
    lexical_chunks: Sequence[dict],
    top_k: Optional[int] = None,
    k: int = RRF_K,
) -> list[dict]:
    """Deterministic reciprocal-rank fusion of two ranked chunk lists.

    Each list contributes 1/(k + rank) for the chunks it ranks, so a chunk
    found by both rankers outranks one found by either alone, without either
    ranker's raw score scale leaking into the comparison. Ranks are 1-based and
    taken from the input order, which is the database's ORDER BY. Ties are
    broken by document/chunk identity, never by input order, so the result is
    reproducible.

    Both scores are preserved on the merged chunk, so a citation can still
    report how its evidence was found.
    """
    merged: dict[tuple, dict] = {}

    for rank, chunk in enumerate(semantic_chunks, start=1):
        identity = _fusion_identity(chunk)
        entry = merged.setdefault(identity, dict(chunk))
        entry.setdefault("similarity", chunk.get("similarity"))
        entry["semantic_rank"] = rank
        entry["fusion_score"] = entry.get("fusion_score", 0.0) + 1.0 / (k + rank)

    for rank, chunk in enumerate(lexical_chunks, start=1):
        identity = _fusion_identity(chunk)
        if identity in merged:
            entry = merged[identity]
        else:
            entry = dict(chunk)
            merged[identity] = entry
        entry["lexical_rank_position"] = rank
        if "lexical_rank" not in entry and "lexical_rank" in chunk:
            entry["lexical_rank"] = chunk["lexical_rank"]
        entry["fusion_score"] = entry.get("fusion_score", 0.0) + 1.0 / (k + rank)

    fused = sorted(
        merged.values(),
        key=lambda chunk: (-chunk.get("fusion_score", 0.0), _tie_break_key(chunk)),
    )
    if top_k is not None:
        fused = fused[:top_k]
    return fused
