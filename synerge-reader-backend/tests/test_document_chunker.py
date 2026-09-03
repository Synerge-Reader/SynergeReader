import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from document_chunker import DocumentChunk, build_chunk_locator, chunk_document
from document_parser import ParsedDocument, ParsedPage


def legacy_chunk_text(text: str, max_chunk_size: int = 500) -> list:
    """Verbatim copy of the pre-refactor main.py chunk_text() algorithm."""
    if not text.strip():
        return []

    words = text.split()
    chunks = []
    current = []

    for word in words:
        current_size = sum(len(w) for w in current) + len(current) - 1
        if current_size + len(word) + 1 <= max_chunk_size:
            current.append(word)
        else:
            if current:
                chunks.append(" ".join(current))
            current = [word]
    if current:
        chunks.append(" ".join(current))
    return chunks


def _lorem_words(n: int) -> str:
    bank = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit", "sed", "do"]
    return " ".join(bank[i % len(bank)] for i in range(n))


# --- Parity tests: chunk_document must reproduce legacy_chunk_text(document.text) ---


def test_parity_txt_document():
    document = ParsedDocument(
        pages=[ParsedPage(page_number=None, text=_lorem_words(400))],
        document_type="text",
    )
    expected = legacy_chunk_text(document.text, 500)
    actual = [c.text for c in chunk_document(document, 500)]
    assert actual == expected
    assert len(actual) > 1  # sanity: content is long enough to span multiple chunks


def test_parity_docx_document():
    document = ParsedDocument(
        pages=[
            ParsedPage(page_number=None, text=_lorem_words(60)),
            ParsedPage(page_number=None, text=_lorem_words(45)),
            ParsedPage(page_number=None, text=_lorem_words(80)),
        ],
        document_type="docx",
    )
    expected = legacy_chunk_text(document.text, 200)
    actual = [c.text for c in chunk_document(document, 200)]
    assert actual == expected
    assert len(actual) > 1


def test_parity_multi_page_pdf_document():
    document = ParsedDocument(
        pages=[
            ParsedPage(page_number=1, text=_lorem_words(50)),
            ParsedPage(page_number=2, text=_lorem_words(30)),
            ParsedPage(page_number=4, text=_lorem_words(70)),  # page 3 skipped (empty in source)
        ],
        document_type="pdf",
    )
    expected = legacy_chunk_text(document.text, 150)
    actual = [c.text for c in chunk_document(document, 150)]
    assert actual == expected
    assert len(actual) > 1


# --- Independently-reasoned correctness tests ---


def test_empty_document_returns_no_chunks():
    document = ParsedDocument(pages=[], document_type="text")
    assert chunk_document(document) == []


def test_whitespace_only_content_returns_no_chunks():
    document = ParsedDocument(
        pages=[ParsedPage(page_number=None, text="   \n\n   \t  ")],
        document_type="text",
    )
    assert chunk_document(document) == []


def test_one_page_pdf_page_numbers():
    document = ParsedDocument(
        pages=[ParsedPage(page_number=5, text="hello world")],
        document_type="pdf",
    )
    chunks = chunk_document(document)
    assert len(chunks) == 1
    assert chunks[0].page_numbers == (5,)


def test_chunk_spanning_adjacent_pages():
    document = ParsedDocument(
        pages=[
            ParsedPage(page_number=1, text="alpha beta"),
            ParsedPage(page_number=2, text="gamma delta"),
        ],
        document_type="pdf",
    )
    chunks = chunk_document(document)
    assert len(chunks) == 1
    assert chunks[0].page_numbers == (1, 2)


def test_chunk_spanning_gap_preserves_true_page_numbers():
    document = ParsedDocument(
        pages=[
            ParsedPage(page_number=1, text="alpha beta"),
            ParsedPage(page_number=3, text="gamma delta"),  # page 2 was empty, filtered upstream
        ],
        document_type="pdf",
    )
    chunks = chunk_document(document)
    assert len(chunks) == 1
    assert chunks[0].page_numbers == (1, 3)
    assert chunks[0].page_numbers != (1, 2)
    assert chunks[0].page_numbers != (1, 2, 3)


def test_sequential_chunk_indices():
    document = ParsedDocument(
        pages=[ParsedPage(page_number=None, text=_lorem_words(200))],
        document_type="text",
    )
    chunks = chunk_document(document, max_chunk_size=50)
    assert len(chunks) >= 3
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_page_start_end_derive_from_page_numbers_including_gap():
    document = ParsedDocument(
        pages=[
            ParsedPage(page_number=1, text="alpha beta"),
            ParsedPage(page_number=3, text="gamma delta"),
        ],
        document_type="pdf",
    )
    chunk = chunk_document(document)[0]
    assert chunk.page_numbers == (1, 3)
    assert chunk.page_start == 1
    assert chunk.page_end == 3


@pytest.mark.parametrize("document_type", ["text", "docx"])
def test_docx_and_txt_have_empty_page_numbers_and_none_bounds(document_type):
    document = ParsedDocument(
        pages=[ParsedPage(page_number=None, text="hello world")],
        document_type=document_type,
    )
    chunk = chunk_document(document)[0]
    assert chunk.page_numbers == ()
    assert chunk.page_start is None
    assert chunk.page_end is None


def test_page_provenance_resets_after_chunk_flush():
    document = ParsedDocument(
        pages=[
            ParsedPage(page_number=1, text="alpha beta"),
            ParsedPage(page_number=2, text="gamma delta"),
        ],
        document_type="pdf",
    )
    # max_chunk_size=11: "alpha beta" is 10 chars and fits; adding "gamma"
    # would push past 11, forcing a flush; "gamma delta" is exactly 11 chars.
    chunks = chunk_document(document, max_chunk_size=11)
    assert [chunk.text for chunk in chunks] == ["alpha beta", "gamma delta"]
    assert chunks[0].page_numbers == (1,)
    assert chunks[1].page_numbers == (2,)


def test_no_marker_text_in_chunks():
    document = ParsedDocument(
        pages=[
            ParsedPage(page_number=1, text="alpha beta"),
            ParsedPage(page_number=2, text="gamma delta"),
        ],
        document_type="pdf",
    )
    for chunk in chunk_document(document):
        assert "--- Page" not in chunk.text
        assert "[Page" not in chunk.text


def test_source_word_order_preserved():
    document = ParsedDocument(
        pages=[
            ParsedPage(page_number=1, text="one two three"),
            ParsedPage(page_number=2, text="four five six"),
        ],
        document_type="pdf",
    )
    chunks = chunk_document(document, max_chunk_size=10)
    all_words = " ".join(c.text for c in chunks).split()
    assert all_words == ["one", "two", "three", "four", "five", "six"]


def test_exact_max_chunk_size_boundary_hand_constructed():
    # "abc" (3) + " " + "def" (3) = "abc def" is exactly 7 chars == max_chunk_size,
    # so it must be included in the same chunk (inclusive <= boundary). The next
    # word "g" cannot fit (7 + 1 + 1 = 9 > 7) and starts a new chunk.
    document = ParsedDocument(
        pages=[ParsedPage(page_number=None, text="abc def g")],
        document_type="text",
    )
    chunks = chunk_document(document, max_chunk_size=7)
    assert [c.text for c in chunks] == ["abc def", "g"]
    assert len(chunks[0].text) == 7


def test_single_word_longer_than_max_chunk_size():
    # An oversized word can never fit within max_chunk_size. It still gets
    # flushed as its own chunk (exceeding the limit) rather than being split
    # or dropped, and it does not merge with a preceding word.
    document = ParsedDocument(
        pages=[ParsedPage(page_number=None, text="hi abcdefghij")],
        document_type="text",
    )
    chunks = chunk_document(document, max_chunk_size=5)
    assert [c.text for c in chunks] == ["hi", "abcdefghij"]
    assert len(chunks[1].text) > 5


def test_chunk_document_returns_document_chunk_instances():
    document = ParsedDocument(
        pages=[ParsedPage(page_number=None, text="hello world")],
        document_type="text",
    )
    chunks = chunk_document(document)
    assert all(isinstance(c, DocumentChunk) for c in chunks)


# --- build_chunk_locator ---


def test_build_chunk_locator_single_pdf_page():
    chunk = DocumentChunk(text="...", chunk_index=0, page_numbers=(4,))
    locator = build_chunk_locator(chunk, document_type="pdf")
    assert locator == {"locator_type": "pdf_pages", "page_numbers": [4]}


def test_build_chunk_locator_preserves_page_gap():
    chunk = DocumentChunk(text="...", chunk_index=0, page_numbers=(1, 3))
    locator = build_chunk_locator(chunk, document_type="pdf")
    assert locator == {"locator_type": "pdf_pages", "page_numbers": [1, 3]}
    assert locator["page_numbers"] != [1, 2]
    assert locator["page_numbers"] != [1, 2, 3]


def test_build_chunk_locator_docx():
    chunk = DocumentChunk(text="...", chunk_index=0, page_numbers=())
    locator = build_chunk_locator(chunk, document_type="docx")
    assert locator == {"locator_type": "docx"}


def test_build_chunk_locator_text():
    chunk = DocumentChunk(text="...", chunk_index=0, page_numbers=())
    locator = build_chunk_locator(chunk, document_type="text")
    assert locator == {"locator_type": "text"}


def test_build_chunk_locator_defensive_empty_pages_with_pdf_type():
    # Shouldn't occur given Commit 1's page-numbering invariants, but
    # must not crash if it ever does.
    chunk = DocumentChunk(text="...", chunk_index=0, page_numbers=())
    locator = build_chunk_locator(chunk, document_type="pdf")
    assert locator == {"locator_type": "pdf_pages", "page_numbers": []}


def test_build_chunk_locator_is_json_serializable():
    import json
    chunk = DocumentChunk(text="...", chunk_index=0, page_numbers=(1, 3))
    locator = build_chunk_locator(chunk, document_type="pdf")
    json.dumps(locator)  # must not raise
