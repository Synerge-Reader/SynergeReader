"""Stage-by-stage diagnosis of the PDF word-boundary collapse.

Pure and offline: every PDF here is built byte-by-byte in memory by
``_build_pdf`` below, so nothing reads a fixture file, opens a database,
contacts Ollama, or touches the network. No new dependency is used -- the
extraction capability exercised here (``x_tolerance_ratio``) already ships
with the installed pdfplumber.

What these prove: the collapse reported on real research PDFs
("Inpractice,wecomputetheattentionfunction") is produced by the PDF page
parser itself, at the very first stage, and not by normalization, chunking,
storage, evidence assembly, or excerpting -- each of which is shown here to
carry whatever boundaries it is handed. They then pin the corrected parser
and prove the repaired boundaries survive all the way to a citation excerpt,
with page metadata intact.

What these do NOT prove: how any particular real-world PDF is typeset. The
fixtures reproduce the *mechanism* (an inter-word gap narrower than the
library's fixed 3-point default) rather than re-shipping a copyrighted paper.
"""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pdfplumber = pytest.importorskip("pdfplumber")

from answer_evidence import EvidenceItem, EvidenceBundle, EvidenceMode
from citation_generation import CitationRegistry, bounded_excerpt
from document_chunker import build_chunk_locator, chunk_document
from document_parser import (
    PDF_X_TOLERANCE_RATIO,
    ParsedDocument,
    ParsedPage,
    extract_text_from_upload,
)


# --- a PDF built from glyph positions, so the gap is exactly what we say ----

# Helvetica advance widths, per 1000 em units.
_WIDTHS = {
    " ": 278, ",": 278, ".": 278, "(": 333, ")": 333, "=": 584, "-": 333,
    "A": 667, "B": 667, "D": 722, "E": 667, "G": 778, "I": 278, "K": 667,
    "L": 611, "P": 667, "Q": 778, "T": 611, "V": 722, "W": 944,
    "0": 556, "1": 556, "2": 556, "3": 556, "4": 556, "5": 556, "6": 556,
    "7": 556, "8": 556, "9": 556,
    "a": 556, "b": 556, "c": 500, "d": 556, "e": 556, "f": 278, "g": 556,
    "h": 556, "i": 222, "j": 222, "k": 500, "l": 222, "m": 833, "n": 556,
    "o": 556, "p": 556, "q": 556, "r": 333, "s": 500, "t": 278, "u": 556,
    "v": 500, "w": 722, "x": 500, "y": 500, "z": 500,
}


def _text_width(word, size):
    return sum(_WIDTHS.get(ch, 500) for ch in word) * size / 1000.0


def _build_pdf(pages_of_lines, size=9.0, word_gap=2.5):
    """A real PDF whose words sit ``word_gap`` points apart.

    ``word_gap`` is the whole point of the fixture. At 2.5pt it is narrower
    than pdfplumber's fixed 3pt default tolerance -- exactly the condition
    that makes a typeset paper extract as one run-on token -- while still
    being a perfectly ordinary inter-word space for 9pt type.
    """
    content_streams = []
    for lines in pages_of_lines:
        parts = []
        y = 700.0
        for words in lines:
            x = 72.0
            for word in words:
                escaped = (
                    word.replace("\\", "\\\\").replace("(", r"\(").replace(")", r"\)")
                )
                parts.append(
                    "BT /F1 %.2f Tf %.3f %.3f Td (%s) Tj ET"
                    % (size, x, y, escaped)
                )
                x += _text_width(word, size) + word_gap
            y -= 14.0
        content_streams.append("\n".join(parts).encode("latin-1"))

    objects = []
    page_count = len(content_streams)
    # 1 catalog, 2 pages, 3 font, then one page object + one stream per page.
    page_obj_ids = [4 + 2 * i for i in range(page_count)]
    kids = " ".join("%d 0 R" % pid for pid in page_obj_ids)

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(
        ("<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, page_count)).encode()
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, stream in enumerate(content_streams):
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                "/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>"
                % (page_obj_ids[i] + 1)
            ).encode()
        )
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
            + stream + b"\nendstream"
        )

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(("%d 0 obj\n" % i).encode())
        out.write(body)
        out.write(b"\nendobj\n")
    xref = out.tell()
    out.write(("xref\n0 %d\n" % (len(objects) + 1)).encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets:
        out.write(("%010d 00000 n \n" % off).encode())
    out.write(
        ("trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
         % (len(objects) + 1, xref)).encode()
    )
    return out.getvalue()


_SENTENCE = ["In", "practice,", "we", "compute", "the", "attention", "function"]
_SECOND = ["Due", "to", "the", "reduced", "dimension"]


@pytest.fixture(scope="module")
def tight_pdf_bytes():
    return _build_pdf([[_SENTENCE, _SECOND]])


@pytest.fixture(scope="module")
def two_page_pdf_bytes():
    return _build_pdf([[_SENTENCE], [_SECOND]])


# --- 1: the fixture really does reproduce the reported corruption ----------


def test_the_library_default_is_what_collapses_the_words(tight_pdf_bytes):
    """The first stage is the parser, and this is the proof.

    Reading the same bytes with pdfplumber's default tolerance reproduces the
    reported string character for character. Nothing downstream has run yet.
    """
    with pdfplumber.open(io.BytesIO(tight_pdf_bytes)) as pdf:
        default_text = pdf.pages[0].extract_text()

    assert "Inpractice,wecomputetheattentionfunction" in default_text, (
        "the fixture must reproduce the real-document failure, or it is not "
        "diagnosing the reported defect"
    )
    assert "Duetothereduceddimension" in default_text


def test_scaling_the_tolerance_to_the_font_size_restores_the_boundaries(
    tight_pdf_bytes,
):
    with pdfplumber.open(io.BytesIO(tight_pdf_bytes)) as pdf:
        fixed = pdf.pages[0].extract_text()
        scaled = pdf.pages[0].extract_text(x_tolerance_ratio=PDF_X_TOLERANCE_RATIO)

    assert "In practice, we compute the attention function" in scaled
    assert "Due to the reduced dimension" in scaled
    assert scaled != fixed


def test_the_repaired_parser_emits_separated_words(tight_pdf_bytes):
    parsed = extract_text_from_upload("attention.pdf", tight_pdf_bytes)

    assert parsed.document_type == "pdf"
    assert "In practice, we compute the attention function" in parsed.text
    assert "Inpractice" not in parsed.text, (
        "the parser is the stage that must not collapse word boundaries"
    )


def test_the_parser_invents_no_spaces_inside_words(tight_pdf_bytes):
    """Separating words must not fragment them.

    A tolerance small enough to split words apart would also split letters
    apart. Every word written into the fixture must come back whole.
    """
    parsed = extract_text_from_upload("attention.pdf", tight_pdf_bytes)
    tokens = parsed.text.split()

    for word in _SENTENCE + _SECOND:
        assert word in tokens, f"{word!r} was fragmented by the tolerance"
    assert len(tokens) == len(_SENTENCE) + len(_SECOND), (
        "no token may be invented or lost"
    )


# --- 2: the stages after the parser carry whatever they are handed ---------


def test_the_chunker_preserves_whatever_boundaries_it_receives():
    """Chunking is not the corrupting stage, in either direction.

    Given collapsed text it stays collapsed; given separated text it stays
    separated. That is what rules the chunker out as the origin.
    """
    collapsed = ParsedDocument(
        pages=[ParsedPage(page_number=1, text="Inpractice,wecompute")],
        document_type="pdf",
    )
    separated = ParsedDocument(
        pages=[ParsedPage(page_number=1, text="In practice, we compute")],
        document_type="pdf",
    )

    assert chunk_document(collapsed)[0].text == "Inpractice,wecompute"
    assert chunk_document(separated)[0].text == "In practice, we compute"


def test_boundaries_survive_parsing_and_chunking_into_stored_text(tight_pdf_bytes):
    parsed = extract_text_from_upload("attention.pdf", tight_pdf_bytes)
    chunks = chunk_document(parsed)

    stored = " ".join(chunk.text for chunk in chunks)
    assert "In practice, we compute the attention function" in stored
    assert "Inpractice" not in stored


def test_boundaries_survive_all_the_way_into_a_citation_excerpt(tight_pdf_bytes):
    """Parser -> chunk -> evidence bundle -> registry -> public excerpt."""
    parsed = extract_text_from_upload("attention.pdf", tight_pdf_bytes)
    chunk = chunk_document(parsed)[0]

    bundle = EvidenceBundle(
        mode=EvidenceMode.HYBRID_RETRIEVAL,
        items=(
            EvidenceItem(
                text=chunk.text,
                source_type="document_chunk",
                document_id=1,
                filename="attention.pdf",
                chunk_id="1-0",
                chunk_index=chunk.chunk_index,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
            ),
        ),
    )
    record = CitationRegistry.from_bundle(bundle).records[0]

    assert "In practice, we compute the attention function" in record.excerpt
    assert "Inpractice" not in record.excerpt
    assert "Inpractice" not in record.evidence_text


def test_excerpt_bounding_normalises_whitespace_without_joining_words():
    """The excerpt bound collapses runs of whitespace, never word gaps."""
    text, truncated = bounded_excerpt("In  practice,\n we   compute", 200)

    assert text == "In practice, we compute"
    assert truncated is False


# --- page metadata survives extraction and chunking ------------------------


def test_page_numbers_survive_extraction(two_page_pdf_bytes):
    parsed = extract_text_from_upload("attention.pdf", two_page_pdf_bytes)

    assert [page.page_number for page in parsed.pages] == [1, 2]
    assert parsed.page_count == 2


def test_page_metadata_survives_chunking_with_repaired_text(two_page_pdf_bytes):
    parsed = extract_text_from_upload("attention.pdf", two_page_pdf_bytes)
    chunks = chunk_document(parsed, max_chunk_size=10_000)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.page_numbers == (1, 2)
    assert chunk.page_start == 1
    assert chunk.page_end == 2
    assert "In practice, we compute the attention function" in chunk.text
    assert "Due to the reduced dimension" in chunk.text


def test_chunk_locator_metadata_still_carries_real_pages(two_page_pdf_bytes):
    parsed = extract_text_from_upload("attention.pdf", two_page_pdf_bytes)
    chunk = chunk_document(parsed, max_chunk_size=10_000)[0]

    locator = build_chunk_locator(chunk, "pdf")
    assert locator == {"locator_type": "pdf_pages", "page_numbers": [1, 2]}


def _record_for(chunk, filename="attention.pdf"):
    bundle = EvidenceBundle(
        mode=EvidenceMode.HYBRID_RETRIEVAL,
        items=(
            EvidenceItem(
                text=chunk.text,
                source_type="document_chunk",
                document_id=1,
                filename=filename,
                chunk_id="1-%d" % chunk.chunk_index,
                chunk_index=chunk.chunk_index,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
            ),
        ),
    )
    return CitationRegistry.from_bundle(bundle).records[0]


def test_a_single_page_citation_reports_that_real_page(two_page_pdf_bytes):
    """A chunk confined to one page cites that page, not a fabricated one."""
    parsed = extract_text_from_upload("attention.pdf", two_page_pdf_bytes)
    single = chunk_document(
        ParsedDocument(pages=parsed.pages[:1], document_type="pdf"),
        max_chunk_size=10_000,
    )[0]

    record = _record_for(single)
    assert record.page_start == 1
    assert record.page_end == 1
    assert record.locator.label == "page 1"


def test_a_citation_spanning_two_pages_reports_the_real_range(two_page_pdf_bytes):
    parsed = extract_text_from_upload("attention.pdf", two_page_pdf_bytes)
    spanning = chunk_document(parsed, max_chunk_size=10_000)[0]

    record = _record_for(spanning)
    assert record.page_start == 1
    assert record.page_end == 2
    assert record.locator.label == "pages 1-2", (
        "a real extracted page range must reach the citation locator intact"
    )
