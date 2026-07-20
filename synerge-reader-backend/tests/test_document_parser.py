import io
import os
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from document_parser import (
    MAX_EXTRACTED_CHARS,
    ExtractionError,
    ExtractionResult,
    ParsedDocument,
    ParsedPage,
    UnsupportedFileTypeError,
    _detect_file_type,
    extract_text_from_upload,
    looks_like_text,
)


def test_pdf_magic_bytes_detection():
    content = b"%PDF-1.4 fake content"
    assert _detect_file_type("document.pdf", content) == "pdf"


def test_pdf_no_extension_magic_bytes_win():
    content = b"%PDF-1.4 fake content"
    assert _detect_file_type("my_report", content) == "pdf"


def test_unknown_binary_raises_415():
    content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        extract_text_from_upload("image.png", content)
    assert exc_info.value.http_status == 415


def test_empty_file_raises_422():
    with pytest.raises(ExtractionError) as exc_info:
        extract_text_from_upload("test.txt", b"")
    assert exc_info.value.http_status == 422


def test_oversized_file_raises_413():
    with pytest.raises(ExtractionError) as exc_info:
        extract_text_from_upload("big.pdf", b"x" * (51 * 1024 * 1024))
    assert exc_info.value.http_status == 413


def test_txt_extraction():
    result = extract_text_from_upload("notes.txt", b"Hello world")
    assert result.document_type == "text"
    assert result.file_type == "text"  # back-compat alias for document_type
    assert result.text == "Hello world"
    assert len(result.pages) == 1
    assert result.pages[0].page_number is None
    assert result.pages[0].text == "Hello world"


def test_looks_like_text_null_byte():
    assert looks_like_text(b"some text\x00more") is False


def test_looks_like_text_valid_utf8():
    assert looks_like_text(b"Hello, this is clean ASCII text.") is True


def test_pdf_extraction_corrupt():
    pytest.importorskip("pdfplumber")
    content = b"%PDF-1.4\nthis is not a real pdf"
    with pytest.raises(ExtractionError):
        extract_text_from_upload("test.pdf", content)


def _make_minimal_docx(paragraphs=None) -> bytes:
    if paragraphs is None:
        paragraphs = ["Hello DOCX world"]
    body_paras = "".join(
        f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        z.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            "</Relationships>",
        )
        z.writestr(
            "word/_rels/document.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            "</Relationships>",
        )
        z.writestr(
            "word/document.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body>"
            + body_paras
            + "<w:sectPr/>"
            "</w:body>"
            "</w:document>",
        )
    return buf.getvalue()


def test_docx_extraction():
    pytest.importorskip("docx")
    content = _make_minimal_docx()
    result = extract_text_from_upload("test.docx", content)
    assert result.document_type == "docx"
    assert "Hello DOCX world" in result.text


def _make_pdf(pages_text) -> bytes:
    """Build a minimal, valid multi-page PDF by hand.

    pages_text: list of str|None. None produces a page with an empty content
    stream (no extractable text), matching a genuinely blank source page.
    """
    n_pages = len(pages_text)
    page_obj_nums = [4 + i for i in range(n_pages)]
    content_obj_nums = [4 + n_pages + i for i in range(n_pages)]

    objects = []
    objects.append((1, b"<< /Type /Catalog /Pages 2 0 R >>"))
    kids = " ".join(f"{n} 0 R" for n in page_obj_nums)
    objects.append((2, f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode()))
    objects.append((3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))

    for i in range(n_pages):
        pnum = page_obj_nums[i]
        cnum = content_obj_nums[i]
        page_dict = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {cnum} 0 R >>"
        ).encode()
        objects.append((pnum, page_dict))

    for i, text in enumerate(pages_text):
        cnum = content_obj_nums[i]
        if text:
            escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            stream = f"BT /F1 24 Tf 72 700 Td ({escaped}) Tj ET".encode()
        else:
            stream = b""
        content_obj = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        objects.append((cnum, content_obj))

    objects.sort(key=lambda o: o[0])

    buf = bytearray()
    buf += b"%PDF-1.4\n"
    offsets = {}
    for num, body in objects:
        offsets[num] = len(buf)
        buf += f"{num} 0 obj\n".encode()
        buf += body
        buf += b"\nendobj\n"

    xref_offset = len(buf)
    max_num = max(offsets.keys())
    buf += f"xref\n0 {max_num + 1}\n".encode()
    buf += b"0000000000 65535 f \n"
    for num in range(1, max_num + 1):
        off = offsets.get(num, 0)
        buf += f"{off:010d} 00000 n \n".encode()
    buf += f"trailer\n<< /Size {max_num + 1} /Root 1 0 R >>\n".encode()
    buf += f"startxref\n{xref_offset}\n%%EOF".encode()
    return bytes(buf)


def test_pdf_multi_page_extraction():
    pytest.importorskip("pdfplumber")
    content = _make_pdf(["Page one text", "Page two text", "Page three text"])
    result = extract_text_from_upload("doc.pdf", content)

    assert result.document_type == "pdf"
    assert len(result.pages) == 3
    for page in result.pages:
        assert page.text.strip() != ""

    assert result.text == "\n\n".join(p.text for p in result.pages)
    assert "--- Page" not in result.text
    assert "[Page" not in result.text


def test_pdf_empty_middle_page_preserves_true_page_numbers():
    pytest.importorskip("pdfplumber")
    content = _make_pdf(["Page one text", None, "Page three text"])
    result = extract_text_from_upload("doc.pdf", content)

    page_numbers = [p.page_number for p in result.pages]
    assert page_numbers == [1, 3]
    assert page_numbers != [1, 2]


def test_pdf_page_limit_exceeded():
    pytest.importorskip("pdfplumber")
    content = _make_pdf([f"Page {i} text" for i in range(1, 152)])
    with pytest.raises(ExtractionError) as exc_info:
        extract_text_from_upload("big.pdf", content)
    assert exc_info.value.http_status == 422
    assert "page limit" in exc_info.value.user_message


def test_pdf_scanned_no_text_raises():
    pytest.importorskip("pdfplumber")
    content = _make_pdf([None, None])
    with pytest.raises(ExtractionError) as exc_info:
        extract_text_from_upload("scanned.pdf", content)
    assert exc_info.value.http_status == 422
    assert "scanned" in exc_info.value.user_message.lower()


def test_docx_all_pages_have_no_page_number():
    pytest.importorskip("docx")
    content = _make_minimal_docx(["First paragraph", "Second paragraph"])
    result = extract_text_from_upload("doc.docx", content)

    assert result.document_type == "docx"
    assert all(p.page_number is None for p in result.pages)
    assert result.text != ""


def test_docx_text_matches_old_extraction_algorithm():
    pytest.importorskip("docx")
    import docx as python_docx

    content = _make_minimal_docx(["First paragraph", "Second paragraph", "Third one"])

    # Reference implementation of the pre-refactor algorithm: paragraph texts
    # (non-blank) joined with a blank line, exactly as ExtractionResult.text
    # used to be built.
    document = python_docx.Document(io.BytesIO(content))
    expected = "\n\n".join(p.text for p in document.paragraphs if p.text.strip())

    result = extract_text_from_upload("doc.docx", content)
    assert result.text == expected


def test_plain_text_special_characters_preserved_exactly():
    raw = "Section § 1 — “Quoted” text with an em-dash — done.".encode("utf-8")

    # Reference implementation of the pre-refactor algorithm.
    expected = raw.decode("utf-8")

    result = extract_text_from_upload("notes.txt", raw)
    assert result.text == expected
    assert result.text == raw.decode("utf-8")


def test_empty_upload_raises_same_exception():
    with pytest.raises(ExtractionError) as exc_info:
        extract_text_from_upload("empty.txt", b"")
    assert exc_info.value.http_status == 422


def test_combined_pages_exceed_limit_truncates():
    pytest.importorskip("docx")
    half = MAX_EXTRACTED_CHARS // 2
    para1 = "a" * (half + 10)  # individually well under the limit
    para2 = "b" * (half + 10)  # individually well under the limit, combined exceeds it
    content = _make_minimal_docx([para1, para2])

    result = extract_text_from_upload("big.docx", content)

    assert len(result.text) == MAX_EXTRACTED_CHARS
    assert result.truncated is True
    assert result.char_count == MAX_EXTRACTED_CHARS
    assert "Document truncated" in result.warnings[0]


def test_docx_truncation_boundary_lands_mid_separator(monkeypatch):
    pytest.importorskip("docx")
    import document_parser

    paragraphs = ["abcde", "fghij"]
    combined = "\n\n".join(paragraphs)  # "abcde\n\nfghij"
    limit = 6  # cuts after the first of the two "\n\n" separator characters
    expected = combined[:limit]
    assert expected == "abcde\n"  # sanity-check the boundary actually lands mid-separator

    monkeypatch.setattr(document_parser, "MAX_EXTRACTED_CHARS", limit)
    content = _make_minimal_docx(paragraphs)
    result = document_parser.extract_text_from_upload("boundary.docx", content)

    assert result.text == expected
    assert result.char_count == len(expected)
    assert result.truncated is True
    assert result.warnings == ["Document truncated to 300,000 characters."]
    assert len(result.pages) == 1
    assert result.pages[0].page_number is None
    assert result.pages[0].text == expected


def test_extraction_result_alias_and_extra_fields_resolve():
    # ExtractionResult must remain usable as the old type name, and every
    # field that used to live on ExtractionResult (beyond .text) must still
    # resolve correctly on the new type.
    assert ExtractionResult is ParsedDocument

    result = extract_text_from_upload("notes.txt", b"Hello world")
    assert isinstance(result, ExtractionResult)
    assert result.file_type == "text"       # compat property for document_type
    assert result.page_count == 0           # not set for text (only meaningful for pdf)
    assert result.char_count == len("Hello world")
    assert result.truncated is False
    assert result.warnings == []


def test_pdf_page_count_reflects_total_source_pages():
    pytest.importorskip("pdfplumber")
    content = _make_pdf(["Page one text", None, "Page three text"])
    result = extract_text_from_upload("doc.pdf", content)
    assert result.page_count == 3
    assert len(result.pages) == 2


def test_parsed_page_is_a_frozen_dataclass():
    page = ParsedPage(page_number=1, text="hi")
    with pytest.raises(Exception):
        page.text = "changed"
