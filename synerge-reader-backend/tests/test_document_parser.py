import io
import os
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from document_parser import (
    ExtractionError,
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
    assert result.file_type == "text"
    assert result.text == "Hello world"


def test_looks_like_text_null_byte():
    assert looks_like_text(b"some text\x00more") is False


def test_looks_like_text_valid_utf8():
    assert looks_like_text(b"Hello, this is clean ASCII text.") is True


def test_pdf_extraction_corrupt():
    pytest.importorskip("pdfplumber")
    content = b"%PDF-1.4\nthis is not a real pdf"
    with pytest.raises(ExtractionError):
        extract_text_from_upload("test.pdf", content)


def _make_minimal_docx() -> bytes:
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
            "<w:p><w:r><w:t>Hello DOCX world</w:t></w:r></w:p>"
            "<w:sectPr/>"
            "</w:body>"
            "</w:document>",
        )
    return buf.getvalue()


def test_docx_extraction():
    pytest.importorskip("docx")
    content = _make_minimal_docx()
    result = extract_text_from_upload("test.docx", content)
    assert result.file_type == "docx"
    assert "Hello DOCX world" in result.text
