"""Deterministic in-memory original-file fixtures for ingestion tests."""

from __future__ import annotations

import io
import zipfile
from xml.sax.saxutils import escape


def make_docx(paragraphs: list[str] | None = None) -> bytes:
    if paragraphs is None:
        paragraphs = ["Hello DOCX world"]
    body = "".join(
        f"<w:p><w:r><w:t>{escape(text)}</w:t></w:r></w:p>"
        for text in paragraphs
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "word/_rels/document.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            "</Relationships>",
        )
        archive.writestr(
            "word/document.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{body}<w:sectPr/></w:body>"
            "</w:document>",
        )
    return output.getvalue()


def make_pdf(pages: list[str | None]) -> bytes:
    """Build a valid text PDF; ``None`` creates an empty source page."""
    page_numbers = [4 + index for index in range(len(pages))]
    content_numbers = [4 + len(pages) + index for index in range(len(pages))]
    objects: list[tuple[int, bytes]] = [
        (1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        (
            2,
            (
                "<< /Type /Pages /Kids ["
                + " ".join(f"{number} 0 R" for number in page_numbers)
                + f"] /Count {len(pages)} >>"
            ).encode(),
        ),
        (3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]

    for page_number, content_number in zip(page_numbers, content_numbers):
        objects.append(
            (
                page_number,
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 3 0 R >> >> "
                    f"/Contents {content_number} 0 R >>"
                ).encode(),
            )
        )

    for content_number, text in zip(content_numbers, pages):
        if text:
            escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            stream = f"BT /F1 24 Tf 72 700 Td ({escaped}) Tj ET".encode()
        else:
            stream = b""
        objects.append(
            (
                content_number,
                b"<< /Length %d >>\nstream\n" % len(stream)
                + stream
                + b"\nendstream",
            )
        )

    output = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number, body in sorted(objects):
        offsets[number] = len(output)
        output += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(output)
    max_number = max(offsets)
    output += f"xref\n0 {max_number + 1}\n".encode()
    output += b"0000000000 65535 f \n"
    for number in range(1, max_number + 1):
        output += f"{offsets.get(number, 0):010d} 00000 n \n".encode()
    output += (
        f"trailer\n<< /Size {max_number + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF"
    ).encode()
    return bytes(output)
