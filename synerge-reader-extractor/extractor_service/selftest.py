"""In-container self-test: model bytes, offline engine start, OCR and table structure on synthetic pages.

    docker run --rm --network none <image> python -m extractor_service.selftest

Builds two image-only PDFs in memory (a text page and a ruled table page), runs them through the
same supervised engine the service uses, and prints one JSON summary line. It needs no network,
no token and no repository data; the synthetic content is not a validation fixture.
"""

from __future__ import annotations

import io
import json
import sys
import time

from .config import ServiceSettings
from .engine import worker_main
from .model_check import verify
from .supervisor import Supervisor

TEXT_LINES = ["SELFTEST INVOICE 12345", "Amount due 678.90", "Reference ALPHA BRAVO"]
TABLE = [["Item", "2025", "2024"], ["Revenue", "1,200", "1,050"], ["Expenses", "(300)", "(280)"],
         ["Net", "900", "770"]]


def _font(size: int):
    from PIL import ImageFont
    return ImageFont.load_default(size=size)


def _pdf_from_image(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PDF", resolution=300.0)
    return buf.getvalue()


def text_page_pdf() -> bytes:
    from PIL import Image, ImageDraw
    img = Image.new("L", (2550, 3300), 255)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(TEXT_LINES):
        draw.text((200, 300 + i * 160), line, fill=0, font=_font(72))
    return _pdf_from_image(img)


def table_page_pdf() -> bytes:
    from PIL import Image, ImageDraw
    img = Image.new("L", (2550, 3300), 255)
    draw = ImageDraw.Draw(img)
    draw.text((200, 250), "Statement of results (selftest)", fill=0, font=_font(64))
    x0, y0, col_w, row_h = 200, 450, 600, 140
    for r, row in enumerate(TABLE):
        for c, value in enumerate(row):
            x, y = x0 + c * col_w, y0 + r * row_h
            draw.rectangle([x, y, x + col_w, y + row_h], outline=0, width=4)
            draw.text((x + 30, y + 35), value, fill=0, font=_font(60))
    return _pdf_from_image(img)


def table_criterion(tables: list) -> tuple[int, int, bool]:
    """The self-test's table acceptance rule: at least one table, and no more than two of the
    expected synthetic values missing from its cells. Returns (matched, expected, met)."""
    cells = {c["text"].strip() for t in tables for c in t["cells"]}
    expected = {v for row in TABLE for v in row}
    matched = len(expected & cells)
    return matched, len(expected), len(tables) >= 1 and matched >= len(expected) - 2


def main() -> int:
    settings = ServiceSettings(token="selftest-not-used")
    summary: dict = {"schema": "synergereader.extractor.selftest.v1"}
    problems = verify(settings.model_dir, "/app/MODEL_LOCK.json")
    summary["model_check"] = "ok" if not problems else problems
    if problems:
        print(json.dumps(summary))
        return 1
    sup = Supervisor(target=worker_main, engine_config=settings.engine_config(),
                     rss_limit_bytes=settings.worker_rss_limit_bytes)
    ok = True
    try:
        t0 = time.monotonic()
        result, stats = sup.run({"pdf_bytes": text_page_pdf(), "ocr_pages": [1], "table_pages": []}, 300)
        page = result["pages"][0]
        found = [line for line in TEXT_LINES if line in page["text"]]
        summary["ocr"] = {"elapsed_s": round(time.monotonic() - t0, 2), "lines_found": len(found),
                          "lines_expected": len(TEXT_LINES), "mean_confidence": page["mean_confidence"],
                          "peak_rss_mb": (stats.peak_rss_bytes or 0) // 2 ** 20}
        ok &= len(found) == len(TEXT_LINES)
        summary["engine"] = result["engine"]

        t0 = time.monotonic()
        result, stats = sup.run({"pdf_bytes": table_page_pdf(), "ocr_pages": [], "table_pages": [1]}, 900)
        page = result["pages"][0]
        tables = page["tables"]
        matched, expected_count, table_ok = table_criterion(tables)
        summary["table"] = {"elapsed_s": round(time.monotonic() - t0, 2), "tables": len(tables),
                            "shape": [(t["num_rows"], t["num_cols"]) for t in tables],
                            "cells_matched": matched, "cells_expected": expected_count,
                            "column_header_cells": sum(1 for t in tables for c in t["cells"] if c["column_header"]),
                            "peak_rss_mb": (stats.peak_rss_bytes or 0) // 2 ** 20, "warnings": page["warnings"],
                            "text_chars": len(page.get("text") or ""),
                            "diagnostics": page.get("diagnostics")}
        ok &= table_ok
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        ok = False
    finally:
        sup.shutdown()
    summary["result"] = "PASS" if ok else "FAIL"
    print(json.dumps(summary, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
