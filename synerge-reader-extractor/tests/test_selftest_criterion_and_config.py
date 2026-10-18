"""Pins: the synthetic fixture, its expected values and the self-test threshold are unchanged, and the
only table-configuration amendment is the Docling table OCR page segmentation mode (3 -> 11)."""

import ast
import pathlib

from extractor_service import engine
from extractor_service.selftest import TABLE, TEXT_LINES, table_criterion

ENGINE_SRC = (pathlib.Path(engine.__file__)).read_text(encoding="utf-8")


def _table(values):
    return [{"cells": [{"text": v} for v in values]}]


ALL = [v for row in TABLE for v in row]


def test_fixture_values_are_unchanged():
    assert TEXT_LINES == ["SELFTEST INVOICE 12345", "Amount due 678.90", "Reference ALPHA BRAVO"]
    assert TABLE == [["Item", "2025", "2024"], ["Revenue", "1,200", "1,050"], ["Expenses", "(300)", "(280)"],
                     ["Net", "900", "770"]]


def test_threshold_is_unchanged_at_most_two_missing():
    assert table_criterion(_table(ALL)) == (12, 12, True)
    assert table_criterion(_table(ALL[:10])) == (10, 12, True)
    assert table_criterion(_table(ALL[:9])) == (9, 12, False)
    assert table_criterion([]) == (0, 12, False)
    assert table_criterion(_table([" Item ", "2025\n"] + ALL[2:])) == (12, 12, True), "cell text is stripped"


def _call_kwargs(func_name):
    """Keyword arguments of every call to ``func_name`` in engine.py (static, no Docling import)."""
    found = []
    for node in ast.walk(ast.parse(ENGINE_SRC)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == func_name:
            found.append({kw.arg: ast.unparse(kw.value) for kw in node.keywords})
    return found


def test_only_the_docling_table_ocr_psm_is_amended():
    assert engine.REVIEWED_TABLE_OCR_PSM == 3 and engine.TABLE_OCR_PSM == 11
    assert engine.TABLE_OCR_AMENDMENT == "TABLE-OCR-PSM11-20261017"
    (ocr,) = _call_kwargs("TesseractOcrOptions")
    assert ocr == {"mode": "OcrMode.FULL_PAGE", "lang": "['eng']", "psm": "TABLE_OCR_PSM",
                   "scale": "OCR_SCALE_300", "path": "self.cfg['tessdata_dir'].rstrip('/') + '/'"}
    (table,) = _call_kwargs("TableStructureOptions")
    assert table == {"do_cell_matching": "True", "mode": "TableFormerMode.ACCURATE"}
    assert engine.OCR_SCALE_300 == 300 / 72
    assert engine.HERON_REVISION == "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8"


def test_standalone_page_ocr_is_unchanged():
    (api,) = _call_kwargs("PyTessBaseAPI")
    assert api == {"path": "self.cfg['tessdata_dir'].rstrip('/') + '/'", "lang": "'eng'",
                   "psm": "tesserocr.PSM.AUTO", "oem": "tesserocr.OEM.LSTM_ONLY"}
