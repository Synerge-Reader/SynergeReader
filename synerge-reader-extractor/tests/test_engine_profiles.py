"""Table profiles: the service routes only pages with an embedded text layer to native_text_tables_v1.

pypdfium2 cannot be imported under the test guard (its import starts a subprocess), so a minimal fake
module exposes the text-layer calls the engine uses; the Docling conversion is a stub. No model, OCR
or inference runs.
"""

import ast
import os
import pathlib
import sys
import types

import pytest

from extractor_service import engine
from extractor_service.engine import (Engines, NATIVE_TEXT_MIN_ALNUM, PROFILE_NATIVE_TEXT, PROFILE_OCR_PSM11,
                                      native_alnum_chars, no_native_text_table_page)

NATIVE = "Revenue 1,200 1,050 Expenses (300) (280) Net 900 770 for the period"


class _TextPage:
    def __init__(self, text):
        self.text, self.closed = text, False

    def get_text_range(self):
        return self.text

    def close(self):
        self.closed = True


class _Page:
    def __init__(self, text):
        self.text = text

    def get_textpage(self):
        return _TextPage(self.text)

    def close(self):
        pass


class FakePdf:
    """Pages given as text (None = no text layer); the bytes passed in are the page texts joined."""

    def __init__(self, pages):
        self.pages = pages

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, index):
        return _Page(self.pages[index] or "")

    def close(self):
        pass


@pytest.fixture
def fake_pdfium(monkeypatch):
    registry = {}
    module = types.ModuleType("pypdfium2")
    module.PdfDocument = lambda data: FakePdf(registry[data])
    monkeypatch.setitem(sys.modules, "pypdfium2", module)

    def make(pages):
        data = repr(pages).encode()
        registry[data] = pages
        return data
    return make


def _engines(tmp_path):
    return Engines({"max_pdf_pages": 150, "work_dir": str(tmp_path), "ocr_dpi": 300, "torch_threads": 1,
                    "tessdata_dir": "/none", "model_dir": "/none"})


def test_native_alnum_chars_reads_the_text_layer():
    pdf = FakePdf([NATIVE, None])
    assert native_alnum_chars(pdf, 1) == sum(ch.isalnum() for ch in NATIVE)
    assert native_alnum_chars(pdf, 2) == 0


class _UnreadablePage(_Page):
    def get_textpage(self):
        raise RuntimeError("no text page")


class _UnreadablePdf(FakePdf):
    def __getitem__(self, index):
        return _UnreadablePage("")


def test_native_alnum_chars_is_zero_when_the_text_layer_cannot_be_read():
    assert native_alnum_chars(_UnreadablePdf([NATIVE]), 1) == 0


def test_only_pages_with_embedded_text_are_converted_and_with_the_native_profile(tmp_path, monkeypatch,
                                                                                         fake_pdfium):
    calls = []

    def fake_table_page(self, path, page_number, profile=PROFILE_NATIVE_TEXT):
        calls.append((page_number, profile, os.path.exists(path)))
        return {"page_number": page_number, "method": "table", "text": "native words", "mean_confidence": None,
                "tables": [], "warnings": [], "diagnostics": {"profile": profile}}

    monkeypatch.setattr(Engines, "_table_page", fake_table_page)
    monkeypatch.setattr(Engines, "versions", lambda self: {"table_profile": PROFILE_NATIVE_TEXT})
    out = _engines(tmp_path).run({"pdf_bytes": fake_pdfium([NATIVE, None]), "ocr_pages": [], "table_pages": [1, 2]})
    assert calls == [(1, PROFILE_NATIVE_TEXT, True)]
    page1, page2 = out["pages"]
    assert page1["diagnostics"] == {"profile": PROFILE_NATIVE_TEXT,
                                    "native_alnum_chars": sum(ch.isalnum() for ch in NATIVE)}
    assert page2 == no_native_text_table_page(2, 0)
    assert page2["tables"] == [] and page2["text"] == "" and page2["diagnostics"]["reason"] == "no_native_text"
    assert "no embedded text layer" in page2["warnings"][0]
    assert list(tmp_path.iterdir()) == [], "the task-scoped work directory is removed"


def test_scanned_table_page_never_starts_a_conversion(tmp_path, monkeypatch, fake_pdfium):
    monkeypatch.setattr(Engines, "_table_page", lambda *a, **k: pytest.fail("no conversion for scanned pages"))
    monkeypatch.setattr(Engines, "versions", lambda self: {})
    out = _engines(tmp_path).run({"pdf_bytes": fake_pdfium([None]), "ocr_pages": [], "table_pages": [1]})
    assert out["pages"][0]["diagnostics"]["profile"] is None


def test_threshold_matches_the_backend_usable_text_rule():
    backend = (pathlib.Path(__file__).resolve().parents[2] / "synerge-reader-backend" / "enhanced_extraction.py")
    assert f"MIN_USABLE_ALNUM_CHARS = {NATIVE_TEXT_MIN_ALNUM}" in backend.read_text(encoding="utf-8")


def test_unknown_profile_is_refused(tmp_path):
    with pytest.raises(ValueError):
        _engines(tmp_path)._docling("not_a_profile")


def _pipeline_kwargs():
    src = pathlib.Path(engine.__file__).read_text(encoding="utf-8")
    (call,) = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
               and getattr(n.func, "id", None) == "PdfPipelineOptions"]
    return {kw.arg: ast.unparse(kw.value) for kw in call.keywords}


def test_profiles_differ_in_do_ocr_only():
    kwargs = _pipeline_kwargs()
    assert kwargs.pop("do_ocr") == "profile == PROFILE_OCR_PSM11"
    assert kwargs.pop("do_table_structure") == "True" and kwargs.pop("force_backend_text") == "False"
    assert kwargs.pop("images_scale") == "OCR_SCALE_300" and kwargs.pop("document_timeout") == "900"
    assert kwargs.pop("generate_page_images") == "True" and kwargs.pop("generate_parsed_pages") == "True"
    for key in ("enable_remote_services", "allow_external_plugins", "do_code_enrichment", "do_formula_enrichment",
                "do_picture_classification", "do_picture_description", "generate_table_images",
                "generate_picture_images"):
        assert kwargs.pop(key) == "False", key
    assert set(kwargs) == {"ocr_batch_size", "layout_batch_size", "table_batch_size", "queue_max_size",
                           "use_reading_order_separators", "artifacts_path", "accelerator_options",
                           "layout_options", "table_structure_options", "ocr_options"}
    assert PROFILE_NATIVE_TEXT == "native_text_tables_v1" and PROFILE_OCR_PSM11 == "ocr_fullpage_psm11"
    assert engine.NATIVE_PROFILE_AMENDMENT == "TABLE-NATIVE-PROFILE-20261017"
