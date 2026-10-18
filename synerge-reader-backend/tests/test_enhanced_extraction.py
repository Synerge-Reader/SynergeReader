"""Enhanced PDF extraction: routing, OCR/table merge, coverage, worker client (LIVE-EXTRACTION-20261017).

No network, database or model is contacted: the worker is a fake client and the
HTTP client is exercised with a fake opener. PDFs are real, built in memory.
Synthetic text only -- no production document identifiers or expected answers.
"""

import io
import json
import os
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from document_ingestion import DocumentIngestionService, ErrorCategory, ResultStatus, UploadDocument  # noqa: E402
from document_parser import ExtractionError, extract_text_from_upload  # noqa: E402
from enhanced_extraction import (  # noqa: E402
    EnhancedPdfExtractor,
    ExtractionSettings,
    ExtractorError,
    ExtractorRejected,
    ExtractorUnavailable,
    HttpExtractorClient,
    WORKER_SCHEMA,
    WorkerPage,
    WorkerResult,
    build_document_extractor,
    page_ranges,
    page_signals,
    parse_worker_response,
    plan_pages,
)
from document_parser import ParsedPage  # noqa: E402
from original_file_fixtures import make_docx, make_pdf  # noqa: E402
from rag_model_profiles import resolve_embedding_profile  # noqa: E402

NATIVE = "This native page carries enough ordinary words to be indexed as it is."
LETTER_SPACED = " ".join("T o t a l a s s e t s a n d l i a b i l i t i e s f o r t h e p e r i o d".split()) + " 1 2 , 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1"
NUMERIC = "Revenue 1,234 2,345 Cost (567) (678) Profit 667 1,667 Tax (12) (34) Net 655 1,633 Other 1 2 3 4"

ENABLED = ExtractionSettings(worker_url="http://worker.invalid:9", worker_token="t" * 32)


class FakeClient:
    def __init__(self, pages=None, error=None):
        self.pages = pages or {}
        self.error = error
        self.calls = []

    def extract(self, pdf_bytes, *, ocr_pages, table_pages, timeout_s):
        self.calls.append({"bytes": pdf_bytes, "ocr": list(ocr_pages), "table": list(table_pages), "timeout": timeout_s})
        if self.error is not None:
            raise self.error
        wanted = set(ocr_pages) | set(table_pages)
        return WorkerResult(pages={n: p for n, p in self.pages.items() if n in wanted}, engine={"ocr": "fake"})


def _ocr(n, text, conf=88.0):
    return WorkerPage(page_number=n, method="ocr", text=text, mean_confidence=conf, tables=(), warnings=())


def _table_page(n, text, tables):
    return WorkerPage(page_number=n, method="table", text=text, mean_confidence=91.0, tables=tuple(tables), warnings=())


_TABLE = {
    "num_rows": 2, "num_cols": 2,
    "cells": [
        {"text": "Period A", "start_row": 0, "end_row": 1, "start_col": 1, "end_col": 2, "column_header": True},
        {"text": "Line item", "start_row": 1, "end_row": 2, "start_col": 0, "end_col": 1, "row_header": True},
        {"text": "(42)", "start_row": 1, "end_row": 2, "start_col": 1, "end_col": 2},
    ],
}


# --- classification and routing ------------------------------------------------


def test_page_signals_classify_text_quality_only():
    assert page_signals(NATIVE).usable and not page_signals(NATIVE).table_signal
    assert not page_signals("").usable
    assert not page_signals("4").usable
    spaced = page_signals(LETTER_SPACED)
    assert spaced.letter_spaced and not spaced.usable and spaced.table_signal
    assert page_signals(NUMERIC).table_signal


def test_standard_mode_sends_only_unusable_pages_to_ocr():
    pages = [ParsedPage(1, NATIVE), ParsedPage(2, ""), ParsedPage(3, NUMERIC), ParsedPage(4, LETTER_SPACED)]
    plans, ocr_over, table_over = plan_pages(pages, mode="standard", settings=ENABLED)
    assert [p.route for p in plans] == ["native", "ocr", "native", "ocr"]
    assert ocr_over == [] and table_over == []


def test_enhanced_mode_routes_table_signals_to_tables_and_caps_both_paths():
    settings = ExtractionSettings(worker_url="http://w.invalid", worker_token="x" * 16, max_ocr_pages=1, max_table_pages=1)
    pages = [ParsedPage(1, ""), ParsedPage(2, ""), ParsedPage(3, NUMERIC), ParsedPage(4, NUMERIC + " 9 9 9")]
    plans, ocr_over, table_over = plan_pages(pages, mode="enhanced", settings=settings)
    routes = {p.page_number: p.route for p in plans}
    assert routes == {1: "ocr", 2: "skipped", 3: "native", 4: "table"}
    assert ocr_over == [2] and table_over == [3]


def test_page_ranges_compress_consecutive_pages():
    assert page_ranges([5, 1, 2, 3, 9, 10]) == "1-3, 5, 9-10"


# --- worker disabled: standard behaviour preserved ------------------------------


def test_disabled_extractor_matches_standard_parser_for_native_pdfs():
    pdf = make_pdf([NATIVE, "Second page with plenty of ordinary readable words."])
    standard = extract_text_from_upload("a.pdf", pdf)
    enhanced = build_document_extractor(ExtractionSettings())("a.pdf", pdf)
    assert [(p.page_number, p.text) for p in enhanced.pages] == [(p.page_number, p.text) for p in standard.pages]
    assert enhanced.page_count == standard.page_count and enhanced.warnings == standard.warnings
    assert enhanced.extraction_report["extractor"] == "disabled"


def test_disabled_extractor_keeps_the_standard_image_only_rejection():
    pdf = make_pdf([None, None])
    with pytest.raises(ExtractionError) as std:
        extract_text_from_upload("s.pdf", pdf)
    with pytest.raises(ExtractionError) as enh:
        build_document_extractor(ExtractionSettings())("s.pdf", pdf)
    assert enh.value.user_message == std.value.user_message and enh.value.http_status == 422


def test_disabled_extractor_reports_unreadable_pages_of_a_mixed_pdf():
    doc = build_document_extractor(ExtractionSettings())("m.pdf", make_pdf([NATIVE, None, NATIVE]))
    assert [p.page_number for p in doc.pages] == [1, 3]
    assert any("Page 2 contains no extractable text and was not indexed" in w for w in doc.warnings)
    assert doc.extraction_report["pages_without_text"] == [2]


def test_docx_and_text_never_reach_the_worker():
    client = FakeClient()
    extractor = EnhancedPdfExtractor(settings=ENABLED, mode="enhanced", client=client)
    docx = extractor("d.docx", make_docx(["Paragraph one"]))
    text = extractor("t.txt", b"plain text body")
    assert docx.document_type == "docx" and text.document_type == "text" and client.calls == []


# --- worker enabled: OCR and tables -----------------------------------------------


def test_scanned_page_is_ocred_with_its_true_physical_page_number():
    pdf = make_pdf([NATIVE, None, NATIVE])
    client = FakeClient(pages={2: _ocr(2, "Recovered scanned text for the middle page.")})
    doc = EnhancedPdfExtractor(settings=ENABLED, client=client)("mixed.pdf", pdf)
    assert client.calls[0]["ocr"] == [2] and client.calls[0]["table"] == []
    assert client.calls[0]["bytes"] == pdf, "the worker must receive the ORIGINAL bytes"
    assert [p.page_number for p in doc.pages] == [1, 2, 3]
    assert doc.pages[1].text == "Recovered scanned text for the middle page."
    assert doc.extraction_report["ocr_pages"] == [2] and doc.extraction_report["native_pages"] == [1, 3]
    assert any("Page 2 had no usable embedded text" in w and "88%" in w for w in doc.warnings)
    assert doc.extraction_report["recovered_text_preview"] == "Recovered scanned text for the middle page."


def test_recovered_text_preview_is_bounded_and_excludes_native_pages():
    long_text = "word " * 2000
    client = FakeClient(pages={2: _ocr(2, long_text)})
    doc = EnhancedPdfExtractor(settings=ENABLED, client=client)("mixed.pdf", make_pdf([NATIVE, None]))
    preview = doc.extraction_report["recovered_text_preview"]
    assert len(preview) == 2500 and preview.startswith("word word")
    assert NATIVE.split()[0] not in preview


def test_native_only_pdf_does_not_call_the_worker():
    client = FakeClient()
    doc = EnhancedPdfExtractor(settings=ENABLED, client=client)("n.pdf", make_pdf([NATIVE]))
    assert client.calls == [] and doc.extraction_report["extractor"] == "not_needed"


def test_low_confidence_and_empty_ocr_are_reported_not_hidden():
    pdf = make_pdf([None, None, NATIVE])
    client = FakeClient(pages={1: _ocr(1, "faint text here", conf=40.0), 2: _ocr(2, "   ")})
    doc = EnhancedPdfExtractor(settings=ENABLED, client=client)("f.pdf", pdf)
    assert [p.page_number for p in doc.pages] == [1, 3]
    assert any("low on page 1" in w for w in doc.warnings)
    assert any("Page 2 contained no readable text" in w for w in doc.warnings)


def test_table_page_with_usable_native_text_keeps_it_and_adds_row_lines():
    pdf = make_pdf([NUMERIC])
    client = FakeClient(pages={1: _table_page(1, "ocr text not needed", [_TABLE])})
    doc = EnhancedPdfExtractor(settings=ENABLED, mode="enhanced", client=client)("t.pdf", pdf)
    text = doc.pages[0].text
    assert text.startswith(NUMERIC.split()[0]) and "ocr text not needed" not in text
    assert "[Table 1, page 1] Line item | Period A: (42)" in text
    assert doc.extraction_report["table_pages"] == [1] and doc.extraction_report["tables_found"] == 1


@pytest.mark.parametrize("empty_table", [
    {"num_rows": 0, "num_cols": 0, "cells": [], "caption": "Under heading: Statement of results"},
    {"num_rows": 2, "num_cols": 2, "cells": [], "caption": ""},
    {"num_rows": 1, "num_cols": 1, "cells": [{"text": "  ", "start_row": 0, "end_row": 1, "start_col": 0, "end_col": 1}]},
    {"num_rows": True, "num_cols": 1, "cells": [{"text": "x", "start_row": 0, "end_row": 1, "start_col": 0, "end_col": 1}]},
])
def test_empty_table_object_is_unusable_never_counted_and_page_text_is_kept(empty_table):
    pdf = make_pdf([LETTER_SPACED])
    client = FakeClient(pages={1: _table_page(1, "Statement of results\nRevenue 1,200 1,050", [empty_table])})
    doc = EnhancedPdfExtractor(settings=ENABLED, mode="enhanced", client=client)("e.pdf", pdf)
    report = doc.extraction_report
    assert report["tables_found"] == 0 and report["table_pages"] == [] and report["tables_unusable"] == 1
    assert doc.pages[0].text == "Statement of results\nRevenue 1,200 1,050", "worker text kept, no table lines"
    assert "[Table" not in doc.pages[0].text
    assert not any(w.startswith("Tables were extracted") for w in doc.warnings)
    assert any("had no usable structure" in w for w in doc.warnings)
    assert report["ocr_pages"] == [1], "the kept text came from the worker, not the broken native glyphs"


def test_usable_and_empty_tables_on_one_page_count_only_the_usable_one():
    empty = {"num_rows": 0, "num_cols": 0, "cells": [], "caption": "Under heading: Notes"}
    client = FakeClient(pages={1: _table_page(1, "", [empty, _TABLE])})
    doc = EnhancedPdfExtractor(settings=ENABLED, mode="enhanced", client=client)("u.pdf", make_pdf([NUMERIC]))
    assert doc.extraction_report["tables_found"] == 1 and doc.extraction_report["tables_unusable"] == 1
    assert "[Table 2, page 1] Line item | Period A: (42)" in doc.pages[0].text
    assert "Under heading: Notes" not in doc.pages[0].text


def test_worker_table_diagnostics_are_compacted_into_the_report():
    diagnostics = {
        "conversion_status": "success", "error_count": 0, "tables_detected": 1, "tables_usable": 0,
        "errors": [{"module": "table", "message": "m" * 999, "extra": "dropped"}],
        "page": {"textline_cells": 14, "ocr_textline_cells": 14, "table_stage": "predicted", "secret": "x"},
        "unusable_tables": [{"reason": "zero_dimensions", "num_rows": 0, "num_cols": 0, "cells": 0,
                             "kept_text_chars": 120, "raw": "dropped"}] * 30,
        "unknown_key": {"nested": "dropped"},
    }
    page = WorkerPage(page_number=1, method="table", text="kept words", mean_confidence=None, tables=(),
                      warnings=("a table was detected but its structure came back empty",), diagnostics=None)
    payload = {"schema": WORKER_SCHEMA, "pages": [{"page_number": 1, "method": "table", "text": "kept words",
                                                   "tables": [], "warnings": list(page.warnings),
                                                   "diagnostics": diagnostics}]}
    parsed = parse_worker_response(payload, requested={1})
    compact = parsed.pages[1].diagnostics
    assert set(compact) == {"conversion_status", "error_count", "tables_detected", "tables_usable", "errors",
                            "textline_cells", "ocr_textline_cells", "table_stage", "unusable_tables"}
    assert len(compact["errors"][0]["message"]) == 200 and "extra" not in compact["errors"][0]
    assert len(compact["unusable_tables"]) == 10 and "raw" not in compact["unusable_tables"][0]
    client = FakeClient(pages={1: parsed.pages[1]})
    doc = EnhancedPdfExtractor(settings=ENABLED, mode="enhanced", client=client)("d.pdf", make_pdf([LETTER_SPACED]))
    report = doc.extraction_report
    assert report["table_diagnostics"]["1"]["conversion_status"] == "success"
    assert report["tables_unusable"] == 10 and report["tables_found"] == 0
    assert any("Page 1: a table was detected but its structure came back empty" in w for w in doc.warnings)
    json.dumps(report)


@pytest.mark.parametrize("value,expected", [(None, False), ("", False), ("0", False), ("no", False),
                                            ("1", True), ("true", True), ("YES", True)])
def test_table_extraction_switch_is_off_unless_explicitly_enabled(monkeypatch, value, expected):
    monkeypatch.delenv("EXTRACTOR_ENABLE_TABLES", raising=False)
    if value is not None:
        monkeypatch.setenv("EXTRACTOR_ENABLE_TABLES", value)
    assert ExtractionSettings.from_env().tables_enabled is expected
    assert ExtractionSettings().tables_enabled is False


def test_profile_reason_and_worker_stats_are_kept_bounded():
    payload = {
        "schema": WORKER_SCHEMA,
        "pages": [{"page_number": 1, "method": "table", "text": "",
                   "warnings": ["this page has no embedded text layer, so its table structure was not extracted"],
                   "diagnostics": {"profile": None, "reason": "no_native_text", "native_alnum_chars": 0}},
                  {"page_number": 2, "method": "table", "text": "Revenue 1,200",
                   "diagnostics": {"profile": "native_text_tables_v1", "native_alnum_chars": 812,
                                   "tables_detected": 1, "tables_usable": 1}}],
        "stats": {"elapsed_s": 41.25, "peak_rss_bytes": 2_000_000_000, "worker_respawned": False,
                  "timings": {"table_page_2_s": 30.1}, "extra": "dropped"},
    }
    parsed = parse_worker_response(payload, requested={1, 2})
    assert parsed.pages[1].diagnostics == {"reason": "no_native_text", "native_alnum_chars": 0}
    assert parsed.pages[2].diagnostics["profile"] == "native_text_tables_v1"
    assert parsed.stats == {"elapsed_s": 41.25, "peak_rss_bytes": 2_000_000_000, "worker_respawned": False}
    client = FakeClient(pages=dict(parsed.pages))
    client.extract = lambda pdf_bytes, **kw: WorkerResult(pages=dict(parsed.pages), engine={}, stats=parsed.stats)
    doc = EnhancedPdfExtractor(settings=ENABLED, mode="enhanced", client=client)(
        "w.pdf", make_pdf([LETTER_SPACED, NUMERIC]))
    assert doc.extraction_report["worker_stats"] == parsed.stats
    assert doc.extraction_report["table_diagnostics"]["1"]["reason"] == "no_native_text"
    assert any("Page 1: this page has no embedded text layer" in w for w in doc.warnings)
    assert parse_worker_response({"schema": WORKER_SCHEMA, "pages": [], "stats": "bad"}, requested={1}).stats is None


def test_non_dict_diagnostics_are_ignored():
    payload = {"schema": WORKER_SCHEMA, "pages": [{"page_number": 1, "method": "table", "text": "t",
                                                   "diagnostics": ["not", "a", "dict"]}]}
    assert parse_worker_response(payload, requested={1}).pages[1].diagnostics is None


def test_letter_spaced_table_page_uses_worker_text_instead_of_broken_glyph_runs():
    pdf = make_pdf([LETTER_SPACED])
    client = FakeClient(pages={1: _table_page(1, "Total assets and liabilities", [_TABLE])})
    doc = EnhancedPdfExtractor(settings=ENABLED, mode="enhanced", client=client)("ls.pdf", pdf)
    assert client.calls[0]["table"] == [1]
    assert doc.pages[0].text.startswith("Total assets and liabilities")
    assert "T o t a l" not in doc.pages[0].text


# --- worker failures: never a false success --------------------------------------


def test_all_scanned_with_worker_down_is_an_infrastructure_failure():
    pdf = make_pdf([None])
    extractor = EnhancedPdfExtractor(settings=ENABLED, client=FakeClient(error=ExtractorUnavailable("down")), log=lambda m: None)
    with pytest.raises(ExtractionError) as exc:
        extractor("s.pdf", pdf)
    assert exc.value.http_status == 503


def test_mixed_pdf_with_worker_down_indexes_native_pages_and_says_what_was_lost():
    pdf = make_pdf([NATIVE, None])
    extractor = EnhancedPdfExtractor(settings=ENABLED, client=FakeClient(error=ExtractorUnavailable("down")), log=lambda m: None)
    doc = extractor("m.pdf", pdf)
    assert [p.page_number for p in doc.pages] == [1]
    assert doc.extraction_report["extractor"] == "unavailable" and doc.extraction_report["pages_without_text"] == [2]
    assert any("Page 2 could not be read" in w for w in doc.warnings)


def test_malformed_worker_response_is_rejected():
    with pytest.raises(ExtractorError):
        parse_worker_response({"schema": "other", "pages": []}, requested={1})
    with pytest.raises(ExtractorError):
        parse_worker_response({"schema": WORKER_SCHEMA, "pages": [{"page_number": 5, "method": "ocr", "text": "x"}]},
                              requested={1})
    with pytest.raises(ExtractorError):
        parse_worker_response({"schema": WORKER_SCHEMA, "pages": [{"page_number": 1, "method": "magic", "text": "x"}]},
                              requested={1})


# --- HTTP client (fake opener, no network) ---------------------------------------


class _FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _FakeOpener:
    def __init__(self, body=None, error=None):
        self.body, self.error, self.requests = body, error, []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        if self.error is not None:
            raise self.error
        return _FakeResponse(self.body)


def _good_body():
    return json.dumps({"schema": WORKER_SCHEMA, "pages": [
        {"page_number": 2, "method": "ocr", "text": "hello", "mean_confidence": 77.5, "tables": [], "warnings": []}],
        "engine": {"tesseract": "5.x"}}).encode()


def test_http_client_posts_original_bytes_with_token_and_page_lists():
    opener = _FakeOpener(_good_body())
    client = HttpExtractorClient(ENABLED, opener_factory=lambda: opener)
    result = client.extract(b"%PDF-1.4 original", ocr_pages=[2], table_pages=[], timeout_s=120)
    request, timeout = opener.requests[0]
    assert request.data == b"%PDF-1.4 original" and request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer " + "t" * 32
    assert "ocr_pages=2" in request.full_url and "timeout_s=120" in request.full_url
    assert timeout == 150 and result.pages[2].text == "hello"


@pytest.mark.parametrize("code,kind", [(503, ExtractorUnavailable), (429, ExtractorUnavailable),
                                       (401, ExtractorUnavailable), (422, ExtractorRejected)])
def test_http_errors_map_to_typed_failures_without_leaking_detail(code, kind):
    err = urllib.error.HTTPError("http://worker.invalid", code, "x", {}, io.BytesIO(b"secret detail"))
    client = HttpExtractorClient(ENABLED, opener_factory=lambda: _FakeOpener(error=err))
    with pytest.raises(kind) as exc:
        client.extract(b"%PDF-", ocr_pages=[1], table_pages=[], timeout_s=60)
    assert "secret" not in str(exc.value)


def test_unreachable_worker_and_oversized_response_are_failures():
    client = HttpExtractorClient(ENABLED, opener_factory=lambda: _FakeOpener(error=ConnectionRefusedError()))
    with pytest.raises(ExtractorUnavailable):
        client.extract(b"%PDF-", ocr_pages=[1], table_pages=[], timeout_s=60)
    small = ExtractionSettings(worker_url="http://w.invalid", worker_token="y" * 16, max_response_bytes=10)
    client = HttpExtractorClient(small, opener_factory=lambda: _FakeOpener(_good_body()))
    with pytest.raises(ExtractorError):
        client.extract(b"%PDF-", ocr_pages=[2], table_pages=[], timeout_s=60)


def test_settings_from_env_require_url_and_token_and_never_repr_the_token(monkeypatch):
    monkeypatch.setenv("EXTRACTOR_URL", "http://synergereader-extractor:8091")
    monkeypatch.setenv("EXTRACTOR_TOKEN", "super-secret-token-value")
    settings = ExtractionSettings.from_env()
    assert settings.enabled and "super-secret" not in repr(settings)
    monkeypatch.setenv("EXTRACTOR_URL", "file:///etc/passwd")
    assert not ExtractionSettings.from_env().enabled


# --- through the real ingestion service (fake DB + embedder) ---------------------


class _Cursor:
    def __init__(self, log):
        self.log = log

    def execute(self, sql, params):
        self.log.append((" ".join(sql.split()), params))

    def fetchone(self):
        return (501,)

    def close(self):
        pass


class _Connection:
    autocommit = False

    def __init__(self, log):
        self.log, self.committed = log, False

    def cursor(self):
        return _Cursor(self.log)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.log.append(("ROLLBACK", None))

    def close(self):
        pass


class _Embedder:
    def __init__(self, dim, fail=False):
        self.dim, self.fail = dim, fail

    def embed_documents(self, texts):
        if self.fail:
            from ollama_embedding_provider import EmbeddingProviderError
            raise EmbeddingProviderError("down")
        return [[0.1] * self.dim for _ in texts]


def _service(extractor, log, fail_embed=False):
    profile = resolve_embedding_profile({})
    return DocumentIngestionService(
        connection_factory=lambda: _Connection(log),
        embedding_provider=_Embedder(profile.dimension, fail=fail_embed),
        embedding_profile=profile,
        extractor=extractor,
        json_adapter=lambda d: d,
    )


def test_ocr_text_is_indexed_with_physical_page_locators_and_coverage():
    pdf = make_pdf([None, NATIVE])
    client = FakeClient(pages={1: _ocr(1, "Scanned invoice words recovered by OCR on the first page.")})
    log = []
    result = _service(EnhancedPdfExtractor(settings=ENABLED, client=client), log).ingest(
        UploadDocument(filename="scan.pdf", content=pdf, uploader_id="owner-1"))
    assert result.status is ResultStatus.INDEXED
    chunk_rows = [p for sql, p in log if sql.startswith("INSERT INTO document_chunks")]
    pages_by_chunk = [(row[4], row[5], row[6]["page_numbers"]) for row in chunk_rows]
    assert pages_by_chunk[0][2][0] == 1 and any(2 in pg for _, _, pg in pages_by_chunk)
    payload = result.to_dict()
    assert payload["extraction"]["ocr_pages"] == [1]
    assert payload["locator_coverage"]["page_numbers_indexed"] == [1, 2]
    assert any("Page 1 had no usable embedded text" in w for w in payload["warnings"])


def test_embedding_failure_after_ocr_writes_nothing():
    pdf = make_pdf([None])
    client = FakeClient(pages={1: _ocr(1, "Recovered text that will never be stored.")})
    log = []
    result = _service(EnhancedPdfExtractor(settings=ENABLED, client=client), log, fail_embed=True).ingest(
        UploadDocument(filename="scan.pdf", content=pdf, uploader_id="owner-1"))
    assert result.status is ResultStatus.FAILED and result.error_category is ErrorCategory.EMBEDDING_UNAVAILABLE
    assert log == [], "no database statement may run when embedding fails"


def test_worker_outage_on_a_fully_scanned_pdf_is_reported_as_parser_unavailable():
    pdf = make_pdf([None])
    extractor = EnhancedPdfExtractor(settings=ENABLED, client=FakeClient(error=ExtractorUnavailable("down")), log=lambda m: None)
    log = []
    result = _service(extractor, log).ingest(UploadDocument(filename="scan.pdf", content=pdf, uploader_id="o"))
    assert result.status is ResultStatus.FAILED and result.error_category is ErrorCategory.PARSER_UNAVAILABLE
    assert log == []
