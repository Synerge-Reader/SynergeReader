"""Enhanced PDF extraction: native text first, OCR and table structure where needed.

This module plugs into ``DocumentIngestionService`` through its existing
``extractor(filename, content) -> ParsedDocument`` seam. It never touches the
database, the embedder or the generation stack, and importing it opens no
connection and starts no thread.

How a PDF is processed
----------------------
1. Native text is read for every physical page with the standard parser.
2. Each page is classified from its own content only (no filenames, document
   ids or page numbers are special-cased):

   * usable native text is kept as it is;
   * a page with no text, almost no text, or letter-spaced glyph runs
     ("T o t a l  a s s e t s") is unusable and is sent for OCR;
   * in ``enhanced`` mode a page with a table signal (dense numeric tokens) is
     sent to the table path, which returns the model's cell grid; its rows are
     serialized with their emitted column headers (``table_serialization``).
3. OCR and table work runs in a separate, private extraction worker service
   reached over the internal network with a deployment-supplied token. When the
   worker is not configured, PDFs behave exactly as before (native text only),
   with an explicit warning for pages that could not be read.

DOCX and plain text never reach the worker. Physical 1-based page numbers are
preserved end to end, so chunk locators and citations keep pointing at the real
page.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence

from document_parser import (
    MAX_EXTRACTED_CHARS,
    ExtractionError,
    ParsedDocument,
    ParsedPage,
    _truncate_pages,
    detect_file_type,
    extract_pdf_native_pages,
    extract_text_from_upload,
)
from table_serialization import TableFormatError, serialize_worker_table

EXTRACTION_MODES = ("standard", "enhanced")
WORKER_SCHEMA = "synergereader.extractor.v1"
# The start of the OCR/table-recovered text, returned to the uploader in the upload result so the
# browser can offer suggestions for a scanned PDF whose local parse found no text layer.
RECOVERED_PREVIEW_CHARS = 2500

# Page-classification thresholds. They describe text quality, not documents.
MIN_USABLE_ALNUM_CHARS = 25
LETTER_SPACED_MIN_ALPHA_TOKENS = 30
LETTER_SPACED_SINGLE_CHAR_RATIO = 0.45
TABLE_MIN_NUMERIC_TOKENS = 12
TABLE_MIN_NUMERIC_RATIO = 0.12
LETTER_SPACED_TABLE_MIN_DIGITS = 20

_NUMERIC_TOKEN = re.compile(
    r"^[(\[]?[-+−–]?(?:[$€£]|RM)?\d[\d,]*(?:\.\d+)?%?[)\]]?[,;:]?$"
)
_DATE_TOKEN = re.compile(r"^\d{1,4}[./-]\d{1,2}[./-]\d{1,4}$")


class ExtractorError(Exception):
    """The extraction worker could not produce a result (never shown verbatim)."""


class ExtractorUnavailable(ExtractorError):
    pass


class ExtractorRejected(ExtractorError):
    pass


# ---------------------------------------------------------------- settings ----


def _env_int(name: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return min(max(value, low), high)


@dataclass(frozen=True)
class ExtractionSettings:
    """Worker connection and bounds. The token is never logged or returned."""

    worker_url: Optional[str] = None
    worker_token: Optional[str] = field(default=None, repr=False)
    max_ocr_pages: int = 40
    max_table_pages: int = 10
    connect_timeout_s: int = 10
    base_timeout_s: int = 60
    per_ocr_page_timeout_s: int = 25
    per_table_page_timeout_s: int = 150
    max_request_timeout_s: int = 1200
    max_response_bytes: int = 25 * 1024 * 1024
    # Table extraction ("enhanced" uploads) stays off until its qualification passes; an operator
    # turns it on explicitly with EXTRACTOR_ENABLE_TABLES=1. OCR for scanned pages does not depend on it.
    tables_enabled: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.worker_url and self.worker_token)

    @classmethod
    def from_env(cls) -> "ExtractionSettings":
        url = (os.environ.get("EXTRACTOR_URL") or "").strip() or None
        token = (os.environ.get("EXTRACTOR_TOKEN") or "").strip() or None
        if url is not None:
            parts = urllib.parse.urlsplit(url)
            if parts.scheme not in ("http", "https") or not parts.netloc:
                url = None
        return cls(
            worker_url=url,
            worker_token=token,
            max_ocr_pages=_env_int("EXTRACTOR_MAX_OCR_PAGES", 40, 1, 150),
            max_table_pages=_env_int("EXTRACTOR_MAX_TABLE_PAGES", 10, 1, 50),
            max_request_timeout_s=_env_int("EXTRACTOR_MAX_REQUEST_TIMEOUT_S", 1200, 60, 3600),
            tables_enabled=(os.environ.get("EXTRACTOR_ENABLE_TABLES") or "").strip().lower() in ("1", "true", "yes"),
        )

    def request_timeout(self, ocr_pages: int, table_pages: int) -> int:
        estimate = (
            self.base_timeout_s
            + self.per_ocr_page_timeout_s * ocr_pages
            + self.per_table_page_timeout_s * table_pages
        )
        return int(min(estimate, self.max_request_timeout_s))


# ------------------------------------------------------- page classification --


@dataclass(frozen=True)
class PageSignals:
    alnum_chars: int
    alpha_tokens: int
    single_char_alpha_tokens: int
    numeric_tokens: int
    token_count: int
    digit_chars: int

    @property
    def letter_spaced(self) -> bool:
        if self.alpha_tokens < LETTER_SPACED_MIN_ALPHA_TOKENS:
            return False
        return self.single_char_alpha_tokens / self.alpha_tokens >= LETTER_SPACED_SINGLE_CHAR_RATIO

    @property
    def usable(self) -> bool:
        return self.alnum_chars >= MIN_USABLE_ALNUM_CHARS and not self.letter_spaced

    @property
    def table_signal(self) -> bool:
        if self.letter_spaced and self.digit_chars >= LETTER_SPACED_TABLE_MIN_DIGITS:
            return True
        if self.token_count == 0:
            return False
        return (
            self.numeric_tokens >= TABLE_MIN_NUMERIC_TOKENS
            and self.numeric_tokens / self.token_count >= TABLE_MIN_NUMERIC_RATIO
        )


def page_signals(text: str) -> PageSignals:
    tokens = (text or "").split()
    alpha = [t for t in tokens if t.isalpha()]
    numeric = [t for t in tokens if _NUMERIC_TOKEN.match(t) or _DATE_TOKEN.match(t)]
    return PageSignals(
        alnum_chars=sum(ch.isalnum() for ch in (text or "")),
        alpha_tokens=len(alpha),
        single_char_alpha_tokens=sum(len(t) == 1 for t in alpha),
        numeric_tokens=len(numeric),
        token_count=len(tokens),
        digit_chars=sum(ch.isdigit() for ch in (text or "")),
    )


@dataclass
class PagePlan:
    page_number: int
    native_text: str
    signals: PageSignals
    route: str = "native"  # native | ocr | table | skipped


def plan_pages(
    pages: Sequence[ParsedPage],
    *,
    mode: str,
    settings: ExtractionSettings,
) -> tuple[list[PagePlan], list[int], list[int]]:
    """Route every page; returns (plans, ocr_overflow_pages, table_overflow_pages)."""
    plans = [
        PagePlan(page_number=p.page_number, native_text=p.text or "", signals=page_signals(p.text or ""))
        for p in pages
    ]
    table_overflow: list[int] = []
    if mode == "enhanced":
        candidates = [pl for pl in plans if pl.signals.table_signal]
        ranked = sorted(candidates, key=lambda pl: (-pl.signals.numeric_tokens, pl.page_number))
        chosen = {pl.page_number for pl in ranked[: settings.max_table_pages]}
        for pl in candidates:
            if pl.page_number in chosen:
                pl.route = "table"
            else:
                table_overflow.append(pl.page_number)
    for pl in plans:
        if pl.route == "native" and not pl.signals.usable:
            pl.route = "ocr"
    ocr_plans = [pl for pl in plans if pl.route == "ocr"]
    ocr_overflow: list[int] = []
    for pl in ocr_plans[settings.max_ocr_pages:]:
        pl.route = "skipped"
        ocr_overflow.append(pl.page_number)
    return plans, ocr_overflow, sorted(table_overflow)


def page_ranges(numbers: Iterable[int]) -> str:
    ordered = sorted(set(numbers))
    if not ordered:
        return ""
    parts: list[str] = []
    start = prev = ordered[0]
    for n in ordered[1:]:
        if n == prev + 1:
            prev = n
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = n
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(parts)


def _pages_word(numbers: Sequence[int]) -> str:
    return "Page" if len(set(numbers)) == 1 else "Pages"


def _agree(numbers: Sequence[int], singular: str, plural: str) -> str:
    return singular if len(set(numbers)) == 1 else plural


# ------------------------------------------------------------ worker client --


@dataclass(frozen=True)
class WorkerPage:
    page_number: int
    method: str
    text: str
    mean_confidence: Optional[float]
    tables: tuple[dict, ...]
    warnings: tuple[str, ...]
    diagnostics: Optional[dict] = None


@dataclass(frozen=True)
class WorkerResult:
    pages: dict[int, WorkerPage]
    engine: dict
    stats: Optional[dict] = None


def compact_worker_stats(raw: object) -> Optional[dict]:
    """The worker's own job measurements (elapsed seconds, peak RSS), typed and bounded."""
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {}
    elapsed = raw.get("elapsed_s")
    if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and 0 <= elapsed < 1e6:
        out["elapsed_s"] = round(float(elapsed), 3)
    peak = _bounded_int(raw.get("peak_rss_bytes"), 1 << 40)
    if peak is not None:
        out["peak_rss_bytes"] = peak
    if isinstance(raw.get("worker_respawned"), bool):
        out["worker_respawned"] = raw["worker_respawned"]
    return out or None


class ExtractorClient(Protocol):
    def extract(self, pdf_bytes: bytes, *, ocr_pages: Sequence[int], table_pages: Sequence[int],
                timeout_s: int) -> WorkerResult: ...


def _clean_warning(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())[:300]
    return text or None


def _bounded_int(value: object, high: int = 1_000_000) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return min(value, high)


def _short(value: object, limit: int) -> Optional[str]:
    return " ".join(value.split())[:limit] if isinstance(value, str) else None


def compact_worker_diagnostics(raw: object) -> Optional[dict]:
    """Keep a small, typed subset of the worker's per-page table diagnostics for the upload report.

    Unknown keys are dropped, strings are truncated and lists are capped, so a misbehaving worker
    cannot inflate the report. Returns None when nothing usable was supplied.
    """
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {}
    status = _short(raw.get("conversion_status"), 40)
    if status:
        out["conversion_status"] = status
    for key in ("profile", "reason"):
        value = _short(raw.get(key), 40)
        if value:
            out[key] = value
    native = _bounded_int(raw.get("native_alnum_chars"))
    if native is not None:
        out["native_alnum_chars"] = native
    for key in ("error_count", "tables_detected", "tables_usable"):
        number = _bounded_int(raw.get(key), 1000)
        if number is not None:
            out[key] = number
    errors = raw.get("errors")
    if isinstance(errors, list):
        out["errors"] = [
            {"module": _short(e.get("module"), 80), "message": _short(e.get("message"), 200)}
            for e in errors[:5] if isinstance(e, dict)
        ]
    page = raw.get("page")
    if isinstance(page, dict):
        for key in ("textline_cells", "ocr_textline_cells"):
            number = _bounded_int(page.get(key))
            if number is not None:
                out[key] = number
        stage = _short(page.get("table_stage"), 40)
        if stage:
            out["table_stage"] = stage
    unusable = raw.get("unusable_tables")
    if isinstance(unusable, list):
        out["unusable_tables"] = [
            {"reason": _short(u.get("reason"), 40), "num_rows": _bounded_int(u.get("num_rows"), 2000),
             "num_cols": _bounded_int(u.get("num_cols"), 200), "cells": _bounded_int(u.get("cells")),
             "kept_text_chars": _bounded_int(u.get("kept_text_chars"))}
            for u in unusable[:10] if isinstance(u, dict)
        ]
    return out or None


def worker_table_usable(table: dict) -> bool:
    """Structure is usable only with >= 1 row, >= 1 column and at least one cell carrying text."""
    rows, cols, cells = table.get("num_rows"), table.get("num_cols"), table.get("cells")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (rows, cols)) or rows < 1 or cols < 1:
        return False
    if not isinstance(cells, list) or not cells:
        return False
    return any(isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip() for c in cells)


def parse_worker_response(payload: Any, *, requested: set[int]) -> WorkerResult:
    """Strictly validate the worker JSON; anything malformed is an ExtractorError."""
    if not isinstance(payload, dict) or payload.get("schema") != WORKER_SCHEMA:
        raise ExtractorError("unexpected worker schema")
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list) or len(raw_pages) > len(requested):
        raise ExtractorError("worker pages missing or too many")
    pages: dict[int, WorkerPage] = {}
    for raw in raw_pages:
        if not isinstance(raw, dict):
            raise ExtractorError("worker page must be an object")
        number = raw.get("page_number")
        if isinstance(number, bool) or not isinstance(number, int) or number not in requested or number in pages:
            raise ExtractorError("worker returned an unexpected page")
        method = raw.get("method")
        if method not in ("ocr", "table"):
            raise ExtractorError("worker returned an unknown method")
        text = raw.get("text")
        if not isinstance(text, str) or len(text) > 400_000:
            raise ExtractorError("worker page text invalid")
        conf = raw.get("mean_confidence")
        if conf is not None and (isinstance(conf, bool) or not isinstance(conf, (int, float))):
            raise ExtractorError("worker confidence invalid")
        tables = raw.get("tables", [])
        if not isinstance(tables, list) or len(tables) > 50 or not all(isinstance(t, dict) for t in tables):
            raise ExtractorError("worker tables invalid")
        warnings = raw.get("warnings", [])
        if not isinstance(warnings, list):
            raise ExtractorError("worker warnings invalid")
        pages[number] = WorkerPage(
            page_number=number,
            method=method,
            text=text,
            mean_confidence=float(conf) if conf is not None else None,
            tables=tuple(tables),
            warnings=tuple(w for w in (_clean_warning(x) for x in warnings[:20]) if w),
            diagnostics=compact_worker_diagnostics(raw.get("diagnostics")),
        )
    engine = payload.get("engine") if isinstance(payload.get("engine"), dict) else {}
    return WorkerResult(pages=pages, engine=engine, stats=compact_worker_stats(payload.get("stats")))


class HttpExtractorClient:
    """POSTs the original PDF bytes to the private worker; no proxies, no retries."""

    def __init__(self, settings: ExtractionSettings,
                 opener_factory: Callable[[], Any] = lambda: urllib.request.build_opener(urllib.request.ProxyHandler({}))):
        if not settings.enabled:
            raise ValueError("extraction worker is not configured")
        self._settings = settings
        self._opener_factory = opener_factory

    def extract(self, pdf_bytes: bytes, *, ocr_pages: Sequence[int], table_pages: Sequence[int],
                timeout_s: int) -> WorkerResult:
        query = urllib.parse.urlencode({
            "ocr_pages": ",".join(str(p) for p in sorted(set(ocr_pages))),
            "table_pages": ",".join(str(p) for p in sorted(set(table_pages))),
            "timeout_s": str(int(timeout_s)),
        })
        url = self._settings.worker_url.rstrip("/") + "/v1/extract?" + query
        request = urllib.request.Request(
            url,
            data=pdf_bytes,
            method="POST",
            headers={
                "Content-Type": "application/pdf",
                "Accept": "application/json",
                "Authorization": f"Bearer {self._settings.worker_token}",
            },
        )
        # The worker enforces its own wall clock and returns 504; the socket
        # timeout adds a margin so a dead worker can never hang an ingestion.
        socket_timeout = timeout_s + 30
        try:
            with self._opener_factory().open(request, timeout=socket_timeout) as response:
                body = response.read(self._settings.max_response_bytes + 1)
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
            if code in (400, 413, 415, 422):
                raise ExtractorRejected(f"worker rejected the request ({code})") from None
            raise ExtractorUnavailable(f"worker returned HTTP {code}") from None
        except Exception as exc:  # connection refused, DNS, timeout, reset
            raise ExtractorUnavailable(f"worker unreachable: {type(exc).__name__}") from None
        if len(body) > self._settings.max_response_bytes:
            raise ExtractorError("worker response exceeds the size limit")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise ExtractorError("worker response is not JSON") from None
        return parse_worker_response(payload, requested=set(ocr_pages) | set(table_pages))


# ---------------------------------------------------------------- extractor --


@dataclass
class EnhancedPdfExtractor:
    """``extractor(filename, content) -> ParsedDocument`` for DocumentIngestionService."""

    settings: ExtractionSettings
    mode: str = "standard"
    client: Optional[ExtractorClient] = None
    native_extractor: Callable[[str, bytes], ParsedDocument] = extract_text_from_upload
    native_pages_reader: Callable[[bytes], tuple[int, list[ParsedPage]]] = extract_pdf_native_pages
    log: Callable[[str], None] = print

    def __post_init__(self) -> None:
        if self.mode not in EXTRACTION_MODES:
            raise ValueError(f"unknown extraction mode {self.mode!r}")

    def __call__(self, filename: str, content: bytes) -> ParsedDocument:
        if not content or detect_file_type(filename, content) != "pdf":
            # DOCX, text, empty and unknown inputs keep the standard parser's
            # exact behaviour and error messages.
            return self.native_extractor(filename, content)
        page_count, native_pages = self.native_pages_reader(content)
        if not self.settings.enabled or self.client is None:
            return self._native_only(page_count, native_pages, reason="disabled")
        plans, ocr_overflow, table_overflow = plan_pages(native_pages, mode=self.mode, settings=self.settings)
        ocr_pages = [pl.page_number for pl in plans if pl.route == "ocr"]
        table_pages = [pl.page_number for pl in plans if pl.route == "table"]
        if not ocr_pages and not table_pages:
            return self._assemble(page_count, plans, None, ocr_overflow, table_overflow, extractor="not_needed")
        timeout_s = self.settings.request_timeout(len(ocr_pages), len(table_pages))
        try:
            result = self.client.extract(content, ocr_pages=ocr_pages, table_pages=table_pages, timeout_s=timeout_s)
        except ExtractorError as exc:
            # Fixed-shape diagnostic: category and class name only.
            self.log(f"[Extraction] worker unavailable: {type(exc).__name__}")
            return self._worker_failed(page_count, plans, ocr_overflow, table_overflow)
        return self._assemble(page_count, plans, result, ocr_overflow, table_overflow, extractor="available")

    # -- assembly ---------------------------------------------------------

    def _native_only(self, page_count: int, native_pages: Sequence[ParsedPage], *, reason: str) -> ParsedDocument:
        kept = [ParsedPage(page_number=p.page_number, text=p.text) for p in native_pages if p.text and p.text.strip()]
        empty = [p.page_number for p in native_pages if not (p.text and p.text.strip())]
        if not kept:
            if reason == "disabled":
                raise ExtractionError(
                    "This PDF appears to be scanned or image-based. Text extraction is not supported for image-only PDFs.",
                    422,
                )
            raise ExtractionError(
                "Scanned-page text recovery is temporarily unavailable. Please try again later.", 503
            )
        warnings: list[str] = []
        if empty:
            if reason == "disabled":
                warnings.append(
                    f"{_pages_word(empty)} {page_ranges(empty)} {_agree(empty, 'contains', 'contain')} no extractable "
                    f"text and {_agree(empty, 'was', 'were')} not indexed (scanned-page OCR is not enabled on this server)."
                )
            else:
                warnings.append(
                    f"{_pages_word(empty)} {page_ranges(empty)} could not be read because scanned-page text "
                    f"recovery is temporarily unavailable; {_agree(empty, 'it was', 'they were')} not indexed."
                )
        report = {
            "mode": self.mode,
            "extractor": reason,
            "pages_total": page_count,
            "native_pages": [p.page_number for p in kept],
            "ocr_pages": [],
            "table_pages": [],
            "tables_found": 0,
            "pages_without_text": empty,
            "pages_not_processed": [],
        }
        return self._finish(page_count, kept, warnings, report)

    def _worker_failed(self, page_count, plans, ocr_overflow, table_overflow) -> ParsedDocument:
        # Pages whose native text is usable survive; nothing is indexed from
        # pages the worker should have read, and the result says so.
        kept = [
            ParsedPage(page_number=pl.page_number, text=pl.native_text)
            for pl in plans
            if pl.native_text.strip() and (pl.route == "native" or (pl.route == "table" and pl.signals.usable))
        ]
        lost = [pl.page_number for pl in plans if pl.route in ("ocr", "skipped")
                or (pl.route == "table" and not pl.signals.usable)]
        table_native = [pl.page_number for pl in plans if pl.route == "table" and pl.signals.usable]
        if not kept:
            raise ExtractionError(
                "Scanned-page text recovery is temporarily unavailable. Please try again later.", 503
            )
        warnings = []
        if lost:
            warnings.append(
                f"{_pages_word(lost)} {page_ranges(lost)} could not be read because scanned-page text "
                f"recovery is temporarily unavailable; {_agree(lost, 'it was', 'they were')} not indexed."
            )
        if table_native:
            warnings.append(
                f"Table structure could not be extracted for {_pages_word(table_native).lower()} "
                f"{page_ranges(table_native)}; {_agree(table_native, 'its', 'their')} plain text was indexed instead."
            )
        report = {
            "mode": self.mode,
            "extractor": "unavailable",
            "pages_total": page_count,
            "native_pages": [p.page_number for p in kept],
            "ocr_pages": [],
            "table_pages": [],
            "tables_found": 0,
            "pages_without_text": lost,
            "pages_not_processed": sorted(set(ocr_overflow)),
        }
        return self._finish(page_count, kept, warnings, report)

    def _assemble(self, page_count, plans, result: Optional[WorkerResult], ocr_overflow, table_overflow,
                  *, extractor: str) -> ParsedDocument:
        pages: list[ParsedPage] = []
        native_used: list[int] = []
        ocr_used: list[int] = []
        table_used: list[int] = []
        no_text: list[int] = []
        low_conf: list[int] = []
        tables_found = 0
        table_errors: list[int] = []
        tables_unusable: list[int] = []
        worker_unusable = 0
        table_diagnostics: dict[str, dict] = {}
        confidences: dict[str, float] = {}
        worker_pages = result.pages if result else {}

        for pl in plans:
            number = pl.page_number
            if pl.route == "native":
                if pl.native_text.strip():
                    pages.append(ParsedPage(page_number=number, text=pl.native_text))
                    native_used.append(number)
                else:
                    no_text.append(number)
                continue
            if pl.route == "skipped":
                continue
            wp = worker_pages.get(number)
            if pl.route == "ocr":
                if wp is not None and wp.text.strip():
                    pages.append(ParsedPage(page_number=number, text=wp.text))
                    ocr_used.append(number)
                    if wp.mean_confidence is not None:
                        confidences[str(number)] = round(wp.mean_confidence, 1)
                        if wp.mean_confidence < 60:
                            low_conf.append(number)
                elif pl.native_text.strip():
                    pages.append(ParsedPage(page_number=number, text=pl.native_text))
                    native_used.append(number)
                else:
                    no_text.append(number)
                continue
            # table route
            base = pl.native_text if pl.signals.usable else (wp.text if wp is not None else "")
            blocks: list[str] = []
            if wp is not None:
                for index, table in enumerate(wp.tables, start=1):
                    # An empty table object (no rows, columns or cell text) is not structure:
                    # never index it as a table or count it as extracted.
                    if not worker_table_usable(table):
                        tables_unusable.append(number)
                        continue
                    try:
                        block = serialize_worker_table(table, number, index)
                    except TableFormatError:
                        table_errors.append(number)
                        continue
                    if block:
                        blocks.append(block)
                        tables_found += 1
                if wp.mean_confidence is not None:
                    confidences[str(number)] = round(wp.mean_confidence, 1)
                if wp.diagnostics:
                    table_diagnostics[str(number)] = wp.diagnostics
                    worker_unusable += len(wp.diagnostics.get("unusable_tables") or [])
            combined = "\n\n".join(part for part in [base.strip(), *blocks] if part)
            if combined:
                pages.append(ParsedPage(page_number=number, text=combined))
                if blocks:
                    table_used.append(number)
                elif pl.signals.usable:
                    native_used.append(number)
                else:
                    ocr_used.append(number)
            else:
                no_text.append(number)

        warnings: list[str] = []
        if ocr_used:
            conf_values = [confidences[str(n)] for n in ocr_used if str(n) in confidences]
            conf_note = f" (mean OCR confidence {sum(conf_values) / len(conf_values):.0f}%)" if conf_values else ""
            warnings.append(
                f"{_pages_word(ocr_used)} {page_ranges(ocr_used)} had no usable embedded text; "
                f"{_agree(ocr_used, 'its', 'their')} text was recovered with OCR{conf_note}. "
                "Check important figures against the original."
            )
        if low_conf:
            warnings.append(
                f"OCR confidence was low on {_pages_word(low_conf).lower()} {page_ranges(low_conf)}; "
                "recovered text there may contain errors."
            )
        if table_used:
            warnings.append(
                f"Tables were extracted on {_pages_word(table_used).lower()} {page_ranges(table_used)} "
                f"({tables_found} table{'s' if tables_found != 1 else ''}); each row is indexed with its column headings."
            )
        if table_errors:
            warnings.append(
                f"A table on {_pages_word(table_errors).lower()} {page_ranges(table_errors)} could not be "
                "structured and was skipped; the page text was still indexed."
            )
        if tables_unusable:
            pages_unusable = sorted(set(tables_unusable))
            warnings.append(
                f"A table on {_pages_word(pages_unusable).lower()} {page_ranges(pages_unusable)} had no usable "
                "structure (no rows, columns or cell text), so it was not indexed as a table."
            )
        if no_text:
            warnings.append(
                f"{_pages_word(no_text)} {page_ranges(no_text)} contained no readable text and "
                f"{_agree(no_text, 'was', 'were')} not indexed."
            )
        if ocr_overflow:
            warnings.append(
                f"Only the first {self.settings.max_ocr_pages} scanned pages were processed; "
                f"{_pages_word(ocr_overflow).lower()} {page_ranges(ocr_overflow)} "
                f"{_agree(ocr_overflow, 'was', 'were')} not indexed."
            )
        if table_overflow:
            warnings.append(
                f"Table extraction is limited to {self.settings.max_table_pages} pages per document; "
                f"{_pages_word(table_overflow).lower()} {page_ranges(table_overflow)} "
                f"{_agree(table_overflow, 'was', 'were')} indexed as plain text."
            )
        if result is not None:
            for number in sorted(worker_pages):
                for w in worker_pages[number].warnings:
                    warnings.append(f"Page {number}: {w}")

        if not pages:
            # Worded to map onto EMPTY_DOCUMENT ("contains no indexable text"),
            # not IMAGE_ONLY_PDF, whose safe message says OCR is unsupported.
            raise ExtractionError(
                "The PDF is effectively empty: no readable text could be recovered, even with OCR.", 422
            )
        report = {
            "mode": self.mode,
            "extractor": extractor,
            "pages_total": page_count,
            "native_pages": native_used,
            "ocr_pages": ocr_used,
            "table_pages": table_used,
            "tables_found": tables_found,
            "tables_unusable": len(tables_unusable) + worker_unusable,
            "pages_without_text": no_text,
            "pages_not_processed": sorted(set(ocr_overflow)),
            "ocr_mean_confidence": confidences,
        }
        if result is not None and result.engine:
            report["engine"] = {k: v for k, v in result.engine.items() if isinstance(v, (str, int, float))}
        if result is not None and result.stats:
            report["worker_stats"] = dict(result.stats)
        if table_diagnostics:
            report["table_diagnostics"] = table_diagnostics
        ordered = sorted(pages, key=lambda p: p.page_number)
        recovered_numbers = set(ocr_used) | set(table_used)
        report["recovered_text_preview"] = "\n\n".join(
            p.text for p in ordered if p.page_number in recovered_numbers
        )[:RECOVERED_PREVIEW_CHARS]
        return self._finish(page_count, ordered, warnings, report)

    @staticmethod
    def _finish(page_count: int, pages: list[ParsedPage], warnings: list[str], report: dict) -> ParsedDocument:
        pages, truncated = _truncate_pages(pages, MAX_EXTRACTED_CHARS)
        if truncated:
            warnings = list(warnings) + ["Document truncated to 300,000 characters."]
        text = "\n\n".join(p.text for p in pages)
        return ParsedDocument(
            pages=pages,
            document_type="pdf",
            page_count=page_count,
            char_count=len(text),
            truncated=truncated,
            warnings=list(warnings),
            extraction_report=report,
        )


def build_document_extractor(
    settings: ExtractionSettings,
    mode: str = "standard",
    client: Optional[ExtractorClient] = None,
) -> EnhancedPdfExtractor:
    """Compose the extractor; a configured worker gets the HTTP client."""
    if client is None and settings.enabled:
        client = HttpExtractorClient(settings)
    return EnhancedPdfExtractor(settings=settings, mode=mode, client=client)
