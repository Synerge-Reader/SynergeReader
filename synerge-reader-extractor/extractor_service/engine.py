"""Engine process: page OCR (Tesseract) and table structure (Docling) for requested pages only.

Runs inside the supervised child process. Libraries are imported lazily on the first job
so the parent never loads torch. Every job writes its PDF into a task-scoped temporary
directory that is removed when the job ends, whatever the outcome.

Table conversion runs a named profile built from the reviewed Docling D12 configuration (Heron
layout at a pinned revision, TableFormer V1 ACCURATE with cell matching, CPU, two threads, no
remote services or plugins). The service's table route uses only ``native_text_tables_v1``
(amendment TABLE-NATIVE-PROFILE-20261017: OCR off, so the parser's positioned native words are
the matching tokens), and only on pages that have an embedded text layer. The full-page-OCR
profile (amendment TABLE-OCR-PSM11-20261017) failed its Linux qualification on an image-only
table and is kept for diagnose_table only. The standalone page OCR used for scanned pages
(``_ocr_page``) is unchanged.

Every table page carries a bounded ``diagnostics`` record (conversion status, errors, stage
timings when recorded, confidence, layout/OCR/table-stage counts). A table object with no
rows, no columns or no cell text is not structure: it is never emitted as a table, a warning
says so, and the words in its area are kept as plain page text.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
from typing import Any, Optional

HERON_REPO = "docling-project/docling-layout-heron"
HERON_REVISION = "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8"
OCR_SCALE_300 = 300 / 72

# Configuration amendment TABLE-OCR-PSM11-20261017 (the only change to the reviewed D12 table
# configuration). Reviewed value: psm=3 (fully automatic page segmentation). Amended: psm=11
# (sparse text: find as much text as possible, in no particular order).
# Evidence: on the Linux candidate (lx-20261017T230526Z-2a37010a, table_diagnostic_synthetic) the
# ruled synthetic table was detected by layout (confidence 0.943) and TableFormer predicted all 12
# cell boxes, but Docling's psm=3 OCR returned a single text line (the heading) with orientation
# 0 degrees, so the table had no tokens to match and came back 0x0. psm=11 asks Tesseract for text
# anywhere on the page, including inside ruled cells. Nothing else changes: same models, lock,
# 300 DPI scale, OCR mode (FULL_PAGE), language, orientation detection (Docling's separate
# OSD-only reader) and TableFormer cell matching. Unverified until the Linux run passes.
REVIEWED_TABLE_OCR_PSM = 3
TABLE_OCR_PSM = 11
TABLE_OCR_AMENDMENT = "TABLE-OCR-PSM11-20261017"
# Result on Linux (lx-20261017T232308Z-83780ac6): FAILED. psm 11 recovered one wide "Net 900 770" box and
# the heading; TableFormer's 12 predicted cells matched one token. The image-only table profile is therefore
# NOT qualified and is never used by the service; it stays available to diagnose_table only.

# Table profiles. The service's table route uses PROFILE_NATIVE_TEXT only.
PROFILE_NATIVE_TEXT = "native_text_tables_v1"
PROFILE_OCR_PSM11 = "ocr_fullpage_psm11"
TABLE_PROFILES = (PROFILE_NATIVE_TEXT, PROFILE_OCR_PSM11)
# Configuration amendment TABLE-NATIVE-PROFILE-20261017 (profile native_text_tables_v1): the reviewed D12
# configuration with do_ocr=False -- same pinned docling-parse backend, Heron layout, TableFormer V1
# ACCURATE with cell matching, CPU threads, page-image scale and limits. With do_ocr=True and FULL_PAGE
# OCR, Docling skips native text-cell extraction altogether (resolve_skip_cell_extraction), so a page's
# embedded text never reached table matching. With OCR off, the parser's positioned native word cells
# are the tokens TableFormer's cells are matched with. Used only for pages that have an embedded text
# layer (generic capability: at least NATIVE_TEXT_MIN_ALNUM alphanumeric characters in pypdfium2's text
# for the page); pages without one get no table conversion and an explicit warning.
NATIVE_PROFILE_AMENDMENT = "TABLE-NATIVE-PROFILE-20261017"
NATIVE_TEXT_MIN_ALNUM = 25


class EngineInputError(Exception):
    """The request cannot be processed (unreadable PDF, page out of range)."""


def _clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


# -- table helpers (no Docling imports; unit-tested directly) ------------------------------

MAX_DIAG_ERRORS = 10
MAX_DIAG_MESSAGE = 300
REGION_MIN_OVERLAP = 0.5


def table_usability(num_rows: Any, num_cols: Any, cells: list) -> tuple[bool, str]:
    """A table is usable structure only with >= 1 row, >= 1 column and at least one cell with text."""
    if not isinstance(num_rows, int) or not isinstance(num_cols, int) or num_rows < 1 or num_cols < 1:
        return False, "zero_dimensions"
    if not cells:
        return False, "no_cells"
    if not any(str(c.get("text") or "").strip() for c in cells):
        return False, "cells_without_text"
    return True, "ok"


def cells_in_region(cells: list[tuple[str, tuple[float, float, float, float]]],
                    region: tuple[float, float, float, float]) -> list[str]:
    """Texts of the page text cells lying mostly inside ``region`` (top-left origin), top-to-bottom.

    A cell counts when it has text and at least half of its own area lies inside the region.
    """
    rl, rt, rr, rb = region
    picked = []
    for text, (l, t, r, b) in cells:
        if not text or not text.strip():
            continue
        area = max(0.0, r - l) * max(0.0, b - t)
        if area <= 0:
            continue
        iw = min(r, rr) - max(l, rl)
        ih = min(b, rb) - max(t, rt)
        if iw <= 0 or ih <= 0 or (iw * ih) / area < REGION_MIN_OVERLAP:
            continue
        picked.append((round(t, 1), l, text.strip()))
    picked.sort()
    return [text for _, _, text in picked]


def region_text(cells: list[tuple[str, tuple[float, float, float, float]]],
                region: tuple[float, float, float, float]) -> str:
    """Used to keep the words of a table area as plain text when its table structure is unusable."""
    return "\n".join(cells_in_region(cells, region))


def unusable_table_warning(reason: str, num_rows: Any, num_cols: Any, cells: int, kept_chars: int) -> str:
    detail = {"zero_dimensions": f"{num_rows} rows x {num_cols} columns",
              "no_cells": "no cells", "cells_without_text": f"{cells} cells, none with text"}.get(reason, reason)
    kept = ("the text in that area was kept as plain text" if kept_chars
            else "no text could be recovered from that area")
    return f"a table was detected but its structure came back empty ({detail}); it was not indexed as a table and {kept}"


def native_alnum_chars(pdf: Any, page_number: int) -> int:
    """Alphanumeric characters in the page's embedded text layer (pypdfium2), 0 when unreadable."""
    page = pdf[page_number - 1]
    try:
        textpage = page.get_textpage()
        try:
            text = textpage.get_text_range() or ""
        finally:
            textpage.close()
    except Exception:
        text = ""
    finally:
        page.close()
    return sum(ch.isalnum() for ch in text)


def no_native_text_table_page(page_number: int, alnum_chars: int) -> dict:
    """Table-route record for a page without an embedded text layer: no conversion is attempted, because
    the only table profile in service needs native text (scanned-table extraction is not qualified)."""
    return {"page_number": page_number, "method": "table", "text": "", "mean_confidence": None, "tables": [],
            "warnings": ["this page has no embedded text layer, so its table structure was not extracted "
                         "(table extraction currently works only on pages with embedded text)"],
            "diagnostics": {"profile": None, "reason": "no_native_text", "native_alnum_chars": alnum_chars,
                            "tables_detected": 0, "tables_usable": 0, "unusable_tables": []}}


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else round(number, 4)


def _enum_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(getattr(value, "value", value))[:80]


def summarize_conversion(result: Any, page_number: int) -> dict:
    """Bounded, JSON-safe record of one Docling conversion for one page (task-owned evidence).

    Reads only public result fields; every lookup is defensive so a missing field is recorded
    as absent rather than failing the job.
    """
    out: dict = {"conversion_status": _enum_text(getattr(result, "status", None))}
    errors = []
    for err in list(getattr(result, "errors", None) or [])[:MAX_DIAG_ERRORS]:
        errors.append({
            "component": _enum_text(getattr(err, "component_type", None)),
            "module": str(getattr(err, "module_name", "") or "")[:80],
            "message": str(getattr(err, "error_message", "") or "")[:MAX_DIAG_MESSAGE],
            "page_no": getattr(err, "page_no", None) if isinstance(getattr(err, "page_no", None), int) else None,
        })
    out["errors"] = errors
    out["error_count"] = len(list(getattr(result, "errors", None) or []))
    try:
        conf = result.confidence.pages.get(page_number)
        out["confidence"] = {name: _finite(getattr(conf, name, None))
                             for name in ("parse_score", "layout_score", "ocr_score", "table_score")}
    except Exception:
        out["confidence"] = None
    try:
        out["timings_s"] = {str(k): round(float(sum(v.times)), 3) for k, v in (result.timings or {}).items()}
    except Exception:
        out["timings_s"] = {}
    page = next((p for p in (getattr(result, "pages", None) or []) if getattr(p, "page_no", None) == page_number),
                None)
    if page is None:
        out["page"] = None
        return out
    page_rec: dict = {}
    predictions = getattr(page, "predictions", None)
    layout = getattr(predictions, "layout", None)
    if layout is not None:
        labels: dict = {}
        for cluster in getattr(layout, "clusters", None) or []:
            label = _enum_text(getattr(cluster, "label", None)) or "unknown"
            labels[label] = labels.get(label, 0) + 1
        page_rec["layout_clusters"] = labels
    else:
        page_rec["layout_clusters"] = None
    parsed = getattr(page, "parsed_page", None)
    cells = list(getattr(parsed, "textline_cells", None) or []) if parsed is not None else []
    page_rec["textline_cells"] = len(cells)
    page_rec["ocr_textline_cells"] = sum(1 for c in cells if getattr(c, "from_ocr", False))
    structure = getattr(predictions, "tablestructure", None)
    if structure is None:
        page_rec["table_stage"] = "no_prediction"
        page_rec["table_predictions"] = []
    else:
        page_rec["table_stage"] = "predicted"
        preds = []
        for table in list(getattr(structure, "table_map", {}).values())[:20]:
            cluster = getattr(table, "cluster", None)
            preds.append({
                "num_rows": getattr(table, "num_rows", None),
                "num_cols": getattr(table, "num_cols", None),
                "table_cells": len(getattr(table, "table_cells", None) or []),
                "otsl_len": len(getattr(table, "otsl_seq", None) or []),
                "cluster_label": _enum_text(getattr(cluster, "label", None)),
                "cluster_cells": len(getattr(cluster, "cells", None) or []),
                "cluster_confidence": _finite(getattr(cluster, "confidence", None)),
            })
        page_rec["table_predictions"] = preds
    out["page"] = page_rec
    return out


class Engines:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._tess = None
        self._converters: dict = {}
        self._versions: Optional[dict] = None

    # -- lazy engines ---------------------------------------------------------

    def _tesseract(self):
        if self._tess is None:
            import tesserocr
            self._tess = tesserocr.PyTessBaseAPI(
                path=self.cfg["tessdata_dir"].rstrip("/") + "/", lang="eng",
                psm=tesserocr.PSM.AUTO, oem=tesserocr.OEM.LSTM_ONLY,
            )
        return self._tess

    def _docling(self, profile: str = PROFILE_NATIVE_TEXT):
        if profile not in TABLE_PROFILES:
            raise ValueError(f"unknown table profile {profile!r}")
        if profile not in self._converters:
            from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
            from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
            from docling.datamodel.backend_options import ThreadedDoclingParseBackendOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.object_detection_engine_options import TransformersObjectDetectionEngineOptions
            from docling.datamodel.pipeline_options import (LayoutObjectDetectionOptions, OcrMode, PdfPipelineOptions,
                                                            TableFormerMode, TableStructureOptions, TesseractOcrOptions)
            from docling.datamodel.stage_model_specs import ObjectDetectionModelSpec
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
            from . import offline_guard

            threads = int(self.cfg["torch_threads"])
            # The two profiles differ in do_ocr only (see PROFILE_* above). With do_ocr=False the OCR
            # options are inert: Docling's OCR model is created disabled and starts no Tesseract reader.
            options = PdfPipelineOptions(
                do_ocr=profile == PROFILE_OCR_PSM11, do_table_structure=True, force_backend_text=False,
                enable_remote_services=False, allow_external_plugins=False,
                do_code_enrichment=False, do_formula_enrichment=False,
                do_picture_classification=False, do_picture_description=False,
                generate_page_images=True, generate_table_images=False, generate_picture_images=False,
                generate_parsed_pages=True, images_scale=OCR_SCALE_300,
                ocr_batch_size=1, layout_batch_size=1, table_batch_size=1, queue_max_size=1,
                document_timeout=900, use_reading_order_separators=True,
                artifacts_path=self.cfg["model_dir"],
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=threads),
                layout_options=LayoutObjectDetectionOptions(
                    model_spec=ObjectDetectionModelSpec(name="layout_heron", repo_id=HERON_REPO, revision=HERON_REVISION),
                    engine_options=TransformersObjectDetectionEngineOptions(torch_dtype="float32", compile_model=False),
                ),
                table_structure_options=TableStructureOptions(do_cell_matching=True, mode=TableFormerMode.ACCURATE),
                ocr_options=TesseractOcrOptions(mode=OcrMode.FULL_PAGE, lang=["eng"], psm=TABLE_OCR_PSM,
                                                scale=OCR_SCALE_300,
                                                path=self.cfg["tessdata_dir"].rstrip("/") + "/"),
            )
            backend = ThreadedDoclingParseBackendOptions(parser_threads=1, max_concurrent_results=1,
                                                         render_pages=True, render_scale=OCR_SCALE_300)
            fmt = PdfFormatOption(pipeline_cls=StandardPdfPipeline, backend=ThreadedDoclingParseDocumentBackend,
                                  backend_options=backend, pipeline_options=options)
            converter = DocumentConverter(allowed_formats=[InputFormat.PDF], format_options={InputFormat.PDF: fmt})
            converter.initialize_pipeline(InputFormat.PDF)
            offline_guard.deny_download_helpers()
            self._converters[profile] = converter
        return self._converters[profile]

    def versions(self) -> dict:
        if self._versions is None:
            import importlib.metadata as md
            out = {}
            for dist in ("docling-slim", "docling-core", "docling-ibm-models", "docling-parse", "torch",
                         "tesserocr", "pypdfium2", "opencv-python-headless", "transformers"):
                try:
                    out[dist] = md.version(dist)
                except Exception:
                    out[dist] = "absent"
            try:
                import tesserocr
                out["tesseract"] = tesserocr.tesseract_version().splitlines()[0]
            except Exception:
                out["tesseract"] = "unavailable"
            out["table_profile"] = PROFILE_NATIVE_TEXT
            out["table_profile_amendment"] = NATIVE_PROFILE_AMENDMENT
            self._versions = out
        return self._versions

    # -- job ----------------------------------------------------------------

    def run(self, job: dict) -> dict:
        pdf_bytes = job["pdf_bytes"]
        ocr_pages = sorted(set(job.get("ocr_pages", [])))
        table_pages = sorted(set(job.get("table_pages", [])))
        import pypdfium2 as pdfium
        try:
            pdf = pdfium.PdfDocument(pdf_bytes)
        except Exception:
            raise EngineInputError("the PDF could not be opened") from None
        try:
            page_count = len(pdf)
            if page_count > int(self.cfg["max_pdf_pages"]):
                raise EngineInputError("the PDF exceeds the page limit")
            for p in ocr_pages + table_pages:
                if p < 1 or p > page_count:
                    raise EngineInputError("a requested page is outside the PDF")
            pages: list[dict] = []
            timings: dict[str, float] = {}
            for p in ocr_pages:
                t0 = time.monotonic()
                pages.append(self._ocr_page(pdf, p))
                timings[f"ocr_page_{p}_s"] = round(time.monotonic() - t0, 3)
            native_chars = {p: native_alnum_chars(pdf, p) for p in table_pages}
        finally:
            pdf.close()
        native_table_pages = [p for p in table_pages if native_chars[p] >= NATIVE_TEXT_MIN_ALNUM]
        for p in table_pages:
            if p not in native_table_pages:
                pages.append(no_native_text_table_page(p, native_chars[p]))
        if native_table_pages:
            os.makedirs(self.cfg["work_dir"], exist_ok=True)
            work = tempfile.mkdtemp(prefix="job-", dir=self.cfg["work_dir"])
            try:
                path = os.path.join(work, "input.pdf")
                with open(path, "xb") as fh:
                    fh.write(pdf_bytes)
                for p in native_table_pages:
                    t0 = time.monotonic()
                    page = self._table_page(path, p, PROFILE_NATIVE_TEXT)
                    page["diagnostics"]["native_alnum_chars"] = native_chars[p]
                    pages.append(page)
                    timings[f"table_page_{p}_s"] = round(time.monotonic() - t0, 3)
            finally:
                shutil.rmtree(work, ignore_errors=True)
        return {"pages": sorted(pages, key=lambda r: r["page_number"]), "engine": self.versions(), "timings": timings}

    def _ocr_page(self, pdf, page_number: int) -> dict:
        page = pdf[page_number - 1]
        try:
            bitmap = page.render(scale=int(self.cfg["ocr_dpi"]) / 72, rotation=0)
            image = bitmap.to_pil()
            bitmap.close()
        finally:
            page.close()
        api = self._tesseract()
        api.SetImage(image)
        api.SetSourceResolution(int(self.cfg["ocr_dpi"]))
        api.Recognize()
        text = _clean_text(api.GetUTF8Text())
        conf = api.MeanTextConf()
        api.Clear()
        return {"page_number": page_number, "method": "ocr", "text": text,
                "mean_confidence": float(conf) if conf is not None and conf >= 0 else None,
                "tables": [], "warnings": []}

    def _table_page(self, path: str, page_number: int, profile: str = PROFILE_NATIVE_TEXT) -> dict:
        page = self._page_from_result(self._convert_table_page(path, page_number, profile), page_number)
        page["diagnostics"]["profile"] = profile
        return page

    def _convert_table_page(self, path: str, page_number: int, profile: str = PROFILE_NATIVE_TEXT):
        converter = self._docling(profile)
        return converter.convert(path, page_range=(page_number, page_number), max_num_pages=10_000,
                                 max_file_size=200 * 1024 * 1024, raises_on_error=True)

    def _page_from_result(self, result: Any, page_number: int) -> dict:
        """Page record from one conversion. Unusable (empty) table objects are never emitted as
        tables: they are reported with a warning and their area's text is kept as plain text."""
        from docling_core.types.doc import DocItemLabel, TableItem
        doc = result.document
        page_height = None
        try:
            page_height = float(doc.pages[page_number].size.height)
        except Exception:
            page_height = None
        text_cells: list = []
        for page in getattr(result, "pages", None) or []:
            if getattr(page, "page_no", None) != page_number or getattr(page, "parsed_page", None) is None:
                continue
            for cell in page.parsed_page.textline_cells or []:
                try:
                    box = cell.rect.to_bounding_box()
                    if page_height is not None:
                        box = box.to_top_left_origin(page_height)
                    text_cells.append((cell.text, (box.l, box.t, box.r, box.b)))
                except Exception:
                    continue
        texts: list[str] = []
        last_heading: Optional[str] = None
        tables: list[dict] = []
        unusable: list[dict] = []
        warnings: list[str] = []
        for item, _level in doc.iterate_items():
            provs = [p for p in (getattr(item, "prov", None) or []) if getattr(p, "page_no", None) == page_number]
            if not provs:
                continue
            if isinstance(item, TableItem):
                caption = ""
                try:
                    caption = item.caption_text(doc) or ""
                except Exception:
                    caption = ""
                if not caption and last_heading:
                    caption = f"Under heading: {last_heading}"
                record = {
                    "num_rows": item.data.num_rows,
                    "num_cols": item.data.num_cols,
                    "caption": caption,
                    "cells": [
                        {"text": c.text, "start_row": c.start_row_offset_idx, "end_row": c.end_row_offset_idx,
                         "start_col": c.start_col_offset_idx, "end_col": c.end_col_offset_idx,
                         "column_header": bool(c.column_header), "row_header": bool(c.row_header),
                         "row_section": bool(c.row_section)}
                        for c in item.data.table_cells
                    ],
                }
                usable, reason = table_usability(record["num_rows"], record["num_cols"], record["cells"])
                if usable:
                    tables.append(record)
                    continue
                kept = ""
                if page_height is not None:
                    try:
                        box = provs[0].bbox.to_top_left_origin(page_height)
                        kept = region_text(text_cells, (box.l, box.t, box.r, box.b))
                    except Exception:
                        kept = ""
                if kept:
                    texts.append(kept)
                unusable.append({"reason": reason, "num_rows": record["num_rows"], "num_cols": record["num_cols"],
                                 "cells": len(record["cells"]), "kept_text_chars": len(kept)})
                warnings.append(unusable_table_warning(reason, record["num_rows"], record["num_cols"],
                                                       len(record["cells"]), len(kept)))
                continue
            text = getattr(item, "text", None)
            if text:
                texts.append(text)
                if getattr(item, "label", None) in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
                    last_heading = " ".join(text.split())[:200]
        conf = None
        try:
            page_conf = result.confidence.pages.get(page_number)
            score = getattr(page_conf, "ocr_score", None)
            if score is not None and not math.isnan(float(score)):
                conf = round(float(score) * 100.0, 1)
        except Exception:
            conf = None
        if not tables and not unusable:
            warnings.append("no table structure was detected on this page")
        diagnostics = summarize_conversion(result, page_number)
        diagnostics["tables_detected"] = len(tables) + len(unusable)
        diagnostics["tables_usable"] = len(tables)
        diagnostics["unusable_tables"] = unusable
        return {"page_number": page_number, "method": "table", "text": _clean_text("\n".join(texts)),
                "mean_confidence": conf, "tables": tables, "warnings": warnings, "diagnostics": diagnostics}


def worker_main(conn: Any, cfg: dict) -> None:
    """Child-process entry point: serve jobs until the pipe closes."""
    from . import offline_guard
    offline_guard.install_network_deny()
    try:
        import torch  # noqa: F401  (thread count before any model work)
        torch.set_num_threads(int(cfg["torch_threads"]))
    except Exception:
        pass
    engines = Engines(cfg)
    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            return
        try:
            result = engines.run(job)
            conn.send({"ok": True, "result": result})
        except EngineInputError as exc:
            conn.send({"ok": False, "kind": "input", "message": str(exc)[:300]})
        except Exception as exc:
            conn.send({"ok": False, "kind": "engine", "message": type(exc).__name__})
