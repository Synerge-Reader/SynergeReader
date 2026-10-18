"""One instrumented, table-only diagnostic conversion (acceptance evidence; not used by the service).

    docker run --rm --network none --read-only --tmpfs /tmp:size=2g,mode=1777 --cap-drop ALL \
        --user "$(id -u):$(id -g)" -v "$DIAG:/out" <image> python -m extractor_service.diagnose_table --out /out

By default it converts the self-test's synthetic ruled-table page (the image-only page that failed)
exactly once with the failed ``ocr_fullpage_psm11`` profile, so that failure stays reproducible.
``--pdf-stdin --page N --profile native_text_tables_v1`` converts one page of a PDF on stdin with the
named engine profile instead (the profile is required with --pdf-stdin). Besides diagnostic.json it
writes engine_page.json (the full page record the service would return: text, structured cells,
warnings, diagnostics) and tableformer_tokens.json (every token handed to TableFormer, i.e. the
native word cells for the native-text profile).

Read-only hooks record what each stage produced and pass every result through unchanged:
Docling's OCR orientation detection, the OCR text lines, layout clusters before and after
post-processing (with the table cluster's assigned cells), the tokens handed to TableFormer,
TableFormer's raw prediction (tag sequence length, predicted cell boxes), cell-matching counts,
and the conversion status, errors and stage timings. Stage timings are recorded by turning on
Docling's profiling switch, which does not change any output. No separate OCR test is run.

Writes diagnostic.json, page_overlay.png and the TableFormer input crop(s) into --out, and prints a
one-line verdict naming the first stage at which the table lost its content. Before blaming cell
assignment it counts the OCR text lines inside the detected table region, so missing OCR coverage
(``STAGE=ocr_coverage``) is reported as such. For the synthetic page it also applies the self-test's
unchanged table criterion (``selftest.table_criterion``). Exit status: 0 criterion met (or a
stdin PDF converted), 3 converted but criterion not met, 1 exception; evidence is written in all cases.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from typing import Any

MAX_ITEMS = 200
MAX_TOKENS = 5000
MAX_TEXT = 80


def _bbox(b: Any) -> dict | None:
    if b is None:
        return None
    try:
        return {"l": round(float(b.l), 2), "t": round(float(b.t), 2), "r": round(float(b.r), 2),
                "b": round(float(b.b), 2), "origin": str(getattr(getattr(b, "coord_origin", None), "value", ""))}
    except Exception:
        return None


def _cluster(c: Any) -> dict:
    cells = list(getattr(c, "cells", None) or [])
    return {"id": getattr(c, "id", None), "label": str(getattr(getattr(c, "label", None), "value", "")),
            "confidence": round(float(getattr(c, "confidence", 0.0) or 0.0), 4), "bbox": _bbox(getattr(c, "bbox", None)),
            "cells": len(cells), "children": len(getattr(c, "children", None) or []),
            "cell_texts": [str(x.text)[:MAX_TEXT] for x in cells[:8]]}


TABLE_LABELS = ("table", "document_index")


def add_region_text(record: dict, cell_boxes: list) -> None:
    """Record how many page text lines lie inside this detected table region (engine's
    at-least-half-of-the-cell rule), independently of which cluster the lines were assigned to."""
    from .engine import cells_in_region
    region = record.get("bbox")
    if not region:
        record["region_text_cells"] = None
        record["region_check"] = "no_region_bbox"
        return
    same_origin = [(t, b) for t, b in cell_boxes if b and b.get("origin") == region.get("origin")]
    if len(same_origin) != len([1 for _, b in cell_boxes if b]):
        record["region_check"] = "some_cells_in_other_origin"
    texts = cells_in_region([(t, (b["l"], b["t"], b["r"], b["b"])) for t, b in same_origin],
                            (region["l"], region["t"], region["r"], region["b"]))
    record["region_text_cells"] = len(texts)
    record["region_text_sample"] = [t[:MAX_TEXT] for t in texts[:8]]


class _Log(logging.Handler):
    def __init__(self, sink: list):
        super().__init__(level=logging.INFO)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        if len(self.sink) < MAX_ITEMS:
            try:
                self.sink.append({"level": record.levelname, "logger": record.name,
                                  "message": record.getMessage()[:300]})
            except Exception:
                pass


class _OsdProxy:
    """Forwards every call to the real tesserocr OSD reader; records DetectOrientationScript results."""

    def __init__(self, real: Any, sink: list):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_sink", sink)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def DetectOrientationScript(self):  # noqa: N802 (tesserocr API name)
        res = self._real.DetectOrientationScript()
        if len(self._sink) < MAX_ITEMS:
            self._sink.append(None if res is None else {k: res.get(k) for k in
                                                        ("orient_deg", "orient_conf", "script_name", "script_conf")})
        return res


def install_hooks(rec: dict, out_dir: str) -> None:
    import numpy as np
    from docling.models.stages.ocr.tesseract_ocr_model import TesseractOcrModel
    from docling.utils.layout_postprocessor import LayoutPostprocessor
    from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
    from PIL import Image

    orig_init = TesseractOcrModel.__init__

    def ocr_init(self, *a, **k):
        orig_init(self, *a, **k)
        if getattr(self, "_osd_reader", None) is not None:
            self._osd_reader = _OsdProxy(self._osd_reader, rec["osd"])
    TesseractOcrModel.__init__ = ocr_init

    orig_pp = LayoutPostprocessor.postprocess

    def postprocess(self):
        raw = [_cluster(c) for c in (self.all_clusters or [])][:MAX_ITEMS]
        cells = list(self.cells or [])
        cell_boxes = []
        for c in cells:
            try:
                cell_boxes.append((str(c.text), _bbox(c.rect.to_bounding_box())))
            except Exception:
                continue
        out = orig_pp(self)
        final = [_cluster(c) for c in out][:MAX_ITEMS]
        for record in raw + final:
            if record["label"] in TABLE_LABELS:
                add_region_text(record, cell_boxes)
        rec["layout"].append({
            "page_no": getattr(self.page, "page_no", None),
            "page_text_cells": len(cells),
            "page_text_cells_from_ocr": sum(1 for c in cells if getattr(c, "from_ocr", False)),
            "page_text_cell_sample": [{"text": t[:MAX_TEXT], "bbox": b} for t, b in cell_boxes[:60]],
            "raw_clusters": raw,
            "final_clusters": final,
        })
        return out
    LayoutPostprocessor.postprocess = postprocess

    orig_mtp = TFPredictor.multi_table_predict

    def multi_table_predict(self, iocr_page, table_bboxes, do_matching=True, *a, **k):
        n = len(rec["tableformer"]) + 1
        tokens = list(iocr_page.get("tokens") or [])
        rec.setdefault("_tokens_full", []).append({
            "call": n, "table_bboxes_scaled": [[round(float(v), 2) for v in b] for b in table_bboxes],
            "tokens": [{"id": t.get("id"), "text": str(t.get("text"))[:200],
                        "bbox": {kk: round(float(t["bbox"][kk]), 2) for kk in ("l", "t", "r", "b")}}
                       for t in tokens[:MAX_TOKENS]],
            "tokens_total": len(tokens)})
        entry = {"call": n, "do_matching": do_matching,
                 "page_input_wh": [round(float(iocr_page["width"]), 1), round(float(iocr_page["height"]), 1)],
                 "page_image_shape": list(np.asarray(iocr_page["image"]).shape),
                 "table_bboxes_scaled": [[round(float(v), 1) for v in b] for b in table_bboxes],
                 "tokens": len(tokens),
                 "token_sample": [{"text": str(t.get("text"))[:MAX_TEXT],
                                   "bbox": {kk: round(float(t["bbox"][kk]), 1) for kk in ("l", "t", "r", "b")}}
                                  for t in tokens[:30]]}
        try:
            img = Image.fromarray(np.asarray(iocr_page["image"]))
            for i, b in enumerate(table_bboxes, 1):
                img.crop(tuple(int(round(float(v))) for v in b)).save(
                    os.path.join(out_dir, f"tableformer_crop_{n}_{i}.png"))
        except Exception as exc:
            entry["crop_save_error"] = type(exc).__name__
        out = orig_mtp(self, iocr_page, table_bboxes, do_matching, *a, **k)
        entry["outputs"] = [{"tf_responses": len(o.get("tf_responses") or []),
                             "num_rows": o.get("predict_details", {}).get("num_rows"),
                             "num_cols": o.get("predict_details", {}).get("num_cols")} for o in out]
        rec["tableformer"].append(entry)
        return out
    TFPredictor.multi_table_predict = multi_table_predict

    orig_predict = TFPredictor.predict

    def predict(self, iocr_page, table_bbox, table_image, scale_factor, eval_res_preds=None,
                correct_overlapping_cells=False):
        raw: dict = {"table_image_shape": list(np.asarray(table_image).shape),
                     "scale_factor": round(float(scale_factor), 5)}
        model = self._model
        real = model.predict

        def model_predict(*a, **k):
            res = real(*a, **k)
            seq, cls, coord = res
            raw["tag_seq_len"] = None if seq is None else len(seq)
            try:
                raw["rs_seq_head"] = self._get_html_tags(seq)[:80]
            except Exception:
                raw["rs_seq_head"] = None
            raw["outputs_coord_shape"] = None if coord is None else list(getattr(coord, "shape", []))
            raw["outputs_class_shape"] = None if cls is None else list(getattr(cls, "shape", []))
            return res
        model.predict = model_predict
        try:
            tf_output, matching = orig_predict(self, iocr_page, table_bbox, table_image, scale_factor,
                                               eval_res_preds, correct_overlapping_cells)
        finally:
            try:
                del model.predict
            except Exception:
                model.predict = real
        pred = matching.get("prediction") or {}
        raw.update({"predicted_cell_boxes": len(pred.get("bboxes") or []),
                    "table_cells_built": len(matching.get("table_cells") or []),
                    "pdf_cells": len(matching.get("pdf_cells") or []),
                    "matched_pdf_cells": len(matching.get("matches") or {}),
                    "docling_responses": len(matching.get("docling_responses") or []),
                    "tf_output_cells": len(tf_output or [])})
        rec["tf_predict"].append(raw)
        return tf_output, matching
    TFPredictor.predict = predict


def verdict(rec: dict) -> str:
    conv = rec.get("conversion") or {}
    if conv.get("conversion_status") not in ("success", "partial_success"):
        return f"STAGE=conversion: status {conv.get('conversion_status')} errors {conv.get('error_count')}"
    layout = rec["layout"][-1] if rec["layout"] else None
    if layout is None:
        return "STAGE=layout_postprocess: layout post-processing was not reached"
    if layout["page_text_cells"] == 0:
        return f"STAGE=ocr: no text lines on the page after OCR (OSD results: {rec['osd'][:3]})"
    raw_tables = [c for c in layout["raw_clusters"] if c["label"] in TABLE_LABELS]
    final_tables = [c for c in layout["final_clusters"] if c["label"] in TABLE_LABELS]
    if not raw_tables:
        return "STAGE=layout: the layout model detected no table region"
    if not final_tables:
        return "STAGE=layout_postprocess: the table region was removed during layout post-processing"
    if all(c["cells"] == 0 for c in final_tables):
        # Before blaming cell assignment, check whether any OCR text exists inside the region at all.
        inside = [c.get("region_text_cells") for c in raw_tables + final_tables]
        if any(n is None for n in inside):
            return ("UNDETERMINED: the table cluster has no assigned text, and the text inside its region "
                    "could not be counted (see region_check in diagnostic.json)")
        if max(inside) == 0:
            return (f"STAGE=ocr_coverage: {layout['page_text_cells']} text lines on the page "
                    f"({layout['page_text_cells_from_ocr']} from OCR) but none inside the detected table "
                    "region, so TableFormer had no tokens to match")
        return (f"STAGE=cell_assignment: {max(inside)} text lines lie inside the detected table region but "
                "none were assigned to the table cluster, so TableFormer had no tokens to match")
    if not rec["tableformer"]:
        return "STAGE=table: the table stage never called TableFormer"
    tf = rec["tableformer"][-1]
    if tf["tokens"] == 0:
        return "STAGE=table_tokens: TableFormer was called with zero tokens"
    pr = rec["tf_predict"][-1] if rec["tf_predict"] else {}
    if not pr.get("predicted_cell_boxes"):
        return f"STAGE=tableformer: no predicted cell boxes (tag sequence length {pr.get('tag_seq_len')})"
    if not pr.get("matched_pdf_cells"):
        return (f"STAGE=matching: {tf['tokens']} tokens and {pr['predicted_cell_boxes']} predicted cells "
                "did not overlap")
    out = tf["outputs"][-1] if tf.get("outputs") else {}
    if out.get("num_rows") and out.get("num_cols"):
        text = f"STRUCTURE_PRODUCED: {out['num_rows']} x {out['num_cols']}, {out['tf_responses']} cells"
        check = rec.get("synthetic_check")
        if check:
            text += (f"; synthetic expected values matched {check['cells_matched']}/{check['cells_expected']}"
                     f" (self-test table criterion {'MET' if check['criterion_met'] else 'NOT MET'})")
        return text
    return "UNDETERMINED: see diagnostic.json"


def overlay(path_pdf: str, page_number: int, rec: dict, out_dir: str) -> None:
    import pypdfium2 as pdfium
    from PIL import ImageDraw
    pdf = pdfium.PdfDocument(path_pdf)
    try:
        page = pdf[page_number - 1]
        image = page.render(scale=2.0).to_pil().convert("RGB")
        page.close()
    finally:
        pdf.close()
    draw = ImageDraw.Draw(image)
    if rec["layout"]:
        lay = rec["layout"][-1]
        for cell in lay["page_text_cell_sample"]:
            b = cell["bbox"]
            if b:
                draw.rectangle([b["l"] * 2, b["t"] * 2, b["r"] * 2, b["b"] * 2], outline=(0, 160, 0), width=2)
        for c in lay["final_clusters"]:
            b = c["bbox"]
            if b:
                color, width = ((0, 70, 220), 5) if c["label"] in TABLE_LABELS else ((220, 0, 0), 2)
                draw.rectangle([b["l"] * 2, b["t"] * 2, b["r"] * 2, b["b"] * 2], outline=color, width=width)
    for tf in rec["tableformer"]:
        for tok in tf["token_sample"]:
            b = tok["bbox"]
            draw.rectangle([b["l"], b["t"], b["r"], b["b"]], outline=(255, 140, 0), width=1)
    image.save(os.path.join(out_dir, "page_overlay.png"))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="extractor_service.diagnose_table")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pdf-stdin", action="store_true")
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--profile", choices=["ocr_fullpage_psm11", "native_text_tables_v1"], default=None,
                        help="table profile; required with --pdf-stdin. The synthetic page defaults to the "
                             "(failed) image-only profile ocr_fullpage_psm11, so that failure stays reproducible.")
    args = parser.parse_args(argv)
    if args.pdf_stdin and args.profile is None:
        parser.error("--profile is required with --pdf-stdin")
    args.profile = args.profile or "ocr_fullpage_psm11"
    os.makedirs(args.out, exist_ok=True)

    from . import offline_guard
    from .config import ServiceSettings
    from .engine import Engines, native_alnum_chars
    offline_guard.install_network_deny()
    settings = ServiceSettings.from_env()
    cfg = settings.engine_config()
    try:
        import torch
        torch.set_num_threads(int(cfg["torch_threads"]))
    except Exception:
        pass
    rec: dict = {"schema": "synergereader.extractor.table-diagnostic.v1", "osd": [], "layout": [],
                 "tableformer": [], "tf_predict": [], "log": [], "input": {}}
    handler = _Log(rec["log"])
    for name in ("docling", "docling_ibm_models"):
        lg = logging.getLogger(name)
        lg.addHandler(handler)
        lg.setLevel(logging.INFO)
    from docling.datamodel.settings import settings as docling_settings
    docling_settings.debug.profile_pipeline_timings = True
    install_hooks(rec, args.out)

    if args.pdf_stdin:
        pdf_bytes = sys.stdin.buffer.read(settings.max_body_bytes + 1)
        if len(pdf_bytes) > settings.max_body_bytes or not pdf_bytes.startswith(b"%PDF-"):
            print(json.dumps({"error": "stdin must be a PDF within the size limit"}))
            return 2
        rec["input"] = {"source": "stdin", "page": args.page}
    else:
        from .selftest import table_page_pdf
        pdf_bytes = table_page_pdf()
        rec["input"] = {"source": "selftest.table_page_pdf", "page": 1}
        args.page = 1
    import hashlib
    rec["input"]["sha256"] = hashlib.sha256(pdf_bytes).hexdigest()
    rec["input"]["bytes"] = len(pdf_bytes)
    rec["profile"] = args.profile
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(pdf_bytes)
        try:
            rec["input"]["native_alnum_chars"] = native_alnum_chars(doc, args.page)
        finally:
            doc.close()
    except Exception as exc:
        rec["input"]["native_alnum_chars_error"] = type(exc).__name__

    engines = Engines(cfg)
    rec["engine"] = engines.versions()
    work = tempfile.mkdtemp(prefix="diag-", dir="/tmp")
    status = 0
    try:
        path = os.path.join(work, "input.pdf")
        with open(path, "wb") as fh:
            fh.write(pdf_bytes)
        t0 = time.monotonic()
        result = engines._convert_table_page(path, args.page, args.profile)
        rec["convert_s"] = round(time.monotonic() - t0, 3)
        page = engines._page_from_result(result, args.page)
        page["diagnostics"]["profile"] = args.profile
        with open(os.path.join(args.out, "engine_page.json"), "w", encoding="utf-8") as fh:
            json.dump(page, fh, indent=1, default=str)
        with open(os.path.join(args.out, "tableformer_tokens.json"), "w", encoding="utf-8") as fh:
            json.dump(rec.pop("_tokens_full", []), fh, indent=1, default=str)
        rec["conversion"] = page.get("diagnostics", {})
        rec["engine_page"] = {"tables": [{"num_rows": t["num_rows"], "num_cols": t["num_cols"],
                                          "cells": len(t["cells"])} for t in page["tables"]],
                              "warnings": page["warnings"], "text_chars": len(page["text"]),
                              "text_head": page["text"][:600],
                              "table_cells_text": [[c["text"][:MAX_TEXT] for c in t["cells"]][:60]
                                                   for t in page["tables"]][:5]}
        if rec["input"]["source"] == "selftest.table_page_pdf":
            from .selftest import table_criterion
            matched, expected_count, met = table_criterion(page["tables"])
            rec["synthetic_check"] = {"cells_matched": matched, "cells_expected": expected_count,
                                      "criterion_met": met}
        try:
            overlay(path, args.page, rec, args.out)
        except Exception as exc:
            rec["overlay_error"] = type(exc).__name__
    except Exception as exc:
        rec["exception"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
        status = 1
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    rec["verdict"] = verdict(rec) if status == 0 else f"STAGE=exception: {rec['exception']['type']}"
    with open(os.path.join(args.out, "diagnostic.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, default=str)
    print(json.dumps({"verdict": rec["verdict"], "convert_s": rec.get("convert_s"),
                      "profile": rec.get("profile"),
                      "osd": rec["osd"][:2], "tokens": [t["tokens"] for t in rec["tableformer"]],
                      "synthetic_check": rec.get("synthetic_check"),
                      "tf_predict": rec["tf_predict"][:2]}, default=str))
    return exit_status(status, rec)


def exit_status(status: int, rec: dict) -> int:
    """0: conversion ran and (for the synthetic page) the self-test table criterion is met;
    3: conversion ran and evidence was written, but the criterion is not met; 1: exception."""
    if status != 0:
        return status
    check = rec.get("synthetic_check")
    if check is not None and not check.get("criterion_met"):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
