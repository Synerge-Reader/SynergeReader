"""The diagnostic's verdict names the first stage at which the table lost its content (pure logic)."""

import pytest

from extractor_service.diagnose_table import add_region_text, exit_status, verdict


def _rec(status="success", text_cells=12, ocr_cells=12, raw=("table",), final=(("table", 5),), region=12,
         tokens=5, boxes=12, matches=5, rows=4, cols=3, tf=True, synthetic=None):
    layout = {"page_text_cells": text_cells, "page_text_cells_from_ocr": ocr_cells,
              "raw_clusters": [{"label": l, "region_text_cells": region if l == "table" else None} for l in raw],
              "final_clusters": [{"label": l, "cells": n, "region_text_cells": region if l == "table" else None}
                                 for l, n in final]}
    rec = {"conversion": {"conversion_status": status, "error_count": 0}, "osd": [{"orient_deg": 0}],
           "layout": [layout], "tableformer": [], "tf_predict": []}
    if tf:
        rec["tableformer"].append({"tokens": tokens, "outputs": [{"num_rows": rows, "num_cols": cols,
                                                                  "tf_responses": 12}]})
        rec["tf_predict"].append({"predicted_cell_boxes": boxes, "matched_pdf_cells": matches, "tag_seq_len": 40})
    if synthetic is not None:
        rec["synthetic_check"] = synthetic
    return rec


@pytest.mark.parametrize("kwargs,prefix", [
    ({"status": "failure"}, "STAGE=conversion"),
    ({"text_cells": 0, "ocr_cells": 0}, "STAGE=ocr:"),
    ({"raw": ("text",), "final": (("text", 3),)}, "STAGE=layout:"),
    ({"final": (("text", 3),)}, "STAGE=layout_postprocess"),
    ({"final": (("table", 0),), "region": 0, "tf": False}, "STAGE=ocr_coverage"),
    ({"final": (("table", 0),), "region": 9, "tf": False}, "STAGE=cell_assignment: 9 text lines lie inside"),
    ({"final": (("table", 0),), "region": None, "tf": False}, "UNDETERMINED"),
    ({"tf": False}, "STAGE=table:"),
    ({"tokens": 0}, "STAGE=table_tokens"),
    ({"boxes": 0}, "STAGE=tableformer"),
    ({"matches": 0}, "STAGE=matching"),
    ({}, "STRUCTURE_PRODUCED: 4 x 3"),
    ({"rows": 0, "cols": 0}, "UNDETERMINED"),
])
def test_verdict_names_the_first_failing_stage(kwargs, prefix):
    assert verdict(_rec(**kwargs)).startswith(prefix)


def test_first_linux_diagnostic_shape_is_now_ocr_coverage_not_cell_assignment():
    # The recorded run: 1 OCR line (the heading, outside the table), table detected, 0 cells assigned.
    rec = _rec(text_cells=1, ocr_cells=1, final=(("section_header", 1), ("table", 0)), region=0, tokens=0,
               rows=0, cols=0)
    assert verdict(rec) == ("STAGE=ocr_coverage: 1 text lines on the page (1 from OCR) but none inside the "
                            "detected table region, so TableFormer had no tokens to match")


def test_structure_verdict_reports_the_synthetic_criterion():
    met = _rec(synthetic={"cells_matched": 12, "cells_expected": 12, "criterion_met": True})
    assert verdict(met).endswith("matched 12/12 (self-test table criterion MET)")
    not_met = _rec(synthetic={"cells_matched": 9, "cells_expected": 12, "criterion_met": False})
    assert verdict(not_met).endswith("matched 9/12 (self-test table criterion NOT MET)")


def test_missing_layout_record_is_reported():
    rec = _rec()
    rec["layout"] = []
    assert verdict(rec).startswith("STAGE=layout_postprocess: layout post-processing was not reached")


def test_exit_status():
    assert exit_status(1, {}) == 1
    assert exit_status(0, {}) == 0, "stdin PDFs have no synthetic criterion"
    assert exit_status(0, {"synthetic_check": {"criterion_met": True}}) == 0
    assert exit_status(0, {"synthetic_check": {"criterion_met": False}}) == 3


def test_stdin_pdf_requires_an_explicit_profile(capsys):
    from extractor_service.diagnose_table import main
    with pytest.raises(SystemExit) as info:
        main(["--out", "/nonexistent", "--pdf-stdin", "--page", "4"])
    assert info.value.code == 2 and "--profile is required" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--out", "/nonexistent", "--profile", "not_a_profile"])


def _box(l, t, r, b, origin="TOPLEFT"):
    return {"l": l, "t": t, "r": r, "b": b, "origin": origin}


def test_region_count_uses_text_inside_the_detected_region_only():
    record = {"label": "table", "bbox": _box(47.3, 106.8, 480.7, 243.0)}
    cells = [("Statement of results (selftest)", _box(48.7, 62.9, 245.0, 76.3)),
             ("Item", _box(55, 115, 80, 128)), ("Revenue", _box(55, 148, 110, 160)),
             ("half outside", _box(470, 115, 520, 128))]
    add_region_text(record, cells)
    assert record["region_text_cells"] == 2 and record["region_text_sample"] == ["Item", "Revenue"]
    assert "region_check" not in record


def test_region_count_flags_mixed_origins_and_missing_region():
    record = {"label": "table", "bbox": _box(0, 0, 100, 100)}
    add_region_text(record, [("a", _box(10, 10, 20, 20)), ("b", _box(10, 90, 20, 80, "BOTTOMLEFT"))])
    assert record["region_text_cells"] == 1 and record["region_check"] == "some_cells_in_other_origin"
    missing = {"label": "table", "bbox": None}
    add_region_text(missing, [])
    assert missing["region_text_cells"] is None and missing["region_check"] == "no_region_bbox"
