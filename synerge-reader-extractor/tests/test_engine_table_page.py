"""Engine table-page handling without Docling: usability rule, kept text, diagnostics.

Docling is not installed in the unit-test environment, so the two names the engine imports from
docling_core are provided by a tiny fake module, and the conversion result is a plain object graph
shaped like Docling's public fields. No model, OCR or inference runs here.
"""

import json
import math
import sys
import types
from types import SimpleNamespace as NS

import pytest

from extractor_service.engine import (Engines, region_text, summarize_conversion, table_usability,
                                      unusable_table_warning)

PAGE_H = 792.0


class Box:
    def __init__(self, l, t, r, b, origin="TOPLEFT"):
        self.l, self.t, self.r, self.b, self.origin = l, t, r, b, origin

    def to_top_left_origin(self, page_height):
        if self.origin == "TOPLEFT":
            return self
        return Box(self.l, page_height - self.t, self.r, page_height - self.b, "TOPLEFT")


class FakeTableItem:
    def __init__(self, num_rows, num_cols, cells, bbox, caption=""):
        self.data = NS(num_rows=num_rows, num_cols=num_cols, table_cells=cells)
        self.prov = [NS(page_no=1, bbox=bbox)]
        self._caption = caption

    def caption_text(self, doc):
        return self._caption


@pytest.fixture
def fake_docling(monkeypatch):
    label = NS(SECTION_HEADER="section_header", TITLE="title")
    doc_mod = types.ModuleType("docling_core.types.doc")
    doc_mod.DocItemLabel = label
    doc_mod.TableItem = FakeTableItem
    monkeypatch.setitem(sys.modules, "docling_core", types.ModuleType("docling_core"))
    monkeypatch.setitem(sys.modules, "docling_core.types", types.ModuleType("docling_core.types"))
    monkeypatch.setitem(sys.modules, "docling_core.types.doc", doc_mod)
    return label


def _cell(text, l, t, r, b):
    return NS(text=text, from_ocr=True, rect=NS(to_bounding_box=lambda: Box(l, t, r, b)))


def _tcell(text, r, c, header=False):
    return NS(text=text, start_row_offset_idx=r, end_row_offset_idx=r + 1, start_col_offset_idx=c,
              end_col_offset_idx=c + 1, column_header=header, row_header=False, row_section=False)


def _result(items, label, table_map_rows=0, cluster_cells=0, status="success", errors=()):
    cells = [_cell("Statement of results", 48, 60, 300, 80), _cell("Item 2025 2024", 60, 115, 400, 130),
             _cell("Revenue 1,200 1,050", 60, 150, 400, 165), _cell("Footer note", 48, 700, 300, 715)]
    layout = NS(clusters=[NS(label=NS(value="section_header")), NS(label=NS(value="table"))])
    table = NS(num_rows=table_map_rows, num_cols=table_map_rows, table_cells=[], otsl_seq=[],
               cluster=NS(label=NS(value="table"), cells=[object()] * cluster_cells, confidence=0.93))
    page = NS(page_no=1, parsed_page=NS(textline_cells=cells),
              predictions=NS(layout=layout, tablestructure=NS(table_map={7: table})))
    doc = NS(pages={1: NS(size=NS(height=PAGE_H))}, iterate_items=lambda: [(i, 0) for i in items])
    return NS(document=doc, pages=[page], status=NS(value=status), errors=list(errors), timings={},
              confidence=NS(pages={1: NS(parse_score=1.0, layout_score=0.9, ocr_score=0.95, table_score=math.nan)}))


def _heading(label):
    return NS(text="Statement of results", label=label.SECTION_HEADER, prov=[NS(page_no=1, bbox=Box(48, 60, 300, 80))])


# Table region (48,108)-(480,276) in top-left points, given in Docling's bottom-left origin.
TABLE_BOX_BL = Box(48, PAGE_H - 108, 480, PAGE_H - 276, "BOTTOMLEFT")


def test_empty_table_object_is_not_emitted_and_its_area_text_is_kept(fake_docling):
    items = [_heading(fake_docling), FakeTableItem(0, 0, [], TABLE_BOX_BL)]
    page = Engines({})._page_from_result(_result(items, fake_docling), 1)
    assert page["tables"] == []
    assert page["text"] == "Statement of results\nItem 2025 2024\nRevenue 1,200 1,050"
    assert "Footer note" not in page["text"]
    assert page["warnings"] == [
        "a table was detected but its structure came back empty (0 rows x 0 columns); it was not indexed as a "
        "table and the text in that area was kept as plain text"]
    diag = page["diagnostics"]
    assert diag["conversion_status"] == "success" and diag["errors"] == []
    assert diag["tables_detected"] == 1 and diag["tables_usable"] == 0
    assert diag["unusable_tables"] == [{"reason": "zero_dimensions", "num_rows": 0, "num_cols": 0, "cells": 0,
                                        "kept_text_chars": len("Item 2025 2024\nRevenue 1,200 1,050")}]
    assert diag["page"]["ocr_textline_cells"] == 4 and diag["page"]["table_stage"] == "predicted"
    assert diag["page"]["table_predictions"][0]["cluster_cells"] == 0
    assert diag["confidence"]["table_score"] is None and diag["confidence"]["ocr_score"] == 0.95
    assert page["mean_confidence"] == 95.0
    json.dumps(page)


def test_cell_empty_table_is_unusable_with_explicit_reason(fake_docling):
    blank = [_tcell(" ", 0, 0), _tcell("", 0, 1)]
    items = [FakeTableItem(1, 2, blank, TABLE_BOX_BL, caption="Results")]
    page = Engines({})._page_from_result(_result(items, fake_docling), 1)
    assert page["tables"] == []
    assert page["diagnostics"]["unusable_tables"][0]["reason"] == "cells_without_text"
    assert "(2 cells, none with text)" in page["warnings"][0]


def test_usable_table_is_emitted_unchanged_with_heading_caption(fake_docling):
    cells = [_tcell("2025", 0, 1, header=True), _tcell("Revenue", 1, 0), _tcell("1,200", 1, 1)]
    items = [_heading(fake_docling), FakeTableItem(2, 2, cells, TABLE_BOX_BL)]
    page = Engines({})._page_from_result(_result(items, fake_docling, table_map_rows=2, cluster_cells=3), 1)
    assert len(page["tables"]) == 1 and page["warnings"] == []
    table = page["tables"][0]
    assert table["caption"] == "Under heading: Statement of results"
    assert table["cells"][0] == {"text": "2025", "start_row": 0, "end_row": 1, "start_col": 1, "end_col": 2,
                                 "column_header": True, "row_header": False, "row_section": False}
    assert page["text"] == "Statement of results", "table words stay in the table, not duplicated as text"
    assert page["diagnostics"]["tables_usable"] == 1 and page["diagnostics"]["unusable_tables"] == []


def test_page_without_any_table_says_so(fake_docling):
    page = Engines({})._page_from_result(_result([_heading(fake_docling)], fake_docling), 1)
    assert page["tables"] == [] and page["warnings"] == ["no table structure was detected on this page"]


@pytest.mark.parametrize("rows,cols,cells,expected", [
    (0, 0, [], (False, "zero_dimensions")),
    (3, 0, [{"text": "x"}], (False, "zero_dimensions")),
    (None, 2, [{"text": "x"}], (False, "zero_dimensions")),
    (2, 2, [], (False, "no_cells")),
    (2, 2, [{"text": " "}, {"text": None}], (False, "cells_without_text")),
    (2, 2, [{"text": ""}, {"text": "(42)"}], (True, "ok")),
])
def test_table_usability(rows, cols, cells, expected):
    assert table_usability(rows, cols, cells) == expected


def test_region_text_keeps_mostly_inside_cells_in_reading_order():
    cells = [("second row", (60, 150, 200, 165)), ("first row", (60, 115, 200, 130)),
             ("straddles", (400, 115, 700, 130)), ("outside", (48, 700, 300, 715)), ("  ", (60, 120, 70, 125))]
    assert region_text(cells, (48, 108, 480, 276)) == "first row\nsecond row"


def test_warning_wording_without_kept_text():
    assert unusable_table_warning("no_cells", 2, 2, 0, 0).endswith("no text could be recovered from that area")


def test_summarize_conversion_bounds_errors_and_survives_missing_fields():
    errors = [NS(component_type=NS(value="model"), module_name="table", error_message="boom " * 200, page_no=1)] * 15
    result = NS(status=NS(value="partial_success"), errors=errors, timings=None, confidence=None, pages=None)
    out = summarize_conversion(result, 1)
    assert out["conversion_status"] == "partial_success" and out["error_count"] == 15
    assert len(out["errors"]) == 10 and len(out["errors"][0]["message"]) == 300
    assert out["confidence"] is None and out["timings_s"] == {} and out["page"] is None
    json.dumps(out)
