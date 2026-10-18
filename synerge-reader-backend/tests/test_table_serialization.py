"""Table serialization keeps emitted header context with every value (LIVE-EXTRACTION-20261017).

The fixtures are synthetic grids shaped like a two-period financial statement;
no production document, id or expected answer is used.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from table_serialization import (  # noqa: E402
    FIELD_SEPARATOR,
    TABLE_LINE_TAG_RE,
    TableFormatError,
    column_header_paths,
    parse_cells,
    serialize_worker_table,
)


def _cell(text, r0, r1, c0, c1, **flags):
    return {"text": text, "start_row": r0, "end_row": r1, "start_col": c0, "end_col": c1, **flags}


def _statement_table():
    # Row 0-2 are header rows (status, label, date), row 3 a unit row, then
    # one section row and two body rows with a negative value in parentheses.
    cells = [
        _cell("Status Alpha", 0, 1, 1, 2, column_header=True),
        _cell("Status Beta", 0, 1, 2, 3, column_header=True),
        _cell("As at", 1, 2, 1, 3, column_header=True),
        _cell("01.02.2030", 2, 3, 1, 2, column_header=True),
        _cell("01.02.2029", 2, 3, 2, 3, column_header=True),
        _cell("XU'000", 3, 4, 1, 2, column_header=True),
        _cell("XU'000", 3, 4, 2, 3, column_header=True),
        _cell("WIDGETS", 4, 5, 0, 3, row_section=True),
        _cell("Gross widgets", 5, 6, 0, 1, row_header=True),
        _cell("1,234", 5, 6, 1, 2),
        _cell("987", 5, 6, 2, 3),
        _cell("Widget losses", 6, 7, 0, 1, row_header=True),
        _cell("(56)", 6, 7, 1, 2),
        _cell("", 6, 7, 2, 3),
    ]
    return {"num_rows": 7, "num_cols": 3, "cells": cells, "caption": "STATEMENT OF WIDGETS"}


def test_each_value_carries_its_full_emitted_column_header_path_and_row_label():
    text = serialize_worker_table(_statement_table(), page_number=7, table_index=1)
    lines = text.splitlines()
    assert lines[0] == "[Table 1, page 7] STATEMENT OF WIDGETS"
    body = [ln for ln in lines if "Gross widgets" in ln]
    assert body == [
        "[Table 1, page 7] Gross widgets | Status Alpha / As at / 01.02.2030 / XU'000: 1,234"
        " | Status Beta / As at / 01.02.2029 / XU'000: 987"
    ]


def test_signs_parentheses_and_units_are_verbatim_and_empty_cells_are_not_filled():
    text = serialize_worker_table(_statement_table(), page_number=7, table_index=1)
    losses = [ln for ln in text.splitlines() if "Widget losses" in ln]
    assert losses == ["[Table 1, page 7] Widget losses | Status Alpha / As at / 01.02.2030 / XU'000: (56)"]
    assert "01.02.2029 / XU'000:" not in losses[0], "an empty cell must not be emitted or filled"


def test_section_rows_and_header_rows_are_not_emitted_as_values():
    text = serialize_worker_table(_statement_table(), page_number=7, table_index=1)
    assert "[Table 1, page 7] Section: WIDGETS" in text.splitlines()
    assert not any(ln.startswith("[Table 1, page 7] Status") for ln in text.splitlines())


def test_no_header_is_invented_when_the_model_emitted_none():
    table = {
        "num_rows": 2,
        "num_cols": 2,
        "cells": [_cell("alpha", 0, 1, 0, 1), _cell("12", 0, 1, 1, 2), _cell("beta", 1, 2, 0, 1), _cell("(3)", 1, 2, 1, 2)],
    }
    text = serialize_worker_table(table, page_number=2, table_index=3)
    assert text.splitlines() == ["[Table 3, page 2] | alpha | 12", "[Table 3, page 2] | beta | (3)"]
    assert "Columns:" not in text


def test_spanning_header_applies_to_every_column_it_covers():
    cells = parse_cells(
        [_cell("Top", 0, 1, 0, 2, column_header=True), _cell("L", 1, 2, 0, 1, column_header=True),
         _cell("R", 1, 2, 1, 2, column_header=True)],
        num_rows=2,
        num_cols=2,
    )
    assert column_header_paths(cells, 2) == ["Top / L", "Top / R"]


@pytest.mark.parametrize(
    "bad",
    [
        {"num_rows": 1, "num_cols": 1, "cells": [_cell("x", 0, 2, 0, 1)]},
        {"num_rows": 1, "num_cols": 1, "cells": [_cell("x", 0, 1, -1, 1)]},
        {"num_rows": 1, "num_cols": 1, "cells": [{"text": "x", "start_row": "0", "end_row": 1, "start_col": 0, "end_col": 1}]},
        {"num_rows": True, "num_cols": 1, "cells": []},
        {"num_rows": 1, "num_cols": 1, "cells": "not-a-list"},
    ],
)
def test_out_of_grid_or_malformed_cells_are_rejected_not_repaired(bad):
    with pytest.raises(TableFormatError):
        serialize_worker_table(bad, page_number=1, table_index=1)


def test_page_number_is_carried_in_every_line():
    text = serialize_worker_table(_statement_table(), page_number=12, table_index=2)
    assert all(ln.startswith("[Table 2, page 12]") for ln in text.splitlines())


def test_every_line_opens_with_exactly_one_tag_the_completion_reader_parses():
    # table_row_completion.py finds cut rows by this tag; writer and reader must agree.
    text = serialize_worker_table(_statement_table(), page_number=12, table_index=2)
    for ln in text.splitlines():
        tag = TABLE_LINE_TAG_RE.match(ln)
        assert tag is not None and (tag.group("table"), tag.group("page")) == ("2", "12")
        assert TABLE_LINE_TAG_RE.search(ln, tag.end()) is None
    assert FIELD_SEPARATOR in _line(text, "Gross widgets")


# --- unit scope: a section heading or row label can declare its rows' unit -----------------------------

def _period_table(rows):
    """Two-period statement whose column headings state a money unit inside one cell each
    ("Period one 01.02.2030 XU'000"), followed by the given (kind, text, values) rows."""
    cells = [
        _cell("Group A", 0, 1, 1, 3, column_header=True),
        _cell("Period one 01.02.2030 XU’000", 1, 2, 1, 2, column_header=True),
        _cell("Period two 01.02.2029 XU’000", 1, 2, 2, 3, column_header=True),
    ]
    for offset, (kind, text, values) in enumerate(rows):
        r = 2 + offset
        if kind == "section":
            cells.append(_cell(text, r, r + 1, 0, 3, row_section=True))
            continue
        cells.append(_cell(text, r, r + 1, 0, 1, row_header=True))
        for c, value in enumerate(values, start=1):
            cells.append(_cell(value, r, r + 1, c, c + 1))
    return {"num_rows": 2 + len(rows), "num_cols": 3, "cells": cells}


def _line(text, label):
    return next(ln for ln in text.splitlines() if f"] {label} |" in ln)


def test_section_declared_unit_replaces_the_column_unit_for_its_rows_only():
    table = _period_table([
        ("row", "Money row", ["1,200", "(300)"]),
        ("section", "Per-share figures (sen)", []),
        ("row", "Basic (2)", ["(0.52)", "1.34"]),
        ("row", "Diluted (2)", ["(0.50)", "1.30"]),
        ("section", "Other items", []),
        ("row", "Later money row", ["15", "(9)"]),
    ])
    text = serialize_worker_table(table, page_number=2, table_index=1)
    assert _line(text, "Money row") == ("[Table 1, page 2] Money row | Group A / Period one 01.02.2030 XU’000: 1,200"
                                        " | Group A / Period two 01.02.2029 XU’000: (300)")
    assert _line(text, "Basic (2)") == ("[Table 1, page 2] Basic (2) | Group A / Period one 01.02.2030 sen: (0.52)"
                                        " | Group A / Period two 01.02.2029 sen: 1.34")
    assert "XU" not in _line(text, "Diluted (2)") and "sen: (0.50)" in _line(text, "Diluted (2)")
    assert _line(text, "Later money row").count("XU’000:") == 2, "a new section without a unit clears the scope"


def test_row_label_unit_applies_to_that_row_only():
    table = _period_table([
        ("row", "Weighted shares ('000)", ["601,250", "601,250"]),
        ("row", "Margin (%)", ["12.5", "(3.0)"]),
        ("row", "Money row", ["7", "8"]),
    ])
    text = serialize_worker_table(table, page_number=3, table_index=1)
    assert "01.02.2030 '000: 601,250" in _line(text, "Weighted shares ('000)")
    assert "01.02.2029 %: (3.0)" in _line(text, "Margin (%)")
    assert _line(text, "Money row").count("XU’000:") == 2


@pytest.mark.parametrize("label", ['Profit for the year ("PAT")', "Basic (2)", "Revenue (note 4)", "Segment (a)",
                                   "Items (i) to (x)"])
def test_footnotes_and_abbreviations_are_not_units(label):
    table = _period_table([("section", label, []), ("row", label, ["1", "2"])])
    assert _line(serialize_worker_table(table, page_number=1, table_index=1), label).count("XU’000:") == 2


def test_label_only_heading_row_can_open_a_unit_scope():
    table = _period_table([("row", "Per-share figures (cents)", []), ("row", "Basic", ["0.4", "0.3"])])
    assert "01.02.2030 cents: 0.4" in _line(serialize_worker_table(table, page_number=1, table_index=1), "Basic")


def test_declared_unit_is_appended_when_the_column_heading_states_none():
    table = {"num_rows": 3, "num_cols": 2, "cells": [
        _cell("Year 2030", 0, 1, 1, 2, column_header=True),
        _cell("Ratios (times)", 1, 2, 0, 2, row_section=True),
        _cell("Cover", 2, 3, 0, 1, row_header=True), _cell("4.2", 2, 3, 1, 2)]}
    assert "Cover | Year 2030 / times: 4.2" in serialize_worker_table(table, page_number=1, table_index=1)


@pytest.mark.parametrize("label,unit", [("Earnings per share (sen)", "sen"), ("Shares ('000)", "'000"),
                                        ("Amounts (USD'000):", "USD'000"), ("Totals (RM mil)", "RM mil"),
                                        ("Yield (per cent)", "per cent"), ("Plain label", None),
                                        ("Profit (PAT)", None), ("Basic (2)", None)])
def test_declared_unit_recognises_units_only(label, unit):
    from table_serialization import declared_unit
    assert declared_unit(label) == unit
