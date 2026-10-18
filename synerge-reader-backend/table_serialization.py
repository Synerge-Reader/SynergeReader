"""Index-friendly serialization of extracted table structure.

The extraction worker returns each detected table as a grid of cells with
row/column offsets, spans and the model's own header flags. The ingestion path
chunks text by words, so a table flattened into reading order would let a value
drift into a chunk without the header that gives it meaning (a figure without
its date column, its unit or its audited/unaudited heading).

This module writes every body row as ONE self-contained line that carries:

* a short table tag with the physical page number;
* the row label, built only from cells the model flagged as row headers;
* every non-empty value paired with its column-header path, built only from
  cells the model flagged as column headers that cover that column.

Nothing is inferred. A column with no emitted header keeps its bare value, a
row with no emitted row header has no label, and an empty cell is skipped
rather than filled. Signs, parentheses, units and dates are copied verbatim
from the cell text.

Units have a scope. A column heading may state a unit for the whole column
("RM'000"), but a section heading or row label of the same table can declare
a different unit for its own rows by ending in one, e.g. "Earnings per share
(sen)" or "Number of shares ('000)". Within that scope the column heading's
unit token is replaced by the declared one; dates, period and audit wording in
the heading are kept. A row label's unit applies to that row; a section's unit
applies to the rows below it until the next section heading, which sets or
clears it. Only recognised unit expressions count, so footnote markers such
as "(2)" or abbreviations such as ("PAT") never change a unit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

MAX_TABLE_TEXT_CHARS = 60_000
MAX_CELL_TEXT_CHARS = 400
_HEADER_SEPARATOR = " / "
FIELD_SEPARATOR = " | "
# The tag that opens every serialized table line, e.g. "[Table 1, page 4]". A line lies on the one physical
# page its tag names. table_row_completion.py reads lines back by this tag, so it must match what
# serialize_table() writes.
TABLE_LINE_TAG_RE = re.compile(r"\[Table (?P<table>\d+), page (?P<page>\d+|\?)\](?= |$)")

# Unit expressions (generic, not tied to any document): an amount scale such as RM'000, USD'000, $'000 or
# '000; a currency with a written scale such as RM mil or USD million; or a measurement word.
_SCALED_AMOUNT = r"(?:(?:[A-Z]{1,4}|[$€£¥])\s?)?['’]\s?000"
_WRITTEN_SCALE = r"(?:[A-Z]{2,4}|[$€£¥])\s?(?:mil|million|mn|bn|billion)"
_UNIT_WORDS = r"sen|cents?|%|per\s?cent|percent|times|days|months|years"
# A unit token inside a column heading, e.g. the "RM'000" in "Current quarter 31.05.2025 RM'000".
_COLUMN_UNIT_RE = re.compile(rf"(?<![\w'’])(?:{_SCALED_AMOUNT}|{_WRITTEN_SCALE})(?!\w)")
# A unit declared by a heading or row label: a parenthesised unit at its very end, e.g. "... (sen)".
_DECLARED_UNIT_RE = re.compile(
    rf"\(\s*(?P<unit>{_SCALED_AMOUNT}|{_WRITTEN_SCALE}|{_UNIT_WORDS})\s*\)\s*:?\s*$", re.IGNORECASE
)


def declared_unit(label: Optional[str]) -> Optional[str]:
    """The unit a heading or row label declares for its rows, or None."""
    match = _DECLARED_UNIT_RE.search(label or "")
    return " ".join(match.group("unit").split()) if match else None


def apply_unit(path: str, unit: Optional[str]) -> str:
    """The column-header path with its unit token replaced by ``unit`` (appended when it has none)."""
    if not unit:
        return path
    if _COLUMN_UNIT_RE.search(path):
        return _COLUMN_UNIT_RE.sub(lambda _m: unit, path)
    return f"{path}{_HEADER_SEPARATOR}{unit}" if path else unit


@dataclass(frozen=True)
class TableCell:
    text: str
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    column_header: bool = False
    row_header: bool = False
    row_section: bool = False

    def covers_row(self, row: int) -> bool:
        return self.start_row <= row < self.end_row

    def covers_col(self, col: int) -> bool:
        return self.start_col <= col < self.end_col


class TableFormatError(ValueError):
    """The extraction worker returned a table that is not a valid cell grid."""


def _clean(text: object) -> str:
    if not isinstance(text, str):
        return ""
    collapsed = " ".join(text.split())
    return collapsed[:MAX_CELL_TEXT_CHARS]


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TableFormatError(f"{name} must be an integer")
    return value


def parse_cells(raw_cells: Iterable[Any], num_rows: int, num_cols: int) -> list[TableCell]:
    """Validate the worker's cell records; reject anything outside the grid."""
    cells: list[TableCell] = []
    for raw in raw_cells:
        if not isinstance(raw, dict):
            raise TableFormatError("cell must be an object")
        start_row = _as_int(raw.get("start_row"), "start_row")
        end_row = _as_int(raw.get("end_row"), "end_row")
        start_col = _as_int(raw.get("start_col"), "start_col")
        end_col = _as_int(raw.get("end_col"), "end_col")
        if not (0 <= start_row < end_row <= num_rows and 0 <= start_col < end_col <= num_cols):
            raise TableFormatError("cell span lies outside the table grid")
        cells.append(
            TableCell(
                text=_clean(raw.get("text")),
                start_row=start_row,
                end_row=end_row,
                start_col=start_col,
                end_col=end_col,
                column_header=bool(raw.get("column_header")),
                row_header=bool(raw.get("row_header")),
                row_section=bool(raw.get("row_section")),
            )
        )
    return cells


def _dedupe_consecutive(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and (not out or out[-1] != value):
            out.append(value)
    return out


def column_header_paths(cells: Sequence[TableCell], num_cols: int) -> list[str]:
    """For each column, the emitted column-header texts covering it, top to bottom."""
    headers = sorted((c for c in cells if c.column_header), key=lambda c: (c.start_row, c.start_col))
    paths: list[str] = []
    for col in range(num_cols):
        parts = _dedupe_consecutive([c.text for c in headers if c.covers_col(col)])
        paths.append(_HEADER_SEPARATOR.join(parts))
    return paths


def _header_only_rows(cells: Sequence[TableCell], num_rows: int) -> set[int]:
    rows = set()
    for row in range(num_rows):
        covering = [c for c in cells if c.covers_row(row)]
        if covering and all(c.column_header for c in covering):
            rows.add(row)
    return rows


def serialize_table(
    *,
    cells: Sequence[TableCell],
    num_rows: int,
    num_cols: int,
    page_number: Optional[int],
    table_index: int,
    caption: Optional[str] = None,
) -> str:
    """Return the index text for one table (empty string if it has no content)."""
    page_label = f"page {page_number}" if page_number is not None else "page ?"
    tag = f"[Table {table_index}, {page_label}]"
    col_paths = column_header_paths(cells, num_cols)
    header_rows = _header_only_rows(cells, num_rows)

    lines: list[str] = []
    heading = _clean(caption)
    if heading:
        lines.append(f"{tag} {heading}")
    named_cols = [p for p in col_paths if p]
    if named_cols:
        lines.append(f"{tag} Columns: " + " ; ".join(_dedupe_consecutive(named_cols)))

    scope_unit: Optional[str] = None
    for row in range(num_rows):
        if row in header_rows:
            continue
        covering = sorted((c for c in cells if c.covers_row(row)), key=lambda c: c.start_col)
        sections = [c.text for c in covering if c.row_section and c.text]
        if sections and all(c.row_section or not c.text for c in covering):
            section_text = _HEADER_SEPARATOR.join(_dedupe_consecutive(sections))
            # Every section heading opens a new unit scope: its own declared unit, or none.
            scope_unit = declared_unit(section_text)
            lines.append(f"{tag} Section: " + section_text)
            continue
        label = _HEADER_SEPARATOR.join(
            _dedupe_consecutive([c.text for c in covering if c.row_header and c.text])
        )
        has_values = any(
            c.text and not (c.row_header or c.column_header or c.row_section) for c in covering
        )
        if label and not has_values and declared_unit(label):
            # A label-only heading row that declares a unit opens that scope too.
            scope_unit = declared_unit(label)
        row_unit = declared_unit(label) or scope_unit
        fields: list[str] = []
        seen_cells: set[int] = set()
        for cell in covering:
            if cell.row_header or cell.column_header or cell.row_section or not cell.text:
                continue
            if id(cell) in seen_cells:
                continue
            seen_cells.add(id(cell))
            path_parts = _dedupe_consecutive(
                [col_paths[c] for c in range(cell.start_col, cell.end_col) if col_paths[c]]
            )
            path = apply_unit(_HEADER_SEPARATOR.join(path_parts), row_unit)
            fields.append(f"{path}: {cell.text}" if path else cell.text)
        if not fields and not label:
            continue
        line = tag
        if label:
            line += f" {label}"
        if fields:
            line += FIELD_SEPARATOR + FIELD_SEPARATOR.join(fields)
        lines.append(line)

    text = "\n".join(lines)
    if len(text) > MAX_TABLE_TEXT_CHARS:
        text = text[:MAX_TABLE_TEXT_CHARS]
    return text


def serialize_worker_table(raw_table: dict, page_number: Optional[int], table_index: int) -> str:
    """Validate one worker table record and serialize it."""
    if not isinstance(raw_table, dict):
        raise TableFormatError("table must be an object")
    num_rows = _as_int(raw_table.get("num_rows"), "num_rows")
    num_cols = _as_int(raw_table.get("num_cols"), "num_cols")
    if num_rows < 0 or num_cols < 0 or num_rows > 2000 or num_cols > 200:
        raise TableFormatError("table dimensions are out of range")
    raw_cells = raw_table.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) > 50_000:
        raise TableFormatError("cells must be a bounded list")
    cells = parse_cells(raw_cells, num_rows, num_cols)
    caption = raw_table.get("caption")
    return serialize_table(
        cells=cells,
        num_rows=num_rows,
        num_cols=num_cols,
        page_number=page_number,
        table_index=table_index,
        caption=caption if isinstance(caption, str) else None,
    )
