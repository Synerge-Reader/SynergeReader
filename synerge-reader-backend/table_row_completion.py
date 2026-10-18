"""Bounded completion of table lines cut at a retrieved chunk's boundaries.

Enhanced extraction writes each table row as ONE line of a document's stored
text, opened by a tag naming its table and physical page (see
table_serialization.py), for example::

    [Table 1, page 4] Other reserves | Unaudited As at 31.05.2026 RM'000: (23,326) | Audited As at 31.05.2025 RM'000: (6,373)

The chunker windows the stored words at a fixed size and knows nothing of
lines, so one row can be split between two chunks: the first ends
"... RM'000: (23,326) | Audited As" and the next begins "at 31.05.2025
RM'000: (6,373)". Retrieval may supply either chunk without the other, and
then one value arrives without its column heading, or the other without its
row label.

This module puts the missing part of such a line back. It locates the
retrieved chunk inside the canonical stored text of the SAME document -- text
the caller has already loaded through its owner-scoped loader -- and returns
only the part of a table line that lies outside the chunk, at its start, its
end, or both. The chunker, the stored chunks and the ranking are untouched.

It refuses rather than guesses:

* the chunk must occur exactly once, word for word and word-aligned, in the
  stored text: no occurrence (missing or changed text) and several
  occurrences (ambiguous) are both refused;
* only a single serialized table line -- opened by a table tag and holding no
  other -- is completed, so prose is never extended and flattened text is
  never mistaken for one row;
* the tag's page must lie inside the chunk's own page range, so the added text
  never comes from a page the chunk's citation does not name;
* each side is bounded, and a side over the bound is refused whole rather than
  cut a second time.

``may_hold_table_text`` is a cheap test of whether a chunk shows table
structure at all, for a caller deciding whether a document is worth reading.
One cut leaves no such trace -- a chunk that opens with only the last value of
a row and continues into prose -- so a caller that already holds the
document's text should check every chunk of it; ``complete_fragment`` itself
needs no hint.

Pure: no route, no connection, no model.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

from table_serialization import FIELD_SEPARATOR, TABLE_LINE_TAG_RE


# One side of one chunk: the part of a single table line outside the chunk.
# Well above one serialized row of a wide statement (eight columns, each value
# carrying its full column-heading path, stays under 700 characters).
DEFAULT_MAX_SIDE_CHARS = 1_000
# Every completion in one answer, together.
DEFAULT_MAX_TOTAL_CHARS = 4_000

STORED_TEXT_SOURCE = "stored_document_text"

LEADING = "leading"
TRAILING = "trailing"


@dataclass(frozen=True)
class RowCompletionLimits:
    max_side_chars: int = DEFAULT_MAX_SIDE_CHARS
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS


DEFAULT_ROW_COMPLETION_LIMITS = RowCompletionLimits()


class CompletionStatus(str, Enum):
    COMPLETED = "completed"
    # Both ends of the chunk fall on a line end, or outside any table line.
    NOT_CUT = "not_cut"
    # The chunk does not occur in the stored text.
    SOURCE_MISSING = "source_missing"
    # The chunk occurs more than once in the stored text.
    AMBIGUOUS = "ambiguous"
    # A cut table line was found, but every completion of it was refused.
    REFUSED = "refused"


class RefusalReason(str, Enum):
    NOT_ONE_TABLE_LINE = "not_one_table_line"
    TOO_LONG = "too_long"
    PAGE_UNKNOWN = "page_unknown"
    PAGE_OUTSIDE_CHUNK = "page_outside_chunk"


@dataclass(frozen=True)
class LineCompletion:
    """The words of one table line that lie outside a chunk, and their source."""

    side: str  # LEADING (before the chunk) or TRAILING (after it)
    text: str  # whitespace-normalised, exactly as the stored line reads
    table: int
    page: int

    @property
    def cost(self) -> int:
        """Characters this completion adds to the evidence, joining space included."""
        return len(self.text) + 1

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "chars": len(self.text),
            "table": self.table,
            "page": self.page,
            "source": STORED_TEXT_SOURCE,
        }


@dataclass(frozen=True)
class FragmentCompletion:
    status: CompletionStatus
    lines: tuple[LineCompletion, ...] = ()
    refusals: tuple[tuple[str, RefusalReason], ...] = ()


_WORD = re.compile(r"\S+")
_TABLE_LINE = re.compile("^" + TABLE_LINE_TAG_RE.pattern, re.MULTILINE)


def may_hold_table_text(text: Optional[str]) -> bool:
    """Cheap test, before any document is loaded: could this chunk touch a table line?

    A chunk that touches a table line carries that line's tag or, when it lies
    inside one long row, the row's field separator.
    """
    return bool(text) and (TABLE_LINE_TAG_RE.search(text) is not None or FIELD_SEPARATOR in text)


class CanonicalText:
    """One document's stored text, indexed word by word.

    The chunker builds every chunk by joining the whitespace-separated words of
    the stored text with single spaces, so a chunk is found by matching its
    words, in order, against the same words of the stored text.
    """

    def __init__(self, text: str):
        self.text = text
        self._starts: list[int] = []
        self._ends: list[int] = []
        self._joined_starts: list[int] = []
        words: list[str] = []
        joined_length = 0
        for match in _WORD.finditer(text):
            self._starts.append(match.start())
            self._ends.append(match.end())
            self._joined_starts.append(joined_length)
            words.append(match.group())
            joined_length += len(match.group()) + 1
        self._joined = " ".join(words)

    @classmethod
    def with_table_lines(cls, text: object) -> Optional["CanonicalText"]:
        """The indexed text, or None when it holds no table line to complete."""
        if not isinstance(text, str) or _TABLE_LINE.search(text) is None:
            return None
        return cls(text)

    def locate(self, fragment: str) -> tuple[Optional[tuple[int, int]], Optional[CompletionStatus]]:
        """The fragment's character span in the stored text, or why there is none.

        Only a word-aligned occurrence counts: "at 31.05.2025" inside
        "flat 31.05.2025" is not an occurrence. Two occurrences are as much a
        refusal as none, because either could be the chunk.
        """
        needle = " ".join((fragment or "").split())
        if not needle:
            return None, CompletionStatus.SOURCE_MISSING
        last_word = needle.count(" ")
        found: list[int] = []
        position = self._joined.find(needle)
        while position >= 0 and len(found) < 2:
            end = position + len(needle)
            word = bisect.bisect_left(self._joined_starts, position)
            starts_on_word = word < len(self._joined_starts) and self._joined_starts[word] == position
            ends_on_word = end == len(self._joined) or self._joined[end] == " "
            if starts_on_word and ends_on_word:
                found.append(word)
            position = self._joined.find(needle, position + 1)
        if not found:
            return None, CompletionStatus.SOURCE_MISSING
        if len(found) > 1:
            return None, CompletionStatus.AMBIGUOUS
        first = found[0]
        return (self._starts[first], self._ends[first + last_word]), None

    def line_bounds(self, offset: int) -> tuple[int, int]:
        """Start and end of the line holding the character at ``offset``."""
        start = self.text.rfind("\n", 0, offset) + 1
        end = self.text.find("\n", offset)
        return start, len(self.text) if end < 0 else end


def _page_number(value: object) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def complete_fragment(
    source: CanonicalText,
    fragment: str,
    page_start: object,
    page_end: object,
    max_side_chars: int = DEFAULT_MAX_SIDE_CHARS,
) -> FragmentCompletion:
    """The parts of table lines cut at the fragment's start or end, from ``source``.

    ``page_start``/``page_end`` are the chunk's own page range; a completion is
    granted only from a table line whose tag names a page inside it.
    """
    span, missing = source.locate(fragment)
    if span is None:
        return FragmentCompletion(status=missing or CompletionStatus.SOURCE_MISSING)
    start, end = span
    first_page = _page_number(page_start)
    last_page = _page_number(page_end)
    lines: list[LineCompletion] = []
    refusals: list[tuple[str, RefusalReason]] = []

    def consider(side: str, line_start: int, line_end: int, outside: str) -> None:
        words = " ".join(outside.split())
        if not words:
            return  # the chunk starts or ends exactly at a line boundary
        tag = TABLE_LINE_TAG_RE.match(source.text, line_start, line_end)
        if tag is None:
            return  # not a table line: prose is never extended
        page = tag.group("page")
        if TABLE_LINE_TAG_RE.search(source.text, tag.end(), line_end) is not None:
            refusals.append((side, RefusalReason.NOT_ONE_TABLE_LINE))
        elif not page.isdigit() or first_page is None or last_page is None:
            refusals.append((side, RefusalReason.PAGE_UNKNOWN))
        elif not first_page <= int(page) <= last_page:
            refusals.append((side, RefusalReason.PAGE_OUTSIDE_CHUNK))
        elif len(words) > max_side_chars:
            refusals.append((side, RefusalReason.TOO_LONG))
        else:
            lines.append(LineCompletion(side=side, text=words, table=int(tag.group("table")), page=int(page)))

    line_start, line_end = source.line_bounds(start)
    consider(LEADING, line_start, line_end, source.text[line_start:start])
    line_start, line_end = source.line_bounds(end - 1)
    consider(TRAILING, line_start, line_end, source.text[end:line_end])

    if lines:
        status = CompletionStatus.COMPLETED
    elif refusals:
        status = CompletionStatus.REFUSED
    else:
        status = CompletionStatus.NOT_CUT
    return FragmentCompletion(status=status, lines=tuple(lines), refusals=tuple(refusals))


def completed_text(fragment: str, lines: Sequence[LineCompletion]) -> str:
    """The fragment with the given line completions joined on, in reading order."""
    text = fragment
    for line in lines:
        text = f"{line.text} {text}" if line.side == LEADING else f"{text} {line.text}"
    return text
