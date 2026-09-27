"""NDJSON event transport for the /ask response stream.

Why this is its own module: the /ask stream now has to carry structured
citation records alongside generated text, and the previous transport encoded
control information as magic sentinels (``__CONTEXT__``, ``__READY__``,
``__ENTRY_ID__``, ``__ERROR__``) inside the same plain-text stream as the
answer. That conflates generated text with transport control -- an answer that
happens to contain ``__CONTEXT__`` corrupts the stream -- and it cannot carry a
citation record at all. Keeping the encoder/decoder here, rather than in
``main.py``, is what lets the transport be unit-tested without importing the
application (no route, no database, no model).

The wire format is newline-delimited JSON: one complete JSON object per line,
each independently parseable. Generated answer text only ever appears as a
JSON string value inside a ``delta`` event, so no answer content can be
mistaken for a control instruction.

Event types:

``evidence``      structured citation records for this answer, sent before the
                  first token so the UI can render sources while streaming.
``delta``         one piece of answer text, in ``text``.
``verification``  structured per-claim verification results.
``entry_id``      the chat_history row id for this answer (unchanged semantics).
``error``         a fixed, safe error code and message. Never raw exceptions.
``done``          explicit terminator. Its absence means the stream was cut.
"""

from __future__ import annotations

import json
from typing import Any, Iterator, Optional, Sequence


STREAM_CONTRACT_VERSION = "e2.askstream.v1"
STREAM_MEDIA_TYPE = "application/x-ndjson"

EVENT_EVIDENCE = "evidence"
EVENT_DELTA = "delta"
EVENT_VERIFICATION = "verification"
EVENT_ENTRY_ID = "entry_id"
EVENT_ERROR = "error"
EVENT_DONE = "done"

EVENT_TYPES = frozenset(
    {
        EVENT_EVIDENCE,
        EVENT_DELTA,
        EVENT_VERIFICATION,
        EVENT_ENTRY_ID,
        EVENT_ERROR,
        EVENT_DONE,
    }
)


def encode_event(event_type: str, **payload: Any) -> str:
    """One NDJSON line for ``event_type``.

    ``ensure_ascii`` is left on so the line is byte-safe over any transport,
    and the payload is JSON-encoded, so text containing newlines, quotes, or
    old sentinel strings cannot break the line framing.
    """
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown ask-stream event type: {event_type!r}")
    event = {"type": event_type}
    event.update(payload)
    return json.dumps(event, ensure_ascii=True) + "\n"


def evidence_event(
    citations: Sequence[dict],
    *,
    mode: str,
    truncated: bool = False,
    warnings: Sequence[str] = (),
) -> str:
    return encode_event(
        EVENT_EVIDENCE,
        citations=list(citations),
        mode=mode,
        truncated=bool(truncated),
        warnings=list(warnings),
    )


def delta_event(text: str) -> str:
    return encode_event(EVENT_DELTA, text=text)


def verification_event(
    claims: Sequence[dict],
    *,
    invalid_citation_ids: Sequence[str] = (),
    used_citation_ids: Sequence[str] = (),
) -> str:
    """Per-claim results, plus the citations the answer actually used.

    ``used_citation_ids`` is the registry ids the generated answer really cited,
    deduplicated and in first-use order. The client renders source cards from
    this list rather than from the candidate evidence, so an answer that used
    one of eight retrieved passages shows one source, not eight.
    """
    return encode_event(
        EVENT_VERIFICATION,
        claims=list(claims),
        invalid_citation_ids=list(invalid_citation_ids),
        used_citation_ids=list(used_citation_ids),
    )


def entry_id_event(entry_id: Any) -> str:
    return encode_event(EVENT_ENTRY_ID, entry_id=entry_id)


def error_event(code: str, message: str) -> str:
    """A safe error line. Callers pass a fixed code/message pair only."""
    return encode_event(EVENT_ERROR, code=code, message=message)


def done_event(*, ok: bool = True) -> str:
    return encode_event(EVENT_DONE, ok=bool(ok))


class EventStreamDecoder:
    """Reassembles NDJSON events across arbitrary chunk boundaries.

    A network chunk can split a line anywhere, including inside a JSON string
    or between the two bytes of an escape sequence, so partial input is held in
    a buffer until a newline completes it. This is the Python mirror of the
    decoder the frontend uses; keeping both means the framing rule is tested on
    both sides.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self.malformed_lines = 0

    def feed(self, chunk: str) -> list[dict]:
        self._buffer += chunk
        events: list[dict] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            event = self._decode(line)
            if event is not None:
                events.append(event)
        return events

    def close(self) -> list[dict]:
        """Decode whatever is left, for a stream that ended without a newline."""
        remaining, self._buffer = self._buffer, ""
        event = self._decode(remaining)
        return [event] if event is not None else []

    def _decode(self, line: str) -> Optional[dict]:
        stripped = line.strip()
        if not stripped:
            return None
        try:
            event = json.loads(stripped)
        except Exception:
            # A malformed line is counted and dropped. It is never treated as
            # answer text, which would let a broken frame print as content.
            self.malformed_lines += 1
            return None
        if not isinstance(event, dict) or event.get("type") not in EVENT_TYPES:
            self.malformed_lines += 1
            return None
        return event


def iter_events(payload: str) -> Iterator[dict]:
    """Decode a complete NDJSON payload."""
    decoder = EventStreamDecoder()
    for event in decoder.feed(payload):
        yield event
    for event in decoder.close():
        yield event
