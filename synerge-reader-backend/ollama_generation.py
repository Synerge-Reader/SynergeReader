"""Reader for Ollama's native ``/api/generate`` NDJSON stream.

Shared by the ``/ask`` route and the operator diagnostic harness, so both judge
a generation by the same rules. Pure: it parses chunks it is handed and never
opens a connection.

What it guarantees:

* Only ``response`` text is ever returned as answer text. A separate
  ``thinking`` field is counted, never returned, so model reasoning cannot
  reach the rendered or persisted answer, claim extraction, citations, or JSON.
* Network chunks may split a line, or a multi-byte UTF-8 character, anywhere.
  Bytes go through an incremental UTF-8 decoder rather than being decoded one
  chunk at a time.
* A stream only counts as finished when Ollama's final ``"done": true`` record
  arrived. An ``{"error": ...}`` line, a malformed line, a missing final
  record, or ``done_reason == "length"`` (the output limit was reached) is
  reported through ``failure``; after an error or malformed line no further
  text is returned. Error text from the server is never kept.
"""

from __future__ import annotations

import codecs
import json
from typing import Optional

FAILURE_MODEL_ERROR = "model_error"
FAILURE_MALFORMED = "malformed"
FAILURE_INCOMPLETE = "incomplete"
FAILURE_OUTPUT_LIMIT = "output_limit"

# Counters and durations from the final record. Durations are nanoseconds.
METRIC_FIELDS = (
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
)


class OllamaGenerateStream:
    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""
        self.done = False
        self.done_reason: Optional[str] = None
        self.model_error = False
        self.malformed_lines = 0
        self.thinking_chars = 0
        self.response_chars = 0
        self.metrics: dict = {}

    @property
    def _stopped(self) -> bool:
        return self.model_error or self.malformed_lines > 0

    def feed(self, chunk) -> list:
        """Answer text completed by this chunk, in order."""
        if isinstance(chunk, (bytes, bytearray)):
            try:
                text = self._decoder.decode(bytes(chunk))
            except UnicodeDecodeError:
                self.malformed_lines += 1
                return []
        elif isinstance(chunk, str):
            text = chunk
        else:
            self.malformed_lines += 1
            return []
        self._buffer += text
        tokens = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            tokens.extend(self._line(line))
        return tokens

    def close(self) -> list:
        """Answer text from a final line that had no trailing newline."""
        try:
            tail = self._decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            self.malformed_lines += 1
            tail = ""
        remaining, self._buffer = self._buffer + tail, ""
        return self._line(remaining)

    def _line(self, line: str) -> list:
        stripped = line.strip()
        if not stripped or self._stopped:
            return []
        if self.done:
            # Nothing may follow the final record.
            self.malformed_lines += 1
            return []
        try:
            record = json.loads(stripped)
        except ValueError:
            self.malformed_lines += 1
            return []
        if not isinstance(record, dict):
            self.malformed_lines += 1
            return []
        if "error" in record:
            self.model_error = True
            return []

        thinking = record.get("thinking")
        if isinstance(thinking, str):
            self.thinking_chars += len(thinking)

        token = record.get("response", "")
        if not isinstance(token, str):
            self.malformed_lines += 1
            return []

        if record.get("done") is True:
            self.done = True
            reason = record.get("done_reason")
            self.done_reason = reason if isinstance(reason, str) else None
            for name in METRIC_FIELDS:
                value = record.get(name)
                if isinstance(value, int) and not isinstance(value, bool):
                    self.metrics[name] = value

        if token:
            self.response_chars += len(token)
            return [token]
        return []

    @property
    def failure(self) -> Optional[str]:
        """Why this generation cannot count as a finished answer, or None."""
        if self.model_error:
            return FAILURE_MODEL_ERROR
        if self.malformed_lines:
            return FAILURE_MALFORMED
        if not self.done:
            return FAILURE_INCOMPLETE
        if self.done_reason == "length":
            return FAILURE_OUTPUT_LIMIT
        return None
