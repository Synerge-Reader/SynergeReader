from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

from document_parser import ParsedDocument, ParsedPage


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    chunk_index: int
    page_numbers: tuple[int, ...]

    @property
    def page_start(self) -> Optional[int]:
        return min(self.page_numbers) if self.page_numbers else None

    @property
    def page_end(self) -> Optional[int]:
        return max(self.page_numbers) if self.page_numbers else None


def _dedupe_preserve_order(values: Iterable[int]) -> list[int]:
    seen = set()
    ordered = []
    for v in values:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def _words_with_pages(pages: list[ParsedPage]) -> list[tuple[str, Optional[int]]]:
    words = []
    for page in pages:
        for word in page.text.split():
            words.append((word, page.page_number))
    return words


def chunk_document(
    document: ParsedDocument,
    max_chunk_size: int = 500,
) -> list[DocumentChunk]:
    words = _words_with_pages(document.pages)

    chunks: list[DocumentChunk] = []
    current_words: list[str] = []
    current_pages: list[Optional[int]] = []

    def flush() -> None:
        if not current_words:
            return
        page_numbers = tuple(
            _dedupe_preserve_order(p for p in current_pages if p is not None)
        )
        chunks.append(
            DocumentChunk(
                text=" ".join(current_words),
                chunk_index=len(chunks),
                page_numbers=page_numbers,
            )
        )

    for word, page_number in words:
        current_size = sum(len(w) for w in current_words) + len(current_words) - 1
        if current_size + len(word) + 1 <= max_chunk_size:
            current_words.append(word)
            current_pages.append(page_number)
        else:
            flush()
            current_words = [word]
            current_pages = [page_number]

    flush()
    return chunks


def build_chunk_locator(chunk: DocumentChunk, document_type: str) -> dict:
    if document_type == "pdf":
        return {
            "locator_type": "pdf_pages",
            "page_numbers": list(chunk.page_numbers),
        }
    return {"locator_type": document_type}
