"""Route-independent answer-evidence planning for E2.

This module decides WHAT information is supplied to the language model. It is
deliberately not a citation module: it produces a finalized ``EvidenceBundle``
and stops there. Registering citation identifiers, building the generation
prompt, parsing citation markers, and verifying claims all belong to
``citation_generation.py``, which consumes the bundle this module returns and
never re-decides selected text versus complete document versus retrieval.

The binding answer-context priority implemented here, in order:

1. Explicit user-selected text. If the user highlighted something, that is the
   answer context, whatever else is in scope.
2. A complete short authorized document. Exactly one authorized document in
   scope whose full text fits the configured limit is supplied whole.
3. Authorized hybrid retrieval. Multiple documents, or one document that
   exceeds the complete-document limit.

Everything the planner loads from the server is authorization-scoped by the
caller: the planner is handed an ``AuthorizedScope`` and can only ever see the
documents inside it. Client-supplied document ids and filenames are treated as
requests, not as proof: an id that is not in the authorized scope is dropped,
and a selection that claims an unauthorized document keeps its text but loses
its document attribution, so it can never be cited as another user's document.

This module imports no route, opens no connection, starts no thread, and
contacts no model. Document loading and retrieval are injected callables.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


EVIDENCE_CONTRACT_VERSION = "e2.evidence.v1"


# --- Named limits -----------------------------------------------------------
#
# These replace the scattered 14000/16000/12000/8000 magic numbers that used to
# live in the /ask route and in the frontend's combined-document dump. The
# defaults are deliberately conservative: they keep a local 8B-class model's
# prompt responsive rather than maximising recall.

# A single document is supplied whole only if its complete text is at most this
# long. Above it the document is retrieved from, never truncated-and-called-
# complete.
DEFAULT_COMPLETE_DOCUMENT_CHAR_LIMIT = 12_000

# Hard ceiling on the total characters of evidence text placed in one prompt,
# across every evidence item.
DEFAULT_TOTAL_EVIDENCE_CHAR_LIMIT = 16_000

# Hard ceiling on how many evidence items (and therefore citations) one answer
# may be built from.
DEFAULT_MAX_EVIDENCE_ITEMS = 8

# Maximum length of the excerpt shown next to a citation in the UI. Defined
# here, with the other evidence limits, and imported by citation_generation so
# the two modules cannot drift apart.
DEFAULT_MAX_CITATION_EXCERPT_CHARS = 480

TRUNCATION_NOTE = "[Truncated to keep the prompt responsive.]"

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class EvidenceLimits:
    complete_document_chars: int = DEFAULT_COMPLETE_DOCUMENT_CHAR_LIMIT
    total_evidence_chars: int = DEFAULT_TOTAL_EVIDENCE_CHAR_LIMIT
    max_evidence_items: int = DEFAULT_MAX_EVIDENCE_ITEMS
    max_citation_excerpt_chars: int = DEFAULT_MAX_CITATION_EXCERPT_CHARS


DEFAULT_LIMITS = EvidenceLimits()


class EvidenceMode(str, Enum):
    SELECTED_TEXT = "selected_text"
    COMPLETE_DOCUMENT = "complete_document"
    HYBRID_RETRIEVAL = "hybrid_retrieval"
    NONE = "none"


class EvidenceWarning(str, Enum):
    SELECTION_SCOPE_UNVERIFIED = "selection_scope_unverified"
    DOCUMENT_NOT_AUTHORIZED = "document_not_authorized"
    DOCUMENT_TOO_LONG_FOR_COMPLETE = "document_too_long_for_complete"
    EVIDENCE_TRUNCATED = "evidence_truncated"
    EVIDENCE_ITEM_LIMIT_REACHED = "evidence_item_limit_reached"
    RETRIEVAL_UNAVAILABLE = "retrieval_unavailable"
    NO_AUTHORIZED_DOCUMENTS = "no_authorized_documents"


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "No authorized document evidence is available for this question."
)


@dataclass(frozen=True)
class AuthorizedDocument:
    """One document the current caller is allowed to read.

    ``filename`` is mandatory: a citation must never be rendered as a bare
    database id, so a document with no filename cannot be cited truthfully and
    is rejected at construction time.
    """

    document_id: Any
    filename: str
    title: Optional[str] = None
    char_length: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.filename, str) or not self.filename.strip():
            raise ValueError("an authorized document must carry a filename")

    @property
    def display_name(self) -> str:
        return self.title.strip() if (self.title or "").strip() else self.filename


@dataclass(frozen=True)
class AuthorizedScope:
    """The complete set of documents the current caller may be shown.

    ``established`` is False when authorization could not be determined at all
    (for example the token lookup failed). The planner then fails closed: no
    server-loaded evidence, whatever the client asked for.
    """

    documents: tuple[AuthorizedDocument, ...] = ()
    established: bool = True
    user_id: Any = None
    anonymous: bool = True

    @classmethod
    def unresolved(cls) -> "AuthorizedScope":
        return cls(documents=(), established=False, user_id=None, anonymous=False)

    @property
    def by_id(self) -> dict[Any, AuthorizedDocument]:
        return {document.document_id: document for document in self.documents}

    def get(self, document_id: Any) -> Optional[AuthorizedDocument]:
        if document_id is None:
            return None
        return self.by_id.get(document_id)

    def authorizes(self, document_id: Any) -> bool:
        return self.get(document_id) is not None


@dataclass(frozen=True)
class SelectedTextInput:
    """One explicit highlight, as the client reported it.

    ``document_id`` and ``filename`` are client claims. They are validated
    against the authorized scope before they are allowed to reach an evidence
    item.
    """

    text: str
    selection_id: Optional[str] = None
    document_id: Any = None
    filename: Optional[str] = None
    locator_json: Optional[Mapping[str, Any]] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None


@dataclass(frozen=True)
class EvidenceItem:
    """One unit of evidence, carrying everything a citation needs."""

    text: str
    source_type: str
    document_id: Any = None
    filename: Optional[str] = None
    chunk_id: Any = None
    chunk_index: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    locator_json: Optional[Mapping[str, Any]] = None
    semantic_score: Optional[float] = None
    lexical_score: Optional[float] = None
    combined_score: Optional[float] = None
    truncated: bool = False
    selection_id: Optional[str] = None

    @property
    def dedupe_key(self) -> tuple:
        """Identity used to collapse duplicate or repeated evidence.

        Chunk identity wins when it exists, because the same chunk retrieved by
        both the semantic and the lexical ranker is one piece of evidence. When
        there is no chunk identity (selections, complete documents) the
        whitespace-normalised text hash is used, so the same passage pasted
        twice does not become two citations.
        """
        if self.chunk_id is not None:
            return ("chunk", self.document_id, self.chunk_id)
        if self.document_id is not None and self.chunk_index is not None:
            return ("chunk_index", self.document_id, self.chunk_index)
        normalized = _WHITESPACE.sub(" ", self.text or "").strip().lower()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return ("text", self.document_id, digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_type": self.source_type,
            "document_id": self.document_id,
            "filename": self.filename,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "locator_json": dict(self.locator_json) if self.locator_json else None,
            "semantic_score": self.semantic_score,
            "lexical_score": self.lexical_score,
            "combined_score": self.combined_score,
            "truncated": self.truncated,
            "selection_id": self.selection_id,
        }


@dataclass(frozen=True)
class EvidenceBundle:
    """The finalized evidence for one answer. Citation generation starts here."""

    mode: EvidenceMode
    items: tuple[EvidenceItem, ...] = ()
    truncated: bool = False
    limits: EvidenceLimits = DEFAULT_LIMITS
    contract_version: str = EVIDENCE_CONTRACT_VERSION

    @property
    def total_chars(self) -> int:
        return sum(len(item.text) for item in self.items)

    @property
    def document_ids(self) -> tuple:
        seen: list[Any] = []
        for item in self.items:
            if item.document_id is not None and item.document_id not in seen:
                seen.append(item.document_id)
        return tuple(seen)

    @property
    def is_empty(self) -> bool:
        return not self.items

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "items": [item.to_dict() for item in self.items],
            "truncated": self.truncated,
            "total_chars": self.total_chars,
            "contract_version": self.contract_version,
        }


@dataclass(frozen=True)
class EvidencePlanningResult:
    bundle: EvidenceBundle
    warnings: tuple[str, ...] = ()
    message: Optional[str] = None

    @property
    def mode(self) -> EvidenceMode:
        return self.bundle.mode

    @property
    def insufficient(self) -> bool:
        return self.bundle.is_empty

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "insufficient": self.insufficient,
            "warnings": list(self.warnings),
            "message": self.message,
            "bundle": self.bundle.to_dict(),
        }


@dataclass(frozen=True)
class EvidenceRequest:
    question: str
    selected_text: str = ""
    selections: tuple[SelectedTextInput, ...] = ()
    requested_document_ids: tuple = ()
    top_k: int = 8


def _clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _dedupe(items: Iterable[EvidenceItem]) -> list[EvidenceItem]:
    """Collapse duplicate evidence, keeping first position and best scores.

    Order is the caller's order, which is already deterministic: fusion rank
    for retrieval, client order for selections.
    """
    merged: dict[tuple, EvidenceItem] = {}
    order: list[tuple] = []
    for item in items:
        key = item.dedupe_key
        if key not in merged:
            merged[key] = item
            order.append(key)
            continue
        kept = merged[key]
        merged[key] = replace(
            kept,
            semantic_score=_max_optional(kept.semantic_score, item.semantic_score),
            lexical_score=_max_optional(kept.lexical_score, item.lexical_score),
            combined_score=_max_optional(kept.combined_score, item.combined_score),
            filename=kept.filename or item.filename,
            page_start=kept.page_start if kept.page_start is not None else item.page_start,
            page_end=kept.page_end if kept.page_end is not None else item.page_end,
            locator_json=kept.locator_json or item.locator_json,
        )
    return [merged[key] for key in order]


def _max_optional(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def apply_budget(
    items: Sequence[EvidenceItem],
    limits: EvidenceLimits,
) -> tuple[tuple[EvidenceItem, ...], bool, list[str]]:
    """Fit evidence into the configured budget, deterministically.

    Items are taken in order until either the item limit or the character
    budget is reached. The item that crosses the character budget is cut to the
    remaining room and flagged ``truncated`` rather than dropped, so the most
    relevant evidence is never silently lost; once the budget is exhausted no
    further item is admitted. The same input always yields the same output.
    """
    kept: list[EvidenceItem] = []
    warnings: list[str] = []
    truncated_any = False
    used = 0

    for item in items:
        if len(kept) >= limits.max_evidence_items:
            warnings.append(EvidenceWarning.EVIDENCE_ITEM_LIMIT_REACHED.value)
            break
        remaining = limits.total_evidence_chars - used
        if remaining <= 0:
            warnings.append(EvidenceWarning.EVIDENCE_TRUNCATED.value)
            break
        text = item.text or ""
        if len(text) > remaining:
            text = text[:remaining]
            truncated_any = True
            kept.append(replace(item, text=text, truncated=True))
            used += len(text)
            warnings.append(EvidenceWarning.EVIDENCE_TRUNCATED.value)
            continue
        kept.append(item)
        used += len(text)
        if item.truncated:
            truncated_any = True

    # Stable, de-duplicated warnings.
    unique: list[str] = []
    for warning in warnings:
        if warning not in unique:
            unique.append(warning)
    return tuple(kept), truncated_any, unique


@dataclass
class AnswerEvidencePlanner:
    """Applies the binding answer-context priority to an authorized scope.

    ``load_document_text`` and ``retrieve`` are injected so this class can be
    unit-tested without a database, an embedding provider, or a route. Both are
    only ever called with document ids that are already inside the authorized
    scope handed to :meth:`plan`.
    """

    load_document_text: Callable[[Any], Optional[str]]
    retrieve: Callable[[str, Sequence[Any], int], Sequence[EvidenceItem]]
    limits: EvidenceLimits = DEFAULT_LIMITS
    # Exception types the caller wants to handle itself rather than have
    # degraded into "no evidence" -- an embedding-provider outage, for example,
    # is a service failure the route reports as such, not an empty result.
    propagate_exceptions: tuple = ()

    # -- priority level 1 ----------------------------------------------------
    def _selected_text_items(
        self,
        request: EvidenceRequest,
        scope: AuthorizedScope,
        warnings: list[str],
    ) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for index, selection in enumerate(request.selections):
            text = _clean(selection.text)
            if not text:
                continue
            document = scope.get(selection.document_id)
            if selection.document_id is not None and document is None:
                # The client claimed a document this caller may not read. The
                # highlighted text is still the user's own screen content, so
                # it stays as evidence -- but stripped of the attribution, so
                # nothing can be cited as that document.
                _note(warnings, EvidenceWarning.SELECTION_SCOPE_UNVERIFIED)
            items.append(
                EvidenceItem(
                    text=text,
                    source_type="selection",
                    document_id=document.document_id if document else None,
                    filename=document.filename if document else None,
                    page_start=selection.page_start if document else None,
                    page_end=selection.page_end if document else None,
                    locator_json=selection.locator_json if document else None,
                    selection_id=selection.selection_id or f"selection-{index + 1}",
                )
            )

        loose = _clean(request.selected_text)
        if loose and not any(item.text == loose for item in items):
            items.append(
                EvidenceItem(
                    text=loose,
                    source_type="selection",
                    selection_id=f"selection-{len(items) + 1}",
                )
            )
        return items

    # -- priority level 2 ----------------------------------------------------
    def _complete_document_items(
        self,
        scoped: Sequence[AuthorizedDocument],
        warnings: list[str],
    ) -> Optional[list[EvidenceItem]]:
        if len(scoped) != 1:
            return None
        document = scoped[0]
        text = self.load_document_text(document.document_id) or ""
        if not text.strip():
            return None
        if len(text) > self.limits.complete_document_chars:
            # Never describe a truncated document as complete: fall through to
            # retrieval instead of cutting it and calling it the whole thing.
            _note(warnings, EvidenceWarning.DOCUMENT_TOO_LONG_FOR_COMPLETE)
            return None
        return [
            EvidenceItem(
                text=text,
                source_type="complete_document",
                document_id=document.document_id,
                filename=document.filename,
                truncated=False,
            )
        ]

    # -- priority level 3 ----------------------------------------------------
    def _retrieved_items(
        self,
        request: EvidenceRequest,
        scoped: Sequence[AuthorizedDocument],
        warnings: list[str],
    ) -> list[EvidenceItem]:
        document_ids = [document.document_id for document in scoped]
        if not document_ids:
            return []
        try:
            retrieved = self.retrieve(request.question, document_ids, request.top_k)
        except self.propagate_exceptions:
            raise
        except Exception:
            # The caller decides how to report infrastructure failure; the
            # planner only reports that no evidence could be gathered.
            _note(warnings, EvidenceWarning.RETRIEVAL_UNAVAILABLE)
            return []

        authorized: list[EvidenceItem] = []
        by_id = {document.document_id: document for document in scoped}
        for item in retrieved:
            document = by_id.get(item.document_id)
            if document is None:
                # Defence in depth: a retrieval implementation that returned a
                # chunk outside the authorized scope is dropped here too.
                _note(warnings, EvidenceWarning.DOCUMENT_NOT_AUTHORIZED)
                continue
            authorized.append(replace(item, filename=item.filename or document.filename))
        return authorized

    # -- entry point ---------------------------------------------------------
    def plan(
        self,
        request: EvidenceRequest,
        scope: AuthorizedScope,
    ) -> EvidencePlanningResult:
        warnings: list[str] = []

        if not scope.established:
            # Fail closed: authorization could not be established at all.
            return EvidencePlanningResult(
                bundle=EvidenceBundle(mode=EvidenceMode.NONE, limits=self.limits),
                warnings=(EvidenceWarning.NO_AUTHORIZED_DOCUMENTS.value,),
                message=INSUFFICIENT_EVIDENCE_MESSAGE,
            )

        # Priority 1: explicit selection beats everything else in scope.
        selection_items = self._selected_text_items(request, scope, warnings)
        if selection_items:
            return self._finalize(EvidenceMode.SELECTED_TEXT, selection_items, warnings)

        scoped = self._scoped_documents(request, scope, warnings)
        if not scoped:
            _note(warnings, EvidenceWarning.NO_AUTHORIZED_DOCUMENTS)
            return EvidencePlanningResult(
                bundle=EvidenceBundle(mode=EvidenceMode.NONE, limits=self.limits),
                warnings=tuple(warnings),
                message=INSUFFICIENT_EVIDENCE_MESSAGE,
            )

        # Priority 2: exactly one authorized document that fits, supplied whole.
        complete = self._complete_document_items(scoped, warnings)
        if complete:
            return self._finalize(EvidenceMode.COMPLETE_DOCUMENT, complete, warnings)

        # Priority 3: authorized hybrid retrieval.
        retrieved = self._retrieved_items(request, scoped, warnings)
        if not retrieved:
            return EvidencePlanningResult(
                bundle=EvidenceBundle(mode=EvidenceMode.NONE, limits=self.limits),
                warnings=tuple(warnings),
                message=INSUFFICIENT_EVIDENCE_MESSAGE,
            )
        return self._finalize(EvidenceMode.HYBRID_RETRIEVAL, retrieved, warnings)

    def _scoped_documents(
        self,
        request: EvidenceRequest,
        scope: AuthorizedScope,
        warnings: list[str],
    ) -> list[AuthorizedDocument]:
        """Requested ids intersected with the authorized scope.

        An id the caller is not authorized for is dropped with a warning and
        never distinguished in the response from an id that does not exist, so
        another user's document cannot be probed for existence.
        """
        if not request.requested_document_ids:
            return list(scope.documents)
        scoped: list[AuthorizedDocument] = []
        for document_id in request.requested_document_ids:
            document = scope.get(document_id)
            if document is None:
                _note(warnings, EvidenceWarning.DOCUMENT_NOT_AUTHORIZED)
                continue
            if document not in scoped:
                scoped.append(document)
        return scoped

    def _finalize(
        self,
        mode: EvidenceMode,
        items: Sequence[EvidenceItem],
        warnings: list[str],
    ) -> EvidencePlanningResult:
        deduped = _dedupe(items)
        kept, truncated, budget_warnings = apply_budget(deduped, self.limits)
        for warning in budget_warnings:
            if warning not in warnings:
                warnings.append(warning)
        bundle = EvidenceBundle(
            mode=mode if kept else EvidenceMode.NONE,
            items=kept,
            truncated=truncated,
            limits=self.limits,
        )
        return EvidencePlanningResult(
            bundle=bundle,
            warnings=tuple(warnings),
            message=INSUFFICIENT_EVIDENCE_MESSAGE if bundle.is_empty else None,
        )


def _note(warnings: list[str], warning: EvidenceWarning) -> None:
    if warning.value not in warnings:
        warnings.append(warning.value)
