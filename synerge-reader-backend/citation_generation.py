"""Route-independent citation generation for E2.

This module owns everything downstream of evidence selection:

* registering the evidence that was actually supplied to the model, with
  deterministic response-local identifiers ``[C1]``, ``[C2]``, ...;
* building the generation prompt from those registered evidence blocks;
* parsing the generated answer and validating every citation marker against the
  registry;
* splitting the answer into claims and attaching a verification status to each.

It deliberately does NOT decide what evidence to use. It consumes a finalized
``EvidenceBundle`` from ``answer_evidence.py``; the selected-text / complete-
document / hybrid-retrieval priority is that module's responsibility and is
never re-litigated here.

The module imports no route, opens no connection, and contacts no model. The
claim verifier is injected, so the whole pipeline is unit-testable without a
live model; the runtime verifier lives with the application's Ollama transport.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

from answer_evidence import (
    DEFAULT_MAX_CITATION_EXCERPT_CHARS,
    DEFAULT_TOTAL_EVIDENCE_CHAR_LIMIT,
    EvidenceBundle,
    EvidenceItem,
    EvidenceMode,
)


class EvidenceBudgetExceeded(ValueError):
    """A bundle arrived carrying more evidence text than one prompt may hold.

    Raised, never silently repaired. Trimming, reordering, or dropping evidence
    here would be an evidence-selection decision, and that belongs exclusively
    to answer_evidence.AnswerEvidencePlanner.
    """


CITATION_CONTRACT_VERSION = "e2.citation.v1"

CITATION_MARKER_PATTERN = re.compile(r"\[C(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")

# A sentence that only talks ABOUT the citation machinery is not a claim about
# the documents, and must not consume one of the few verification slots. The
# pattern deliberately matches the WHOLE marker-stripped sentence, so an
# ordinary factual sentence that happens to mention a citation is untouched.
_CITATION_MECHANICS = re.compile(
    r"^(?:this|that|it|the\s+(?:citation|source|evidence|passage|above))?\s*"
    r"(?:is|are|was|were)?\s*"
    r"(?:cited|referenced|provided|listed|used)\s+"
    r"(?:as|for)\s+(?:the\s+)?(?:evidence|support|source|reference)"
    r"(?:\s+(?:for|of)\s+(?:this|that|the)(?:\s+\w+){0,3})?\s*[.!]?$",
    re.IGNORECASE,
)
# A fragment that carries citation markers and nothing else. The model
# occasionally writes the marker AFTER the full stop ("... is stated. [C1]"),
# and the sentence splitter then hands that marker over as its own fragment.
# It asserts nothing, so it must never become a claim; discarding it outright
# would silently strip the citation off the claim it belongs to.
_MARKER_ONLY_RESIDUE = re.compile(r"^[\s.,;:!?)\]}–—]*$")
# The claim's own terminal punctuation, with any closing quote or bracket, so a
# re-attached marker lands immediately before it rather than after it.
_TRAILING_TERMINAL_PUNCTUATION = re.compile(r"([.!?]+[\"'\)\]]*)\s*$")
_WHITESPACE = re.compile(r"\s+")

# A citation must never be shown as a bare database id, and a page number must
# never be invented. When no page is known the most truthful locator available
# is used instead, in this order of preference.
LOCATION_UNAVAILABLE_LABEL = "location unavailable"
SELECTION_LOCATION_LABEL = "highlighted selection"
COMPLETE_DOCUMENT_LOCATION_LABEL = "complete document"
# A chunk index is an implementation detail of ingestion, not a place a reader
# can turn to. Evidence with no page, paragraph or line is described by what it
# truthfully is -- a passage of the named document -- and the reader is taken
# to it by matching the excerpt text, not by a fabricated coordinate.
PASSAGE_LOCATION_LABEL = "Relevant passage"


class ClaimVerificationStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    UNVERIFIED = "unverified"


class ClaimReason(str, Enum):
    NO_CITATION = "no_citation"
    INVALID_CITATION = "invalid_citation"
    VERIFIER_UNAVAILABLE = "verifier_unavailable"
    VERIFIER_MALFORMED = "verifier_malformed"
    NOT_VERIFIED = "not_verified"
    VERIFIED = "verified"
    # The verifier said more than the cited evidence can carry: a number,
    # model name, dataset name or section identifier the claim states does
    # not appear in the evidence it cited. The status was lowered, never
    # raised, so this can only ever make an answer less green.
    EVIDENCE_TERMS_MISSING = "evidence_terms_missing"


@dataclass(frozen=True)
class CitationLimits:
    max_excerpt_chars: int = DEFAULT_MAX_CITATION_EXCERPT_CHARS
    max_claims_verified: int = 6
    # Defensive ceiling on the evidence text placed in one prompt. The planner's
    # budget in answer_evidence.py is authoritative and already enforces this;
    # the check here exists so a bundle that somehow arrives over budget is
    # REJECTED rather than quietly trimmed, reordered, or reselected -- this
    # module must never make an evidence-selection decision.
    max_prompt_evidence_chars: int = DEFAULT_TOTAL_EVIDENCE_CHAR_LIMIT


DEFAULT_CITATION_LIMITS = CitationLimits()


@dataclass(frozen=True)
class NormalizedLocator:
    kind: str
    label: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }


def _coerce_page(value: object) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    return number if number > 0 else None


def normalize_locator(item: EvidenceItem, mode: EvidenceMode) -> NormalizedLocator:
    """The most truthful location statement the evidence actually supports.

    Page numbers are used only when the ingested chunk really carries them. A
    missing page never becomes a guessed page: the fallback chain is paragraph,
    line range, chunk index, selection, complete document, and finally an
    explicit "location unavailable".
    """
    page_start = _coerce_page(item.page_start)
    page_end = _coerce_page(item.page_end)
    if page_start is not None:
        if page_end is not None and page_end != page_start:
            return NormalizedLocator(
                kind="page_range",
                label=f"pages {page_start}-{page_end}",
                page_start=page_start,
                page_end=page_end,
            )
        return NormalizedLocator(
            kind="page",
            label=f"page {page_start}",
            page_start=page_start,
            page_end=page_start,
        )

    locator = item.locator_json if isinstance(item.locator_json, Mapping) else {}
    paragraph = locator.get("paragraph") or locator.get("paragraph_index")
    if isinstance(paragraph, int) and not isinstance(paragraph, bool):
        return NormalizedLocator(kind="paragraph", label=f"paragraph {paragraph}")

    line_start = locator.get("line_start")
    line_end = locator.get("line_end")
    if isinstance(line_start, int) and not isinstance(line_start, bool):
        if isinstance(line_end, int) and not isinstance(line_end, bool) and line_end != line_start:
            return NormalizedLocator(kind="line_range", label=f"lines {line_start}-{line_end}")
        return NormalizedLocator(kind="line", label=f"line {line_start}")

    if item.source_type == "selection":
        return NormalizedLocator(kind="selection", label=SELECTION_LOCATION_LABEL)

    if item.source_type == "complete_document":
        return NormalizedLocator(kind="document", label=COMPLETE_DOCUMENT_LOCATION_LABEL)

    if isinstance(item.chunk_index, int) and not isinstance(item.chunk_index, bool):
        # chunk_index stays on the record for internal use; it is never a label.
        return NormalizedLocator(kind="passage", label=PASSAGE_LOCATION_LABEL)

    return NormalizedLocator(kind="unavailable", label=LOCATION_UNAVAILABLE_LABEL)


def bounded_excerpt(text: str, limit: int) -> tuple[str, bool]:
    """A whitespace-normalised excerpt of at most ``limit`` characters.

    The ellipsis counts against the limit: a caller that asks for 40 characters
    gets at most 40, never 41. ``len(result) <= limit`` holds for every
    non-negative limit.
    """
    normalized = _WHITESPACE.sub(" ", text or "").strip()
    if limit <= 0:
        return "", bool(normalized)
    if len(normalized) <= limit:
        return normalized, False
    if limit == 1:
        return "…", True
    return normalized[: limit - 1].rstrip() + "…", True


@dataclass(frozen=True)
class CitationRecord:
    citation_id: str
    filename: Optional[str]
    document_id: Any = None
    chunk_id: Any = None
    chunk_index: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    locator: NormalizedLocator = NormalizedLocator(
        kind="unavailable", label=LOCATION_UNAVAILABLE_LABEL
    )
    evidence_mode: EvidenceMode = EvidenceMode.NONE
    source_type: str = "unknown"
    # PUBLIC: the bounded excerpt shown in the UI and serialized to the client.
    excerpt: str = ""
    excerpt_truncated: bool = False
    # INTERNAL: the finalized evidence text exactly as the bundle holds it.
    # This is what the answer model and the claim verifier read. It is
    # deliberately absent from to_dict() (see the allow-list there) and must
    # never reach the client, where it would leak the complete document or the
    # whole selection past the excerpt bound.
    evidence_text: str = ""
    semantic_score: Optional[float] = None
    lexical_score: Optional[float] = None
    combined_score: Optional[float] = None

    @property
    def display_source(self) -> str:
        """What the UI shows. Never a bare document id.

        An unattributed highlight (a selection the caller's scope could not
        confirm) is labelled as the user's own selection rather than given a
        document name it has not earned.
        """
        if self.filename:
            return self.filename
        if self.source_type == "selection":
            return "Your highlighted text"
        return "Unattributed evidence"

    @property
    def display_label(self) -> str:
        return f"{self.display_source} · {self.locator.label}"

    def to_dict(self) -> dict[str, Any]:
        """The public citation object, as an explicit allow-list.

        Hand-written rather than dataclasses.asdict() on purpose: asdict()
        would serialize every field, including ``evidence_text``, and would
        keep doing so for any field added later. Only the keys named below
        reach the client; ``evidence_text`` is excluded deliberately.
        """
        return {
            "citation_id": self.citation_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "display_source": self.display_source,
            "display_label": self.display_label,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "locator": self.locator.to_dict(),
            "evidence_mode": self.evidence_mode.value,
            "source_type": self.source_type,
            "excerpt": self.excerpt,
            "excerpt_truncated": self.excerpt_truncated,
            "scores": {
                "semantic": self.semantic_score,
                "lexical": self.lexical_score,
                "combined": self.combined_score,
            },
        }


@dataclass(frozen=True)
class CitationRegistry:
    """Deterministic response-local citation identifiers for one answer."""

    records: tuple[CitationRecord, ...] = ()
    limits: CitationLimits = DEFAULT_CITATION_LIMITS
    contract_version: str = CITATION_CONTRACT_VERSION

    @classmethod
    def from_bundle(
        cls,
        bundle: EvidenceBundle,
        limits: CitationLimits = DEFAULT_CITATION_LIMITS,
    ) -> "CitationRegistry":
        """C1..Cn in bundle order, with duplicate evidence registered once.

        The bundle's order is already deterministic (fusion rank for retrieval,
        client order for selections), so the same evidence always produces the
        same identifiers.
        """
        records: list[CitationRecord] = []
        seen: dict[tuple, str] = {}
        for item in bundle.items:
            key = item.dedupe_key
            if key in seen:
                continue
            citation_id = f"C{len(records) + 1}"
            seen[key] = citation_id
            # The bundle is authoritative: its finalized text is carried
            # through unchanged as the model's evidence, and separately
            # bounded for public display.
            excerpt, truncated = bounded_excerpt(item.text, limits.max_excerpt_chars)
            records.append(
                CitationRecord(
                    citation_id=citation_id,
                    filename=item.filename,
                    document_id=item.document_id,
                    chunk_id=item.chunk_id,
                    chunk_index=item.chunk_index,
                    page_start=_coerce_page(item.page_start),
                    page_end=_coerce_page(item.page_end),
                    locator=normalize_locator(item, bundle.mode),
                    evidence_mode=bundle.mode,
                    source_type=item.source_type,
                    excerpt=excerpt,
                    excerpt_truncated=truncated,
                    evidence_text=item.text or "",
                    semantic_score=item.semantic_score,
                    lexical_score=item.lexical_score,
                    combined_score=item.combined_score,
                )
            )
        registry = cls(records=tuple(records), limits=limits)
        _assert_prompt_budget(registry.records, limits)
        return registry

    @property
    def citation_ids(self) -> tuple[str, ...]:
        return tuple(record.citation_id for record in self.records)

    @property
    def evidence_mode(self) -> EvidenceMode:
        """The mode of the bundle these citations came from."""
        return self.records[0].evidence_mode if self.records else EvidenceMode.NONE

    def get(self, citation_id: str) -> Optional[CitationRecord]:
        for record in self.records:
            if record.citation_id == citation_id:
                return record
        return None

    def is_valid(self, citation_id: str) -> bool:
        return self.get(citation_id) is not None

    @property
    def is_empty(self) -> bool:
        return not self.records

    def to_dict(self) -> dict[str, Any]:
        return {
            "citations": [record.to_dict() for record in self.records],
            "contract_version": self.contract_version,
        }


# --- prompt construction ----------------------------------------------------

CITATION_RULES = (
    "Answer rules:\n"
    "- Answer the question directly, in as few sentences as the answer needs.\n"
    "- Answer every part of the question. If the question asks for several "
    "things -- a count and a size, a limit and a condition, a list of items "
    "-- give each one explicitly.\n"
    "- Answer the question that was asked. Do not answer a nearby or similar "
    "question because the evidence happens to discuss it.\n"
    "- If the evidence does not establish one of the requested parts, say so "
    "for that part and still answer the parts it does establish.\n"
    "- Do not repeat or restate the question before answering it.\n"
    "- Keep the answer concise; do not pad it with restatements or summaries "
    "of what you just said.\n"
    "- Do not offer further help, suggest follow-up questions, or ask whether "
    "the reader wants more detail.\n"
    "- Paraphrase the evidence in your own words by default.\n"
    "- Do not present your own wording as a direct quotation, and do not wrap "
    "paraphrased text in quotation marks.\n"
    "\n"
    "Citation rules:\n"
    "- Use ONLY the evidence blocks above for any claim about the documents.\n"
    "- Put the citation id in square brackets immediately after each material "
    "claim, for example: The term is three years [C1].\n"
    "- Attach a citation id only to the claim that evidence actually supports; "
    "do not append every id to every sentence.\n"
    "- Place the citation id immediately before the sentence's terminal "
    "punctuation, never after it. Write: The term is three years [C1]. Do not "
    "write: The term is three years. [C1]\n"
    "- A citation is valid only when that specific evidence block contains the "
    "factual material the claim states. Read the block and confirm the facts "
    "are in it before citing it.\n"
    "- A matching document name, heading, or topic is not enough. Do not cite "
    "an evidence block merely because it belongs to the right document.\n"
    "- When a claim combines facts drawn from more than one evidence block, "
    "cite every evidence block that claim needs, for example: The term is "
    "three years and renews annually [C1][C2].\n"
    "- Use only the citation ids listed above. Never invent a citation id, a "
    "document name, a page number, or a quotation.\n"
    "- Cite every material claim you make. A factual statement drawn from "
    "the evidence and left uncited cannot be shown to the reader as sourced.\n"
    "- Do not describe or explain the citation markers themselves. Sentences "
    "such as \'[C1] is cited as evidence for this information\' are not part "
    "of the answer.\n"
    "- Never mention the evidence blocks, their identifiers, or how you "
    "arrived at the answer. Sentences such as \'This answer was found by "
    "combining information from evidence blocks C1 and C4\' must not appear. "
    "Write only the answer itself.\n"
    "- Do not cite an evidence block you did not actually use. A block being "
    "supplied to you is not a reason to cite it.\n"
    "- Do not overstate what an excerpt proves; if an excerpt only partly "
    "supports a point, say so.\n"
    "- This is legal information drawn from the supplied documents, not "
    "definitive legal advice.\n"
    "- Do not output internal tags, metadata, or JSON."
)

# The honest thing to say when the evidence does not answer the question
# depends on what was actually supplied. Claiming "the document does not
# address it" is only truthful when the whole document was supplied; with
# retrieved passages the truthful statement is narrower.
ABSENCE_RULE_COMPLETE_DOCUMENT = (
    "- If the complete document above does not contain the answer, say that "
    "the document does not address it, and say what is missing.\n"
    "- That negative finding is itself a claim about the document, so cite the "
    "complete-document evidence block it rests on, placing the citation id "
    "immediately before the terminal punctuation."
)
ABSENCE_RULE_PARTIAL_EVIDENCE = (
    "- If the passages above do not contain the answer, say that the answer "
    "was not found in the provided passages, and say what is missing. Do not "
    "claim the document as a whole is silent -- only these passages were "
    "supplied.\n"
    "- Say only that the information was not found in the retrieved passages. "
    "Do not state or imply that the entire document is silent on the subject."
)


def absence_rule(mode: EvidenceMode) -> str:
    """The scope-accurate way to report that the evidence falls short."""
    if mode is EvidenceMode.COMPLETE_DOCUMENT:
        return ABSENCE_RULE_COMPLETE_DOCUMENT
    return ABSENCE_RULE_PARTIAL_EVIDENCE

NO_EVIDENCE_INSTRUCTION = (
    "No document evidence is available for this question. Say that plainly, "
    "state what document or selection would be needed, and do not answer from "
    "memory as if a document supported it. Do not output any citation ids."
)


def prompt_evidence_chars(records: Sequence[CitationRecord]) -> int:
    """Total evidence text these records would place in one prompt."""
    return sum(len(record.evidence_text) for record in records)


def _assert_prompt_budget(
    records: Sequence[CitationRecord],
    limits: CitationLimits,
) -> None:
    """Reject an over-budget bundle. Never repair one.

    The planner already enforces this ceiling, so reaching it here means the
    bundle is invalid; failing closed is the only response that does not
    quietly become an evidence-selection decision.
    """
    total = prompt_evidence_chars(records)
    if total > limits.max_prompt_evidence_chars:
        raise EvidenceBudgetExceeded(
            f"bundle carries {total} characters of evidence, above the "
            f"{limits.max_prompt_evidence_chars}-character prompt ceiling"
        )


def build_evidence_blocks(registry: CitationRegistry) -> str:
    """The registered evidence, one labelled block per citation id.

    Uses the INTERNAL ``evidence_text``, not the public excerpt. The excerpt is
    a UI affordance bounded at a few hundred characters; feeding it to the
    model would silently reduce a complete short document, or a long
    highlighted selection, to its opening paragraph.
    """
    _assert_prompt_budget(registry.records, registry.limits)
    blocks = []
    for record in registry.records:
        blocks.append(
            f"[{record.citation_id}]\n"
            f"Document: {record.display_source}\n"
            f"Location: {record.locator.label}\n"
            f"Evidence:\n{record.evidence_text}"
        )
    return "\n\n".join(blocks)


def build_generation_prompt(
    question: str,
    registry: CitationRegistry,
    *,
    extra_context: str = "",
    task_prefix: str = "",
) -> str:
    """The complete generation prompt. Pure and independently unit-testable."""
    header = (task_prefix.strip() + "\n\n") if task_prefix.strip() else ""
    if registry.is_empty:
        return (
            f"{header}<question>\n{question.strip()}\n</question>\n\n"
            f"{NO_EVIDENCE_INSTRUCTION}"
        )
    extra = f"\n\n{extra_context.strip()}" if extra_context.strip() else ""
    rules = f"{CITATION_RULES}\n{absence_rule(registry.evidence_mode)}"
    return (
        f"{header}<evidence>\n{build_evidence_blocks(registry)}\n</evidence>{extra}\n\n"
        f"<question>\n{question.strip()}\n</question>\n\n"
        f"{rules}"
    )


# --- answer modes ------------------------------------------------------------
#
# The mode says what the caller will render, and so which rules the model
# answers under and whether its claims are graded. It never changes which
# evidence is supplied: planning and authorization are identical in every mode.
# Only document_qa output is shown as cited prose with per-claim support, so
# only document_qa runs claim extraction and verification.


class AnswerMode(str, Enum):
    DOCUMENT_QA = "document_qa"
    STRUCTURED_JSON = "structured_json"
    MODEL_REASONING = "model_reasoning"


def evidence_scope_statement(mode: EvidenceMode) -> str:
    """What the supplied evidence is, so no mode overstates its coverage."""
    if mode is EvidenceMode.COMPLETE_DOCUMENT:
        return "The evidence above is the complete text of the document."
    if mode is EvidenceMode.SELECTED_TEXT:
        return "The evidence above is text the user selected, not a complete document."
    return (
        "The evidence above is a set of passages retrieved from the documents, "
        "not their complete text. Do not describe it as covering a whole "
        "document or every document."
    )


STRUCTURED_OUTPUT_RULES = (
    "Output rules:\n"
    "- Produce exactly the output the request above asks for. When it asks for "
    "JSON, output only that JSON, with no prose and no markdown fences.\n"
    "- Take every statement about the documents from the evidence blocks above. "
    "Never invent a party, date, amount, clause, quotation, or page number.\n"
    "- Refer to a location by the document name and location shown in its "
    "evidence block, never by its citation id: do not write ids such as [C1].\n"
    "- If the evidence does not establish something the request asks for, "
    "leave it empty or say so inside the requested format rather than guessing."
)

STRUCTURED_NO_EVIDENCE_INSTRUCTION = (
    "No document evidence is available for this request. Produce the requested "
    "output format, leave every document-derived field empty, and say inside "
    "that format that no document evidence was available. Never invent "
    "document content."
)

MODEL_REASONING_RULES = (
    "Analysis rules:\n"
    "- This answer is AI analysis, not a lookup of verified sources.\n"
    "- You may draw on general legal knowledge such as doctrines and well-known "
    "precedents, but never present a case, statute, or citation as confirmed. "
    "If you are not certain one exists, say so plainly.\n"
    "- Describe what the evidence above says accurately, and do not attribute "
    "to it anything it does not state.\n"
    "- Do not write citation ids such as [C1]; refer to the supplied text by "
    "its document name or as the passage.\n"
    "- Do not output internal tags, metadata, or JSON."
)

MODEL_REASONING_NO_EVIDENCE_INSTRUCTION = (
    "No document text was supplied. Say so, and keep any analysis general: do "
    "not describe the contents of a document you were not given."
)


def build_mode_prompt(
    mode: AnswerMode,
    question: str,
    registry: CitationRegistry,
    *,
    extra_context: str = "",
) -> str:
    """The prompt for a structured_json or model_reasoning answer.

    document_qa is deliberately not handled here: its citation rules live in
    build_generation_prompt and stay the single path for graded answers.
    """
    if mode is AnswerMode.STRUCTURED_JSON:
        rules, no_evidence = STRUCTURED_OUTPUT_RULES, STRUCTURED_NO_EVIDENCE_INSTRUCTION
    elif mode is AnswerMode.MODEL_REASONING:
        rules, no_evidence = MODEL_REASONING_RULES, MODEL_REASONING_NO_EVIDENCE_INSTRUCTION
    else:
        raise ValueError(f"build_mode_prompt does not build {mode!r} prompts")
    if registry.is_empty:
        return f"<question>\n{question.strip()}\n</question>\n\n{no_evidence}"
    extra = f"\n\n{extra_context.strip()}" if extra_context.strip() else ""
    return (
        f"<evidence>\n{build_evidence_blocks(registry)}\n</evidence>{extra}\n\n"
        f"<question>\n{question.strip()}\n</question>\n\n"
        f"{evidence_scope_statement(registry.evidence_mode)}\n{rules}"
    )


# --- parsing and claim verification ----------------------------------------


@dataclass(frozen=True)
class ParsedCitations:
    valid_ids: tuple[str, ...] = ()
    invalid_ids: tuple[str, ...] = ()


def parse_citation_markers(text: str, registry: CitationRegistry) -> ParsedCitations:
    """Every ``[Cn]`` marker in ``text``, split into registered and unknown.

    An unknown id is recorded so the UI can refuse to make it clickable; it is
    never resolved into a record and never becomes a citation.
    """
    valid: list[str] = []
    invalid: list[str] = []
    for match in CITATION_MARKER_PATTERN.finditer(text or ""):
        citation_id = f"C{int(match.group(1))}"
        target = valid if registry.is_valid(citation_id) else invalid
        if citation_id not in target:
            target.append(citation_id)
    return ParsedCitations(valid_ids=tuple(valid), invalid_ids=tuple(invalid))


def _marker_only_tokens(fragment: str, marker_stripped: str) -> Optional[str]:
    """The citation markers of a fragment that consists of nothing else.

    Returns the markers as written (``"[C1]"``, ``"[C1] [C2]"``) when the
    fragment carries at least one marker and no other content, and ``None``
    otherwise. Deliberately not a broad "sentence that looks like a citation"
    test: a fragment keeping any word at all is a claim and is never matched
    here, so no legal sentence can be swallowed.
    """
    markers = [match.group(0) for match in CITATION_MARKER_PATTERN.finditer(fragment)]
    if not markers:
        return None
    if not _MARKER_ONLY_RESIDUE.match(marker_stripped):
        return None
    return " ".join(markers)


def _attach_markers(claim: str, markers: str) -> str:
    """Fold stray markers back into the claim they were written for.

    Placed immediately before the claim's terminal punctuation -- where the
    prompt asks the model to put them -- rather than trailing after it. Only
    the claim boundary moves; the generated answer text is never rewritten.
    """
    match = _TRAILING_TERMINAL_PUNCTUATION.search(claim)
    if match:
        return f"{claim[: match.start()].rstrip()} {markers}{match.group(1)}"
    return f"{claim} {markers}"


def extract_claims(answer: str) -> list[str]:
    """Split an answer into material claims (sentences/bulleted lines).

    Two narrow exclusions, both about not wasting a verification slot on
    something that asserts nothing:

    * an interrogative sentence -- a question echoed back from the prompt
      states no fact, so it can be neither supported nor unsupported;
    * a sentence that only describes the citation machinery, such as
      "[C1] is cited as evidence for this information."

    Everything else is kept, including ordinary factual sentences that happen
    to carry a citation marker.

    One repair, not an exclusion: when the model writes the marker after the
    full stop ("No monetary penalty is stated. [C1]") the splitter hands that
    marker over as its own fragment. Dropping it would leave a real, cited
    claim looking uncited and mark it unsupported, so the marker is folded
    back into the claim it followed. A marker-only fragment never becomes a
    claim of its own, and one with no retained claim before it -- after a
    skipped question, or at the very start -- is discarded rather than
    attached to something it does not belong to.

    "The claim it followed" means the one DIRECTLY before it. Retaining a
    claim opens it as the marker target and consecutive marker-only fragments
    keep folding into it, but any other meaningful fragment closes the target,
    including one that was skipped: a suppressed citation-commentary sentence
    or an echoed question stands between the marker and the earlier claim, so
    the marker belongs to neither and is dropped. Only an empty split fragment
    passes over without closing the target.
    """
    claims: list[str] = []
    # The claim a stray marker may still attach to, or None once an
    # intervening fragment -- retained or skipped -- has closed it.
    target_open = False
    for raw in _SENTENCE_SPLIT.split(answer or ""):
        claim = raw.strip()
        if not claim:
            # Nothing was written here; the target survives untouched.
            continue
        stripped = CITATION_MARKER_PATTERN.sub("", claim).strip(" -*•\t")
        markers = _marker_only_tokens(claim, stripped)
        if markers is not None:
            if target_open:
                claims[-1] = _attach_markers(claims[-1], markers)
            # Either way the target stays as it was: a run of stray markers
            # all belongs to the same claim, and a marker with no target does
            # not create one.
            continue
        if len(stripped) < 3:
            # Punctuation alone is not a claim, but it did intervene.
            target_open = False
            continue
        if stripped.endswith("?"):
            # A question asserts nothing; it cannot be supported or refuted.
            target_open = False
            continue
        if _CITATION_MECHANICS.match(stripped):
            target_open = False
            continue
        claims.append(claim)
        target_open = True
    return claims


@dataclass(frozen=True)
class CitedClaim:
    text: str
    citation_ids: tuple[str, ...] = ()
    invalid_citation_ids: tuple[str, ...] = ()
    status: ClaimVerificationStatus = ClaimVerificationStatus.UNVERIFIED
    reason: ClaimReason = ClaimReason.NOT_VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citation_ids": list(self.citation_ids),
            "invalid_citation_ids": list(self.invalid_citation_ids),
            "status": self.status.value,
            "reason": self.reason.value,
        }


_STATUS_BY_NAME = {status.value: status for status in ClaimVerificationStatus}


def _coerce_status(response: object) -> Optional[ClaimVerificationStatus]:
    """Map a verifier response onto a status, or None when malformed.

    A malformed or unrecognised response is never quietly upgraded to
    supported; the caller turns None into ``unverified``.
    """
    value: object = response
    if isinstance(response, Mapping):
        value = response.get("status")
    if isinstance(value, ClaimVerificationStatus):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    status = _STATUS_BY_NAME.get(normalized)
    if status is ClaimVerificationStatus.UNVERIFIED:
        return None
    return status


# --- deterministic support floor -------------------------------------------
#
# A green "Supported by your documents" badge is a high-confidence statement,
# so it must fail closed. A small local model asked "supported or not?" will
# happily answer "supported" for a passage that is merely ON THE SAME TOPIC --
# that is how an answer citing the wrong pages came back fully supported.
#
# This floor is the deterministic half of the check. It never contacts a
# model, never raises a status, and never invents support: it only asks
# whether the hard, checkable terms a claim asserts -- numbers, model and
# dataset names, section identifiers -- actually occur in the evidence THAT
# CLAIM CITED. Anything it cannot check it leaves alone for the verifier.

# Digits: "8", "100,000", "3.5", "27%".
_NUMBER_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")
# A token is a checkable name only when it is distinctive enough that its
# absence is meaningful. An ordinary Capitalised word is not: it is usually
# just the start of a sentence, and treating "The" or "Due" as evidence of
# anything would reject correct claims.
#   * ALL-CAPS runs of two or more:        BLEU, STEM, GPU, QA
#   * a token mixing letters and digits:   P100, GPT-4, PaLM-2L, Llama2-70B
#   * internal capitals (camel/mixed):     IRCoT, PaLM, StepBack
#   A pluralised acronym ("GPUs", "LLMs") is the same name as its singular,
#   so the trailing lowercase s is dropped when normalising. Without that,
#   a claim saying "GPUs" would fail against evidence saying "GPU".
_ALLCAPS_TOKEN = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)*s?\b")
_ALNUM_TOKEN = re.compile(r"\b(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b")
_MIXEDCASE_TOKEN = re.compile(
    r"\b[A-Za-z]*[a-z][A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b"
)

# Written numerals a claim may use where the document prints a digit, and the
# reverse. Bounded on purpose: this maps the small numbers that appear in
# model and training descriptions ("eight heads", "8 heads"), not arbitrary
# language. Without it a correct "eight attention heads" claim would be
# rejected against evidence that says "h = 8".
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "million": "1000000",
}
_WORD_TOKEN = re.compile(r"[A-Za-z]+")


def _normalize_number(raw: str) -> str:
    """"100,000" and "100000" are the same number; "3.50" and "3.5" are not.

    Only grouping commas are removed. Nothing is rounded, and a trailing
    decimal is kept exactly as written, so 3.5 never matches 35.
    """
    return raw.replace(",", "")


def salient_terms(text: str) -> set[str]:
    """The hard, checkable assertions in a piece of text.

    Citation markers are removed first: "[C1]" is machinery, not a claim
    about the documents, and C1 would otherwise read as a mixed-case name.
    """
    body = CITATION_MARKER_PATTERN.sub(" ", text or "")
    terms: set[str] = set()
    for match in _NUMBER_TOKEN.finditer(body):
        terms.add("#" + _normalize_number(match.group(0)))
    for word in _WORD_TOKEN.finditer(body):
        mapped = _NUMBER_WORDS.get(word.group(0).lower())
        if mapped is not None:
            terms.add("#" + mapped)
    for pattern in (_ALLCAPS_TOKEN, _ALNUM_TOKEN, _MIXEDCASE_TOKEN):
        for match in pattern.finditer(body):
            token = match.group(0)
            if token.endswith("s") and not token.endswith("ss"):
                # A pluralised acronym names the same thing as its singular.
                token = token[:-1]
            terms.add("@" + token.lower())
    return terms


# A status is only ever lowered, so the floor can make an answer amber or red
# but never green. Ordered least to most confident.
_STATUS_RANK = {
    ClaimVerificationStatus.UNSUPPORTED: 0,
    ClaimVerificationStatus.PARTIALLY_SUPPORTED: 1,
    ClaimVerificationStatus.SUPPORTED: 2,
}


def apply_support_floor(
    claim: str,
    records: Sequence[CitationRecord],
    status: ClaimVerificationStatus,
) -> tuple[ClaimVerificationStatus, bool]:
    """Lower ``status`` to what the cited evidence can actually carry.

    Returns the status and whether the floor changed it. The rules:

    * a claim asserting no checkable term is left to the verifier -- generic
      framing is not evidence of anything, in either direction;
    * every checkable term present in the cited evidence: unchanged;
    * some present, some missing (a compound claim only half evidenced):
      capped at ``partially_supported``;
    * none present: ``unsupported``. Topical similarity alone can never
      produce a green result.

    ``records`` are the ones the claim itself cited, so evidence the answer
    did not cite cannot rescue it.
    """
    if status not in _STATUS_RANK:
        return status, False
    claim_terms = salient_terms(claim)
    if not claim_terms:
        return status, False

    evidence_terms = salient_terms(" ".join(r.evidence_text for r in records))
    present = claim_terms & evidence_terms
    if len(present) == len(claim_terms):
        return status, False

    floor = (
        ClaimVerificationStatus.UNSUPPORTED
        if not present
        else ClaimVerificationStatus.PARTIALLY_SUPPORTED
    )
    if _STATUS_RANK[floor] < _STATUS_RANK[status]:
        return floor, True
    return status, False


ClaimVerifier = Callable[[str, Sequence[CitationRecord]], object]


def verify_claims(
    claims: Sequence[str],
    registry: CitationRegistry,
    verifier: Optional[ClaimVerifier] = None,
    limits: CitationLimits = DEFAULT_CITATION_LIMITS,
) -> list[CitedClaim]:
    """Attach a verification status to each claim.

    Guarantees:

    * a claim with no valid citation is never ``supported``;
    * a verifier exception or malformed response yields ``unverified``, never
      ``supported``;
    * no verifier exception detail escapes -- the exception is not bound to the
      result in any form.
    """
    verified_count = 0
    results: list[CitedClaim] = []

    for claim in claims:
        parsed = parse_citation_markers(claim, registry)
        if not parsed.valid_ids:
            results.append(
                CitedClaim(
                    text=claim,
                    citation_ids=(),
                    invalid_citation_ids=parsed.invalid_ids,
                    status=ClaimVerificationStatus.UNSUPPORTED,
                    reason=(
                        ClaimReason.INVALID_CITATION
                        if parsed.invalid_ids
                        else ClaimReason.NO_CITATION
                    ),
                )
            )
            continue

        if verifier is None or verified_count >= limits.max_claims_verified:
            results.append(
                CitedClaim(
                    text=claim,
                    citation_ids=parsed.valid_ids,
                    invalid_citation_ids=parsed.invalid_ids,
                    status=ClaimVerificationStatus.UNVERIFIED,
                    reason=ClaimReason.NOT_VERIFIED,
                )
            )
            continue

        records = tuple(
            record
            for record in (registry.get(citation_id) for citation_id in parsed.valid_ids)
            if record is not None
        )
        verified_count += 1
        try:
            response = verifier(claim, records)
        except Exception:
            # Deliberately not bound: no verifier exception text, stack, or
            # provider detail may reach the response.
            results.append(
                CitedClaim(
                    text=claim,
                    citation_ids=parsed.valid_ids,
                    invalid_citation_ids=parsed.invalid_ids,
                    status=ClaimVerificationStatus.UNVERIFIED,
                    reason=ClaimReason.VERIFIER_UNAVAILABLE,
                )
            )
            continue

        status = _coerce_status(response)
        if status is None:
            results.append(
                CitedClaim(
                    text=claim,
                    citation_ids=parsed.valid_ids,
                    invalid_citation_ids=parsed.invalid_ids,
                    status=ClaimVerificationStatus.UNVERIFIED,
                    reason=ClaimReason.VERIFIER_MALFORMED,
                )
            )
            continue

        # The verifier has spoken; the deterministic floor may only lower it.
        status, lowered = apply_support_floor(claim, records, status)
        results.append(
            CitedClaim(
                text=claim,
                citation_ids=parsed.valid_ids,
                invalid_citation_ids=parsed.invalid_ids,
                status=status,
                reason=(
                    ClaimReason.EVIDENCE_TERMS_MISSING
                    if lowered
                    else ClaimReason.VERIFIED
                ),
            )
        )
    return results


@dataclass(frozen=True)
class CitationGenerationResult:
    registry: CitationRegistry
    claims: tuple[CitedClaim, ...] = ()
    invalid_citation_ids: tuple[str, ...] = ()
    used_citation_ids: tuple[str, ...] = ()

    @property
    def has_unverified(self) -> bool:
        return any(
            claim.status is ClaimVerificationStatus.UNVERIFIED for claim in self.claims
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "citations": [record.to_dict() for record in self.registry.records],
            "claims": [claim.to_dict() for claim in self.claims],
            "invalid_citation_ids": list(self.invalid_citation_ids),
            "used_citation_ids": list(self.used_citation_ids),
            "contract_version": CITATION_CONTRACT_VERSION,
        }


def generate_citations(
    answer: str,
    registry: CitationRegistry,
    verifier: Optional[ClaimVerifier] = None,
    limits: CitationLimits = DEFAULT_CITATION_LIMITS,
) -> CitationGenerationResult:
    """Parse one generated answer and resolve its citations end to end."""
    parsed = parse_citation_markers(answer, registry)
    claims = verify_claims(extract_claims(answer), registry, verifier, limits)
    return CitationGenerationResult(
        registry=registry,
        claims=tuple(claims),
        invalid_citation_ids=parsed.invalid_ids,
        used_citation_ids=parsed.valid_ids,
    )


def citations_without_claims(answer: str, registry: CitationRegistry) -> CitationGenerationResult:
    """The registered ids an answer used, with no claim extraction or grading.

    For modes whose output is not shown as graded prose. No verifier runs and
    no claim is produced, so nothing can be labelled supported; the used ids
    still say which supplied evidence the answer drew on.
    """
    parsed = parse_citation_markers(answer, registry)
    return CitationGenerationResult(
        registry=registry,
        invalid_citation_ids=parsed.invalid_ids,
        used_citation_ids=parsed.valid_ids,
    )


# --- safe error objects -----------------------------------------------------

SAFE_ERROR_MESSAGES = {
    "evidence_unavailable": "Document evidence is temporarily unavailable.",
    "search_unavailable": "Search is temporarily unavailable. Please try again shortly.",
    "generation_unavailable": "The local model is not reachable.",
    "verification_unavailable": "Answer verification is temporarily unavailable.",
    "not_authorized": "No authorized document evidence is available for this question.",
    "internal_error": "Something went wrong while answering. Please try again.",
}


def safe_error(code: str) -> dict[str, str]:
    """A fixed, content-free error object.

    Only codes in ``SAFE_ERROR_MESSAGES`` produce a message, so no caller can
    smuggle an exception string, SQL, a connection string, or document text
    into a client-visible error by passing it through here.
    """
    key = code if code in SAFE_ERROR_MESSAGES else "internal_error"
    return {"code": key, "message": SAFE_ERROR_MESSAGES[key]}
