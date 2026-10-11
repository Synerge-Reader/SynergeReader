from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class SelectionContext(BaseModel):
    id: str
    text: str
    document_name: str
    # E2: a highlight may report which document it came from, and where. These
    # are client claims, not authorization -- the backend validates document_id
    # against the caller's authorized documents and drops the attribution when
    # it does not match. All four are optional so older clients still parse.
    document_id: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    locator: Optional[Dict[str, Any]] = None

class AskRequest(BaseModel):
    selected_text: str = ""
    question: str
    # Legacy compatibility input only. Older clients send a model tag here; the
    # backend never forwards it and never lets it choose a model -- every
    # answer uses the backend-resolved profile.
    model: Optional[str] = None
    # RETIRED per-request model override. Every answer uses the one shared
    # selection an administrator switches; the backend refuses ANY non-null
    # value (an administrator's too) with 400, before any lookup. The field is
    # kept only so such a request is refused explicitly rather than ignored.
    model_profile_id: Optional[str] = Field(default=None, min_length=1, max_length=64)
    auth_token: Optional[str] = None
    active_document_name: Optional[str] = None
    selections: List[SelectionContext] = Field(default_factory=list)
    # E2: document scope is expressed with backend document ids, which are the
    # only thing that can be checked against what the caller owns. The
    # name-based fields above are retained for compatibility with older clients
    # but are never treated as authorization.
    active_document_id: Optional[int] = None
    document_ids: List[int] = Field(default_factory=list)
    # What the caller will render, which decides the prompt rules and whether
    # claims are verified -- never which documents may be read. document_qa is
    # cited prose graded claim by claim; structured_json is JSON-only tool
    # output; model_reasoning is open-ended AI analysis shown with an
    # "AI analysis only" notice. The values match citation_generation.AnswerMode.
    mode: Literal["document_qa", "structured_json", "model_reasoning"] = "document_qa"


class GenerationSelectRequest(BaseModel):
    """An administrator's request to switch the shared answer model. The id
    must name a registered, qualified backend profile; nothing else about the
    model can be chosen."""

    profile_id: str = Field(min_length=1, max_length=64)


class AskResponse(BaseModel):
    id: int
    answer: str
    question: str
    context_chunks: List[str]
    relevant_history: List[dict]



class CorrectionRequest(BaseModel):
    chat_id: int
    corrected_answer: str
    comment: Optional[str] = None


class RatingRequest(BaseModel):
    id: int
    rating: int
    comment: str


