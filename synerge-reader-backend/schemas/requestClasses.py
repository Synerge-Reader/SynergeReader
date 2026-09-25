from typing import Any, Dict, List, Optional
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
    model: str
    auth_token: Optional[str] = None
    active_document_name: Optional[str] = None
    selections: List[SelectionContext] = Field(default_factory=list)
    # E2: document scope is expressed with backend document ids, which are the
    # only thing that can be checked against what the caller owns. The
    # name-based fields above are retained for compatibility with older clients
    # but are never treated as authorization.
    active_document_id: Optional[int] = None
    document_ids: List[int] = Field(default_factory=list)


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


