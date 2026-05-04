from typing import List, Optional
from pydantic import BaseModel, Field


class SelectionContext(BaseModel):
    id: str
    text: str
    document_name: str

class AskRequest(BaseModel):
    selected_text: str = ""
    question: str
    model: str
    auth_token: Optional[str] = None
    active_document_name: Optional[str] = None
    selections: List[SelectionContext] = Field(default_factory=list)


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


