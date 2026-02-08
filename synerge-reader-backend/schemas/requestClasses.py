from typing import List, Optional
from pydantic import BaseModel

class AskRequest(BaseModel):
    selected_text: str
    question: str
    model: str
    auth_token: Optional[str] = None


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


