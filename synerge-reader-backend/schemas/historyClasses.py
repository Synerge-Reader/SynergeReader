from pydantic import BaseModel
from typing import List, Optional

class HistoryItem(BaseModel):
    id: int
    timestamp: str
    selected_text: str
    question: str
    answer: str


class HistoryRequest(BaseModel):
    token: Optional[str] = None
