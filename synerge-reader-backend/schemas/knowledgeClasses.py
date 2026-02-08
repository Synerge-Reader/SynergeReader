from typing import List, Optional
from pydantic import BaseModel

class KnowledgeItem(BaseModel):
    question: str
    answer: str
    source: Optional[str] = None


class KnowledgeInsertRequest(BaseModel):
    items: List[KnowledgeItem]
