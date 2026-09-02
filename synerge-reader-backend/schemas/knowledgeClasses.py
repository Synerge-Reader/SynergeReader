from typing import List, Optional
from pydantic import BaseModel

class KnowledgeItem(BaseModel):
    question: str
    answer: str
    source: Optional[str] = None


class KnowledgeInsertRequest(BaseModel):
    items: List[KnowledgeItem]
    source_type: Optional[str] = "manual"   # "manual" | "external_import"
    token: Optional[str] = None


class KnowledgeUrlImportRequest(BaseModel):
    url: str
    token: Optional[str] = None
