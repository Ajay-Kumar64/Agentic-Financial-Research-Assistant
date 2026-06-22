# rag/schemas.py
from pydantic import BaseModel
from typing import List

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    page: int = 0
    section: str = ""

class Answer(BaseModel):
    text: str
    citations: List[str]
