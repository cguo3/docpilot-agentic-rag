from __future__ import annotations

from pydantic import BaseModel


class Citation(BaseModel):
    chunk_id: str
    span: str
    source: str


class CitationExtractor:
    """Parse inline citations from the generated answer text."""

    def extract(self, answer: str) -> list[Citation]:
        raise NotImplementedError
