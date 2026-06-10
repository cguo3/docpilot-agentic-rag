from __future__ import annotations

from ...core.schema import Document
from .base import ChunkerBase


class SemanticChunker(ChunkerBase):
    """Split documents at natural semantic boundaries (sentences, paragraphs, topics)."""

    def __init__(self, max_chunk_size: int = 1024) -> None:
        self.max_chunk_size = max_chunk_size

    async def chunk(self, documents: list[Document]) -> list[Document]:
        raise NotImplementedError
