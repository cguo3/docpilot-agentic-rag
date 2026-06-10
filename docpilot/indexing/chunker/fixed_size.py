from __future__ import annotations

from ...core.schema import Document
from .base import ChunkerBase


class FixedSizeChunker(ChunkerBase):
    """Split documents into fixed-size token/character windows with optional overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    async def chunk(self, documents: list[Document]) -> list[Document]:
        raise NotImplementedError
