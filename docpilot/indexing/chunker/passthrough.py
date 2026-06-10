from __future__ import annotations

from ...core.schema import Document
from .base import ChunkerBase


class PassThroughChunker(ChunkerBase):
    """Return documents as-is for datasets that are already pre-chunked."""

    async def chunk(self, documents: list[Document]) -> list[Document]:
        return documents
