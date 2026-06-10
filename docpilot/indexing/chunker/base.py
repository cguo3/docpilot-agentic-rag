from __future__ import annotations

from abc import ABC, abstractmethod

from ...core.schema import Document


class ChunkerBase(ABC):

    @abstractmethod
    async def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents into smaller retrievable chunks."""
