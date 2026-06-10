from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.schema import Document, SearchResult


class VectorStoreBase(ABC):

    @abstractmethod
    async def ingest(self, documents: list[Document], embeddings: list[list[float]]) -> list[str]:
        """Index documents with their pre-computed embeddings; return assigned IDs."""

    @abstractmethod
    async def query(self, vector: list[float], top_k: int = 10) -> list[SearchResult]:
        """ANN search by vector; return top_k results ranked by similarity."""

    @abstractmethod
    async def filtered_query(
        self,
        vector: list[float],
        filter: dict[str, Any],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """ANN search with metadata filter applied before ranking."""

    @abstractmethod
    async def get_by_id(self, id: str) -> Document | None:
        """Fetch a single document by ID; None if not found."""

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Remove documents by ID."""

    @abstractmethod
    async def count(self) -> int:
        """Return total number of documents in the store."""
