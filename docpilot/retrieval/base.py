from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.schema import SearchResult


class RetrieverBase(ABC):

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Return top_k results ranked by relevance."""
