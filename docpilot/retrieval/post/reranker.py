from __future__ import annotations

from abc import ABC, abstractmethod

from ...core.schema import SearchResult


class RerankerBase(ABC):

    @abstractmethod
    async def rerank(self, query: str, results: list[SearchResult], top_k: int = 10) -> list[SearchResult]:
        """Score and reorder results using a cross-encoder or similar model."""


class CrossEncoderReranker(RerankerBase):
    """Rerank using a cross-encoder that jointly encodes query and document."""

    async def rerank(self, query: str, results: list[SearchResult], top_k: int = 10) -> list[SearchResult]:
        raise NotImplementedError
