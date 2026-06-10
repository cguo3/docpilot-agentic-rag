from __future__ import annotations

from abc import ABC, abstractmethod

from ...core.schema import SearchResult


class ContextCompressorBase(ABC):

    @abstractmethod
    async def compress(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Strip irrelevant sentences from each result, keeping only query-relevant spans."""


class LLMContextCompressor(ContextCompressorBase):
    """Use an LLM to extract only the relevant sentences from each chunk."""

    async def compress(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        raise NotImplementedError
