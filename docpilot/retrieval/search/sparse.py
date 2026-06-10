from __future__ import annotations

from ...core.schema import SearchResult
from ..base import RetrieverBase


class SparseRetriever(RetrieverBase):
    """Keyword retrieval using BM25 scoring."""

    async def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        raise NotImplementedError
