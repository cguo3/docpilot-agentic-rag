from __future__ import annotations

import asyncio

from ...core.schema import SearchResult
from ..base import RetrieverBase
from .dense import DenseRetriever
from .sparse import SparseRetriever


class HybridRetriever(RetrieverBase):
    """Fuse dense and sparse results with Reciprocal Rank Fusion (RRF)."""

    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever, rrf_k: int = 60) -> None:
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k

    async def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        dense_results, sparse_results = await asyncio.gather(
            self.dense.retrieve(query, top_k * 2),
            self.sparse.retrieve(query, top_k * 2),
        )
        # TODO: RRF fusion
        raise NotImplementedError
