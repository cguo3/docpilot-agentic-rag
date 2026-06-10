from __future__ import annotations

from ...core.schema import SearchResult
from ...embedder.base import EmbedderBase
from ...vector_store.base import VectorStoreBase
from ..base import RetrieverBase


class DenseRetriever(RetrieverBase):
    """Semantic retrieval via vector similarity search."""

    def __init__(self, vector_store: VectorStoreBase, embedder: EmbedderBase) -> None:
        self.vector_store = vector_store
        self.embedder = embedder

    async def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        vector = await self.embedder.embed_query(query)
        return await self.vector_store.query(vector, top_k)
