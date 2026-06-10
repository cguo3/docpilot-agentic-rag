from __future__ import annotations

from typing import Any

import numpy as np

from ..core.schema import Document, SearchResult
from .base import VectorStoreBase


class ChromaStore(VectorStoreBase):
    """Vector store backed by ChromaDB."""

    def __init__(self, collection_name: str, persist_directory: str | None = None) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory

    async def ingest(self, documents: list[Document], embeddings: list[np.ndarray]) -> list[str]:
        raise NotImplementedError

    async def query(self, vector: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        raise NotImplementedError

    async def filtered_query(
        self,
        vector: np.ndarray,
        filter: dict[str, Any],
        top_k: int = 10,
    ) -> list[SearchResult]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Document | None:
        raise NotImplementedError

    async def delete(self, ids: list[str]) -> None:
        raise NotImplementedError

    async def count(self) -> int:
        raise NotImplementedError
