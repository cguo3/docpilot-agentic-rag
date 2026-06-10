from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.schema import Document


class EmbedderBase(ABC):

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the underlying embedding model."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Output vector dimensionality; must match the target index."""

    @abstractmethod
    async def embed_documents(self, documents: list[Document]) -> list[np.ndarray]:
        """Encode documents for indexing (passage-side embedding)."""

    @abstractmethod
    async def embed_query(self, query: str) -> np.ndarray:
        """Encode a query for retrieval (query-side embedding)."""
