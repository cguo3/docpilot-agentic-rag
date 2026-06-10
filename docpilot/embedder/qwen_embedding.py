from __future__ import annotations

import numpy as np

from ..core.schema import Document
from .base import EmbedderBase


class QwenEmbedder(EmbedderBase):
    """Embedder backed by Qwen text-embedding models via DashScope API."""

    def __init__(self, model: str = "text-embedding-v3", api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return 1024

    async def embed_documents(self, documents: list[Document]) -> list[np.ndarray]:
        raise NotImplementedError

    async def embed_query(self, query: str) -> np.ndarray:
        raise NotImplementedError
