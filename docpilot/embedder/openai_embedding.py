from __future__ import annotations

from openai import AsyncOpenAI

from ..core.schema import Document
from .base import EmbedderBase


class OpenAIEmbedder(EmbedderBase):

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return 1536

    async def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        raise NotImplementedError

    async def embed_query(self, query: str) -> list[float]:
        raise NotImplementedError
