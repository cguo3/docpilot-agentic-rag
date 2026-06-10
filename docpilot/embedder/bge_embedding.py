from __future__ import annotations

import asyncio

from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

from ..core.schema import Document
from .base import EmbedderBase

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BGEEmbedder(EmbedderBase):
    """Local embedder using BAAI/bge-small-en-v1.5 via sentence-transformers.

    Produces 384-dimensional normalized vectors suitable for cosine similarity
    via inner product (IndexFlatIP in FAISS).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return 384

    async def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        texts = [doc.text for doc in documents]
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._get_model().encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            ),
        )
        return embeddings.tolist()

    async def embed_query(self, query: str) -> list[float]:
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._get_model().encode(
                [_QUERY_PREFIX + query],
                normalize_embeddings=True,
            ),
        )
        return embedding[0].tolist()
