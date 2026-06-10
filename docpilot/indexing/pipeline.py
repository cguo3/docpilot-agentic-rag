from __future__ import annotations

from pathlib import Path
from typing import Any

from ..embedder.base import EmbedderBase
from ..vector_store.base import VectorStoreBase
from .chunker.base import ChunkerBase
from .loader import Cleaner, DataLoader, MetadataExtractor


class IngestionPipeline:
    """Orchestrates the full write path: load → clean → chunk → embed → ingest."""

    def __init__(
        self,
        loader: DataLoader,
        cleaner: Cleaner,
        metadata_extractor: MetadataExtractor,
        chunker: ChunkerBase,
        embedder: EmbedderBase,
        vector_store: VectorStoreBase,
    ) -> None:
        self._loader = loader
        self._cleaner = cleaner
        self._metadata_extractor = metadata_extractor
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store

    async def run(self, source: str | Path, **kwargs: Any) -> list[str]:
        """Run the full pipeline; return the IDs of indexed documents."""
        documents = await self._loader.load(source, **kwargs)
        documents = await self._cleaner.clean(documents)
        documents = await self._metadata_extractor.extract(documents)
        chunks = await self._chunker.chunk(documents)
        embeddings = await self._embedder.embed_documents(chunks)
        return await self._vector_store.ingest(chunks, embeddings)
