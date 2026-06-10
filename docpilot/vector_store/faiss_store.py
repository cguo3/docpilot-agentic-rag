from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import faiss  # type: ignore[import-untyped]
import numpy as np

from ..core.schema import Document, SearchResult
from .base import VectorStoreBase

logger = logging.getLogger(__name__)


class FAISSStore(VectorStoreBase):
    """Vector store backed by FAISS IndexFlatIP (cosine similarity via normalized vectors).

    Persists three files to disk:
        index.faiss      — FAISS binary index
        documents.json   — {faiss_int_id: document_dict}
        index_meta.json  — {embed_model, dimensions, doc_count, created_at}
    """

    def __init__(
        self,
        dimensions: int,
        persist_directory: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self._dimensions = dimensions
        self._persist_directory = Path(persist_directory) if persist_directory else None
        self._embed_model = embed_model
        self._index: faiss.Index = faiss.IndexFlatIP(dimensions)
        self._documents: dict[int, Document] = {}
        self._id_to_faiss: dict[str, int] = {}

        if self._persist_directory:
            meta_path = self._persist_directory / "index_meta.json"
            if meta_path.exists():
                stored = json.loads(meta_path.read_text()).get("embed_model")
                if stored and embed_model and stored != embed_model:
                    logger.warning(
                        "FAISSStore: embed_model mismatch — existing index at '%s' was built "
                        "with '%s' but current model is '%s'. Writing new embeddings into this "
                        "index will corrupt search results. Use a separate persist_directory "
                        "per model, e.g. data/yolo/%s.",
                        persist_directory, stored, embed_model,
                        embed_model.replace("/", "_"),
                    )

    # ------------------------------------------------------------------ #
    # Write path                                                           #
    # ------------------------------------------------------------------ #

    async def ingest(self, documents: list[Document], embeddings: list[list[float]]) -> list[str]:
        if not documents:
            return []

        duplicates = [doc.id for doc in documents if doc.id in self._id_to_faiss]
        if duplicates:
            logger.warning(
                "FAISSStore: %d document(s) already exist in the index and will be overwritten: %s",
                len(duplicates),
                ", ".join(duplicates[:5]) + (" ..." if len(duplicates) > 5 else ""),
            )

        vectors = np.array(embeddings, dtype=np.float32)
        start_id = self._index.ntotal

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._index.add(vectors))

        for i, doc in enumerate(documents):
            faiss_id = start_id + i
            self._documents[faiss_id] = doc
            self._id_to_faiss[doc.id] = faiss_id

        if self._persist_directory:
            await self._save()

        return [doc.id for doc in documents]

    # ------------------------------------------------------------------ #
    # Read path                                                            #
    # ------------------------------------------------------------------ #

    async def query(self, vector: list[float], top_k: int = 10) -> list[SearchResult]:
        q = np.array(vector, dtype=np.float32).reshape(1, -1)
        loop = asyncio.get_running_loop()
        scores, indices = await loop.run_in_executor(
            None, lambda: self._index.search(q, top_k)
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx in self._documents:
                results.append(SearchResult(document=self._documents[idx], score=float(score)))
        return results

    async def filtered_query(
        self,
        vector: list[float],
        filter: dict[str, Any],
        top_k: int = 10,
    ) -> list[SearchResult]:
        candidates = await self.query(vector, top_k=top_k * 4)
        results = []
        for result in candidates:
            if all(result.document.metadata.get(k) == v for k, v in filter.items()):
                results.append(result)
                if len(results) == top_k:
                    break
        return results

    async def get_by_id(self, id: str) -> Document | None:
        faiss_id = self._id_to_faiss.get(id)
        if faiss_id is None:
            return None
        return self._documents.get(faiss_id)

    async def delete(self, ids: list[str]) -> None:
        for doc_id in ids:
            faiss_id = self._id_to_faiss.pop(doc_id, None)
            if faiss_id is not None:
                self._documents.pop(faiss_id, None)
        # Rebuild index from remaining documents
        remaining = list(self._documents.values())
        remaining_ids = list(self._documents.keys())
        if remaining:
            old_embeddings: list[list[float]] = []
            for fid in remaining_ids:
                vec = np.zeros((1, self._dimensions), dtype=np.float32)
                self._index.reconstruct(fid, vec[0])
                old_embeddings.append(vec[0].tolist())
            self._index = faiss.IndexFlatIP(self._dimensions)
            self._documents = {}
            self._id_to_faiss = {}
            await self.ingest(remaining, old_embeddings)
        else:
            self._index = faiss.IndexFlatIP(self._dimensions)

    async def count(self) -> int:
        return int(self._index.ntotal)

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    async def _save(self) -> None:
        assert self._persist_directory is not None
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_running_loop()

        index_path = str(self._persist_directory / "index.faiss")
        await loop.run_in_executor(None, lambda: faiss.write_index(self._index, index_path))

        docs_path = self._persist_directory / "documents.json"
        docs_data = {str(k): v.model_dump() for k, v in self._documents.items()}
        docs_path.write_text(json.dumps(docs_data))

        meta_path = self._persist_directory / "index_meta.json"
        meta_path.write_text(json.dumps({
            "embed_model": self._embed_model,
            "dimensions": self._dimensions,
            "doc_count": self._index.ntotal,
            "created_at": time.time(),
        }))

    @classmethod
    async def load(cls, persist_directory: str) -> FAISSStore:
        path = Path(persist_directory)
        meta = json.loads((path / "index_meta.json").read_text())
        store = cls(
            dimensions=meta["dimensions"],
            persist_directory=persist_directory,
            embed_model=meta.get("embed_model"),
        )
        loop = asyncio.get_running_loop()
        store._index = await loop.run_in_executor(
            None, lambda: faiss.read_index(str(path / "index.faiss"))
        )
        docs_data = json.loads((path / "documents.json").read_text())
        store._documents = {int(k): Document(**v) for k, v in docs_data.items()}
        store._id_to_faiss = {doc.id: int(k) for k, doc in store._documents.items()}
        return store
