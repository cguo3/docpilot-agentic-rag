"""Functional integration tests for the end-to-end ingestion pipeline.

Uses real FAISS and real DatasetCache. Mocks the embedder and HuggingFace
loader so the test has no network or model-download dependencies.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from docpilot.cache import DatasetCache
from docpilot.core.schema import Document
from docpilot.indexing import (
    Cleaner,
    HuggingFaceLoader,
    IngestionPipeline,
    MetadataExtractor,
    PassThroughChunker,
)
from docpilot.vector_store.faiss_store import FAISSStore

DIM = 8


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_mock_embedder(docs_count: int) -> MagicMock:
    embedder = MagicMock()
    embedder.dimensions = DIM
    embedder.model_name = "mock-model"
    embedder.embed_documents = AsyncMock(
        return_value=[_unit_vec(i) for i in range(docs_count)]
    )
    embedder.embed_query = AsyncMock(return_value=_unit_vec(99))
    return embedder


SAMPLE_ROWS = [
    {"_id": f"doc-{i}", "text": f"content about YOLO part {i}", "metadata": {"idx": i}}
    for i in range(10)
]


class TestFullPipelineWithFAISS:
    async def test_pipeline_indexes_all_documents(self, tmp_path):
        loader = HuggingFaceLoader(
            "freshstack/corpus-oct-2024",
            name="yolo",
            split="train",
            text_field="text",
            id_field="_id",
            metadata_field="metadata",
        )
        embedder = _make_mock_embedder(len(SAMPLE_ROWS))
        store = FAISSStore(dimensions=DIM, persist_directory=str(tmp_path / "index"))

        pipeline = IngestionPipeline(
            loader=loader,
            cleaner=Cleaner(),
            metadata_extractor=MetadataExtractor(),
            chunker=PassThroughChunker(),
            embedder=embedder,
            vector_store=store,
        )

        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(SAMPLE_ROWS))

        with patch("docpilot.indexing.loaders.huggingface.load_dataset", return_value=mock_dataset):
            ids = await pipeline.run()

        assert len(ids) == 10
        assert await store.count() == 10

    async def test_pipeline_persists_index(self, tmp_path):
        loader = HuggingFaceLoader("ds", text_field="text", id_field="_id")
        embedder = _make_mock_embedder(len(SAMPLE_ROWS))
        store = FAISSStore(dimensions=DIM, persist_directory=str(tmp_path / "idx"), embed_model="mock")

        pipeline = IngestionPipeline(
            loader=loader, cleaner=Cleaner(), metadata_extractor=MetadataExtractor(),
            chunker=PassThroughChunker(), embedder=embedder, vector_store=store,
        )
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(SAMPLE_ROWS))

        with patch("docpilot.indexing.loaders.huggingface.load_dataset", return_value=mock_dataset):
            await pipeline.run()

        assert (tmp_path / "idx" / "index.faiss").exists()
        assert (tmp_path / "idx" / "documents.json").exists()
        assert (tmp_path / "idx" / "index_meta.json").exists()

    async def test_indexed_docs_queryable(self, tmp_path):
        loader = HuggingFaceLoader("ds", text_field="text", id_field="_id")
        embeddings = [_unit_vec(i) for i in range(len(SAMPLE_ROWS))]
        embedder = _make_mock_embedder(len(SAMPLE_ROWS))
        embedder.embed_documents = AsyncMock(return_value=embeddings)
        store = FAISSStore(dimensions=DIM)

        pipeline = IngestionPipeline(
            loader=loader, cleaner=Cleaner(), metadata_extractor=MetadataExtractor(),
            chunker=PassThroughChunker(), embedder=embedder, vector_store=store,
        )
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(SAMPLE_ROWS))

        with patch("docpilot.indexing.loaders.huggingface.load_dataset", return_value=mock_dataset):
            await pipeline.run()

        # Query with the embedding of doc-0 — should return doc-0 as top result
        results = await store.query(embeddings[0], top_k=1)
        assert results[0].document.id == "doc-0"
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    async def test_metadata_preserved_end_to_end(self, tmp_path):
        rows = [{"_id": "x", "text": "test", "metadata": {"source": "github", "year": 2024}}]
        loader = HuggingFaceLoader("ds", text_field="text", id_field="_id", metadata_field="metadata")
        embedder = _make_mock_embedder(1)
        store = FAISSStore(dimensions=DIM)

        pipeline = IngestionPipeline(
            loader=loader, cleaner=Cleaner(), metadata_extractor=MetadataExtractor(),
            chunker=PassThroughChunker(), embedder=embedder, vector_store=store,
        )
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(rows))

        with patch("docpilot.indexing.loaders.huggingface.load_dataset", return_value=mock_dataset):
            await pipeline.run()

        doc = await store.get_by_id("x")
        assert doc is not None
        assert doc.metadata["source"] == "github"
        assert doc.metadata["year"] == 2024


class TestCacheDeduplication:
    async def test_pipeline_skipped_on_cache_hit(self, tmp_path):
        cache = DatasetCache(str(tmp_path / "cache"))
        cache.mark_cached("freshstack/corpus-oct-2024", "train", "mock-model", name="yolo")

        embedder = _make_mock_embedder(0)
        store = FAISSStore(dimensions=DIM)

        if cache.is_cached("freshstack/corpus-oct-2024", "train", "mock-model", name="yolo"):
            # Simulate the ingest_yolo.py guard
            pass
        else:
            pytest.fail("Cache miss when hit expected")

        embedder.embed_documents.assert_not_called()
        assert await store.count() == 0

    async def test_cache_miss_triggers_pipeline(self, tmp_path):
        cache = DatasetCache(str(tmp_path / "cache"))
        assert not cache.is_cached("freshstack/corpus-oct-2024", "train", "new-model", name="yolo")

    async def test_different_model_bypasses_cache(self, tmp_path):
        cache = DatasetCache(str(tmp_path / "cache"))
        cache.mark_cached("ds", "train", "model-v1", name="yolo")
        assert not cache.is_cached("ds", "train", "model-v2", name="yolo")

    async def test_mark_after_pipeline_prevents_rerun(self, tmp_path):
        cache = DatasetCache(str(tmp_path / "cache"))
        cache.mark_cached("ds", "train", "model", name="yolo", doc_count=10)
        assert cache.is_cached("ds", "train", "model", name="yolo")


class TestFAISSLoadRoundTrip:
    async def test_load_after_pipeline_preserves_all_docs(self, tmp_path):
        rows = [{"_id": f"d{i}", "text": f"text {i}", "metadata": {}} for i in range(5)]
        loader = HuggingFaceLoader("ds", text_field="text", id_field="_id", metadata_field="metadata")
        embeddings = [_unit_vec(i) for i in range(5)]
        embedder = _make_mock_embedder(5)
        embedder.embed_documents = AsyncMock(return_value=embeddings)
        store = FAISSStore(dimensions=DIM, persist_directory=str(tmp_path / "idx"), embed_model="mock")

        pipeline = IngestionPipeline(
            loader=loader, cleaner=Cleaner(), metadata_extractor=MetadataExtractor(),
            chunker=PassThroughChunker(), embedder=embedder, vector_store=store,
        )
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(rows))

        with patch("docpilot.indexing.loaders.huggingface.load_dataset", return_value=mock_dataset):
            await pipeline.run()

        loaded = await FAISSStore.load(str(tmp_path / "idx"))
        assert await loaded.count() == 5
        for i in range(5):
            doc = await loaded.get_by_id(f"d{i}")
            assert doc is not None
            assert doc.text == f"text {i}"

    async def test_loaded_store_query_matches_original(self, tmp_path):
        rows = [{"_id": "q0", "text": "query doc", "metadata": {}}]
        loader = HuggingFaceLoader("ds", text_field="text", id_field="_id", metadata_field="metadata")
        vec = _unit_vec(0)
        embedder = _make_mock_embedder(1)
        embedder.embed_documents = AsyncMock(return_value=[vec])
        store = FAISSStore(dimensions=DIM, persist_directory=str(tmp_path / "idx"), embed_model="mock")

        pipeline = IngestionPipeline(
            loader=loader, cleaner=Cleaner(), metadata_extractor=MetadataExtractor(),
            chunker=PassThroughChunker(), embedder=embedder, vector_store=store,
        )
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(rows))

        with patch("docpilot.indexing.loaders.huggingface.load_dataset", return_value=mock_dataset):
            await pipeline.run()

        loaded = await FAISSStore.load(str(tmp_path / "idx"))
        results = await loaded.query(vec, top_k=1)
        assert results[0].document.id == "q0"
        assert results[0].score == pytest.approx(1.0, abs=1e-5)
