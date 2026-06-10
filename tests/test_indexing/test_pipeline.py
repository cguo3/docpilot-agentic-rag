"""Unit tests for IngestionPipeline."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from docpilot.core.schema import Document
from docpilot.indexing.pipeline import IngestionPipeline


def _doc(id: str) -> Document:
    return Document(id=id, text=f"text-{id}")


def _make_pipeline(
    load_return=None,
    clean_return=None,
    extract_return=None,
    chunk_return=None,
    embed_return=None,
    ingest_return=None,
) -> tuple[IngestionPipeline, MagicMock, MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
    docs = load_return or [_doc("a"), _doc("b")]
    embs = embed_return or [[0.1, 0.2], [0.3, 0.4]]

    loader = MagicMock()
    loader.load = AsyncMock(return_value=docs)

    cleaner = MagicMock()
    cleaner.clean = AsyncMock(return_value=clean_return or docs)

    extractor = MagicMock()
    extractor.extract = AsyncMock(return_value=extract_return or docs)

    chunker = MagicMock()
    chunker.chunk = AsyncMock(return_value=chunk_return or docs)

    embedder = MagicMock()
    embedder.embed_documents = AsyncMock(return_value=embs)

    store = MagicMock()
    store.ingest = AsyncMock(return_value=ingest_return or [d.id for d in docs])

    pipeline = IngestionPipeline(
        loader=loader,
        cleaner=cleaner,
        metadata_extractor=extractor,
        chunker=chunker,
        embedder=embedder,
        vector_store=store,
    )
    return pipeline, loader, cleaner, extractor, chunker, embedder, store


class TestPipelineCallChain:
    async def test_all_stages_called(self):
        pipeline, loader, cleaner, extractor, chunker, embedder, store = _make_pipeline()
        await pipeline.run()
        loader.load.assert_called_once()
        cleaner.clean.assert_called_once()
        extractor.extract.assert_called_once()
        chunker.chunk.assert_called_once()
        embedder.embed_documents.assert_called_once()
        store.ingest.assert_called_once()

    async def test_load_output_passed_to_cleaner(self):
        docs = [_doc("x")]
        pipeline, loader, cleaner, *_ = _make_pipeline(load_return=docs)
        await pipeline.run()
        cleaner.clean.assert_called_once_with(docs)

    async def test_clean_output_passed_to_extractor(self):
        docs_cleaned = [_doc("y")]
        pipeline, _, cleaner, extractor, *_ = _make_pipeline(clean_return=docs_cleaned)
        cleaner.clean = AsyncMock(return_value=docs_cleaned)
        await pipeline.run()
        extractor.extract.assert_called_once_with(docs_cleaned)

    async def test_chunk_output_passed_to_embedder(self):
        chunks = [_doc("c1"), _doc("c2")]
        pipeline, _, _, _, chunker, embedder, _ = _make_pipeline(chunk_return=chunks)
        chunker.chunk = AsyncMock(return_value=chunks)
        await pipeline.run()
        embedder.embed_documents.assert_called_once_with(chunks)

    async def test_embed_output_passed_to_store(self):
        embs = [[1.0, 0.0], [0.0, 1.0]]
        docs = [_doc("a"), _doc("b")]
        pipeline, *_, embedder, store = _make_pipeline(embed_return=embs)
        embedder.embed_documents = AsyncMock(return_value=embs)
        await pipeline.run()
        call_args = store.ingest.call_args
        assert call_args[0][1] == embs

    async def test_returns_ids_from_store(self):
        pipeline, *_ = _make_pipeline(ingest_return=["id-1", "id-2"])
        result = await pipeline.run()
        assert result == ["id-1", "id-2"]


class TestPipelineSource:
    async def test_source_none_works(self):
        pipeline, loader, *_ = _make_pipeline()
        await pipeline.run(source=None)
        loader.load.assert_called_once_with(None)

    async def test_source_string_forwarded(self):
        pipeline, loader, *_ = _make_pipeline()
        await pipeline.run(source="path/to/data")
        loader.load.assert_called_once_with("path/to/data")

    async def test_source_path_forwarded(self):
        pipeline, loader, *_ = _make_pipeline()
        p = Path("/some/path")
        await pipeline.run(source=p)
        loader.load.assert_called_once_with(p)

    async def test_kwargs_forwarded_to_loader(self):
        pipeline, loader, *_ = _make_pipeline()
        await pipeline.run(trust_remote_code=True)
        loader.load.assert_called_once_with(None, trust_remote_code=True)


class TestDefaultLoaderBehavior:
    async def test_cleaner_base_class_is_identity(self):
        from docpilot.indexing.loader import Cleaner
        docs = [_doc("a")]
        result = await Cleaner().clean(docs)
        assert result is docs

    async def test_metadata_extractor_base_class_is_identity(self):
        from docpilot.indexing.loader import MetadataExtractor
        docs = [_doc("a")]
        result = await MetadataExtractor().extract(docs)
        assert result is docs
