"""Unit tests for FAISSStore."""
from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from docpilot.core.schema import Document
from docpilot.vector_store.faiss_store import FAISSStore

DIM = 4  # small dimension for speed


def unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def make_doc(id: str, **meta) -> Document:
    return Document(id=id, text=f"text for {id}", metadata=meta)


@pytest.fixture
def store() -> FAISSStore:
    return FAISSStore(dimensions=DIM)


@pytest.fixture
def docs() -> list[Document]:
    return [make_doc("a", topic="cv"), make_doc("b", topic="nlp"), make_doc("c", topic="nlp")]


@pytest.fixture
def embeddings() -> list[list[float]]:
    return [unit_vec(i) for i in range(3)]


class TestIngest:
    async def test_returns_document_ids(self, store, docs, embeddings):
        ids = await store.ingest(docs, embeddings)
        assert ids == ["a", "b", "c"]

    async def test_count_increases(self, store, docs, embeddings):
        assert await store.count() == 0
        await store.ingest(docs, embeddings)
        assert await store.count() == 3

    async def test_incremental_ingest(self, store, docs, embeddings):
        await store.ingest(docs[:2], embeddings[:2])
        await store.ingest(docs[2:], embeddings[2:])
        assert await store.count() == 3

    async def test_empty_ingest_is_noop(self, store):
        ids = await store.ingest([], [])
        assert ids == []
        assert await store.count() == 0


class TestQuery:
    async def test_returns_top_k_results(self, store, docs, embeddings):
        await store.ingest(docs, embeddings)
        results = await store.query(embeddings[0], top_k=2)
        assert len(results) == 2

    async def test_exact_match_scores_near_one(self, store, docs, embeddings):
        await store.ingest(docs, embeddings)
        results = await store.query(embeddings[0], top_k=1)
        assert len(results) == 1
        assert results[0].document.id == "a"
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    async def test_results_ordered_by_score_descending(self, store, docs, embeddings):
        await store.ingest(docs, embeddings)
        results = await store.query(embeddings[0], top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_top_k_capped_by_index_size(self, store, docs, embeddings):
        await store.ingest(docs, embeddings)
        results = await store.query(embeddings[0], top_k=100)
        assert len(results) == 3  # only 3 docs

    async def test_query_empty_store_returns_empty(self, store, embeddings):
        results = await store.query(embeddings[0], top_k=5)
        assert results == []


class TestFilteredQuery:
    async def test_filters_by_metadata(self, store, docs, embeddings):
        await store.ingest(docs, embeddings)
        results = await store.filtered_query(embeddings[0], filter={"topic": "nlp"}, top_k=10)
        assert all(r.document.metadata["topic"] == "nlp" for r in results)
        assert len(results) == 2

    async def test_no_match_returns_empty(self, store, docs, embeddings):
        await store.ingest(docs, embeddings)
        results = await store.filtered_query(embeddings[0], filter={"topic": "unknown"}, top_k=10)
        assert results == []

    async def test_top_k_limits_filtered_results(self, store, docs, embeddings):
        await store.ingest(docs, embeddings)
        results = await store.filtered_query(embeddings[0], filter={"topic": "nlp"}, top_k=1)
        assert len(results) == 1

    async def test_multi_key_filter(self, store):
        docs = [
            make_doc("x", topic="cv", lang="en"),
            make_doc("y", topic="cv", lang="zh"),
            make_doc("z", topic="nlp", lang="en"),
        ]
        embs = [unit_vec(i) for i in range(3)]
        await store.ingest(docs, embs)
        results = await store.filtered_query(embs[0], filter={"topic": "cv", "lang": "en"})
        assert len(results) == 1
        assert results[0].document.id == "x"


class TestGetById:
    async def test_found(self, store, docs, embeddings):
        await store.ingest(docs, embeddings)
        doc = await store.get_by_id("b")
        assert doc is not None
        assert doc.id == "b"

    async def test_not_found_returns_none(self, store, docs, embeddings):
        await store.ingest(docs, embeddings)
        assert await store.get_by_id("nonexistent") is None

    async def test_not_found_empty_store(self, store):
        assert await store.get_by_id("x") is None


class TestDelete:
    async def test_removes_document(self, store, docs, embeddings):
        await store.ingest(docs, embeddings)
        await store.delete(["a"])
        assert await store.get_by_id("a") is None

    async def test_count_decreases(self, store, docs, embeddings):
        await store.ingest(docs, embeddings)
        await store.delete(["a"])
        assert await store.count() == 2

    async def test_remaining_docs_still_queryable(self, store, docs, embeddings):
        await store.ingest(docs, embeddings)
        await store.delete(["a"])
        results = await store.query(embeddings[1], top_k=5)
        ids = {r.document.id for r in results}
        assert "a" not in ids
        assert "b" in ids
        assert "c" in ids

    async def test_delete_unknown_id_is_safe(self, store, docs, embeddings):
        await store.ingest(docs, embeddings)
        await store.delete(["nonexistent"])
        assert await store.count() == 3

    async def test_delete_all_empties_store(self, store, docs, embeddings):
        await store.ingest(docs, embeddings)
        await store.delete(["a", "b", "c"])
        assert await store.count() == 0

    async def test_delete_partial(self, store, docs, embeddings):
        await store.ingest(docs, embeddings)
        await store.delete(["b"])
        assert await store.get_by_id("a") is not None
        assert await store.get_by_id("b") is None
        assert await store.get_by_id("c") is not None


class TestPersistence:
    async def test_save_creates_all_three_files(self, tmp_path, docs, embeddings):
        store = FAISSStore(DIM, persist_directory=str(tmp_path), embed_model="bge")
        await store.ingest(docs, embeddings)
        assert (tmp_path / "index.faiss").exists()
        assert (tmp_path / "documents.json").exists()
        assert (tmp_path / "index_meta.json").exists()

    async def test_load_restores_count(self, tmp_path, docs, embeddings):
        store = FAISSStore(DIM, persist_directory=str(tmp_path), embed_model="bge")
        await store.ingest(docs, embeddings)

        loaded = await FAISSStore.load(str(tmp_path))
        assert await loaded.count() == 3

    async def test_load_restores_documents(self, tmp_path, docs, embeddings):
        store = FAISSStore(DIM, persist_directory=str(tmp_path), embed_model="bge")
        await store.ingest(docs, embeddings)

        loaded = await FAISSStore.load(str(tmp_path))
        for doc in docs:
            found = await loaded.get_by_id(doc.id)
            assert found is not None
            assert found.text == doc.text
            assert found.metadata == doc.metadata

    async def test_load_restores_query_capability(self, tmp_path, docs, embeddings):
        store = FAISSStore(DIM, persist_directory=str(tmp_path), embed_model="bge")
        await store.ingest(docs, embeddings)

        loaded = await FAISSStore.load(str(tmp_path))
        results = await loaded.query(embeddings[0], top_k=1)
        assert results[0].document.id == "a"

    async def test_load_restores_meta(self, tmp_path, docs, embeddings):
        store = FAISSStore(DIM, persist_directory=str(tmp_path), embed_model="bge-small")
        await store.ingest(docs, embeddings)

        loaded = await FAISSStore.load(str(tmp_path))
        assert loaded._embed_model == "bge-small"
        assert loaded._dimensions == DIM

    async def test_index_meta_json_content(self, tmp_path, docs, embeddings):
        store = FAISSStore(DIM, persist_directory=str(tmp_path), embed_model="test-model")
        await store.ingest(docs, embeddings)

        meta = json.loads((tmp_path / "index_meta.json").read_text())
        assert meta["embed_model"] == "test-model"
        assert meta["dimensions"] == DIM
        assert meta["doc_count"] == 3
        assert "created_at" in meta

    async def test_no_persist_dir_does_not_save(self, tmp_path, docs, embeddings):
        store = FAISSStore(DIM, persist_directory=None)
        await store.ingest(docs, embeddings)
        assert not list(tmp_path.iterdir())


class TestEmbedModelMismatchWarning:
    async def test_no_warning_when_model_matches(self, tmp_path, docs, embeddings, caplog):
        store = FAISSStore(DIM, persist_directory=str(tmp_path), embed_model="model-a")
        await store.ingest(docs, embeddings)

        with caplog.at_level(logging.WARNING):
            FAISSStore(DIM, persist_directory=str(tmp_path), embed_model="model-a")
        assert not caplog.records

    async def test_warning_when_model_differs(self, tmp_path, docs, embeddings, caplog):
        store = FAISSStore(DIM, persist_directory=str(tmp_path), embed_model="model-a")
        await store.ingest(docs, embeddings)

        with caplog.at_level(logging.WARNING):
            FAISSStore(DIM, persist_directory=str(tmp_path), embed_model="model-b")

        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert any("model-a" in r.message for r in caplog.records)
        assert any("model-b" in r.message for r in caplog.records)

    async def test_warning_suggests_separate_directory(self, tmp_path, docs, embeddings, caplog):
        store = FAISSStore(DIM, persist_directory=str(tmp_path), embed_model="model-a")
        await store.ingest(docs, embeddings)

        with caplog.at_level(logging.WARNING):
            FAISSStore(DIM, persist_directory=str(tmp_path), embed_model="org/model-b")

        assert any("org_model-b" in r.message for r in caplog.records)

    async def test_no_warning_when_no_existing_index(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            FAISSStore(DIM, persist_directory=str(tmp_path), embed_model="model-a")
        assert not caplog.records

    async def test_no_warning_when_no_persist_directory(self, caplog):
        with caplog.at_level(logging.WARNING):
            FAISSStore(DIM, persist_directory=None, embed_model="model-a")
        assert not caplog.records


class TestDuplicateWarnings:
    async def test_no_warning_on_fresh_ingest(self, store, docs, embeddings, caplog):
        with caplog.at_level(logging.WARNING):
            await store.ingest(docs, embeddings)
        assert not caplog.records

    async def test_warning_on_duplicate_ingest(self, store, docs, embeddings, caplog):
        await store.ingest(docs, embeddings)
        with caplog.at_level(logging.WARNING):
            await store.ingest(docs[:1], embeddings[:1])  # re-ingest doc "a"
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert any("a" in r.message for r in caplog.records)

    async def test_warning_includes_duplicate_count(self, store, docs, embeddings, caplog):
        await store.ingest(docs, embeddings)
        with caplog.at_level(logging.WARNING):
            await store.ingest(docs[:2], embeddings[:2])  # "a" and "b" are duplicates
        assert any("2" in r.message for r in caplog.records)

    async def test_ingest_still_proceeds_despite_warning(self, store, docs, embeddings, caplog):
        await store.ingest(docs, embeddings)
        count_before = await store.count()
        with caplog.at_level(logging.WARNING):
            await store.ingest(docs[:1], embeddings[:1])
        assert await store.count() == count_before + 1
