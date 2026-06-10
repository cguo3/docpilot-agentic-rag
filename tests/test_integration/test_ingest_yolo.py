"""End-to-end tests for scripts/ingest_yolo.py.

All external I/O is mocked:
  - HuggingFace load_dataset  → avoids network
  - SentenceTransformer       → avoids model download

Everything else is real: FAISSStore, DatasetCache, IngestionPipeline,
PassThroughChunker, Cleaner, MetadataExtractor.
"""
from __future__ import annotations

import json
import logging
from argparse import Namespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.ingest_yolo import (
    DATASET_NAME,
    DATASET_SUBSET,
    EMBED_MODEL,
    SPLIT,
    _persist_dir,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 384
_ROWS = [
    {"_id": f"yolo-{i}", "text": f"YOLO detection text {i}", "metadata": {"idx": i}}
    for i in range(8)
]


def _mock_dataset(rows: list = _ROWS) -> MagicMock:
    ds = MagicMock()
    ds.__iter__ = MagicMock(return_value=iter(rows))
    return ds


def _mock_st_model(dim: int = _DIM) -> MagicMock:
    model = MagicMock()

    def encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False):
        n = len(texts)
        vecs = np.random.default_rng(0).standard_normal((n, dim)).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    model.encode = MagicMock(side_effect=encode)
    return model


def _args(tmp_path, *, embed_model: str = EMBED_MODEL, force: bool = False) -> Namespace:
    """Build a Namespace that drives main() into tmp_path."""
    return Namespace(
        dataset=DATASET_NAME,
        subset=DATASET_SUBSET,
        split=SPLIT,
        embed_model=embed_model,
        embedder_type="bge",
        force=force,
        data_root=str(tmp_path),
        list_indexes=False,
    )


def _run(tmp_path, *, embed_model: str = EMBED_MODEL, force: bool = False):
    """Context manager: patches load_dataset + SentenceTransformer, then runs main()."""
    args = _args(tmp_path, embed_model=embed_model, force=force)

    class _ctx:
        async def __aenter__(self):
            self._ld = patch(
                "docpilot.indexing.loaders.huggingface.load_dataset",
                return_value=_mock_dataset(),
            )
            self._st = patch(
                "docpilot.embedder.bge_embedding.SentenceTransformer",
                return_value=_mock_st_model(),
            )
            self._ld.start()
            self._st.start()
            await main(args)

        async def __aexit__(self, *_):
            self._ld.stop()
            self._st.stop()

    return _ctx()


def _index_dir(tmp_path, embed_model: str = EMBED_MODEL) -> object:
    from pathlib import Path
    return Path(_persist_dir(Path(tmp_path), DATASET_SUBSET, embed_model))


# ---------------------------------------------------------------------------
# First-run behaviour
# ---------------------------------------------------------------------------

class TestFirstRun:
    async def test_creates_faiss_index_file(self, tmp_path):
        async with _run(tmp_path):
            pass
        assert (_index_dir(tmp_path) / "index.faiss").exists()

    async def test_creates_documents_json(self, tmp_path):
        async with _run(tmp_path):
            pass
        assert (_index_dir(tmp_path) / "documents.json").exists()

    async def test_creates_index_meta_json(self, tmp_path):
        async with _run(tmp_path):
            pass
        assert (_index_dir(tmp_path) / "index_meta.json").exists()

    async def test_creates_cache_file(self, tmp_path):
        async with _run(tmp_path):
            pass
        assert (tmp_path / "cache" / "dataset_hashes.json").exists()

    async def test_indexes_all_documents(self, tmp_path):
        async with _run(tmp_path):
            pass
        docs = json.loads((_index_dir(tmp_path) / "documents.json").read_text())
        assert len(docs) == len(_ROWS)

    async def test_index_meta_records_correct_model(self, tmp_path):
        async with _run(tmp_path):
            pass
        meta = json.loads((_index_dir(tmp_path) / "index_meta.json").read_text())
        assert meta["embed_model"] == EMBED_MODEL
        assert meta["dimensions"] == _DIM
        assert meta["doc_count"] == len(_ROWS)

    async def test_cache_entry_records_correct_metadata(self, tmp_path):
        async with _run(tmp_path):
            pass
        data = json.loads((tmp_path / "cache" / "dataset_hashes.json").read_text())
        entry = next(iter(data.values()))
        assert entry["dataset"] == DATASET_NAME
        assert entry["name"] == DATASET_SUBSET
        assert entry["split"] == SPLIT
        assert entry["embed_model"] == EMBED_MODEL
        assert entry["doc_count"] == len(_ROWS)

    async def test_document_ids_match_source_id_field(self, tmp_path):
        async with _run(tmp_path):
            pass
        docs = json.loads((_index_dir(tmp_path) / "documents.json").read_text())
        assert {v["id"] for v in docs.values()} == {r["_id"] for r in _ROWS}

    async def test_document_metadata_flat_merged(self, tmp_path):
        async with _run(tmp_path):
            pass
        docs = json.loads((_index_dir(tmp_path) / "documents.json").read_text())
        for entry in docs.values():
            assert "idx" in entry["metadata"]
            assert "metadata" not in entry["metadata"]


# ---------------------------------------------------------------------------
# Multi-model support
# ---------------------------------------------------------------------------

class TestMultiModel:
    async def test_different_models_get_separate_directories(self, tmp_path):
        async with _run(tmp_path, embed_model="BAAI/bge-small-en-v1.5"):
            pass
        async with _run(tmp_path, embed_model="BAAI/bge-base-en-v1.5"):
            pass

        assert (_index_dir(tmp_path, "BAAI/bge-small-en-v1.5") / "index.faiss").exists()
        assert (_index_dir(tmp_path, "BAAI/bge-base-en-v1.5") / "index.faiss").exists()

    async def test_different_model_directories_are_independent(self, tmp_path):
        async with _run(tmp_path, embed_model="BAAI/bge-small-en-v1.5"):
            pass
        async with _run(tmp_path, embed_model="BAAI/bge-base-en-v1.5"):
            pass

        meta_small = json.loads((_index_dir(tmp_path, "BAAI/bge-small-en-v1.5") / "index_meta.json").read_text())
        meta_base  = json.loads((_index_dir(tmp_path, "BAAI/bge-base-en-v1.5") / "index_meta.json").read_text())
        assert meta_small["embed_model"] != meta_base["embed_model"]

    async def test_each_model_has_own_cache_entry(self, tmp_path):
        from docpilot.cache import DatasetCache
        async with _run(tmp_path, embed_model="BAAI/bge-small-en-v1.5"):
            pass
        async with _run(tmp_path, embed_model="BAAI/bge-base-en-v1.5"):
            pass

        cache = DatasetCache(str(tmp_path / "cache"))
        assert cache.is_cached(DATASET_NAME, SPLIT, "BAAI/bge-small-en-v1.5", name=DATASET_SUBSET)
        assert cache.is_cached(DATASET_NAME, SPLIT, "BAAI/bge-base-en-v1.5", name=DATASET_SUBSET)

    async def test_slug_replaces_slash_in_directory_name(self, tmp_path):
        async with _run(tmp_path, embed_model="org/my-model"):
            pass
        assert (_index_dir(tmp_path, "org/my-model") / "index.faiss").exists()
        # directory name must not contain a slash
        index_dir = _index_dir(tmp_path, "org/my-model")
        assert "/" not in index_dir.name


# ---------------------------------------------------------------------------
# Cache deduplication
# ---------------------------------------------------------------------------

class TestCacheDedup:
    async def test_second_run_skips_load_dataset(self, tmp_path):
        load_mock = MagicMock(return_value=_mock_dataset())
        with patch("docpilot.indexing.loaders.huggingface.load_dataset", load_mock), \
             patch("docpilot.embedder.bge_embedding.SentenceTransformer", return_value=_mock_st_model()):
            await main(_args(tmp_path))
            await main(_args(tmp_path))
        assert load_mock.call_count == 1

    async def test_force_reruns_despite_cache(self, tmp_path):
        load_mock = MagicMock(return_value=_mock_dataset())
        with patch("docpilot.indexing.loaders.huggingface.load_dataset", load_mock), \
             patch("docpilot.embedder.bge_embedding.SentenceTransformer", return_value=_mock_st_model()):
            await main(_args(tmp_path))
            await main(_args(tmp_path, force=True))
        assert load_mock.call_count == 2

    async def test_second_run_does_not_overwrite_index(self, tmp_path):
        async with _run(tmp_path):
            pass
        mtime_first = (_index_dir(tmp_path) / "index.faiss").stat().st_mtime

        async with _run(tmp_path):
            pass
        assert (_index_dir(tmp_path) / "index.faiss").stat().st_mtime == mtime_first

    async def test_different_model_bypasses_cache(self, tmp_path):
        load_mock = MagicMock(return_value=_mock_dataset())
        with patch("docpilot.indexing.loaders.huggingface.load_dataset", load_mock), \
             patch("docpilot.embedder.bge_embedding.SentenceTransformer", return_value=_mock_st_model()):
            await main(_args(tmp_path, embed_model="BAAI/bge-small-en-v1.5"))
            await main(_args(tmp_path, embed_model="BAAI/bge-base-en-v1.5"))
        assert load_mock.call_count == 2


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

class TestWarnings:
    async def test_cache_hit_emits_warning(self, tmp_path, caplog):
        async with _run(tmp_path):
            pass
        with caplog.at_level(logging.WARNING):
            await main(_args(tmp_path))
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert any(DATASET_NAME in r.message for r in caplog.records)
        assert any(EMBED_MODEL in r.message for r in caplog.records)

    async def test_cache_hit_warning_mentions_force_flag(self, tmp_path, caplog):
        async with _run(tmp_path):
            pass
        with caplog.at_level(logging.WARNING):
            await main(_args(tmp_path))
        assert any("--force" in r.message for r in caplog.records)

    async def test_force_flag_suppresses_cache_skip_warning(self, tmp_path, caplog):
        async with _run(tmp_path):
            pass
        with caplog.at_level(logging.WARNING), \
             patch("docpilot.indexing.loaders.huggingface.load_dataset", return_value=_mock_dataset()), \
             patch("docpilot.embedder.bge_embedding.SentenceTransformer", return_value=_mock_st_model()):
            await main(_args(tmp_path, force=True))
        skip_warnings = [r for r in caplog.records if "Skipping" in r.message]
        assert not skip_warnings


# ---------------------------------------------------------------------------
# Index correctness
# ---------------------------------------------------------------------------

class TestIndexCorrectness:
    async def test_loaded_index_returns_correct_top1(self, tmp_path):
        from docpilot.vector_store.faiss_store import FAISSStore

        fixed_vecs = np.eye(_DIM, dtype=np.float32)[:len(_ROWS)]
        call_count = 0

        def encode(texts, **kwargs):
            nonlocal call_count
            result = fixed_vecs[call_count:call_count + len(texts)]
            call_count += len(texts)
            return result

        st_mock = MagicMock()
        st_mock.encode = MagicMock(side_effect=encode)

        with patch("docpilot.indexing.loaders.huggingface.load_dataset", return_value=_mock_dataset()), \
             patch("docpilot.embedder.bge_embedding.SentenceTransformer", return_value=st_mock):
            await main(_args(tmp_path))

        store = await FAISSStore.load(str(_index_dir(tmp_path)))
        results = await store.query(fixed_vecs[0].tolist(), top_k=1)
        assert results[0].document.id == "yolo-0"
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    async def test_loaded_index_count_matches_dataset_size(self, tmp_path):
        from docpilot.vector_store.faiss_store import FAISSStore

        async with _run(tmp_path):
            pass

        store = await FAISSStore.load(str(_index_dir(tmp_path)))
        assert await store.count() == len(_ROWS)

    async def test_list_indexes_shows_all_models(self, tmp_path, capsys):
        async with _run(tmp_path, embed_model="BAAI/bge-small-en-v1.5"):
            pass
        async with _run(tmp_path, embed_model="BAAI/bge-base-en-v1.5"):
            pass

        list_args = Namespace(list_indexes=True, data_root=str(tmp_path))
        await main(list_args)

        output = capsys.readouterr().out
        assert "bge-small" in output
        assert "bge-base" in output
