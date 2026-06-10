"""Unit tests for DatasetCache."""
from __future__ import annotations

import json

import pytest

from docpilot.cache import DatasetCache


class TestInit:
    def test_creates_cache_dir(self, tmp_path):
        DatasetCache(cache_dir=str(tmp_path / "sub" / "cache"))
        assert (tmp_path / "sub" / "cache").exists()

    def test_cache_file_absent_until_first_write(self, tmp_path):
        DatasetCache(cache_dir=str(tmp_path))
        assert not (tmp_path / "dataset_hashes.json").exists()


class TestCacheKey:
    def test_deterministic(self):
        k1 = DatasetCache._make_key("ds", "train", "sub", "model")
        k2 = DatasetCache._make_key("ds", "train", "sub", "model")
        assert k1 == k2

    def test_length_is_16(self):
        assert len(DatasetCache._make_key("ds", "train", None, "model")) == 16

    def test_differs_by_dataset_name(self):
        assert DatasetCache._make_key("ds1", "train", None, "m") != \
               DatasetCache._make_key("ds2", "train", None, "m")

    def test_differs_by_split(self):
        assert DatasetCache._make_key("ds", "train", None, "m") != \
               DatasetCache._make_key("ds", "test", None, "m")

    def test_differs_by_embed_model(self):
        assert DatasetCache._make_key("ds", "train", None, "model-a") != \
               DatasetCache._make_key("ds", "train", None, "model-b")

    def test_differs_by_subset_name(self):
        assert DatasetCache._make_key("ds", "train", "sub-a", "m") != \
               DatasetCache._make_key("ds", "train", "sub-b", "m")

    def test_none_name_treated_as_empty_string(self):
        assert DatasetCache._make_key("ds", "train", None, "m") == \
               DatasetCache._make_key("ds", "train", "", "m")


class TestIsCached:
    def test_not_cached_initially(self, tmp_path):
        cache = DatasetCache(str(tmp_path))
        assert not cache.is_cached("ds", "train", "model")

    def test_cached_after_mark(self, tmp_path):
        cache = DatasetCache(str(tmp_path))
        cache.mark_cached("ds", "train", "model")
        assert cache.is_cached("ds", "train", "model")

    def test_different_model_not_cached(self, tmp_path):
        cache = DatasetCache(str(tmp_path))
        cache.mark_cached("ds", "train", "model-a")
        assert not cache.is_cached("ds", "train", "model-b")

    def test_different_split_not_cached(self, tmp_path):
        cache = DatasetCache(str(tmp_path))
        cache.mark_cached("ds", "train", "model")
        assert not cache.is_cached("ds", "test", "model")

    def test_different_name_not_cached(self, tmp_path):
        cache = DatasetCache(str(tmp_path))
        cache.mark_cached("ds", "train", "model", name="sub-a")
        assert not cache.is_cached("ds", "train", "model", name="sub-b")

    def test_none_and_absent_name_are_equivalent(self, tmp_path):
        cache = DatasetCache(str(tmp_path))
        cache.mark_cached("ds", "train", "model", name=None)
        assert cache.is_cached("ds", "train", "model")  # name defaults to None


class TestMarkCached:
    def test_persists_across_instances(self, tmp_path):
        DatasetCache(str(tmp_path)).mark_cached("ds", "train", "model")
        assert DatasetCache(str(tmp_path)).is_cached("ds", "train", "model")

    def test_stores_all_metadata_fields(self, tmp_path):
        cache = DatasetCache(str(tmp_path))
        cache.mark_cached("ds", "train", "bge-small", name="yolo", doc_count=27207)

        data = json.loads((tmp_path / "dataset_hashes.json").read_text())
        entry = next(iter(data.values()))
        assert entry["dataset"] == "ds"
        assert entry["split"] == "train"
        assert entry["embed_model"] == "bge-small"
        assert entry["name"] == "yolo"
        assert entry["doc_count"] == 27207
        assert "cached_at" in entry

    def test_overwrites_existing_entry(self, tmp_path):
        cache = DatasetCache(str(tmp_path))
        cache.mark_cached("ds", "train", "model", doc_count=10)
        cache.mark_cached("ds", "train", "model", doc_count=20)

        data = json.loads((tmp_path / "dataset_hashes.json").read_text())
        assert len(data) == 1
        assert next(iter(data.values()))["doc_count"] == 20

    def test_multiple_independent_entries(self, tmp_path):
        cache = DatasetCache(str(tmp_path))
        cache.mark_cached("ds1", "train", "model")
        cache.mark_cached("ds2", "train", "model")

        assert cache.is_cached("ds1", "train", "model")
        assert cache.is_cached("ds2", "train", "model")
        data = json.loads((tmp_path / "dataset_hashes.json").read_text())
        assert len(data) == 2
