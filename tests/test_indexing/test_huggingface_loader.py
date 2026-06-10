"""Unit tests for HuggingFaceLoader."""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from docpilot.indexing.loaders.huggingface import HuggingFaceLoader


def _mock_dataset(rows: list[dict]) -> MagicMock:
    """Return a mock object that iterates like a HuggingFace Dataset."""
    ds = MagicMock()
    ds.__iter__ = MagicMock(return_value=iter(rows))
    ds.__len__ = MagicMock(return_value=len(rows))
    return ds


def _patch_load(rows: list[dict]):
    """Context manager that patches datasets.load_dataset with given rows."""
    return patch(
        "docpilot.indexing.loaders.huggingface.load_dataset",
        return_value=_mock_dataset(rows),
    )


class TestBasicLoad:
    async def test_returns_documents(self):
        rows = [{"_id": "1", "text": "hello"}, {"_id": "2", "text": "world"}]
        loader = HuggingFaceLoader("ds", id_field="_id", text_field="text")
        with _patch_load(rows):
            docs = await loader.load()
        assert len(docs) == 2
        assert docs[0].id == "1"
        assert docs[0].text == "hello"
        assert docs[1].id == "2"

    async def test_empty_dataset_returns_empty_list(self):
        loader = HuggingFaceLoader("ds")
        with _patch_load([]):
            docs = await loader.load()
        assert docs == []

    async def test_source_overrides_dataset_name(self):
        loader = HuggingFaceLoader("original-ds")
        with patch("docpilot.indexing.loaders.huggingface.load_dataset") as mock_ld:
            mock_ld.return_value = _mock_dataset([])
            await loader.load(source="override-ds")
        mock_ld.assert_called_once()
        assert mock_ld.call_args[0][0] == "override-ds"

    async def test_constructor_dataset_used_when_source_is_none(self):
        loader = HuggingFaceLoader("my-ds")
        with patch("docpilot.indexing.loaders.huggingface.load_dataset") as mock_ld:
            mock_ld.return_value = _mock_dataset([])
            await loader.load()
        assert mock_ld.call_args[0][0] == "my-ds"

    async def test_kwargs_forwarded_to_load_dataset(self):
        loader = HuggingFaceLoader("ds")
        with patch("docpilot.indexing.loaders.huggingface.load_dataset") as mock_ld:
            mock_ld.return_value = _mock_dataset([])
            await loader.load(trust_remote_code=True)
        assert mock_ld.call_args[1].get("trust_remote_code") is True

    async def test_name_param_forwarded(self):
        loader = HuggingFaceLoader("ds", name="subset")
        with patch("docpilot.indexing.loaders.huggingface.load_dataset") as mock_ld:
            mock_ld.return_value = _mock_dataset([])
            await loader.load()
        assert mock_ld.call_args[1]["name"] == "subset"

    async def test_split_param_forwarded(self):
        loader = HuggingFaceLoader("ds", split="validation")
        with patch("docpilot.indexing.loaders.huggingface.load_dataset") as mock_ld:
            mock_ld.return_value = _mock_dataset([])
            await loader.load()
        assert mock_ld.call_args[1]["split"] == "validation"


class TestIdField:
    async def test_custom_id_field_used(self):
        rows = [{"_id": "abc123", "text": "content"}]
        loader = HuggingFaceLoader("ds", id_field="_id", text_field="text")
        with _patch_load(rows):
            docs = await loader.load()
        assert docs[0].id == "abc123"

    async def test_id_field_excluded_from_metadata(self):
        rows = [{"_id": "x", "text": "t", "other": "val"}]
        loader = HuggingFaceLoader("ds", id_field="_id", text_field="text")
        with _patch_load(rows):
            docs = await loader.load()
        assert "_id" not in docs[0].metadata

    async def test_auto_id_generated_when_no_id_field(self):
        rows = [{"text": "hello"}]
        loader = HuggingFaceLoader("ds", text_field="text")
        with _patch_load(rows):
            docs = await loader.load()
        assert docs[0].id != ""
        assert len(docs[0].id) == 16  # MD5 hex[:16]

    async def test_auto_id_is_deterministic(self):
        rows = [{"text": "hello"}]
        loader = HuggingFaceLoader("my-ds", text_field="text")
        with _patch_load(rows):
            docs1 = await loader.load()
        with _patch_load(rows):
            docs2 = await loader.load()
        assert docs1[0].id == docs2[0].id

    async def test_auto_ids_differ_across_rows(self):
        rows = [{"text": "hello"}, {"text": "world"}]
        loader = HuggingFaceLoader("ds", text_field="text")
        with _patch_load(rows):
            docs = await loader.load()
        assert docs[0].id != docs[1].id


class TestTextField:
    async def test_text_field_used(self):
        rows = [{"_id": "1", "body": "the body text"}]
        loader = HuggingFaceLoader("ds", text_field="body", id_field="_id")
        with _patch_load(rows):
            docs = await loader.load()
        assert docs[0].text == "the body text"

    async def test_text_field_excluded_from_metadata(self):
        rows = [{"_id": "1", "text": "t", "other": "v"}]
        loader = HuggingFaceLoader("ds", id_field="_id", text_field="text")
        with _patch_load(rows):
            docs = await loader.load()
        assert "text" not in docs[0].metadata

    async def test_missing_text_field_gives_empty_string(self):
        rows = [{"_id": "1", "other": "v"}]
        loader = HuggingFaceLoader("ds", id_field="_id", text_field="text")
        with _patch_load(rows):
            docs = await loader.load()
        assert docs[0].text == ""


class TestMetadataField:
    async def test_metadata_field_merged_flat(self):
        rows = [{"_id": "1", "text": "t", "metadata": {"title": "Doc A", "year": 2024}}]
        loader = HuggingFaceLoader("ds", id_field="_id", text_field="text", metadata_field="metadata")
        with _patch_load(rows):
            docs = await loader.load()
        assert docs[0].metadata["title"] == "Doc A"
        assert docs[0].metadata["year"] == 2024
        assert "metadata" not in docs[0].metadata  # not nested

    async def test_metadata_field_excluded_as_key(self):
        rows = [{"_id": "1", "text": "t", "metadata": {"k": "v"}}]
        loader = HuggingFaceLoader("ds", id_field="_id", text_field="text", metadata_field="metadata")
        with _patch_load(rows):
            docs = await loader.load()
        assert "metadata" not in docs[0].metadata

    async def test_non_dict_metadata_field_stored_as_key(self):
        rows = [{"_id": "1", "text": "t", "metadata": "plain-string"}]
        loader = HuggingFaceLoader("ds", id_field="_id", text_field="text", metadata_field="metadata")
        with _patch_load(rows):
            docs = await loader.load()
        assert docs[0].metadata["metadata"] == "plain-string"

    async def test_extra_scalar_fields_also_in_metadata(self):
        rows = [{"_id": "1", "text": "t", "metadata": {"k": "v"}, "source": "web"}]
        loader = HuggingFaceLoader("ds", id_field="_id", text_field="text", metadata_field="metadata")
        with _patch_load(rows):
            docs = await loader.load()
        assert docs[0].metadata["source"] == "web"
        assert docs[0].metadata["k"] == "v"

    async def test_flat_merge_overwrite_order(self):
        """Scalar field 'title' is overwritten by metadata dict's 'title'."""
        rows = [{"_id": "1", "text": "t", "title": "outer", "metadata": {"title": "inner"}}]
        loader = HuggingFaceLoader("ds", id_field="_id", text_field="text", metadata_field="metadata")
        with _patch_load(rows):
            docs = await loader.load()
        assert docs[0].metadata["title"] == "inner"

    async def test_no_metadata_field_all_extras_in_metadata(self):
        rows = [{"_id": "1", "text": "t", "source": "web", "year": 2024}]
        loader = HuggingFaceLoader("ds", id_field="_id", text_field="text")
        with _patch_load(rows):
            docs = await loader.load()
        assert docs[0].metadata == {"source": "web", "year": 2024}


class TestDuplicateWarnings:
    async def test_no_warning_when_ids_are_unique(self, caplog):
        rows = [{"_id": "1", "text": "a"}, {"_id": "2", "text": "b"}]
        loader = HuggingFaceLoader("ds", id_field="_id", text_field="text")
        with caplog.at_level(logging.WARNING), _patch_load(rows):
            await loader.load()
        assert not caplog.records

    async def test_warning_emitted_on_duplicate_ids(self, caplog):
        rows = [{"_id": "dup", "text": "first"}, {"_id": "dup", "text": "second"}]
        loader = HuggingFaceLoader("ds", id_field="_id", text_field="text")
        with caplog.at_level(logging.WARNING), _patch_load(rows):
            await loader.load()
        assert any("dup" in r.message for r in caplog.records)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    async def test_warning_includes_duplicate_count(self, caplog):
        rows = [{"_id": "x", "text": "a"}, {"_id": "x", "text": "b"}, {"_id": "y", "text": "c"}, {"_id": "y", "text": "d"}]
        loader = HuggingFaceLoader("ds", id_field="_id", text_field="text")
        with caplog.at_level(logging.WARNING), _patch_load(rows):
            await loader.load()
        assert any("2" in r.message for r in caplog.records)

    async def test_warning_includes_dataset_name(self, caplog):
        rows = [{"_id": "dup", "text": "a"}, {"_id": "dup", "text": "b"}]
        loader = HuggingFaceLoader("my-dataset", id_field="_id", text_field="text")
        with caplog.at_level(logging.WARNING), _patch_load(rows):
            await loader.load()
        assert any("my-dataset" in r.message for r in caplog.records)

    async def test_documents_still_returned_despite_duplicates(self, caplog):
        rows = [{"_id": "dup", "text": "a"}, {"_id": "dup", "text": "b"}, {"_id": "unique", "text": "c"}]
        loader = HuggingFaceLoader("ds", id_field="_id", text_field="text")
        with caplog.at_level(logging.WARNING), _patch_load(rows):
            docs = await loader.load()
        assert len(docs) == 3
