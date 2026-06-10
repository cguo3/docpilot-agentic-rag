"""Unit tests for BGEEmbedder (mocked) and property contracts."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from docpilot.core.schema import Document
from docpilot.embedder.bge_embedding import BGEEmbedder, _QUERY_PREFIX


def _mock_model(dim: int = 384) -> MagicMock:
    """Return a SentenceTransformer mock whose encode() returns unit vectors."""
    model = MagicMock()

    def encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False):
        n = len(texts)
        vecs = np.random.default_rng(42).standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    model.encode = MagicMock(side_effect=encode)
    return model


@pytest.fixture
def embedder_with_mock():
    embedder = BGEEmbedder(model_name="BAAI/bge-small-en-v1.5", batch_size=32)
    embedder._model = _mock_model(dim=384)
    return embedder


class TestProperties:
    def test_model_name(self):
        assert BGEEmbedder().model_name == "BAAI/bge-small-en-v1.5"

    def test_custom_model_name(self):
        e = BGEEmbedder(model_name="BAAI/bge-base-en-v1.5")
        assert e.model_name == "BAAI/bge-base-en-v1.5"

    def test_dimensions(self):
        assert BGEEmbedder().dimensions == 384

    def test_model_not_loaded_at_construction(self):
        e = BGEEmbedder()
        assert e._model is None


class TestEmbedDocuments:
    async def test_returns_list_of_list_float(self, embedder_with_mock):
        docs = [Document(id="a", text="hello"), Document(id="b", text="world")]
        result = await embedder_with_mock.embed_documents(docs)
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], float)

    async def test_output_length_matches_input(self, embedder_with_mock):
        docs = [Document(id=str(i), text=f"doc {i}") for i in range(5)]
        result = await embedder_with_mock.embed_documents(docs)
        assert len(result) == 5

    async def test_vector_dimension_is_384(self, embedder_with_mock):
        docs = [Document(id="a", text="test")]
        result = await embedder_with_mock.embed_documents(docs)
        assert len(result[0]) == 384

    async def test_vectors_are_normalized(self, embedder_with_mock):
        docs = [Document(id="a", text="test")]
        result = await embedder_with_mock.embed_documents(docs)
        norm = np.linalg.norm(result[0])
        assert norm == pytest.approx(1.0, abs=1e-5)

    async def test_empty_docs_returns_empty(self, embedder_with_mock):
        result = await embedder_with_mock.embed_documents([])
        assert result == []

    async def test_model_encode_called_with_texts(self, embedder_with_mock):
        docs = [Document(id="a", text="foo"), Document(id="b", text="bar")]
        await embedder_with_mock.embed_documents(docs)
        call_args = embedder_with_mock._model.encode.call_args
        assert call_args[0][0] == ["foo", "bar"]

    async def test_normalize_embeddings_true(self, embedder_with_mock):
        docs = [Document(id="a", text="t")]
        await embedder_with_mock.embed_documents(docs)
        call_kwargs = embedder_with_mock._model.encode.call_args[1]
        assert call_kwargs["normalize_embeddings"] is True

    async def test_batch_size_forwarded(self, embedder_with_mock):
        docs = [Document(id="a", text="t")]
        await embedder_with_mock.embed_documents(docs)
        call_kwargs = embedder_with_mock._model.encode.call_args[1]
        assert call_kwargs["batch_size"] == 32  # matches fixture batch_size


class TestEmbedQuery:
    async def test_returns_list_float(self, embedder_with_mock):
        result = await embedder_with_mock.embed_query("what is YOLO?")
        assert isinstance(result, list)
        assert isinstance(result[0], float)

    async def test_vector_dimension_is_384(self, embedder_with_mock):
        result = await embedder_with_mock.embed_query("test query")
        assert len(result) == 384

    async def test_vector_is_normalized(self, embedder_with_mock):
        result = await embedder_with_mock.embed_query("test query")
        norm = np.linalg.norm(result)
        assert norm == pytest.approx(1.0, abs=1e-5)

    async def test_query_prefix_prepended(self, embedder_with_mock):
        await embedder_with_mock.embed_query("test query")
        call_args = embedder_with_mock._model.encode.call_args
        encoded_text = call_args[0][0][0]
        assert encoded_text == _QUERY_PREFIX + "test query"

    async def test_query_prefix_not_in_document_encoding(self, embedder_with_mock):
        docs = [Document(id="a", text="test query")]
        await embedder_with_mock.embed_documents(docs)
        call_args = embedder_with_mock._model.encode.call_args
        encoded_text = call_args[0][0][0]
        assert not encoded_text.startswith(_QUERY_PREFIX)


class TestLazyModelLoading:
    async def test_model_loaded_on_first_embed_call(self):
        embedder = BGEEmbedder()
        assert embedder._model is None

        with patch("docpilot.embedder.bge_embedding.SentenceTransformer") as mock_st:
            mock_st.return_value = _mock_model()
            await embedder.embed_query("test")

        mock_st.assert_called_once_with("BAAI/bge-small-en-v1.5")

    def test_model_not_reloaded_on_subsequent_calls(self, embedder_with_mock):
        original_model = embedder_with_mock._model
        embedder_with_mock._get_model()
        assert embedder_with_mock._model is original_model
