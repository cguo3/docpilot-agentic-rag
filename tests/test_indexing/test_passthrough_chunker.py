"""Unit tests for PassThroughChunker."""
from __future__ import annotations

from docpilot.core.schema import Document
from docpilot.indexing.chunker.passthrough import PassThroughChunker


class TestPassThroughChunker:
    async def test_returns_same_documents(self):
        docs = [Document(id="a", text="hello"), Document(id="b", text="world")]
        chunker = PassThroughChunker()
        result = await chunker.chunk(docs)
        assert result is docs  # same object, not a copy

    async def test_empty_list(self):
        chunker = PassThroughChunker()
        result = await chunker.chunk([])
        assert result == []

    async def test_preserves_order(self):
        docs = [Document(id=str(i), text=f"doc {i}") for i in range(10)]
        chunker = PassThroughChunker()
        result = await chunker.chunk(docs)
        assert [d.id for d in result] == [str(i) for i in range(10)]

    async def test_preserves_metadata(self):
        docs = [Document(id="x", text="t", metadata={"key": "value"})]
        chunker = PassThroughChunker()
        result = await chunker.chunk(docs)
        assert result[0].metadata == {"key": "value"}
