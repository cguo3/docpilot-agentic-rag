from __future__ import annotations

from ..core.schema import Citation, GeneratedAnswer, SearchResult


class ContextPacker:
    """Select and format retrieved chunks into a prompt context window."""

    async def pack(self, results: list[SearchResult], max_tokens: int = 8000) -> str:
        raise NotImplementedError


class AnswerGenerator:
    """Call the LLM to produce a grounded answer from packed context."""

    async def generate(self, query: str, context: str) -> GeneratedAnswer:
        raise NotImplementedError


class CitationExtractor:
    """Parse inline citations from the generated answer text."""

    async def extract(self, answer: str) -> list[Citation]:
        raise NotImplementedError
