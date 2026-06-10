from __future__ import annotations

from abc import ABC, abstractmethod

from ...llm.base import LLMClientBase


class QueryRewriterBase(ABC):

    @abstractmethod
    async def rewrite(self, query: str) -> str:
        """Rewrite query into a form better suited for retrieval."""


class LLMQueryRewriter(QueryRewriterBase):
    """Rewrite query using an LLM to improve retrieval precision."""

    def __init__(self, llm: LLMClientBase, model: str) -> None:
        self._llm = llm
        self._model = model

    async def rewrite(self, query: str) -> str:
        raise NotImplementedError
