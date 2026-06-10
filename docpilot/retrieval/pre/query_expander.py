from __future__ import annotations

from abc import ABC, abstractmethod

from ...llm.base import LLMClientBase


class QueryExpanderBase(ABC):

    @abstractmethod
    async def expand(self, query: str) -> list[str]:
        """Return the original query plus synonyms/related terms to improve recall."""


class LLMQueryExpander(QueryExpanderBase):
    """Generate query variants using an LLM."""

    def __init__(self, llm: LLMClientBase, model: str) -> None:
        self._llm = llm
        self._model = model

    async def expand(self, query: str) -> list[str]:
        raise NotImplementedError
