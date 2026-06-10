from __future__ import annotations

from abc import ABC, abstractmethod

from .rag_workflow import RAGWorkflow


class ToolBase(ABC):
    name: str
    description: str

    @abstractmethod
    async def run(self, input: str) -> str:
        """Execute the tool and return a string observation."""


class RAGTool(ToolBase):
    name = "rag_search"
    description = (
        "Search the knowledge base for relevant information. "
        "Input should be a clear, specific question or search query."
    )

    def __init__(self, workflow: RAGWorkflow) -> None:
        self._workflow = workflow

    async def run(self, input: str) -> str:
        answer = await self._workflow.run(input)
        return answer.answer
