from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.schema import GeneratedAnswer, SearchResult
from ..generation.answer_generator import AnswerGenerator
from ..retrieval.base import RetrieverBase
from ..retrieval.pre.query_rewriter import QueryRewriterBase
from .state_machine import Machine, State


class RAGState(str, Enum):
    ANALYZE  = "analyze"
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    DONE     = "done"


@dataclass
class RAGContext:
    query: str
    rewritten_query: str = ""
    results: list[SearchResult] = field(default_factory=list)
    answer: GeneratedAnswer | None = None


class RAGWorkflow:
    """Single-pass RAG pipeline driven by the state machine."""

    def __init__(
        self,
        retriever: RetrieverBase,
        generator: AnswerGenerator,
        rewriter: QueryRewriterBase | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._rewriter = rewriter
        self._machine = self._build_machine()

    # ------------------------------------------------------------------ #
    # State actions                                                        #
    # ------------------------------------------------------------------ #

    async def _do_analyze(self, context: dict[str, Any]) -> None:
        ctx: RAGContext = context["ctx"]
        ctx.rewritten_query = (
            await self._rewriter.rewrite(ctx.query) if self._rewriter else ctx.query
        )

    async def _do_retrieve(self, context: dict[str, Any]) -> None:
        ctx: RAGContext = context["ctx"]
        ctx.results = await self._retriever.retrieve(ctx.rewritten_query)

    async def _do_generate(self, context: dict[str, Any]) -> None:
        ctx: RAGContext = context["ctx"]
        packed = "\n\n".join(r.document.text for r in ctx.results)
        ctx.answer = await self._generator.generate(ctx.rewritten_query, packed)

    # ------------------------------------------------------------------ #
    # Machine wiring                                                       #
    # ------------------------------------------------------------------ #

    def _build_machine(self) -> Machine:
        analyze  = State(RAGState.ANALYZE.value,  on_enter=self._do_analyze)
        retrieve = State(RAGState.RETRIEVE.value, on_enter=self._do_retrieve)
        generate = State(RAGState.GENERATE.value, on_enter=self._do_generate)
        done     = State(RAGState.DONE.value,     is_terminal=True)

        machine = Machine([analyze, retrieve, generate, done], initial=analyze)
        machine.add_transition(analyze,  "next", retrieve)
        machine.add_transition(retrieve, "next", generate)
        machine.add_transition(generate, "next", done)
        return machine

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def run(self, query: str) -> GeneratedAnswer:
        ctx = RAGContext(query=query)
        self._machine.reset()
        self._machine.context["ctx"] = ctx

        await self._machine.current.enter(self._machine.context)

        while not self._machine.current.is_terminal:
            await self._machine.trigger("next")

        assert ctx.answer is not None, "Workflow completed without generating an answer"
        return ctx.answer
