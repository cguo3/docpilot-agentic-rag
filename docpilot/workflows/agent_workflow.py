from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.schema import Message
from ..llm.base import LLMClientBase
from .state_machine import Machine, State
from .tools import ToolBase

_SYSTEM_PROMPT = """\
You are a helpful assistant with access to the following tools:

{tool_descriptions}

Use this format strictly:

Thought: reason about what to do
Action: <tool_name>
Action Input: <input to the tool>

When you have enough information to answer, use:

Thought: I now know the final answer
Final Answer: <your answer>
"""

_USER_PROMPT = """\
Question: {query}

{scratchpad}\
"""


# ------------------------------------------------------------------ #
# States                                                               #
# ------------------------------------------------------------------ #

class AgentState(str, Enum):
    THINK   = "think"
    ACT     = "act"
    OBSERVE = "observe"
    DONE    = "done"


# ------------------------------------------------------------------ #
# Context                                                              #
# ------------------------------------------------------------------ #

@dataclass
class Step:
    thought: str
    action: str
    action_input: str
    observation: str = ""


@dataclass
class AgentContext:
    query: str
    steps: list[Step] = field(default_factory=list)
    pending: Step | None = None
    final_answer: str = ""


# ------------------------------------------------------------------ #
# Workflow                                                             #
# ------------------------------------------------------------------ #

class AgentWorkflow:
    """ReAct agent loop driven by the state machine."""

    def __init__(
        self,
        llm: LLMClientBase,
        tools: list[ToolBase],
        model: str = "gpt-4o",
        max_iterations: int = 6,
    ) -> None:
        self._llm = llm
        self._tools: dict[str, ToolBase] = {t.name: t for t in tools}
        self._model = model
        self._max_iterations = max_iterations
        self._machine = self._build_machine()

    # ------------------------------------------------------------------ #
    # Prompt helpers                                                       #
    # ------------------------------------------------------------------ #

    def _tool_descriptions(self) -> str:
        return "\n".join(
            f"- {t.name}: {t.description}" for t in self._tools.values()
        )

    def _scratchpad(self, ctx: AgentContext) -> str:
        lines: list[str] = []
        for step in ctx.steps:
            lines += [
                f"Thought: {step.thought}",
                f"Action: {step.action}",
                f"Action Input: {step.action_input}",
                f"Observation: {step.observation}",
                "",
            ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # LLM output parser                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse(output: str) -> tuple[str, str, str, str]:
        """Return (thought, action, action_input, final_answer)."""
        thought = action = action_input = final_answer = ""
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("Thought:"):
                thought = line[len("Thought:"):].strip()
            elif line.startswith("Action:"):
                action = line[len("Action:"):].strip()
            elif line.startswith("Action Input:"):
                action_input = line[len("Action Input:"):].strip()
            elif line.startswith("Final Answer:"):
                final_answer = line[len("Final Answer:"):].strip()
        return thought, action, action_input, final_answer

    # ------------------------------------------------------------------ #
    # State actions                                                        #
    # ------------------------------------------------------------------ #

    async def _do_think(self, context: dict[str, Any]) -> None:
        ctx: AgentContext = context["ctx"]
        messages = [
            Message(
                role="system",
                content=_SYSTEM_PROMPT.format(tool_descriptions=self._tool_descriptions()),
            ),
            Message(
                role="user",
                content=_USER_PROMPT.format(query=ctx.query, scratchpad=self._scratchpad(ctx)),
            ),
        ]
        response = await self._llm.complete(messages, model=self._model)
        thought, action, action_input, final_answer = self._parse(response.content)

        if final_answer:
            ctx.final_answer = final_answer
            context["next_trigger"] = "finish"
        else:
            ctx.pending = Step(thought=thought, action=action, action_input=action_input)
            context["next_trigger"] = "act"

    async def _do_act(self, context: dict[str, Any]) -> None:
        ctx: AgentContext = context["ctx"]
        assert ctx.pending is not None
        tool = self._tools.get(ctx.pending.action)
        if tool is None:
            ctx.pending.observation = f"Unknown tool '{ctx.pending.action}'. Available: {list(self._tools)}"
        else:
            ctx.pending.observation = await tool.run(ctx.pending.action_input)

    async def _do_observe(self, context: dict[str, Any]) -> None:
        ctx: AgentContext = context["ctx"]
        assert ctx.pending is not None
        ctx.steps.append(ctx.pending)
        ctx.pending = None

    # ------------------------------------------------------------------ #
    # Machine wiring                                                       #
    # ------------------------------------------------------------------ #

    def _build_machine(self) -> Machine:
        think   = State(AgentState.THINK.value,   on_enter=self._do_think)
        act     = State(AgentState.ACT.value,     on_enter=self._do_act)
        observe = State(AgentState.OBSERVE.value, on_enter=self._do_observe)
        done    = State(AgentState.DONE.value,    is_terminal=True)

        machine = Machine([think, act, observe, done], initial=think)
        machine.add_transition(think,   "act",    act)
        machine.add_transition(think,   "finish", done)
        machine.add_transition(act,     "next",   observe)
        machine.add_transition(observe, "next",   think)
        return machine

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def run(self, query: str) -> str:
        ctx = AgentContext(query=query)
        self._machine.reset()
        self._machine.context["ctx"] = ctx

        await self._machine.current.enter(self._machine.context)

        for _ in range(self._max_iterations):
            if self._machine.current.is_terminal:
                break
            trigger = self._machine.context.pop("next_trigger", "next")
            await self._machine.trigger(trigger)

        return ctx.final_answer or "Agent reached max iterations without a final answer."
