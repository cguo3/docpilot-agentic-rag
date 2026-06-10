from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class State:
    name: str
    on_enter: Callable[..., Awaitable[None]] | None = None
    on_exit: Callable[..., Awaitable[None]] | None = None
    is_terminal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    async def enter(self, context: dict[str, Any]) -> None:
        if self.on_enter:
            await self.on_enter(context)

    async def exit(self, context: dict[str, Any]) -> None:
        if self.on_exit:
            await self.on_exit(context)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, State) and self.name == other.name


class InvalidTransitionError(Exception):
    pass


class Machine:
    def __init__(self, states: list[State], initial: State) -> None:
        self.states: dict[str, State] = {s.name: s for s in states}
        self.initial = initial
        self._transitions: dict[tuple[str, str], State] = {}  # (state, trigger) -> target
        self.current: State = initial
        self.context: dict[str, Any] = {}

    def add_transition(self, source: State, trigger: str, target: State) -> None:
        self._transitions[(source.name, trigger)] = target

    async def trigger(self, event: str) -> State:
        key = (self.current.name, event)
        target = self._transitions.get(key)
        if target is None:
            raise InvalidTransitionError(
                f"No transition from '{self.current.name}' on '{event}'"
            )
        await self.current.exit(self.context)
        self.current = target
        await self.current.enter(self.context)
        return self.current

    def reset(self) -> None:
        self.current = self.initial
        self.context = {}
