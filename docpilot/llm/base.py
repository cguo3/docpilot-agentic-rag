from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from ..core.schema import LLMResponse, Message


class LLMClientBase(ABC):

    @abstractmethod
    async def complete(self, messages: list[Message], model: str, **kwargs: object) -> LLMResponse:
        """Send messages and return the full response."""

    @abstractmethod
    async def stream(self, messages: list[Message], model: str, **kwargs: object) -> AsyncGenerator[str, None]:
        """Send messages and yield response content token by token."""
