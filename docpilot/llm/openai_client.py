from __future__ import annotations

from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from ..core.schema import LLMResponse, Message
from .base import LLMClientBase


class OpenAIClient(LLMClientBase):

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, messages: list[Message], model: str, **kwargs: object) -> LLMResponse:
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        )
        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    async def stream(self, messages: list[Message], model: str, **kwargs: object) -> AsyncGenerator[str, None]:
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            stream=True,
            **kwargs,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
