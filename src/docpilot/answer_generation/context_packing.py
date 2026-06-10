from __future__ import annotations

from typing import Any


class ContextPacker:
    """Select and format retrieved chunks into a prompt context window."""

    def pack(self, chunks: list[dict[str, Any]], max_tokens: int = 8000) -> str:
        raise NotImplementedError
