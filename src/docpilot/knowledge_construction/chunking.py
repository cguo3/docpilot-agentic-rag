from __future__ import annotations

from typing import Any


class Chunker:
    """Split documents into retrievable chunks."""

    def chunk(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError
