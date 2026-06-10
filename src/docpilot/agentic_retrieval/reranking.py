from __future__ import annotations

from typing import Any


class Reranker:
    """Re-score and reorder retrieved chunks relative to the query."""

    def rerank(self, query: str, chunks: list[dict[str, Any]], top_k: int = 10) -> list[dict[str, Any]]:
        raise NotImplementedError
