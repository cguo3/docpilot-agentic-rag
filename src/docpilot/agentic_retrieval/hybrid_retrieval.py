from __future__ import annotations

from typing import Any


class HybridRetriever:
    """Fuse dense vector search and sparse keyword (BM25) results."""

    def retrieve(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        raise NotImplementedError
