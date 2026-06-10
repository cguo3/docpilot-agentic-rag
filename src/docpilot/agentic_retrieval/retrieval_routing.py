from __future__ import annotations

from typing import Any

from .retrieval_planning import RetrievalPlan


class RetrievalRouter:
    """Route each retrieval step to the appropriate retriever (dense, sparse, SQL, etc.)."""

    def route(self, plan: RetrievalPlan, query: str) -> list[dict[str, Any]]:
        raise NotImplementedError
