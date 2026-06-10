from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class RetrievalMetrics(BaseModel):
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float


class RetrievalEvaluator:
    """Measure retrieval quality against ground-truth relevant chunks."""

    def evaluate(
        self,
        retrieved: list[dict[str, Any]],
        relevant: list[str],
        k: int = 10,
    ) -> RetrievalMetrics:
        raise NotImplementedError
