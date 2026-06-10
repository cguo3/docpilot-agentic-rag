from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class FaithfulnessResult(BaseModel):
    score: float
    hallucinated_spans: list[str]


class FaithfulnessEvaluator:
    """Score how well the answer is grounded in the retrieved context."""

    def evaluate(self, answer: str, context_chunks: list[dict[str, Any]]) -> FaithfulnessResult:
        raise NotImplementedError
