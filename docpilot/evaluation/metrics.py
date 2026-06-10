from __future__ import annotations

from ..core.schema import (
    CitationMetrics,
    FaithfulnessResult,
    RetrievalMetrics,
    SearchResult,
    VerificationResult,
)


class RetrievalEvaluator:
    """Measure retrieval quality against ground-truth relevant chunks."""

    async def evaluate(
        self,
        retrieved: list[SearchResult],
        relevant: list[str],
        k: int = 10,
    ) -> RetrievalMetrics:
        raise NotImplementedError


class FaithfulnessEvaluator:
    """Score how well the answer is grounded in the retrieved context."""

    async def evaluate(self, answer: str, context: list[SearchResult]) -> FaithfulnessResult:
        raise NotImplementedError


class CitationEvaluator:
    """Aggregate citation verification results into precision/recall/F1."""

    async def evaluate(self, verifications: list[VerificationResult]) -> CitationMetrics:
        raise NotImplementedError
