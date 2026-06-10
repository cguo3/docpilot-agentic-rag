from __future__ import annotations

from pydantic import BaseModel

from ..answer_generation.citation_verification import VerificationResult


class CitationMetrics(BaseModel):
    precision: float
    recall: float
    f1: float


class CitationEvaluator:
    """Aggregate citation verification results into precision/recall/F1."""

    def evaluate(self, verifications: list[VerificationResult]) -> CitationMetrics:
        raise NotImplementedError
