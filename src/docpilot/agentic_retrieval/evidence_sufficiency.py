from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SufficiencyResult(BaseModel):
    is_sufficient: bool
    missing_aspects: list[str]
    confidence: float


class EvidenceSufficiencyChecker:
    """Determine whether retrieved evidence is enough to answer the query."""

    def check(self, query: str, chunks: list[dict[str, Any]]) -> SufficiencyResult:
        raise NotImplementedError
