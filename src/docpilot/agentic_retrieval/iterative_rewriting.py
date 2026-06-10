from __future__ import annotations

from .evidence_sufficiency import SufficiencyResult


class IterativeQueryRewriter:
    """Rewrite the query to fill gaps identified by the sufficiency checker."""

    def rewrite(self, original_query: str, sufficiency: SufficiencyResult) -> str:
        raise NotImplementedError
