from __future__ import annotations

from pydantic import BaseModel

from .query_analysis import QueryIntent


class RetrievalPlan(BaseModel):
    steps: list[str]
    strategy: str
    max_iterations: int = 3


class RetrievalPlanner:
    """Decide how many retrieval steps to run and in what order."""

    def plan(self, intent: QueryIntent) -> RetrievalPlan:
        raise NotImplementedError
