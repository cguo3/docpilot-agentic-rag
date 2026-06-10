from __future__ import annotations

from pydantic import BaseModel


class QueryIntent(BaseModel):
    original_query: str
    intent_type: str
    sub_questions: list[str]
    filters: dict[str, str]


class QueryAnalyzer:
    """Decompose and classify the user query into structured intent."""

    def analyze(self, query: str) -> QueryIntent:
        raise NotImplementedError
