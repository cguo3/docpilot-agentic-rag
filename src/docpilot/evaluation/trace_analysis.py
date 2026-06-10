from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Trace(BaseModel):
    trace_id: str
    query: str
    stages: list[dict[str, Any]]
    final_answer: str
    error: str | None = None


class TraceAnalyzer:
    """Store, inspect, and surface failure patterns from request traces."""

    def record(self, trace: Trace) -> None:
        raise NotImplementedError

    def failures(self) -> list[Trace]:
        raise NotImplementedError

    def summary(self) -> dict[str, Any]:
        raise NotImplementedError
