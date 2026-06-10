from __future__ import annotations

from pydantic import BaseModel


class LatencyCostRecord(BaseModel):
    stage: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


class LatencyCostMetrics:
    """Collect and report per-stage latency and token cost."""

    def record(self, record: LatencyCostRecord) -> None:
        raise NotImplementedError

    def summary(self) -> list[LatencyCostRecord]:
        raise NotImplementedError
