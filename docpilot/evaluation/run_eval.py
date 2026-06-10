from __future__ import annotations

from typing import Any

from ..core.schema import LatencyCostRecord, Trace


class LatencyCostMetrics:
    """Collect and report per-stage latency and token cost."""

    async def record(self, record: LatencyCostRecord) -> None:
        raise NotImplementedError

    async def summary(self) -> list[LatencyCostRecord]:
        raise NotImplementedError


class TraceAnalyzer:
    """Store, inspect, and surface failure patterns from request traces."""

    async def record(self, trace: Trace) -> None:
        raise NotImplementedError

    async def failures(self) -> list[Trace]:
        raise NotImplementedError

    async def summary(self) -> dict[str, Any]:
        raise NotImplementedError
