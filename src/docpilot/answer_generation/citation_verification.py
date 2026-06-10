from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .citation_extraction import Citation


class VerificationResult(BaseModel):
    citation: Citation
    is_supported: bool
    support_score: float


class CitationVerifier:
    """Check each citation against the source chunks it claims to reference."""

    def verify(self, citations: list[Citation], chunks: list[dict[str, Any]]) -> list[VerificationResult]:
        raise NotImplementedError
