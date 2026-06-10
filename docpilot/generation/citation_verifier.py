from __future__ import annotations

from ..core.schema import Citation, Document, VerificationResult


class CitationVerifier:
    """Check each citation against the source chunks it claims to reference."""

    async def verify(self, citations: list[Citation], documents: list[Document]) -> list[VerificationResult]:
        raise NotImplementedError
