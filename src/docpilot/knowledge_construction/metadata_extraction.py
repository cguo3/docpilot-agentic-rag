from __future__ import annotations

from typing import Any


class MetadataExtractor:
    """Extract structured metadata from chunks (title, date, entities, etc.)."""

    def extract(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError
