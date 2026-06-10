from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.schema import Document


class DataLoader:
    """Load raw documents from files, URLs, or other sources."""

    async def load(self, source: str | Path, **kwargs: Any) -> list[Document]:
        raise NotImplementedError


class Cleaner:
    """Normalize and clean raw document text."""

    async def clean(self, documents: list[Document]) -> list[Document]:
        raise NotImplementedError


class MetadataExtractor:
    """Extract structured metadata from documents (title, date, entities, etc.)."""

    async def extract(self, documents: list[Document]) -> list[Document]:
        raise NotImplementedError
