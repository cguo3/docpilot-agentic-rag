from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..core.schema import Document


class DataLoader(ABC):
    """Load raw documents from files, URLs, or other sources."""

    @abstractmethod
    async def load(self, source: str | Path | None = None, **kwargs: Any) -> list[Document]:
        """Load and return documents from the given source."""


class Cleaner:
    """Normalize and clean raw document text. Base implementation is a no-op pass-through."""

    async def clean(self, documents: list[Document]) -> list[Document]:
        return documents


class MetadataExtractor:
    """Extract structured metadata from documents. Base implementation is a no-op pass-through."""

    async def extract(self, documents: list[Document]) -> list[Document]:
        return documents
