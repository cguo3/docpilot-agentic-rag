from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]

from ...core.schema import Document
from ..loader import DataLoader

logger = logging.getLogger(__name__)


class HuggingFaceLoader(DataLoader):
    """Load a HuggingFace dataset and convert rows to Documents.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. "squad", "Salesforce/wikitext").
        split: Dataset split to load ("train", "validation", "test", etc.).
        text_field: Row field to use as Document.text.
        id_field: Row field to use as Document.id; auto-generated if None.
        name: Optional dataset config name (e.g. "wikitext-2-raw-v1").
        metadata_field: If set, the named field must be a dict whose contents are
            merged flat into Document.metadata (rather than nested under a key).
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        text_field: str = "text",
        id_field: str | None = None,
        name: str | None = None,
        metadata_field: str | None = None,
    ) -> None:
        self._dataset_name = dataset_name
        self._split = split
        self._text_field = text_field
        self._id_field = id_field
        self._name = name
        self._metadata_field = metadata_field

    async def load(self, source: str | Path | None = None, **kwargs: Any) -> list[Document]:
        """Download and convert the dataset.

        Args:
            source: Overrides constructor dataset_name when provided.
            **kwargs: Forwarded to datasets.load_dataset (e.g. trust_remote_code=True).
        """
        dataset_name = str(source) if source else self._dataset_name

        loop = asyncio.get_running_loop()
        dataset = await loop.run_in_executor(
            None,
            lambda: load_dataset(
                dataset_name,
                name=self._name,
                split=self._split,
                **kwargs,
            ),
        )

        skip_fields = {self._text_field}
        if self._id_field:
            skip_fields.add(self._id_field)
        if self._metadata_field:
            skip_fields.add(self._metadata_field)

        documents: list[Document] = []
        for i, row in enumerate(dataset):
            text = str(row.get(self._text_field, ""))
            doc_id = (
                str(row[self._id_field])
                if self._id_field and self._id_field in row
                else self._make_id(dataset_name, i, text)
            )

            # Start with any remaining scalar fields
            metadata: dict[str, Any] = {k: v for k, v in row.items() if k not in skip_fields}

            # Merge the dedicated metadata dict flat into the top-level metadata
            if self._metadata_field and self._metadata_field in row:
                field_value = row[self._metadata_field]
                if isinstance(field_value, dict):
                    metadata.update(field_value)
                else:
                    metadata[self._metadata_field] = field_value

            documents.append(Document(id=doc_id, text=text, metadata=metadata))

        seen: set[str] = set()
        duplicates: set[str] = set()
        for doc in documents:
            if doc.id in seen:
                duplicates.add(doc.id)
            seen.add(doc.id)
        if duplicates:
            logger.warning(
                "Duplicate document IDs detected in dataset '%s' (split=%s): "
                "%d duplicate(s) — %s. Only the last occurrence will be stored.",
                dataset_name, self._split, len(duplicates),
                ", ".join(sorted(duplicates)[:5]) + (" ..." if len(duplicates) > 5 else ""),
            )

        return documents

    @staticmethod
    def _make_id(dataset_name: str, index: int, text: str) -> str:
        key = f"{dataset_name}:{index}:{text[:64]}"
        return hashlib.md5(key.encode()).hexdigest()[:16]
