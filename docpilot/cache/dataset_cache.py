from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


class DatasetCache:
    """File-based cache that tracks ingested dataset+model combinations.

    The cache key is a SHA-256 hash of (dataset_name, name, split, embed_model),
    so changing any of these triggers a fresh ingestion run.
    """

    def __init__(self, cache_dir: str = ".cache") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._cache_dir / "dataset_hashes.json"

    def _load(self) -> dict[str, dict]:
        if self._cache_file.exists():
            return json.loads(self._cache_file.read_text())
        return {}

    def _save(self, data: dict[str, dict]) -> None:
        self._cache_file.write_text(json.dumps(data, indent=2))

    @staticmethod
    def _make_key(
        dataset_name: str,
        split: str,
        name: str | None,
        embed_model: str,
    ) -> str:
        raw = f"{dataset_name}:{name or ''}:{split}:{embed_model}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def is_cached(
        self,
        dataset_name: str,
        split: str,
        embed_model: str,
        name: str | None = None,
    ) -> bool:
        key = self._make_key(dataset_name, split, name, embed_model)
        return key in self._load()

    def mark_cached(
        self,
        dataset_name: str,
        split: str,
        embed_model: str,
        name: str | None = None,
        doc_count: int = 0,
    ) -> None:
        key = self._make_key(dataset_name, split, name, embed_model)
        data = self._load()
        data[key] = {
            "dataset": dataset_name,
            "name": name,
            "split": split,
            "embed_model": embed_model,
            "doc_count": doc_count,
            "cached_at": time.time(),
        }
        self._save(data)
