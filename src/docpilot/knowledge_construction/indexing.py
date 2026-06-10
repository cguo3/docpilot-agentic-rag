from __future__ import annotations

from typing import Any

import numpy as np


class Indexer:
    """Store embeddings and metadata in a vector/keyword index."""

    def index(self, chunks: list[dict[str, Any]], embeddings: list[np.ndarray]) -> None:
        raise NotImplementedError

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[dict[str, Any]]:
        raise NotImplementedError
