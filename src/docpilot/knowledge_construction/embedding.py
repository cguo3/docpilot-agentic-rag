from __future__ import annotations

from typing import Any

import numpy as np


class Embedder:
    """Encode chunks into dense vector embeddings."""

    def embed(self, chunks: list[dict[str, Any]]) -> list[np.ndarray]:
        raise NotImplementedError
