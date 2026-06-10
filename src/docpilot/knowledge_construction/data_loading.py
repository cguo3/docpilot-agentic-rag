from __future__ import annotations

from pathlib import Path
from typing import Any


class DataLoader:
    """Load raw documents from files, URLs, or other sources."""

    def load(self, source: str | Path, **kwargs: Any) -> list[dict[str, Any]]:
        raise NotImplementedError
