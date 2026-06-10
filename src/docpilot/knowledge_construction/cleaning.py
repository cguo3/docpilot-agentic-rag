from __future__ import annotations

from typing import Any


class Cleaner:
    """Normalize and clean raw document text."""

    def clean(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError
