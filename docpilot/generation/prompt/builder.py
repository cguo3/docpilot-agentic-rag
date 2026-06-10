from __future__ import annotations

from pathlib import Path

from ...core.schema import Document

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _load(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text()


class PromptBuilder:
    """Render prompt templates with retrieved context and query."""

    def build_rag_prompt(self, query: str, chunks: list[Document]) -> str:
        template = _load("rag.txt")
        context = "\n\n".join(c.text for c in chunks)
        return template.format(query=query, context=context)

    def build_citation_prompt(self, query: str, chunks: list[Document]) -> str:
        template = _load("citation.txt")
        context = "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)
        return template.format(query=query, context=context)
