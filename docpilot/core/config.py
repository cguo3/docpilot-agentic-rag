from __future__ import annotations

from pydantic import BaseModel


class DocPilotConfig(BaseModel):
    model: str = "claude-sonnet-4-6"
    max_retrieval_iterations: int = 3
    top_k_retrieval: int = 20
    top_k_rerank: int = 10
    max_context_tokens: int = 8000
