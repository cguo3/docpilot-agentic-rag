from __future__ import annotations

from pydantic import BaseModel


class GeneratedAnswer(BaseModel):
    answer: str
    raw_response: str
    model: str
    input_tokens: int
    output_tokens: int


class AnswerGenerator:
    """Call the LLM to produce a grounded answer from packed context."""

    def generate(self, query: str, context: str) -> GeneratedAnswer:
        raise NotImplementedError
