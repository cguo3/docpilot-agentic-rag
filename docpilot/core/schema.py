from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Document(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = {}


class SearchResult(BaseModel):
    document: Document
    score: float


class Message(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str
    input_tokens: int
    output_tokens: int


class QueryIntent(BaseModel):
    original_query: str
    intent_type: str
    sub_questions: list[str]
    filters: dict[str, str]


class RetrievalPlan(BaseModel):
    steps: list[str]
    strategy: str
    max_iterations: int = 3


class SufficiencyResult(BaseModel):
    is_sufficient: bool
    missing_aspects: list[str]
    confidence: float


class GeneratedAnswer(BaseModel):
    answer: str
    raw_response: str
    model: str
    input_tokens: int
    output_tokens: int


class Citation(BaseModel):
    chunk_id: str
    span: str
    source: str


class VerificationResult(BaseModel):
    citation: Citation
    is_supported: bool
    support_score: float


class RetrievalMetrics(BaseModel):
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float


class CitationMetrics(BaseModel):
    precision: float
    recall: float
    f1: float


class FaithfulnessResult(BaseModel):
    score: float
    hallucinated_spans: list[str]


class LatencyCostRecord(BaseModel):
    stage: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


class Trace(BaseModel):
    trace_id: str
    query: str
    stages: list[dict[str, Any]]
    final_answer: str
    error: str | None = None
