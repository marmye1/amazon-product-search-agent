"""API 的 Pydantic 请求、响应和错误契约。"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchRequest(_StrictModel):
    query: str = Field(min_length=1)
    locale: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    top_k: StrictInt = Field(default=10, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query 不能为空")
        return value

    @field_validator("locale", "category", "brand")
    @classmethod
    def normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


class ChatRequest(_StrictModel):
    message: str = Field(min_length=1)
    session_id: Optional[str] = None
    locale: Optional[str] = None
    top_k: Optional[StrictInt] = Field(default=None, ge=1, le=100)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message 不能为空")
        return value

    @field_validator("session_id", "locale")
    @classmethod
    def normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


class SearchResult(_StrictModel):
    product_id: str
    title: str
    brand: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    bullet_points: List[str] = Field(default_factory=list)
    score: float
    matched_fields: List[str] = Field(default_factory=list)
    source_ref: str
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    retrieval_channels: List[str] = Field(default_factory=list)


class SearchResponse(_StrictModel):
    query: str
    results: List[SearchResult]
    total: int
    search_backend: str = "opensearch"
    retrieval_method: Literal["bm25", "hybrid_rrf"]
    warnings: List[str] = Field(default_factory=list)
    retrieval_channels: dict[str, List[str]] = Field(default_factory=dict)
    fusion_config: dict = Field(default_factory=dict)
    query_embedding_model: Optional[str] = None
    trace_id: str


class Recommendation(_StrictModel):
    product_id: str
    title: str
    reason: str
    evidence_source_ids: List[str] = Field(default_factory=list)


class Evidence(_StrictModel):
    source_id: str
    product_id: str
    field_name: str
    quoted_or_paraphrased_fact: str


class ModuleTrace(_StrictModel):
    sequence: int
    node_name: str
    display_name: str
    input_description: str
    output_description: str
    input_format: str
    output_format: str
    input_keys: List[str] = Field(default_factory=list)
    output_keys: List[str] = Field(default_factory=list)
    input: dict = Field(default_factory=dict)
    output: dict = Field(default_factory=dict)
    duration_ms: int = 0
    status: Literal["completed", "degraded"]


class ChatResponse(_StrictModel):
    answer: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    related_categories: List[str] = Field(default_factory=list)
    trace_id: str
    agent_version: str
    retrieval_method: Literal["not_run", "bm25", "hybrid_rrf"]
    execution_trace: List[ModuleTrace] = Field(default_factory=list)


class HealthResponse(_StrictModel):
    status: Literal["ok", "degraded"]
    api: Literal["ok"] = "ok"
    opensearch: str
    llm: str
    index_version: str
    app_version: str


class ErrorBody(_StrictModel):
    code: str
    message: str
    trace_id: str


class ErrorResponse(_StrictModel):
    error: ErrorBody
