"""重排序、硬约束、证据校验和查询改写契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping

from .models import ContractError, SearchResult
from .hybrid_models import HybridSearchResult


def _required_text(value: Any, field_name: str, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(code, "%s 不能为空" % field_name)
    return value.strip()


def _positive_int(value: Any, field_name: str, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractError(code, "%s 必须是大于 0 的整数" % field_name)
    return value


@dataclass(frozen=True)
class RerankRequest:
    """候选重排序输入。"""

    user_query: str
    parsed_constraints: Mapping[str, Any]
    candidates: List[SearchResult]
    rerank_top_k: int = 5
    rerank_model_id: str = "rule-rerank-"

    def __post_init__(self) -> None:
        _required_text(self.user_query, "user_query", "invalid_rerank_request")
        if not isinstance(self.parsed_constraints, Mapping):
            raise ContractError("invalid_rerank_request", "parsed_constraints 必须是对象")
        if not isinstance(self.candidates, list) or not all(isinstance(item, SearchResult) for item in self.candidates):
            raise ContractError("invalid_rerank_request", "candidates 必须是 SearchResult 数组")
        _positive_int(self.rerank_top_k, "rerank_top_k", "invalid_rerank_request")
        _required_text(self.rerank_model_id, "rerank_model_id", "invalid_rerank_request")


@dataclass(frozen=True)
class RerankedResult(HybridSearchResult):
    """保留原始商品字段，并增加排序信息。"""

    rerank_score: float = 0.0
    original_score: float = 0.0
    retrieval_method: str = "unknown"
    violated_constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RerankResponse:
    """候选重排序输出。"""

    user_query: str
    results: List[RerankedResult]
    violated_constraints: Dict[str, List[str]] = field(default_factory=dict)
    rerank_model_id: str = "rule-rerank-"
    fallback_used: bool = False

    def __post_init__(self) -> None:
        _required_text(self.user_query, "user_query", "invalid_rerank_response")
        if not isinstance(self.results, list) or not all(isinstance(item, RerankedResult) for item in self.results):
            raise ContractError("invalid_rerank_response", "results 必须是 RerankedResult 数组")
        if not isinstance(self.violated_constraints, dict):
            raise ContractError("invalid_rerank_response", "violated_constraints 必须是对象")
        _required_text(self.rerank_model_id, "rerank_model_id", "invalid_rerank_response")
        if not isinstance(self.fallback_used, bool):
            raise ContractError("invalid_rerank_response", "fallback_used 必须是布尔值")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_query": self.user_query,
            "results": [item.to_dict() for item in self.results],
            "violated_constraints": {key: list(value) for key, value in self.violated_constraints.items()},
            "rerank_model_id": self.rerank_model_id,
            "fallback_used": self.fallback_used,
        }


@dataclass(frozen=True)
class ConstraintCheck:
    """硬约束检查输出；unknown 不会进入 valid_results。"""

    valid_results: List[RerankedResult] = field(default_factory=list)
    violated_constraints: Dict[str, List[str]] = field(default_factory=dict)
    unknown_constraints: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid_results": [item.to_dict() for item in self.valid_results],
            "violated_constraints": {key: list(value) for key, value in self.violated_constraints.items()},
            "unknown_constraints": {key: list(value) for key, value in self.unknown_constraints.items()},
        }


@dataclass(frozen=True)
class EvidenceCheck:
    """回答到商品字段的证据校验结果。"""

    grounded: bool
    unsupported_claims: List[str] = field(default_factory=list)
    invalid_product_ids: List[str] = field(default_factory=list)
    evidence_links: List[Dict[str, Any]] = field(default_factory=list)
    confidence_reason: str = ""
    invalid_source_ids: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueryRewriteResult:
    """一次有限查询改写的结果。"""

    rewritten_query: str
    rewrite_reason: str
    allow_retry: bool
    attempt_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
