"""两步 RAG 的请求、上下文和回答数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from .models import ContractError, SearchResponse


def _required_text(value: Any, field_name: str, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(code, "%s 不能为空" % field_name)
    return value.strip()


def _string_list(value: Any, field_name: str, code: str) -> List[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ContractError(code, "%s 必须是字符串数组" % field_name)
    return [item.strip() for item in value]


def _validate_search_response(value: Any) -> SearchResponse:
    if not isinstance(value, SearchResponse):
        raise ContractError("invalid_rag_request", "search_response 必须是 SearchResponse")

    for index, result in enumerate(value.results, start=1):
        suffix = "（第 %s 个结果）" % index
        _required_text(result.product_id, "product_id%s" % suffix, "invalid_rag_request")
        _required_text(result.title, "title%s" % suffix, "invalid_rag_request")
        if isinstance(result.score, bool) or not isinstance(result.score, (int, float)):
            raise ContractError("invalid_rag_request", "score%s 必须是数字" % suffix)
        _required_text(result.source_ref, "source_ref%s" % suffix, "invalid_rag_request")
    return value


@dataclass(frozen=True)
class RAGRequest:
    """RAG 链的输入。"""

    user_query: str
    search_response: SearchResponse
    answer_language: str = "zh-CN"
    max_products: int = 5

    def __post_init__(self) -> None:
        _required_text(self.user_query, "user_query", "invalid_rag_request")
        _required_text(self.answer_language, "answer_language", "invalid_rag_request")
        if isinstance(self.max_products, bool) or not isinstance(self.max_products, int):
            raise ContractError("invalid_rag_request", "max_products 必须是整数")
        if self.max_products < 1:
            raise ContractError("invalid_rag_request", "max_products 必须大于 0")
        _validate_search_response(self.search_response)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RAGRequest":
        if not isinstance(value, Mapping):
            raise ContractError("invalid_rag_request", "RAGRequest 必须是对象")
        return cls(
            user_query=value.get("user_query"),
            search_response=value.get("search_response"),
            answer_language=value.get("answer_language", "zh-CN"),
            max_products=value.get("max_products", 5),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_query": self.user_query,
            "search_response": self.search_response.to_dict(),
            "answer_language": self.answer_language,
            "max_products": self.max_products,
        }


@dataclass(frozen=True)
class ContextBlock:
    """一个可追溯到商品和字段的上下文片段。"""

    source_id: str
    product_id: str
    field_name: str
    text: str
    rank: int

    def __post_init__(self) -> None:
        for field_name in ("source_id", "product_id", "field_name", "text"):
            _required_text(getattr(self, field_name), field_name, "invalid_context_block")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ContractError("invalid_context_block", "rank 必须是大于 0 的整数")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "product_id": self.product_id,
            "field_name": self.field_name,
            "text": self.text,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class ContextBuildResult:
    """上下文构造结果及固定截断统计。"""

    blocks: List[ContextBlock] = field(default_factory=list)
    truncation_stats: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocks": [block.to_dict() for block in self.blocks],
            "truncation_stats": dict(self.truncation_stats),
        }


@dataclass(frozen=True)
class RecommendationItem:
    """回答中的一个商品建议。"""

    product_id: str
    title: str
    reason: str
    evidence_source_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for field_name in ("product_id", "title", "reason"):
            _required_text(getattr(self, field_name), field_name, "invalid_recommendation")
        _string_list(self.evidence_source_ids, "evidence_source_ids", "invalid_recommendation")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "title": self.title,
            "reason": self.reason,
            "evidence_source_ids": list(self.evidence_source_ids),
        }


@dataclass(frozen=True)
class EvidenceItem:
    """回答使用的一条商品字段证据。"""

    source_id: str
    product_id: str
    field_name: str
    quoted_or_paraphrased_fact: str

    def __post_init__(self) -> None:
        for field_name in ("source_id", "product_id", "field_name", "quoted_or_paraphrased_fact"):
            _required_text(getattr(self, field_name), field_name, "invalid_evidence")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "product_id": self.product_id,
            "field_name": self.field_name,
            "quoted_or_paraphrased_fact": self.quoted_or_paraphrased_fact,
        }


@dataclass(frozen=True)
class RecommendationResponse:
    """面向用户的结构化回答契约。"""

    answer: str
    recommendations: List[RecommendationItem] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    grounded: bool = False
    retrieval_method: str = "bm25"
    answer_version: str = "v2"

    def __post_init__(self) -> None:
        _required_text(self.answer, "answer", "invalid_recommendation_response")
        if not isinstance(self.recommendations, list) or not all(
            isinstance(item, RecommendationItem) for item in self.recommendations
        ):
            raise ContractError("invalid_recommendation_response", "recommendations 必须是 RecommendationItem 数组")
        if not isinstance(self.evidence, list) or not all(isinstance(item, EvidenceItem) for item in self.evidence):
            raise ContractError("invalid_recommendation_response", "evidence 必须是 EvidenceItem 数组")
        _string_list(self.limitations, "limitations", "invalid_recommendation_response")
        if not isinstance(self.grounded, bool):
            raise ContractError("invalid_recommendation_response", "grounded 必须是布尔值")
        if self.retrieval_method != "bm25":
            raise ContractError("invalid_recommendation_response", "retrieval_method 必须固定为 bm25")
        if self.answer_version != "v2":
            raise ContractError("invalid_recommendation_response", "answer_version 必须固定为 v2")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "recommendations": [item.to_dict() for item in self.recommendations],
            "evidence": [item.to_dict() for item in self.evidence],
            "limitations": list(self.limitations),
            "grounded": self.grounded,
            "retrieval_method": self.retrieval_method,
            "answer_version": self.answer_version,
        }
