"""混合检索请求、结果和响应契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .models import ContractError, SearchResponse, SearchResult


def _optional_filter(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContractError("invalid_hybrid_request", "%s 必须是字符串或 null" % field_name)
    value = value.strip()
    return value or None


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractError("invalid_hybrid_request", "%s 必须是大于 0 的整数" % field_name)
    return value


@dataclass(frozen=True)
class HybridSearchRequest:
    """两路召回共用的在线检索请求。"""

    query: str
    locale: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    top_k: int = 10
    bm25_k: int = 50
    vector_k: int = 50

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ContractError("invalid_hybrid_request", "query 不能为空")
        object.__setattr__(self, "query", self.query.strip())
        for field_name in ("locale", "category", "brand"):
            object.__setattr__(self, field_name, _optional_filter(getattr(self, field_name), field_name))
        for field_name in ("top_k", "bm25_k", "vector_k"):
            object.__setattr__(self, field_name, _positive_int(getattr(self, field_name), field_name))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HybridSearchRequest":
        if not isinstance(value, Mapping):
            raise ContractError("invalid_hybrid_request", "HybridSearchRequest 必须是对象")
        return cls(
            query=value.get("query"),
            locale=value.get("locale"),
            category=value.get("category"),
            brand=value.get("brand"),
            top_k=value.get("top_k", 10),
            bm25_k=value.get("bm25_k", 50),
            vector_k=value.get("vector_k", 50),
        )

    @classmethod
    def from_search_request(
        cls,
        request: Any,
        *,
        bm25_k: int = 50,
        vector_k: int = 50,
    ) -> "HybridSearchRequest":
        return cls(
            query=request.query,
            locale=request.locale,
            category=request.category,
            brand=request.brand,
            top_k=request.top_k,
            bm25_k=bm25_k,
            vector_k=vector_k,
        )

    def filter_clauses(self) -> List[Dict[str, Any]]:
        filters: List[Dict[str, Any]] = []
        if self.locale:
            filters.append({"term": {"locale": self.locale.lower()}})
        if self.category:
            filters.append({"term": {"category.keyword": self.category}})
        if self.brand:
            filters.append({"term": {"brand.keyword": self.brand}})
        return filters

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "locale": self.locale,
            "category": self.category,
            "brand": self.brand,
            "top_k": self.top_k,
            "bm25_k": self.bm25_k,
            "vector_k": self.vector_k,
        }


@dataclass(frozen=True)
class HybridSearchResult(SearchResult):
    """保留基础商品字段，同时记录两路召回和 RRF 信息。"""

    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    retrieval_channels: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class HybridSearchResponse(SearchResponse):
    """混合检索响应；仍是 SearchResponse 的子类型。"""

    retrieval_method: str = "hybrid_rrf"
    retrieval_channels: Dict[str, List[str]] = field(default_factory=dict)
    fusion_config: Dict[str, Any] = field(default_factory=dict)
    query_embedding_model: str = ""

    def __post_init__(self) -> None:
        if self.retrieval_method not in {"bm25", "hybrid_rrf"}:
            raise ContractError("invalid_hybrid_response", "retrieval_method 必须是 bm25 或 hybrid_rrf")
        if not isinstance(self.retrieval_channels, dict):
            raise ContractError("invalid_hybrid_response", "retrieval_channels 必须是对象")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [result.to_dict() for result in self.results],
            "total": self.total,
            "search_backend": self.search_backend,
            "retrieval_method": self.retrieval_method,
            "retrieval_channels": {key: list(value) for key, value in self.retrieval_channels.items()},
            "fusion_config": dict(self.fusion_config),
            "query_embedding_model": self.query_embedding_model,
            "warnings": list(self.warnings),
        }
