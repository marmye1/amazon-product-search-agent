"""搜索输入、商品文档和输出契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


class ContractError(ValueError):
    """请求或数据契约不满足时抛出的错误。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> Dict[str, str]:
        return {"code": self.code, "message": self.message}


def _optional_text(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContractError("invalid_field", "%s 必须是字符串或 null" % field_name)
    value = value.strip()
    return value or None


@dataclass(frozen=True)
class ProductDocument:
    """写入 OpenSearch 的一个商品文档。"""

    product_id: str
    locale: str
    title: str
    brand: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    bullet_points: List[str] = field(default_factory=list)
    source_url: Optional[str] = None
    data_provenance: Optional[str] = None
    source_ref: Optional[str] = None
    color: Optional[str] = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, line_number: Optional[int] = None) -> "ProductDocument":
        suffix = "" if line_number is None else "（第 %s 行）" % line_number
        product_id = value.get("product_id")
        locale = value.get("locale")
        title = value.get("title")
        if not isinstance(product_id, str) or not product_id.strip():
            raise ContractError("missing_product_id", "product_id 不能为空%s" % suffix)
        if not isinstance(locale, str) or not locale.strip():
            raise ContractError("missing_locale", "locale 不能为空%s" % suffix)
        if not isinstance(title, str) or not title.strip():
            raise ContractError("missing_title", "title 不能为空%s" % suffix)

        raw_bullets = value.get("bullet_points", [])
        if raw_bullets is None:
            bullets: List[str] = []
        elif isinstance(raw_bullets, str):
            bullets = [raw_bullets.strip()] if raw_bullets.strip() else []
        elif isinstance(raw_bullets, list) and all(isinstance(item, str) for item in raw_bullets):
            bullets = [item.strip() for item in raw_bullets if item.strip()]
        else:
            raise ContractError("invalid_bullet_points", "bullet_points 必须是字符串数组或 null%s" % suffix)

        return cls(
            product_id=product_id.strip(),
            locale=locale.strip().lower(),
            title=title.strip(),
            brand=_optional_text(value.get("brand"), "brand"),
            category=_optional_text(value.get("category"), "category"),
            description=_optional_text(value.get("description"), "description"),
            bullet_points=bullets,
            source_url=_optional_text(value.get("source_url"), "source_url"),
            data_provenance=_optional_text(value.get("data_provenance"), "data_provenance"),
            source_ref=_optional_text(value.get("source_ref"), "source_ref"),
            color=_optional_text(value.get("color"), "color"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchRequest:
    query: str
    locale: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    top_k: int = 10

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, default_top_k: int = 10, max_top_k: int = 100) -> "SearchRequest":
        query = value.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ContractError("invalid_query", "query 不能为空")

        raw_top_k = value.get("top_k", default_top_k)
        if isinstance(raw_top_k, bool) or not isinstance(raw_top_k, int):
            raise ContractError("invalid_top_k", "top_k 必须是整数")
        if raw_top_k < 1 or raw_top_k > max_top_k:
            raise ContractError("invalid_top_k", "top_k 必须在 1 到 %s 之间" % max_top_k)

        fields: Dict[str, Optional[str]] = {}
        for field_name in ("locale", "category", "brand"):
            raw_value = value.get(field_name)
            if raw_value is not None and not isinstance(raw_value, str):
                raise ContractError("invalid_field", "%s 必须是字符串或 null" % field_name)
            fields[field_name] = raw_value.strip() if isinstance(raw_value, str) and raw_value.strip() else None

        return cls(query=query.strip(), top_k=raw_top_k, **fields)


@dataclass(frozen=True)
class SearchResult:
    product_id: str
    title: str
    brand: Optional[str]
    category: Optional[str]
    description: Optional[str]
    bullet_points: List[str]
    score: float
    matched_fields: List[str]
    source_ref: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResponse:
    query: str
    results: List[SearchResult]
    total: int
    search_backend: str = "opensearch"
    retrieval_method: str = "bm25"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [result.to_dict() for result in self.results],
            "total": self.total,
            "search_backend": self.search_backend,
            "retrieval_method": self.retrieval_method,
            "warnings": list(self.warnings),
        }
