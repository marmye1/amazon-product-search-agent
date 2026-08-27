"""把 OpenSearch hits 转成稳定的结果契约。"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .models import ContractError, SearchResponse, SearchResult


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContractError("invalid_result_field", "商品结果文本字段必须是字符串或 null")
    value = value.strip()
    return value or None


def _bullet_points(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [item.strip() for item in value if item.strip()]
    raise ContractError("invalid_result_field", "bullet_points 必须是字符串数组或 null")


def _matched_fields(hit: Mapping[str, Any]) -> List[str]:
    highlight = hit.get("highlight")
    if isinstance(highlight, Mapping):
        return sorted(str(field_name) for field_name in highlight.keys())
    raw_fields = hit.get("_matched_fields", [])
    if isinstance(raw_fields, list) and all(isinstance(item, str) for item in raw_fields):
        return sorted(set(raw_fields))
    return []


def normalize_hit(hit: Mapping[str, Any]) -> SearchResult:
    source = hit.get("_source")
    if not isinstance(source, Mapping):
        raise ContractError("invalid_result", "OpenSearch hit 缺少 _source")

    product_id = source.get("product_id") or hit.get("_id")
    title = source.get("title")
    if not isinstance(product_id, str) or not product_id.strip():
        raise ContractError("invalid_result", "OpenSearch hit 缺少 product_id")
    if not isinstance(title, str) or not title.strip():
        raise ContractError("invalid_result", "OpenSearch hit 缺少 title")

    raw_score = hit.get("_score", 0.0)
    try:
        score = float(raw_score if raw_score is not None else 0.0)
    except (TypeError, ValueError) as exc:
        raise ContractError("invalid_result", "OpenSearch score 不是数字") from exc

    source_ref = source.get("source_ref")
    if not isinstance(source_ref, str) or not source_ref.strip():
        source_ref = "opensearch:%s" % product_id

    return SearchResult(
        product_id=product_id.strip(),
        title=title.strip(),
        brand=_text(source.get("brand")),
        category=_text(source.get("category")),
        description=_text(source.get("description")),
        bullet_points=_bullet_points(source.get("bullet_points")),
        score=score,
        matched_fields=_matched_fields(hit),
        source_ref=source_ref.strip(),
    )


def total_hits(body: Mapping[str, Any]) -> int:
    hits = body.get("hits")
    if not isinstance(hits, Mapping):
        raise ContractError("invalid_result", "OpenSearch 返回缺少 hits")
    total = hits.get("total", 0)
    if isinstance(total, Mapping):
        total = total.get("value", 0)
    if isinstance(total, bool) or not isinstance(total, int):
        raise ContractError("invalid_result", "OpenSearch hits.total 不是整数")
    return max(total, 0)


def normalize_search_response(query: str, body: Mapping[str, Any]) -> SearchResponse:
    hits = body.get("hits")
    if not isinstance(hits, Mapping):
        raise ContractError("invalid_result", "OpenSearch 返回缺少 hits")
    raw_hits = hits.get("hits", [])
    if not isinstance(raw_hits, list):
        raise ContractError("invalid_result", "OpenSearch hits.hits 不是数组")

    results: List[SearchResult] = []
    seen_ids = set()
    for raw_hit in raw_hits:
        if not isinstance(raw_hit, Mapping):
            raise ContractError("invalid_result", "OpenSearch hit 不是对象")
        result = normalize_hit(raw_hit)
        if result.product_id in seen_ids:
            raise ContractError("duplicate_result", "搜索结果包含重复 product_id: %s" % result.product_id)
        seen_ids.add(result.product_id)
        results.append(result)

    warnings = ["no_results"] if total_hits(body) == 0 else []
    return SearchResponse(
        query=query,
        results=results,
        total=total_hits(body),
        warnings=warnings,
    )
