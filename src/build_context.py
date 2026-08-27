"""把 SearchResponse 转成有限长度、可追溯的上下文。"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional

from .models import ContractError, SearchResponse, SearchResult
from .rag_models import ContextBlock, ContextBuildResult


DEFAULT_FIELD_LENGTH_LIMITS: Dict[str, int] = {
    "title": 200,
    "brand": 100,
    "category": 100,
    "description": 1200,
    "bullet_points": 1200,
}
CONTEXT_FIELDS = tuple(DEFAULT_FIELD_LENGTH_LIMITS.keys())


def _validated_limits(field_length_limits: Optional[Mapping[str, int]]) -> Dict[str, int]:
    limits = dict(DEFAULT_FIELD_LENGTH_LIMITS)
    if field_length_limits is None:
        return limits
    unknown_fields = set(field_length_limits) - set(CONTEXT_FIELDS)
    if unknown_fields:
        raise ContractError(
            "invalid_field_limits",
            "不支持的上下文字段长度限制: %s" % ", ".join(sorted(unknown_fields)),
        )
    for field_name, limit in field_length_limits.items():
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ContractError("invalid_field_limits", "%s 的长度限制必须是大于 0 的整数" % field_name)
        limits[field_name] = limit
    return limits


def _field_text(result: SearchResult, field_name: str) -> Optional[str]:
    value = getattr(result, field_name)
    if field_name == "bullet_points":
        if value is None:
            return None
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ContractError("invalid_context_field", "bullet_points 必须是字符串数组")
        text = "；".join(item.strip() for item in value if item.strip())
        return text or None

    if value is None:
        return None
    if not isinstance(value, str):
        raise ContractError("invalid_context_field", "%s 必须是字符串或 null" % field_name)
    value = value.strip()
    return value or None


def build_context(
    search_response: SearchResponse,
    *,
    max_products: int = 5,
    field_length_limits: Optional[Mapping[str, int]] = None,
) -> ContextBuildResult:
    """按商品排名生成 ContextBlock，并记录每个字段的截断次数。"""

    if not isinstance(search_response, SearchResponse):
        raise ContractError("invalid_context_input", "search_response 必须是 SearchResponse")
    if isinstance(max_products, bool) or not isinstance(max_products, int) or max_products < 1:
        raise ContractError("invalid_context_input", "max_products 必须是大于 0 的整数")

    limits = _validated_limits(field_length_limits)
    blocks: List[ContextBlock] = []
    truncation_stats: Dict[str, int] = {}

    for rank, result in enumerate(search_response.results[:max_products], start=1):
        for field_name in CONTEXT_FIELDS:
            text = _field_text(result, field_name)
            if text is None:
                continue

            limit = limits[field_name]
            if len(text) > limit:
                text = text[:limit].rstrip()
                truncation_stats[field_name] = truncation_stats.get(field_name, 0) + 1

            blocks.append(
                ContextBlock(
                    source_id="%s:%s" % (result.product_id, field_name),
                    product_id=result.product_id,
                    field_name=field_name,
                    text=text,
                    rank=rank,
                )
            )

    return ContextBuildResult(blocks=blocks, truncation_stats=truncation_stats)
