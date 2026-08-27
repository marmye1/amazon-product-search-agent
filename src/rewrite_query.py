"""有上限、不会放松硬约束的规则查询改写。"""

from __future__ import annotations

from typing import Any, List, Mapping

from .agent_models import QueryRewriteResult


MAX_QUERY_REWRITES = 2
_UNKNOWN = {"", "unknown", "null", "none", "n/a"}


def _known(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    return "" if cleaned.casefold() in _UNKNOWN else cleaned


def _constraint_terms(parsed_constraints: Mapping[str, Any]) -> List[str]:
    values: List[str] = []
    for field_name in (
        "search_terms_en",
        "category_en",
        "brand_en",
        "use_case_en",
        "search_terms",
        "category",
        "brand",
        "use_case",
    ):
        raw_value = parsed_constraints.get(field_name, [])
        if isinstance(raw_value, list):
            values.extend(item.strip() for item in raw_value if isinstance(item, str) and item.strip())
        else:
            value = _known(raw_value)
            if value:
                values.append(value)
    for field_name in ("must_have_en", "must_have"):
        raw_values = parsed_constraints.get(field_name, [])
        if isinstance(raw_values, list):
            values.extend(item.strip() for item in raw_values if isinstance(item, str) and item.strip())
    deduplicated: List[str] = []
    seen = set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            deduplicated.append(value)
    return deduplicated


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())


def _search_terms(parsed_constraints: Mapping[str, Any]) -> List[str]:
    for field_name in ("search_terms_en", "search_terms"):
        raw_values = parsed_constraints.get(field_name, [])
        if isinstance(raw_values, list) and raw_values:
            return [item.strip() for item in raw_values if isinstance(item, str) and item.strip()]
    retrieval_query = _known(parsed_constraints.get("retrieval_query"))
    return retrieval_query.split() if retrieval_query else []


def rewrite_query(
    user_query: str,
    parsed_constraints: Mapping[str, Any],
    failure_reasons: List[str],
    attempt_count: int,
) -> QueryRewriteResult:
    """最多执行两次改写，始终保留 brand/category/must_have/avoid 条件。"""

    current = user_query.strip()
    if attempt_count >= MAX_QUERY_REWRITES:
        return QueryRewriteResult(current, "已达到查询改写上限。", False, attempt_count)

    terms = _constraint_terms(parsed_constraints)
    if not terms:
        return QueryRewriteResult(current, "没有可安全加入查询的结构化检索词。", False, attempt_count)

    if attempt_count == 0:
        # 第一次只补充商品词，不删除用户原始表达；第二次再收敛到全部结构化词。
        candidate = " ".join([current] + (_search_terms(parsed_constraints) or terms))
    else:
        # 第二次只保留结构化词，避免继续扩张语义；硬约束仍在 terms 中。
        candidate = " ".join(terms)

    if _normalise(candidate) == _normalise(current):
        return QueryRewriteResult(current, "改写不会产生新的检索条件。", False, attempt_count)

    reason = "；".join(item for item in failure_reasons if item) or "候选不足或证据不足"
    return QueryRewriteResult(
        rewritten_query=candidate,
        rewrite_reason="第 %s 次有限改写：%s" % (attempt_count + 1, reason[:180]),
        allow_retry=True,
        attempt_count=attempt_count + 1,
    )
