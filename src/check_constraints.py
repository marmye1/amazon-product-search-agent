"""商品硬约束检查。"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Sequence

from .agent_models import ConstraintCheck, RerankedResult


_UNKNOWN = {"", "unknown", "null", "none", "n/a"}
_SOURCE_LOCALE = re.compile(r":products:([^:]+):", re.I)
_TERM_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.I)
_CONSTRAINT_STOPWORDS = {
    "support",
    "supports",
    "supported",
    "with",
    "must",
    "have",
    "has",
    "need",
    "needs",
    "compatible",
    "compatibility",
    "feature",
    "features",
}
_UNVERIFIABLE_PATTERN = re.compile(
    r"(?:budget|price|priced|dollar|usd|\$|\u9884\u7b97|\u4ef7\u683c|\u5143|\u4eba\u6c11\u5e01)",
    re.I,
)


def _known(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    return "" if cleaned.casefold() in _UNKNOWN else cleaned


def _terms(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _product_text(result: RerankedResult) -> str:
    values = [result.title, result.brand, result.category, result.description, getattr(result, "color", None)]
    values.extend(result.bullet_points or [])
    return " ".join(value for value in values if isinstance(value, str)).casefold()


def _is_unverifiable_catalog_constraint(term: str) -> bool:
    """当前商品库没有价格字段，预算条件不能被当作商品文本硬约束。"""

    return bool(_UNVERIFIABLE_PATTERN.search(term or ""))


def _constraint_satisfied(term: str, text: str) -> bool:
    normalized_term = term.casefold().strip()
    normalized_text = text.casefold()
    if normalized_term in normalized_text:
        return True
    tokens = [token for token in _TERM_PATTERN.findall(normalized_term) if token not in _CONSTRAINT_STOPWORDS]
    return bool(tokens) and all(token in normalized_text for token in tokens)


def _add(target: Dict[str, List[str]], product_id: str, value: str) -> None:
    target.setdefault(product_id, [])
    if value not in target[product_id]:
        target[product_id].append(value)


def _locale_from_source_ref(source_ref: str) -> str:
    match = _SOURCE_LOCALE.search(source_ref)
    return match.group(1).casefold() if match else ""


def check_constraints(
    parsed_constraints: Mapping[str, Any],
    results: Sequence[RerankedResult],
) -> ConstraintCheck:
    """返回满足全部硬约束的商品；无法判断的条件按 unknown 排除。"""

    violated: Dict[str, List[str]] = {}
    unknown: Dict[str, List[str]] = {}
    valid: List[RerankedResult] = []

    expected_brand = _known(parsed_constraints.get("brand_en")) or _known(parsed_constraints.get("brand"))
    expected_category = _known(parsed_constraints.get("category_en")) or _known(parsed_constraints.get("category"))
    expected_locale = _known(parsed_constraints.get("locale"))
    must_have = _terms(parsed_constraints.get("must_have_en")) or _terms(parsed_constraints.get("must_have"))
    avoid = _terms(parsed_constraints.get("avoid_en")) or _terms(parsed_constraints.get("avoid"))

    for result in results:
        product_violations: List[str] = []
        product_unknown: List[str] = []
        text = _product_text(result)

        if expected_brand:
            if not result.brand:
                product_unknown.append("brand:unknown")
            elif result.brand.casefold() != expected_brand.casefold():
                product_violations.append("brand:%s" % expected_brand)

        if expected_category:
            # 当前已加载的  数据 category 字段为空，缺失不能等同于类别不匹配；
            # 如果字段存在且明确冲突，仍然排除该候选。
            if result.category and result.category.casefold() != expected_category.casefold():
                product_violations.append("category:%s" % expected_category)

        if expected_locale:
            actual_locale = _locale_from_source_ref(result.source_ref)
            if not actual_locale:
                product_unknown.append("locale:unknown")
            elif actual_locale != expected_locale.casefold():
                product_violations.append("locale:%s" % expected_locale)

        for term in must_have:
            if _is_unverifiable_catalog_constraint(term):
                continue
            if not _constraint_satisfied(term, text):
                # 商品字段没有该特性时不能把“没提到”当成满足。
                product_unknown.append("must_have:%s:unknown" % term)

        for term in avoid:
            if term.casefold() in text:
                product_violations.append("avoid:%s" % term)

        if product_violations:
            violated[result.product_id] = product_violations
        if product_unknown:
            unknown[result.product_id] = product_unknown
        if not product_violations and not product_unknown:
            valid.append(replace(result, violated_constraints=[]))

    return ConstraintCheck(
        valid_results=valid,
        violated_constraints=violated,
        unknown_constraints=unknown,
    )
