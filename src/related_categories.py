"""无合规商品时的相关商品类别建议。

类别建议只返回类别名称，不返回具体商品。每个类别都要再经过一次小范围
商品库检索确认，避免把模型常识误报成商品库中存在的类别。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from .search_config import config_value
from .hybrid_models import HybridSearchRequest


@dataclass(frozen=True)
class RelatedCategoryCandidate:
    """一个待验证的相关类别。"""

    label: str
    query: str


@dataclass(frozen=True)
class _TopicRule:
    aliases: Tuple[str, ...]
    related: Tuple[RelatedCategoryCandidate, ...]


_TOPIC_RULES: Tuple[_TopicRule, ...] = (
    _TopicRule(
        aliases=("mouse pad", "鼠标垫"),
        related=(
            RelatedCategoryCandidate("鼠标", "mouse"),
            RelatedCategoryCandidate("桌垫", "desk mat"),
            RelatedCategoryCandidate("键盘", "keyboard"),
        ),
    ),
    _TopicRule(
        aliases=("wireless mouse", "computer mouse", "mice", "mouse", "鼠标"),
        related=(
            RelatedCategoryCandidate("鼠标垫", "mouse pad"),
            RelatedCategoryCandidate("键盘", "keyboard"),
            RelatedCategoryCandidate("USB 集线器", "USB hub"),
        ),
    ),
    _TopicRule(
        aliases=("keyboard", "keyboards", "键盘"),
        related=(
            RelatedCategoryCandidate("鼠标", "mouse"),
            RelatedCategoryCandidate("键盘腕托", "keyboard wrist rest"),
            RelatedCategoryCandidate("桌垫", "desk mat"),
        ),
    ),
    _TopicRule(
        aliases=("monitor", "display", "screen", "显示器", "屏幕"),
        related=(
            RelatedCategoryCandidate("显示器支架", "monitor stand"),
            RelatedCategoryCandidate("笔记本支架", "laptop stand"),
            RelatedCategoryCandidate("摄像头", "webcam"),
        ),
    ),
    _TopicRule(
        aliases=("laptop", "notebook", "笔记本电脑"),
        related=(
            RelatedCategoryCandidate("笔记本支架", "laptop stand"),
            RelatedCategoryCandidate("扩展坞", "docking station"),
            RelatedCategoryCandidate("无线鼠标", "wireless mouse"),
        ),
    ),
    _TopicRule(
        aliases=("headphones", "headset", "earbuds", "耳机"),
        related=(
            RelatedCategoryCandidate("耳机支架", "headphone stand"),
            RelatedCategoryCandidate("麦克风", "microphone"),
            RelatedCategoryCandidate("音箱", "speakers"),
        ),
    ),
    _TopicRule(
        aliases=("smartphone", "mobile phone", "phone", "手机"),
        related=(
            RelatedCategoryCandidate("手机壳", "phone case"),
            RelatedCategoryCandidate("手机钢化膜", "screen protector"),
            RelatedCategoryCandidate("手机充电器", "phone charger"),
        ),
    ),
    _TopicRule(
        aliases=("camera", "相机"),
        related=(
            RelatedCategoryCandidate("三脚架", "camera tripod"),
            RelatedCategoryCandidate("相机包", "camera bag"),
            RelatedCategoryCandidate("存储卡", "memory card"),
        ),
    ),
    _TopicRule(
        aliases=("printer", "打印机"),
        related=(
            RelatedCategoryCandidate("墨盒", "ink cartridge"),
            RelatedCategoryCandidate("打印纸", "printer paper"),
            RelatedCategoryCandidate("标签打印机", "label printer"),
        ),
    ),
    _TopicRule(
        aliases=("office desk", "desk", "办公桌", "书桌"),
        related=(
            RelatedCategoryCandidate("办公椅", "office chair"),
            RelatedCategoryCandidate("桌面台灯", "desk lamp"),
            RelatedCategoryCandidate("显示器支架", "monitor stand"),
        ),
    ),
    _TopicRule(
        aliases=("running shoes", "running shoe", "跑鞋"),
        related=(
            RelatedCategoryCandidate("跑步袜", "running socks"),
            RelatedCategoryCandidate("鞋垫", "shoe insoles"),
            RelatedCategoryCandidate("运动手表", "sports watch"),
        ),
    ),
)

_WORD_PATTERN = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]", re.I)


def _contains_alias(text: str, alias: str) -> bool:
    normalized_text = text.casefold()
    normalized_alias = alias.casefold()
    if any("\u3400" <= char <= "\u9fff" for char in normalized_alias):
        return normalized_alias in normalized_text
    return bool(
        re.search(
            r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(normalized_alias),
            normalized_text,
        )
    )


def _topic_text(user_query: str, parsed_constraints: Mapping[str, Any]) -> str:
    values: List[str] = [user_query]
    for field_name in (
        "category",
        "category_en",
        "use_case",
        "use_case_en",
        "search_terms",
        "search_terms_en",
        "retrieval_query",
    ):
        value = parsed_constraints.get(field_name)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(item for item in value if isinstance(item, str))
    return " ".join(values)


def related_category_candidates(
    user_query: str,
    parsed_constraints: Mapping[str, Any],
) -> List[RelatedCategoryCandidate]:
    """根据用户明确的商品语义生成有限的类别候选，不调用模型。"""

    text = _topic_text(user_query, parsed_constraints)
    for rule in _TOPIC_RULES:
        if any(_contains_alias(text, alias) for alias in rule.aliases):
            return list(rule.related)
    return []


def _search_text(result: Any) -> str:
    values = [
        getattr(result, "title", None),
        getattr(result, "category", None),
        getattr(result, "description", None),
    ]
    values.extend(getattr(result, "bullet_points", []) or [])
    return " ".join(value for value in values if isinstance(value, str)).casefold()


def _verified(response: Any, query: str) -> bool:
    results = getattr(response, "results", None)
    if not isinstance(results, list) or not results:
        return False
    terms = [term.casefold() for term in _WORD_PATTERN.findall(query) if len(term.strip()) > 1]
    if not terms:
        return False
    return any(any(term in _search_text(result) for term in terms) for result in results)


def find_verified_related_categories(
    user_query: str,
    parsed_constraints: Mapping[str, Any],
    *,
    search: Callable[[HybridSearchRequest], Any] | None,
    search_config: Mapping[str, Any],
    locale: str | None = None,
    max_items: int = 3,
) -> List[Dict[str, Any]]:
    """只返回在当前商品库中检索到证据的相关类别。"""

    if search is None:
        return []
    candidates = related_category_candidates(user_query, parsed_constraints)
    if not candidates:
        return []

    bm25_k = max(5, min(10, int(config_value(search_config, "retrieval", "bm25_k", 50))))
    vector_k = max(5, min(10, int(config_value(search_config, "retrieval", "vector_k", 50))))
    verified: List[Dict[str, Any]] = []
    seen_queries = set()
    for candidate in candidates:
        if candidate.query.casefold() in seen_queries:
            continue
        seen_queries.add(candidate.query.casefold())
        try:
            response = search(
                HybridSearchRequest(
                    query=candidate.query,
                    locale=locale,
                    top_k=3,
                    bm25_k=bm25_k,
                    vector_k=vector_k,
                )
            )
        except Exception:
            # 相关类别只是降级提示，验证失败不能影响主回答。
            continue
        if not _verified(response, candidate.query):
            continue
        verified.append(
            {
                "label": candidate.label,
                "query": candidate.query,
                "matched_count": len(getattr(response, "results", []) or []),
            }
        )
        if len(verified) >= max_items:
            break
    return verified
