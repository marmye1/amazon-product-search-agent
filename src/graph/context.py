"""Agent 进入需求解析前的短期上下文合并节点。"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from ..conversation_memory import ConversationMemoryError, decide_context_relation
from ..generate_recommendation import LocalQwenConfig
from .state import AgentState


_UNKNOWN = {"", "unknown", "null", "none", "n/a"}


def _known(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    return "" if cleaned.casefold() in _UNKNOWN else cleaned


def _list_values(value: Any) -> Iterable[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _append_unique(target: list[str], values: Iterable[str]) -> None:
    seen = {item.casefold() for item in target}
    for value in values:
        if value.casefold() not in seen:
            target.append(value)
            seen.add(value.casefold())


def _context_phrase(previous_context: Mapping[str, Any]) -> str:
    """把上一轮已提取的显式条件还原成可供本轮解析的短语。"""

    parts: list[str] = []
    active_topic = _known(previous_context.get("active_topic"))
    if active_topic and active_topic.casefold() != "unknown":
        parts.append(active_topic)

    previous_constraints = previous_context.get("parsed_constraints")
    if not isinstance(previous_constraints, Mapping):
        previous_constraints = {}
    for field_name, english_name in (
        ("category", "category_en"),
        ("brand", "brand_en"),
        ("use_case", "use_case_en"),
    ):
        value = _known(previous_constraints.get(english_name)) or _known(previous_constraints.get(field_name))
        if value:
            parts.append(value)

    english_terms = _list_values(previous_constraints.get("search_terms_en"))
    _append_unique(parts, english_terms or _list_values(previous_constraints.get("search_terms")))

    must_have = list(_list_values(previous_constraints.get("must_have_en")))
    if not must_have:
        must_have = list(_list_values(previous_constraints.get("must_have")))
    if must_have:
        parts.append("必须 " + " ".join(must_have))

    avoid = list(_list_values(previous_constraints.get("avoid_en")))
    if not avoid:
        avoid = list(_list_values(previous_constraints.get("avoid")))
    if avoid:
        parts.append("不要 " + " ".join(avoid))

    return " ".join(parts)


def _context_for_parse(previous_context: Mapping[str, Any]) -> Dict[str, Any]:
    """只把模型提取的上一轮问答摘要交给需求解析，不直接拼完整答案。"""

    return {
        key: previous_context.get(key)
        for key in (
            "active_topic",
            "user_summary",
            "answer_summary",
            "unresolved_question",
            "mentioned_products",
            "parsed_constraints",
        )
        if previous_context.get(key) not in (None, "", {}, [])
    }


def merge_context_node(
    state: AgentState,
    *,
    config: Optional[LocalQwenConfig] = None,
    invoke_model: Optional[Callable[[List[Dict[str, str]]], str]] = None,
) -> Dict[str, Any]:
    """判断当前问题是否追问，并按判断结果合并或替换上一轮记忆。"""

    current_query = str(state.get("user_query", "")).strip()
    previous = state.get("previous_context") or {}
    if not isinstance(previous, Mapping) or not previous:
        return {
            "effective_query": current_query,
            "context_used": False,
            "context_status": "no_previous_context",
            "topic_relation": "none",
            "context_decision": {
                "topic_relation": "none",
                "use_previous_context": False,
                "reason": "没有可用的上一轮临时记忆",
            },
            "conversation_context": {},
        }

    try:
        decision = decide_context_relation(
            current_query,
            previous,
            config=config,
            invoke_model=invoke_model,
        )
    except ConversationMemoryError as exc:
        return {
            "effective_query": current_query,
            "context_used": False,
            "context_status": "context_decision_failed",
            "topic_relation": "unknown",
            "context_decision": {
                "topic_relation": "unknown",
                "use_previous_context": False,
                "reason": str(exc),
            },
            "conversation_context": {},
        }

    if not decision["use_previous_context"]:
        return {
            "effective_query": current_query,
            "context_used": False,
            "context_status": "new_topic_replaced",
            "topic_relation": "new_topic",
            "context_decision": decision,
            "conversation_context": {},
        }

    previous_phrase = _context_phrase(previous)
    effective_query = " ".join(part for part in (previous_phrase, current_query) if part).strip()
    return {
        "effective_query": effective_query,
        "context_used": True,
        "context_status": "follow_up_merged",
        "topic_relation": "follow_up",
        "context_decision": decision,
        "context_query": previous_phrase,
        "conversation_context": _context_for_parse(previous),
    }
