"""条件路由、澄清、搜索请求构造和最终响应节点。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ..models import SearchRequest
from ..output_language import chinese_or_fallback, contains_chinese
from ..search_config import config_value
from ..hybrid_models import HybridSearchRequest
from .parse_query import is_generic_request
from .state import AgentState, ParsedConstraints


def decide_next(constraints: ParsedConstraints) -> str:
    """根据结构化约束返回固定路由枚举。"""

    if not constraints or constraints.get("in_scope") is False:
        return "reject"
    # needs_clarification 只表示结果可能宽泛，不再阻断已有商品/功能语义的检索。
    if constraints.get("retrieval_eligible") is False:
        return "clarify"
    # 兼容旧的注入式解析器：没有新增 retrieval_eligible 时，仍允许有检索词的请求进入搜索。
    if constraints.get("retrieval_eligible") is None and not constraints.get("search_terms"):
        return "clarify"
    return "search"


def decide_next_node(state: AgentState) -> Dict[str, Any]:
    constraints = state.get("parsed_constraints", {})
    if constraints.get("in_scope") is False and is_generic_request(state.get("user_query", "")):
        return {"next_action": "clarify"}
    if state.get("errors") and not state.get("parsed_constraints"):
        return {"next_action": "reject"}
    if constraints.get("constraint_conflicts"):
        return {"next_action": "clarify"}
    return {"next_action": decide_next(constraints)}


def route_after_decision(state: AgentState) -> str:
    action = state.get("next_action")
    return action if action in {"clarify", "search", "reject"} else "reject"


def ask_clarification_node(state: AgentState) -> Dict[str, Any]:
    constraints = state.get("parsed_constraints", {})
    reason = constraints.get("clarification_reason", "unknown")
    if reason == "unknown" or not reason.strip() or not contains_chinese(reason):
        reason = "商品类型、用途或关键筛选条件"
    question = "为了帮你检索商品，请补充：%s。" % reason
    return {
        "clarification_question": question,
        "final_response": question,
        "next_action": "clarify",
    }


def _known(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value and value.lower() != "unknown" else None


def _has_unverifiable_budget_constraint(constraints: Mapping[str, Any]) -> bool:
    values = []
    for field_name in ("must_have", "must_have_en"):
        raw_values = constraints.get(field_name, [])
        if isinstance(raw_values, list):
            values.extend(item for item in raw_values if isinstance(item, str))
    lowered = " ".join(values).casefold()
    return any(token in lowered for token in ("budget", "price", "dollar", "usd", "$", "预算", "价格", "美元", "元"))


def build_search_request_node(
    state: AgentState,
    *,
    search_config: Mapping[str, Any],
    hybrid: bool = False,
) -> Dict[str, Any]:
    """把约束映射到已支持的 SearchRequest 字段。"""

    constraints = state.get("parsed_constraints", {})
    default_top_k = int(config_value(search_config, "search", "default_top_k", 10))
    max_top_k = int(config_value(search_config, "search", "max_top_k", 100))
    strict_category_filter = bool(
        config_value(search_config, "retrieval", "strict_category_filter", False)
    )
    category_value = _known(constraints.get("category_en")) or _known(constraints.get("category"))
    brand_value = _known(constraints.get("brand_en")) or _known(constraints.get("brand"))
    retrieval_query = _known(constraints.get("retrieval_query"))
    values = {
        # user_query 保留本轮原文；effective_query 已合并临时上下文， 改写后再由 active_query 覆盖。
        "query": state.get("active_query") or retrieval_query or state.get("effective_query") or state.get("user_query", ""),
        "locale": _known(constraints.get("locale")),
        # 当前商品库的 category 字段并未完整填充；默认把类别作为召回/排序语义，
        # 不把缺失 category 的商品在 OpenSearch 过滤阶段全部删掉。
        "category": category_value if strict_category_filter else None,
        "brand": brand_value,
        "top_k": state.get("top_k", default_top_k),
    }
    if hybrid:
        request = HybridSearchRequest.from_mapping(
            {
                **values,
                "bm25_k": int(config_value(search_config, "retrieval", "bm25_k", 50)),
                "vector_k": int(config_value(search_config, "retrieval", "vector_k", 50)),
            }
        )
    else:
        request = SearchRequest.from_mapping(
            values,
            default_top_k=default_top_k,
            max_top_k=max_top_k,
        )
    return {"search_request": request, "next_action": "search"}


def route_after_validation(state: AgentState) -> str:
    report = state.get("validation_report") or {}
    answer = state.get("answer")
    if answer is not None and answer.grounded and report.get("grounded") is True:
        return "finalize"
    return "reject"


def reject_node(state: AgentState) -> Dict[str, Any]:
    errors = state.get("errors", [])
    constraints = state.get("parsed_constraints", {})
    report = state.get("validation_report") or {}
    if any("没有检索到商品上下文" in error or "no_results" in error for error in errors):
        message = "没有检索到可用于推荐的商品信息，因此不能编造商品建议。"
    elif report and report.get("grounded") is not True:
        message = "当前回答没有通过证据检查，因此不输出未经验证的商品推荐。"
    elif errors:
        message = "当前请求未能完成：%s" % "；".join(errors)
    elif constraints and constraints.get("in_scope") is False:
        message = "当前 Agent 只支持亚马逊商品检索、比较和购买建议。"
    else:
        message = "当前无法生成可靠的商品推荐。"
    return {"final_response": message, "next_action": "reject"}


def finalize_node(state: AgentState) -> Dict[str, Any]:
    if state.get("final_response"):
        return {"next_action": state.get("next_action", "finalize")}

    answer = state.get("answer")
    if answer is None:
        return {"final_response": "当前没有可输出的回答。", "next_action": "reject"}
    constraints = state.get("parsed_constraints", {})

    lines = [
        chinese_or_fallback(
            answer.answer,
            "根据检索到的商品字段，以下结果可供参考。",
        )
    ]
    for item in answer.recommendations:
        lines.append(
            "- %s（%s）：%s"
            % (
                item.title,
                item.product_id,
                chinese_or_fallback(item.reason, "该商品的相关字段与当前检索条件匹配。"),
            )
        )
    if answer.limitations:
        limitations = [
            chinese_or_fallback(item, "仅依据商品字段，未包含实时价格、库存、评分和配送信息。")
            for item in answer.limitations
        ]
        if _has_unverifiable_budget_constraint(constraints) and not any(
            any(token in item for token in ("价格", "预算", "美元", "price", "budget"))
            for item in limitations
        ):
            limitations.append("当前商品库未提供价格字段，无法验证预算条件。")
        lines.append("说明：%s" % "；".join(limitations))
    elif _has_unverifiable_budget_constraint(constraints):
        lines.append("说明：当前商品库未提供价格字段，无法验证预算条件。")
    if constraints.get("needs_clarification"):
        reason = constraints.get("clarification_reason", "").strip()
        if reason and reason.casefold() != "unknown" and contains_chinese(reason):
            lines.append("如补充%s，可以进一步缩小检索范围。" % reason)
    return {"final_response": "\n".join(lines), "next_action": "finalize"}
