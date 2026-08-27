"""LangGraph 节点：重排序、硬约束、证据校验和有限改写。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping

from ..build_context import build_context
from ..check_constraints import check_constraints
from ..check_evidence import check_evidence
from ..models import SearchResponse
from ..rerank_candidates import rerank_candidates
from ..related_categories import find_verified_related_categories
from ..rewrite_query import MAX_QUERY_REWRITES, rewrite_query
from ..hybrid_models import HybridSearchResponse
from ..agent_models import RerankRequest
from .state import AgentState


def _append_error(state: AgentState, message: str) -> List[str]:
    errors = list(state.get("errors", []))
    if message not in errors:
        errors.append(message)
    return errors


def _filtered_response(response: SearchResponse, results: List[Any]) -> SearchResponse:
    if isinstance(response, HybridSearchResponse):
        channels = {
            result.product_id: list(response.retrieval_channels.get(result.product_id, []))
            for result in results
        }
        return replace(
            response,
            results=results,
            total=len(results),
            retrieval_channels=channels,
        )
    return replace(response, results=results, total=len(results))


def _canonicalize_recommendation_titles(answer: Any, candidates: List[Any]) -> Any:
    """用候选商品的真实标题校正模型可能截断或改写的展示标题。"""

    if answer is None or not hasattr(answer, "recommendations"):
        return answer
    candidates_by_id = {item.product_id: item for item in candidates}
    recommendations = []
    changed = False
    for recommendation in answer.recommendations:
        candidate = candidates_by_id.get(recommendation.product_id)
        if candidate is not None and recommendation.title.strip() != candidate.title.strip():
            recommendation = replace(recommendation, title=candidate.title)
            changed = True
        recommendations.append(recommendation)
    return replace(answer, recommendations=recommendations) if changed else answer


def prepare_candidates_node(state: AgentState) -> Dict[str, Any]:
    """把 HybridSearchResponse 变成通过硬约束的 RAG 输入。"""

    search_response = state.get("search_response")
    if not isinstance(search_response, SearchResponse):
        return {
            "errors": _append_error(state, "缺少 SearchResponse，无法执行候选处理"),
            "pipeline_next_action": "reject",
        }
    if getattr(search_response, "retrieval_method", "bm25") != "hybrid_rrf":
        return {
            "errors": _append_error(
                state,
                "需要 hybrid_rrf 候选，当前检索结果不是 hybrid_rrf",
            ),
            "pipeline_next_action": "reject",
        }

    parsed_constraints = state.get("parsed_constraints", {})
    request = RerankRequest(
        user_query=state.get("active_query") or state.get("user_query", ""),
        parsed_constraints=parsed_constraints,
        candidates=list(search_response.results),
        # 先对全部候选做硬约束检查，再截取最终推荐数量，避免合规商品因初始排名靠后被漏掉。
        rerank_top_k=max(len(search_response.results), int(state.get("max_products", 5))),
    )
    rerank_response = rerank_candidates(request)
    constraint_report = check_constraints(parsed_constraints, rerank_response.results)
    rerank_response = replace(
        rerank_response,
        violated_constraints=constraint_report.violated_constraints,
    )
    filtered = _filtered_response(
        search_response,
        list(constraint_report.valid_results[: int(state.get("max_products", 5))]),
    )
    return {
        "rerank_response": rerank_response,
        "constraint_report": constraint_report.to_dict(),
        "retrieved_search_response": search_response,
        "retrieved_candidate_count": len(search_response.results),
        "search_response": filtered,
        "validation_report": {
            "rerank": rerank_response.to_dict(),
            "constraints": constraint_report.to_dict(),
        },
        "pipeline_next_action": "answer" if constraint_report.valid_results else "rewrite",
    }


def route_after_candidates(state: AgentState) -> str:
    if state.get("pipeline_next_action") == "reject" or state.get("errors"):
        return "reject"
    search_response = state.get("search_response")
    if isinstance(search_response, SearchResponse) and search_response.results:
        return "answer"
    if int(state.get("attempt_count", 0)) < MAX_QUERY_REWRITES:
        return "rewrite"
    return "reject"


def _rewrite_reasons(state: AgentState) -> List[str]:
    reasons: List[str] = []
    constraint_report = state.get("constraint_report") or {}
    for field_name in ("violated_constraints", "unknown_constraints"):
        values = constraint_report.get(field_name, {})
        if isinstance(values, Mapping):
            for product_id, constraints in values.items():
                reasons.append("%s: %s" % (product_id, ", ".join(constraints)))
    evidence_check = state.get("evidence_check") or {}
    for field_name in ("unsupported_claims", "invalid_product_ids", "missing_evidence"):
        values = evidence_check.get(field_name, [])
        if isinstance(values, list):
            reasons.extend(str(value) for value in values)
    if not reasons:
        reasons.append("候选不足或证据不足")
    return reasons


def rewrite_query_node(state: AgentState) -> Dict[str, Any]:
    current_query = state.get("active_query") or state.get("user_query", "")
    result = rewrite_query(
        current_query,
        state.get("parsed_constraints", {}),
        _rewrite_reasons(state),
        int(state.get("attempt_count", 0)),
    )
    history = list(state.get("rewrite_history", []))
    history.append(result.to_dict())
    if not result.allow_retry:
        return {
            "rewrite_result": result.to_dict(),
            "rewrite_history": history,
            "errors": _append_error(state, result.rewrite_reason),
            "pipeline_next_action": "reject",
        }
    return {
        "active_query": result.rewritten_query,
        "attempt_count": result.attempt_count,
        "rewrite_result": result.to_dict(),
        "rewrite_history": history,
        "search_request": None,
        "search_response": None,
        "retrieved_search_response": None,
        "retrieved_candidate_count": 0,
        "answer": None,
        "rerank_response": None,
        "constraint_report": None,
        "evidence_check": None,
        "validation_report": None,
        "pipeline_next_action": "search",
    }


def route_after_rewrite(state: AgentState) -> str:
    return "search" if state.get("pipeline_next_action") == "search" else "reject"


def validate_pipeline_node(state: AgentState) -> Dict[str, Any]:
    """在 RAG 生成后执行字段级 EvidenceCheck。"""

    answer = state.get("answer")
    search_response = state.get("search_response")
    if answer is None or not isinstance(search_response, SearchResponse):
        report = {
            "grounded": False,
            "unsupported_claims": [],
            "invalid_product_ids": [],
            "evidence_links": [],
            "confidence_reason": "answer 或 search_response 缺失。",
            "invalid_source_ids": [],
            "missing_evidence": ["answer 或 search_response 缺失"],
        }
        return {"evidence_check": report, "validation_report": report}

    # product_id 已通过结构化契约校验；展示标题必须直接取候选字段，
    # 否则 Qwen 偶尔把长标题压缩后会被证据校验误判为未找到商品。
    answer = _canonicalize_recommendation_titles(answer, list(search_response.results))
    context = build_context(
        search_response,
        max_products=int(state.get("max_products", 5)),
    )
    report = check_evidence(answer, search_response.results, context.blocks).to_dict()
    return {"answer": answer, "evidence_check": report, "validation_report": report}


def route_after_evidence_validation(state: AgentState) -> str:
    evidence_check = state.get("evidence_check") or {}
    answer = state.get("answer")
    if answer is not None and answer.grounded and evidence_check.get("grounded") is True:
        return "finalize"
    if answer is None:
        return "reject"
    if int(state.get("attempt_count", 0)) < MAX_QUERY_REWRITES:
        return "rewrite"
    return "reject"


def reject_pipeline_node(state: AgentState) -> Dict[str, Any]:
    """输出可解释拒答，不把失败候选伪装成推荐。"""

    errors = state.get("errors", [])
    constraint_report = state.get("constraint_report") or {}
    evidence_check = state.get("evidence_check") or {}
    attempts = int(state.get("attempt_count", 0))
    if any("需要 hybrid_rrf" in error for error in errors):
        message = "需要先获得 hybrid_rrf 混合候选，当前检索结果不满足前置条件。"
    elif attempts >= MAX_QUERY_REWRITES:
        if constraint_report.get("unknown_constraints"):
            message = "检索已执行，但当前商品字段不足以验证用户提出的硬性条件，因此暂不输出商品推荐。"
        else:
            message = "检索已执行，但没有找到与当前描述足够匹配且证据完整的商品，因此暂不输出商品推荐。"
    elif constraint_report and not constraint_report.get("valid_results"):
        message = "没有商品同时满足当前明确筛选条件，因此不输出未经验证的商品推荐。"
    elif evidence_check and evidence_check.get("grounded") is not True:
        message = "回答没有通过字段级证据校验，因此不输出未经验证的商品推荐。"
    elif errors:
        message = "当前请求未能完成：%s" % "；".join(errors)
    else:
        message = "当前无法生成可靠的商品推荐。"
    return {"final_response": message, "next_action": "reject"}


def category_fallback_node(
    state: AgentState,
    *,
    related_category_search: Any = None,
    search_config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """无合规商品时说明原因，并给出已在商品库中验证过的相关类别。"""

    retrieved_response = state.get("retrieved_search_response")
    if not isinstance(retrieved_response, SearchResponse):
        return {"related_categories": [], "related_category_queries": []}

    parsed_constraints = state.get("parsed_constraints")
    if not isinstance(parsed_constraints, Mapping):
        parsed_constraints = {}
    retrieved_count = len(retrieved_response.results)
    constraint_report = state.get("constraint_report") or {}
    if retrieved_count:
        if constraint_report.get("unknown_constraints"):
            message = (
                "检索已执行并召回 %s 个候选，但当前商品字段不足以验证你的硬性条件，"
                "所以没有输出未经验证的具体商品。"
                % retrieved_count
            )
        else:
            message = (
                "检索已执行并召回 %s 个候选，但没有商品同时满足当前条件，"
                "所以没有输出不符合条件的具体商品。"
                % retrieved_count
            )
    else:
        message = "检索已执行，但当前商品库没有找到与当前描述匹配的商品类别。"

    raw_locale = parsed_constraints.get("locale")
    locale = (
        raw_locale
        if isinstance(raw_locale, str) and raw_locale.strip().casefold() not in {"", "unknown", "null", "none", "n/a"}
        else None
    )
    suggestions = find_verified_related_categories(
        str(state.get("user_query") or state.get("effective_query") or ""),
        parsed_constraints,
        search=related_category_search,
        search_config=search_config or {},
        locale=locale,
    )
    labels = [item["label"] for item in suggestions]
    if labels:
        message += "\n可以尝试这些相关商品类别：%s。" % "、".join(labels)
    else:
        message += "\n当前没有可验证的相关商品类别，建议更换商品名称或放宽筛选条件。"

    return {
        "final_response": message,
        "next_action": "reject",
        "related_categories": labels,
        "related_category_queries": suggestions,
        "retrieved_candidate_count": retrieved_count,
    }
