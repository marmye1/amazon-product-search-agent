"""RAG 适配和最终证据复核节点。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, Optional

from ..build_context import build_context
from ..generate_recommendation import (
    LocalQwenConfig,
    RAGExecutionResult,
    run_rag,
)
from ..grounding_check import check_grounding
from ..models import ContractError, SearchResponse
from ..rag_models import EvidenceItem, RAGRequest, RecommendationItem, RecommendationResponse
from .state import AgentState


def _grounded_candidate_fallback(
    result: RAGExecutionResult,
    request: RAGRequest,
) -> RAGExecutionResult:
    """模型没有给出推荐时，用候选标题生成最小、可回溯的商品结果。"""

    if result.generation.error or result.response.recommendations or not result.context.blocks:
        return result

    title_blocks = {
        block.product_id: block
        for block in result.context.blocks
        if block.field_name == "title"
    }
    recommendations = []
    evidence = []
    for candidate in request.search_response.results[: request.max_products]:
        block = title_blocks.get(candidate.product_id)
        if block is None:
            continue
        recommendations.append(
            RecommendationItem(
                product_id=candidate.product_id,
                title=candidate.title,
                reason="商品标题与当前检索描述相关。",
                evidence_source_ids=[block.source_id],
            )
        )
        evidence.append(
            EvidenceItem(
                source_id=block.source_id,
                product_id=block.product_id,
                field_name=block.field_name,
                quoted_or_paraphrased_fact=block.text,
            )
        )

    if not recommendations:
        return result
    fallback = RecommendationResponse(
        answer="已检索到以下与当前描述相关的商品。",
        recommendations=recommendations,
        evidence=evidence,
        limitations=["仅依据检索到的商品字段，未包含实时价格、库存、评分、销量和配送信息。"],
        grounded=True,
    )
    grounding = check_grounding(fallback, result.context.blocks)
    return replace(result, response=fallback, grounding=grounding)


def answer_node(
    state: AgentState,
    *,
    llm_config: Optional[LocalQwenConfig] = None,
    rag_runner: Callable[..., RAGExecutionResult] = run_rag,
) -> Dict[str, Any]:
    """复用两步 RAG，不复制 Prompt，也不绕过 grounding。"""

    search_response = state.get("search_response")
    if not isinstance(search_response, SearchResponse):
        return {"errors": list(state.get("errors", [])) + ["缺少 SearchResponse，无法调用 RAG"]}

    request = RAGRequest(
        user_query=state.get("effective_query") or state.get("user_query", ""),
        search_response=search_response,
        max_products=int(state.get("max_products", 5)),
    )
    try:
        result = rag_runner(request, llm_config=llm_config)
    except (ContractError, ValueError) as exc:
        return {"errors": list(state.get("errors", [])) + [str(exc)]}

    result = _grounded_candidate_fallback(result, request)

    errors = list(state.get("errors", []))
    if result.generation.error:
        errors.append(result.generation.error.get("message", " RAG 生成失败"))
    return {
        "answer": result.response,
        "validation_report": result.grounding.to_dict(),
        "errors": errors,
    }


def validate_answer_node(state: AgentState) -> Dict[str, Any]:
    """validate 节点再次检查回答输出，并保留模型的 grounded=false。"""

    answer = state.get("answer")
    search_response = state.get("search_response")
    if not isinstance(answer, RecommendationResponse) or not isinstance(search_response, SearchResponse):
        report = {
            "grounded": False,
            "missing_evidence": ["answer 或 search_response 缺失"],
            "invalid_product_ids": [],
            "invalid_source_ids": [],
            "unsupported_claims": [],
        }
        return {"validation_report": report}

    context = build_context(
        search_response,
        max_products=int(state.get("max_products", 5)),
    )
    report = check_grounding(answer, context.blocks).to_dict()
    if not answer.grounded:
        report["grounded"] = False
        missing = list(report.get("missing_evidence", []))
        missing.append(" RecommendationResponse.grounded=false")
        report["missing_evidence"] = sorted(set(missing))
    return {"validation_report": report}
