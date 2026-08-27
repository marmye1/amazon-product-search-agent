"""LangGraph 单 Agent 主图。"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Mapping, Optional

from langgraph.graph import END, START, StateGraph

from ..generate_recommendation import LocalQwenConfig, RAGExecutionResult, run_rag
from ..models import ContractError
from ..opensearch_client import BackendError
from .answer_node import answer_node, validate_answer_node
from .context import merge_context_node
from .field_validation import validate_fields_node
from .parse_query import parse_query_node
from .router import (
    ask_clarification_node,
    build_search_request_node,
    decide_next_node,
    finalize_node,
    reject_node,
    route_after_decision,
    route_after_validation,
)
from .state import AgentState, TRACE_NODE_META, trace_input, trace_output
from .pipeline_router import (
    prepare_candidates_node,
    reject_pipeline_node,
    rewrite_query_node,
    route_after_rewrite,
    route_after_candidates,
    route_after_evidence_validation,
    category_fallback_node,
    validate_pipeline_node,
)


def build_graph(
    search_tool: Any,
    *,
    search_config: Mapping[str, Any],
    llm_config: Optional[LocalQwenConfig] = None,
    invoke_query_model: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    invoke_context_model: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    rag_runner: Optional[Callable[..., RAGExecutionResult]] = None,
    related_category_search: Optional[Callable[..., Any]] = None,
    use_agent_pipeline: bool = False,
):
    """构建并编译只有一个 Agent、一个搜索工具的主图。"""

    if use_agent_pipeline and not bool(getattr(search_tool, "is_hybrid", False)):
        raise ValueError("当前流程必须使用 HybridSearchTool")

    builder = StateGraph(AgentState)

    def merge_context(state: AgentState) -> Dict[str, Any]:
        return merge_context_node(
            state,
            config=llm_config,
            invoke_model=invoke_context_model,
        )

    def parse_node(state: AgentState) -> Dict[str, Any]:
        result = parse_query_node(
            state,
            config=llm_config,
            invoke_model=invoke_query_model,
        )
        requested_locale = state.get("request_locale")
        parsed_constraints = result.get("parsed_constraints")
        if requested_locale and isinstance(parsed_constraints, dict):
            parsed_constraints = dict(parsed_constraints)
            parsed_constraints["locale"] = requested_locale
            result["parsed_constraints"] = parsed_constraints
        if use_agent_pipeline:
            result.setdefault("active_query", state.get("effective_query") or state.get("user_query", ""))
            result.setdefault("attempt_count", 0)
            result.setdefault("rewrite_history", [])
        return result

    def validate_fields(state: AgentState) -> Dict[str, Any]:
        return validate_fields_node(state)

    def search_node(state: AgentState) -> Dict[str, Any]:
        request = state.get("search_request")
        if request is None:
            return {"errors": list(state.get("errors", [])) + ["SearchRequest 缺失"]}
        try:
            response = search_tool.invoke(request)
        except (BackendError, ContractError, ValueError) as exc:
            return {"errors": list(state.get("errors", [])) + [str(exc)]}
        return {"search_response": response}

    def answer(state: AgentState) -> Dict[str, Any]:
        return answer_node(
            state,
            llm_config=llm_config,
            rag_runner=rag_runner or run_rag,
        )

    def finalize(state: AgentState) -> Dict[str, Any]:
        """输出用户回答，并在 Agent 外层记录真实工作流元数据。"""

        result = finalize_node(state)
        search_response = state.get("search_response")
        return {
            **result,
            "agent_version": "production",
            "retrieval_method": getattr(
                search_response,
                "retrieval_method",
                "not_run",
            ),
        }

    def category_fallback(state: AgentState) -> Dict[str, Any]:
        return category_fallback_node(
            state,
            related_category_search=related_category_search,
            search_config=search_config,
        )

    def traced_node(name: str, node: Callable[[AgentState], Dict[str, Any]]) -> Callable[[AgentState], Dict[str, Any]]:
        """把实际经过的 LangGraph 节点写入本次请求的非敏感状态摘要。"""

        def wrapped(state: AgentState) -> Dict[str, Any]:
            started = time.monotonic()
            input_payload = trace_input(state, name)
            result = node(state)
            nodes = list(state.get("trace_nodes", []))
            nodes.append(name)
            snapshots = list(state.get("trace_snapshots", []))
            metadata = TRACE_NODE_META.get(
                name,
                {
                    "display_name": name,
                    "input_description": "节点状态输入",
                    "output_description": "节点状态输出",
                },
            )
            snapshots.append(
                {
                    "sequence": len(snapshots) + 1,
                    "node_name": name,
                    "display_name": metadata["display_name"],
                    "input_description": metadata["input_description"],
                    "output_description": metadata["output_description"],
                    "input_format": "JSON",
                    "output_format": "JSON state patch",
                    "input_keys": list(input_payload),
                    "output_keys": list(trace_output(result)),
                    "input": input_payload,
                    "output": trace_output(result),
                    "duration_ms": int(round((time.monotonic() - started) * 1000)),
                    "status": "degraded" if result.get("errors") else "completed",
                }
            )
            return {**result, "trace_nodes": nodes, "trace_snapshots": snapshots}

        return wrapped

    def build_request(state: AgentState) -> Dict[str, Any]:
        return build_search_request_node(
            state,
            search_config=search_config,
            hybrid=bool(getattr(search_tool, "is_hybrid", False)),
        )

    builder.add_node("merge_context", traced_node("merge_context", merge_context))
    builder.add_node("parse_query", traced_node("parse_query", parse_node))
    builder.add_node("validate_fields", traced_node("validate_fields", validate_fields))
    builder.add_node("decide_next", traced_node("decide_next", decide_next_node))
    builder.add_node("ask_clarification", traced_node("ask_clarification", ask_clarification_node))
    builder.add_node("build_search_request", traced_node("build_search_request", build_request))
    builder.add_node("search_products", traced_node("search_products", search_node))
    # State 中已有 answer 字段，LangGraph 不允许节点名与状态字段重名。
    builder.add_node("answer_rag", traced_node("answer_rag", answer))
    builder.add_node("reject", traced_node("reject", reject_node))
    if use_agent_pipeline:
        builder.add_node("prepare_candidates", traced_node("prepare_candidates", prepare_candidates_node))
        builder.add_node("rewrite_query", traced_node("rewrite_query", rewrite_query_node))
        builder.add_node("validate_evidence", traced_node("validate_evidence", validate_pipeline_node))
        builder.add_node("reject_pipeline", traced_node("reject_pipeline", reject_pipeline_node))
        builder.add_node("category_fallback", traced_node("category_fallback", category_fallback))
    else:
        builder.add_node("validate", traced_node("validate", validate_answer_node))

    builder.add_edge(START, "merge_context")
    builder.add_edge("merge_context", "parse_query")
    builder.add_edge("parse_query", "validate_fields")
    builder.add_edge("validate_fields", "decide_next")
    builder.add_conditional_edges(
        "decide_next",
        route_after_decision,
        {
            "clarify": "ask_clarification",
            "search": "build_search_request",
            "reject": "reject",
        },
    )
    builder.add_edge("ask_clarification", "finalize")
    builder.add_edge("build_search_request", "search_products")
    if use_agent_pipeline:
        builder.add_edge("search_products", "prepare_candidates")
        builder.add_conditional_edges(
            "prepare_candidates",
            route_after_candidates,
            {"answer": "answer_rag", "rewrite": "rewrite_query", "reject": "reject_pipeline"},
        )
        builder.add_edge("answer_rag", "validate_evidence")
        builder.add_conditional_edges(
            "validate_evidence",
            route_after_evidence_validation,
            {"finalize": "finalize", "rewrite": "rewrite_query", "reject": "reject_pipeline"},
        )
        builder.add_conditional_edges(
            "rewrite_query",
            route_after_rewrite,
            {"search": "build_search_request", "reject": "reject_pipeline"},
        )
        builder.add_edge("reject_pipeline", "category_fallback")
        builder.add_edge("category_fallback", "finalize")
    else:
        builder.add_edge("search_products", "answer_rag")
        builder.add_edge("answer_rag", "validate")
        builder.add_conditional_edges(
            "validate",
            route_after_validation,
            {"finalize": "finalize", "reject": "reject"},
        )
    builder.add_edge("reject", "finalize")
    builder.add_node("finalize", traced_node("finalize", finalize))
    builder.add_edge("finalize", END)
    return builder.compile()
