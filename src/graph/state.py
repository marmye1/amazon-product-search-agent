"""AgentState 和状态序列化。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, TypedDict, Union

from ..models import SearchRequest, SearchResponse
from ..rag_models import RecommendationResponse
from ..hybrid_models import HybridSearchRequest, HybridSearchResponse
from ..agent_models import RerankResponse


Action = Literal["clarify", "search", "reject", "finalize"]


class ParsedConstraints(TypedDict, total=False):
    """只保存用户明确表达或模型明确标记为 unknown 的约束。"""

    category: str
    category_en: str
    brand: str
    brand_en: str
    use_case: str
    use_case_en: str
    must_have: List[str]
    must_have_en: List[str]
    avoid: List[str]
    avoid_en: List[str]
    locale: str
    search_terms: List[str]
    search_terms_en: List[str]
    retrieval_query: str
    constraint_conflicts: List[str]
    in_scope: bool
    retrieval_eligible: bool
    needs_clarification: bool
    clarification_reason: str


class AgentState(TypedDict, total=False):
    """ 主图在节点之间传递的业务状态。"""

    user_query: str
    effective_query: str
    previous_context: Dict[str, Any]
    context_used: bool
    context_status: str
    context_query: str
    topic_relation: str
    context_decision: Dict[str, Any]
    conversation_context: Dict[str, Any]
    conversation_memory: Dict[str, Any]
    memory_update_status: str
    parsed_constraints: ParsedConstraints
    field_validation_status: str
    clarification_question: Optional[str]
    search_request: Optional[Union[SearchRequest, HybridSearchRequest]]
    search_response: Optional[Union[SearchResponse, HybridSearchResponse]]
    retrieved_search_response: Optional[Union[SearchResponse, HybridSearchResponse]]
    retrieved_candidate_count: int
    related_categories: List[str]
    related_category_queries: List[Dict[str, Any]]
    answer: Optional[RecommendationResponse]
    active_query: str
    rerank_response: Optional[RerankResponse]
    constraint_report: Optional[Dict[str, Any]]
    evidence_check: Optional[Dict[str, Any]]
    rewrite_result: Optional[Dict[str, Any]]
    rewrite_history: List[Dict[str, Any]]
    attempt_count: int
    pipeline_next_action: str
    validation_report: Optional[Dict[str, Any]]
    next_action: Action
    agent_version: str
    retrieval_method: str
    trace_id: str
    trace_nodes: List[str]
    trace_snapshots: List[Dict[str, Any]]
    request_locale: Optional[str]
    errors: List[str]
    final_response: Optional[str]
    top_k: int
    max_products: int


TRACE_INPUT_KEYS: Dict[str, List[str]] = {
    "merge_context": ["user_query", "previous_context"],
    "parse_query": [
        "user_query",
        "effective_query",
        "request_locale",
        "context_used",
        "topic_relation",
        "conversation_context",
    ],
    "validate_fields": ["parsed_constraints", "effective_query", "errors"],
    "decide_next": ["parsed_constraints", "errors"],
    "ask_clarification": ["parsed_constraints"],
    "build_search_request": ["user_query", "active_query", "parsed_constraints", "top_k"],
    "search_products": ["search_request"],
    "prepare_candidates": ["search_response", "parsed_constraints", "max_products"],
    "rewrite_query": [
        "active_query",
        "parsed_constraints",
        "constraint_report",
        "evidence_check",
        "attempt_count",
    ],
    "answer_rag": ["user_query", "effective_query", "search_response", "max_products"],
    "validate_evidence": ["answer", "search_response"],
    "validate": ["answer", "search_response"],
    "reject_pipeline": ["errors", "constraint_report", "evidence_check", "attempt_count"],
    "category_fallback": [
        "user_query",
        "parsed_constraints",
        "retrieved_search_response",
        "constraint_report",
        "final_response",
    ],
    "reject": ["errors", "parsed_constraints", "validation_report"],
    "finalize": ["answer", "search_response", "errors", "next_action"],
    "memory_update": ["user_query", "final_response", "parsed_constraints", "answer"],
}

TRACE_NODE_META: Dict[str, Dict[str, str]] = {
    "merge_context": {
        "display_name": "话题判断与上下文合并",
        "input_description": "本轮问题和上一轮模型提取的问答记忆",
        "output_description": "追问/新话题判断、有效查询和上下文状态",
    },
    "parse_query": {
        "display_name": "需求解析",
        "input_description": "用户自然语言和可选市场条件",
        "output_description": "结构化商品约束和检索词",
    },
    "validate_fields": {
        "display_name": "字段校验",
        "input_description": "模型结构化字段和有效查询",
        "output_description": "归一化字段、检索资格和校验状态",
    },
    "decide_next": {
        "display_name": "路径决策",
        "input_description": "结构化约束和前序错误",
        "output_description": "search、clarify 或 reject",
    },
    "ask_clarification": {
        "display_name": "澄清问题",
        "input_description": "缺失或不明确的商品约束",
        "output_description": "给用户的澄清问题",
    },
    "build_search_request": {
        "display_name": "检索请求构造",
        "input_description": "原问题、活动查询和结构化约束",
        "output_description": "HybridSearchRequest JSON",
    },
    "search_products": {
        "display_name": "混合检索",
        "input_description": "HybridSearchRequest",
        "output_description": "BM25、向量和 RRF 商品候选",
    },
    "prepare_candidates": {
        "display_name": "重排序与硬约束",
        "input_description": "混合候选、约束和推荐数量",
        "output_description": "排序结果、合规候选和约束报告",
    },
    "rewrite_query": {
        "display_name": "有限查询改写",
        "input_description": "当前查询、失败原因和尝试次数",
        "output_description": "下一次查询或停止改写",
    },
    "answer_rag": {
        "display_name": "RAG 回答",
        "input_description": "候选商品和用户问题",
        "output_description": "结构化推荐回答和初步证据结果",
    },
    "validate_evidence": {
        "display_name": "字段级证据校验",
        "input_description": "结构化回答和商品候选",
        "output_description": "grounded、证据链接和不支持声明",
    },
    "validate": {
        "display_name": "回答校验",
        "input_description": "结构化回答和检索结果",
        "output_description": "回答是否通过基础校验",
    },
    "reject_pipeline": {
        "display_name": "安全拒答",
        "input_description": "错误、约束和证据报告",
        "output_description": "不编造推荐的拒答文本",
    },
    "category_fallback": {
        "display_name": "相关类别提示",
        "input_description": "检索候选、约束报告和用户问题",
        "output_description": "无合规商品说明和已验证的相关类别",
    },
    "reject": {
        "display_name": "安全拒答",
        "input_description": "错误、约束和校验报告",
        "output_description": "不编造推荐的拒答文本",
    },
    "finalize": {
        "display_name": "最终输出",
        "input_description": "回答、检索结果和流程状态",
        "output_description": "用户可读回答和运行元数据",
    },
    "memory_update": {
        "display_name": "对话记忆更新",
        "input_description": "本轮用户问题、答案、结构化条件和商品证据",
        "output_description": "供下一轮判断话题和解析指代的短期记忆",
    },
}


def to_serializable(value: Any) -> Any:
    """把内部 dataclass/TypedDict 转成 JSON 可序列化对象。"""

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_serializable(value.to_dict())
    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, set):
        return [to_serializable(item) for item in sorted(value, key=str)]
    return value


def trace_input(state: AgentState, node_name: str) -> Dict[str, Any]:
    """按模块契约提取输入，不把整个历史 AgentState 倾倒到界面。"""

    keys = TRACE_INPUT_KEYS.get(node_name, [])
    return {
        key: to_serializable(state[key])
        for key in keys
        if key in state and key not in {"trace_snapshots", "trace_nodes"}
    }


def trace_output(result: Mapping[str, Any]) -> Dict[str, Any]:
    """把节点返回的 state patch 转成界面可读输出。"""

    return {
        str(key): to_serializable(value)
        for key, value in result.items()
        if key not in {"trace_snapshots", "trace_nodes"}
    }


def state_to_dict(state: Dict[str, Any]) -> Dict[str, Any]:
    """把 LangGraph 返回状态转换成可打印的 JSON 对象。"""

    return to_serializable(state)
