"""LangGraph 单 Agent 的三条主路径和节点契约测试。"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from src.build_context import build_context
from src.generate_recommendation import GenerationMetadata, RAGExecutionResult
from src.graph.build_graph import build_graph
from src.graph.parse_query import QueryParseError, parse_query
from src.graph.state import state_to_dict
from src.grounding_check import GroundingReport
from src.models import SearchRequest
from src.rag_models import (
    EvidenceItem,
    RecommendationItem,
    RecommendationResponse,
)
from src.tools.search_products_tool import SearchProductsTool
from src.fake_search_backend import FakeSearchBackend


SEARCH_CONFIG: Dict[str, Any] = {
    "opensearch": {"index_name": "amazon_products_v1"},
    "search": {
        "default_top_k": 5,
        "max_top_k": 20,
        "field_weights": {"title": 5, "brand": 3, "category": 2, "description": 1, "bullet_points": 1},
    },
    "data": {},
}


def _raw_search_response(*, empty: bool = False) -> Dict[str, Any]:
    if empty:
        return {"hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}}
    return {
        "hits": {
            "total": {"value": 1, "relation": "eq"},
            "hits": [
                {
                    "_score": 4.2,
                    "_source": {
                        "product_id": "p-1",
                        "locale": "us",
                        "title": "Wireless Mouse",
                        "brand": "Brand A",
                        "category": "Computer Mice",
                        "description": "A mouse for office use",
                        "bullet_points": ["Quiet click"],
                        "source_ref": "esci:v0:products:us:p-1",
                    },
                    "highlight": {"title": ["Wireless Mouse"]},
                }
            ],
        }
    }


def _constraints(
    *,
    in_scope: bool = True,
    needs_clarification: bool = False,
    search_terms: Optional[List[str]] = None,
    clarification_reason: str = "unknown",
) -> str:
    return json.dumps(
        {
            "category": "unknown",
            "category_en": "unknown",
            "brand": "unknown",
            "brand_en": "unknown",
            "use_case": "unknown",
            "use_case_en": "unknown",
            "must_have": [],
            "must_have_en": [],
            "avoid": [],
            "avoid_en": [],
            "locale": "unknown",
            "search_terms": search_terms if search_terms is not None else ["wireless", "mouse"],
            "search_terms_en": search_terms if search_terms is not None else ["wireless", "mouse"],
            "retrieval_query": " ".join(search_terms if search_terms is not None else ["wireless", "mouse"]),
            "in_scope": in_scope,
            "needs_clarification": needs_clarification,
            "clarification_reason": clarification_reason,
        },
        ensure_ascii=False,
    )


def _fake_parser(output: str):
    def invoke(_: List[Dict[str, str]]) -> str:
        return output

    return invoke


def _follow_up_context_model(_: List[Dict[str, str]]) -> str:
    return json.dumps(
        {
            "topic_relation": "follow_up",
            "use_previous_context": True,
            "reason": "当前问题补充上一轮商品条件",
        },
        ensure_ascii=False,
    )


def _new_topic_context_model(_: List[Dict[str, str]]) -> str:
    return json.dumps(
        {
            "topic_relation": "new_topic",
            "use_previous_context": False,
            "reason": "当前问题明确切换了商品主题",
        },
        ensure_ascii=False,
    )


def _rag_result(request, *, grounded: bool = True) -> RAGExecutionResult:
    context = build_context(request.search_response, max_products=request.max_products)
    if grounded:
        recommendation = RecommendationItem(
            "p-1",
            "Wireless Mouse",
            "标题和品牌符合无线鼠标检索词。",
            ["p-1:title"],
        )
        evidence = EvidenceItem(
            "p-1:title",
            "p-1",
            "title",
            "Wireless Mouse",
        )
        response = RecommendationResponse(
            "可以考虑这款商品。",
            [recommendation],
            [evidence],
            ["仅依据商品字段。"],
            True,
        )
        report = GroundingReport(True)
        metadata = GenerationMetadata("qwen-local", "http://local.test")
    else:
        response = RecommendationResponse(
            "没有检索到可用于推荐的商品信息。",
            [],
            [],
            ["仅依据商品字段。"],
            False,
        )
        report = GroundingReport(False, missing_evidence=["没有可用于检查的商品上下文"])
        metadata = GenerationMetadata(
            "qwen-local",
            "http://local.test",
            error={"code": "no_results", "message": "没有检索到商品上下文"},
            skipped=True,
        )
    return RAGExecutionResult(
        response=response,
        context=context,
        generation=metadata,
        grounding=report,
    )


def _make_graph(raw_response: Dict[str, Any], parser_output: str, *, grounded: bool = True):
    backend = FakeSearchBackend(raw_response)
    tool = SearchProductsTool(backend, SEARCH_CONFIG)

    def fake_rag(request, **_: Any) -> RAGExecutionResult:
        return _rag_result(request, grounded=grounded)

    graph = build_graph(
        tool,
        search_config=SEARCH_CONFIG,
        invoke_query_model=_fake_parser(parser_output),
        rag_runner=fake_rag,
    )
    return graph, backend


def test_normal_path_is_parse_search_answer_validate() -> None:
    graph, backend = _make_graph(_raw_search_response(), _constraints())

    result = graph.invoke({"user_query": "wireless mouse", "errors": [], "max_products": 3})

    assert result["next_action"] == "finalize"
    assert result["search_request"].query == "wireless mouse"
    assert result["search_response"].results[0].product_id == "p-1"
    assert result["answer"].grounded is True
    assert result["validation_report"]["grounded"] is True
    assert "Wireless Mouse" in result["final_response"]
    assert len(backend.calls) == 1


def test_clarify_path_does_not_call_search_tool() -> None:
    graph, backend = _make_graph(
        _raw_search_response(),
        _constraints(
            needs_clarification=True,
            search_terms=[],
            clarification_reason="商品类型或使用场景",
        ),
    )

    result = graph.invoke({"user_query": "我想买一个东西", "errors": []})

    assert result["next_action"] == "clarify"
    assert result["retrieval_method"] == "not_run"
    assert "商品类型或使用场景" in result["final_response"]
    assert "search_response" not in result
    assert backend.calls == []


def test_function_terms_search_even_when_clarification_is_needed() -> None:
    graph, backend = _make_graph(
        _raw_search_response(),
        _constraints(
            needs_clarification=True,
            search_terms=["办公", "连接电脑"],
            clarification_reason="具体商品类型",
        ),
    )

    result = graph.invoke({"user_query": "办公的、连接电脑的", "errors": [], "max_products": 3})

    assert result["next_action"] == "finalize"
    assert len(backend.calls) == 1
    assert "办公 连接电脑" == result["search_request"].query
    assert "具体商品类型" in result["final_response"]


def test_merge_context_before_query_parse() -> None:
    backend = FakeSearchBackend(_raw_search_response())
    # 上一轮记忆是否参与由独立的上下文判断模型决定。
    graph = build_graph(
        SearchProductsTool(backend, SEARCH_CONFIG),
        search_config=SEARCH_CONFIG,
        invoke_query_model=_fake_parser(_constraints()),
        invoke_context_model=_follow_up_context_model,
        rag_runner=lambda request, **_: _rag_result(request),
    )

    result = graph.invoke(
        {
            "user_query": "mouse",
            "previous_context": {
                "pending_clarification": True,
                "parsed_constraints": {
                    "category": "unknown",
                    "brand": "unknown",
                    "use_case": "unknown",
                    "must_have": [],
                    "avoid": [],
                    "locale": "unknown",
                    "search_terms": ["wireless"],
                },
            },
            "errors": [],
            "max_products": 3,
        }
    )

    assert result["trace_snapshots"][0]["node_name"] == "merge_context"
    assert result["trace_snapshots"][0]["output"]["effective_query"] == "wireless mouse"
    assert result["trace_snapshots"][1]["input"]["context_used"] is True
    assert result["search_request"].query == "wireless mouse"
    assert len(backend.calls) == 1


def test_new_topic_replaces_previous_context() -> None:
    backend = FakeSearchBackend(_raw_search_response())
    graph = build_graph(
        SearchProductsTool(backend, SEARCH_CONFIG),
        search_config=SEARCH_CONFIG,
        invoke_query_model=_fake_parser(_constraints()),
        invoke_context_model=_new_topic_context_model,
        rag_runner=lambda request, **_: _rag_result(request),
    )

    result = graph.invoke(
        {
            "user_query": "running shoes",
            "previous_context": {
                "active_topic": "办公电脑外设",
                "user_summary": "用户想找办公电脑外设",
                "answer_summary": "上一轮返回了鼠标建议",
                "parsed_constraints": {"search_terms_en": ["office computer accessory"]},
            },
            "errors": [],
            "max_products": 3,
        }
    )

    assert result["trace_snapshots"][0]["output"]["context_status"] == "new_topic_replaced"
    assert result["trace_snapshots"][0]["output"]["context_used"] is False
    assert result["trace_snapshots"][1]["input"]["conversation_context"] == {}


def test_no_result_path_rejects_unverified_answer() -> None:
    graph, backend = _make_graph(_raw_search_response(empty=True), _constraints(), grounded=False)

    result = graph.invoke({"user_query": "不存在的无线鼠标", "errors": [], "max_products": 3})

    assert len(backend.calls) == 1
    assert result["search_response"].total == 0
    assert result["validation_report"]["grounded"] is False
    assert result["next_action"] == "reject"
    assert "不能编造商品建议" in result["final_response"]


def test_rejects_out_of_scope_request() -> None:
    graph, backend = _make_graph(
        _raw_search_response(),
        _constraints(in_scope=False, search_terms=[]),
    )

    result = graph.invoke({"user_query": "今天的天气怎么样", "errors": []})

    assert result["next_action"] == "reject"
    assert "只支持亚马逊商品检索" in result["final_response"]
    assert backend.calls == []


def test_generic_product_intent_asks_for_details_instead_of_rejecting() -> None:
    graph, backend = _make_graph(
        _raw_search_response(),
        _constraints(in_scope=False, search_terms=[]),
    )

    result = graph.invoke({"user_query": "我想要一个", "errors": []})

    assert result["next_action"] == "clarify"
    assert result["retrieval_method"] == "not_run"
    assert "请补充" in result["final_response"]
    assert backend.calls == []


def test_parse_query_rejects_missing_structured_field() -> None:
    payload = json.loads(_constraints())
    payload.pop("search_terms")

    with pytest.raises(QueryParseError) as error:
        parse_query("wireless mouse", invoke_model=_fake_parser(json.dumps(payload)))

    assert error.value.code == "query_parse_missing_field"


def test_state_to_dict_serializes_dataclass_without_to_dict() -> None:
    serialized = state_to_dict({"search_request": SearchRequest(query="mouse", top_k=5)})

    assert serialized["search_request"]["query"] == "mouse"
    assert serialized["search_request"]["top_k"] == 5
