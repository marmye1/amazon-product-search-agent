"""API、服务层和稳定错误契约测试。"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.generate_recommendation import LocalQwenConfig
from src.models import SearchRequest as DomainSearchRequest, SearchResponse as DomainSearchResponse, SearchResult
from src.opensearch_client import BackendError
from src.rag_models import EvidenceItem, RecommendationItem, RecommendationResponse
from src.service.agent_service import AgentRuntime, AgentService
from src.hybrid_models import HybridSearchResponse, HybridSearchResult


def _result(product_id: str = "p-1") -> HybridSearchResult:
    return HybridSearchResult(
        product_id=product_id,
        title="Wireless Mouse",
        brand="Brand A",
        category="Computer Mice",
        description="An office mouse",
        bullet_points=["Quiet click"],
        score=1.0,
        matched_fields=["title"],
        source_ref="esci:v0:products:us:%s" % product_id,
        bm25_rank=1,
        vector_rank=1,
        rrf_score=0.03,
        retrieval_channels=["bm25", "vector"],
    )


def _hybrid_response() -> HybridSearchResponse:
    return HybridSearchResponse(
        query="wireless mouse",
        results=[_result()],
        total=1,
        retrieval_method="hybrid_rrf",
        retrieval_channels={"p-1": ["bm25", "vector"]},
        fusion_config={"algorithm": "rrf"},
        query_embedding_model="embedding-test",
    )


def _fake_memory_model(messages: list[dict[str, str]]) -> str:
    payload = json.loads(messages[-1]["content"])
    constraints = payload.get("parsed_constraints") or {}
    terms = constraints.get("search_terms", [])
    return json.dumps(
        {
            "active_topic": " ".join(terms) or "unknown",
            "user_summary": payload.get("user_query", "unknown"),
            "answer_summary": payload.get("assistant_answer", "unknown"),
            "unresolved_question": payload.get("assistant_answer", "")
            if payload.get("next_action") == "clarify"
            else "",
            "pending_clarification": payload.get("next_action") == "clarify",
            "mentioned_products": [],
        },
        ensure_ascii=False,
    )


class _FakeService:
    def __init__(self) -> None:
        self.last_trace_id: Optional[str] = None

    def search(self, request: DomainSearchRequest, *, trace_id: str) -> DomainSearchResponse:
        self.last_trace_id = trace_id
        return _hybrid_response()

    def chat(
        self,
        *,
        message: str,
        locale: Optional[str],
        top_k: Optional[int],
        trace_id: str,
        session_id: Optional[str] = None,
    ):
        self.last_trace_id = trace_id
        return type(
            "ChatResult",
            (),
            {
                "answer": RecommendationResponse(
                    answer="可以考虑这款商品。",
                    recommendations=[
                        RecommendationItem(
                            "p-1",
                            "Wireless Mouse",
                            "标题符合无线鼠标。",
                            ["esci:v0:products:us:p-1:title"],
                        )
                    ],
                    evidence=[
                        EvidenceItem(
                            "esci:v0:products:us:p-1:title",
                            "p-1",
                            "title",
                            "Wireless Mouse",
                        )
                    ],
                    limitations=["仅依据商品字段。"],
                    grounded=True,
                ),
                "final_response": "可以考虑这款商品。",
                "agent_version": "production",
                "retrieval_method": "hybrid_rrf",
                "errors": [],
                "trace_snapshots": [
                    {
                        "sequence": 1,
                        "node_name": "parse_query",
                        "display_name": "需求解析",
                        "input_description": "用户自然语言和可选市场条件",
                        "output_description": "结构化商品约束和检索词",
                        "input_format": "JSON",
                        "output_format": "JSON state patch",
                        "input_keys": ["user_query"],
                        "output_keys": ["parsed_constraints"],
                        "input": {"user_query": "wireless mouse"},
                        "output": {"parsed_constraints": {"search_terms": ["wireless mouse"]}},
                        "duration_ms": 1,
                        "status": "completed",
                    }
                ],
            },
        )()

    def health(self) -> Dict[str, str]:
        return {
            "status": "ok",
            "api": "ok",
            "opensearch": "ok",
            "llm": "ready",
            "index_version": "amazon_products_v4",
            "app_version": "production",
        }


class _FailingService(_FakeService):
    def search(self, request: DomainSearchRequest, *, trace_id: str) -> DomainSearchResponse:
        raise BackendError("backend_timeout", "OpenSearch 请求超时")


def test_search_chat_health_and_trace_id() -> None:
    service = _FakeService()
    app = create_app(service=service)

    with TestClient(app) as client:
        root = client.get("/")
        ui = client.get("/ui/")
        css = client.get("/ui/assets/styles.css")
        favicon = client.get("/favicon.ico")
        health = client.get("/health")
        search = client.post(
            "/v1/search",
            headers={"X-Trace-ID": "trace-search"},
            json={"query": "wireless mouse", "top_k": 1},
        )
        chat = client.post(
            "/v1/chat",
            headers={"X-Trace-ID": "trace-chat"},
            json={"message": "wireless mouse", "locale": "us", "top_k": 1},
        )

    assert root.status_code == 200
    assert "Agent" in root.text
    assert "聊天界面" in root.text
    assert "流程图" in root.text
    assert 'class="brand-copy"' not in root.text
    assert 'class="brand-title"' not in root.text
    assert "不展示模型隐藏思维链" in root.text
    assert ui.status_code == 200
    assert "完整运行流程" not in ui.text
    assert "数据如何穿过每一层" in ui.text
    assert css.status_code == 200
    assert favicon.status_code == 204
    assert health.status_code == 200
    assert health.json()["app_version"] == "production"
    assert search.status_code == 200
    assert search.json()["trace_id"] == "trace-search"
    assert search.json()["retrieval_method"] == "hybrid_rrf"
    assert search.json()["results"][0]["product_id"] == "p-1"
    assert chat.status_code == 200
    assert chat.json()["trace_id"] == "trace-chat"
    assert chat.json()["agent_version"] == "production"
    assert chat.json()["retrieval_method"] == "hybrid_rrf"
    assert chat.json()["recommendations"][0]["product_id"] == "p-1"
    assert chat.json()["execution_trace"][0]["node_name"] == "parse_query"
    assert service.last_trace_id == "trace-chat"


def test_invalid_request_uses_stable_error_contract() -> None:
    app = create_app(service=_FakeService())

    with TestClient(app) as client:
        response = client.post("/v1/search", json={"query": "", "top_k": 0})

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "request_validation_error"
    assert payload["error"]["trace_id"]


def test_backend_error_uses_stable_error_contract() -> None:
    app = create_app(service=_FailingService())

    with TestClient(app) as client:
        response = client.post("/v1/search", json={"query": "wireless mouse"})

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "backend_timeout",
            "message": "OpenSearch 请求超时",
            "trace_id": response.json()["error"]["trace_id"],
        }
    }


class _FakeSearchTool:
    is_hybrid = True

    def invoke(self, request: Any) -> HybridSearchResponse:
        assert request.query == "wireless mouse"
        return _hybrid_response()


class _FakeGraph:
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        assert state["request_locale"] == "us"
        assert state["top_k"] == 2
        return {
            "final_response": "可以考虑这款商品。",
            "answer": RecommendationResponse(
                answer="可以考虑这款商品。",
                recommendations=[],
                evidence=[],
                limitations=["仅依据商品字段。"],
                grounded=True,
            ),
            "agent_version": "production",
            "retrieval_method": "hybrid_rrf",
            "trace_nodes": ["parse_query", "finalize"],
            "trace_snapshots": [
                {
                    "sequence": 1,
                    "node_name": "parse_query",
                    "display_name": "需求解析",
                    "input_description": "用户自然语言和可选市场条件",
                    "output_description": "结构化商品约束和检索词",
                    "input_format": "JSON",
                    "output_format": "JSON state patch",
                    "input_keys": ["user_query"],
                    "output_keys": ["parsed_constraints"],
                    "input": {"user_query": state["user_query"]},
                    "output": {"parsed_constraints": {"search_terms": ["wireless mouse"]}},
                    "duration_ms": 1,
                    "status": "completed",
                }
            ],
            "attempt_count": 0,
            "errors": [],
        }


def test_agent_service_passes_request_state_and_returns_trace_summary() -> None:
    runtime = AgentRuntime(
        search_config={},
        client=object(),
        search_tool=_FakeSearchTool(),
        graph=_FakeGraph(),
        llm_config=LocalQwenConfig(),
        app_version="production",
        agent_version="production",
        retrieval_method="hybrid_rrf",
        index_name="amazon_products_v4",
        embedding_model_id="embedding-test",
    )
    service = AgentService(runtime, default_chat_top_k=5)

    response = service.search(
        DomainSearchRequest(query="wireless mouse", top_k=1),
        trace_id="trace-search",
    )
    execution = service.chat(
        message="wireless mouse",
        locale="us",
        top_k=2,
        trace_id="trace-chat",
    )

    assert response.retrieval_method == "hybrid_rrf"
    assert execution.agent_version == "production"
    assert execution.retrieval_method == "hybrid_rrf"
    assert execution.trace_nodes == ["parse_query", "finalize"]
    assert execution.trace_snapshots[0]["input"]["user_query"] == "wireless mouse"
    assert execution.attempt_count == 0


class _ContextGraph:
    def __init__(self) -> None:
        self.states = []

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.states.append(state)
        has_previous = bool(state.get("previous_context"))
        return {
            "effective_query": "wireless mouse display" if has_previous else state["user_query"],
            "parsed_constraints": {
                "search_terms": ["wireless mouse", "display"],
                "needs_clarification": not has_previous,
                "retrieval_eligible": True,
            },
            "next_action": "search" if has_previous else "clarify",
            "final_response": "可以继续检索。",
            "agent_version": "production",
            "retrieval_method": "hybrid_rrf" if has_previous else "not_run",
            "trace_nodes": [],
            "trace_snapshots": [],
            "attempt_count": 0,
            "errors": [],
        }


def test_agent_service_merges_pending_context_for_the_next_turn() -> None:
    graph = _ContextGraph()
    runtime = AgentRuntime(
        search_config={},
        client=object(),
        search_tool=_FakeSearchTool(),
        graph=graph,
        llm_config=LocalQwenConfig(),
        app_version="production",
        agent_version="production",
        retrieval_method="hybrid_rrf",
        index_name="amazon_products_v4",
        embedding_model_id="embedding-test",
    )
    service = AgentService(runtime, invoke_memory_model=_fake_memory_model)

    service.chat(
        message="wireless mouse",
        locale="us",
        top_k=2,
        trace_id="trace-context-1",
        session_id="session-demo",
    )
    execution = service.chat(
        message="display",
        locale="us",
        top_k=2,
        trace_id="trace-context-2",
        session_id="session-demo",
    )

    assert graph.states[0]["previous_context"] == {}
    assert graph.states[1]["previous_context"]["pending_clarification"] is True
    assert graph.states[1]["previous_context"]["parsed_constraints"]["search_terms"] == [
        "wireless mouse",
        "display",
    ]
    assert service.context_store.get("session-demo") is not None
    assert execution.trace_snapshots[-1]["node_name"] == "memory_update"
    assert execution.trace_snapshots[-1]["output"]["stored"] is True


def test_agent_service_keeps_context_after_a_search() -> None:
    class _SearchDespiteClarificationGraph(_ContextGraph):
        def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
            self.states.append(state)
            return {
                "parsed_constraints": {
                    "search_terms": ["office", "computer"],
                    "needs_clarification": True,
                    "retrieval_eligible": True,
                },
                "next_action": "search",
                "final_response": "已执行检索。",
                "agent_version": "production",
                "retrieval_method": "hybrid_rrf",
                "trace_nodes": [],
                "trace_snapshots": [],
                "attempt_count": 0,
                "errors": [],
            }

    graph = _SearchDespiteClarificationGraph()
    runtime = AgentRuntime(
        search_config={},
        client=object(),
        search_tool=_FakeSearchTool(),
        graph=graph,
        llm_config=LocalQwenConfig(),
        app_version="production",
        agent_version="production",
        retrieval_method="hybrid_rrf",
        index_name="amazon_products_v4",
        embedding_model_id="embedding-test",
    )
    service = AgentService(runtime, invoke_memory_model=_fake_memory_model)

    service.chat(
        message="办公的、连接电脑的",
        locale="us",
        top_k=2,
        trace_id="trace-context-search",
        session_id="session-search",
    )

    stored = service.context_store.get("session-search")
    assert stored is not None
    assert stored["parsed_constraints"]["search_terms"] == ["office", "computer"]
