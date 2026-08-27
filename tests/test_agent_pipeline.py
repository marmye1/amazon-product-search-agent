"""重排序、约束、证据和有限改写测试。"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from src.build_context import build_context
from src.check_constraints import check_constraints
from src.check_evidence import check_evidence
from src.generate_recommendation import GenerationMetadata, RAGExecutionResult
from src.graph.answer_node import answer_node
from src.graph.build_graph import build_graph
from src.graph.pipeline_router import validate_pipeline_node
from src.grounding_check import GroundingReport
from src.models import SearchResponse
from src.parse_response import ParseError, parse_recommendation_response
from src.rag_models import EvidenceItem, RecommendationItem, RecommendationResponse
from src.rerank_candidates import rerank_candidates
from src.rewrite_query import rewrite_query
from src.hybrid_models import HybridSearchResponse, HybridSearchResult
from src.agent_models import RerankRequest


CONFIG: Dict[str, Any] = {
    "opensearch": {"index_name": "amazon_products_v1"},
    "search": {
        "default_top_k": 5,
        "max_top_k": 20,
        "field_weights": {"title": 5, "brand": 3, "category": 2, "description": 1, "bullet_points": 1},
    },
    "retrieval": {"bm25_k": 5, "vector_k": 5, "rrf_k": 60},
    "data": {},
}


def _item(
    product_id: str,
    *,
    title: str = "Wireless Mouse",
    brand: str = "Brand A",
    description: str = "Office mouse",
    bullets: List[str] | None = None,
    rank: int = 1,
    category: Optional[str] = "Mice",
) -> HybridSearchResult:
    return HybridSearchResult(
        product_id=product_id,
        title=title,
        brand=brand,
        category=category,
        description=description,
        bullet_points=bullets or [],
        score=1.0,
        matched_fields=["title"],
        source_ref="esci:v0:products:us:%s" % product_id,
        bm25_rank=rank,
        vector_rank=rank,
        rrf_score=0.03,
        retrieval_channels=["bm25", "vector"],
    )


def _hybrid_response(results: List[HybridSearchResult]) -> HybridSearchResponse:
    return HybridSearchResponse(
        query="wireless mouse",
        results=results,
        total=len(results),
        retrieval_method="hybrid_rrf",
        retrieval_channels={item.product_id: ["bm25", "vector"] for item in results},
        fusion_config={"algorithm": "rrf", "rrf_k": 60},
        query_embedding_model="test-model",
    )


def _parser_output(
    *,
    search_terms: List[str] | None = None,
    retrieval_query: str | None = None,
) -> str:
    resolved_terms = search_terms or ["wireless", "mouse"]
    return json.dumps(
        {
            "category": "Mice",
            "category_en": "mice",
            "brand": "Brand A",
            "brand_en": "Brand A",
            "use_case": "unknown",
            "use_case_en": "unknown",
            "must_have": [],
            "must_have_en": [],
            "avoid": [],
            "avoid_en": [],
            "locale": "us",
            "search_terms": resolved_terms,
            "search_terms_en": resolved_terms,
            "retrieval_query": retrieval_query or " ".join(resolved_terms),
            "in_scope": True,
            "needs_clarification": False,
            "clarification_reason": "unknown",
        }
    )


def _valid_rag_result(request: Any, **_: Any) -> RAGExecutionResult:
    result = request.search_response.results[0]
    context = build_context(request.search_response, max_products=request.max_products)
    response = RecommendationResponse(
        answer="可以考虑这款商品。",
        recommendations=[
            RecommendationItem(
                result.product_id,
                result.title,
                "商品标题和字段符合检索条件。",
                ["%s:title" % result.product_id],
            )
        ],
        evidence=[
            EvidenceItem(
                "%s:title" % result.product_id,
                result.product_id,
                "title",
                result.title,
            )
        ],
        limitations=["仅依据商品字段，不包含实时价格和库存。"],
        grounded=True,
    )
    return RAGExecutionResult(
        response=response,
        context=context,
        generation=GenerationMetadata("qwen-local", "http://local.test"),
        grounding=GroundingReport(True),
    )


class _HybridTool:
    is_hybrid = True

    def __init__(self, *, always_empty: bool = False) -> None:
        self.always_empty = always_empty
        self.calls: List[str] = []
        self.product = _item("p-1", title="Wireless Mouse Quiet Click", description="Office mouse")

    def invoke(self, request: Any) -> HybridSearchResponse:
        self.calls.append(request.query)
        if self.always_empty or "wireless" not in request.query.casefold():
            return _hybrid_response([])
        return _hybrid_response([self.product])


def test_rerank_preserves_hybrid_fields_and_adds_explainable_score() -> None:
    response = rerank_candidates(
        RerankRequest(
            user_query="wireless mouse",
            parsed_constraints={"must_have": ["quiet click"]},
            candidates=[
                _item("p-1", title="Wireless Mouse Quiet Click", bullets=["Quiet click"], rank=2),
                _item("p-2", title="Wireless Mouse Basic", rank=1),
            ],
            rerank_top_k=2,
        )
    )

    assert response.results[0].product_id == "p-1"
    assert response.results[0].rerank_score > 0
    assert response.results[0].original_score == 1.0
    assert response.results[0].retrieval_method == "hybrid_rrf"
    assert response.results[0].source_ref.endswith(":p-1")


def test_constraints_exclude_unknown_must_have_and_wrong_brand() -> None:
    results = [
        rerank_candidates(
            RerankRequest("mouse", {}, [_item("p-1", bullets=["Quiet click"]), _item("p-2", brand="Brand B")], 2)
        ).results[0],
        rerank_candidates(
            RerankRequest("mouse", {}, [_item("p-1", bullets=["Quiet click"]), _item("p-2", brand="Brand B")], 2)
        ).results[1],
    ]
    checked = check_constraints(
        {"brand": "Brand A", "locale": "us", "must_have": ["Quiet click"]},
        results,
    )

    assert [item.product_id for item in checked.valid_results] == ["p-1"]
    assert "p-2" in checked.violated_constraints


def test_constraints_use_english_terms_and_allow_missing_category_field() -> None:
    bluetooth = _item("p-1", title="Bluetooth Wireless Mouse", category=None)
    result = rerank_candidates(
        RerankRequest("bluetooth mouse", {"retrieval_query": "bluetooth mouse"}, [bluetooth], 1)
    ).results[0]

    checked = check_constraints(
        {
            "category_en": "mouse",
            "must_have_en": ["support Bluetooth"],
            "must_have": ["支持蓝牙"],
        },
        [result],
    )

    assert [item.product_id for item in checked.valid_results] == ["p-1"]
    assert checked.unknown_constraints == {}


def test_constraints_do_not_treat_unindexed_budget_as_missing_product_fact() -> None:
    result = rerank_candidates(
        RerankRequest("wireless mouse", {}, [_item("p-1")], 1)
    ).results[0]

    checked = check_constraints(
        {"must_have_en": ["budget within $30"]},
        [result],
    )

    assert [item.product_id for item in checked.valid_results] == ["p-1"]
    assert checked.unknown_constraints == {}


def test_agent_validation_canonicalizes_model_shortened_title() -> None:
    candidate = _item(
        "p-1",
        title="Calico Designs Arch Tower Corner Computer Tower Multipurpose Home Office Computer Writing Desk",
    )
    answer = RecommendationResponse(
        answer="可以考虑这款办公电脑桌。",
        recommendations=[
            RecommendationItem(
                "p-1",
                "Calico Designs电脑桌",
                "提供电脑和办公配件空间。",
                ["p-1:title"],
            )
        ],
        evidence=[EvidenceItem("p-1:title", "p-1", "title", candidate.title)],
        limitations=["仅依据商品字段。"],
        grounded=True,
    )

    result = validate_pipeline_node(
        {
            "answer": answer,
            "search_response": _hybrid_response([candidate]),
            "max_products": 1,
        }
    )

    assert result["answer"].recommendations[0].title == candidate.title
    assert result["evidence_check"]["grounded"] is True


def test_evidence_rejects_unsupported_collective_claim() -> None:
    candidates = [
        _item("p-1", title="Wireless Mouse Quiet Click", description="Quiet click"),
        _item("p-2", title="Wireless Mouse Basic", description="Office mouse"),
    ]
    response = RecommendationResponse(
        answer="所有商品都支持 quiet click。",
        recommendations=[
            RecommendationItem("p-1", "Wireless Mouse Quiet Click", "有 quiet click。", ["p-1:title"]),
            RecommendationItem("p-2", "Wireless Mouse Basic", "适合办公。", ["p-2:title"]),
        ],
        evidence=[
            EvidenceItem("p-1:title", "p-1", "title", "Wireless Mouse Quiet Click"),
            EvidenceItem("p-2:title", "p-2", "title", "Wireless Mouse Basic"),
        ],
        limitations=["仅依据商品字段。"],
        grounded=True,
    )
    report = check_evidence(response, candidates, build_context(SearchResponse("mouse", candidates, 2)).blocks)

    assert report.grounded is False
    assert any("集体性断言" in claim for claim in report.unsupported_claims)


def test_evidence_rejects_unsupported_realtime_claim() -> None:
    candidate = _item("p-1")
    response = RecommendationResponse(
        answer="这款商品价格是99元。",
        recommendations=[RecommendationItem("p-1", candidate.title, "适合办公。", ["p-1:title"])],
        evidence=[EvidenceItem("p-1:title", "p-1", "title", candidate.title)],
        limitations=["仅依据商品字段。"],
        grounded=True,
    )
    report = check_evidence(response, [candidate], build_context(SearchResponse("mouse", [candidate], 1)).blocks)

    assert report.grounded is False
    assert report.unsupported_claims == ["价格是99元"]


def test_query_rewrite_preserves_constraints_and_stops_after_two_attempts() -> None:
    constraints = {
        "search_terms": ["wireless", "mouse"],
        "brand": "Brand A",
        "must_have": ["quiet click"],
        "avoid": ["used"],
    }
    first = rewrite_query("generic", constraints, ["no candidates"], 0)
    second = rewrite_query(first.rewritten_query, constraints, ["no candidates"], 1)
    third = rewrite_query(second.rewritten_query, constraints, ["no candidates"], 2)

    assert first.allow_retry is True
    assert second.allow_retry is True
    assert "used" not in first.rewritten_query
    assert "quiet click" in second.rewritten_query
    assert third.allow_retry is False
    assert third.attempt_count == 2


def test_parser_rejects_missing_answer_version_fields() -> None:
    output = {
        "answer": "可以考虑。",
        "recommendations": [],
        "evidence": [],
        "limitations": ["仅依据商品字段。"],
        "grounded": False,
        "retrieval_method": "bm25",
        "answer_version": "v2",
    }
    output.pop("retrieval_method")
    with pytest.raises(ParseError) as error:
        parse_recommendation_response(json.dumps(output))
    assert error.value.code == "missing_output_field"


def test_agent_graph_normal_path_runs_one_agent_and_checks_evidence() -> None:
    tool = _HybridTool()
    graph = build_graph(
        tool,
        search_config=CONFIG,
        invoke_query_model=lambda _: _parser_output(),
        rag_runner=_valid_rag_result,
        use_agent_pipeline=True,
    )

    result = graph.invoke({"user_query": "wireless mouse", "errors": [], "max_products": 3})

    assert result["next_action"] == "finalize"
    assert result["attempt_count"] == 0
    assert result["evidence_check"]["grounded"] is True
    assert result["agent_version"] == "production"
    assert result["retrieval_method"] == "hybrid_rrf"
    assert result["answer"].retrieval_method == "bm25"
    assert result["answer"].answer_version == "v2"
    assert len(tool.calls) == 1
    assert result["trace_snapshots"][0]["node_name"] == "merge_context"
    assert result["trace_snapshots"][1]["node_name"] == "parse_query"
    assert result["trace_snapshots"][1]["input"]["effective_query"] == "wireless mouse"
    assert result["trace_snapshots"][1]["output"]["parsed_constraints"]["search_terms"]
    assert result["trace_snapshots"][-1]["node_name"] == "finalize"


def test_agent_graph_rewrites_once_when_first_query_has_no_candidate() -> None:
    tool = _HybridTool()
    graph = build_graph(
        tool,
        search_config=CONFIG,
        invoke_query_model=lambda _: _parser_output(retrieval_query="generic"),
        rag_runner=_valid_rag_result,
        use_agent_pipeline=True,
    )

    result = graph.invoke({"user_query": "generic", "errors": [], "max_products": 3})

    assert result["next_action"] == "finalize"
    assert result["attempt_count"] == 1
    assert len(result["rewrite_history"]) == 1
    assert len(tool.calls) == 2


def test_agent_no_compliant_candidates_shows_verified_related_categories() -> None:
    tool = _HybridTool(always_empty=True)

    def related_search(request: Any) -> HybridSearchResponse:
        if request.query == "mouse pad":
            return _hybrid_response([_item("pad-1", title="Mouse Pad", description="Desk mouse pad")])
        return _hybrid_response([])

    graph = build_graph(
        tool,
        search_config=CONFIG,
        invoke_query_model=lambda _: _parser_output(),
        rag_runner=_valid_rag_result,
        related_category_search=related_search,
        use_agent_pipeline=True,
    )

    result = graph.invoke({"user_query": "wireless mouse", "errors": [], "max_products": 3})

    assert result["next_action"] == "reject"
    assert result["related_categories"] == ["鼠标垫"]
    assert "检索已执行" in result["final_response"]
    assert "鼠标垫" in result["final_response"]
    assert result["trace_snapshots"][-2]["node_name"] == "category_fallback"


def test_answer_node_uses_grounded_candidate_fallback_when_model_returns_no_items() -> None:
    search_response = _hybrid_response([_item("p-1", title="Wireless Mouse")])

    def empty_rag(request: Any, **_: Any) -> RAGExecutionResult:
        context = build_context(request.search_response, max_products=request.max_products)
        return RAGExecutionResult(
            response=RecommendationResponse(
                answer="检索已执行。",
                recommendations=[],
                evidence=[],
                limitations=["仅依据商品字段。"],
                grounded=False,
            ),
            context=context,
            generation=GenerationMetadata("qwen-local", "http://local.test"),
            grounding=GroundingReport(False, missing_evidence=["模型未返回商品项"]),
        )

    result = answer_node(
        {
            "user_query": "wireless mouse",
            "effective_query": "wireless mouse",
            "search_response": search_response,
            "max_products": 1,
            "errors": [],
        },
        rag_runner=empty_rag,
    )

    assert result["answer"].recommendations[0].product_id == "p-1"
    assert result["answer"].grounded is True
    assert result["validation_report"]["grounded"] is True


def test_agent_graph_rejects_after_two_rewrites() -> None:
    tool = _HybridTool(always_empty=True)
    graph = build_graph(
        tool,
        search_config=CONFIG,
        invoke_query_model=lambda _: _parser_output(),
        rag_runner=_valid_rag_result,
        use_agent_pipeline=True,
    )

    result = graph.invoke({"user_query": "generic", "errors": [], "max_products": 3})

    assert result["next_action"] == "reject"
    assert result["attempt_count"] == 2
    assert len(result["rewrite_history"]) == 2
    assert "检索已执行" in result["final_response"]
