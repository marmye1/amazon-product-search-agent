"""提示词、解析、证据检查和固定 RAG 链测试。"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.build_context import build_context
from src.generate_recommendation import (
    GenerationMetadata,
    GenerationResult,
    LocalQwenConfig,
    invoke_local_qwen,
    run_rag,
)
from src.grounding_check import check_grounding
from src.models import ContractError, SearchResponse, SearchResult
from src.parse_response import ParseError, parse_recommendation_response
from src.prompts.recommendation_prompt import build_recommendation_messages
from src.rag_models import ContextBlock, RAGRequest, RecommendationItem, RecommendationResponse, EvidenceItem


def _search_response(*, empty: bool = False) -> SearchResponse:
    if empty:
        return SearchResponse(query="wireless mouse", results=[], total=0, warnings=["no_results"])
    result = SearchResult(
        product_id="p-1",
        title="Wireless Mouse",
        brand="Brand A",
        category=None,
        description="A quiet mouse for office use",
        bullet_points=["Quiet click"],
        score=2.5,
        matched_fields=["title"],
        source_ref="esci:v0:products:us:p-1",
    )
    return SearchResponse(query="wireless mouse", results=[result], total=1)


def _context() -> List[ContextBlock]:
    return [
        ContextBlock("p-1:title", "p-1", "title", "Wireless Mouse", 1),
        ContextBlock("p-1:brand", "p-1", "brand", "Brand A", 1),
    ]


def _valid_model_output() -> str:
    return (
        '{"answer":"可以考虑这款商品。",'
        '"recommendations":[{"product_id":"p-1","title":"Wireless Mouse",'
        '"reason":"标题明确包含无线鼠标。",'
        '"evidence_source_ids":["p-1:title"]}],'
        '"evidence":[{"source_id":"p-1:title","product_id":"p-1",'
        '"field_name":"title","quoted_or_paraphrased_fact":"Wireless Mouse"}],'
        '"limitations":["仅使用检索到的商品字段。"],'
        '"grounded":false,"retrieval_method":"bm25","answer_version":"v2"}'
    )


def _metadata() -> GenerationMetadata:
    return GenerationMetadata(model="qwen-local", endpoint="http://127.0.0.1:1234/v1/chat/completions")


def test_prompt_contains_context_source_ids_and_grounding_rules() -> None:
    messages = build_recommendation_messages("我想买无线鼠标", _context())

    assert [message["role"] for message in messages] == ["system", "user"]
    combined = "\n".join(message["content"] for message in messages)
    assert "p-1:title" in combined
    assert "只能使用" in combined
    assert "实时价格" in combined


def test_parser_accepts_structured_output_and_rejects_unknown_product() -> None:
    response = parse_recommendation_response(
        _valid_model_output(),
        candidate_product_ids={"p-1"},
        allowed_source_ids={"p-1:title"},
    )
    assert response.recommendations[0].product_id == "p-1"
    assert response.evidence[0].source_id == "p-1:title"

    invalid = _valid_model_output().replace('"p-1"', '"p-999"', 1)
    with pytest.raises(ParseError) as error:
        parse_recommendation_response(
            invalid,
            candidate_product_ids={"p-1"},
            allowed_source_ids={"p-1:title"},
        )
    assert error.value.code == "unknown_reference"


def test_grounding_rejects_missing_evidence_and_unsupported_price_claim() -> None:
    response = RecommendationResponse(
        answer="这款商品价格是99元。",
        recommendations=[RecommendationItem("p-1", "Wireless Mouse", "适合办公。", [])],
        evidence=[],
        limitations=["仅使用检索到的字段。"],
        grounded=False,
    )
    report = check_grounding(response, _context())

    assert report.grounded is False
    assert "p-1 缺少 evidence_source_ids" in report.missing_evidence
    assert report.unsupported_claims == ["价格是99元"]


def test_run_rag_returns_grounded_success_with_injected_model() -> None:
    calls: List[List[Dict[str, str]]] = []

    def fake_model(messages: List[Dict[str, str]]) -> GenerationResult:
        calls.append(messages)
        return GenerationResult(raw_output=_valid_model_output(), metadata=_metadata())

    result = run_rag(
        RAGRequest("我想买无线鼠标", _search_response()),
        invoke_model=fake_model,
    )

    assert len(calls) == 1
    assert result.response.grounded is True
    assert result.response.recommendations[0].product_id == "p-1"
    assert result.grounding.grounded is True


def test_run_rag_skips_model_when_search_has_no_results() -> None:
    called = False

    def fake_model(messages: List[Dict[str, str]]) -> GenerationResult:
        nonlocal called
        called = True
        return GenerationResult(raw_output=_valid_model_output(), metadata=_metadata())

    result = run_rag(
        RAGRequest("找一个不存在的商品", _search_response(empty=True)),
        invoke_model=fake_model,
    )

    assert called is False
    assert result.generation.skipped is True
    assert result.response.recommendations == []
    assert result.response.grounded is False


def test_run_rag_degrades_on_invalid_model_output() -> None:
    def fake_model(messages: List[Dict[str, str]]) -> GenerationResult:
        return GenerationResult(raw_output="不是 JSON", metadata=_metadata())

    result = run_rag(RAGRequest("无线鼠标", _search_response()), invoke_model=fake_model)

    assert result.response.recommendations == []
    assert result.response.grounded is False
    assert result.generation.error is not None
    assert result.generation.error["code"] == "invalid_json"


class _FakeResponse:
    status_code = 200
    text = ""

    def json(self) -> Dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {"content": _valid_model_output()},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }


class _FakeSession:
    trust_env = True

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return _FakeResponse()


def test_local_qwen_adapter_uses_openai_compatible_payload() -> None:
    session = _FakeSession()
    result = invoke_local_qwen(
        [{"role": "user", "content": "test"}],
        LocalQwenConfig(timeout_seconds=3),
        session=session,  # type: ignore[arg-type]
    )

    assert result.raw_output.startswith("{")
    assert result.metadata.model == "qwen-local"
    assert result.metadata.total_tokens == 30
    assert session.calls[0]["url"].endswith("/chat/completions")
    assert session.calls[0]["json"]["response_format"]["type"] == "json_schema"
    assert session.calls[0]["json"]["response_format"]["json_schema"]["name"] == "recommendation_response"


def search_result(product_id: str = "p-1", *, description: str | None = "A mouse") -> SearchResult:
    return SearchResult(
        product_id=product_id,
        title="Wireless Mouse",
        brand="Brand A",
        category=None,
        description=description,
        bullet_points=["Quiet click", "USB receiver"],
        score=2.5,
        matched_fields=["title"],
        source_ref="esci:v0:products:us:%s" % product_id,
    )


def search_response(*results: SearchResult) -> SearchResponse:
    return SearchResponse(query="wireless mouse", results=list(results), total=len(results))


def test_rag_request_wraps_search_response_contract() -> None:
    response = search_response(search_result())
    request = RAGRequest(user_query="我想要无线鼠标", search_response=response, max_products=3)

    payload = request.to_dict()
    assert payload["user_query"] == "我想要无线鼠标"
    assert payload["answer_language"] == "zh-CN"
    assert payload["max_products"] == 3
    assert payload["search_response"]["results"][0]["product_id"] == "p-1"
    assert payload["search_response"]["results"][0]["source_ref"] == "esci:v0:products:us:p-1"


def test_rag_request_rejects_result_without_required_source_ref() -> None:
    result = search_result()
    invalid_result = SearchResult(
        product_id=result.product_id,
        title=result.title,
        brand=result.brand,
        category=result.category,
        description=result.description,
        bullet_points=result.bullet_points,
        score=result.score,
        matched_fields=result.matched_fields,
        source_ref="",
    )

    with pytest.raises(ContractError) as error:
        RAGRequest(user_query="鼠标", search_response=search_response(invalid_result))
    assert error.value.code == "invalid_rag_request"


def test_response_models_have_required_fields() -> None:
    context = ContextBlock(
        source_id="p-1:title",
        product_id="p-1",
        field_name="title",
        text="Wireless Mouse",
        rank=1,
    )
    recommendation = RecommendationItem(
        product_id="p-1",
        title="Wireless Mouse",
        reason="标题明确包含 wireless mouse。",
        evidence_source_ids=[context.source_id],
    )
    evidence = EvidenceItem(
        source_id=context.source_id,
        product_id="p-1",
        field_name="title",
        quoted_or_paraphrased_fact="Wireless Mouse",
    )
    response = RecommendationResponse(
        answer="可以考虑这款商品。",
        recommendations=[recommendation],
        evidence=[evidence],
        limitations=["仅依据商品字段，不包含实时价格和库存。"],
        grounded=True,
    )

    payload = response.to_dict()
    assert payload["recommendations"][0]["evidence_source_ids"] == ["p-1:title"]
    assert payload["evidence"][0]["field_name"] == "title"
    assert payload["retrieval_method"] == "bm25"
    assert payload["answer_version"] == "v2"


def test_build_context_limits_products_and_keeps_source_trace() -> None:
    response = search_response(search_result("p-1"), search_result("p-2"))
    built = build_context(response, max_products=1)

    assert {block.product_id for block in built.blocks} == {"p-1"}
    assert all(block.rank == 1 for block in built.blocks)
    assert "p-1:title" in {block.source_id for block in built.blocks}
    assert all(block.text for block in built.blocks)
    assert built.truncation_stats == {}


def test_build_context_truncates_deterministically_and_records_stats() -> None:
    long_description = "x" * 20
    built = build_context(
        search_response(search_result(description=long_description)),
        max_products=1,
        field_length_limits={"description": 10},
    )

    description_blocks = [block for block in built.blocks if block.field_name == "description"]
    assert len(description_blocks) == 1
    assert description_blocks[0].text == "x" * 10
    assert built.truncation_stats == {"description": 1}


def test_build_context_empty_search_response_returns_empty_context() -> None:
    built = build_context(search_response(), max_products=5)

    assert built.blocks == []
    assert built.truncation_stats == {}
