"""搜索请求、结果、OpenSearch body 和错误边界测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import pandas as pd

from src.evaluate_bm25 import evaluate_examples, evaluate_one_query
from src.fake_search_backend import FakeSearchBackend
from src.index_products import _bulk_counts, build_index_body, iter_jsonl_documents
from src.models import ContractError, ProductDocument, SearchRequest
from src.normalize_search_result import normalize_search_response
from src.opensearch_client import BackendError, InsecureRequestWarning, OpenSearchClient
from src.prepare_search_input import export_products_jsonl
from src.search_products import build_search_body, search_products


CONFIG = {
    "opensearch": {"index_name": "amazon_products_v1"},
    "search": {
        "default_top_k": 10,
        "max_top_k": 100,
        "field_weights": {
            "title": 5,
            "brand": 3,
            "category": 2,
            "description": 1,
            "bullet_points": 1,
        },
    },
}


def hit(product_id: str = "p-1") -> dict:
    return {
        "_id": product_id,
        "_score": 2.5,
        "_source": {
            "product_id": product_id,
            "locale": "us",
            "title": "Wireless Mouse",
            "brand": "Brand A",
            "category": None,
            "description": "A mouse",
            "bullet_points": ["Quiet click"],
            "source_ref": "esci:v0:products:us:%s" % product_id,
        },
        "highlight": {"title": ["Wireless <em>Mouse</em>"]},
    }


def test_request_rejects_empty_query_and_top_k_over_limit() -> None:
    with pytest.raises(ContractError) as empty_error:
        SearchRequest.from_mapping({"query": "  "}, default_top_k=10, max_top_k=100)
    assert empty_error.value.code == "invalid_query"

    with pytest.raises(ContractError) as top_k_error:
        SearchRequest.from_mapping({"query": "mouse", "top_k": 101}, default_top_k=10, max_top_k=100)
    assert top_k_error.value.code == "invalid_top_k"


def test_search_body_contains_bm25_fields_and_filters() -> None:
    request = SearchRequest.from_mapping(
        {"query": "mouse", "locale": "us", "brand": "Brand A", "top_k": 5},
        default_top_k=10,
        max_top_k=100,
    )
    body = build_search_body(request, CONFIG)
    multi_match = body["query"]["bool"]["must"][0]["multi_match"]
    assert multi_match["query"] == "mouse"
    assert "title^5" in multi_match["fields"]
    assert {"term": {"locale": "us"}} in body["query"]["bool"]["filter"]
    assert {"term": {"brand.keyword": "Brand A"}} in body["query"]["bool"]["filter"]
    assert body["size"] == 5


def test_search_response_is_stable_and_no_result_has_warning() -> None:
    request = SearchRequest.from_mapping({"query": "mouse"}, default_top_k=10, max_top_k=100)
    backend = FakeSearchBackend({"hits": {"total": {"value": 1, "relation": "eq"}, "hits": [hit()]}})
    response = search_products(request, backend, CONFIG)
    assert response.search_backend == "opensearch"
    assert response.retrieval_method == "bm25"
    assert response.results[0].product_id == "p-1"
    assert response.results[0].matched_fields == ["title"]
    assert backend.calls[0]["index_name"] == "amazon_products_v1"

    empty = normalize_search_response("missing", {"hits": {"total": 0, "hits": []}})
    assert empty.results == []
    assert empty.warnings == ["no_results"]


def test_duplicate_result_is_rejected() -> None:
    with pytest.raises(ContractError) as error:
        normalize_search_response(
            "mouse",
            {"hits": {"total": 2, "hits": [hit("p-1"), hit("p-1")]}} ,
        )
    assert error.value.code == "duplicate_result"


def test_index_mapping_and_bulk_counts() -> None:
    mapping = build_index_body()
    assert mapping["mappings"]["properties"]["product_id"]["type"] == "keyword"
    assert mapping["mappings"]["properties"]["title"]["type"] == "text"
    docs = [ProductDocument(product_id="p-1", locale="us", title="Mouse")]
    result = _bulk_counts({"items": [{"index": {"status": 201}}]}, docs)
    assert result == {"successful": 1, "failed": 0, "failed_ids": []}
    failed = _bulk_counts({"items": [{"index": {"status": 400, "error": {"type": "bad"}}}]}, docs)
    assert failed == {"successful": 0, "failed": 1, "failed_ids": ["p-1"]}


def test_jsonl_product_contract_rejects_missing_title(tmp_path: Path) -> None:
    path = tmp_path / "products.jsonl"
    path.write_text(json.dumps({"product_id": "p-1", "locale": "us"}) + "\n", encoding="utf-8")
    with pytest.raises(ContractError) as error:
        list(iter_jsonl_documents(path))
    assert error.value.code == "missing_title"


def test_backend_timeout_is_explicit() -> None:
    class TimeoutSession:
        def request(self, *args, **kwargs):
            import requests

            raise requests.Timeout("timeout")

    client = OpenSearchClient("http://localhost:9200", session=TimeoutSession())
    with pytest.raises(BackendError) as error:
        client.search("amazon_products_v1", {"query": {}})
    assert error.value.code == "backend_timeout"


def test_backend_auth_error_is_explicit() -> None:
    class AuthSession:
        def request(self, *args, **kwargs):
            import requests

            response = requests.Response()
            response.status_code = 401
            response._content = b"Unauthorized"
            response.url = "https://localhost:9200/"
            return response

    client = OpenSearchClient("https://localhost:9200", verify_ssl=False, session=AuthSession())
    with pytest.raises(BackendError) as error:
        client.search("amazon_products_v1", {"query": {}})
    assert error.value.code == "backend_auth_error"


def test_evaluation_metrics_and_parquet_to_jsonl_adapter(tmp_path: Path) -> None:
    response = normalize_search_response(
        "mouse",
        {"hits": {"total": 2, "hits": [hit("p-1"), hit("p-2")] }},
    )
    metrics = evaluate_one_query(
        {"p-1": "E", "p-2": "I", "p-3": "S"},
        response,
        top_k=2,
        positive_labels={"E", "S"},
        grades={"E": 3, "S": 2, "C": 1, "I": 0},
    )
    assert metrics["recall"] == 0.5
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg"] is not None

    frame = pd.DataFrame(
        {
            "product_id": ["p-1"],
            "product_locale": ["US"],
            "product_title": ["Wireless Mouse"],
            "product_description": ["A mouse"],
            "product_bullet_point": ["Quiet click"],
            "product_brand": ["Brand A"],
            "product_color": ["Black"],
        }
    )
    parquet_path = tmp_path / "products.parquet"
    jsonl_path = tmp_path / "products.jsonl"
    frame.to_parquet(parquet_path, engine="pyarrow", index=False)
    summary = export_products_jsonl(parquet_path, jsonl_path)
    assert summary["rows"] == 1
    exported = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
    assert exported["locale"] == "us"
    assert exported["title"] == "Wireless Mouse"
    assert exported["bullet_points"] == ["Quiet click"]


AUTH_CONFIG = {
    "opensearch": {
        "base_url": "https://localhost:9200",
        "username_env": "TEST_OPENSEARCH_USERNAME",
        "password_env": "TEST_OPENSEARCH_PASSWORD",
        "verify_ssl": False,
    }
}


def _clear_auth_credentials(monkeypatch) -> None:
    monkeypatch.delenv("TEST_OPENSEARCH_USERNAME", raising=False)
    monkeypatch.delenv("TEST_OPENSEARCH_PASSWORD", raising=False)


def test_from_config_uses_environment_credentials_without_prompt(monkeypatch) -> None:
    monkeypatch.setenv("TEST_OPENSEARCH_USERNAME", "env-user")
    monkeypatch.setenv("TEST_OPENSEARCH_PASSWORD", "env-password")

    def fail_if_prompted(_: str) -> str:
        raise AssertionError("不应在凭证完整时提示输入")

    client = OpenSearchClient.from_config(
        AUTH_CONFIG,
        prompt_for_missing=True,
        input_func=fail_if_prompted,
        password_func=fail_if_prompted,
    )

    assert client.auth == ("env-user", "env-password")


def test_from_config_prompts_only_for_missing_password(monkeypatch) -> None:
    monkeypatch.setenv("TEST_OPENSEARCH_USERNAME", "env-user")
    monkeypatch.delenv("TEST_OPENSEARCH_PASSWORD", raising=False)
    prompts = []

    def fail_if_username_prompted(_: str) -> str:
        raise AssertionError("已有用户名时不应再次提示用户名")

    def read_password(prompt: str) -> str:
        prompts.append(prompt)
        return "typed-password"

    client = OpenSearchClient.from_config(
        AUTH_CONFIG,
        prompt_for_missing=True,
        input_func=fail_if_username_prompted,
        password_func=read_password,
    )

    assert client.auth == ("env-user", "typed-password")
    assert prompts == ["请输入 OpenSearch 密码: "]


def test_from_config_rejects_missing_credentials_without_tty(monkeypatch) -> None:
    _clear_auth_credentials(monkeypatch)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    with pytest.raises(BackendError) as error:
        OpenSearchClient.from_config(AUTH_CONFIG, prompt_for_missing=True)

    assert error.value.code == "missing_credentials"
    assert "TEST_OPENSEARCH_USERNAME" in error.value.message
    assert "TEST_OPENSEARCH_PASSWORD" in error.value.message


def test_from_config_rejects_blank_interactive_credentials(monkeypatch) -> None:
    _clear_auth_credentials(monkeypatch)

    with pytest.raises(BackendError) as error:
        OpenSearchClient.from_config(
            AUTH_CONFIG,
            prompt_for_missing=True,
            input_func=lambda _: "  ",
            password_func=lambda _: "",
        )

    assert error.value.code == "missing_credentials"


def test_unverified_https_disables_repeated_warning(monkeypatch) -> None:
    categories = []
    monkeypatch.setattr(
        "src.opensearch_client.disable_warnings",
        lambda category: categories.append(category),
    )

    OpenSearchClient("https://localhost:9200", verify_ssl=False)

    assert categories == [InsecureRequestWarning]
