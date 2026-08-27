"""向量、BM25、RRF 和失败降级契约测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.embedding_client import EmbeddingClient, EmbeddingConfig, EmbeddingError
from src.fuse_rrf import fuse_rrf
from src.hybrid_search_tool import HybridSearchTool
from src.index_embeddings import build_hybrid_index_body, index_embeddings, product_embedding_text
from src.models import ProductDocument, SearchResult
from src.fake_search_backend import FakeSearchBackend
from src.retrieve_bm25 import retrieve_bm25
from src.retrieve_vector import build_vector_search_body
from src.hybrid_models import HybridSearchRequest
from src.search_acceptance import validate_hybrid_mapping


CONFIG: Dict[str, Any] = {
    "opensearch": {"index_name": "amazon_products_v1"},
    "search": {
        "default_top_k": 5,
        "max_top_k": 100,
        "field_weights": {"title": 5, "brand": 3, "category": 2, "description": 1, "bullet_points": 1},
    },
    "retrieval": {
        "hybrid_index_name": "amazon_products_v4",
        "bm25_k": 10,
        "vector_k": 10,
        "rrf_k": 60,
        "fallback_on_vector_error": True,
    },
    "embedding": {
        "base_url": "http://127.0.0.1:1234/v1",
        "model": "text-embedding-nomic-embed-text-v1.5",
        "dimension": 3,
    },
}


def _result(product_id: str) -> SearchResult:
    return SearchResult(
        product_id=product_id,
        title="Wireless Mouse %s" % product_id,
        brand="Brand A",
        category=None,
        description="A mouse",
        bullet_points=["Quiet click"],
        score=1.0,
        matched_fields=["title"],
        source_ref="esci:v0:products:us:%s" % product_id,
    )


class _EmbeddingResponse:
    status_code = 200
    text = ""

    def __init__(self, count: int) -> None:
        self.count = count

    def json(self) -> Dict[str, Any]:
        return {
            "model": "text-embedding-nomic-embed-text-v1.5",
            "data": [
                {"index": index, "embedding": [0.1 + index, 0.2 + index, 0.3 + index]}
                for index in reversed(range(self.count))
            ],
        }


class _EmbeddingSession:
    trust_env = True

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _EmbeddingResponse:
        self.calls.append({"url": url, **kwargs})
        return _EmbeddingResponse(len(kwargs["json"]["input"]))


def test_embedding_client_uses_nomic_task_prefixes_and_orders_vectors() -> None:
    session = _EmbeddingSession()
    client = EmbeddingClient(
        EmbeddingConfig(
            model="text-embedding-nomic-embed-text-v1.5",
            dimension=3,
        ),
        session=session,  # type: ignore[arg-type]
    )

    documents = client.embed_documents(["product title", "brand text"])
    query = client.embed_query("wireless mouse")

    assert documents[0] == [0.1, 0.2, 0.3]
    assert documents[1] == [1.1, 1.2, 1.3]
    assert query == [0.1, 0.2, 0.3]
    assert session.calls[0]["json"]["input"][0].startswith("search_document: ")
    assert session.calls[1]["json"]["input"][0].startswith("search_query: ")


def test_hybrid_index_mapping_contains_knn_vector_and_dimension() -> None:
    body = build_hybrid_index_body(768)

    assert body["settings"]["index"]["knn"] is True
    assert body["mappings"]["properties"]["embedding"] == {
        "type": "knn_vector",
        "dimension": 768,
        "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"},
    }
    assert body["mappings"]["properties"]["embedding_model_id"]["type"] == "keyword"


def test_product_embedding_text_is_stable_and_bounded() -> None:
    document = ProductDocument(
        product_id="p-1",
        locale="us",
        title="Wireless Mouse",
        brand="Brand A",
        description="x" * 100,
        bullet_points=["Quiet click"],
    )

    text = product_embedding_text(document, max_chars=40)

    assert text.startswith("title: Wireless Mouse")
    assert len(text) <= 40


def test_vector_search_body_contains_knn_and_hard_filters() -> None:
    request = HybridSearchRequest(
        query="wireless mouse",
        locale="us",
        brand="Brand A",
        vector_k=7,
    )

    body = build_vector_search_body(request, [0.1, 0.2, 0.3])
    clause = body["query"]["knn"]["embedding"]

    assert body["size"] == 7
    assert clause["k"] == 7
    assert clause["vector"] == [0.1, 0.2, 0.3]
    assert {"term": {"locale": "us"}} in clause["filter"]["bool"]["filter"]
    assert {"term": {"brand.keyword": "Brand A"}} in clause["filter"]["bool"]["filter"]


def test_hybrid_bm25_retrieval_uses_hybrid_index() -> None:
    raw_response = {
        "hits": {
            "total": {"value": 1, "relation": "eq"},
            "hits": [
                {
                    "_score": 2.0,
                    "_source": {
                        "product_id": "p-1",
                        "locale": "us",
                        "title": "Wireless Mouse",
                        "source_ref": "esci:v0:products:us:p-1",
                    },
                    "highlight": {"title": ["Wireless Mouse"]},
                }
            ],
        }
    }
    backend = FakeSearchBackend(raw_response)

    retrieve_bm25(HybridSearchRequest(query="wireless mouse", bm25_k=3), backend, CONFIG)

    assert backend.calls[0]["index_name"] == "amazon_products_v4"


def test_rrf_deduplicates_and_records_channels() -> None:
    results, channels = fuse_rrf(
        [_result("p-1"), _result("p-2")],
        [_result("p-2"), _result("p-3")],
        top_k=3,
        rrf_k=60,
    )

    assert [item.product_id for item in results] == ["p-2", "p-1", "p-3"]
    assert channels["p-2"] == ["bm25", "vector"]
    assert results[0].bm25_rank == 2
    assert results[0].vector_rank == 1
    assert results[0].rrf_score == results[0].score


class _UnavailableEmbedding:
    model = "text-embedding-nomic-embed-text-v1.5"

    def embed_query(self, _: str) -> List[float]:
        raise EmbeddingError("embedding_unavailable", "本地 Embedding 不可用")


def test_hybrid_tool_falls_back_to_bm25_when_vector_unavailable() -> None:
    raw_response = {
        "hits": {
            "total": {"value": 1, "relation": "eq"},
            "hits": [
                {
                    "_score": 2.0,
                    "_source": {
                        "product_id": "p-1",
                        "locale": "us",
                        "title": "Wireless Mouse",
                        "brand": "Brand A",
                        "source_ref": "esci:v0:products:us:p-1",
                    },
                    "highlight": {"title": ["Wireless Mouse"]},
                }
            ],
        }
    }
    tool = HybridSearchTool(FakeSearchBackend(raw_response), CONFIG, _UnavailableEmbedding())  # type: ignore[arg-type]

    response = tool.invoke(HybridSearchRequest(query="wireless mouse", top_k=3, bm25_k=3, vector_k=3))

    assert response.retrieval_method == "bm25"
    assert response.results[0].product_id == "p-1"
    assert response.warnings == ["vector_fallback_bm25:embedding_unavailable"]


def test_embedding_client_rejects_wrong_dimension() -> None:
    class WrongDimensionSession(_EmbeddingSession):
        def post(self, url: str, **kwargs: Any) -> Any:
            class Response:
                status_code = 200
                text = ""

                def json(self) -> Dict[str, Any]:
                    return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

            return Response()

    client = EmbeddingClient(EmbeddingConfig(dimension=3), session=WrongDimensionSession())  # type: ignore[arg-type]
    with pytest.raises(EmbeddingError) as error:
        client.embed_query("mouse")

    assert error.value.code == "embedding_dimension_mismatch"


def test_index_embeddings_processes_final_partial_batch(tmp_path: Path) -> None:
    input_path = tmp_path / "products.jsonl"
    documents = [
        {
            "product_id": "p-%s" % index,
            "locale": "us",
            "title": "Mouse %s" % index,
            "source_ref": "esci:v0:products:us:p-%s" % index,
        }
        for index in range(3)
    ]
    input_path.write_text("\n".join(json.dumps(item) for item in documents) + "\n", encoding="utf-8")

    class FakeIndexClient:
        def __init__(self) -> None:
            self.bulk_calls = []

        def ensure_index(self, index_name: str, body: Dict[str, Any]) -> str:
            assert index_name == "amazon_products_v4"
            return "created"

        def bulk(self, payload: str) -> Dict[str, Any]:
            self.bulk_calls.append(payload)
            item_count = len(payload.strip().splitlines()) // 2
            return {"items": [{"index": {"status": 201}} for _ in range(item_count)]}

    class FakeEmbeddingClient:
        model = "text-embedding-nomic-embed-text-v1.5"
        dimension = 3

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

    fake_client = FakeIndexClient()
    report = index_embeddings(
        input_path,
        fake_client,  # type: ignore[arg-type]
        FakeEmbeddingClient(),  # type: ignore[arg-type]
        CONFIG,
        batch_size=2,
    )

    assert report.total == 3
    assert report.successful == 3
    assert report.failed == 0
    assert len(fake_client.bulk_calls) == 2
    assert '"embedding_model_id":"text-embedding-nomic-embed-text-v1.5"' in fake_client.bulk_calls[0]


def test_index_embeddings_starts_from_requested_record(tmp_path: Path) -> None:
    input_path = tmp_path / "products.jsonl"
    documents = [
        {
            "product_id": "p-%s" % index,
            "locale": "us",
            "title": "Mouse %s" % index,
            "source_ref": "esci:v0:products:us:p-%s" % index,
        }
        for index in range(5)
    ]
    input_path.write_text("\n".join(json.dumps(item) for item in documents) + "\n", encoding="utf-8")

    class FakeIndexClient:
        def __init__(self) -> None:
            self.bulk_calls = []

        def ensure_index(self, index_name: str, body: Dict[str, Any]) -> str:
            return "exists"

        def bulk(self, payload: str) -> Dict[str, Any]:
            self.bulk_calls.append(payload)
            item_count = len(payload.strip().splitlines()) // 2
            return {"items": [{"index": {"status": 200}} for _ in range(item_count)]}

    class FakeEmbeddingClient:
        model = "text-embedding-nomic-embed-text-v1.5"
        dimension = 3

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

    fake_client = FakeIndexClient()
    report = index_embeddings(
        input_path,
        fake_client,  # type: ignore[arg-type]
        FakeEmbeddingClient(),  # type: ignore[arg-type]
        CONFIG,
        batch_size=2,
        start_record=3,
    )

    assert report.start_record == 3
    assert report.total == 3
    assert report.successful == 3
    assert len(fake_client.bulk_calls) == 2
    assert '"_id":"p-0"' not in "".join(fake_client.bulk_calls)
    assert '"_id":"p-2"' in fake_client.bulk_calls[0]


def test_validate_hybrid_mapping_accepts_required_text_and_vector_fields() -> None:
    mapping = build_hybrid_index_body(768)["mappings"]

    result = validate_hybrid_mapping(
        {"amazon_products_v4": {"mappings": mapping}},
        "amazon_products_v4",
        768,
    )

    assert result["passed"] is True
    assert result["errors"] == []


def test_validate_hybrid_mapping_rejects_wrong_vector_dimension() -> None:
    mapping = build_hybrid_index_body(384)["mappings"]

    result = validate_hybrid_mapping(
        {"amazon_products_v4": {"mappings": mapping}},
        "amazon_products_v4",
        768,
    )

    assert result["passed"] is False
    assert any("embedding.dimension" in error for error in result["errors"])
