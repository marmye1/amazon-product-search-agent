"""向量候选召回。"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from .embedding_client import EmbeddingClient
from .normalize_search_result import normalize_hit
from .hybrid_models import HybridSearchRequest, HybridSearchResult


def build_vector_search_body(
    request: HybridSearchRequest,
    query_vector: List[float],
    *,
    vector_field: str = "embedding",
) -> Dict[str, Any]:
    if not query_vector:
        raise ValueError("query_vector 不能为空")
    knn_clause: Dict[str, Any] = {"vector": query_vector, "k": request.vector_k}
    filters = request.filter_clauses()
    if filters:
        knn_clause["filter"] = {"bool": {"filter": filters}}
    return {
        "size": request.vector_k,
        "track_total_hits": True,
        "_source": [
            "product_id",
            "locale",
            "title",
            "brand",
            "category",
            "description",
            "bullet_points",
            "source_ref",
        ],
        "query": {"knn": {vector_field: knn_clause}},
    }


def retrieve_vector(
    request: HybridSearchRequest,
    client: Any,
    embedding_client: EmbeddingClient,
    config: Mapping[str, Any],
) -> List[HybridSearchResult]:
    """把查询转成向量后调用 OpenSearch kNN。"""

    query_vector = embedding_client.embed_query(request.query)
    index_name = str(config.get("retrieval", {}).get("hybrid_index_name", "amazon_products_v4"))
    body = build_vector_search_body(request, query_vector)
    raw_response = client.search(index_name, body)
    raw_hits = raw_response.get("hits", {}).get("hits", [])
    if not isinstance(raw_hits, list):
        raise ValueError("向量检索返回的 hits.hits 不是数组")

    results: List[HybridSearchResult] = []
    seen_ids = set()
    for rank, raw_hit in enumerate(raw_hits, start=1):
        result = normalize_hit(raw_hit)
        if result.product_id in seen_ids:
            continue
        seen_ids.add(result.product_id)
        results.append(
            HybridSearchResult(
                product_id=result.product_id,
                title=result.title,
                brand=result.brand,
                category=result.category,
                description=result.description,
                bullet_points=list(result.bullet_points),
                score=result.score,
                matched_fields=list(result.matched_fields),
                source_ref=result.source_ref,
                vector_rank=rank,
                retrieval_channels=["vector"],
            )
        )
    return results
