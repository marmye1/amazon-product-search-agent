"""BM25 候选召回，复用配置中的字段权重和过滤语义。"""

from __future__ import annotations

from typing import Any, List, Mapping

from .models import SearchRequest
from .search_config import config_value
from .search_products import search_products
from .hybrid_models import HybridSearchRequest, HybridSearchResult


def retrieve_bm25(
    request: HybridSearchRequest,
    client: Any,
    config: Mapping[str, Any],
) -> List[HybridSearchResult]:
    """返回带 BM25 排名的候选，不执行向量或融合逻辑。"""

    search_request = SearchRequest.from_mapping(
        {
            "query": request.query,
            "locale": request.locale,
            "category": request.category,
            "brand": request.brand,
            "top_k": request.bm25_k,
        },
        default_top_k=request.bm25_k,
        max_top_k=max(request.bm25_k, int(config.get("search", {}).get("max_top_k", request.bm25_k))),
    )
    hybrid_index_name = str(config_value(config, "retrieval", "hybrid_index_name", "amazon_products_v4"))
    response = search_products(search_request, client, config, index_name=hybrid_index_name)
    results: List[HybridSearchResult] = []
    for rank, result in enumerate(response.results, start=1):
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
                bm25_rank=rank,
                retrieval_channels=["bm25"],
            )
        )
    return results
