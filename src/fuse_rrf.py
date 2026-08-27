"""Reciprocal Rank Fusion 融合模块。"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .models import SearchResult
from .hybrid_models import HybridSearchResult


def fuse_rrf(
    bm25_candidates: Sequence[SearchResult],
    vector_candidates: Sequence[SearchResult],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> Tuple[List[HybridSearchResult], Dict[str, List[str]]]:
    """按固定 RRF 参数融合两路候选，并按商品编号稳定处理平分。"""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k 必须是大于 0 的整数")
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 1:
        raise ValueError("rrf_k 必须是大于 0 的整数")

    by_id: Dict[str, Dict[str, object]] = {}

    def add_candidates(candidates: Sequence[SearchResult], channel: str) -> None:
        for position, candidate in enumerate(candidates, start=1):
            item = by_id.setdefault(
                candidate.product_id,
                {
                    "result": candidate,
                    "score": 0.0,
                    "channels": [],
                    "bm25_rank": None,
                    "vector_rank": None,
                },
            )
            item["score"] = float(item["score"]) + 1.0 / float(rrf_k + position)
            channels = item["channels"]
            if isinstance(channels, list) and channel not in channels:
                channels.append(channel)
            rank_name = "bm25_rank" if channel == "bm25" else "vector_rank"
            item[rank_name] = position
            # BM25 结果优先作为商品字段来源；向量结果只提供补充候选和排序。
            if channel == "bm25":
                item["result"] = candidate

    add_candidates(bm25_candidates, "bm25")
    add_candidates(vector_candidates, "vector")

    ranked = sorted(
        by_id.items(),
        key=lambda pair: (-float(pair[1]["score"]), pair[0]),
    )[:top_k]
    results: List[HybridSearchResult] = []
    retrieval_channels: Dict[str, List[str]] = {}
    for product_id, item in ranked:
        result = item["result"]
        if not isinstance(result, SearchResult):
            raise ValueError("RRF 内部候选类型无效")
        channels = list(item["channels"]) if isinstance(item["channels"], list) else []
        retrieval_channels[product_id] = channels
        results.append(
            HybridSearchResult(
                product_id=result.product_id,
                title=result.title,
                brand=result.brand,
                category=result.category,
                description=result.description,
                bullet_points=list(result.bullet_points),
                score=float(item["score"]),
                matched_fields=list(result.matched_fields),
                source_ref=result.source_ref,
                bm25_rank=item["bm25_rank"] if isinstance(item["bm25_rank"], int) else None,
                vector_rank=item["vector_rank"] if isinstance(item["vector_rank"], int) else None,
                rrf_score=float(item["score"]),
                retrieval_channels=channels,
            )
        )
    return results, retrieval_channels
