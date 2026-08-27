"""可解释规则重排序。"""

from __future__ import annotations

from typing import Any, List, Mapping

from .models import SearchResult
from .agent_models import RerankRequest, RerankResponse, RerankedResult


RERANK_MODEL_ID = "rule-rerank-"
_UNKNOWN = {"", "unknown", "null", "none", "n/a"}


def _known(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    return "" if cleaned.casefold() in _UNKNOWN else cleaned


def _constraint_terms(constraints: Mapping[str, Any]) -> List[str]:
    values: List[str] = []
    for field_name in ("brand_en", "category_en", "use_case_en", "brand", "category", "use_case"):
        value = _known(constraints.get(field_name))
        if value:
            values.append(value)
    for field_name in ("must_have_en", "search_terms_en", "must_have", "search_terms"):
        raw_values = constraints.get(field_name, [])
        if isinstance(raw_values, list):
            values.extend(value.strip() for value in raw_values if isinstance(value, str) and value.strip())
    retrieval_query = _known(constraints.get("retrieval_query"))
    if retrieval_query:
        values.extend(part for part in retrieval_query.split() if part)
    deduplicated: List[str] = []
    seen = set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            deduplicated.append(value)
    return deduplicated


def _product_text(result: SearchResult) -> str:
    values = [result.title, result.brand, result.category, result.description, getattr(result, "color", None)]
    values.extend(result.bullet_points or [])
    return " ".join(value for value in values if isinstance(value, str)).casefold()


def _rank_component(rank: Any) -> float:
    if isinstance(rank, int) and not isinstance(rank, bool) and rank > 0:
        return 1.0 / float(rank)
    return 0.0


def _score(result: SearchResult, constraints: Mapping[str, Any]) -> float:
    bm25_rank = getattr(result, "bm25_rank", None)
    vector_rank = getattr(result, "vector_rank", None)
    if bm25_rank is not None and vector_rank is not None:
        rank_score = 0.5 * _rank_component(bm25_rank) + 0.5 * _rank_component(vector_rank)
    elif bm25_rank is not None:
        rank_score = 0.7 * _rank_component(bm25_rank)
    elif vector_rank is not None:
        rank_score = 0.7 * _rank_component(vector_rank)
    else:
        rank_score = 0.0

    text = _product_text(result)
    terms = _constraint_terms(constraints)
    if terms:
        matched = sum(1 for term in terms if term.casefold() in text)
        coverage = matched / float(len(terms))
    else:
        coverage = 0.0
    # 明确表达的 must_have/brand/category 在候选中应有可解释的排序增益，
    # 但最终是否合格仍由 check_constraints 决定。
    return round(rank_score + 0.75 * coverage, 8)


def _to_reranked(result: SearchResult, score: float) -> RerankedResult:
    retrieval_method = getattr(result, "retrieval_method", None)
    if not isinstance(retrieval_method, str) or not retrieval_method:
        retrieval_method = "hybrid_rrf" if getattr(result, "bm25_rank", None) or getattr(result, "vector_rank", None) else "bm25"
    return RerankedResult(
        product_id=result.product_id,
        title=result.title,
        brand=result.brand,
        category=result.category,
        description=result.description,
        bullet_points=list(result.bullet_points),
        score=result.score,
        matched_fields=list(result.matched_fields),
        source_ref=result.source_ref,
        bm25_rank=getattr(result, "bm25_rank", None),
        vector_rank=getattr(result, "vector_rank", None),
        rrf_score=getattr(result, "rrf_score", None),
        retrieval_channels=list(getattr(result, "retrieval_channels", [])),
        rerank_score=score,
        original_score=float(result.score),
        retrieval_method=retrieval_method,
    )


def rerank_candidates(request: RerankRequest) -> RerankResponse:
    """按召回排名和约束词覆盖率排序，不修改商品事实字段。"""

    scored = [(_score(candidate, request.parsed_constraints), candidate) for candidate in request.candidates]
    scored.sort(key=lambda pair: (-pair[0], pair[1].product_id))
    results = [_to_reranked(candidate, score) for score, candidate in scored[: request.rerank_top_k]]
    return RerankResponse(
        user_query=request.user_query,
        results=results,
        rerank_model_id=request.rerank_model_id or RERANK_MODEL_ID,
        fallback_used=False,
    )
