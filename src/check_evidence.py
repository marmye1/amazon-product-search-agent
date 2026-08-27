"""字段级回答证据校验。"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

from .grounding_check import check_grounding
from .models import SearchResult
from .rag_models import ContextBlock, RecommendationResponse
from .agent_models import EvidenceCheck


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.I)
_COLLECTIVE_PATTERN = re.compile(
    r"(?:所有|全部|这些|各款|每款|每个)(?:推荐)?(?:商品)?[^。！？\n]{0,12}"
    r"(?:支持|具备|包含|拥有|适合|提供|采用)([^。！？\n]{1,24})"
)


def _tokens(value: str) -> Set[str]:
    return {token.casefold() for token in _TOKEN_PATTERN.findall(value or "")}


def _supported_fact(fact: str, source_text: str) -> bool:
    fact_norm = " ".join((fact or "").casefold().split())
    source_norm = " ".join((source_text or "").casefold().split())
    if fact_norm and fact_norm in source_norm:
        return True
    fact_tokens = _tokens(fact_norm)
    source_tokens = _tokens(source_norm)
    if not fact_tokens:
        return False
    overlap = len(fact_tokens & source_tokens)
    return overlap >= max(1, min(2, len(fact_tokens)))


def _result_text(result: SearchResult) -> str:
    values = [result.title, result.brand, result.category, result.description, getattr(result, "color", None)]
    values.extend(result.bullet_points or [])
    return " ".join(value for value in values if isinstance(value, str))


def _collective_claims(
    response: RecommendationResponse,
    candidates: Mapping[str, SearchResult],
) -> List[str]:
    text = "\n".join([response.answer] + [item.reason for item in response.recommendations])
    claims: List[str] = []
    for match in _COLLECTIVE_PATTERN.finditer(text):
        claim = match.group(1).strip()
        claim_tokens = _tokens(claim)
        if not claim_tokens:
            continue
        missing_products = []
        for recommendation in response.recommendations:
            candidate = candidates.get(recommendation.product_id)
            if candidate is None or not _supported_fact(claim, _result_text(candidate)):
                missing_products.append(recommendation.product_id)
        if missing_products:
            claims.append("集体性断言未覆盖全部商品: %s（缺少 %s）" % (match.group(0).strip(), ", ".join(missing_products)))
    return claims


def check_evidence(
    response: RecommendationResponse,
    candidates: Sequence[SearchResult],
    context_blocks: Iterable[ContextBlock],
) -> EvidenceCheck:
    """验证推荐商品、证据 source_id、字段归属和集体性断言。"""

    blocks = list(context_blocks)
    blocks_by_source = {block.source_id: block for block in blocks}
    candidates_by_id = {candidate.product_id: candidate for candidate in candidates}
    evidence_by_source = {item.source_id: item for item in response.evidence}
    grounding = check_grounding(response, blocks)

    unsupported_claims = list(grounding.unsupported_claims)
    invalid_product_ids = list(grounding.invalid_product_ids)
    invalid_source_ids = list(grounding.invalid_source_ids)
    missing_evidence = list(grounding.missing_evidence)
    evidence_links: List[Dict[str, Any]] = []

    for recommendation in response.recommendations:
        candidate = candidates_by_id.get(recommendation.product_id)
        if candidate is None:
            if recommendation.product_id not in invalid_product_ids:
                invalid_product_ids.append(recommendation.product_id)
            continue
        if recommendation.title.strip() != candidate.title.strip():
            unsupported_claims.append("商品 %s 的回答标题与候选 title 不一致" % recommendation.product_id)
        if not recommendation.evidence_source_ids:
            missing_evidence.append("%s 缺少 evidence_source_ids" % recommendation.product_id)

        for source_id in recommendation.evidence_source_ids:
            block = blocks_by_source.get(source_id)
            evidence_item = evidence_by_source.get(source_id)
            if block is None:
                invalid_source_ids.append(source_id)
                continue
            if block.product_id != recommendation.product_id:
                invalid_product_ids.append(recommendation.product_id)
                missing_evidence.append("%s 的证据 %s 属于 %s" % (recommendation.product_id, source_id, block.product_id))
            if evidence_item is None:
                missing_evidence.append("缺少证据对象: %s" % source_id)
                evidence_links.append(
                    {"source_id": source_id, "product_id": recommendation.product_id, "supported": False}
                )
                continue
            fact_supported = _supported_fact(evidence_item.quoted_or_paraphrased_fact, block.text)
            if evidence_item.product_id != block.product_id or evidence_item.field_name != block.field_name:
                missing_evidence.append("证据字段与上下文不一致: %s" % source_id)
                fact_supported = False
            if not fact_supported:
                unsupported_claims.append("证据事实无法由字段支撑: %s" % source_id)
            evidence_links.append(
                {
                    "source_id": source_id,
                    "product_id": block.product_id,
                    "field_name": block.field_name,
                    "supported": fact_supported,
                }
            )

    for evidence_item in response.evidence:
        block = blocks_by_source.get(evidence_item.source_id)
        if block is None:
            invalid_source_ids.append(evidence_item.source_id)
            continue
        if evidence_item.product_id not in candidates_by_id:
            invalid_product_ids.append(evidence_item.product_id)

    unsupported_claims.extend(_collective_claims(response, candidates_by_id))
    unsupported_claims = sorted(set(unsupported_claims))
    invalid_product_ids = sorted(set(invalid_product_ids))
    invalid_source_ids = sorted(set(invalid_source_ids))
    missing_evidence = sorted(set(missing_evidence))
    grounded = bool(response.grounded) and not any(
        (unsupported_claims, invalid_product_ids, invalid_source_ids, missing_evidence)
    )
    if grounded:
        confidence_reason = "通过：推荐商品、字段证据和限制声明均可回溯到候选上下文。"
    else:
        confidence_reason = "未通过：存在商品、字段证据或回答断言无法回溯到候选上下文。"
    return EvidenceCheck(
        grounded=grounded,
        unsupported_claims=unsupported_claims,
        invalid_product_ids=invalid_product_ids,
        evidence_links=evidence_links,
        confidence_reason=confidence_reason,
        invalid_source_ids=invalid_source_ids,
        missing_evidence=missing_evidence,
    )
