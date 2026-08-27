"""最小证据检查。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from .rag_models import ContextBlock, RecommendationResponse


@dataclass(frozen=True)
class GroundingReport:
    """记录回答是否能被检索上下文支撑。"""

    grounded: bool
    missing_evidence: List[str] = field(default_factory=list)
    invalid_product_ids: List[str] = field(default_factory=list)
    invalid_source_ids: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "grounded": self.grounded,
            "missing_evidence": list(self.missing_evidence),
            "invalid_product_ids": list(self.invalid_product_ids),
            "invalid_source_ids": list(self.invalid_source_ids),
            "unsupported_claims": list(self.unsupported_claims),
        }


_UNSUPPORTED_CLAIM_PATTERNS = (
    re.compile(
        r"(?:价格|售价|费用|库存|现货|配送|销量|评分|星级)[^。！？\n]{0,20}"
        r"(?:¥\s*\d|￥\s*\d|\$\s*\d|\d+(?:\.\d+)?\s*(?:元|美元|星|分)|是|为|有|无|不足|充足|高|低|快|慢|售罄)",
        re.I,
    ),
    re.compile(r"(?:¥|￥|\$)\s*\d", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:元|美元|星|分)\b", re.I),
    re.compile(
        r"(?:符合|满足|低于|不超过|within|under|below)[^。！？\n]{0,20}"
        r"(?:预算|budget|价格|price|\$|美元)",
        re.I,
    ),
)


def _find_unsupported_claims(response: RecommendationResponse) -> List[str]:
    text = "\n".join(
        [response.answer]
        + [item.reason for item in response.recommendations]
        + list(response.limitations)
    )
    claims = []
    for pattern in _UNSUPPORTED_CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            window = text[max(0, match.start() - 18) : min(len(text), match.end() + 18)]
            # “未提供/无法验证/不包含价格”是限制声明，不是声称有价格或满足预算。
            if any(
                phrase in window
                for phrase in ("未提供", "未包含", "无法验证", "无法确认", "不包含", "没有价格")
            ):
                continue
            claims.append(match.group(0))
    return sorted(set(claims))


def check_grounding(
    response: RecommendationResponse,
    context_blocks: Iterable[ContextBlock],
) -> GroundingReport:
    """检查商品 ID、证据 ID、证据归属和禁止的无依据核心事实。"""

    blocks = list(context_blocks)
    blocks_by_source = {block.source_id: block for block in blocks}
    product_ids = {block.product_id for block in blocks}
    evidence_by_source = {item.source_id: item for item in response.evidence}
    missing_evidence: List[str] = []
    invalid_product_ids: List[str] = []
    invalid_source_ids: List[str] = []

    for recommendation in response.recommendations:
        if recommendation.product_id not in product_ids:
            invalid_product_ids.append(recommendation.product_id)
        if not recommendation.evidence_source_ids:
            missing_evidence.append("%s 缺少 evidence_source_ids" % recommendation.product_id)
        for source_id in recommendation.evidence_source_ids:
            block = blocks_by_source.get(source_id)
            if block is None:
                invalid_source_ids.append(source_id)
                continue
            if block.product_id != recommendation.product_id:
                missing_evidence.append(
                    "%s 的证据 %s 属于 %s"
                    % (recommendation.product_id, source_id, block.product_id)
                )
            if source_id not in evidence_by_source:
                missing_evidence.append("缺少证据对象: %s" % source_id)

    for item in response.evidence:
        block = blocks_by_source.get(item.source_id)
        if block is None:
            invalid_source_ids.append(item.source_id)
            continue
        if block.product_id != item.product_id or block.field_name != item.field_name:
            missing_evidence.append("证据字段与上下文不一致: %s" % item.source_id)

    unsupported_claims = _find_unsupported_claims(response)
    return GroundingReport(
        grounded=not any(
            (missing_evidence, invalid_product_ids, invalid_source_ids, unsupported_claims)
        ),
        missing_evidence=sorted(set(missing_evidence)),
        invalid_product_ids=sorted(set(invalid_product_ids)),
        invalid_source_ids=sorted(set(invalid_source_ids)),
        unsupported_claims=unsupported_claims,
    )
