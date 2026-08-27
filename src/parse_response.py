"""把本地 Qwen 的原始输出解析为 RecommendationResponse。"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Optional, Set

from langchain_core.output_parsers import JsonOutputParser

from .models import ContractError
from .rag_models import EvidenceItem, RecommendationItem, RecommendationResponse


class ParseError(ContractError):
    """模型输出无法满足结构化回答契约。"""


_JSON_PARSER = JsonOutputParser()


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParseError("invalid_output_field", "%s 必须是非空字符串" % field_name)
    return value.strip()


def _string_array(value: Any, field_name: str) -> list:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ParseError("invalid_output_field", "%s 必须是非空字符串数组" % field_name)
    return [item.strip() for item in value]


def _parse_json(raw_output: str) -> Mapping[str, Any]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ParseError("invalid_output", "模型没有返回文本")
    cleaned = raw_output.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    try:
        # LangChain parser 负责 JSON 结构识别，json.loads 拒绝被截断的“部分 JSON”。
        parsed = _JSON_PARSER.parse(cleaned)
        strict_parsed = json.loads(cleaned)
    except Exception as exc:
        raise ParseError("invalid_json", "模型输出不是合法 JSON") from exc
    if parsed != strict_parsed:
        raise ParseError("invalid_json", "模型输出不是完整 JSON")
    if not isinstance(strict_parsed, Mapping):
        raise ParseError("invalid_json", "模型输出必须是 JSON 对象")
    return strict_parsed


def _check_allowed(value: str, allowed: Optional[Set[str]], field_name: str) -> None:
    if allowed is not None and value not in allowed:
        raise ParseError("unknown_reference", "%s 不在检索上下文中: %s" % (field_name, value))


def parse_recommendation_response(
    raw_output: str,
    *,
    candidate_product_ids: Optional[Iterable[str]] = None,
    allowed_source_ids: Optional[Iterable[str]] = None,
) -> RecommendationResponse:
    """严格解析模型 JSON，并提前拦截不存在的商品和证据 ID。"""

    payload = _parse_json(raw_output)
    candidate_ids = set(candidate_product_ids) if candidate_product_ids is not None else None
    source_ids = set(allowed_source_ids) if allowed_source_ids is not None else None

    answer = _required_string(payload.get("answer"), "answer")
    raw_recommendations = payload.get("recommendations")
    raw_evidence = payload.get("evidence")
    limitations = _string_array(payload.get("limitations"), "limitations")
    grounded = payload.get("grounded")
    if not isinstance(raw_recommendations, list):
        raise ParseError("invalid_output_field", "recommendations 必须是数组")
    if not isinstance(raw_evidence, list):
        raise ParseError("invalid_output_field", "evidence 必须是数组")
    if not isinstance(grounded, bool):
        raise ParseError("invalid_output_field", "grounded 必须是布尔值")

    recommendations = []
    for index, item in enumerate(raw_recommendations, start=1):
        if not isinstance(item, Mapping):
            raise ParseError("invalid_recommendation", "第 %s 个推荐项必须是对象" % index)
        product_id = _required_string(item.get("product_id"), "recommendations[%s].product_id" % index)
        _check_allowed(product_id, candidate_ids, "product_id")
        title = _required_string(item.get("title"), "recommendations[%s].title" % index)
        reason = _required_string(item.get("reason"), "recommendations[%s].reason" % index)
        evidence_source_ids = _string_array(
            item.get("evidence_source_ids"),
            "recommendations[%s].evidence_source_ids" % index,
        )
        for source_id in evidence_source_ids:
            _check_allowed(source_id, source_ids, "evidence_source_id")
        recommendations.append(
            RecommendationItem(
                product_id=product_id,
                title=title,
                reason=reason,
                evidence_source_ids=evidence_source_ids,
            )
        )

    evidence = []
    for index, item in enumerate(raw_evidence, start=1):
        if not isinstance(item, Mapping):
            raise ParseError("invalid_evidence", "第 %s 条证据必须是对象" % index)
        source_id = _required_string(item.get("source_id"), "evidence[%s].source_id" % index)
        _check_allowed(source_id, source_ids, "source_id")
        evidence.append(
            EvidenceItem(
                source_id=source_id,
                product_id=_required_string(item.get("product_id"), "evidence[%s].product_id" % index),
                field_name=_required_string(item.get("field_name"), "evidence[%s].field_name" % index),
                quoted_or_paraphrased_fact=_required_string(
                    item.get("quoted_or_paraphrased_fact"),
                    "evidence[%s].quoted_or_paraphrased_fact" % index,
                ),
            )
        )

    if "retrieval_method" not in payload:
        raise ParseError("missing_output_field", "模型输出缺少 retrieval_method")
    if "answer_version" not in payload:
        raise ParseError("missing_output_field", "模型输出缺少 answer_version")
    retrieval_method = payload.get("retrieval_method")
    answer_version = payload.get("answer_version")
    if retrieval_method != "bm25":
        raise ParseError("invalid_output_field", "retrieval_method 必须是 bm25")
    if answer_version != "v2":
        raise ParseError("invalid_output_field", "answer_version 必须是 v2")

    return RecommendationResponse(
        answer=answer,
        recommendations=recommendations,
        evidence=evidence,
        limitations=limitations,
        grounded=grounded,
        retrieval_method="bm25",
        answer_version="v2",
    )
