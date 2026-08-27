"""固定的“检索后生成”链，以及本地 LM Studio Qwen 适配器。"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import requests
from langchain_core.runnables import RunnableLambda

from .build_context import build_context
from .grounding_check import GroundingReport, check_grounding
from .models import ContractError, SearchRequest
from .opensearch_client import BackendError, OpenSearchClient
from .parse_response import ParseError, parse_recommendation_response
from .prompts.recommendation_prompt import build_recommendation_messages
from .rag_models import ContextBuildResult, RAGRequest, RecommendationItem, RecommendationResponse
from .search_config import load_search_config, config_value
from .search_products import search_products


BASE_LIMITATIONS = [
    "回答只依据  BM25 检索到的商品字段。",
    "数据不包含实时价格、库存、评分、销量和配送信息。",
]
_BUDGET_CLAIM_PATTERN = re.compile(
    r"(?:符合|满足|低于|不超过|within|under|below)[^。！？\n]{0,20}"
    r"(?:预算|budget|价格|price|\$|美元)",
    re.I,
)


def _contains_budget_condition(value: str) -> bool:
    lowered = (value or "").casefold()
    return any(token in lowered for token in ("budget", "price", "dollar", "usd", "$", "预算", "价格", "美元", "元"))


def _neutralize_unverifiable_budget_claims(
    response: RecommendationResponse,
    user_query: str,
) -> RecommendationResponse:
    """价格字段缺失时，把模型的预算满足断言改成可验证的中性说明。"""

    if not _contains_budget_condition(user_query):
        return response
    has_budget_claim = bool(
        _BUDGET_CLAIM_PATTERN.search(
            "\n".join([response.answer] + [item.reason for item in response.recommendations])
        )
    )
    if not has_budget_claim:
        return response

    safe_recommendations = [
        RecommendationItem(
            product_id=item.product_id,
            title=item.title,
            reason=(
                "商品字段与当前商品条件相关；价格信息未提供，无法验证预算条件。"
                if _BUDGET_CLAIM_PATTERN.search(item.reason)
                else item.reason
            ),
            evidence_source_ids=list(item.evidence_source_ids),
        )
        for item in response.recommendations
    ]
    safe_limitations = list(response.limitations)
    safe_limitations.append("商品上下文未提供价格信息，无法验证预算条件。")
    return replace(
        response,
        answer="以下商品与当前商品条件相关，但商品上下文未提供价格信息，无法验证预算条件。",
        recommendations=safe_recommendations,
        limitations=list(dict.fromkeys(safe_limitations)),
    )


RECOMMENDATION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "maxLength": 240},
        "recommendations": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "title": {"type": "string", "maxLength": 240},
                    "reason": {"type": "string", "maxLength": 120},
                    "evidence_source_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                },
                "required": ["product_id", "title", "reason", "evidence_source_ids"],
                "additionalProperties": False,
            },
        },
        "evidence": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "product_id": {"type": "string"},
                    "field_name": {"type": "string"},
                    "quoted_or_paraphrased_fact": {"type": "string", "maxLength": 300},
                },
                "required": ["source_id", "product_id", "field_name", "quoted_or_paraphrased_fact"],
                "additionalProperties": False,
            },
        },
        "limitations": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "grounded": {"type": "boolean"},
        "retrieval_method": {"type": "string", "enum": ["bm25"]},
        "answer_version": {"type": "string", "enum": ["v2"]},
    },
    "required": [
        "answer",
        "recommendations",
        "evidence",
        "limitations",
        "grounded",
        "retrieval_method",
        "answer_version",
    ],
    "additionalProperties": False,
}


class LocalQwenError(RuntimeError):
    """本地 Qwen 请求或响应错误。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class LocalQwenConfig:
    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = "qwen-local"
    api_key: str = "lm-studio"
    timeout_seconds: float = 180.0
    max_tokens: int = 1200
    temperature: float = 0.0
    json_mode: bool = True
    reasoning_effort: str = "none"

    @classmethod
    def from_env(cls) -> "LocalQwenConfig":
        return cls(
            base_url=os.getenv("QWEN_BASE_URL", cls.base_url),
            model=os.getenv("QWEN_MODEL", cls.model),
            api_key=os.getenv("QWEN_API_KEY", cls.api_key),
            timeout_seconds=float(os.getenv("QWEN_TIMEOUT_SECONDS", str(cls.timeout_seconds))),
            max_tokens=int(os.getenv("QWEN_MAX_TOKENS", str(cls.max_tokens))),
            temperature=float(os.getenv("QWEN_TEMPERATURE", str(cls.temperature))),
            json_mode=os.getenv("QWEN_JSON_MODE", "true").lower() not in {"0", "false", "no"},
            reasoning_effort=os.getenv("QWEN_REASONING_EFFORT", cls.reasoning_effort),
        )

    @property
    def chat_url(self) -> str:
        return "%s/chat/completions" % self.base_url.rstrip("/")


@dataclass(frozen=True)
class GenerationMetadata:
    model: str
    endpoint: str
    elapsed_ms: int = 0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    response_chars: int = 0
    error: Optional[Dict[str, str]] = None
    skipped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationResult:
    raw_output: str
    metadata: GenerationMetadata


@dataclass(frozen=True)
class RAGExecutionResult:
    response: RecommendationResponse
    context: ContextBuildResult
    generation: GenerationMetadata
    grounding: GroundingReport
    raw_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response.to_dict(),
            "context": self.context.to_dict(),
            "generation": self.generation.to_dict(),
            "grounding": self.grounding.to_dict(),
            "raw_output": self.raw_output,
        }


def invoke_local_qwen(
    messages: List[Dict[str, str]],
    config: LocalQwenConfig,
    *,
    session: Optional[requests.Session] = None,
) -> GenerationResult:
    """通过 LM Studio OpenAI 兼容接口调用本地 Qwen。"""

    payload: Dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": False,
        "reasoning_effort": config.reasoning_effort,
    }
    if config.json_mode:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "recommendation_response",
                "schema": RECOMMENDATION_JSON_SCHEMA,
                "strict": True,
            },
        }

    http = session or requests.Session()
    if hasattr(http, "trust_env"):
        http.trust_env = False
    started = time.monotonic()
    try:
        response = http.post(
            config.chat_url,
            headers={"Authorization": "Bearer %s" % config.api_key},
            json=payload,
            timeout=config.timeout_seconds,
        )
    except requests.Timeout as exc:
        raise LocalQwenError("llm_timeout", "本地 Qwen 请求超时") from exc
    except requests.ConnectionError as exc:
        raise LocalQwenError("llm_unavailable", "无法连接本地 Qwen: %s" % config.chat_url) from exc
    except requests.RequestException as exc:
        raise LocalQwenError("llm_request_error", "本地 Qwen 请求失败") from exc

    if response.status_code >= 400:
        detail = response.text.strip().replace("\n", " ")[:500]
        raise LocalQwenError("llm_http_error", "本地 Qwen 返回 HTTP %s: %s" % (response.status_code, detail))
    try:
        body = response.json()
    except ValueError as exc:
        raise LocalQwenError("llm_invalid_response", "本地 Qwen 返回的不是合法 JSON") from exc

    choices = body.get("choices") if isinstance(body, Mapping) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise LocalQwenError("llm_invalid_response", "本地 Qwen 返回缺少 choices")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, Mapping) else str(part) for part in content
        )
    if (not isinstance(content, str) or not content.strip()) and isinstance(message, Mapping):
        # Qwen 在未关闭 thinking 时可能把可解析内容放进 reasoning_content。
        content = message.get("reasoning_content")
    if not isinstance(content, str) or not content.strip():
        raise LocalQwenError("llm_invalid_response", "本地 Qwen 返回缺少 message.content")

    usage = body.get("usage") if isinstance(body, Mapping) else {}
    usage = usage if isinstance(usage, Mapping) else {}
    choice = choices[0]
    metadata = GenerationMetadata(
        model=config.model,
        endpoint=config.chat_url,
        elapsed_ms=int(round((time.monotonic() - started) * 1000)),
        prompt_tokens=usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None,
        completion_tokens=usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None,
        total_tokens=usage.get("total_tokens") if isinstance(usage.get("total_tokens"), int) else None,
        finish_reason=choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None,
        response_chars=len(content),
    )
    return GenerationResult(raw_output=content.strip(), metadata=metadata)


def _fallback_response(answer: str, limitations: List[str]) -> RecommendationResponse:
    return RecommendationResponse(
        answer=answer,
        recommendations=[],
        evidence=[],
        limitations=list(dict.fromkeys(BASE_LIMITATIONS + limitations)),
        grounded=False,
    )


def _failure_report(message: str) -> GroundingReport:
    return GroundingReport(grounded=False, missing_evidence=[message])


def run_rag(
    request: RAGRequest,
    *,
    llm_config: Optional[LocalQwenConfig] = None,
    invoke_model: Optional[Callable[[List[Dict[str, str]]], GenerationResult]] = None,
) -> RAGExecutionResult:
    """执行固定顺序：build_context → prompt → Qwen → parse → grounding。"""

    config = llm_config or LocalQwenConfig.from_env()
    context = build_context(request.search_response, max_products=request.max_products)
    if not context.blocks:
        generation = GenerationMetadata(
            model=config.model,
            endpoint=config.chat_url,
            error={"code": "no_results", "message": "没有检索到商品上下文"},
            skipped=True,
        )
        report = _failure_report("没有可用于检查的商品上下文")
        response = _fallback_response("没有检索到可用于推荐的商品信息，因此不能编造商品建议。", [])
        return RAGExecutionResult(response=response, context=context, generation=generation, grounding=report)

    messages = build_recommendation_messages(
        request.user_query,
        context.blocks,
        request.answer_language,
        request.max_products,
    )
    runnable = RunnableLambda(
        invoke_model
        or (lambda input_messages: invoke_local_qwen(input_messages, config))
    )
    try:
        generated = runnable.invoke(messages)
    except LocalQwenError as exc:
        metadata = GenerationMetadata(
            model=config.model,
            endpoint=config.chat_url,
            error={"code": exc.code, "message": exc.message},
        )
        report = _failure_report(exc.message)
        response = _fallback_response("本地模型暂时不可用，当前无法生成可靠的商品建议。", [exc.message])
        return RAGExecutionResult(response=response, context=context, generation=metadata, grounding=report)

    if not isinstance(generated, GenerationResult):
        raise TypeError("invoke_model 必须返回 GenerationResult")

    if generated.metadata.finish_reason == "length":
        message = "本地 Qwen 输出达到 max_tokens 上限，回答不完整"
        metadata = replace(
            generated.metadata,
            error={"code": "llm_truncated", "message": message},
        )
        report = _failure_report(message)
        fallback = _fallback_response("模型回答不完整，因此不输出未经验证的商品推荐。", [message])
        return RAGExecutionResult(
            response=fallback,
            context=context,
            generation=metadata,
            grounding=report,
            raw_output=generated.raw_output,
        )

    candidate_ids = {block.product_id for block in context.blocks}
    source_ids = {block.source_id for block in context.blocks}
    try:
        response = parse_recommendation_response(
            generated.raw_output,
            candidate_product_ids=candidate_ids,
            allowed_source_ids=source_ids,
        )
    except (ParseError, ContractError) as exc:
        metadata = replace(
            generated.metadata,
            error={"code": getattr(exc, "code", "parse_error"), "message": str(exc)},
        )
        report = _failure_report(str(exc))
        fallback = _fallback_response("模型回答未通过结构化校验，因此不输出未经验证的商品推荐。", [str(exc)])
        return RAGExecutionResult(
            response=fallback,
            context=context,
            generation=metadata,
            grounding=report,
            raw_output=generated.raw_output,
        )

    response = _neutralize_unverifiable_budget_claims(response, request.user_query)
    report = check_grounding(response, context.blocks)
    if not report.grounded:
        reasons = report.missing_evidence + report.invalid_product_ids + report.invalid_source_ids + report.unsupported_claims
        fallback = _fallback_response(
            "模型回答没有通过基础证据检查，因此不输出未经验证的商品推荐。",
            ["；".join(reasons)] if reasons else ["证据关联不完整"],
        )
        return RAGExecutionResult(
            response=fallback,
            context=context,
            generation=generated.metadata,
            grounding=report,
            raw_output=generated.raw_output,
        )

    grounded_response = replace(response, grounded=True)
    return RAGExecutionResult(
        response=grounded_response,
        context=context,
        generation=generated.metadata,
        grounding=report,
        raw_output=generated.raw_output,
    )


def run_rag_query(
    search_request: SearchRequest,
    client: Any,
    search_config: Mapping[str, Any],
    *,
    max_products: int = 5,
    llm_config: Optional[LocalQwenConfig] = None,
) -> RAGExecutionResult:
    """对外入口：先调用 BM25，再执行固定 RAG 链。"""

    search_response = search_products(search_request, client, search_config)
    request = RAGRequest(
        user_query=search_request.query,
        search_response=search_response,
        max_products=max_products,
    )
    return run_rag(request, llm_config=llm_config)


def main() -> int:
    parser = argparse.ArgumentParser(description="执行本地 Qwen 两步 RAG 商品回答")
    parser.add_argument("--config", type=Path, default=Path("config/search.yaml"))
    parser.add_argument("--query", required=True)
    parser.add_argument("--locale")
    parser.add_argument("--category")
    parser.add_argument("--brand")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-products", type=int, default=5)
    parser.add_argument("--qwen-model", default=None)
    args = parser.parse_args()

    config_path = args.config.resolve()
    search_config = load_search_config(config_path)
    default_top_k = int(config_value(search_config, "search", "default_top_k", 10))
    max_top_k = int(config_value(search_config, "search", "max_top_k", 100))
    request = SearchRequest.from_mapping(
        {
            "query": args.query,
            "locale": args.locale,
            "category": args.category,
            "brand": args.brand,
            "top_k": default_top_k if args.top_k is None else args.top_k,
        },
        default_top_k=default_top_k,
        max_top_k=max_top_k,
    )
    llm_config = LocalQwenConfig.from_env()
    if args.qwen_model:
        llm_config = replace(llm_config, model=args.qwen_model)

    try:
        client = OpenSearchClient.from_config(search_config, prompt_for_missing=True)
        result = run_rag_query(
            request,
            client,
            search_config,
            max_products=args.max_products,
            llm_config=llm_config,
        )
    except (BackendError, ContractError, ValueError) as exc:
        print(json.dumps({"error": {"code": getattr(exc, "code", "rag_failed"), "message": str(exc)}}, ensure_ascii=False))
        return 2

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
