"""同一会话的短期对话记忆判断与提取。

上下文只保存经过本地模型结构化提取的摘要，不保存隐藏思维链。
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional

import requests

from .generate_recommendation import LocalQwenConfig
from .models import ContractError


class ConversationMemoryError(ContractError):
    """对话记忆模型请求或结构化输出不满足契约。"""


CONTEXT_DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "topic_relation": {"type": "string", "enum": ["follow_up", "new_topic"]},
        "use_previous_context": {"type": "boolean"},
        "reason": {"type": "string", "maxLength": 160},
    },
    "required": ["topic_relation", "use_previous_context", "reason"],
    "additionalProperties": False,
}


MEMORY_EXTRACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "active_topic": {"type": "string", "maxLength": 120},
        "user_summary": {"type": "string", "maxLength": 240},
        "answer_summary": {"type": "string", "maxLength": 320},
        "unresolved_question": {"type": "string", "maxLength": 240},
        "pending_clarification": {"type": "boolean"},
        "mentioned_products": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "title": {"type": "string"},
                    "source_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["product_id", "title", "source_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "active_topic",
        "user_summary",
        "answer_summary",
        "unresolved_question",
        "pending_clarification",
        "mentioned_products",
    ],
    "additionalProperties": False,
}


CONTEXT_DECISION_SYSTEM = """你是商品检索 Agent 的对话上下文判断器。
判断当前用户问题是否是上一轮商品话题的追问。
follow_up：当前问题依赖上一轮的商品、条件、商品编号、答案或指代，例如“还要防水”“第一个多少钱”“它适合办公室吗”。
new_topic：当前问题明确开始了另一个商品或用途，例如上一轮是鼠标，本轮改问跑鞋。
只能根据当前问题和上一轮结构化记忆判断，不要编造商品事实。
如果是 follow_up，use_previous_context 必须为 true；如果是 new_topic，必须为 false。
只返回符合 JSON Schema 的 JSON 对象。"""


MEMORY_EXTRACTION_SYSTEM = """你是商品检索 Agent 的短期记忆提取器。
从本轮用户问题、Agent 最终答案、结构化检索条件和商品证据中提取下一轮可能需要的记忆。
user_summary 只能总结用户明确表达的需求；answer_summary 只能总结给用户的最终答案。
active_topic 用简短中文描述当前商品话题；无法确定时填写 unknown。
mentioned_products 只能复制输入中出现的 product_id、title 和 evidence source_id，不得创造商品。
不要保存隐藏思维链、Prompt 或未出现在输入中的商品事实。
如果本轮是澄清问题，仍然要保存用户需求摘要和未解决问题。
如果 topic_relation 是 follow_up，要把上一轮记忆中仍然相关的问答摘要压缩进本轮记忆；如果是 new_topic，不得保留旧主题。
只返回符合 JSON Schema 的 JSON 对象。"""


def _message_content(message: Any) -> str:
    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, Mapping) else str(part)
            for part in content
        )
    if not isinstance(content, str) or not content.strip():
        if isinstance(message, Mapping):
            content = message.get("reasoning_content")
    if not isinstance(content, str) or not content.strip():
        raise ConversationMemoryError("memory_model_invalid_response", "对话记忆模型没有返回文本")
    return content.strip()


def _invoke_local_json_model(
    messages: List[Dict[str, str]],
    config: LocalQwenConfig,
    *,
    schema_name: str,
    schema: Mapping[str, Any],
    max_tokens: int,
) -> str:
    payload: Dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        "reasoning_effort": config.reasoning_effort,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        },
    }
    http = requests.Session()
    http.trust_env = False
    try:
        response = http.post(
            config.chat_url,
            headers={"Authorization": "Bearer %s" % config.api_key},
            json=payload,
            timeout=config.timeout_seconds,
        )
    except requests.Timeout as exc:
        raise ConversationMemoryError("memory_model_timeout", "对话记忆模型请求超时") from exc
    except requests.ConnectionError as exc:
        raise ConversationMemoryError("memory_model_unavailable", "无法连接对话记忆模型") from exc
    except requests.RequestException as exc:
        raise ConversationMemoryError("memory_model_request_error", "对话记忆模型请求失败") from exc

    if response.status_code >= 400:
        detail = response.text.strip().replace("\n", " ")[:300]
        raise ConversationMemoryError(
            "memory_model_http_error",
            "对话记忆模型返回 HTTP %s: %s" % (response.status_code, detail),
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise ConversationMemoryError("memory_model_invalid_response", "对话记忆模型返回的不是 JSON") from exc

    choices = body.get("choices") if isinstance(body, Mapping) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ConversationMemoryError("memory_model_invalid_response", "对话记忆模型返回缺少 choices")
    return _message_content(choices[0].get("message"))


def _parse_strict_json(raw_output: str, schema: Mapping[str, Any], *, code: str) -> Dict[str, Any]:
    try:
        payload = json.loads(raw_output.strip())
    except (TypeError, ValueError) as exc:
        raise ConversationMemoryError(code, "对话记忆模型输出不是完整 JSON") from exc
    if not isinstance(payload, Mapping):
        raise ConversationMemoryError(code, "对话记忆模型输出必须是 JSON 对象")
    required = set(schema.get("required", []))
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        detail = []
        if missing:
            detail.append("缺少 " + ", ".join(missing))
        if extra:
            detail.append("多余 " + ", ".join(extra))
        raise ConversationMemoryError(code, "对话记忆模型字段不符合契约：%s" % "；".join(detail))
    return dict(payload)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def build_context_decision_messages(
    current_query: str,
    previous_context: Mapping[str, Any],
) -> List[Dict[str, str]]:
    user_content = "当前用户问题：%s\n上一轮结构化记忆：%s" % (
        current_query.strip(),
        _json_text(previous_context),
    )
    return [
        {"role": "system", "content": CONTEXT_DECISION_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def decide_context_relation(
    current_query: str,
    previous_context: Mapping[str, Any],
    *,
    config: Optional[LocalQwenConfig] = None,
    invoke_model: Optional[Callable[[List[Dict[str, str]]], str]] = None,
) -> Dict[str, Any]:
    """判断当前问题是否延续上一轮话题；没有记忆时不额外调用模型。"""

    if not previous_context:
        return {
            "topic_relation": "new_topic",
            "use_previous_context": False,
            "reason": "没有可用的上一轮临时记忆",
        }

    messages = build_context_decision_messages(current_query, previous_context)
    raw_output = (invoke_model or (lambda value: _invoke_local_json_model(
        value,
        config or LocalQwenConfig.from_env(),
        schema_name="context_decision",
        schema=CONTEXT_DECISION_SCHEMA,
        max_tokens=240,
    )))(messages)
    payload = _parse_strict_json(
        raw_output,
        CONTEXT_DECISION_SCHEMA,
        code="context_decision_invalid_schema",
    )
    relation = payload["topic_relation"]
    if relation not in {"follow_up", "new_topic"}:
        raise ConversationMemoryError("context_decision_invalid_schema", "topic_relation 不合法")
    if not isinstance(payload["use_previous_context"], bool):
        raise ConversationMemoryError("context_decision_invalid_schema", "use_previous_context 必须是布尔值")
    if (relation == "follow_up") != payload["use_previous_context"]:
        raise ConversationMemoryError(
            "context_decision_invalid_schema",
            "topic_relation 与 use_previous_context 不一致",
        )
    reason = payload["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ConversationMemoryError("context_decision_invalid_schema", "reason 不能为空")
    payload["reason"] = reason.strip()
    return payload


def build_memory_extraction_messages(
    user_query: str,
    assistant_answer: str,
    parsed_constraints: Mapping[str, Any],
    answer_payload: Any,
    *,
    topic_relation: str,
    next_action: str,
    previous_context: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, str]]:
    payload = {
        "user_query": user_query,
        "assistant_answer": assistant_answer,
        "parsed_constraints": dict(parsed_constraints),
        "answer_payload": answer_payload,
        "topic_relation": topic_relation,
        "next_action": next_action,
        "previous_context": previous_context or {},
    }
    return [
        {"role": "system", "content": MEMORY_EXTRACTION_SYSTEM},
        {"role": "user", "content": _json_text(payload)},
    ]


def _non_empty_text(value: Any, field_name: str, *, fallback: str = "unknown") -> str:
    if not isinstance(value, str):
        raise ConversationMemoryError("memory_model_invalid_schema", "%s 必须是字符串" % field_name)
    value = value.strip()
    return value or fallback


def _string_list(value: Any, field_name: str) -> List[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConversationMemoryError("memory_model_invalid_schema", "%s 必须是字符串数组" % field_name)
    return [item.strip() for item in value if item.strip()]


def extract_conversation_memory(
    user_query: str,
    assistant_answer: str,
    parsed_constraints: Mapping[str, Any],
    answer_payload: Any,
    *,
    effective_query: str,
    topic_relation: str,
    next_action: str,
    previous_context: Optional[Mapping[str, Any]] = None,
    config: Optional[LocalQwenConfig] = None,
    invoke_model: Optional[Callable[[List[Dict[str, str]]], str]] = None,
) -> Dict[str, Any]:
    """用模型从本轮问答提取下一轮可用的短期记忆。"""

    messages = build_memory_extraction_messages(
        user_query,
        assistant_answer,
        parsed_constraints,
        answer_payload,
        topic_relation=topic_relation,
        next_action=next_action,
        previous_context=previous_context if topic_relation == "follow_up" else {},
    )
    raw_output = (invoke_model or (lambda value: _invoke_local_json_model(
        value,
        config or LocalQwenConfig.from_env(),
        schema_name="conversation_memory",
        schema=MEMORY_EXTRACTION_SCHEMA,
        max_tokens=600,
    )))(messages)
    payload = _parse_strict_json(
        raw_output,
        MEMORY_EXTRACTION_SCHEMA,
        code="memory_model_invalid_schema",
    )

    mentioned_products: List[Dict[str, Any]] = []
    answer_items = answer_payload.get("recommendations", []) if isinstance(answer_payload, Mapping) else []
    evidence_items = answer_payload.get("evidence", []) if isinstance(answer_payload, Mapping) else []
    allowed: Dict[str, Dict[str, Any]] = {}
    previous_products = (
        previous_context.get("mentioned_products", [])
        if topic_relation == "follow_up" and isinstance(previous_context, Mapping)
        else []
    )
    for item in previous_products if isinstance(previous_products, list) else []:
        if not isinstance(item, Mapping) or not isinstance(item.get("product_id"), str):
            continue
        allowed[item["product_id"]] = {
            "product_id": item["product_id"],
            "title": item.get("title") if isinstance(item.get("title"), str) else "unknown",
            "source_ids": [
                source_id
                for source_id in item.get("source_ids", [])
                if isinstance(source_id, str) and source_id.strip()
            ],
        }
    for item in answer_items if isinstance(answer_items, list) else []:
        if not isinstance(item, Mapping) or not isinstance(item.get("product_id"), str):
            continue
        allowed[item["product_id"]] = {
            "product_id": item["product_id"],
            "title": item.get("title") if isinstance(item.get("title"), str) else "unknown",
            "source_ids": [
                source_id
                for source_id in item.get("evidence_source_ids", [])
                if isinstance(source_id, str) and source_id.strip()
            ],
        }
    for item in evidence_items if isinstance(evidence_items, list) else []:
        if not isinstance(item, Mapping) or not isinstance(item.get("product_id"), str):
            continue
        product = allowed.setdefault(
            item["product_id"],
            {
                "product_id": item["product_id"],
                "title": "unknown",
                "source_ids": [],
            },
        )
        source_id = item.get("source_id")
        if isinstance(source_id, str) and source_id.strip() and source_id not in product["source_ids"]:
            product["source_ids"].append(source_id)

    raw_products = payload["mentioned_products"]
    if not isinstance(raw_products, list):
        raise ConversationMemoryError("memory_model_invalid_schema", "mentioned_products 必须是数组")
    for item in raw_products:
        if not isinstance(item, Mapping) or not isinstance(item.get("product_id"), str):
            raise ConversationMemoryError("memory_model_invalid_schema", "mentioned_products 字段不合法")
        product = allowed.get(item["product_id"])
        if product is None:
            continue
        mentioned_products.append(dict(product))

    if not isinstance(payload["pending_clarification"], bool):
        raise ConversationMemoryError("memory_model_invalid_schema", "pending_clarification 必须是布尔值")
    return {
        "memory_version": 1,
        "active_topic": _non_empty_text(payload["active_topic"], "active_topic"),
        "user_summary": _non_empty_text(payload["user_summary"], "user_summary"),
        "answer_summary": _non_empty_text(payload["answer_summary"], "answer_summary"),
        "unresolved_question": _non_empty_text(
            payload["unresolved_question"],
            "unresolved_question",
            fallback="",
        ),
        "pending_clarification": payload["pending_clarification"],
        "mentioned_products": mentioned_products,
        "parsed_constraints": dict(parsed_constraints),
        "effective_query": effective_query,
        "topic_relation": topic_relation,
        "next_action": next_action,
    }
