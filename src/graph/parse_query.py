"""需求解析节点：自然语言问题 -> 结构化商品约束。"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional

import requests
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from ..generate_recommendation import LocalQwenConfig
from ..models import ContractError
from .state import AgentState, ParsedConstraints


class QueryParseError(ContractError):
    """需求解析失败。"""


PARSE_QUERY_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "category_en": {"type": "string"},
        "brand": {"type": "string"},
        "brand_en": {"type": "string"},
        "use_case": {"type": "string"},
        "use_case_en": {"type": "string"},
        "must_have": {"type": "array", "items": {"type": "string"}},
        "must_have_en": {"type": "array", "items": {"type": "string"}},
        "avoid": {"type": "array", "items": {"type": "string"}},
        "avoid_en": {"type": "array", "items": {"type": "string"}},
        "locale": {"type": "string"},
        "search_terms": {"type": "array", "items": {"type": "string"}},
        "search_terms_en": {"type": "array", "items": {"type": "string"}},
        "retrieval_query": {"type": "string"},
        "in_scope": {"type": "boolean"},
        "needs_clarification": {"type": "boolean"},
        "clarification_reason": {"type": "string"},
    },
    "required": [
        "category",
        "category_en",
        "brand",
        "brand_en",
        "use_case",
        "use_case_en",
        "must_have",
        "must_have_en",
        "avoid",
        "avoid_en",
        "locale",
        "search_terms",
        "search_terms_en",
        "retrieval_query",
        "in_scope",
        "needs_clarification",
        "clarification_reason",
    ],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """你是亚马逊商品检索 Agent 的需求解析器。
只从当前用户问题和已经标记为同一话题的上一轮结构化记忆中提取约束，不要根据常识补充用户没有说过的品牌、类目、用途或规格。
上一轮答案摘要只能用于解析“它、这个、第一个、刚才”等指代，不能把答案中的未验证内容当成当前商品事实。
没有明确出现的单值字段必须填 unknown，没有明确出现的数组字段必须返回空数组。
in_scope 只有在用户请求商品搜索、商品比较或购买建议时为 true。
如果缺少商品类型、预算或其他可选条件，needs_clarification 可以为 true，但这不代表禁止检索。
search_terms 保留用户明确表达的商品词、功能词和场景词，不要生成新的查询词。
must_have 只记录用户明确使用“必须、一定要、需要支持”等表达的硬性条件；普通功能词和场景词不要放入 must_have。
category_en、brand_en、use_case_en、must_have_en、avoid_en、search_terms_en 是同一字段的英文检索表达：
只翻译用户已经明确说出的内容，不得增加新的条件；品牌名、型号和数字优先保留可检索的英文形式。
retrieval_query 是只用于 BM25 和向量检索的英文查询词；中文问题必须翻译成英文，英文问题原样保留。
如果请求没有可检索的商品语义（例如“帮我推荐一个东西”），retrieval_query 和英文数组可以为空。
如果超出商品检索范围，in_scope 为 false；如果无法确定原因，clarification_reason 填 unknown。
只返回符合要求的 JSON 对象。"""

USER_PROMPT = """当前有效用户问题：{user_query}

上一轮模型提取的对话记忆（如果为空则不要使用）：{conversation_context}

请返回以下字段：
category、category_en、brand、brand_en、use_case、use_case_en、
must_have、must_have_en、avoid、avoid_en、locale、search_terms、search_terms_en、retrieval_query、
in_scope、needs_clarification、clarification_reason。"""


PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
)
_JSON_PARSER = JsonOutputParser()
_REQUIRED_FIELDS = tuple(PARSE_QUERY_JSON_SCHEMA["required"])
_GENERIC_QUERY_KEYS = {
    "我想买一个东西",
    "我想买东西",
    "帮我推荐一个",
    "帮我推荐一个东西",
    "推荐一个东西",
    "推荐商品",
    "我想要一个",
    "我想要一个东西",
    "给我一个",
    "给我一个东西",
    "随便推荐",
    "recommend something",
    "help me choose",
    "recommend",
    "suggest",
}
_ACTION_ONLY_TERMS = {
    "帮我",
    "推荐",
    "推荐一个",
    "推荐商品",
    "想买",
    "我想买",
    "我想要",
    "一个",
    "东西",
    "商品",
    "recommend",
    "recommendation",
    "suggest",
    "suggestion",
    "help",
    "choose",
    "something",
    "product",
}


def build_parse_query_messages(
    user_query: str,
    conversation_context: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, str]]:
    """生成需求解析节点发送给模型的消息。"""

    prompt_value = PROMPT_TEMPLATE.invoke(
        {
            "user_query": user_query,
            "conversation_context": json.dumps(
                conversation_context or {},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
        }
    )
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    return [
        {"role": role_map.get(message.type, message.type), "content": str(message.content)}
        for message in prompt_value.to_messages()
    ]


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
        raise QueryParseError("query_parse_invalid_response", "需求解析模型没有返回文本")
    return content.strip()


def _invoke_local_qwen_parser(
    messages: List[Dict[str, str]],
    config: LocalQwenConfig,
) -> str:
    """调用本地 Qwen 的结构化需求解析接口。"""

    payload: Dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 600,
        "stream": False,
        "reasoning_effort": config.reasoning_effort,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "parsed_constraints",
                "schema": PARSE_QUERY_JSON_SCHEMA,
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
        raise QueryParseError("query_parse_timeout", "需求解析模型请求超时") from exc
    except requests.ConnectionError as exc:
        raise QueryParseError("query_parse_unavailable", "无法连接需求解析模型") from exc
    except requests.RequestException as exc:
        raise QueryParseError("query_parse_request_error", "需求解析模型请求失败") from exc

    if response.status_code >= 400:
        detail = response.text.strip().replace("\n", " ")[:300]
        raise QueryParseError(
            "query_parse_http_error",
            "需求解析模型返回 HTTP %s: %s" % (response.status_code, detail),
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise QueryParseError("query_parse_invalid_response", "需求解析模型返回的不是合法 JSON") from exc

    choices = body.get("choices") if isinstance(body, Mapping) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise QueryParseError("query_parse_invalid_response", "需求解析模型返回缺少 choices")
    return _message_content(choices[0].get("message"))


def _string_list(value: Any, field_name: str) -> List[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise QueryParseError("query_parse_invalid_schema", "%s 必须是字符串数组" % field_name)
    return [item.strip() for item in value if item.strip()]


def _normalise_query_key(value: str) -> str:
    return "".join(char for char in value.casefold() if char.isalnum() or "\u4e00" <= char <= "\u9fff")


def _has_known_constraint(payload: Mapping[str, Any]) -> bool:
    for field_name in ("category", "brand", "use_case"):
        value = payload.get(field_name)
        if isinstance(value, str) and value.strip().casefold() not in {"", "unknown", "null", "none", "n/a"}:
            return True
    for field_name in ("must_have", "avoid", "search_terms"):
        raw_values = payload.get(field_name)
        if not isinstance(raw_values, list):
            continue
        for value in raw_values:
            if not isinstance(value, str):
                continue
            key = _normalise_query_key(value)
            if key and key not in {_normalise_query_key(item) for item in _ACTION_ONLY_TERMS}:
                return True
    return False


def is_generic_request(user_query: str) -> bool:
    key = _normalise_query_key(user_query.strip())
    generic_keys = {_normalise_query_key(item) for item in _GENERIC_QUERY_KEYS}
    if key in generic_keys:
        return True
    # 对“帮我推荐 / 我想要一个 / 给我一个”这类没有商品语义的短句做
    # 确定性拦截，避免模型把“推荐”误当成可检索商品词；带有“鼠标”等
    # 商品词的请求不会命中这里。
    generic_prefixes = (
        "帮我推荐",
        "推荐",
        "我想买",
        "我想要",
        "给我一个",
        "给我",
        "随便推荐",
        "recommend",
        "suggest",
        "helpmechoose",
    )
    generic_suffixes = {
        "",
        "一个",
        "一个东西",
        "东西",
        "商品",
        "一下",
        "aproduct",
        "something",
    }
    for prefix in generic_prefixes:
        normalized_prefix = _normalise_query_key(prefix)
        if key.startswith(normalized_prefix) and key[len(normalized_prefix) :] in generic_suffixes:
            return True
    return False


def retrieval_eligible(user_query: str, payload: Mapping[str, Any]) -> bool:
    """判断是否已有足够的商品相关语义进入宽泛检索。

    这不是相关性分数阈值，也不要求 category 存在；功能词、场景词和原始问题
    都可以作为检索入口。只有明显没有任何可操作语义的泛化请求才先澄清。
    """

    if payload.get("in_scope") is not True:
        return False
    if is_generic_request(user_query):
        return False
    if _has_known_constraint(payload):
        return True
    key = _normalise_query_key(user_query.strip())
    return bool(key) and len(key) > 1


def _validate_constraints(payload: Mapping[str, Any]) -> ParsedConstraints:
    missing = [field_name for field_name in _REQUIRED_FIELDS if field_name not in payload]
    if missing:
        raise QueryParseError(
            "query_parse_missing_field",
            "需求解析缺少字段: %s" % ", ".join(missing),
        )

    text_fields = (
        "category",
        "category_en",
        "brand",
        "brand_en",
        "use_case",
        "use_case_en",
        "locale",
        "retrieval_query",
        "clarification_reason",
    )
    for field_name in text_fields:
        if not isinstance(payload[field_name], str):
            raise QueryParseError("query_parse_invalid_schema", "%s 必须是字符串" % field_name)
    for field_name in (
        "must_have",
        "must_have_en",
        "avoid",
        "avoid_en",
        "search_terms",
        "search_terms_en",
    ):
        _string_list(payload[field_name], field_name)
    for field_name in ("in_scope", "needs_clarification"):
        if not isinstance(payload[field_name], bool):
            raise QueryParseError("query_parse_invalid_schema", "%s 必须是布尔值" % field_name)

    locale = payload["locale"].strip().lower() or "unknown"
    return {
        "category": payload["category"].strip() or "unknown",
        "category_en": payload["category_en"].strip() or "unknown",
        "brand": payload["brand"].strip() or "unknown",
        "brand_en": payload["brand_en"].strip() or "unknown",
        "use_case": payload["use_case"].strip() or "unknown",
        "use_case_en": payload["use_case_en"].strip() or "unknown",
        "must_have": _string_list(payload["must_have"], "must_have"),
        "must_have_en": _string_list(payload["must_have_en"], "must_have_en"),
        "avoid": _string_list(payload["avoid"], "avoid"),
        "avoid_en": _string_list(payload["avoid_en"], "avoid_en"),
        "locale": locale,
        "search_terms": _string_list(payload["search_terms"], "search_terms"),
        "search_terms_en": _string_list(payload["search_terms_en"], "search_terms_en"),
        "retrieval_query": payload["retrieval_query"].strip(),
        "in_scope": payload["in_scope"],
        "needs_clarification": payload["needs_clarification"],
        "clarification_reason": payload["clarification_reason"].strip() or "unknown",
    }


def parse_query(
    user_query: str,
    *,
    conversation_context: Optional[Mapping[str, Any]] = None,
    config: Optional[LocalQwenConfig] = None,
    invoke_model: Optional[Callable[[List[Dict[str, str]]], str]] = None,
) -> ParsedConstraints:
    """解析用户问题；模型输出不符合结构时直接失败，不猜测约束。"""

    if not isinstance(user_query, str) or not user_query.strip():
        raise QueryParseError("invalid_query", "user_query 不能为空")
    llm_config = config or LocalQwenConfig.from_env()
    messages = build_parse_query_messages(user_query.strip(), conversation_context)
    runnable = RunnableLambda(
        invoke_model
        or (lambda input_messages: _invoke_local_qwen_parser(input_messages, llm_config))
    )
    try:
        raw_output = runnable.invoke(messages)
    except QueryParseError:
        raise
    except Exception as exc:
        raise QueryParseError("query_parse_failed", "需求解析节点执行失败") from exc
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise QueryParseError("query_parse_invalid_response", "需求解析模型输出为空")

    try:
        parsed = _JSON_PARSER.parse(raw_output.strip())
        strict_parsed = json.loads(raw_output.strip())
    except Exception as exc:
        raise QueryParseError("query_parse_invalid_json", "需求解析输出不是完整 JSON") from exc
    if parsed != strict_parsed or not isinstance(strict_parsed, Mapping):
        raise QueryParseError("query_parse_invalid_json", "需求解析输出不是完整 JSON 对象")
    constraints = _validate_constraints(strict_parsed)
    constraints["retrieval_eligible"] = retrieval_eligible(user_query.strip(), constraints)
    return constraints


def parse_query_node(
    state: AgentState,
    *,
    config: Optional[LocalQwenConfig] = None,
    invoke_model: Optional[Callable[[List[Dict[str, str]]], str]] = None,
) -> Dict[str, Any]:
    """LangGraph parse_query 节点。"""

    errors = list(state.get("errors", []))
    try:
        constraints = parse_query(
            state.get("effective_query") or state.get("user_query", ""),
            conversation_context=state.get("conversation_context"),
            config=config,
            invoke_model=invoke_model,
        )
    except QueryParseError as exc:
        errors.append(str(exc))
        return {"parsed_constraints": {}, "next_action": "reject", "errors": errors}
    result: Dict[str, Any] = {"parsed_constraints": constraints, "errors": errors}
    if constraints:
        result["active_query"] = constraints.get("retrieval_query") or state.get("effective_query") or state.get("user_query", "")
    return result
