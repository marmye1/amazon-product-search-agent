"""导购回答提示词和 LangChain 消息模板。"""

from __future__ import annotations

from typing import Dict, List, Sequence

from langchain_core.prompts import ChatPromptTemplate

from ..rag_models import ContextBlock


SYSTEM_PROMPT = """你是一个严格受商品检索上下文约束的中文导购回答器。

你只能使用用户问题下方 CONTEXT 中出现的商品事实。禁止使用模型常识、外部网页或上下文之外的信息。
禁止声称实时价格、库存、评分、销量、配送、促销或任何上下文中没有出现的商品规格。
如果用户提出预算或价格条件，而 CONTEXT 没有价格字段，不能声称商品“符合预算”“低于某价格”或“满足价格要求”，必须明确说明无法验证预算。
每个推荐项必须使用 CONTEXT 中真实存在的 product_id，并且 evidence_source_ids 必须引用真实存在的 source_id。
如果证据不足，recommendations 和 evidence 必须为空，并明确说明信息不足。
只返回一个合法 JSON 对象，不要返回 Markdown、代码围栏、解释文字或思考过程。
answer 控制在 120 个汉字以内；每个 reason 控制在 80 个汉字以内；每条证据事实控制在 160 个汉字以内。
最多输出 3 个推荐项。limitations 至少输出一条非空的数据范围或实时性限制。

JSON 结构必须是：
{{
  "answer": "面向用户的中文回答",
  "recommendations": [
    {{
      "product_id": "商品 ID",
      "title": "上下文中的商品标题",
      "reason": "只根据上下文写出的推荐理由",
      "evidence_source_ids": ["商品 ID:字段名"]
    }}
  ],
  "evidence": [
    {{
      "source_id": "商品 ID:字段名",
      "product_id": "商品 ID",
      "field_name": "字段名",
      "quoted_or_paraphrased_fact": "上下文中的原文或忠实改写"
    }}
  ],
  "limitations": ["数据范围或实时性限制"],
  "grounded": false,
  "retrieval_method": "bm25",
  "answer_version": "v2"
}}

grounded 字段只按你是否使用了给定上下文填写；程序会再次检查它。"""


USER_PROMPT = """用户问题：{user_query}
回答语言：{answer_language}
最多推荐商品数：{max_products}

CONTEXT：
{context_text}

请严格按照系统规定的 JSON 结构回答。"""


PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
)


def format_context(blocks: Sequence[ContextBlock]) -> str:
    """把上下文块逐条格式化，保留 source_id 和商品边界。"""

    if not blocks:
        return "（没有可用商品上下文）"
    return "\n".join(
        "[%s] product_id=%s rank=%s field=%s text=%s"
        % (block.source_id, block.product_id, block.rank, block.field_name, block.text)
        for block in blocks
    )


def build_recommendation_messages(
    user_query: str,
    blocks: Sequence[ContextBlock],
    answer_language: str = "zh-CN",
    max_products: int = 5,
) -> List[Dict[str, str]]:
    """生成发送给本地 Qwen 的 LangChain 消息列表。"""

    prompt_value = PROMPT_TEMPLATE.invoke(
        {
            "user_query": user_query,
            "answer_language": answer_language,
            "max_products": max_products,
            "context_text": format_context(blocks),
        }
    )
    role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    return [
        {"role": role_map.get(message.type, message.type), "content": str(message.content)}
        for message in prompt_value.to_messages()
    ]
