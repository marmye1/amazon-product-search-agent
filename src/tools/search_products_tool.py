"""唯一商品搜索工具：只封装 SearchProducts。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..models import ContractError, SearchRequest, SearchResponse
from ..search_products import search_products


@dataclass(frozen=True)
class SearchProductsTool:
    """单 Agent 可调用的唯一工具，不在工具内调用 LLM 或其他 Agent。"""

    client: Any
    search_config: Mapping[str, Any]
    name: str = "search_products"

    def invoke(self, request: SearchRequest) -> SearchResponse:
        if not isinstance(request, SearchRequest):
            raise ContractError("invalid_tool_input", "search_products 工具输入必须是 SearchRequest")
        return search_products(request, self.client, self.search_config)
