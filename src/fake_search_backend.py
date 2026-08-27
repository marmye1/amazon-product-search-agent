"""仅供单元测试的固定响应后端，不实现产品级 BM25。"""

from __future__ import annotations

from typing import Any, Mapping


class FakeSearchBackend:
    """用预置 JSON 响应测试查询和结果契约。"""

    def __init__(self, response: Mapping[str, Any]) -> None:
        self.response = response
        self.calls = []

    def search(self, index_name: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append({"index_name": index_name, "body": body})
        return self.response
