"""当前 Agent 进程内的短期会话上下文。

只保存模型提取的当前对话主题、问题摘要、答案摘要、结构化条件和商品证据引用，不写入数据库，也不保存模型隐藏推理。
进程重启、会话过期或上下文被清除后，这些内容都会消失。
"""

from __future__ import annotations

import time
from threading import RLock
from typing import Any, Dict, Mapping, Optional


class SessionContextStore:
    """有 TTL 和数量上限的进程内会话上下文存储。"""

    def __init__(self, *, ttl_seconds: float = 1800.0, max_sessions: int = 128) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds 必须大于 0")
        if max_sessions < 1:
            raise ValueError("max_sessions 必须大于 0")
        self.ttl_seconds = float(ttl_seconds)
        self.max_sessions = int(max_sessions)
        self._items: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()

    def get(self, session_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """返回尚未过期的上下文副本；没有 session_id 时不启用上下文。"""

        key = self._normalise_id(session_id)
        if key is None:
            return None
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            if now - float(item["saved_at"]) >= self.ttl_seconds:
                self._items.pop(key, None)
                return None
            item["saved_at"] = now
            return self._copy(item["context"])

    def put(self, session_id: Optional[str], context: Mapping[str, Any]) -> None:
        """保存当前轮的模型提取记忆，不保存完整 Prompt 或隐藏思维链。"""

        key = self._normalise_id(session_id)
        if key is None:
            return
        now = time.monotonic()
        with self._lock:
            self._items[key] = {"saved_at": now, "context": self._copy(context)}
            self._evict_oldest_if_needed()

    def clear(self, session_id: Optional[str]) -> None:
        key = self._normalise_id(session_id)
        if key is None:
            return
        with self._lock:
            self._items.pop(key, None)

    @staticmethod
    def _normalise_id(session_id: Optional[str]) -> Optional[str]:
        if not isinstance(session_id, str):
            return None
        value = session_id.strip()
        return value or None

    @classmethod
    def _copy(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): cls._copy(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._copy(item) for item in value]
        if isinstance(value, tuple):
            return [cls._copy(item) for item in value]
        return value

    def _evict_oldest_if_needed(self) -> None:
        while len(self._items) > self.max_sessions:
            oldest = min(self._items.items(), key=lambda pair: float(pair[1]["saved_at"]))[0]
            self._items.pop(oldest, None)
