""" 结构化日志和请求追踪。

默认只记录摘要、哈希和节点路径，不记录完整用户问题、Prompt 或账号密码。
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Optional


LOGGER = logging.getLogger("amazon_retrieval_agent")


def new_trace_id() -> str:
    return uuid.uuid4().hex


def query_fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(message)s")
    LOGGER.setLevel(getattr(logging, level.upper(), logging.INFO))


def log_event(
    event: str,
    *,
    trace_id: str,
    fields: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "trace_id": trace_id,
    }
    if fields:
        payload.update(dict(fields))
    LOGGER.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))
