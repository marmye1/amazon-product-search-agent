from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import pytest

from src.conversation_memory import (
    ConversationMemoryError,
    extract_conversation_memory,
    decide_context_relation,
)
from src.session_context import SessionContextStore


def _model_output(payload: Dict[str, Any]):
    def invoke(_: List[Dict[str, str]]) -> str:
        return json.dumps(payload, ensure_ascii=False)

    return invoke


def test_context_decision_uses_structured_follow_up_contract() -> None:
    messages: List[Dict[str, str]] = []

    def invoke(value: List[Dict[str, str]]) -> str:
        messages.extend(value)
        return json.dumps(
            {
                "topic_relation": "follow_up",
                "use_previous_context": True,
                "reason": "当前问题补充上一轮条件",
            },
            ensure_ascii=False,
        )

    result = decide_context_relation(
        "还要防水",
        {
            "active_topic": "办公电脑外设",
            "user_summary": "用户想找办公电脑外设",
            "answer_summary": "上一轮返回了相关商品",
        },
        invoke_model=invoke,
    )

    assert result["topic_relation"] == "follow_up"
    assert result["use_previous_context"] is True
    assert "办公电脑外设" in messages[-1]["content"]
    assert "上一轮返回了相关商品" in messages[-1]["content"]


def test_context_decision_rejects_relation_mismatch() -> None:
    with pytest.raises(ConversationMemoryError) as error:
        decide_context_relation(
            "跑鞋",
            {"active_topic": "鼠标"},
            invoke_model=_model_output(
                {
                    "topic_relation": "new_topic",
                    "use_previous_context": True,
                    "reason": "主题不同",
                }
            ),
        )

    assert error.value.code == "context_decision_invalid_schema"


def test_memory_extraction_keeps_only_evidence_backed_products() -> None:
    memory = extract_conversation_memory(
        "办公鼠标，要防水",
        "可以考虑这款商品。",
        {"search_terms": ["办公", "防水"], "search_terms_en": ["office", "waterproof"]},
        {
            "recommendations": [
                {
                    "product_id": "p-1",
                    "title": "Office Mouse",
                    "evidence_source_ids": ["p-1:title"],
                }
            ],
            "evidence": [
                {
                    "product_id": "p-1",
                    "source_id": "p-1:title",
                }
            ],
        },
        effective_query="office mouse waterproof",
        topic_relation="follow_up",
        next_action="search",
        previous_context={
            "active_topic": "办公鼠标",
            "mentioned_products": [
                {
                    "product_id": "p-0",
                    "title": "Previous Mouse",
                    "source_ids": ["p-0:title"],
                }
            ],
        },
        invoke_model=_model_output(
            {
                "active_topic": "办公防水鼠标",
                "user_summary": "用户要办公鼠标并补充防水条件",
                "answer_summary": "返回了一款办公鼠标",
                "unresolved_question": "",
                "pending_clarification": False,
                "mentioned_products": [
                    {"product_id": "p-1", "title": "Office Mouse", "source_ids": ["p-1:title"]},
                    {"product_id": "not-in-answer", "title": "不可引用", "source_ids": []},
                ],
            }
        ),
    )

    assert memory["active_topic"] == "办公防水鼠标"
    assert memory["parsed_constraints"]["search_terms_en"] == ["office", "waterproof"]
    assert [item["product_id"] for item in memory["mentioned_products"]] == ["p-1"]
    assert memory["topic_relation"] == "follow_up"


def test_session_context_is_scoped_and_clearable() -> None:
    store = SessionContextStore(ttl_seconds=60, max_sessions=2)
    store.put("session-a", {"pending_clarification": True, "parsed_constraints": {"search_terms": ["office"]}})

    assert store.get("session-a")["parsed_constraints"]["search_terms"] == ["office"]
    assert store.get("session-b") is None

    store.clear("session-a")
    assert store.get("session-a") is None


def test_session_context_expires_without_persisting() -> None:
    store = SessionContextStore(ttl_seconds=0.01, max_sessions=2)
    store.put("session-a", {"pending_clarification": True})
    time.sleep(0.02)

    assert store.get("session-a") is None
