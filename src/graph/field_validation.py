"""需求解析后的字段归一化和检索资格校验。"""

from __future__ import annotations

from typing import Any, Dict

from .parse_query import retrieval_eligible
from .state import AgentState


def validate_fields_node(state: AgentState) -> Dict[str, Any]:
    """确认解析结果可供后续路由使用，不为 unknown 字段猜测值。"""

    constraints = state.get("parsed_constraints")
    if not isinstance(constraints, dict) or not constraints:
        return {
            "field_validation_status": "failed",
            "errors": list(state.get("errors", [])) + ["结构化字段为空，无法继续路由"],
        }

    normalized = dict(constraints)
    must_have = {
        str(item).strip().casefold()
        for item in normalized.get("must_have", [])
        if isinstance(item, str) and item.strip()
    }
    avoid = {
        str(item).strip().casefold()
        for item in normalized.get("avoid", [])
        if isinstance(item, str) and item.strip()
    }
    conflicts = sorted(must_have & avoid)
    if conflicts:
        normalized["constraint_conflicts"] = conflicts
    normalized["retrieval_eligible"] = retrieval_eligible(
        state.get("effective_query") or state.get("user_query", ""),
        normalized,
    )
    return {
        "parsed_constraints": normalized,
        "field_validation_status": "passed",
    }
