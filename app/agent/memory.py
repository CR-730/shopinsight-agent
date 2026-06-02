"""Conversation memory helpers for multi-turn analytics."""

from __future__ import annotations

from typing import Any, TypedDict


class ConversationSnapshot(TypedDict, total=False):
    """Persisted state used to rewrite follow-up questions."""

    last_metric_bindings: list[dict[str, Any]]
    last_resolved_filters: list[dict[str, Any]]
    last_time_binding: dict[str, Any] | None
    last_sql: str | None
    last_answer_summary: str | None
    recent_turns_summary: list[dict[str, Any]]


def build_answer_summary(answer: Any) -> str:
    """Build a compact, non-sensitive summary of a SQL result."""

    if not isinstance(answer, list):
        return "未返回表格结果"
    if not answer:
        return "返回 0 行"

    first_row = answer[0]
    if isinstance(first_row, dict):
        columns = ", ".join(str(key) for key in first_row.keys())
        return f"返回 {len(answer)} 行，字段：{columns}"
    return f"返回 {len(answer)} 行"


def build_snapshot_from_state(
    state: dict[str, Any],
) -> ConversationSnapshot | None:
    """Create a snapshot from a completed graph state."""

    metric_bindings = list(
        state.get("metric_bindings")
        or (state.get("business_binding") or {}).get("metrics")
        or []
    )
    if (
        state.get("blocked_by")
        or state.get("safety_error")
        or state.get("error")
        or state.get("exception_stage")
        or state.get("unresolved_bindings")
        or state.get("ambiguous_bindings")
        or state.get("final_answer") is None
        or not metric_bindings
    ):
        return None

    return {
        "last_metric_bindings": metric_bindings,
        "last_resolved_filters": list(state.get("resolved_filters") or []),
        "last_time_binding": state.get("time_binding"),
        "last_sql": state.get("sql"),
        "last_answer_summary": build_answer_summary(state.get("final_answer")),
        "recent_turns_summary": [],
    }
