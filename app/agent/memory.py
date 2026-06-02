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


def rewrite_followup_query(
    query: str, snapshot: ConversationSnapshot | dict[str, Any] | None
) -> str:
    """Rewrite short follow-up questions into self-contained analytics questions."""

    normalized_query = query.strip()
    if not normalized_query or not snapshot or not _looks_like_followup(normalized_query):
        return normalized_query

    context_parts = []
    metric_text = _metric_context(snapshot.get("last_metric_bindings") or [])
    if metric_text:
        context_parts.append(f"指标 {metric_text}")

    filter_text = _filter_context(snapshot.get("last_resolved_filters") or [])
    if filter_text:
        context_parts.append(f"过滤 {filter_text}")

    time_text = _time_context(snapshot.get("last_time_binding"))
    if time_text:
        context_parts.append(f"上一轮时间 {time_text}")

    if not context_parts:
        return normalized_query

    return f"基于上一轮条件：{'，'.join(context_parts)}；本轮问题：{normalized_query}"


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

    if (
        state.get("blocked_by")
        or state.get("safety_error")
        or state.get("error")
        or state.get("exception_stage")
        or state.get("final_answer") is None
    ):
        return None

    return {
        "last_metric_bindings": list(state.get("metric_bindings") or []),
        "last_resolved_filters": list(state.get("resolved_filters") or []),
        "last_time_binding": state.get("time_binding"),
        "last_sql": state.get("sql"),
        "last_answer_summary": build_answer_summary(state.get("final_answer")),
        "recent_turns_summary": [],
    }


def _looks_like_followup(query: str) -> bool:
    followup_markers = (
        "那",
        "呢",
        "再",
        "换成",
        "改成",
        "上个月",
        "下个月",
        "这个月",
        "按",
        "分",
        "继续",
    )
    if len(query) <= 12:
        return True
    return any(marker in query for marker in followup_markers)


def _metric_context(metric_bindings: list[dict[str, Any]]) -> str:
    metrics = [
        str(item.get("canonical_metric") or item.get("raw_mention") or "").strip()
        for item in metric_bindings
    ]
    return "、".join(metric for metric in metrics if metric)


def _filter_context(resolved_filters: list[dict[str, Any]]) -> str:
    filters = [
        str(item.get("canonical_value") or item.get("raw_value") or "").strip()
        for item in resolved_filters
    ]
    return "、".join(filter_value for filter_value in filters if filter_value)


def _time_context(time_binding: dict[str, Any] | None) -> str:
    if not time_binding:
        return ""
    return str(
        time_binding.get("raw_text")
        or time_binding.get("start_date")
        or time_binding.get("grain")
        or ""
    ).strip()
