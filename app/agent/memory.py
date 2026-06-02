"""Conversation memory helpers for multi-turn analytics."""

from __future__ import annotations

import re
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

    metric_text = _metric_context(snapshot.get("last_metric_bindings") or [])
    filters = _filter_contexts(snapshot.get("last_resolved_filters") or [])
    filter_override = _filter_override(normalized_query, filters)
    focus_text = _followup_focus(normalized_query, filters)
    if filter_override:
        filter_index, filter_value = filter_override
        filters = [
            (
                {**filter_context, "value": filter_value}
                if index == filter_index
                else filter_context
            )
            for index, filter_context in enumerate(filters)
        ]
        focus_text = ""
    if (
        _is_condition_update(normalized_query)
        and not filter_override
        and not _looks_like_measure_focus(focus_text)
    ):
        focus_text = ""
    time_text = _time_context(snapshot.get("last_time_binding"))

    if not metric_text and not filters and not time_text:
        return normalized_query

    parts = ["统计"]
    if time_text:
        parts.append(time_text)
    parts.extend(_format_filter(filter_context) for filter_context in filters)
    parts.append(focus_text or metric_text)
    rewritten = " ".join(part for part in parts if part)

    if _has_relative_time(normalized_query):
        return f"{rewritten}，{normalized_query}"
    return rewritten


def is_followup_query(query: str) -> bool:
    """Return whether a query depends on previous conversation context."""

    normalized_query = query.strip()
    return bool(normalized_query and _looks_like_followup(normalized_query))


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


def _looks_like_followup(query: str) -> bool:
    if _looks_like_complete_query(query):
        return False
    return (
        query.startswith(("那", "再", "继续", "改成", "换成"))
        or query.endswith("呢")
        or _has_relative_time(query)
    )


def _looks_like_complete_query(query: str) -> bool:
    return query.startswith(("统计", "查询", "查看", "分析", "计算"))


def _metric_context(metric_bindings: list[dict[str, Any]]) -> str:
    metrics = [
        str(item.get("canonical_metric") or item.get("raw_mention") or "").strip()
        for item in metric_bindings
    ]
    return "、".join(metric for metric in metrics if metric)


def _filter_context(resolved_filters: list[dict[str, Any]]) -> str:
    return "、".join(
        filter_context["value"] for filter_context in _filter_contexts(resolved_filters)
    )


def _filter_contexts(resolved_filters: list[dict[str, Any]]) -> list[dict[str, str]]:
    contexts = []
    for item in resolved_filters:
        value = str(item.get("canonical_value") or item.get("raw_value") or "").strip()
        if not value:
            continue
        contexts.append(
            {
                "value": value,
                "field_alias": _filter_field_alias(item),
            }
        )
    return contexts


def _filter_field_alias(filter_context: dict[str, Any]) -> str:
    return str(filter_context.get("field_alias") or "").strip()


def _format_filter(filter_context: dict[str, str]) -> str:
    return f"{filter_context['value']}{filter_context.get('field_alias') or ''}"


def _filter_override(
    query: str, filters: list[dict[str, str]]
) -> tuple[int, str] | None:
    value = _followup_focus(query, filters, strip_alias=False)
    if not value:
        return None
    if _looks_like_measure_focus(value):
        return None

    alias_matches = [
        (index, filter_context["field_alias"])
        for index, filter_context in enumerate(filters)
        if filter_context.get("field_alias") and filter_context["field_alias"] in value
    ]
    if alias_matches:
        index, field_alias = alias_matches[0]
        clean_value = value.replace(field_alias, "").strip(" ，,;；")
        if _is_compact_slot_value(clean_value):
            return index, clean_value
        return None

    if len(filters) == 1 and _is_compact_slot_value(value):
        return 0, value
    return None


def _followup_focus(
    query: str,
    filters: list[dict[str, str]],
    *,
    strip_alias: bool = True,
) -> str:
    if _has_relative_time(query):
        return ""
    value = query.strip()
    value = re.sub(r"^(那|再|继续|改成|换成|统计|查询|查看|分析|计算)", "", value)
    value = re.sub(r"(呢|吗|吧|的)$", "", value)
    if strip_alias:
        for filter_context in filters:
            field_alias = filter_context.get("field_alias")
            if field_alias:
                value = value.replace(field_alias, "")
    value = value.strip(" ，,;；")
    if _is_compact_slot_value(value):
        return value
    return ""


def _is_condition_update(query: str) -> bool:
    return query.startswith(("改成", "换成"))


def _is_compact_slot_value(value: str) -> bool:
    return bool(value and re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9_-]{1,12}", value))


def _looks_like_measure_focus(value: str) -> bool:
    return any(token in value for token in ("价", "额", "量", "率", "数"))


def _time_context(time_binding: dict[str, Any] | None) -> str:
    if not time_binding:
        return ""
    return str(
        time_binding.get("raw_text")
        or time_binding.get("start_date")
        or time_binding.get("grain")
        or ""
    ).strip()


def _has_relative_time(query: str) -> bool:
    return any(token in query for token in ("上个月", "下个月", "这个月"))
