"""Conversation evaluation loading, SSE parsing, and assertions."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_conversation_eval_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load multi-turn conversation evaluation cases."""

    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []


def parse_sse_message(message: str) -> dict[str, Any] | None:
    """Parse one SSE message emitted by QueryService."""

    payload = (
        "\n".join(
            line.replace("data:", "", 1).strip()
            for line in message.splitlines()
            if line.startswith("data:")
        )
        .strip()
    )
    return json.loads(payload) if payload else None


async def collect_query_events(
    query_service: Any,
    query: str,
    conversation_id: str | None,
    user_id: str | None,
) -> list[dict[str, Any]]:
    """Run one turn through QueryService and collect parsed SSE events."""

    events = []
    async for message in query_service.query(
        query=query,
        conversation_id=conversation_id,
        user_id=user_id,
        include_trace=True,
    ):
        event = parse_sse_message(message)
        if event is not None:
            events.append(event)
    return events


async def evaluate_conversation_case(
    case: dict[str, Any],
    query_service: Any,
    memory_repository: Any,
    default_user_id: str | None = None,
) -> dict[str, Any]:
    """Run and evaluate one multi-turn conversation case."""

    failures: list[dict[str, Any]] = []
    turns_payload = []
    conversation_id = None
    snapshots_by_turn: dict[int, dict[str, Any] | None] = {}
    default_user_id = case.get("user_id") or default_user_id or "conversation-eval"

    for turn_index, turn in enumerate(case.get("turns", []), start=1):
        supplied_conversation_id = _supplied_conversation_id(turn, conversation_id)
        user_id = turn.get("user_id", default_user_id)
        snapshot_before = await _get_snapshot(
            memory_repository, supplied_conversation_id, user_id
        )
        events = await collect_query_events(
            query_service=query_service,
            query=turn["query"],
            conversation_id=supplied_conversation_id,
            user_id=user_id,
        )
        conversation_event = _latest_event(events, "conversation") or {}
        trace_event = _latest_event(events, "trace") or {"data": {}}
        actual_conversation_id = conversation_event.get("conversation_id")
        if actual_conversation_id:
            conversation_id = actual_conversation_id

        snapshot_after = await _get_snapshot(
            memory_repository, actual_conversation_id, user_id
        )
        snapshots_by_turn[turn_index] = deepcopy(snapshot_after)

        failures.extend(
            _evaluate_turn(
                case_id=case.get("id", ""),
                turn_index=turn_index,
                turn=turn,
                conversation_id_before=supplied_conversation_id,
                conversation_id_after=actual_conversation_id,
                rewritten_query=conversation_event.get("rewritten_query") or "",
                trace=trace_event.get("data") or {},
                snapshot_before=snapshot_before,
                snapshot_after=snapshot_after,
                snapshots_by_turn=snapshots_by_turn,
            )
        )
        turns_payload.append(
            {
                "turn_index": turn_index,
                "query": turn["query"],
                "conversation_id": actual_conversation_id,
                "rewritten_query": conversation_event.get("rewritten_query"),
                "trace": trace_event.get("data") or {},
                "snapshot_before": snapshot_before,
                "snapshot_after": snapshot_after,
                "events": events,
            }
        )

    return {
        "case_id": case.get("id"),
        "passed": not failures,
        "failures": failures,
        "turns": turns_payload,
    }


def _supplied_conversation_id(
    turn: dict[str, Any], previous_conversation_id: str | None
) -> str | None:
    supplied = turn.get("supplied_conversation_id")
    if supplied == "previous":
        return previous_conversation_id
    return supplied if supplied is not None else previous_conversation_id


async def _get_snapshot(
    memory_repository: Any, conversation_id: str | None, user_id: str | None
) -> dict[str, Any] | None:
    if not conversation_id:
        return None
    return await memory_repository.get_snapshot(conversation_id, user_id)


def _latest_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("type") == event_type:
            return event
    return None


def _evaluate_turn(
    case_id: str,
    turn_index: int,
    turn: dict[str, Any],
    conversation_id_before: str | None,
    conversation_id_after: str | None,
    rewritten_query: str,
    trace: dict[str, Any],
    snapshot_before: dict[str, Any] | None,
    snapshot_after: dict[str, Any] | None,
    snapshots_by_turn: dict[int, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    failures.extend(
        _evaluate_conversation_id(
            case_id,
            turn_index,
            turn.get("expected_conversation_id"),
            conversation_id_before,
            conversation_id_after,
        )
    )
    failures.extend(
        _evaluate_rewritten_query(
            case_id,
            turn_index,
            turn.get("expected_rewritten") or {},
            rewritten_query,
            turn.get("query") or "",
        )
    )
    failures.extend(
        _evaluate_trace(
            case_id,
            turn_index,
            turn.get("expected_trace") or {},
            trace,
        )
    )
    failures.extend(
        _evaluate_memory(
            case_id,
            turn_index,
            turn.get("expected_memory") or {},
            snapshot_before,
            snapshot_after,
            snapshots_by_turn,
        )
    )
    return failures


def _evaluate_conversation_id(
    case_id: str,
    turn_index: int,
    expected: str | None,
    before: str | None,
    after: str | None,
) -> list[dict[str, Any]]:
    if expected == "same" and before and after != before:
        return [_failure(case_id, turn_index, "conversation_id_changed")]
    if expected == "new" and before and after == before:
        return [_failure(case_id, turn_index, "conversation_id_not_recreated")]
    if after is None:
        return [_failure(case_id, turn_index, "missing_conversation_id")]
    return []


def _evaluate_rewritten_query(
    case_id: str,
    turn_index: int,
    expected: dict[str, Any],
    rewritten_query: str,
    original_query: str = "",
) -> list[dict[str, Any]]:
    failures = []
    normalized_rewritten = _normalize_text(rewritten_query)
    normalized_original = _normalize_text(original_query)
    if expected.get("mode") == "unchanged" and normalized_rewritten != normalized_original:
        failures.append(_failure(case_id, turn_index, "rewritten_query_changed"))
    if expected.get("mode") == "contextualized":
        if normalized_rewritten == normalized_original:
            failures.append(
                _failure(case_id, turn_index, "rewritten_query_not_contextualized")
            )
        if not _has_contextual_signal(rewritten_query, original_query, expected):
            failures.append(
                _failure(case_id, turn_index, "rewritten_query_missing_context")
            )
    for fragment in expected.get("contains") or []:
        if fragment not in rewritten_query:
            failures.append(
                _failure(case_id, turn_index, "missing_rewritten_fragment", fragment)
            )
    for fragment in expected.get("forbidden_contains") or []:
        if fragment in rewritten_query:
            failures.append(
                _failure(case_id, turn_index, "forbidden_rewritten_fragment", fragment)
            )
    return failures


def _evaluate_trace(
    case_id: str,
    turn_index: int,
    expected: dict[str, Any],
    trace: dict[str, Any],
) -> list[dict[str, Any]]:
    failures = []
    for metric in expected.get("metric_bindings") or []:
        if metric not in _metric_names(trace.get("metric_bindings") or []):
            failures.append(_failure(case_id, turn_index, "missing_metric", metric))
    for value in expected.get("resolved_filters") or []:
        if value not in _filter_values(trace.get("resolved_filters") or []):
            failures.append(_failure(case_id, turn_index, "missing_filter", value))
    for value in expected.get("forbidden_filters") or []:
        if value in _filter_values(trace.get("resolved_filters") or []):
            failures.append(_failure(case_id, turn_index, "forbidden_filter", value))
    if "blocked_by" in expected and trace.get("blocked_by") != expected["blocked_by"]:
        failures.append(_failure(case_id, turn_index, "blocked_by_mismatch"))
    if expected.get("final_answer") == "required" and trace.get("final_answer") is None:
        failures.append(_failure(case_id, turn_index, "missing_final_answer"))
    if expected.get("final_answer") == "absent" and trace.get("final_answer") is not None:
        failures.append(_failure(case_id, turn_index, "unexpected_final_answer"))
    if expected.get("exception_stage") and trace.get("exception_stage") != expected["exception_stage"]:
        failures.append(_failure(case_id, turn_index, "exception_stage_mismatch"))
    if expected.get("error_contains") and expected["error_contains"] not in str(
        trace.get("error") or ""
    ):
        failures.append(_failure(case_id, turn_index, "error_text_mismatch"))
    expected_time = expected.get("time_binding")
    if isinstance(expected_time, dict):
        actual_time = trace.get("time_binding") or {}
        for key, value in expected_time.items():
            if actual_time.get(key) != value:
                failures.append(_failure(case_id, turn_index, "time_binding_mismatch"))
    return failures


def _evaluate_memory(
    case_id: str,
    turn_index: int,
    expected: dict[str, Any],
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    snapshots_by_turn: dict[int, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    failures = []
    if expected.get("snapshot_source") == "isolated_empty" and before is not None:
        failures.append(
            _failure(case_id, turn_index, "snapshot_source_not_isolated_empty")
        )
    if expected.get("snapshot_write") is True and after is None:
        failures.append(_failure(case_id, turn_index, "snapshot_not_written"))
        return failures
    if expected.get("snapshot_write") is False and after != before:
        failures.append(_failure(case_id, turn_index, "snapshot_changed_unexpectedly"))
    unchanged_turn = expected.get("snapshot_unchanged_from_turn")
    if (
        unchanged_turn
        and int(unchanged_turn) in snapshots_by_turn
        and after != snapshots_by_turn.get(int(unchanged_turn))
    ):
        failures.append(_failure(case_id, turn_index, "snapshot_not_preserved"))
    if after:
        expected_metric = expected.get("snapshot_metric")
        if expected_metric and expected_metric not in _metric_names(
            after.get("last_metric_bindings") or []
        ):
            failures.append(_failure(case_id, turn_index, "snapshot_metric_mismatch"))
        for value in expected.get("snapshot_filters") or []:
            if value not in _filter_values(after.get("last_resolved_filters") or []):
                failures.append(_failure(case_id, turn_index, "snapshot_filter_mismatch"))
        expected_time = expected.get("snapshot_time")
        if isinstance(expected_time, dict):
            actual_time = after.get("last_time_binding") or {}
            for key, value in expected_time.items():
                if actual_time.get(key) != value:
                    failures.append(
                        _failure(case_id, turn_index, "snapshot_time_mismatch")
                    )
    return failures


def _metric_names(bindings: list[dict[str, Any]]) -> list[str]:
    return [
        str(item.get("canonical_metric") or item.get("raw_mention") or "")
        for item in bindings
    ]


def _filter_values(filters: list[dict[str, Any]]) -> list[str]:
    return [
        str(item.get("canonical_value") or item.get("raw_value") or "")
        for item in filters
    ]


def _failure(
    case_id: str, turn_index: int, code: str, detail: Any | None = None
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "turn_index": turn_index,
        "code": code,
        "detail": detail,
    }


def _normalize_text(value: str) -> str:
    return " ".join(str(value).split())


def _has_contextual_signal(
    rewritten_query: str, original_query: str, expected: dict[str, Any]
) -> bool:
    if "基于上一轮条件" in rewritten_query or "上一轮" in rewritten_query:
        return True
    return any(
        fragment in rewritten_query and fragment not in original_query
        for fragment in expected.get("contains") or []
    )
