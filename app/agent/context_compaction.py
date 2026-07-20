"""Deterministic helpers for compacting SQL generation context."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from typing import Any

from app.agent.state import (
    DataAgentState,
    DateInfoState,
    DBInfoState,
    MetricInfoState,
    TableInfoState,
)
from app.core.log import logger


async def compile_context_from_plan(
    state: DataAgentState, context: dict[str, Any]
) -> dict[str, Any]:
    """Compile minimal authoritative SQL context from a validated plan."""

    plan = deepcopy(state.get("semantic_plan") or {})
    if not plan:
        return _metadata_issue(
            issue_type="plan",
            reason="semantic_plan_missing",
            candidate_ids=[],
        )
    repository = context.get("meta_mysql_repository")
    if repository is None:
        return _metadata_issue(
            issue_type="repository",
            reason="metadata_repository_unavailable",
            candidate_ids=[],
        )

    try:
        required_tables = {
            str(table_id).strip().casefold()
            for table_id in plan.get("required_table_ids") or []
            if str(table_id).strip()
        }
        required_columns = {
            _normalize_column_id(column_id)
            for column_id in plan.get("required_column_ids") or []
        }
        for join in plan.get("joins") or []:
            required_columns.add(_normalize_column_id(join.get("left_column_id") or ""))
            required_columns.add(
                _normalize_column_id(join.get("right_column_id") or "")
            )
        metric_ids = {
            str(measure.get("metric_id") or "").strip()
            for measure in plan.get("measures") or []
            if str(measure.get("metric_id") or "").strip()
        }
    except AttributeError, ValueError:
        return _metadata_issue(
            issue_type="plan",
            reason="semantic_plan_dependency_invalid",
            candidate_ids=[],
        )

    metadata_columns = await repository.list_column_infos()
    metadata_by_id = {
        _normalize_column_id(str(_value(column, "id") or "")): column
        for column in metadata_columns
        if _value(column, "id")
    }
    missing_columns = required_columns - set(metadata_by_id)
    if missing_columns:
        return _metadata_issue(
            issue_type="column",
            reason="metadata_column_not_found",
            candidate_ids=sorted(missing_columns),
        )
    column_tables = {
        str(_value(metadata_by_id[column_id], "table_id") or "").strip().casefold()
        for column_id in required_columns
    }
    if not column_tables <= required_tables:
        return _metadata_issue(
            issue_type="plan",
            reason="semantic_plan_dependency_invalid",
            candidate_ids=sorted(column_tables - required_tables),
        )

    tables: list[TableInfoState] = []
    tables_by_id: dict[str, TableInfoState] = {}
    for table_id in sorted(required_tables):
        table_info = await repository.get_table_info_by_id(table_id)
        if table_info is None:
            return _metadata_issue(
                issue_type="table",
                reason="metadata_table_not_found",
                candidate_ids=[table_id],
            )
        table = _table_info_to_state(table_info)
        tables.append(table)
        tables_by_id[table_id] = table

    for column_id in sorted(required_columns):
        column_info = metadata_by_id[column_id]
        table_id = str(_value(column_info, "table_id") or "").strip().casefold()
        _upsert_metadata_column(tables_by_id[table_id], column_id, column_info)

    metadata_metrics = await repository.list_metric_infos()
    metrics_by_id = {
        str(_value(metric, "id") or ""): metric
        for metric in metadata_metrics
        if _value(metric, "id")
    }
    missing_metrics = metric_ids - set(metrics_by_id)
    if missing_metrics:
        return _metadata_issue(
            issue_type="metric",
            reason="metadata_metric_not_found",
            candidate_ids=sorted(missing_metrics),
        )
    metric_infos = [
        _metric_info_to_state(metrics_by_id[metric_id])
        for metric_id in sorted(metric_ids)
    ]
    logger.info(
        "计划编译后的表信息：{}",
        [item["name"] for item in tables],
    )
    return {"table_infos": tables, "metric_infos": metric_infos}


async def add_runtime_context(state: DataAgentState, context: dict[str, Any]):
    today = context.get("semantic_reference_date") or date.today()
    date_info = DateInfoState(
        date=today.strftime("%Y-%m-%d"),
        weekday=today.strftime("%A"),
        quarter=f"Q{(today.month - 1) // 3 + 1}",
    )

    db = await context["dw_mysql_repository"].get_db_info()
    db_info = DBInfoState(**db)
    logger.info(f"数据库信息：{db_info}")
    logger.info(f"日期信息：{date_info}")
    return {"date_info": date_info, "db_info": db_info}


def _normalize_column_id(column_id: str) -> str:
    parts = [part.strip().casefold() for part in str(column_id).split(".")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"invalid column id: {column_id!r}")
    return ".".join(parts)


def _table_id(table: TableInfoState) -> str:
    return str(table.get("name") or "").strip().casefold()


def _table_info_to_state(table_info: Any) -> TableInfoState:
    return {
        "name": str(_value(table_info, "name") or _value(table_info, "id") or ""),
        "role": str(_value(table_info, "role") or ""),
        "description": str(_value(table_info, "description") or ""),
        "columns": [],
    }


def _column_info_to_state(column_info: Any) -> dict[str, Any]:
    return {
        "name": _value(column_info, "name"),
        "type": _value(column_info, "type"),
        "role": _value(column_info, "role"),
        "description": _value(column_info, "description"),
        "alias": list(_value(column_info, "alias") or []),
        "examples": list(_value(column_info, "examples") or []),
    }


def _metric_info_to_state(metric_info: Any) -> MetricInfoState:
    return {
        "id": str(_value(metric_info, "id") or ""),
        "name": str(_value(metric_info, "name") or ""),
        "description": str(_value(metric_info, "description") or ""),
        "relevant_columns": list(_value(metric_info, "relevant_columns") or []),
        "alias": list(_value(metric_info, "alias") or []),
        "aggregation": str(_value(metric_info, "aggregation") or ""),
        "expression": _value(metric_info, "expression"),
    }


def _upsert_metadata_column(
    table: TableInfoState, column_id: str, column_info: Any
) -> None:
    columns = table.setdefault("columns", [])
    table_id = _table_id(table)
    for index, column in enumerate(columns):
        candidate_id = f"{table_id}.{str(column.get('name') or '').strip().casefold()}"
        if candidate_id == column_id:
            columns[index] = _column_info_to_state(column_info)
            return
    columns.append(_column_info_to_state(column_info))


def _metadata_issue(
    *, issue_type: str, reason: str, candidate_ids: list[str]
) -> dict[str, Any]:
    return {
        "table_infos": [],
        "issue": {
            "category": "system",
            "type": issue_type,
            "reason": reason,
            "candidate_ids": candidate_ids,
        },
    }


def _value(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


__all__ = ["add_runtime_context", "compile_context_from_plan"]
