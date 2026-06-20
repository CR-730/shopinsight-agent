"""Helpers for compacting SQL generation context."""

from datetime import date
from typing import Any

import yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from app.agent.llm import llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.agent.state import (
    DataAgentState,
    DateInfoState,
    DBInfoState,
    MetricInfoState,
    TableInfoState,
)
from app.conf.app_config import app_config
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def filter_table_context(
    state: DataAgentState, context: dict[str, Any]
) -> dict[str, list[TableInfoState]]:
    query = state["query"]
    sql_context = state.get("sql_context") or {}
    table_infos: list[TableInfoState] = sql_context.get("tables") or []
    if _ablation_options(context).get("disable_context_compaction"):
        prompt_table_infos = table_infos
    else:
        prompt_table_infos = compact_table_context_for_filtering(
            table_infos, state.get("business_binding") or {}
        )

    prompt = PromptTemplate(
        template=load_prompt("filter_table_info"),
        input_variables=["query", "table_infos"],
    )
    result = await ainvoke_llm_with_usage(
        prompt,
        llm,
        JsonOutputParser(),
        {
            "query": query,
            "table_infos": yaml.dump(
                prompt_table_infos, allow_unicode=True, sort_keys=False
            ),
        },
        "过滤表信息",
        context["cost_tracker"],
        app_config.llm.timeout_seconds,
        cacheable=not _ablation_options(context).get("disable_non_sql_llm_cache"),
    )

    filtered_table_infos: list[TableInfoState] = []
    protected_columns = _protected_binding_columns(state.get("business_binding") or {})
    table_infos = await _ensure_protected_columns(table_infos, protected_columns, context)
    for table_info in table_infos:
        selected_columns = set(result.get(table_info["name"]) or [])
        selected_columns.update(
            column_id.partition(".")[2]
            for column_id in protected_columns
            if column_id.startswith(f"{table_info['name']}.")
        )
        if selected_columns:
            table_info["columns"] = [
                column_info
                for column_info in table_info["columns"]
                if column_info["name"] in selected_columns
            ]
            filtered_table_infos.append(table_info)

    logger.info(f"过滤后的表信息：{[item['name'] for item in filtered_table_infos]}")
    return {"table_infos": filtered_table_infos}


def filter_metric_context(state: DataAgentState) -> dict[str, list[MetricInfoState]]:
    sql_context = state.get("sql_context") or {}
    metric_infos: list[MetricInfoState] = sql_context.get("metrics") or []
    business_binding = state.get("business_binding") or {}
    bound_metric_names = {
        binding["canonical_metric"]
        for binding in business_binding.get("metrics") or []
        if binding.get("canonical_metric")
    }

    if bound_metric_names:
        filtered_metric_infos = [
            metric_info
            for metric_info in metric_infos
            if metric_info["name"] in bound_metric_names
        ]
    else:
        filtered_metric_infos = metric_infos

    logger.info(f"过滤后的指标信息：{[item['name'] for item in filtered_metric_infos]}")
    return {"metric_infos": filtered_metric_infos}


async def add_runtime_context(state: DataAgentState, context: dict[str, Any]):
    today = date.today()
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


def compact_table_context_for_filtering(
    table_infos: list[TableInfoState], business_binding: dict
) -> list[dict]:
    """Return a smaller table context without removing candidate columns."""

    if not business_binding:
        return table_infos

    return [
        {
            "name": table_info["name"],
            "role": table_info.get("role", ""),
            "columns": [
                _compact_column(column_info)
                for column_info in table_info.get("columns") or []
            ],
        }
        for table_info in table_infos
    ]


def _compact_column(column_info: dict) -> dict:
    return {
        "name": column_info["name"],
        "role": column_info.get("role", ""),
        "alias": column_info.get("alias") or [],
    }


def _protected_binding_columns(business_binding: dict) -> set[str]:
    columns: set[str] = set()
    for group in business_binding.get("groups") or []:
        column = str(group.get("column") or "").strip()
        if column:
            columns.add(column)
    for metric in business_binding.get("metrics") or []:
        columns.update(str(item) for item in metric.get("relevant_columns") or [] if item)
    for item in business_binding.get("filters") or []:
        column = str(item.get("column") or "").strip()
        if column:
            columns.add(column)
    time_binding = business_binding.get("time") or {}
    columns.update(str(item) for item in time_binding.get("required_columns") or [] if item)
    return columns


async def _ensure_protected_columns(
    table_infos: list[TableInfoState],
    protected_columns: set[str],
    context: dict[str, Any],
) -> list[TableInfoState]:
    missing_columns = _missing_protected_columns(table_infos, protected_columns)
    if not missing_columns or "meta_mysql_repository" not in context:
        return table_infos

    column_infos = await context["meta_mysql_repository"].list_column_infos()
    column_by_id = {getattr(column_info, "id", ""): column_info for column_info in column_infos}
    table_by_name = {table_info["name"]: table_info for table_info in table_infos}
    for column_id in missing_columns:
        column_info = column_by_id.get(column_id)
        if not column_info:
            continue
        table_name, _, _ = column_id.partition(".")
        table_info = table_by_name.get(table_name)
        if not table_info:
            continue
        table_info.setdefault("columns", []).append(_column_info_to_state(column_info))
    return table_infos


def _missing_protected_columns(
    table_infos: list[TableInfoState],
    protected_columns: set[str],
) -> set[str]:
    existing = {
        f"{table_info['name']}.{column_info['name']}"
        for table_info in table_infos
        for column_info in table_info.get("columns") or []
    }
    return protected_columns - existing


def _column_info_to_state(column_info) -> dict[str, Any]:
    return {
        "name": column_info.name,
        "type": column_info.type,
        "role": column_info.role,
        "description": column_info.description,
        "alias": list(column_info.alias or []),
        "examples": list(column_info.examples or []),
    }


def _ablation_options(context: dict[str, Any]) -> dict[str, Any]:
    return dict(context.get("ablation_options") or {})
