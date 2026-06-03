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
    table_infos: list[TableInfoState] = state["table_infos"]
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
    )

    filtered_table_infos: list[TableInfoState] = []
    for table_info in table_infos:
        if table_info["name"] in result:
            table_info["columns"] = [
                column_info
                for column_info in table_info["columns"]
                if column_info["name"] in result[table_info["name"]]
            ]
            filtered_table_infos.append(table_info)

    logger.info(f"过滤后的表信息：{[item['name'] for item in filtered_table_infos]}")
    return {"table_infos": filtered_table_infos}


def filter_metric_context(state: DataAgentState) -> dict[str, list[MetricInfoState]]:
    metric_infos: list[MetricInfoState] = state["metric_infos"]
    bound_metric_names = {
        binding["canonical_metric"] for binding in state.get("metric_bindings") or []
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
