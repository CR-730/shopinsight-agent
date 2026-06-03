"""SQL correction helpers used by the sql_executor graph node."""

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from app.agent.llm import correct_sql_llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.agent.sql.sql_guard import (
    normalize_sql_for_execution,
    repair_invalid_join_relationship,
)
from app.agent.sql_loop import DEFAULT_MAX_SQL_CORRECTION_ATTEMPTS
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def correct_sql_candidate(state: DataAgentState, context: dict):
    correction_attempts = state.get("correction_attempts", 0) + 1
    sql = state["sql"]

    repaired_sql = repair_invalid_join_relationship(state, sql)
    if repaired_sql and not is_same_sql_after_normalization(sql, repaired_sql):
        logger.info(f"基于元数据关系校正 SQL：{repaired_sql}")
        return {"sql": repaired_sql, "correction_attempts": correction_attempts}

    result = await ainvoke_llm_with_usage(
        PromptTemplate(
            template=load_prompt("correct_sql"),
            input_variables=[
                "table_infos",
                "metric_infos",
                "date_info",
                "db_info",
                "query",
                "sql",
                "error",
            ],
        ),
        correct_sql_llm,
        StrOutputParser(),
        {
            "table_infos": yaml.dump(
                state["table_infos"], allow_unicode=True, sort_keys=False
            ),
            "metric_infos": yaml.dump(
                state["metric_infos"], allow_unicode=True, sort_keys=False
            ),
            "date_info": yaml.dump(
                state["date_info"], allow_unicode=True, sort_keys=False
            ),
            "db_info": yaml.dump(state["db_info"], allow_unicode=True, sort_keys=False),
            "query": state["query"],
            "sql": sql,
            "error": state["error"],
        },
        "校正SQL",
        context["cost_tracker"],
        app_config.llm.timeout_seconds,
        cacheable=False,
    )

    logger.info(f"校正后的 SQL：{result}")
    if is_same_sql_after_normalization(sql, result):
        max_attempts = int(
            state.get("max_correction_attempts")
            or DEFAULT_MAX_SQL_CORRECTION_ATTEMPTS
        )
        logger.warning("SQL 修正结果与原 SQL 相同，停止无效修正循环")
        return {
            "sql": result,
            "error": "SQL 修正无效：修正后 SQL 与原 SQL 相同",
            "correction_attempts": max(correction_attempts, max_attempts),
        }

    return {"sql": result, "correction_attempts": correction_attempts}


def is_same_sql_after_normalization(original_sql: str, corrected_sql: str) -> bool:
    return normalize_sql_for_execution(original_sql).lower() == normalize_sql_for_execution(
        corrected_sql
    ).lower()
