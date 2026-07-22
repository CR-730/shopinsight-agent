"""SQL correction helpers used by the sql_executor graph node."""

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from app.agent.llm import correct_sql_llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.agent.sql.plan_projection import project_plan_for_sql
from app.agent.sql.sql_guard import (
    normalize_sql_for_execution,
    repair_invalid_join_relationship,
)
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def correct_sql_candidate(
    state: DataAgentState,
    context: dict,
    validation_error: str,
    *,
    correction_attempts: int,
    max_correction_attempts: int,
    plan_differences: list[dict] | None = None,
):
    correction_attempts = correction_attempts + 1
    sql = state["sql"]

    repaired_sql = repair_invalid_join_relationship(state, sql)
    if repaired_sql and not is_same_sql_after_normalization(sql, repaired_sql):
        logger.info(f"基于元数据关系校正 SQL：{repaired_sql}")
        return {"sql": repaired_sql, "attempts": correction_attempts}

    result = await ainvoke_llm_with_usage(
        PromptTemplate(
            template=load_prompt("correct_sql"),
            input_variables=[
                "semantic_plan",
                "sql",
                "differences",
            ],
        ),
        correct_sql_llm,
        StrOutputParser(),
        {
            "semantic_plan": yaml.dump(
                project_plan_for_sql(state.get("semantic_plan") or {}),
                allow_unicode=True,
                sort_keys=False,
            ),
            "sql": sql,
            "differences": yaml.dump(
                plan_differences
                or [
                    {
                        "code": "sql_validation_error",
                        "message": validation_error,
                    }
                ],
                allow_unicode=True,
                sort_keys=False,
            ),
        },
        "校正SQL",
        context["cost_tracker"],
        app_config.llm.timeout_seconds,
        cacheable=False,
    )

    logger.info(f"校正后的 SQL：{result}")
    if is_same_sql_after_normalization(sql, result):
        logger.warning("SQL 修正结果与原 SQL 相同，停止无效修正循环")
        return {
            "sql": result,
            "attempts": max(correction_attempts, max_correction_attempts),
            "correction_error": "SQL 修正无效：修正后 SQL 与原 SQL 相同",
        }

    return {"sql": result, "attempts": correction_attempts}


def is_same_sql_after_normalization(original_sql: str, corrected_sql: str) -> bool:
    return (
        normalize_sql_for_execution(original_sql).lower()
        == normalize_sql_for_execution(corrected_sql).lower()
    )
