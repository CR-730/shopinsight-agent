"""SQL 修正节点。"""

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import correct_sql_llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.agent.nodes.pre_sql_execution_validation import normalize_sql_for_execution
from app.agent.sql_loop import DEFAULT_MAX_SQL_CORRECTION_ATTEMPTS
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def correct_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """根据 SQL 校验错误修正 SQL。"""

    writer = runtime.stream_writer
    step = "校正SQL"
    writer({"type": "progress", "step": step, "status": "running"})

    try:
        table_infos = state["table_infos"]
        metric_infos = state["metric_infos"]
        date_info = state["date_info"]
        db_info = state["db_info"]
        query = state["query"]
        sql = state["sql"]
        error = state["error"]

        prompt = PromptTemplate(
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
        )
        output_parser = StrOutputParser()

        result = await ainvoke_llm_with_usage(
            prompt,
            correct_sql_llm,
            output_parser,
            {
                "table_infos": yaml.dump(
                    table_infos, allow_unicode=True, sort_keys=False
                ),
                "metric_infos": yaml.dump(
                    metric_infos, allow_unicode=True, sort_keys=False
                ),
                "date_info": yaml.dump(date_info, allow_unicode=True, sort_keys=False),
                "db_info": yaml.dump(db_info, allow_unicode=True, sort_keys=False),
                "query": query,
                "sql": sql,
                "error": error,
            },
            step,
            runtime.context["cost_tracker"],
            app_config.llm.timeout_seconds,
            cacheable=False,
        )

        logger.info(f"校正后的SQL：{result}")
        correction_attempts = state.get("correction_attempts", 0) + 1
        if _is_same_sql_after_normalization(sql, result):
            max_attempts = int(
                state.get("max_correction_attempts")
                or DEFAULT_MAX_SQL_CORRECTION_ATTEMPTS
            )
            logger.warning("SQL 修正结果与原 SQL 相同，停止无效修正循环")
            writer({"type": "progress", "step": step, "status": "unchanged"})
            return {
                "sql": result,
                "error": "SQL 修正无效：修正后 SQL 与原 SQL 相同",
                "correction_attempts": max(correction_attempts, max_attempts),
            }

        writer({"type": "progress", "step": step, "status": "success"})
        return {
            "sql": result,
            "correction_attempts": correction_attempts,
        }
    except Exception as e:
        logger.error(f"{step} failed: {e}")
        writer({"type": "progress", "step": step, "status": "error"})
        raise


def _is_same_sql_after_normalization(original_sql: str, corrected_sql: str) -> bool:
    return normalize_sql_for_execution(original_sql).lower() == normalize_sql_for_execution(
        corrected_sql
    ).lower()
