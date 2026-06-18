"""Single graph node for SQL validation, correction, and execution."""

import json
from datetime import date, datetime
from decimal import Decimal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.agent.sql.sql_correction import correct_sql_candidate
from app.agent.sql.sql_executor import SqlExecutionRequest, SqlExecutor
from app.agent.sql.sql_guard import normalize_sql_for_execution
from app.agent.sql_loop import (
    DEFAULT_MAX_SQL_CORRECTION_ATTEMPTS,
    route_after_pre_sql_execution_validation,
)
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def sql_executor(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """Run the existing SQL close-loop behind one graph node."""

    writer = runtime.stream_writer
    current_state = dict(state)
    accumulated_update = {}
    executor = SqlExecutor(runtime.context["dw_mysql_repository"])
    for _ in range(_max_sql_executor_iterations(current_state)):
        validation_update = await _pre_validate_sql(current_state, executor)
        current_state.update(validation_update)
        accumulated_update.update(validation_update)

        route = route_after_pre_sql_execution_validation(current_state)
        if route == "pass":
            run_update = await _execute_sql(current_state, executor, writer, runtime)
            current_state.update(run_update)
            accumulated_update.update(run_update)
            return accumulated_update
        if route == "blocked":
            return accumulated_update
        if route == "fail_sql_correction":
            fail_update = _fail_sql_correction(current_state)
            current_state.update(fail_update)
            accumulated_update.update(fail_update)
            return accumulated_update

        correction_update = await correct_sql_candidate(current_state, runtime.context)
        current_state.update(correction_update)
        accumulated_update.update(correction_update)
    error = "SQL executor exceeded internal correction loop limit"
    return {**accumulated_update, "error": error, "blocked_by": "sql_executor"}


def _max_sql_executor_iterations(state: dict) -> int:
    max_attempts = int(
        state.get("max_correction_attempts") or DEFAULT_MAX_SQL_CORRECTION_ATTEMPTS
    )
    return max(1, max_attempts + 1)


async def _pre_validate_sql(state: dict, executor: SqlExecutor) -> dict:
    sql = normalize_sql_for_execution(state["sql"])
    result = await executor.pre_validate(state, SqlExecutionRequest(sql=sql))
    if result.status == "repairable_error":
        return {"sql": sql, "error": result.error, "safety_error": None}
    if result.status == "blocked":
        return {
            "sql": sql,
            "error": None,
            "safety_error": result.error,
            "blocked_by": "sql_executor",
        }
    return {"sql": sql, "error": None, "safety_error": None}


async def _execute_sql(state: dict, executor: SqlExecutor, writer, runtime: Runtime) -> dict:
    writer({"type": "progress", "step": "执行查询", "status": "running"})
    result = await executor.execute(SqlExecutionRequest(sql=state["sql"]))
    if not result.ok:
        writer({"type": "progress", "step": "执行查询", "status": "error"})
        return {
            "error": result.error or "SQL 执行失败",
            "exception_stage": result.audit.get("exception_stage"),
            "blocked_by": None,
        }

    meta = _result_meta(state)
    writer({"type": "progress", "step": "执行查询", "status": "success"})
    writer({"type": "result", "data": result.result, "meta": meta})

    analysis = await _analyze_result(state["query"], result.result, runtime)
    if analysis:
        _write_answer_delta(writer, "\n\n" + analysis)

    return {"final_answer": result.result, "result_meta": meta, "result_analysis": analysis}


def _fail_sql_correction(state: dict) -> dict:
    return {
        "error": state.get("safety_error") or state.get("error") or "SQL 校验失败",
        "blocked_by": "sql_correction",
    }


def _result_meta(state: dict) -> dict:
    tables = []
    for table in state.get("table_infos") or []:
        name = str(table.get("name") or "").strip()
        if name and name not in tables:
            tables.append(name)
    return {"tables": tables[:5]}


async def _analyze_result(query: str, rows: list[dict], runtime: Runtime) -> str:
    """执行成功后用 LLM 对结果做简短的自然语言解读。"""
    try:
        prompt = PromptTemplate(
            template=load_prompt("result_analyzer"),
            input_variables=["query", "result"],
        )
        analysis = await ainvoke_llm_with_usage(
            prompt,
            llm,
            StrOutputParser(),
            {"query": query, "result": _format_result_for_llm(rows)},
            "结果分析",
            runtime.context["cost_tracker"],
            app_config.llm.timeout_seconds,
            cacheable=False,
        )
        return str(analysis).strip()
    except Exception as exc:
        logger.warning(f"结果分析失败，跳过: {exc}")
        return ""


def _format_result_for_llm(rows: list[dict]) -> str:
    """把查询结果格式化为 LLM 易读的文本。"""
    if not rows:
        return "（空结果）"

    def _default(obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return str(obj)

    return "\n".join(json.dumps(row, ensure_ascii=False, default=_default) for row in rows)


def _write_answer_delta(writer, text: str) -> None:
    content = str(text or "")
    for index in range(0, len(content), 12):
        writer({"type": "answer_delta", "delta": content[index: index + 12]})
