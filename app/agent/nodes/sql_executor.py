"""Single graph node for SQL validation, correction, and execution."""

import json
from dataclasses import asdict
from datetime import date, datetime
from decimal import Decimal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime
from sqlglot import parse_one
from sqlglot.errors import ParseError

from app.agent.context import DataAgentContext
from app.agent.failure import build_failure
from app.agent.llm import llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.agent.predicate_normalization import stable_fingerprint
from app.agent.semantic_planning.plan import SemanticQueryPlan
from app.agent.sql.plan_consistency import validate_sql_plan_consistency
from app.agent.sql.sql_correction import correct_sql_candidate
from app.agent.sql.sql_executor import SqlExecutionRequest, SqlExecutor
from app.agent.sql.sql_guard import normalize_sql_for_execution
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def sql_executor(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """Run the existing SQL close-loop behind one graph node."""

    writer = runtime.stream_writer
    current_state = dict(state)
    accumulated_update = {"failure": None}
    executor = SqlExecutor(runtime.context["dw_mysql_repository"])
    last_validation_error = ""
    correction_attempts = 0
    max_correction_attempts = app_config.agent.max_sql_correction_attempts
    seen_sql_fingerprints: set[str] = set()
    previous_difference_fingerprint: str | None = None
    for _ in range(max(1, max_correction_attempts + 1)):
        sql_fingerprint = _sql_fingerprint(current_state["sql"])
        sql_repeated = sql_fingerprint in seen_sql_fingerprints
        seen_sql_fingerprints.add(sql_fingerprint)
        validation = await _pre_validate_sql(current_state, executor)
        current_state["sql"] = validation["sql"]
        accumulated_update["sql"] = validation["sql"]
        last_validation_error = str(validation.get("validation_error") or "")
        plan_differences = list(validation.get("plan_differences") or [])
        accumulated_update["trace"] = {
            **(current_state.get("trace") or {}),
            **(accumulated_update.get("trace") or {}),
            "sql_plan_consistency": {
                "status": "pass" if not plan_differences else "failed",
                "differences": plan_differences,
            },
        }

        if validation["status"] == "pass":
            run_update = await _execute_sql(current_state, executor, writer, runtime)
            current_state.update(run_update)
            accumulated_update.update(run_update)
            return accumulated_update
        if validation["status"] == "blocked":
            return {
                **accumulated_update,
                "failure": build_failure(
                    category="sql_validation",
                    stage="sql_executor",
                    code="sql_safety_blocked",
                    message=last_validation_error or "SQL 安全校验失败",
                    disposition="blocked",
                ),
            }

        difference_fingerprint = stable_fingerprint(
            {
                "differences": plan_differences,
                "validation_error": last_validation_error,
            }
        )
        if difference_fingerprint == previous_difference_fingerprint:
            return _stopped_correction(
                accumulated_update,
                last_validation_error,
                "differences_unchanged",
            )
        if sql_repeated:
            return _stopped_correction(
                accumulated_update,
                last_validation_error,
                "sql_cycle",
            )
        previous_difference_fingerprint = difference_fingerprint

        if correction_attempts >= max_correction_attempts:
            return {
                **accumulated_update,
                "failure": _correction_failure(last_validation_error),
            }

        correction_update = await correct_sql_candidate(
            current_state,
            runtime.context,
            last_validation_error,
            correction_attempts=correction_attempts,
            max_correction_attempts=max_correction_attempts,
            plan_differences=plan_differences,
        )
        correction_attempts = max(
            int(correction_update.get("attempts") or 0),
            correction_attempts + 1,
        )
        current_state.update(correction_update)
        accumulated_update["trace"] = {
            **(current_state.get("trace") or {}),
            **(accumulated_update.get("trace") or {}),
            "sql_correction_attempts": correction_attempts,
        }
        accumulated_update.update(
            {key: value for key, value in correction_update.items() if key in {"sql"}}
        )
    return {
        **accumulated_update,
        "failure": _correction_failure(last_validation_error),
    }


async def _pre_validate_sql(state: dict, executor: SqlExecutor) -> dict:
    sql = normalize_sql_for_execution(state["sql"])
    semantic_plan = state.get("semantic_plan")
    if not semantic_plan:
        difference = {
            "code": "semantic_plan_missing",
            "path": "semantic_plan",
            "expected": "trusted SemanticQueryPlan",
            "actual": None,
        }
        return {
            "sql": sql,
            "status": "repairable_error",
            "validation_error": format_plan_differences([difference]),
            "plan_differences": [difference],
        }

    consistency = validate_sql_plan_consistency(sql, semantic_plan)
    if not consistency.ok:
        differences = [asdict(item) for item in consistency.differences]
        return {
            "sql": sql,
            "status": "repairable_error",
            "validation_error": format_plan_differences(differences),
            "plan_differences": differences,
        }

    result = await executor.pre_validate(state, SqlExecutionRequest(sql=sql))
    if result.status == "repairable_error":
        return {
            "sql": sql,
            "status": "repairable_error",
            "validation_error": result.error,
            "plan_differences": [],
        }
    if result.status == "blocked":
        return {
            "sql": sql,
            "status": "blocked",
            "validation_error": result.error,
            "plan_differences": [],
        }
    return {
        "sql": sql,
        "status": "pass",
        "validation_error": None,
        "plan_differences": [],
    }


def format_plan_differences(differences: list[dict]) -> str:
    return json.dumps(differences, ensure_ascii=False, sort_keys=True)


async def _execute_sql(
    state: dict, executor: SqlExecutor, writer, runtime: Runtime
) -> dict:
    writer({"type": "progress", "step": "执行查询", "status": "running"})
    result = await executor.execute(SqlExecutionRequest(sql=state["sql"]))
    if not result.ok:
        writer({"type": "progress", "step": "执行查询", "status": "error"})
        return {
            "failure": build_failure(
                category="sql_execution",
                stage=str(result.audit.get("exception_stage") or "tool_execution"),
                code=str(result.audit.get("error_type") or "execution_failed"),
                message=result.error or "SQL 执行失败",
                disposition="failed",
            )
        }

    meta = _result_meta(state)
    writer({"type": "progress", "step": "执行查询", "status": "success"})
    writer({"type": "result", "data": result.result, "meta": meta})

    analysis = await _analyze_result(state["query"], result.result, runtime)
    if analysis:
        _write_answer_delta(writer, "\n\n" + analysis)

    return {
        "output": {
            "rows": result.result,
            "meta": meta,
            "analysis": analysis,
        }
    }


def _correction_failure(message: str) -> dict:
    return build_failure(
        category="sql_validation",
        stage="sql_correction",
        code="correction_exhausted",
        message=message or "SQL 校验失败",
        disposition="failed",
    )


def _stopped_correction(
    accumulated_update: dict,
    message: str,
    reason: str,
) -> dict:
    return {
        **accumulated_update,
        "trace": {
            **(accumulated_update.get("trace") or {}),
            "sql_correction_stop_reason": reason,
        },
        "failure": _correction_failure(message),
    }


def _sql_fingerprint(sql: str) -> str:
    normalized = normalize_sql_for_execution(sql)
    try:
        expression = parse_one(normalized, read="mysql")
        normalized = expression.sql(
            dialect="mysql",
            pretty=False,
            normalize=True,
        )
    except ParseError, ValueError:
        pass
    return stable_fingerprint(normalized)


def _result_meta(state: dict) -> dict:
    raw_plan = state.get("semantic_plan")
    if not raw_plan:
        return {"tables": []}
    try:
        plan = SemanticQueryPlan.model_validate(raw_plan)
    except ValueError:
        return {"tables": []}
    return {"tables": list(dict.fromkeys(plan.required_table_ids))[:5]}


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

    return "\n".join(
        json.dumps(row, ensure_ascii=False, default=_default) for row in rows
    )


def _write_answer_delta(writer, text: str) -> None:
    content = str(text or "")
    for index in range(0, len(content), 12):
        writer({"type": "answer_delta", "delta": content[index : index + 12]})
