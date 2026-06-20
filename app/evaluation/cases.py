"""Evaluation case loading, diagnostics, and rule-based scoring."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import yaml

FailureStage = Literal[
    "keyword_extraction",
    "rag_recall",
    "context_filter",
    "sql_generation",
    "sql_validation",
    "tool_execution",
    "answer_generation",
    "safety",
]

FAILURE_STAGE_ORDER: tuple[FailureStage, ...] = (
    "safety",
    "keyword_extraction",
    "rag_recall",
    "context_filter",
    "sql_generation",
    "sql_validation",
    "tool_execution",
    "answer_generation",
)


@dataclass
class EvalCase:
    id: str
    query: str
    business_source: str = ""
    suite: str = "regression"
    difficulty: str = "medium"
    capabilities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    risk_points: list[str] = field(default_factory=list)
    expected_sql_contains: list[str] = field(default_factory=list)
    expected_columns: list[str] = field(default_factory=list)
    expected_metrics: list[str] = field(default_factory=list)
    expected_values: list[str] = field(default_factory=list)
    expected_time_binding: dict[str, Any] | None = None
    expected_unresolved_binding: dict[str, Any] | None = None
    expected_result: Any = None
    expected_blocked_by: str | None = None
    forbidden_sql: list[str] = field(default_factory=list)
    must_call_tools: list[str] = field(default_factory=list)
    forbidden_behavior: list[str] = field(default_factory=list)
    fatal_errors: list[str] = field(default_factory=list)
    timeout_seconds: int = 300

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalFailure:
    code: str
    message: str
    stage: FailureStage
    fatal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalResult:
    case_id: str
    suite: str
    difficulty: str
    capabilities: list[str]
    tags: list[str]
    passed: bool
    failure_stage: FailureStage | None
    failures: list[EvalFailure]
    trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failures"] = [failure.to_dict() for failure in self.failures]
        return payload


def load_eval_cases(path) -> list[EvalCase]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [EvalCase(**item) for item in data]


def evaluate_case(case: EvalCase, state: dict[str, Any]) -> EvalResult:
    trace = build_trace(state)
    failures = _evaluate_failures(case, trace)
    failure_stage = _first_failure_stage(failures)
    fatal_codes = set(case.fatal_errors)
    for failure in failures:
        if failure.code in fatal_codes:
            failure.fatal = True

    return EvalResult(
        case_id=case.id,
        suite=case.suite,
        difficulty=case.difficulty,
        capabilities=case.capabilities,
        tags=case.tags,
        passed=not failures,
        failure_stage=failure_stage,
        failures=failures,
        trace=trace,
    )


def build_trace(state: dict[str, Any]) -> dict[str, Any]:
    debug_trace = state.get("trace") or {}
    sql_context = state.get("sql_context") or {}
    retrieved_columns = sorted(debug_trace.get("retrieved_columns") or [])
    retrieved_metrics = sorted(debug_trace.get("retrieved_metrics") or [])
    retrieved_values = sorted(debug_trace.get("retrieved_values") or [])
    filtered_columns = sorted(_extract_column_ids(sql_context.get("tables") or []))
    filtered_metrics = sorted(
        metric_info.get("name")
        for metric_info in sql_context.get("metrics") or []
        if metric_info.get("name")
    )
    generated_sql = str(state.get("sql") or "")
    failure = state.get("failure") or {}
    failure_category = str(failure.get("category") or "")
    failure_disposition = str(failure.get("disposition") or "")
    failure_message = str(failure.get("message") or "")
    sql_error = (
        failure_message
        if failure_category == "sql_validation"
        and failure_disposition == "failed"
        else None
    )
    safety_error = failure_message if failure_disposition == "blocked" else None
    blocked_by = (
        _normalize_stage_alias(str(failure.get("stage") or ""))
        if failure_disposition == "blocked"
        else None
    )
    output = state.get("output") or {}
    final_answer = output.get("rows")
    exception_stage = (
        failure.get("stage")
        if failure_disposition == "failed"
        and failure_category in {"sql_execution", "system"}
        else None
    )
    business_binding = state.get("business_binding") or {}

    return {
        "exception_stage": exception_stage,
        "keywords": debug_trace.get("keywords") or [],
        "retrieved_columns": retrieved_columns,
        "retrieved_metrics": retrieved_metrics,
        "retrieved_values": retrieved_values,
        "filtered_columns": filtered_columns,
        "filtered_metrics": filtered_metrics,
        "metric_bindings": business_binding.get("metrics") or [],
        "resolved_filters": business_binding.get("filters") or [],
        "business_binding": business_binding,
        "time_binding": business_binding.get("time"),
        "unresolved_bindings": business_binding.get("unresolved") or [],
        "ambiguous_bindings": business_binding.get("ambiguous") or [],
        "generated_sql": generated_sql,
        "sql_error": sql_error,
        "safety_error": safety_error,
        "blocked_by": blocked_by,
        "failure": failure,
        "tool_calls": _infer_tool_calls(state),
        "final_answer": final_answer,
    }


def _evaluate_failures(case: EvalCase, trace: dict[str, Any]) -> list[EvalFailure]:
    failures: list[EvalFailure] = []
    sql = str(trace["generated_sql"] or "")
    sql_lower = sql.lower()

    if trace["exception_stage"]:
        failures.append(
            EvalFailure(
                code="tool_or_node_exception",
                message=f"节点或工具异常：{trace['sql_error']}",
                stage=trace["exception_stage"],
            )
        )
        if _is_empty_exception_trace(trace):
            return failures

    if (
        not trace["keywords"]
        and not trace["exception_stage"]
        and not trace.get("blocked_by")
    ):
        failures.append(
            EvalFailure(
                code="missing_keywords",
                message="关键词抽取结果为空",
                stage="keyword_extraction",
            )
        )

    for fragment in case.forbidden_sql:
        fragment = str(fragment)
        if fragment.lower() in sql_lower:
            failures.append(
                EvalFailure(
                    code="forbidden_sql_fragment",
                    message=f"SQL 包含禁止片段：{fragment}",
                    stage="safety",
                )
            )

    if trace["sql_error"] is not None and not trace["exception_stage"]:
        failures.append(
            EvalFailure(
                code="sql_validation_error",
                message=f"SQL 校验错误：{trace['sql_error']}",
                stage="sql_validation",
            )
        )

    expected_blocked_by = _normalize_stage_alias(case.expected_blocked_by or "")
    expected_any_guard = expected_blocked_by == "any_guard"
    if (
        trace.get("safety_error") is not None
        and trace.get("blocked_by") != expected_blocked_by
        and not expected_any_guard
    ):
        blocked_by = trace.get("blocked_by") or "safety_guard"
        failures.append(
            EvalFailure(
                code=f"{blocked_by}_blocked",
                message=f"{blocked_by} 拦截：{trace['safety_error']}",
                stage="safety",
            )
        )

    if expected_any_guard and not trace.get("blocked_by"):
        failures.append(
            EvalFailure(
                code="missing_expected_block",
                message="未被任一安全或业务闸门拦截",
                stage="safety",
            )
        )
    elif (
        expected_blocked_by
        and not expected_any_guard
        and trace.get("blocked_by") != expected_blocked_by
    ):
        failures.append(
            EvalFailure(
                code="missing_expected_block",
                message=f"未被预期闸门拦截：{expected_blocked_by}",
                stage="safety",
            )
        )

    if not sql and case.expected_sql_contains:
        failures.append(
            EvalFailure(
                code="missing_sql",
                message="未生成 SQL",
                stage="sql_generation",
            )
        )

    for fragment in case.expected_sql_contains:
        fragment = str(fragment)
        if fragment.lower() not in sql_lower:
            failures.append(
                EvalFailure(
                    code="missing_sql_fragment",
                    message=f"SQL 缺少片段：{fragment}",
                    stage="sql_generation",
                )
            )

    for column_id in case.expected_columns:
        if column_id not in trace["filtered_columns"]:
            stage: FailureStage = (
                "rag_recall"
                if column_id not in trace["retrieved_columns"]
                else "context_filter"
            )
            failures.append(
                EvalFailure(
                    code="missing_expected_column",
                    message=f"缺少字段上下文：{column_id}",
                    stage=stage,
                )
            )

    for metric in case.expected_metrics:
        if metric not in trace["filtered_metrics"]:
            stage = (
                "rag_recall"
                if metric not in trace["retrieved_metrics"]
                else "context_filter"
            )
            failures.append(
                EvalFailure(
                    code="missing_expected_metric",
                    message=f"缺少指标上下文：{metric}",
                    stage=stage,
                )
            )

    for value_id in case.expected_values:
        if value_id not in trace["retrieved_values"]:
            failures.append(
                EvalFailure(
                    code="missing_expected_value",
                    message=f"缺少字段取值召回：{value_id}",
                    stage="rag_recall",
                )
            )

    if case.expected_time_binding:
        time_binding = trace.get("time_binding") or {}
        for key, expected_value in case.expected_time_binding.items():
            if time_binding.get(key) != expected_value:
                failures.append(
                    EvalFailure(
                        code="missing_expected_time_binding",
                        message=f"时间绑定不匹配：{key}={expected_value}",
                        stage="context_filter",
                    )
                )

    if case.expected_unresolved_binding and not _has_expected_binding_issue(
        trace["unresolved_bindings"], case.expected_unresolved_binding
    ):
        failures.append(
            EvalFailure(
                code="missing_expected_unresolved_binding",
                message=f"缺少预期未解析业务绑定：{case.expected_unresolved_binding}",
                stage="safety",
            )
        )

    for tool in case.must_call_tools:
        if tool not in trace["tool_calls"]:
            failures.append(
                EvalFailure(
                    code="missing_tool_call",
                    message=f"未观察到工具调用：{tool}",
                    stage=_stage_for_tool(tool),
                )
            )

    if case.expected_result is not None:
        expected_mode = (
            case.expected_result.get("mode")
            if isinstance(case.expected_result, dict)
            else None
        )
        final_answer = trace["final_answer"]
        if final_answer is None or (expected_mode == "non_empty" and not final_answer):
            failures.append(
                EvalFailure(
                    code="missing_or_empty_final_answer",
                    message="缺少非空最终执行结果，无法满足 expected_result",
                    stage="answer_generation",
                )
            )

    return failures


def _is_empty_exception_trace(trace: dict[str, Any]) -> bool:
    return (
        not trace["keywords"]
        and not trace["generated_sql"]
        and not trace["filtered_columns"]
        and not trace["filtered_metrics"]
        and not trace["retrieved_columns"]
        and not trace["retrieved_metrics"]
        and not trace["retrieved_values"]
    )


def _has_expected_binding_issue(
    issues: list[dict[str, Any]], expected: dict[str, Any]
) -> bool:
    for issue in issues:
        if all(issue.get(key) == value for key, value in expected.items()):
            return True
    return False


def _first_failure_stage(failures: list[EvalFailure]) -> FailureStage | None:
    if not failures:
        return None
    by_stage = {failure.stage for failure in failures}
    for stage in FAILURE_STAGE_ORDER:
        if stage in by_stage:
            return stage
    return failures[0].stage


def _extract_column_ids(table_infos: list[dict]) -> set[str]:
    column_ids: set[str] = set()
    for table_info in table_infos:
        table_name = table_info.get("name")
        for column_info in table_info.get("columns") or []:
            column_name = column_info.get("name")
            if table_name and column_name:
                column_ids.add(f"{table_name}.{column_name}")
    return column_ids


def _entity_ids(items: list[Any]) -> list[str]:
    result = []
    for item in items:
        value = getattr(item, "id", None)
        if value:
            result.append(value)
    return sorted(set(result))


def _metric_names(items: list[Any]) -> list[str]:
    result = []
    for item in items:
        value = getattr(item, "name", None)
        if value:
            result.append(value)
    return sorted(set(result))


def _value_ids(items: list[Any]) -> list[str]:
    result = []
    for item in items:
        value = getattr(item, "id", None)
        if value:
            result.append(value)
    return sorted(set(result))


def _infer_tool_calls(state: dict[str, Any]) -> list[str]:
    calls = {"intent_recognition"}
    debug_trace = state.get("trace") or {}
    if debug_trace.get("keywords") is not None:
        calls.add("keyword_extraction")
    if debug_trace.get("retrieved_columns") is not None:
        calls.add("qdrant.column.search")
    if debug_trace.get("retrieved_metrics") is not None:
        calls.add("qdrant.metric.search")
    if debug_trace.get("retrieved_values") is not None:
        calls.add("hybrid.value.search")
        calls.add("es.value.search")
        calls.add("qdrant.value.search")
    if state.get("sql"):
        calls.add("llm.sql.generate")
        calls.add("mysql.dw.validate")
    failure = state.get("failure") or {}
    if failure.get("disposition") == "blocked":
        calls.add(_normalize_stage_alias(failure.get("stage") or "safety_guard"))
    if (state.get("trace") or {}).get("sql_correction_attempts", 0) > 0:
        calls.add("llm.sql.correct")
    if (state.get("output") or {}).get("rows") is not None:
        calls.add("mysql.dw.execute")
    return sorted(calls)


def _normalize_stage_alias(stage: str) -> str:
    if stage == "pre_rag_guard":
        return "intent_recognition"
    return stage


def _stage_for_tool(tool: str) -> FailureStage:
    if tool.startswith(("qdrant", "es", "hybrid")):
        return "rag_recall"
    if tool.startswith("mysql.dw.validate"):
        return "sql_validation"
    if tool.startswith("mysql.dw.execute"):
        return "tool_execution"
    if tool.startswith("llm.sql"):
        return "sql_generation"
    if tool.startswith("business_binding"):
        return "safety"
    return "tool_execution"
