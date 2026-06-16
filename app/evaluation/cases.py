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
    retrieved_columns = _entity_ids(state.get("retrieved_column_infos") or [])
    retrieved_metrics = _metric_names(state.get("retrieved_metric_infos") or [])
    retrieved_values = _value_ids(state.get("retrieved_value_infos") or [])
    filtered_columns = sorted(_extract_column_ids(state.get("table_infos") or []))
    filtered_metrics = sorted(
        metric_info.get("name")
        for metric_info in state.get("metric_infos") or []
        if metric_info.get("name")
    )
    generated_sql = str(state.get("sql") or "")
    sql_error = state.get("error")
    safety_error = state.get("safety_error")
    blocked_by = state.get("blocked_by")
    final_answer = state.get("final_answer")
    exception_stage = state.get("exception_stage")

    return {
        "exception_stage": exception_stage,
        "keywords": state.get("keywords") or [],
        "retrieved_columns": retrieved_columns,
        "retrieved_metrics": retrieved_metrics,
        "retrieved_values": retrieved_values,
        "filtered_columns": filtered_columns,
        "filtered_metrics": filtered_metrics,
        "metric_bindings": state.get("metric_bindings") or [],
        "resolved_filters": state.get("resolved_filters") or [],
        "business_binding": state.get("business_binding") or {},
        "time_binding": state.get("time_binding"),
        "unresolved_bindings": state.get("unresolved_bindings") or [],
        "ambiguous_bindings": state.get("ambiguous_bindings") or [],
        "generated_sql": generated_sql,
        "sql_error": sql_error,
        "safety_error": safety_error,
        "blocked_by": blocked_by,
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

    if trace.get("safety_error") is not None and trace.get("blocked_by") != (
        case.expected_blocked_by
    ):
        blocked_by = trace.get("blocked_by") or "safety_guard"
        failures.append(
            EvalFailure(
                code=f"{blocked_by}_blocked",
                message=f"{blocked_by} 拦截：{trace['safety_error']}",
                stage="safety",
            )
        )

    if case.expected_blocked_by and trace.get("blocked_by") != case.expected_blocked_by:
        failures.append(
            EvalFailure(
                code="missing_expected_block",
                message=f"未被预期闸门拦截：{case.expected_blocked_by}",
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
    calls = {"pre_rag_guard"}
    if state.get("keywords") is not None:
        calls.add("keyword_extraction")
    if state.get("retrieved_column_infos") is not None:
        calls.add("qdrant.column.search")
    if state.get("retrieved_metric_infos") is not None:
        calls.add("qdrant.metric.search")
    if state.get("retrieved_value_infos") is not None:
        calls.add("hybrid.value.search")
        calls.add("es.value.search")
        calls.add("qdrant.value.search")
    if state.get("sql"):
        calls.add("llm.sql.generate")
        calls.add("mysql.dw.validate")
    if state.get("safety_error") is not None:
        calls.add(state.get("blocked_by") or "safety_guard")
    if state.get("correction_attempts", 0) > 0:
        calls.add("llm.sql.correct")
    if state.get("final_answer") is not None:
        calls.add("mysql.dw.execute")
    return sorted(calls)


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
