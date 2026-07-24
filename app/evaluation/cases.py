"""Evaluation case loading, diagnostics, and rule-based scoring."""

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

import yaml
from sqlglot import exp, parse_one

from app.entities.value_info import build_value_candidate_id

FailureStage = Literal[
    "keyword_extraction",
    "rag_recall",
    "context_filter",
    "semantic_planning",
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
    "semantic_planning",
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
    expected_retrieved_columns: list[str] | None = None
    expected_metrics: list[str] = field(default_factory=list)
    expected_values: list[str] = field(default_factory=list)
    expected_value_bindings: list[dict[str, str]] = field(default_factory=list)
    expected_time_binding: dict[str, Any] | None = None
    expected_unresolved_binding: dict[str, Any] | None = None
    expected_semantic_plan: dict[str, Any] | None = None
    expected_planning_issue: dict[str, Any] | None = None
    expected_sql_plan_consistent: bool | None = None
    oracle_sql: str | None = None
    order_sensitive: bool = False
    ablation_options: dict[str, bool] | None = None
    expected_result: Any = None
    expected_blocked_by: str | None = None
    forbidden_sql: list[str] = field(default_factory=list)
    must_call_tools: list[str] = field(default_factory=list)
    forbidden_behavior: list[str] = field(default_factory=list)
    fatal_errors: list[str] = field(default_factory=list)
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if self.oracle_sql and not self.expected_blocked_by:
            if self.expected_sql_plan_consistent is None:
                self.expected_sql_plan_consistent = True
            self.capabilities = list(
                dict.fromkeys(
                    [
                        *self.capabilities,
                        "semantic_planning",
                        "plan_consistency",
                        "sql_validation",
                        "tool_execution",
                        "answer_generation",
                    ]
                )
            )
        if self.oracle_sql and self.expected_retrieved_columns is None:
            self.expected_retrieved_columns = _retrieval_columns_from_oracle(
                self.oracle_sql,
                exclude_aggregate_columns=bool(self.expected_metrics),
                recoverable_value_columns={
                    str(binding["column_id"])
                    for binding in self.expected_value_bindings
                },
            )
        elif self.expected_retrieved_columns is None:
            self.expected_retrieved_columns = []
        if self.expected_value_bindings and not self.expected_values:
            self.expected_values = [
                build_value_candidate_id(
                    str(binding["column_id"]),
                    str(binding["value"]),
                )
                for binding in self.expected_value_bindings
            ]
        if (
            self.expected_planning_issue is None
            and self.expected_unresolved_binding is not None
        ):
            self.expected_planning_issue = dict(self.expected_unresolved_binding)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _retrieval_columns_from_oracle(
    sql: str,
    *,
    exclude_aggregate_columns: bool = False,
    recoverable_value_columns: set[str] | None = None,
) -> list[str]:
    """Derive query-facing column Gold while excluding physical join/time plumbing."""

    statement = parse_one(sql, read="mysql")
    aliases = {
        table.alias_or_name: table.name for table in statement.find_all(exp.Table)
    }
    tables = set(aliases.values())
    output_aliases = {
        projection.alias for projection in statement.expressions if projection.alias
    }
    join_column_nodes = {
        id(column)
        for join in statement.find_all(exp.Join)
        for column in (join.args.get("on") or exp.Null()).find_all(exp.Column)
    }
    aggregate_column_nodes = {
        id(column)
        for aggregate in statement.find_all(exp.AggFunc)
        for column in aggregate.find_all(exp.Column)
    }
    recoverable_value_columns = recoverable_value_columns or set()
    result: set[str] = set()
    for column in statement.find_all(exp.Column):
        if id(column) in join_column_nodes:
            continue
        if exclude_aggregate_columns and id(column) in aggregate_column_nodes:
            continue
        if not column.table and column.name in output_aliases:
            continue
        if column.table:
            table_name = aliases.get(column.table, column.table)
        elif len(tables) == 1:
            table_name = next(iter(tables))
        else:
            continue
        column_id = f"{table_name}.{column.name}"
        if column_id == "fact_order.date_id":
            continue
        if column_id in recoverable_value_columns:
            continue
        result.add(column_id)
    return sorted(result)


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
    semantic_plan = state.get("semantic_plan") or {}
    retrieved_columns = sorted(debug_trace.get("retrieved_columns") or [])
    retrieved_metrics = sorted(debug_trace.get("retrieved_metrics") or [])
    retrieved_values = sorted(debug_trace.get("retrieved_values") or [])
    filtered_columns = sorted(
        str(column_id)
        for column_id in semantic_plan.get("required_column_ids") or []
        if str(column_id)
    )
    filtered_metrics = sorted(
        str(measure.get("metric_id"))
        for measure in semantic_plan.get("measures") or []
        if measure.get("metric_id")
    )
    generated_sql = str(state.get("sql") or "")
    failure = state.get("failure") or {}
    failure_category = str(failure.get("category") or "")
    failure_disposition = str(failure.get("disposition") or "")
    failure_message = str(failure.get("message") or "")
    sql_error = (
        failure_message
        if failure_category == "sql_validation" and failure_disposition == "failed"
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
    planning_issues = debug_trace.get("planning_issues") or []

    return {
        "exception_stage": exception_stage,
        "keywords": debug_trace.get("keywords") or [],
        "retrieved_columns": retrieved_columns,
        "retrieved_metrics": retrieved_metrics,
        "retrieved_values": retrieved_values,
        "filtered_columns": filtered_columns,
        "filtered_metrics": filtered_metrics,
        "semantic_plan": semantic_plan,
        "planning_issues": planning_issues,
        "sql_plan_consistency": debug_trace.get("sql_plan_consistency"),
        "time_binding": _time_binding_from_plan(semantic_plan),
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

    expects_executed_answer = bool(case.oracle_sql and not case.expected_blocked_by)
    if not sql and (case.expected_sql_contains or expects_executed_answer):
        failures.append(
            EvalFailure(
                code="missing_sql",
                message="未生成 SQL",
                stage="sql_generation",
            )
        )
    if expects_executed_answer and trace["final_answer"] is None:
        failures.append(
            EvalFailure(
                code="missing_or_empty_final_answer",
                message="Oracle 用例缺少 SQL 执行结果",
                stage="answer_generation",
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
                else "semantic_planning"
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
                else "semantic_planning"
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

    if case.expected_semantic_plan is not None and not _semantic_plan_contains_subset(
        trace["semantic_plan"], case.expected_semantic_plan
    ):
        failures.append(
            EvalFailure(
                code="semantic_plan_mismatch",
                message="semantic_plan 与预期子集不一致",
                stage="semantic_planning",
            )
        )

    if case.expected_planning_issue and not _has_expected_planning_issue(
        trace["planning_issues"], case.expected_planning_issue
    ):
        failures.append(
            EvalFailure(
                code="missing_expected_planning_issue",
                message=f"缺少预期规划问题：{case.expected_planning_issue}",
                stage="semantic_planning",
            )
        )

    if case.expected_sql_plan_consistent is not None:
        consistency = trace.get("sql_plan_consistency") or {}
        actual_consistent = consistency.get("status") == "pass"
        if actual_consistent != case.expected_sql_plan_consistent:
            failures.append(
                EvalFailure(
                    code="sql_plan_consistency_mismatch",
                    message=(
                        "SQL 与 semantic_plan 一致性状态不符合预期："
                        f"expected={case.expected_sql_plan_consistent}, "
                        f"actual={actual_consistent}"
                    ),
                    stage="sql_validation",
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
                        stage="semantic_planning",
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
        elif expected_mode != "non_empty":
            expected_rows = (
                case.expected_result.get("rows")
                if isinstance(case.expected_result, dict)
                else case.expected_result
            )
            plan_has_order = bool((trace.get("semantic_plan") or {}).get("order_by"))
            if not results_match(
                final_answer,
                expected_rows,
                order_sensitive=case.order_sensitive or plan_has_order,
            ):
                failures.append(
                    EvalFailure(
                        code="exact_result_mismatch",
                        message="执行结果与精确预期不一致",
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


def _has_expected_planning_issue(
    issues: list[dict[str, Any]], expected: dict[str, Any]
) -> bool:
    for issue in issues:
        if all(issue.get(key) == value for key, value in expected.items()):
            return True
    return False


def _contains_subset(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains_subset(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        remaining = list(actual)
        for expected_item in expected:
            match_index = next(
                (
                    index
                    for index, actual_item in enumerate(remaining)
                    if _contains_subset(actual_item, expected_item)
                ),
                None,
            )
            if match_index is None:
                return False
            remaining.pop(match_index)
        return True
    return actual == expected


def _semantic_plan_contains_subset(
    actual: dict[str, Any], expected: dict[str, Any]
) -> bool:
    expected_without_joins = {
        key: value for key, value in expected.items() if key != "joins"
    }
    if not _contains_subset(actual, expected_without_joins):
        return False
    expected_joins = expected.get("joins")
    if expected_joins is None:
        return True
    actual_endpoints = [
        frozenset(
            {
                str(item.get("left_column_id") or ""),
                str(item.get("right_column_id") or ""),
            }
        )
        for item in actual.get("joins") or []
    ]
    remaining = list(actual_endpoints)
    for item in expected_joins:
        endpoints = frozenset(
            {
                str(item.get("left_column_id") or ""),
                str(item.get("right_column_id") or ""),
            }
        )
        if endpoints not in remaining:
            return False
        remaining.remove(endpoints)
    return True


def results_match(
    actual_rows: Any,
    expected_rows: Any,
    *,
    order_sensitive: bool,
    ignore_column_names: bool = False,
) -> bool:
    """Compare exact result rows while normalizing numeric representation."""

    if not isinstance(actual_rows, list) or not isinstance(expected_rows, list):
        return False
    actual = [
        _canonical_result_row(row, ignore_column_names=ignore_column_names)
        for row in actual_rows
    ]
    expected = [
        _canonical_result_row(row, ignore_column_names=ignore_column_names)
        for row in expected_rows
    ]
    if order_sensitive:
        return actual == expected
    return sorted(actual, key=repr) == sorted(expected, key=repr)


def _canonical_result_row(row: Any, *, ignore_column_names: bool) -> Any:
    if ignore_column_names and isinstance(row, dict):
        return tuple(_canonical_value(value) for value in row.values())
    return _canonical_value(row)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (Decimal, int, float)):
        try:
            number = Decimal(str(value))
        except InvalidOperation:
            return ("number", str(value))
        return ("number", format(number.normalize(), "f"))
    if isinstance(value, (date, datetime)):
        return ("datetime", value.isoformat())
    if isinstance(value, dict):
        return tuple(
            (str(key), _canonical_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_value(item) for item in value)
    return ("scalar", str(value))


def _time_binding_from_plan(plan: dict[str, Any]) -> dict[str, Any] | None:
    temporal = next(
        (
            predicate
            for predicate in plan.get("predicates") or []
            if predicate.get("kind") == "temporal"
        ),
        None,
    )
    if temporal is None:
        return None
    result = dict(temporal)
    start_date = str(temporal.get("start_date") or "")
    if len(start_date) >= 4 and start_date[:4].isdigit():
        result["year"] = int(start_date[:4])
    return result


def _first_failure_stage(failures: list[EvalFailure]) -> FailureStage | None:
    if not failures:
        return None
    by_stage = {failure.stage for failure in failures}
    for stage in FAILURE_STAGE_ORDER:
        if stage in by_stage:
            return stage
    return failures[0].stage


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
    if state.get("semantic_plan") or debug_trace.get("planning_issues") is not None:
        calls.add("semantic_planning")
    if state.get("sql"):
        calls.add("llm.sql.generate")
        consistency = debug_trace.get("sql_plan_consistency") or {}
        if consistency.get("status") != "failed":
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
    if tool.startswith("semantic_planning"):
        return "semantic_planning"
    return "tool_execution"
