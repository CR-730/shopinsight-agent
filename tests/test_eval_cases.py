from pathlib import Path

from app.evaluation.cases import EvalCase, evaluate_case, load_eval_cases
from app.scripts.run_eval import _infer_exception_stage


def test_infer_exception_stage_uses_traceback_node_name():
    def recall_value_context():
        raise RuntimeError("LLM quota exhausted")

    try:
        recall_value_context()
    except RuntimeError as exc:
        assert _infer_exception_stage(exc) == "rag_recall"


def test_infer_exception_stage_uses_compacted_graph_names():
    def context_compaction():
        raise RuntimeError("context failed")

    def correct_sql_candidate():
        raise RuntimeError("sql correction failed")

    def sql_executor():
        raise RuntimeError("sql execution failed")

    cases = [
        (context_compaction, "context_filter"),
        (correct_sql_candidate, "sql_validation"),
        (sql_executor, "tool_execution"),
    ]
    for fn, expected_stage in cases:
        try:
            fn()
        except RuntimeError as exc:
            assert _infer_exception_stage(exc) == expected_stage


def test_evaluate_case_passes_when_sql_and_context_match():
    case = EvalCase(
        id="sales_by_region",
        query="统计华北地区销售总额",
        suite="smoke",
        capabilities=["rag_column_recall", "rag_metric_recall", "sql_generation"],
        expected_sql_contains=["sum(", "order_amount", "华北"],
        expected_columns=["fact_order.order_amount", "dim_region.region_name"],
        expected_metrics=["GMV"],
        expected_time_binding={"grain": "quarter", "year": 2025},
        must_call_tools=["mysql.dw.validate"],
    )
    state = {
        "error": None,
        "sql": "select sum(fact_order.order_amount) from fact_order join dim_region where dim_region.region_name = '华北'",
        "trace": {
            "keywords": ["华北", "销售额"],
            "retrieved_columns": [],
            "retrieved_metrics": [],
            "retrieved_values": [],
        },
        "sql_context": {
            "tables": [
                {"name": "fact_order", "columns": [{"name": "order_amount"}]},
                {"name": "dim_region", "columns": [{"name": "region_name"}]},
            ],
            "metrics": [{"name": "GMV"}],
        },
        "business_binding": {
            "metrics": [{"canonical_metric": "GMV", "raw_mention": "销售额"}],
            "filters": [{"canonical_value": "华北", "raw_value": "北方区域"}],
            "time": {"grain": "quarter", "year": 2025},
        },
        "output": {"rows": [{"销售总额": 1}]},
    }

    result = evaluate_case(case, state)

    assert result.passed is True
    assert result.failure_stage is None
    assert result.failures == []
    assert result.trace["generated_sql"].startswith("select")
    assert result.trace["metric_bindings"] == [
        {"canonical_metric": "GMV", "raw_mention": "销售额"}
    ]
    assert result.trace["resolved_filters"] == [
        {"canonical_value": "华北", "raw_value": "北方区域"}
    ]
    assert result.trace["time_binding"] == {"grain": "quarter", "year": 2025}
    assert "mysql.dw.execute" in result.trace["tool_calls"]


def test_evaluate_case_checks_expected_unresolved_binding():
    case = EvalCase(
        id="unknown_region",
        query="火星区域的销售额是多少",
        expected_blocked_by="business_binding",
        expected_unresolved_binding={
            "type": "enum_value",
            "raw_text": "火星",
            "candidate_column": "dim_region.region_name",
        },
    )
    state = {
        "trace": {"keywords": ["火星", "区域", "销售额"]},
        "sql": "",
        "failure": {
            "category": "business_binding",
            "stage": "business_binding",
            "code": "value_not_found",
            "message": "业务绑定未解析",
            "disposition": "blocked",
        },
        "business_binding": {
            "unresolved": [
                {
                    "type": "enum_value",
                    "raw_text": "火星",
                    "candidate_column": "dim_region.region_name",
                    "reason": "value_not_found",
                }
            ]
        },
    }

    result = evaluate_case(case, state)

    assert result.passed is True


def test_evaluate_case_reports_missing_expected_value_as_rag_failure():
    case = EvalCase(
        id="value_recall",
        query="华东地区销售额",
        expected_values=["dim_region.region_name.华东"],
    )
    state = {
        "trace": {
            "keywords": ["华东"],
            "retrieved_values": [],
        },
    }

    result = evaluate_case(case, state)

    assert result.passed is False
    assert result.failure_stage == "rag_recall"
    assert result.failures[0].code == "missing_expected_value"


def test_evaluate_case_accepts_any_guard_expected_block():
    case = EvalCase(
        id="unsafe",
        query="导出所有手机号",
        expected_blocked_by="any_guard",
    )
    state = {
        "trace": {"keywords": []},
        "failure": {
            "category": "input_guard",
            "stage": "pre_rag_guard",
            "code": "privacy_detail",
            "message": "拦截",
            "disposition": "blocked",
        },
    }

    result = evaluate_case(case, state)

    assert result.passed is True


def test_evaluate_case_reports_structured_failures_and_stage():
    case = EvalCase(
        id="sales_by_region",
        query="统计华北地区销售总额",
        expected_sql_contains=["order_amount"],
        expected_columns=["fact_order.order_amount"],
        expected_metrics=["GMV"],
        fatal_errors=["sql_validation_error"],
    )
    state = {
        "trace": {"keywords": []},
        "failure": {
            "category": "sql_validation",
            "stage": "sql_validation",
            "code": "sql_validation_error",
            "message": "Unknown column",
            "disposition": "failed",
        },
        "sql": "select 1",
        "sql_context": {"tables": [], "metrics": []},
    }

    result = evaluate_case(case, state)

    assert result.passed is False
    assert result.failure_stage == "keyword_extraction"
    failures = [failure.to_dict() for failure in result.failures]
    assert {
        "code": "sql_validation_error",
        "message": "SQL 校验错误：Unknown column",
        "stage": "sql_validation",
        "fatal": True,
    } in failures
    assert any(failure["code"] == "missing_sql_fragment" for failure in failures)
    assert any(failure["stage"] == "rag_recall" for failure in failures)


def test_evaluate_case_does_not_add_missing_context_noise_for_empty_timeout_state():
    case = EvalCase(
        id="timeout_case",
        query="2025 年第一季度各大区 GMV",
        expected_sql_contains=["order_amount", "region_name"],
        expected_columns=["fact_order.order_amount", "dim_region.region_name"],
        expected_metrics=["GMV"],
        must_call_tools=["qdrant.column.search", "mysql.dw.validate"],
        expected_result={"mode": "non_empty"},
    )
    state = {
        "trace": {"keywords": []},
        "failure": {
            "category": "sql_execution",
            "stage": "tool_execution",
            "code": "timeout",
            "message": "case timeout while waiting for run_sql",
            "disposition": "failed",
        },
        "sql": "",
        "sql_context": {"tables": [], "metrics": []},
    }

    result = evaluate_case(case, state)

    failures = [failure.to_dict() for failure in result.failures]
    assert result.failure_stage == "tool_execution"
    assert [failure["code"] for failure in failures] == ["tool_or_node_exception"]
    assert failures[0]["stage"] == "tool_execution"


def test_evaluate_case_keeps_tool_execution_timeout_stage_for_sql_state():
    case = EvalCase(
        id="run_sql_timeout",
        query="统计 GMV",
        expected_sql_contains=["order_amount"],
        expected_columns=["fact_order.order_amount"],
        expected_metrics=["GMV"],
        expected_result={"mode": "non_empty"},
    )
    state = {
        "trace": {"keywords": ["GMV"]},
        "failure": {
            "category": "sql_execution",
            "stage": "tool_execution",
            "code": "timeout",
            "message": "SQL 执行超时：60 秒",
            "disposition": "failed",
        },
        "sql": "select sum(order_amount) from fact_order",
        "sql_context": {
            "tables": [
                {"name": "fact_order", "columns": [{"name": "order_amount"}]}
            ],
            "metrics": [{"name": "GMV"}],
        },
    }

    result = evaluate_case(case, state)

    failures = [failure.to_dict() for failure in result.failures]
    assert result.failure_stage == "tool_execution"
    assert any(failure["code"] == "tool_or_node_exception" for failure in failures)
    assert not any(failure["code"] == "sql_validation_error" for failure in failures)


def test_load_eval_cases_supports_extended_schema(tmp_path):
    cases_path = tmp_path / "cases.yaml"
    cases_path.write_text(
        """
- id: slow_case
  query: 统计客单价
  suite: regression
  difficulty: hard
  capabilities: [sql_generation]
  tags: [aov]
  risk_points: [metric_mapping]
  expected_sql_contains: [order_amount]
  expected_columns: [fact_order.order_amount]
  expected_metrics: [AOV]
  expected_values: [dim_region.region_name.华东]
  expected_result: {mode: non_empty}
  forbidden_sql: [delete]
  must_call_tools: [mysql.dw.validate]
  forbidden_behavior: [编造字段]
  fatal_errors: [sql_validation_error]
  timeout_seconds: 45
""",
        encoding="utf-8",
    )

    cases = load_eval_cases(cases_path)

    assert cases[0].suite == "regression"
    assert cases[0].difficulty == "hard"
    assert cases[0].capabilities == ["sql_generation"]
    assert cases[0].expected_result == {"mode": "non_empty"}
    assert cases[0].expected_values == ["dim_region.region_name.华东"]
    assert cases[0].timeout_seconds == 45


def test_eval_cases_cover_required_suites_and_count():
    cases = load_eval_cases(Path("examples/eval_cases.yaml"))
    suites = {case.suite for case in cases}

    assert len(cases) >= 20
    assert {"smoke", "regression", "adversarial", "realistic"} <= suites
    assert sum(1 for case in cases if case.suite == "smoke") >= 4
    assert sum(1 for case in cases if case.suite == "adversarial") >= 3
    assert any("rag_value_hybrid_recall" in case.capabilities for case in cases)
    assert any("sql_correction_loop" in case.capabilities for case in cases)
