from pathlib import Path

from app.evaluation.cases import EvalCase, evaluate_case, load_eval_cases
from app.scripts.run_eval import _infer_exception_stage


def test_infer_exception_stage_uses_traceback_node_name():
    def recall_value():
        raise RuntimeError("LLM quota exhausted")

    try:
        recall_value()
    except RuntimeError as exc:
        assert _infer_exception_stage(exc) == "rag_recall"


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
        "keywords": ["华北", "销售额"],
        "error": None,
        "sql": "select sum(fact_order.order_amount) from fact_order join dim_region where dim_region.region_name = '华北'",
        "retrieved_column_infos": [],
        "retrieved_metric_infos": [],
        "retrieved_value_infos": [],
        "table_infos": [
            {"name": "fact_order", "columns": [{"name": "order_amount"}]},
            {"name": "dim_region", "columns": [{"name": "region_name"}]},
        ],
        "metric_infos": [{"name": "GMV"}],
        "metric_bindings": [{"canonical_metric": "GMV", "raw_mention": "销售额"}],
        "resolved_filters": [{"canonical_value": "华北", "raw_value": "北方区域"}],
        "time_binding": {"grain": "quarter", "year": 2025},
        "final_answer": [{"销售总额": 1}],
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
        expected_blocked_by="semantic_guard",
        expected_unresolved_binding={
            "type": "enum_value",
            "raw_text": "火星",
            "candidate_column": "dim_region.region_name",
        },
    )
    state = {
        "keywords": ["火星", "区域", "销售额"],
        "error": None,
        "sql": "",
        "safety_error": "业务绑定未解析",
        "blocked_by": "semantic_guard",
        "unresolved_bindings": [
            {
                "type": "enum_value",
                "raw_text": "火星",
                "candidate_column": "dim_region.region_name",
                "reason": "value_not_found",
            }
        ],
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
        "keywords": [],
        "error": "Unknown column",
        "sql": "select 1",
        "table_infos": [],
        "metric_infos": [],
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
        "keywords": [],
        "exception_stage": "tool_execution",
        "error": "case timeout while waiting for run_sql",
        "sql": "",
        "table_infos": [],
        "metric_infos": [],
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
        "keywords": ["GMV"],
        "exception_stage": "tool_execution",
        "error": "SQL 执行超时：60 秒",
        "sql": "select sum(order_amount) from fact_order",
        "table_infos": [{"name": "fact_order", "columns": [{"name": "order_amount"}]}],
        "metric_infos": [{"name": "GMV"}],
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
