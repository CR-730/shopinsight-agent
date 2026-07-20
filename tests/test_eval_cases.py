import asyncio
import importlib
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from app.evaluation.cases import (
    EvalCase,
    build_trace,
    evaluate_case,
    load_eval_cases,
    results_match,
)
from app.scripts.run_eval import ALL_CAPABILITIES, _infer_exception_stage

run_eval_module = importlib.import_module("app.scripts.run_eval")


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
        "semantic_plan": {
            "measures": [{"metric_id": "GMV", "output_alias": "GMV"}],
            "dimensions": [],
            "predicates": [
                {
                    "kind": "temporal",
                    "grain": "quarter",
                    "start_date": "2025-01-01",
                    "end_date": "2025-03-31",
                }
            ],
            "order_by": [],
        },
        "output": {"rows": [{"销售总额": 1}]},
    }

    result = evaluate_case(case, state)

    assert result.passed is True
    assert result.failure_stage is None
    assert result.failures == []
    assert result.trace["generated_sql"].startswith("select")
    assert result.trace["semantic_plan"] == state["semantic_plan"]
    assert result.trace["time_binding"]["grain"] == "quarter"
    assert result.trace["time_binding"]["year"] == 2025
    assert "mysql.dw.execute" in result.trace["tool_calls"]


def test_evaluate_case_checks_expected_unresolved_binding():
    case = EvalCase(
        id="unknown_region",
        query="火星区域的销售额是多少",
        expected_blocked_by="semantic_planning",
        expected_unresolved_binding={
            "type": "enum_value",
            "raw_text": "火星",
            "candidate_column": "dim_region.region_name",
        },
    )
    state = {
        "trace": {
            "keywords": ["火星", "区域", "销售额"],
            "planning_issues": [
                {
                    "type": "enum_value",
                    "raw_text": "火星",
                    "candidate_column": "dim_region.region_name",
                    "reason": "value_not_found",
                }
            ],
        },
        "sql": "",
        "failure": {
            "category": "semantic_planning",
            "stage": "semantic_planning",
            "code": "value_not_found",
            "message": "业务绑定未解析",
            "disposition": "blocked",
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
            "tables": [{"name": "fact_order", "columns": [{"name": "order_amount"}]}],
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


def test_eval_trace_reads_the_semantic_plan():
    plan = {"version": "1", "measures": [{"metric_id": "GMV"}]}
    trace = build_trace(
        {
            "semantic_plan": plan,
            "trace": {"planning_issues": []},
        }
    )

    assert trace["semantic_plan"] == plan
    assert "metric_bindings" not in trace


def test_eval_checks_expected_semantic_plan_subset():
    case = EvalCase(
        id="plan",
        query="统计销售额",
        expected_semantic_plan={
            "measures": [{"metric_id": "GMV", "output_alias": "GMV"}],
            "required_table_ids": ["fact_order"],
        },
    )
    state = {
        "trace": {"keywords": ["销售额"]},
        "semantic_plan": {
            "version": "1",
            "measures": [{"metric_id": "GMV", "output_alias": "错误别名"}],
            "required_table_ids": ["fact_order"],
        },
    }

    result = evaluate_case(case, state)

    assert result.passed is False
    assert "semantic_plan_mismatch" in {item.code for item in result.failures}


def test_eval_treats_join_endpoints_as_commutative():
    case = EvalCase(
        id="join",
        query="按地区统计销售额",
        expected_semantic_plan={
            "joins": [
                {
                    "left_column_id": "fact_order.region_id",
                    "right_column_id": "dim_region.region_id",
                    "join_type": "inner",
                }
            ]
        },
    )
    state = {
        "trace": {"keywords": ["地区", "销售额"]},
        "semantic_plan": {
            "joins": [
                {
                    "left_column_id": "dim_region.region_id",
                    "right_column_id": "fact_order.region_id",
                    "join_type": "inner",
                }
            ]
        },
    }

    assert evaluate_case(case, state).passed is True


def test_eval_checks_expected_planning_issue():
    case = EvalCase(
        id="ambiguous",
        query="华南销售额",
        expected_blocked_by="semantic_planning",
        expected_planning_issue={"code": "ambiguous_enum_value"},
    )
    state = {
        "trace": {
            "keywords": ["华南"],
            "planning_issues": [{"code": "value_not_found"}],
        },
        "failure": {
            "category": "semantic_planning",
            "stage": "semantic_planning",
            "code": "value_not_found",
            "message": "需要澄清",
            "disposition": "blocked",
        },
    }

    result = evaluate_case(case, state)

    assert result.failure_stage == "semantic_planning"
    assert "missing_expected_planning_issue" in {item.code for item in result.failures}


def test_eval_reports_semantic_planning_block_stage():
    case = EvalCase(
        id="blocked",
        query="华南销售额",
        expected_blocked_by="semantic_planning",
    )
    state = {
        "trace": {"keywords": ["华南"], "planning_issues": []},
        "failure": {
            "category": "semantic_planning",
            "stage": "semantic_planning",
            "code": "ambiguous",
            "message": "需要澄清",
            "disposition": "blocked",
        },
    }

    assert evaluate_case(case, state).passed is True


def test_eval_checks_sql_plan_consistency_status():
    case = EvalCase(
        id="consistency",
        query="统计销售额",
        expected_sql_plan_consistent=True,
    )
    state = {
        "trace": {
            "keywords": ["销售额"],
            "sql_plan_consistency": {"status": "failed", "differences": []},
        }
    }

    result = evaluate_case(case, state)

    assert "sql_plan_consistency_mismatch" in {item.code for item in result.failures}


def test_exact_result_normalizes_decimal_float_and_row_order():
    expected = [{"GMV": Decimal("2.00")}, {"GMV": 1}]
    actual = [{"GMV": 1.0}, {"GMV": Decimal("2")}]

    assert results_match(actual, expected, order_sensitive=False) is True
    assert results_match(actual, expected, order_sensitive=True) is False


def test_nonempty_wrong_result_fails_exact_comparison():
    case = EvalCase(
        id="exact",
        query="统计销售额",
        expected_result=[{"GMV": 100}],
    )
    state = {
        "trace": {"keywords": ["销售额"]},
        "output": {"rows": [{"GMV": 99}]},
    }

    result = evaluate_case(case, state)

    assert "exact_result_mismatch" in {item.code for item in result.failures}


def test_eval_loads_legacy_unresolved_expectation_temporarily(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text(
        """
- id: legacy
  query: 火星销售额
  expected_unresolved_binding: {type: enum_value, reason: value_not_found}
""",
        encoding="utf-8",
    )

    case = load_eval_cases(path)[0]

    assert case.expected_planning_issue == {
        "type": "enum_value",
        "reason": "value_not_found",
    }


def test_tool_calls_use_semantic_planning_name():
    trace = build_trace(
        {
            "trace": {"keywords": [], "planning_issues": []},
            "semantic_plan": {"version": "1"},
        }
    )

    assert "semantic_planning" in trace["tool_calls"]
    assert "plan_consistency" in ALL_CAPABILITIES


def test_run_eval_case_sets_fixed_semantic_reference_date(monkeypatch):
    captured = {}

    class MetaRepository:
        async def get_active_build_version(self):
            return "build-v1"

        async def get_metadata_cache_version(self):
            return "meta-v1"

    async def invoke(*, input, context):
        captured["context"] = context
        return {"trace": {"keywords": ["GMV"]}}

    monkeypatch.setattr(run_eval_module, "graph", SimpleNamespace(ainvoke=invoke))
    repositories = {
        "column_qdrant_repository": object(),
        "metric_qdrant_repository": object(),
        "value_es_repository": object(),
        "value_qdrant_repository": object(),
        "meta_mysql_repository": MetaRepository(),
        "dw_mysql_repository": object(),
    }

    asyncio.run(
        run_eval_module._run_case(
            EvalCase(id="reference-date", query="统计GMV"),
            repositories,
        )
    )

    assert captured["context"]["semantic_reference_date"].isoformat() == (
        run_eval_module.date.today().isoformat()
    )
