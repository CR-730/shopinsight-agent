from pathlib import Path

from sqlglot import expressions as exp
from sqlglot import parse_one

from app.evaluation.cases import load_eval_cases
from app.scripts.run_eval import _validated_oracle_sql, summarize_repeat_results

CASES_PATH = Path("examples/eval_semantic_planning.yaml")


def test_semantic_e2e_cases_keep_required_complex_queries():
    queries = {case.query for case in load_eval_cases(CASES_PATH)}

    assert "2025年第一季度销售额最高的前5个商品" in queries
    assert "2025年第一季度销售额大于10000元的地区，按销售额降序" in queries
    assert "2025年第一季度华北地区的销售额" in queries


def test_every_resolved_e2e_case_has_reviewed_oracle_sql():
    cases = load_eval_cases(CASES_PATH)

    for case in cases:
        if not case.expected_blocked_by:
            assert case.oracle_sql
            assert case.expected_semantic_plan
            assert case.expected_sql_plan_consistent is True


def test_oracle_sql_is_single_read_only_select():
    for case in load_eval_cases(CASES_PATH):
        sql = _validated_oracle_sql(case.oracle_sql or "")
        expression = parse_one(sql, read="mysql")

        assert isinstance(expression, exp.Select)
        assert expression.find(exp.Insert) is None
        assert expression.find(exp.Update) is None
        assert expression.find(exp.Delete) is None


def test_oracle_aliases_match_expected_plan_output_aliases():
    for case in load_eval_cases(CASES_PATH):
        expression = parse_one(case.oracle_sql or "", read="mysql")
        actual_aliases = {item.alias for item in expression.expressions if item.alias}
        plan = case.expected_semantic_plan or {}
        expected_aliases = {
            item["output_alias"]
            for key in ("measures", "dimensions")
            for item in plan.get(key) or []
            if item.get("output_alias")
        }

        assert actual_aliases == expected_aliases


def test_semantic_e2e_time_predicates_use_fact_date_id():
    for case in load_eval_cases(CASES_PATH):
        for predicate in (case.expected_semantic_plan or {}).get("predicates") or []:
            if predicate.get("kind") == "temporal":
                assert predicate["column_id"] == "fact_order.date_id"


def test_repeat_summary_fails_on_any_plan_consistency_violation():
    summary = summarize_repeat_results(
        [
            {
                "case_id": "top5",
                "passed": True,
                "oracle_result_match": True,
                "trace": {"sql_plan_consistency": {"status": "pass"}},
            },
            {
                "case_id": "top5",
                "passed": True,
                "oracle_result_match": True,
                "trace": {"sql_plan_consistency": {"status": "failed"}},
            },
        ]
    )

    assert summary["top5"] == {"passed": False, "runs": 2}
