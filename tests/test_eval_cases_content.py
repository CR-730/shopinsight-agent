from pathlib import Path

from app.evaluation.cases import load_eval_cases


def test_eval_cases_cover_typical_retrieval_regressions():
    cases = load_eval_cases(Path("examples/eval_cases.yaml"))
    queries = {case.query for case in cases}

    assert "统计华北地区的销售总额" in queries
    assert "统计客单价" in queries
    assert "按大区统计 GMV" in queries
    assert "北方区域销售额" in queries


def test_eval_cases_have_required_diagnostic_fields():
    cases = load_eval_cases(Path("examples/eval_cases.yaml"))

    for case in cases:
        assert case.id
        assert case.query
        assert case.suite in {"smoke", "regression", "adversarial", "realistic"}
        assert case.difficulty in {"easy", "medium", "hard"}
        assert case.capabilities
        assert case.tags
        assert case.risk_points
        assert isinstance(case.forbidden_sql, list)
        assert isinstance(case.must_call_tools, list)
        assert isinstance(case.forbidden_behavior, list)
        assert isinstance(case.fatal_errors, list)
