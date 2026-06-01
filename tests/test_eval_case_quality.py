from pathlib import Path

from app.evaluation.cases import load_eval_cases


def _quality_score() -> tuple[int, list[str]]:
    cases = load_eval_cases(Path("examples/eval_cases.yaml"))
    issues: list[str] = []

    if len(cases) < 20:
        issues.append("case 数量少于 20")

    ids = [case.id for case in cases]
    queries = [case.query for case in cases]
    if len(ids) != len(set(ids)):
        issues.append("case id 不唯一")
    if len(queries) != len(set(queries)):
        issues.append("query 不唯一")

    suites = {case.suite for case in cases}
    if {"smoke", "regression", "realistic", "adversarial"} - suites:
        issues.append("缺少必要 suite 分层")

    for case in cases:
        prefix = f"{case.id}: "
        if not case.business_source:
            issues.append(prefix + "缺少 business_source")
        if len(case.capabilities) > 6:
            issues.append(prefix + "capabilities 过多，目标不够聚焦")
        if not case.tags or not case.risk_points:
            issues.append(prefix + "缺少 tags 或 risk_points")
        if not case.forbidden_sql or not case.fatal_errors:
            issues.append(prefix + "缺少安全边界或一票否决错误")

        is_negative = case.suite == "adversarial"
        if is_negative:
            if "safety" not in case.capabilities:
                issues.append(prefix + "负例缺少 safety 能力标签")
            if not {"select", "from", "delete", "drop"} & set(case.forbidden_sql):
                issues.append(prefix + "负例没有明确禁止 SQL 或危险 SQL")
        else:
            if not case.expected_sql_contains:
                issues.append(prefix + "正例缺少 expected_sql_contains")
            if not case.expected_columns:
                issues.append(prefix + "正例缺少 expected_columns")
            if case.expected_result != {"mode": "non_empty"}:
                issues.append(prefix + "正例缺少可验证的非空结果要求")
            if not case.must_call_tools:
                issues.append(prefix + "正例缺少 must_call_tools")

    score = max(0, 100 - len(issues) * 2)
    return score, issues


def test_eval_cases_quality_score_is_at_least_95():
    score, issues = _quality_score()

    assert score >= 95, f"quality_score={score}, issues={issues}"
