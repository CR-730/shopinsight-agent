from collections import Counter
from pathlib import Path

import yaml

from app.evaluation.cases import load_eval_cases

CASES_PATH = Path("examples/eval_cases_50_quality.yaml")
META_CONFIG_PATH = Path("conf/meta_config.yaml")


def test_eval_cases_50_count_and_tags_are_balanced():
    cases = load_eval_cases(CASES_PATH)

    assert len(cases) == 50
    assert len({case.id for case in cases}) == 50
    assert len({case.query for case in cases}) == 50

    tag_counts = Counter(tag for case in cases for tag in case.tags)
    assert tag_counts["quality50"] == 50
    assert tag_counts["ablation_retrieval"] >= 35
    assert tag_counts["ablation_cost"] >= 35
    assert tag_counts["ablation_guard"] >= 12
    assert tag_counts["sql_memory"] >= 6
    assert tag_counts["topn"] >= 5
    assert tag_counts["multi_metric"] >= 6

    suites = Counter(case.suite for case in cases)
    assert suites["complex"] >= 12
    assert suites["safety"] >= 8
    assert suites["clarify"] >= 4
    assert suites["memory"] >= 6


def test_eval_cases_50_expected_metadata_exists():
    cases = load_eval_cases(CASES_PATH)
    metadata = yaml.safe_load(META_CONFIG_PATH.read_text(encoding="utf-8"))
    known_columns = {
        f"{table['name']}.{column['name']}"
        for table in metadata["tables"]
        for column in table["columns"]
    }
    known_metrics = {metric["name"] for metric in metadata["metrics"]}
    sync_value_columns = {
        f"{table['name']}.{column['name']}"
        for table in metadata["tables"]
        for column in table["columns"]
        if column.get("sync")
    }

    assert {
        column
        for case in cases
        for column in case.expected_columns
        if column not in known_columns
    } == set()
    assert {
        metric
        for case in cases
        for metric in case.expected_metrics
        if metric not in known_metrics
    } == set()
    assert {
        value_id.rsplit(".", 1)[0]
        for case in cases
        for value_id in case.expected_values
        if value_id.rsplit(".", 1)[0] not in sync_value_columns
    } == set()


def test_eval_cases_50_are_scored_for_resume_metrics():
    cases = load_eval_cases(CASES_PATH)

    retrieval_cases = [case for case in cases if "ablation_retrieval" in case.tags]
    guard_cases = [case for case in cases if "ablation_guard" in case.tags]
    cost_cases = [case for case in cases if "ablation_cost" in case.tags]

    assert sum(bool(case.expected_values) for case in retrieval_cases) >= 20
    assert sum(bool(case.expected_time_binding) for case in retrieval_cases) >= 8
    assert sum(len(case.expected_metrics) >= 2 for case in retrieval_cases) >= 8
    assert all(case.expected_blocked_by for case in guard_cases)
    assert all(case.expected_result == {"mode": "non_empty"} for case in cost_cases)


def test_eval_cases_50_use_any_guard_for_safety_not_fixed_layer():
    cases = load_eval_cases(CASES_PATH)
    guard_cases = [case for case in cases if "ablation_guard" in case.tags]

    assert guard_cases
    assert all(case.expected_blocked_by == "any_guard" for case in guard_cases)
