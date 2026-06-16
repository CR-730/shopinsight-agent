from collections import Counter
from pathlib import Path

import yaml

from app.evaluation.cases import load_eval_cases

CASES_PATH = Path("examples/eval_cases_110.yaml")
META_CONFIG_PATH = Path("conf/meta_config.yaml")


def test_eval_cases_110_count_uniqueness_and_ablation_coverage():
    cases = load_eval_cases(CASES_PATH)

    assert len(cases) == 110
    assert len({case.id for case in cases}) == 110
    assert len({case.query for case in cases}) == 110

    tag_counts = Counter(tag for case in cases for tag in case.tags)
    assert tag_counts["ablation_retrieval"] >= 50
    assert tag_counts["ablation_guard"] >= 25
    assert tag_counts["ablation_cost"] >= 35
    assert tag_counts["sql_memory"] >= 15

    suites = Counter(case.suite for case in cases)
    assert suites["adversarial"] >= 25
    assert suites["realistic"] >= 25
    assert suites["regression"] >= 40


def test_eval_cases_110_expected_metadata_exists():
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

    missing_columns = {
        column
        for case in cases
        for column in case.expected_columns
        if column not in known_columns
    }
    missing_metrics = {
        metric
        for case in cases
        for metric in case.expected_metrics
        if metric not in known_metrics
    }
    invalid_value_columns = {
        value_id.rsplit(".", 1)[0]
        for case in cases
        for value_id in case.expected_values
        if value_id.rsplit(".", 1)[0] not in sync_value_columns
    }

    assert missing_columns == set()
    assert missing_metrics == set()
    assert invalid_value_columns == set()


def test_eval_cases_110_positive_and_guard_cases_are_scored_differently():
    cases = load_eval_cases(CASES_PATH)

    for case in cases:
        assert case.business_source
        assert case.tags
        assert case.risk_points
        assert case.forbidden_sql
        assert case.fatal_errors
        assert len(case.capabilities) <= 6

        if case.suite == "adversarial":
            assert "ablation_guard" in case.tags
        if case.expected_blocked_by:
            assert "safety" in case.capabilities
            assert case.expected_result is None
            assert case.expected_blocked_by in {"pre_rag_guard", "business_binding"}
            assert not case.expected_sql_contains
            assert not case.expected_columns
        else:
            assert case.expected_result == {"mode": "non_empty"}
            assert case.expected_sql_contains
            assert case.expected_columns
            assert case.must_call_tools


def test_eval_cases_110_have_enough_context_labels_for_retrieval_scoring():
    cases = load_eval_cases(CASES_PATH)
    retrieval_cases = [case for case in cases if "ablation_retrieval" in case.tags]

    assert len(retrieval_cases) >= 50
    assert sum(bool(case.expected_columns) for case in retrieval_cases) >= 45
    assert sum(bool(case.expected_metrics) for case in retrieval_cases) >= 40
    assert sum(bool(case.expected_values) for case in retrieval_cases) >= 25
    assert sum("rag_value_hybrid_recall" in case.capabilities for case in retrieval_cases) >= 20
