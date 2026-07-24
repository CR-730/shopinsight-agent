from types import SimpleNamespace

import pytest

from app.evaluation.cases import EvalCase
from app.evaluation.retrieval_ab import (
    RetrievalCandidates,
    apply_candidate_budget,
    score_retrieval_case,
    summarize_retrieval_scores,
)
from app.scripts.run_retrieval_ab import (
    _baseline_route_keywords,
    _current_route_keywords,
    _top_raw_points_by_score,
)


def _case() -> EvalCase:
    return EvalCase(
        id="north-sales",
        query="华北销售额",
        expected_retrieved_columns=[
            "fact_order.order_amount",
            "dim_region.region_name",
        ],
        expected_metrics=["GMV"],
        expected_values=["value:north"],
    )


def test_retrieval_score_is_candidate_coverage_not_case_pass_rate():
    score = score_retrieval_case(
        _case(),
        RetrievalCandidates(
            columns=["fact_order.order_amount"],
            metrics=["GMV"],
            values=["value:north"],
        ),
    )

    assert score.hit_count == 3
    assert score.gold_count == 4
    assert score.recall == 0.75
    assert score.components["columns"].recall == 0.5
    assert score.components["metrics"].recall == 1.0
    assert score.components["values"].recall == 1.0


def test_retrieval_summary_reports_macro_average_and_component_coverage():
    first = score_retrieval_case(
        _case(),
        RetrievalCandidates(
            columns=["fact_order.order_amount"],
            metrics=["GMV"],
            values=["value:north"],
        ),
    )
    second = score_retrieval_case(
        EvalCase(
            id="orders",
            query="订单数",
            expected_retrieved_columns=["fact_order.order_id"],
            expected_metrics=["ORDER_COUNT"],
        ),
        RetrievalCandidates(
            columns=["fact_order.order_id"],
            metrics=[],
            values=[],
        ),
    )

    summary = summarize_retrieval_scores([first, second])

    assert summary["case_count"] == 2
    assert summary["average_recall"] == 0.625
    assert summary["components"]["columns"]["recall"] == 2 / 3
    assert summary["components"]["metrics"]["recall"] == 0.5


def test_retrieval_summary_deduplicates_repeated_gold_candidates():
    repeated_case = EvalCase(
        id="north-orders",
        query="华北订单量",
        expected_retrieved_columns=["dim_region.region_name"],
        expected_metrics=["ORDER_COUNT"],
        expected_values=["value:north", "value:south"],
    )
    first = score_retrieval_case(
        _case(),
        RetrievalCandidates(
            columns=["fact_order.order_amount"],
            metrics=["GMV"],
            values=["value:north"],
        ),
    )
    second = score_retrieval_case(
        repeated_case,
        RetrievalCandidates(
            columns=["dim_region.region_name"],
            metrics=["ORDER_COUNT"],
            values=[],
        ),
    )

    summary = summarize_retrieval_scores([first, second])

    assert summary["unique_gold"]["gold_count"] == 6
    assert summary["unique_gold"]["hit_count"] == 5
    assert summary["unique_gold"]["recall"] == 5 / 6
    assert summary["unique_gold"]["components"]["values"] == {
        "hit_count": 1,
        "gold_count": 2,
        "recall": 0.5,
    }


def test_candidate_budget_is_applied_per_retrieval_route():
    candidates = RetrievalCandidates(
        columns=[f"column:{index}" for index in range(25)],
        metrics=[f"metric:{index}" for index in range(22)],
        values=[f"value:{index}" for index in range(21)],
    )

    budgeted = apply_candidate_budget(candidates, limit=20)

    assert budgeted.columns == candidates.columns[:20]
    assert budgeted.metrics == candidates.metrics[:20]
    assert budgeted.values == candidates.values[:20]


@pytest.mark.anyio
async def test_baseline_uses_full_query_and_jieba_without_llm(monkeypatch):
    def fake_extract_tags(query, **kwargs):
        assert query == "统计华北地区销售额"
        return ["华北", "地区", "销售额"]

    monkeypatch.setattr(
        "app.agent.retrieval_context.jieba.analyse.extract_tags",
        fake_extract_tags,
    )

    result = await _baseline_route_keywords("统计华北地区销售额")

    assert result == {
        "columns": ["统计华北地区销售额", "华北", "地区", "销售额"],
        "metrics": ["统计华北地区销售额", "华北", "地区", "销售额"],
        "values": ["统计华北地区销售额", "华北", "地区", "销售额"],
    }


@pytest.mark.anyio
async def test_current_queries_use_only_full_query_and_route_expansion(monkeypatch):
    expansions = {
        "extend_keywords_for_column_recall": ["地区"],
        "extend_keywords_for_metric_recall": ["销售额", "GMV"],
        "extend_keywords_for_value_recall": ["华北"],
    }

    async def fake_extend_keywords(*, prompt_name, query, step, context):
        assert query == "统计华北地区销售额"
        return expansions[prompt_name]

    monkeypatch.setattr(
        "app.scripts.run_retrieval_ab._extend_keywords",
        fake_extend_keywords,
    )

    result = await _current_route_keywords(
        "统计华北地区销售额",
        {"cost_tracker": object()},
    )

    assert result == {
        "columns": ["统计华北地区销售额", "地区"],
        "metrics": ["统计华北地区销售额", "销售额", "GMV"],
        "values": ["统计华北地区销售额", "华北"],
    }


def test_baseline_top5_keeps_distinct_points_without_candidate_grouping():
    points = _top_raw_points_by_score(
        [
            [
                SimpleNamespace(id="field-name", score=0.95),
                SimpleNamespace(id="field-alias", score=0.94),
                SimpleNamespace(id="other-name", score=0.90),
            ],
            [
                SimpleNamespace(id="field-name", score=0.93),
                SimpleNamespace(id="field-description", score=0.92),
                SimpleNamespace(id="third-name", score=0.89),
            ],
        ],
        limit=5,
    )

    assert [point.id for point in points] == [
        "field-name",
        "field-alias",
        "field-description",
        "other-name",
        "third-name",
    ]
