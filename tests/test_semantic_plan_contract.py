from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.agent.semantic_planning.issues import (
    PlanningIssue,
    SemanticPlanningResult,
)
from app.agent.semantic_planning.plan import (
    DimensionPlan,
    EnumPredicate,
    JoinPlan,
    MeasurePlan,
    NumericPredicate,
    OrderByPlan,
    PlanProvenance,
    RequiredColumnPlan,
    SemanticQueryPlan,
    TemporalPredicate,
)


def make_plan(**overrides: object) -> SemanticQueryPlan:
    payload: dict[str, object] = {
        "version": "1.0",
        "metadata_version": "meta-v1",
        "measures": [
            {
                "metric_id": "GMV",
                "name": "销售额",
                "aggregation": "sum",
                "expression": None,
                "source_column_ids": ["fact_order.order_amount"],
                "output_alias": "销售额",
            }
        ],
        "dimensions": [
            {
                "column_id": "dim_region.region_name",
                "role": "group_by",
                "output_alias": "地区",
            }
        ],
        "predicates": [
            {
                "kind": "enum",
                "column_id": "dim_region.region_name",
                "operator": "in",
                "canonical_values": ["华北地区", "华南地区"],
            }
        ],
        "order_by": [
            {"target_type": "measure", "target_id": "GMV", "direction": "desc"}
        ],
        "limit": 5,
        "joins": [
            {
                "left_column_id": "fact_order.region_id",
                "right_column_id": "dim_region.region_id",
                "join_type": "inner",
            }
        ],
        "required_table_ids": ["fact_order", "dim_region"],
        "required_column_ids": [
            "fact_order.order_amount",
            "fact_order.region_id",
            "dim_region.region_id",
            "dim_region.region_name",
        ],
        "provenance": [
            {
                "raw_text": "华北和华南销售额前五",
                "resolved_id": "GMV",
                "method": "metric_alias",
                "evidence": "销售额 -> GMV",
            }
        ],
    }
    payload.update(overrides)
    return SemanticQueryPlan.model_validate(payload)


def test_plan_has_stable_top_level_fields():
    assert set(SemanticQueryPlan.model_fields) == {
        "version",
        "metadata_version",
        "measures",
        "dimensions",
        "predicates",
        "order_by",
        "limit",
        "joins",
        "required_table_ids",
        "required_column_ids",
        "required_columns",
        "provenance",
    }


def test_plan_models_cover_the_approved_contract():
    assert set(MeasurePlan.model_fields) == {
        "metric_id",
        "name",
        "aggregation",
        "expression",
        "source_column_ids",
        "output_alias",
    }
    assert set(DimensionPlan.model_fields) == {"column_id", "role", "output_alias"}
    assert set(OrderByPlan.model_fields) == {"target_type", "target_id", "direction"}
    assert set(JoinPlan.model_fields) == {
        "left_column_id",
        "right_column_id",
        "join_type",
    }
    assert set(RequiredColumnPlan.model_fields) == {"column_id", "data_type"}
    assert set(PlanProvenance.model_fields) == {
        "raw_text",
        "resolved_id",
        "method",
        "evidence",
    }


def test_predicates_are_discriminated_and_json_serializable():
    plan = make_plan(
        predicates=[
            {
                "kind": "numeric",
                "target_type": "measure",
                "target_id": "GMV",
                "operator": "between",
                "values": ["10000.00", "20000.00"],
                "clause": "having",
            },
            {
                "kind": "temporal",
                "column_id": "dim_date.full_date",
                "operator": "between",
                "start_date": "2025-01-01",
                "end_date": "2025-03-31",
                "start_date_id": 20250101,
                "end_date_id": 20250331,
                "grain": "quarter",
            },
        ]
    )

    assert isinstance(plan.predicates[0], NumericPredicate)
    assert isinstance(plan.predicates[1], TemporalPredicate)
    assert plan.predicates[0].values == ["10000.00", "20000.00"]
    assert json.loads(json.dumps(plan.model_dump(mode="json"), ensure_ascii=False))


def test_enum_predicate_uses_only_plural_canonical_values():
    assert "canonical_values" in EnumPredicate.model_fields
    assert "canonical_value" not in EnumPredicate.model_fields
    assert "allowed_sql_literals" not in EnumPredicate.model_fields

    predicate = EnumPredicate.model_validate(
        {
            "kind": "enum",
            "column_id": "dim_region.region_name",
            "operator": "in",
            "canonical_values": ["华北地区", "华南地区"],
        }
    )
    assert predicate.canonical_values == ["华北地区", "华南地区"]


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            MeasurePlan,
            {
                "metric_id": "GMV",
                "name": "销售额",
                "aggregation": "sum",
                "expression": None,
                "source_column_ids": ["fact_order.order_amount"],
                "output_alias": "销售额",
                "unexpected": True,
            },
        ),
        (
            DimensionPlan,
            {
                "column_id": "dim_region.region_name",
                "role": "group_by",
                "output_alias": "地区",
                "unexpected": True,
            },
        ),
        (
            EnumPredicate,
            {
                "kind": "enum",
                "column_id": "dim_region.region_name",
                "operator": "eq",
                "canonical_values": ["华北地区"],
                "unexpected": True,
            },
        ),
        (
            NumericPredicate,
            {
                "kind": "numeric",
                "target_type": "measure",
                "target_id": "GMV",
                "operator": "gt",
                "values": ["10000"],
                "clause": "having",
                "unexpected": True,
            },
        ),
        (
            TemporalPredicate,
            {
                "kind": "temporal",
                "column_id": "dim_date.full_date",
                "operator": "between",
                "start_date": "2025-01-01",
                "end_date": "2025-03-31",
                "start_date_id": 20250101,
                "end_date_id": 20250331,
                "grain": "quarter",
                "unexpected": True,
            },
        ),
        (
            OrderByPlan,
            {
                "target_type": "measure",
                "target_id": "GMV",
                "direction": "desc",
                "unexpected": True,
            },
        ),
        (
            JoinPlan,
            {
                "left_column_id": "fact_order.region_id",
                "right_column_id": "dim_region.region_id",
                "join_type": "inner",
                "unexpected": True,
            },
        ),
        (
            PlanProvenance,
            {
                "raw_text": "销售额",
                "resolved_id": "GMV",
                "method": "metric_alias",
                "evidence": "销售额 -> GMV",
                "unexpected": True,
            },
        ),
        (
            SemanticQueryPlan,
            {
                "version": "1.0",
                "metadata_version": "meta-v1",
                "measures": [],
                "dimensions": [],
                "predicates": [],
                "order_by": [],
                "limit": None,
                "joins": [],
                "required_table_ids": [],
                "required_column_ids": [],
                "provenance": [],
                "unexpected": True,
            },
        ),
    ],
)
def test_every_plan_model_rejects_extra_fields(model, payload):
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_plan_rejects_issue_buckets():
    with pytest.raises(ValidationError):
        make_plan(unresolved=[{"code": "metric_not_bound"}])


def test_contract_rejects_invalid_enums_and_numeric_numbers():
    with pytest.raises(ValidationError):
        MeasurePlan.model_validate(
            {
                "metric_id": "GMV",
                "name": "销售额",
                "aggregation": "median",
                "source_column_ids": ["fact_order.order_amount"],
                "output_alias": "销售额",
            }
        )
    with pytest.raises(ValidationError):
        DimensionPlan.model_validate(
            {
                "column_id": "dim_region.region_name",
                "role": "filter",
                "output_alias": "地区",
            }
        )
    with pytest.raises(ValidationError):
        NumericPredicate.model_validate(
            {
                "kind": "numeric",
                "target_type": "measure",
                "target_id": "GMV",
                "operator": "gt",
                "values": [10000],
                "clause": "having",
            }
        )
    with pytest.raises(ValidationError):
        OrderByPlan.model_validate(
            {"target_type": "table", "target_id": "fact_order", "direction": "desc"}
        )


def test_numeric_predicate_rejects_unapproved_not_equal_operator():
    with pytest.raises(ValidationError):
        NumericPredicate.model_validate(
            {
                "kind": "numeric",
                "target_type": "measure",
                "target_id": "GMV",
                "operator": "neq",
                "values": ["10000"],
                "clause": "having",
            }
        )


def test_join_requires_column_ids():
    join = JoinPlan(
        left_column_id="fact_order.region_id",
        right_column_id="dim_region.region_id",
        join_type="inner",
    )
    assert join.left_column_id == "fact_order.region_id"
    assert join.right_column_id == "dim_region.region_id"
    assert join.join_type == "inner"


def test_trusted_plan_rejects_attribute_assignment():
    plan = make_plan()

    with pytest.raises(ValidationError):
        plan.limit = 10


def test_resolved_planning_result_requires_one_trusted_plan():
    result = SemanticPlanningResult(status="resolved", plan=make_plan(), issues=[])

    assert result.plan == make_plan()
    with pytest.raises(ValidationError):
        SemanticPlanningResult(status="resolved", plan=None, issues=[])


def test_blocked_or_failed_result_cannot_carry_a_plan():
    issue = PlanningIssue(
        phase="resolution",
        code="value_ambiguous",
        source_span="华南",
        candidate_ids=["value-1", "value-2"],
        details={"column_count": 2},
    )

    for status in ("unresolved", "ambiguous", "failed"):
        result = SemanticPlanningResult(status=status, plan=None, issues=[issue])
        assert result.plan is None
        with pytest.raises(ValidationError):
            SemanticPlanningResult(status=status, plan=make_plan(), issues=[issue])


def test_non_resolved_result_requires_a_structured_issue():
    with pytest.raises(ValidationError):
        SemanticPlanningResult(status="ambiguous", plan=None, issues=[])


def test_planning_issue_and_result_reject_extra_fields():
    with pytest.raises(ValidationError):
        PlanningIssue(
            phase="resolution",
            code="value_not_found",
            source_span="火星",
            candidate_ids=[],
            details={},
            reason="free-form legacy reason",
        )

    with pytest.raises(ValidationError):
        SemanticPlanningResult(
            status="resolved",
            plan=make_plan(),
            issues=[],
            semantic_draft={},
        )
