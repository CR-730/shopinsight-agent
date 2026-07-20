"""Trusted, JSON-serializable semantic query-plan contracts.

The models in this module contain only canonical values that deterministic
resolution has already validated.  They define data contracts only; parsing,
resolution, and validation behavior live in later semantic-planning stages.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _StrictPlanModel(BaseModel):
    # Pydantic's frozen models are shallowly immutable. Mutable members such as
    # lists remain mutable; deep immutability is intentionally outside this
    # contract task.
    model_config = ConfigDict(extra="forbid", frozen=True)


class MeasurePlan(_StrictPlanModel):
    metric_id: str
    name: str
    aggregation: Literal[
        "sum",
        "avg",
        "count",
        "count_distinct",
        "min",
        "max",
        "expression",
    ]
    expression: str | None = None
    source_column_ids: list[str] = Field(default_factory=list)
    output_alias: str


class DimensionPlan(_StrictPlanModel):
    column_id: str
    role: Literal["group_by", "projection"]
    output_alias: str


class EnumPredicate(_StrictPlanModel):
    kind: Literal["enum"] = "enum"
    column_id: str
    operator: Literal["eq", "neq", "in", "not_in"]
    canonical_values: list[str]
    allowed_sql_literals: list[str] = Field(default_factory=list)


class NumericPredicate(_StrictPlanModel):
    kind: Literal["numeric"] = "numeric"
    target_type: Literal["column", "measure"]
    target_id: str
    operator: Literal["eq", "gt", "gte", "lt", "lte", "between"]
    values: list[str]
    clause: Literal["where", "having"]


class TemporalPredicate(_StrictPlanModel):
    kind: Literal["temporal"] = "temporal"
    column_id: str
    operator: Literal["on", "between", "before", "after", "since", "until", "during"]
    start_date: str | None = None
    end_date: str | None = None
    start_date_id: int | None = None
    end_date_id: int | None = None
    grain: Literal["day", "week", "month", "quarter", "year"]


PredicatePlan = Annotated[
    EnumPredicate | NumericPredicate | TemporalPredicate,
    Field(discriminator="kind"),
]


class OrderByPlan(_StrictPlanModel):
    target_type: Literal["measure", "dimension"]
    target_id: str
    direction: Literal["asc", "desc"]


class JoinPlan(_StrictPlanModel):
    left_column_id: str
    right_column_id: str
    join_type: Literal["inner", "left"]


class PlanProvenance(_StrictPlanModel):
    raw_text: str
    resolved_id: str
    method: str
    evidence: str


class SemanticQueryPlan(_StrictPlanModel):
    version: str
    metadata_version: str
    measures: list[MeasurePlan] = Field(default_factory=list)
    dimensions: list[DimensionPlan] = Field(default_factory=list)
    predicates: list[PredicatePlan] = Field(default_factory=list)
    order_by: list[OrderByPlan] = Field(default_factory=list)
    limit: int | None = None
    joins: list[JoinPlan] = Field(default_factory=list)
    required_table_ids: list[str] = Field(default_factory=list)
    required_column_ids: list[str] = Field(default_factory=list)
    provenance: list[PlanProvenance] = Field(default_factory=list)


__all__ = [
    "DimensionPlan",
    "EnumPredicate",
    "JoinPlan",
    "MeasurePlan",
    "NumericPredicate",
    "OrderByPlan",
    "PlanProvenance",
    "PredicatePlan",
    "SemanticQueryPlan",
    "TemporalPredicate",
]
