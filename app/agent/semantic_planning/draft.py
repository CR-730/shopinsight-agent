"""Strict, untrusted semantic evidence emitted by the LLM.

These models deliberately contain only semantic observations and controlled
candidate identifiers. Canonical values, computed dates, SQL expressions, and
join conditions belong to deterministic resolution. The model may only choose
an allowed join type for an exposed relationship identifier.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _StrictDraftModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MeasureMention(_StrictDraftModel):
    raw_text: str
    candidate_ids: list[str] = Field(default_factory=list)


class DimensionMention(_StrictDraftModel):
    raw_text: str
    candidate_ids: list[str] = Field(default_factory=list)
    role: Literal["group_by", "projection"]


class EnumPredicateMention(_StrictDraftModel):
    kind: Literal["enum"] = "enum"
    raw_text: str
    value_candidate_ids: list[str] = Field(default_factory=list)
    operator_intent: Literal["eq", "neq", "in", "not_in"] = "eq"


class NumericPredicateMention(_StrictDraftModel):
    kind: Literal["numeric"] = "numeric"
    raw_text: str
    target_candidate_ids: list[str] = Field(default_factory=list)
    operator_intent: Literal["eq", "gt", "gte", "lt", "lte", "between"]
    value_texts: list[str] = Field(default_factory=list)


class TemporalPredicateMention(_StrictDraftModel):
    kind: Literal["temporal"] = "temporal"
    raw_text: str
    relation_intent: Literal["during", "on", "before", "after", "since", "until"]


PredicateMention = Annotated[
    EnumPredicateMention | NumericPredicateMention | TemporalPredicateMention,
    Field(discriminator="kind"),
]


class OrderMention(_StrictDraftModel):
    raw_text: str
    target_candidate_ids: list[str] = Field(default_factory=list)
    direction: Literal["asc", "desc"]


class LimitMention(_StrictDraftModel):
    raw_text: str


class JoinMention(_StrictDraftModel):
    raw_text: str
    relationship_candidate_id: str
    join_type: Literal["inner", "left"]
    left_table_candidate_id: str | None = None


class AmbiguityReport(_StrictDraftModel):
    raw_text: str
    candidate_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class SemanticDraft(_StrictDraftModel):
    measure_mentions: list[MeasureMention] = Field(default_factory=list)
    dimension_mentions: list[DimensionMention] = Field(default_factory=list)
    predicate_mentions: list[PredicateMention] = Field(default_factory=list)
    order_mentions: list[OrderMention] = Field(default_factory=list)
    limit_mentions: list[LimitMention] = Field(default_factory=list)
    join_mentions: list[JoinMention] = Field(default_factory=list)
    ambiguity_reports: list[AmbiguityReport] = Field(default_factory=list)


__all__ = [
    "AmbiguityReport",
    "DimensionMention",
    "EnumPredicateMention",
    "JoinMention",
    "LimitMention",
    "MeasureMention",
    "NumericPredicateMention",
    "OrderMention",
    "PredicateMention",
    "SemanticDraft",
    "TemporalPredicateMention",
]
