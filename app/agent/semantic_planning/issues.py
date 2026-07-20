"""Structured outcomes for deterministic semantic planning."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agent.semantic_planning.plan import SemanticQueryPlan

PlanningPhase = Literal[
    "catalog",
    "interpretation",
    "resolution",
    "validation",
    "system",
]
PlanningStatus = Literal["resolved", "unresolved", "ambiguous", "failed"]


class PlanningIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: PlanningPhase
    code: str
    source_span: str
    candidate_ids: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class SemanticPlanningResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: PlanningStatus
    plan: SemanticQueryPlan | None = None
    issues: list[PlanningIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_outcome(self) -> SemanticPlanningResult:
        if self.status == "resolved":
            if self.plan is None:
                raise ValueError("resolved_plan_required")
            if self.issues:
                raise ValueError("resolved_result_cannot_have_issues")
            return self
        if self.plan is not None:
            raise ValueError("blocked_result_cannot_have_plan")
        if not self.issues:
            raise ValueError("planning_issue_required")
        return self


__all__ = [
    "PlanningIssue",
    "PlanningPhase",
    "PlanningStatus",
    "SemanticPlanningResult",
]
