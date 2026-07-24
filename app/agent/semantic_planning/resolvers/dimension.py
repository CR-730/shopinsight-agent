"""Resolve group-by and projection mentions against controlled columns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.agent.semantic_planning.catalog import (
    ColumnCandidate,
    SemanticCandidateCatalog,
)
from app.agent.semantic_planning.draft import DimensionMention
from app.agent.semantic_planning.issues import PlanningIssue
from app.agent.semantic_planning.plan import DimensionPlan
from app.agent.semantic_planning.resolvers.common import select_one_candidate

DimensionResolutionStatus = Literal["resolved", "unresolved", "ambiguous"]


@dataclass(frozen=True)
class DimensionResolutionContext:
    catalog: SemanticCandidateCatalog


@dataclass(frozen=True)
class DimensionResolution:
    status: DimensionResolutionStatus
    plan: DimensionPlan | None = None
    issue: PlanningIssue | None = None


def resolve_dimension(
    mention: DimensionMention,
    context: DimensionResolutionContext,
) -> DimensionResolution:
    issue_prefix = "group_by" if mention.role == "group_by" else "projection"
    selection = select_one_candidate(
        raw_text=mention.raw_text,
        candidate_ids=mention.candidate_ids,
        catalog=context.catalog.columns,
        issue_prefix=issue_prefix,
    )
    if selection.status != "resolved":
        return DimensionResolution(status=selection.status, issue=selection.issue)

    column = selection.candidate
    assert isinstance(column, ColumnCandidate)
    if mention.role == "group_by" and column.role != "dimension":
        return _blocked("group_by_role_invalid", mention, column.candidate_id)
    if mention.role == "projection" and not column.projectable:
        return _blocked("projection_not_allowed", mention, column.candidate_id)
    return DimensionResolution(
        status="resolved",
        plan=DimensionPlan(
            column_id=column.candidate_id,
            role=mention.role,
            output_alias=mention.raw_text,
        ),
    )


def _blocked(
    code: str,
    mention: DimensionMention,
    candidate_id: str,
) -> DimensionResolution:
    return DimensionResolution(
        status="unresolved",
        issue=PlanningIssue(
            phase="resolution",
            code=code,
            source_span=mention.raw_text,
            candidate_ids=[candidate_id],
            details={},
        ),
    )


__all__ = [
    "DimensionResolution",
    "DimensionResolutionContext",
    "resolve_dimension",
]
