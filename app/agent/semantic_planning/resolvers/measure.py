"""Resolve measure mentions against authoritative metric candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.agent.semantic_planning.catalog import (
    MetricCandidate,
    SemanticCandidateCatalog,
)
from app.agent.semantic_planning.draft import MeasureMention
from app.agent.semantic_planning.issues import PlanningIssue
from app.agent.semantic_planning.plan import MeasurePlan
from app.agent.semantic_planning.resolvers.common import select_one_candidate

MeasureResolutionStatus = Literal["resolved", "unresolved", "ambiguous"]
_AGGREGATIONS = {
    "sum",
    "avg",
    "count",
    "count_distinct",
    "min",
    "max",
    "expression",
}


@dataclass(frozen=True)
class MeasureResolutionContext:
    catalog: SemanticCandidateCatalog
    trusted_sources: tuple[str, ...]


@dataclass(frozen=True)
class MeasureResolution:
    status: MeasureResolutionStatus
    plan: MeasurePlan | None = None
    issue: PlanningIssue | None = None


def resolve_measure(
    mention: MeasureMention,
    context: MeasureResolutionContext,
) -> MeasureResolution:
    selection = select_one_candidate(
        raw_text=mention.raw_text,
        candidate_ids=mention.candidate_ids,
        catalog=context.catalog.metrics,
        trusted_sources=context.trusted_sources,
        issue_prefix="metric",
    )
    if selection.status != "resolved":
        return MeasureResolution(status=selection.status, issue=selection.issue)

    metric = selection.candidate
    assert isinstance(metric, MetricCandidate)
    if metric.aggregation not in _AGGREGATIONS:
        return MeasureResolution(
            status="unresolved",
            issue=_issue(
                "metric_definition_missing",
                mention,
                [metric.candidate_id],
            ),
        )
    return MeasureResolution(
        status="resolved",
        plan=MeasurePlan(
            metric_id=metric.candidate_id,
            name=metric.name,
            aggregation=metric.aggregation,
            expression=metric.expression,
            source_column_ids=list(metric.relevant_columns),
            output_alias=mention.raw_text,
        ),
    )


def _issue(
    code: str, mention: MeasureMention, candidate_ids: list[str]
) -> PlanningIssue:
    return PlanningIssue(
        phase="resolution",
        code=code,
        source_span=mention.raw_text,
        candidate_ids=candidate_ids,
        details={},
    )


__all__ = [
    "MeasureResolution",
    "MeasureResolutionContext",
    "resolve_measure",
]
