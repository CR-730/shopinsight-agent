"""Resolve LLM-selected time text through one deterministic parser adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from app.agent.semantic_planning.catalog import SemanticCandidateCatalog
from app.agent.semantic_planning.draft import TemporalPredicateMention
from app.agent.semantic_planning.issues import PlanningIssue
from app.agent.semantic_planning.plan import TemporalPredicate
from app.agent.semantic_planning.time_adapter import (
    TimeParseAmbiguousError,
    TimeParserFailure,
    parse_time_span,
)

TemporalResolutionStatus = Literal["resolved", "unresolved", "ambiguous", "failed"]


@dataclass(frozen=True)
class TemporalResolutionContext:
    catalog: SemanticCandidateCatalog
    reference_date: date
    temporal_column_id: str


@dataclass(frozen=True)
class TemporalPredicateResolution:
    status: TemporalResolutionStatus
    plan: TemporalPredicate | None = None
    issue: PlanningIssue | None = None


def resolve_temporal_predicate(
    mention: TemporalPredicateMention,
    context: TemporalResolutionContext,
) -> TemporalPredicateResolution:
    if context.temporal_column_id not in context.catalog.columns:
        return _blocked(
            "unresolved",
            "temporal_column_invalid",
            mention,
            [context.temporal_column_id],
        )
    if any(marker in mention.raw_text for marker in ("同比", "环比")):
        return _blocked("unresolved", "temporal_comparison_unsupported", mention)

    try:
        parsed = parse_time_span(
            mention.raw_text,
            reference_date=context.reference_date,
        )
    except TimeParseAmbiguousError:
        return _blocked("ambiguous", "temporal_ambiguous", mention)
    except TimeParserFailure as exc:
        return TemporalPredicateResolution(
            status="failed",
            issue=PlanningIssue(
                phase="system",
                code="temporal_parser_failed",
                source_span=mention.raw_text,
                candidate_ids=[],
                details={"error_type": exc.__cause__.__class__.__name__},
            ),
        )

    return TemporalPredicateResolution(
        status="resolved",
        plan=TemporalPredicate(
            column_id=context.temporal_column_id,
            operator=mention.relation_intent,
            start_date=parsed.start_date.isoformat(),
            end_date=parsed.end_date.isoformat(),
            grain=parsed.grain,
        ),
    )


def _blocked(
    status: Literal["unresolved", "ambiguous"],
    code: str,
    mention: TemporalPredicateMention,
    candidate_ids: list[str] | None = None,
) -> TemporalPredicateResolution:
    return TemporalPredicateResolution(
        status=status,
        issue=PlanningIssue(
            phase="resolution",
            code=code,
            source_span=mention.raw_text,
            candidate_ids=candidate_ids or [],
            details={},
        ),
    )


__all__ = [
    "TemporalPredicateResolution",
    "TemporalResolutionContext",
    "resolve_temporal_predicate",
]
