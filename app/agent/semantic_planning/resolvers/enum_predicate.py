"""Resolve enum mentions from controlled value candidates only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.agent.semantic_planning.catalog import SemanticCandidateCatalog
from app.agent.semantic_planning.draft import EnumPredicateMention
from app.agent.semantic_planning.issues import PlanningIssue
from app.agent.semantic_planning.plan import EnumPredicate

EnumResolutionStatus = Literal["resolved", "unresolved", "ambiguous", "failed"]


@dataclass(frozen=True)
class EnumResolutionContext:
    catalog: SemanticCandidateCatalog


@dataclass(frozen=True)
class EnumPredicateResolution:
    status: EnumResolutionStatus
    plan: EnumPredicate | None = None
    issue: PlanningIssue | None = None


async def resolve_enum_predicate(
    mention: EnumPredicateMention,
    context: EnumResolutionContext,
) -> EnumPredicateResolution:
    invalid_value_ids = _invalid_ids(
        mention.value_candidate_ids, context.catalog.values
    )
    if invalid_value_ids:
        return _blocked(
            "unresolved", "invalid_candidate_id", mention, invalid_value_ids
        )

    value_ids = list(dict.fromkeys(mention.value_candidate_ids))
    if not value_ids:
        return _blocked("unresolved", "value_not_bound", mention, [])
    return await _resolve_catalog_values(mention, value_ids, context)


async def _resolve_catalog_values(
    mention: EnumPredicateMention,
    value_ids: list[str],
    context: EnumResolutionContext,
) -> EnumPredicateResolution:
    values = [context.catalog.value_by_id(candidate_id) for candidate_id in value_ids]
    owning_columns = list(dict.fromkeys(value.column_id for value in values))
    if len(owning_columns) != 1:
        return _blocked("ambiguous", "filter_column_ambiguous", mention, value_ids)
    column_id = owning_columns[0]

    if mention.operator_intent in {"eq", "neq"} and len(value_ids) != 1:
        return _blocked("ambiguous", "value_ambiguous", mention, value_ids)

    canonical_values = list(dict.fromkeys(value.canonical_value for value in values))
    if len(canonical_values) != len(values):
        return _blocked("unresolved", "duplicate_value_candidate", mention, value_ids)

    return EnumPredicateResolution(
        status="resolved",
        plan=EnumPredicate(
            column_id=column_id,
            operator=mention.operator_intent,
            canonical_values=canonical_values,
        ),
    )


def _blocked(
    status: Literal["unresolved", "ambiguous"],
    code: str,
    mention: EnumPredicateMention,
    candidate_ids: list[str],
) -> EnumPredicateResolution:
    return EnumPredicateResolution(
        status=status,
        issue=PlanningIssue(
            phase="resolution",
            code=code,
            source_span=mention.raw_text,
            candidate_ids=list(dict.fromkeys(candidate_ids)),
            details={},
        ),
    )


def _invalid_ids(candidate_ids: list[str], catalog) -> list[str]:
    return [
        candidate_id
        for candidate_id in dict.fromkeys(candidate_ids)
        if candidate_id not in catalog
    ]


__all__ = [
    "EnumPredicateResolution",
    "EnumResolutionContext",
    "resolve_enum_predicate",
]
