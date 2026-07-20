"""Resolve enum mentions from controlled value candidates only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.agent.semantic_planning.catalog import (
    ColumnCandidate,
    SemanticCandidateCatalog,
)
from app.agent.semantic_planning.draft import EnumPredicateMention
from app.agent.semantic_planning.issues import PlanningIssue
from app.agent.semantic_planning.plan import EnumPredicate

EnumResolutionStatus = Literal["resolved", "unresolved", "ambiguous", "failed"]


@dataclass(frozen=True)
class EnumResolutionContext:
    catalog: SemanticCandidateCatalog
    dw_repository: Any
    trusted_sources: tuple[str, ...]


@dataclass(frozen=True)
class EnumPredicateResolution:
    status: EnumResolutionStatus
    plan: EnumPredicate | None = None
    issue: PlanningIssue | None = None


async def resolve_enum_predicate(
    mention: EnumPredicateMention,
    context: EnumResolutionContext,
) -> EnumPredicateResolution:
    if not _is_trusted_span(mention.raw_text, context.trusted_sources):
        return _blocked(
            "unresolved",
            "untrusted_source_span",
            mention,
            mention.value_candidate_ids,
        )

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
        return _blocked(
            "ambiguous", "filter_column_ambiguous", mention, value_ids
        )
    column_id = owning_columns[0]

    if mention.operator_intent in {"eq", "neq"} and len(value_ids) != 1:
        return _blocked("ambiguous", "value_ambiguous", mention, value_ids)

    canonical_values = list(
        dict.fromkeys(value.canonical_value for value in values)
    )
    if len(canonical_values) != len(values):
        return _blocked(
            "unresolved", "duplicate_value_candidate", mention, value_ids
        )

    column = context.catalog.column_by_id(column_id)
    for value in values:
        if value.source != "meta_alias":
            continue
        verified = await _exact_value_exists(
            column,
            value.canonical_value,
            mention,
            value_ids,
            context,
        )
        if isinstance(verified, EnumPredicateResolution):
            return verified

    return EnumPredicateResolution(
        status="resolved",
        plan=EnumPredicate(
            column_id=column_id,
            operator=mention.operator_intent,
            canonical_values=canonical_values,
            allowed_sql_literals=list(canonical_values),
        ),
    )


async def _exact_value_exists(
    column: ColumnCandidate,
    value: str,
    mention: EnumPredicateMention,
    candidate_ids: list[str],
    context: EnumResolutionContext,
) -> bool | EnumPredicateResolution:
    try:
        exists = await context.dw_repository.column_value_exists(
            column.table, column.name, value
        )
    except Exception as exc:
        return EnumPredicateResolution(
            status="failed",
            issue=PlanningIssue(
                phase="system",
                code="dw_value_lookup_failed",
                source_span=mention.raw_text,
                candidate_ids=candidate_ids,
                details={"error_type": exc.__class__.__name__},
            ),
        )
    if not exists:
        return _blocked(
            "unresolved", "value_not_found", mention, candidate_ids
        )
    return True


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


def _is_trusted_span(raw_text: str, trusted_sources: tuple[str, ...]) -> bool:
    return bool(raw_text) and any(raw_text in source for source in trusted_sources)


__all__ = [
    "EnumPredicateResolution",
    "EnumResolutionContext",
    "resolve_enum_predicate",
]
