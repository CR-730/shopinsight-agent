"""Resolve numeric predicates without guessing units or target semantics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal

from app.agent.semantic_planning.catalog import (
    ColumnCandidate,
    MetricCandidate,
    SemanticCandidateCatalog,
)
from app.agent.semantic_planning.draft import NumericPredicateMention
from app.agent.semantic_planning.issues import PlanningIssue
from app.agent.semantic_planning.plan import NumericPredicate
from app.agent.semantic_planning.resolvers.common import select_one_candidate

NumericResolutionStatus = Literal["resolved", "unresolved", "ambiguous"]
_PLAIN_DECIMAL = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_NUMERIC_TYPES = (
    "int",
    "decimal",
    "numeric",
    "float",
    "double",
    "real",
)


@dataclass(frozen=True)
class NumericResolutionContext:
    catalog: SemanticCandidateCatalog


@dataclass(frozen=True)
class NumericPredicateResolution:
    status: NumericResolutionStatus
    plan: NumericPredicate | None = None
    issue: PlanningIssue | None = None


def resolve_numeric_predicate(
    mention: NumericPredicateMention,
    context: NumericResolutionContext,
) -> NumericPredicateResolution:
    targets = {**context.catalog.columns, **context.catalog.metrics}
    selection = select_one_candidate(
        raw_text=mention.raw_text,
        candidate_ids=mention.target_candidate_ids,
        catalog=targets,
        issue_prefix="numeric_target",
    )
    if selection.status != "resolved":
        return NumericPredicateResolution(
            status=selection.status,
            issue=selection.issue,
        )

    target = selection.candidate
    if isinstance(target, MetricCandidate):
        target_type = "measure"
        clause = "having"
    else:
        assert isinstance(target, ColumnCandidate)
        if not _is_numeric_type(target.data_type):
            return _blocked(
                "numeric_target_type_invalid", mention, mention.target_candidate_ids
            )
        target_type = "column"
        clause = "where"

    expected_count = 2 if mention.operator_intent == "between" else 1
    if len(mention.value_texts) != expected_count:
        return _blocked(
            "numeric_boundary_count_invalid",
            mention,
            mention.target_candidate_ids,
        )

    normalized_values: list[tuple[Decimal, str]] = []
    for value_text in mention.value_texts:
        if any(unit in value_text for unit in ("万", "亿", "%", "％")):
            return _blocked(
                "numeric_unit_not_declared", mention, mention.target_candidate_ids
            )
        if not _PLAIN_DECIMAL.fullmatch(value_text.strip()):
            return _blocked(
                "numeric_value_invalid", mention, mention.target_candidate_ids
            )
        try:
            decimal_value = Decimal(value_text.strip())
        except InvalidOperation:
            return _blocked(
                "numeric_value_invalid", mention, mention.target_candidate_ids
            )
        normalized_values.append((decimal_value, _decimal_text(decimal_value)))

    if mention.operator_intent == "between":
        normalized_values.sort(key=lambda item: item[0])
    return NumericPredicateResolution(
        status="resolved",
        plan=NumericPredicate(
            target_type=target_type,
            target_id=selection.candidate_id,
            operator=mention.operator_intent,
            values=[text for _, text in normalized_values],
            clause=clause,
        ),
    )


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _is_numeric_type(data_type: str) -> bool:
    normalized = data_type.strip().casefold()
    return any(normalized.startswith(prefix) for prefix in _NUMERIC_TYPES)


def _blocked(
    code: str,
    mention: NumericPredicateMention,
    candidate_ids: list[str],
) -> NumericPredicateResolution:
    return NumericPredicateResolution(
        status="unresolved",
        issue=PlanningIssue(
            phase="resolution",
            code=code,
            source_span=mention.raw_text,
            candidate_ids=list(dict.fromkeys(candidate_ids)),
            details={},
        ),
    )


__all__ = [
    "NumericPredicateResolution",
    "NumericResolutionContext",
    "resolve_numeric_predicate",
]
