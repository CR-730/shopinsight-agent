"""Canonicalize resolved plan predicates without guessing user intent."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Sequence

from app.agent.predicate_normalization import (
    canonical_number,
    canonical_set_values,
)
from app.agent.semantic_planning.issues import PlanningIssue
from app.agent.semantic_planning.plan import (
    EnumPredicate,
    NumericPredicate,
    PredicatePlan,
)


@dataclass(frozen=True)
class PredicateNormalizationResult:
    predicates: tuple[PredicatePlan, ...]
    issues: tuple[PlanningIssue, ...]


def normalize_plan_predicates(
    predicates: Sequence[PredicatePlan],
) -> PredicateNormalizationResult:
    enum_groups: dict[str, list[tuple[int, EnumPredicate]]] = {}
    passthrough: list[tuple[int, PredicatePlan]] = []
    for index, predicate in enumerate(predicates):
        if isinstance(predicate, EnumPredicate) and _valid_enum_shape(predicate):
            enum_groups.setdefault(predicate.column_id.casefold(), []).append(
                (index, predicate)
            )
        elif isinstance(predicate, NumericPredicate):
            passthrough.append((index, _normalize_numeric_values(predicate)))
        else:
            passthrough.append((index, predicate))

    issues: list[PlanningIssue] = []
    normalized: list[tuple[int, PredicatePlan]] = list(passthrough)
    for items in enum_groups.values():
        group_predicates, issue = _normalize_enum_group(
            [predicate for _, predicate in items]
        )
        if issue is not None:
            issues.append(issue)
            continue
        first_index = min(index for index, _ in items)
        normalized.extend(
            (first_index + offset, predicate)
            for offset, predicate in enumerate(group_predicates)
        )

    numeric_issue = _numeric_conflict(
        [
            predicate
            for _, predicate in normalized
            if isinstance(predicate, NumericPredicate)
        ]
    )
    if numeric_issue is not None:
        issues.append(numeric_issue)
    if issues:
        return PredicateNormalizationResult((), tuple(issues))

    normalized.sort(key=lambda item: item[0])
    deduplicated: list[PredicatePlan] = []
    for _, predicate in normalized:
        if predicate not in deduplicated:
            deduplicated.append(predicate)
    return PredicateNormalizationResult(tuple(deduplicated), ())


def _valid_enum_shape(predicate: EnumPredicate) -> bool:
    if predicate.allowed_sql_literals != predicate.canonical_values:
        return False
    if not predicate.canonical_values:
        return False
    return not (
        predicate.operator in {"eq", "neq"} and len(predicate.canonical_values) != 1
    )


def _normalize_enum_group(
    predicates: list[EnumPredicate],
) -> tuple[list[EnumPredicate], PlanningIssue | None]:
    column_id = predicates[0].column_id
    include_values = canonical_set_values(
        value
        for predicate in predicates
        if predicate.operator == "in"
        for value in predicate.canonical_values
    )
    eq_values = canonical_set_values(
        predicate.canonical_values[0]
        for predicate in predicates
        if predicate.operator == "eq"
    )
    exclude_values = canonical_set_values(
        value
        for predicate in predicates
        if predicate.operator in {"neq", "not_in"}
        for value in predicate.canonical_values
    )

    if len(eq_values) > 1 or (
        eq_values and include_values and not set(eq_values).issubset(include_values)
    ):
        return [], _conflict(column_id, "enum_positive_conflict")

    effective_include = eq_values or include_values
    if effective_include and set(effective_include).issubset(exclude_values):
        return [], _conflict(column_id, "enum_positive_fully_excluded")

    output: list[EnumPredicate] = []
    if eq_values:
        output.append(_enum(column_id, "eq", eq_values))
    elif include_values:
        output.append(_enum(column_id, "in", include_values))

    if exclude_values:
        negative_inputs = [
            predicate
            for predicate in predicates
            if predicate.operator in {"neq", "not_in"}
        ]
        operator = (
            "neq"
            if len(exclude_values) == 1
            and all(item.operator == "neq" for item in negative_inputs)
            else "not_in"
        )
        output.append(_enum(column_id, operator, exclude_values))
    return output, None


def _enum(column_id: str, operator: str, values: tuple[str, ...]) -> EnumPredicate:
    return EnumPredicate(
        column_id=column_id,
        operator=operator,
        canonical_values=list(values),
        allowed_sql_literals=list(values),
    )


def _normalize_numeric_values(predicate: NumericPredicate) -> NumericPredicate:
    try:
        values = [canonical_number(value) for value in predicate.values]
    except InvalidOperation, ValueError:
        return predicate
    return predicate.model_copy(update={"values": values})


def _numeric_conflict(
    predicates: list[NumericPredicate],
) -> PlanningIssue | None:
    groups: dict[tuple[str, str, str], list[NumericPredicate]] = {}
    for predicate in predicates:
        groups.setdefault(
            (predicate.clause, predicate.target_type, predicate.target_id), []
        ).append(predicate)

    for (_clause, _target_type, target_id), items in groups.items():
        issue = _numeric_group_conflict(target_id, items)
        if issue is not None:
            return issue
    return None


def _numeric_group_conflict(
    target_id: str,
    predicates: list[NumericPredicate],
) -> PlanningIssue | None:
    try:
        equals = {
            _finite_decimal(item.values[0])
            for item in predicates
            if item.operator == "eq" and len(item.values) == 1
        }
    except InvalidOperation, ValueError:
        return None
    if len(equals) > 1:
        return _conflict(target_id, "numeric_eq_conflict")

    lower_bounds: list[tuple[Decimal, bool]] = []
    upper_bounds: list[tuple[Decimal, bool]] = []
    try:
        for item in predicates:
            if item.operator in {"gt", "gte"} and len(item.values) == 1:
                lower_bounds.append(
                    (_finite_decimal(item.values[0]), item.operator == "gte")
                )
            elif item.operator in {"lt", "lte"} and len(item.values) == 1:
                upper_bounds.append(
                    (_finite_decimal(item.values[0]), item.operator == "lte")
                )
            elif item.operator == "between" and len(item.values) == 2:
                lower_bounds.append((_finite_decimal(item.values[0]), True))
                upper_bounds.append((_finite_decimal(item.values[1]), True))
    except InvalidOperation, ValueError:
        return None

    lower = max(lower_bounds, default=None, key=lambda item: (item[0], not item[1]))
    upper = min(upper_bounds, default=None, key=lambda item: (item[0], item[1]))
    if (
        lower
        and upper
        and (
            lower[0] > upper[0]
            or (lower[0] == upper[0] and not (lower[1] and upper[1]))
        )
    ):
        return _conflict(target_id, "numeric_range_empty")
    if equals:
        value = next(iter(equals))
        if lower and (value < lower[0] or (value == lower[0] and not lower[1])):
            return _conflict(target_id, "numeric_eq_outside_range")
        if upper and (value > upper[0] or (value == upper[0] and not upper[1])):
            return _conflict(target_id, "numeric_eq_outside_range")
    return None


def _finite_decimal(value: str) -> Decimal:
    return Decimal(canonical_number(value))


def _conflict(target_id: str, reason: str) -> PlanningIssue:
    return PlanningIssue(
        phase="validation",
        code="predicate_conflict",
        source_span="",
        candidate_ids=[target_id],
        details={"reason": reason},
    )


__all__ = [
    "PredicateNormalizationResult",
    "normalize_plan_predicates",
]
