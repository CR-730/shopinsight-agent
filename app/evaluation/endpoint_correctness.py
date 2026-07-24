"""Execution-accuracy scoring for endpoint C/D comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlglot import exp, parse_one

from app.evaluation.cases import EvalCase, results_match


@dataclass(frozen=True)
class EndpointScore:
    correct: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


def score_endpoint_result(
    case: EvalCase,
    *,
    generated_sql: str | None,
    actual_rows: Any,
    oracle_rows: list[dict[str, Any]],
    oracle_full_rows: list[dict[str, Any]] | None = None,
    blocked_by: str | None = None,
) -> EndpointScore:
    """Score one generated SQL by execution result with ambiguity safeguards."""

    if case.expected_blocked_by:
        expected = case.expected_blocked_by
        blocked_as_expected = (
            bool(blocked_by) if expected == "any_guard" else blocked_by == expected
        )
        return EndpointScore(
            blocked_as_expected,
            (
                "expected_safety_block"
                if blocked_as_expected
                else "missing_expected_block"
            ),
            {"expected_blocked_by": expected, "actual_blocked_by": blocked_by},
        )

    if not str(generated_sql or "").strip():
        return EndpointScore(False, "missing_sql")
    if not isinstance(actual_rows, list):
        return EndpointScore(False, "missing_execution_result")

    matched = results_match(
        actual_rows,
        oracle_rows,
        order_sensitive=case.order_sensitive,
        ignore_column_names=True,
    )
    if matched:
        if (
            case.expected_metrics
            and _aggregate_projection_fingerprint(str(generated_sql))
            != _aggregate_projection_fingerprint(case.oracle_sql or "")
        ):
            return EndpointScore(False, "metric_formula_mismatch")
        if _is_empty_like(oracle_rows):
            semantic_match = _semantic_fingerprint(
                str(generated_sql)
            ) == _semantic_fingerprint(case.oracle_sql or "")
            return EndpointScore(
                semantic_match,
                (
                    "empty_result_semantic_match"
                    if semantic_match
                    else "empty_result_semantic_mismatch"
                ),
            )
        return EndpointScore(True, "result_match")

    if _matches_topn_with_ties(
        case,
        actual_rows,
        oracle_rows,
        oracle_full_rows,
    ):
        return EndpointScore(True, "topn_tie_match")
    return EndpointScore(False, "result_mismatch")


def _aggregate_projection_fingerprint(sql: str) -> list[str]:
    statement = parse_one(sql, read="mysql")
    result = []
    for projection in statement.expressions:
        expression = (
            projection.this if isinstance(projection, exp.Alias) else projection
        )
        if not any(expression.find_all(exp.AggFunc)):
            continue
        normalized = expression.copy().transform(_strip_column_table)
        result.append(normalized.sql(dialect="mysql", normalize=True))
    return sorted(result)


def _strip_column_table(node: exp.Expression):
    if not isinstance(node, exp.Column):
        return node
    result = node.copy()
    result.set("table", None)
    return result


def _is_empty_like(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return True
    return all(
        value is None
        for row in rows
        if isinstance(row, dict)
        for value in row.values()
    )


def _matches_topn_with_ties(
    case: EvalCase,
    actual_rows: list[Any],
    oracle_rows: list[Any],
    oracle_full_rows: list[Any] | None,
) -> bool:
    if not case.order_sensitive or not oracle_full_rows:
        return False
    statement = parse_one(case.oracle_sql or "", read="mysql")
    limit = _integer_arg(statement.args.get("limit"))
    order = statement.args.get("order")
    if (
        limit is None
        or limit < 1
        or not order
        or len(order.expressions) != 1
        or len(actual_rows) != limit
        or len(oracle_rows) != limit
        or len(oracle_full_rows) <= limit
    ):
        return False

    score_index = _order_projection_index(statement, order.expressions[0])
    if score_index is None:
        return False
    expected_rows = [_row_values(row) for row in oracle_full_rows]
    actual = [_row_values(row) for row in actual_rows]
    if any(score_index >= len(row) for row in [*expected_rows, *actual]):
        return False

    descending = bool(order.expressions[0].args.get("desc"))
    cutoff = expected_rows[limit - 1][score_index]
    allowed = {
        row
        for row in expected_rows
        if _passes_cutoff(row[score_index], cutoff, descending)
    }
    scores = [row[score_index] for row in actual]
    ordered = all(
        (left >= right if descending else left <= right)
        for left, right in zip(scores, scores[1:])
    )
    return ordered and len(set(actual)) == len(actual) and all(
        row in allowed for row in actual
    )


def _order_projection_index(statement: exp.Expression, ordered: exp.Expression):
    target = ordered.this
    projections = list(statement.expressions)
    if isinstance(target, exp.Column) and not target.table:
        for index, projection in enumerate(projections):
            if projection.alias == target.name:
                return index
    target_sql = target.sql(dialect="mysql", normalize=True)
    for index, projection in enumerate(projections):
        expression = (
            projection.this if isinstance(projection, exp.Alias) else projection
        )
        if expression.sql(dialect="mysql", normalize=True) == target_sql:
            return index
    return None


def _integer_arg(expression: exp.Expression | None) -> int | None:
    if expression is None:
        return None
    value = expression.args.get("expression")
    if isinstance(value, exp.Literal) and not value.is_string:
        return int(value.this)
    return None


def _passes_cutoff(value: Any, cutoff: Any, descending: bool) -> bool:
    return value >= cutoff if descending else value <= cutoff


def _row_values(row: Any) -> tuple[Any, ...]:
    values = row.values() if isinstance(row, dict) else row
    return tuple(_comparable_value(value) for value in values)


def _comparable_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return str(value)


def _semantic_fingerprint(sql: str) -> dict[str, Any]:
    statement = parse_one(sql, read="mysql")
    aliases = {
        table.alias_or_name: table.name for table in statement.find_all(exp.Table)
    }
    tables = sorted({table.name for table in statement.find_all(exp.Table)})
    projection_aliases = {
        projection.alias: projection.this
        for projection in statement.expressions
        if isinstance(projection, exp.Alias) and projection.alias
    }

    projections = [
        _canonical_sql(
            projection.this if isinstance(projection, exp.Alias) else projection,
            aliases,
            tables,
            projection_aliases,
        )
        for projection in statement.expressions
    ]
    where = statement.args.get("where")
    having = statement.args.get("having")
    joins = list(statement.args.get("joins") or [])
    group = statement.args.get("group")
    order = statement.args.get("order")
    return {
        "distinct": bool(statement.args.get("distinct")),
        "tables": tables,
        "projections": projections,
        "where": _predicate_parts(
            where.this if where else None,
            aliases,
            tables,
            projection_aliases,
        ),
        "having": _predicate_parts(
            having.this if having else None,
            aliases,
            tables,
            projection_aliases,
        ),
        "joins": sorted(
            (
                _join_kind(join),
                tuple(
                    _predicate_parts(
                        join.args.get("on"),
                        aliases,
                        tables,
                        projection_aliases,
                    )
                ),
            )
            for join in joins
        ),
        "group_by": sorted(
            _canonical_sql(item, aliases, tables, projection_aliases)
            for item in (group.expressions if group else [])
        ),
        "order_by": [
            (
                _canonical_sql(
                    item.this,
                    aliases,
                    tables,
                    projection_aliases,
                ),
                bool(item.args.get("desc")),
            )
            for item in (order.expressions if order else [])
        ],
        "limit": _integer_arg(statement.args.get("limit")),
        "offset": _integer_arg(statement.args.get("offset")),
    }


def _predicate_parts(
    expression: exp.Expression | None,
    aliases: dict[str, str],
    tables: list[str],
    projection_aliases: dict[str, exp.Expression],
) -> list[str]:
    if expression is None:
        return []
    if isinstance(expression, exp.And):
        return sorted(
            [
                *_predicate_parts(
                    expression.left,
                    aliases,
                    tables,
                    projection_aliases,
                ),
                *_predicate_parts(
                    expression.right,
                    aliases,
                    tables,
                    projection_aliases,
                ),
            ]
        )
    canonical = _canonical_expression(
        expression,
        aliases,
        tables,
        projection_aliases,
    )
    if isinstance(canonical, (exp.EQ, exp.NEQ)):
        left = canonical.left.sql(dialect="mysql", normalize=True)
        right = canonical.right.sql(dialect="mysql", normalize=True)
        if right < left:
            original_left = canonical.left.copy()
            original_right = canonical.right.copy()
            canonical.set("this", original_right)
            canonical.set("expression", original_left)
    return [canonical.sql(dialect="mysql", normalize=True)]


def _canonical_sql(
    expression: exp.Expression,
    aliases: dict[str, str],
    tables: list[str],
    projection_aliases: dict[str, exp.Expression],
) -> str:
    return _canonical_expression(
        expression,
        aliases,
        tables,
        projection_aliases,
    ).sql(dialect="mysql", normalize=True)


def _canonical_expression(
    expression: exp.Expression,
    aliases: dict[str, str],
    tables: list[str],
    projection_aliases: dict[str, exp.Expression],
) -> exp.Expression:
    def transform(node: exp.Expression):
        if isinstance(node, exp.Column):
            if not node.table and node.name in projection_aliases:
                return projection_aliases[node.name].copy()
            result = node.copy()
            if node.table in aliases:
                result.set("table", exp.to_identifier(aliases[node.table]))
            elif not node.table and len(tables) == 1:
                result.set("table", exp.to_identifier(tables[0]))
            return result
        return node

    return expression.copy().transform(transform)


def _join_kind(join: exp.Join) -> str:
    side = str(join.args.get("side") or "").upper()
    kind = str(join.args.get("kind") or "INNER").upper()
    return f"{side} {kind}".strip()


__all__ = ["EndpointScore", "score_endpoint_result"]
