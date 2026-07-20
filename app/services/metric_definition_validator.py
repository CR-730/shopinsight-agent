"""Validate authoritative metric definitions before they enter metadata stores."""

from sqlglot import exp, parse
from sqlglot.errors import ParseError

from app.entities.metric_info import MetricInfo

_SUPPORTED_AGGREGATIONS = {
    "sum",
    "avg",
    "count",
    "count_distinct",
    "min",
    "max",
    "expression",
}


def validate_metric_definition(metric: MetricInfo) -> None:
    """Fail closed when a metric formula exceeds the supported safe subset."""

    if metric.aggregation not in _SUPPORTED_AGGREGATIONS:
        raise ValueError(f"unsupported_metric_aggregation: {metric.aggregation}")
    if not metric.relevant_columns:
        raise ValueError("metric_columns_required")
    if metric.aggregation != "expression":
        if metric.expression:
            raise ValueError("expression_not_allowed")
        return
    if not metric.expression or not metric.expression.strip():
        raise ValueError("expression_required")
    _validate_readonly_scalar_aggregate(
        metric.expression,
        allowed_columns=set(metric.relevant_columns),
    )


def _validate_readonly_scalar_aggregate(
    expression: str,
    *,
    allowed_columns: set[str],
) -> None:
    if ";" in expression:
        raise ValueError("metric_expression_multiple")
    try:
        parsed = parse(expression, read="mysql")
    except ParseError as exc:
        raise ValueError("metric_expression_invalid") from exc
    if len(parsed) != 1:
        raise ValueError("metric_expression_multiple")

    root = parsed[0]
    if isinstance(root, (exp.Query, exp.DML, exp.DDL)) or root.find(exp.Subquery):
        raise ValueError("metric_expression_not_scalar")
    if root.find(exp.Window):
        raise ValueError("metric_expression_window_not_allowed")
    if not any(root.find_all(exp.AggFunc)):
        raise ValueError("metric_expression_aggregate_required")

    allowed = {column.casefold() for column in allowed_columns}
    referenced = {
        _qualified_column_name(column).casefold()
        for column in root.find_all(exp.Column)
    }
    undeclared = referenced - allowed
    if undeclared:
        raise ValueError(
            "undeclared_metric_column: " + ", ".join(sorted(undeclared))
        )


def _qualified_column_name(column: exp.Column) -> str:
    if not column.table:
        return column.name
    return f"{column.table}.{column.name}"


__all__ = ["validate_metric_definition"]
