"""Compare generated SQL semantics with a validated SemanticQueryPlan."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import InvalidOperation
from typing import Any, Sequence

from sqlglot import expressions as exp
from sqlglot import parse, parse_one
from sqlglot.errors import ParseError

from app.agent.predicate_normalization import (
    canonical_number,
    canonical_set_values,
)
from app.agent.semantic_planning.plan import (
    EnumPredicate,
    NumericPredicate,
    SemanticQueryPlan,
    TemporalPredicate,
)


@dataclass(frozen=True)
class SqlPlanDifference:
    code: str
    path: str
    expected: object
    actual: object


@dataclass(frozen=True)
class SqlPlanConsistencyResult:
    ok: bool
    differences: Sequence[SqlPlanDifference]


@dataclass(frozen=True)
class _SqlContext:
    aliases: dict[str, str]
    column_ids_by_name: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class _JoinSignature:
    left_column_id: str
    right_column_id: str
    join_type: str

    @property
    def endpoints(self) -> frozenset[str]:
        return frozenset({self.left_column_id, self.right_column_id})


def validate_sql_plan_consistency(
    sql: str,
    semantic_plan: SemanticQueryPlan | dict[str, Any],
) -> SqlPlanConsistencyResult:
    """Fail closed when the SQL adds, removes, or changes plan semantics."""

    plan = (
        semantic_plan
        if isinstance(semantic_plan, SemanticQueryPlan)
        else SemanticQueryPlan.model_validate(semantic_plan)
    )
    try:
        statements = parse(sql, read="mysql")
    except ParseError as exc:
        return _result([_difference("sql_parse_failed", "sql", "SELECT", str(exc))])
    if len(statements) != 1:
        return _result(
            [
                _difference(
                    "sql_statement_count_invalid",
                    "sql",
                    1,
                    len(statements),
                )
            ]
        )
    expression = statements[0]
    if not isinstance(expression, exp.Select):
        return _result([_difference("sql_not_select", "sql", "SELECT", expression.key)])

    context = _sql_context(expression, plan)
    differences: list[SqlPlanDifference] = []
    expected_select, expected_targets = _expected_select(plan)
    actual_select = _actual_select(expression, context)
    if len(expression.expressions) != len(expected_select):
        differences.append(
            _difference(
                "select_item_count_mismatch",
                "select",
                len(expected_select),
                len(expression.expressions),
            )
        )
    _compare_select(expected_select, actual_select, plan, differences)
    _compare_groups(expression, plan, context, differences)
    _compare_tables(expression, plan, differences)
    _compare_joins(expression, plan, context, differences)
    _compare_predicates(expression, plan, context, differences)
    _compare_order(
        expression,
        plan,
        context,
        expected_targets,
        differences,
    )
    _compare_limit(expression, plan, differences)
    _compare_offset(expression, differences)
    return _result(differences)


def _expected_select(plan):
    context = _plan_context(plan)
    selected: dict[str, str] = {}
    targets: dict[str, tuple[str, str]] = {}
    for measure in plan.measures:
        selected[measure.output_alias.casefold()] = _measure_signature(measure, context)
        targets[measure.output_alias.casefold()] = ("measure", measure.metric_id)
    for dimension in plan.dimensions:
        selected[dimension.output_alias.casefold()] = (
            f"col:{dimension.column_id.casefold()}"
        )
        targets[dimension.output_alias.casefold()] = (
            "dimension",
            dimension.column_id,
        )
    return selected, targets


def _actual_select(expression, context):
    selected: dict[str, str] = {}
    for item in expression.expressions:
        alias = item.alias
        inner = item.this if isinstance(item, exp.Alias) else item
        key = str(alias or inner.sql(dialect="mysql")).casefold()
        selected[key] = _expression_signature(inner, context)
    return selected


def _compare_select(expected, actual, plan, differences):
    measure_aliases = {item.output_alias.casefold() for item in plan.measures}
    for alias, signature in expected.items():
        if alias not in actual:
            code = (
                "measure_missing" if alias in measure_aliases else "dimension_missing"
            )
            differences.append(_difference(code, f"select.{alias}", signature, None))
        elif actual[alias] != signature:
            code = (
                "metric_aggregation_mismatch"
                if alias in measure_aliases
                else "dimension_expression_mismatch"
            )
            differences.append(
                _difference(code, f"select.{alias}", signature, actual[alias])
            )
    for alias in actual.keys() - expected.keys():
        differences.append(
            _difference("select_item_extra", f"select.{alias}", None, actual[alias])
        )


def _compare_groups(expression, plan, context, differences):
    expected = {
        f"col:{item.column_id.casefold()}"
        for item in plan.dimensions
        if item.role == "group_by"
    }
    group = expression.args.get("group")
    actual = {
        _expression_signature(item, context)
        for item in (group.expressions if group else [])
    }
    for item in sorted(expected - actual):
        differences.append(_difference("group_by_missing", "group_by", item, None))
    for item in sorted(actual - expected):
        differences.append(_difference("group_by_extra", "group_by", None, item))


def _compare_tables(expression, plan, differences):
    actual = {table.name.casefold() for table in expression.find_all(exp.Table)}
    expected = {table_id.casefold() for table_id in plan.required_table_ids}
    for table_id in sorted(expected - actual):
        differences.append(_difference("table_missing", "tables", table_id, None))
    for table_id in sorted(actual - expected):
        differences.append(_difference("table_extra", "tables", None, table_id))


def _compare_joins(expression, plan, context, differences):
    expected = [
        _JoinSignature(
            left_column_id=join.left_column_id.casefold(),
            right_column_id=join.right_column_id.casefold(),
            join_type=join.join_type,
        )
        for join in plan.joins
    ]
    actual: list[_JoinSignature] = []
    invalid_endpoints: list[frozenset[str]] = []
    unsupported_types: list[str] = []
    available_table_ids = {_from_table_id(expression, context)}
    for join in expression.args.get("joins") or []:
        joined_table = _joined_table_id(join, context)
        join_type = _actual_join_type(join)
        if join_type is None:
            unsupported_types.append(
                str(join.side or join.kind or "unknown").casefold()
            )
            continue
        condition = join.args.get("on")
        if condition is None:
            invalid_endpoints.append(frozenset({"<missing-on>"}))
            continue
        for predicate in _flatten_and(condition):
            predicate = _unwrap_parens(predicate)
            if not isinstance(predicate, exp.EQ):
                continue
            left = _column_id(predicate.left, context)
            right = _column_id(predicate.right, context)
            if left and right:
                endpoint_tables = {
                    left.rsplit(".", 1)[0],
                    right.rsplit(".", 1)[0],
                }
                prior_tables = endpoint_tables - {joined_table}
                if joined_table not in endpoint_tables or not prior_tables.issubset(
                    available_table_ids
                ):
                    differences.append(
                        _difference(
                            "join_scope_invalid",
                            f"joins.{joined_table}",
                            {
                                "joined_table": joined_table,
                                "available_tables": sorted(available_table_ids),
                            },
                            predicate.sql(dialect="mysql"),
                        )
                    )
                if join_type == "left":
                    if left.rsplit(".", 1)[0] == joined_table:
                        left, right = right, left
                actual.append(
                    _JoinSignature(
                        left_column_id=left,
                        right_column_id=right,
                        join_type=join_type,
                    )
                )
        available_table_ids.add(joined_table)

    if unsupported_types:
        differences.append(
            _difference(
                "join_type_unsupported",
                "joins",
                ["inner", "left"],
                unsupported_types,
            )
        )
        return
    expected_endpoints = Counter(item.endpoints for item in expected)
    actual_endpoints = Counter(item.endpoints for item in actual)
    actual_endpoints.update(invalid_endpoints)
    if expected_endpoints != actual_endpoints:
        _report_join_endpoint_difference(
            expected_endpoints,
            actual_endpoints,
            differences,
        )
        return

    remaining = list(actual)
    for planned in expected:
        index = next(
            index
            for index, candidate in enumerate(remaining)
            if candidate.endpoints == planned.endpoints
        )
        generated = remaining.pop(index)
        if generated.join_type != planned.join_type:
            differences.append(
                _difference(
                    "join_type_mismatch",
                    "joins",
                    planned.join_type,
                    generated.join_type,
                )
            )
        elif planned.join_type == "left" and (
            generated.left_column_id != planned.left_column_id
        ):
            differences.append(
                _difference(
                    "join_direction_mismatch",
                    "joins",
                    planned.left_column_id,
                    generated.left_column_id,
                )
            )


def _report_join_endpoint_difference(expected, actual, differences):
    expected_count = sum(expected.values())
    actual_count = sum(actual.values())
    if actual_count == 0 and expected_count:
        differences.append(_difference("join_missing", "joins", list(expected), []))
    elif expected_count == actual_count:
        differences.append(
            _difference("join_endpoint_mismatch", "joins", list(expected), list(actual))
        )
    else:
        if expected - actual:
            differences.append(
                _difference("join_missing", "joins", list(expected), list(actual))
            )
        if actual - expected:
            differences.append(
                _difference("join_extra", "joins", list(expected), list(actual))
            )


def _actual_join_type(join: exp.Join) -> str | None:
    side = str(join.side or "").casefold()
    kind = str(join.kind or "").casefold()
    if side == "left":
        return "left"
    if not side and kind in {"", "inner"}:
        return "inner"
    return None


def _joined_table_id(join: exp.Join, context: _SqlContext) -> str:
    table_name = str(getattr(join.this, "name", "")).casefold()
    return context.aliases.get(table_name, table_name)


def _from_table_id(expression: exp.Select, context: _SqlContext) -> str:
    from_clause = expression.args.get("from_")
    table_name = str(getattr(getattr(from_clause, "this", None), "name", "")).casefold()
    return context.aliases.get(table_name, table_name)


def _compare_predicates(expression, plan, context, differences):
    expected = []
    kinds = []
    for predicate in plan.predicates:
        expected.append(_expected_predicate(predicate, plan))
        kinds.append(predicate.kind)
    expected_kinds = {
        (atom[0], atom[1]): kind for atom, kind in zip(expected, kinds, strict=True)
    }
    actual = []
    for clause in ("where", "having"):
        node = expression.args.get(clause)
        if node is not None:
            actual.extend(
                _sql_predicates(
                    node.this,
                    clause,
                    context,
                    expected_kinds,
                )
            )
    for join in expression.args.get("joins") or []:
        condition = join.args.get("on")
        if condition is not None:
            actual.extend(
                _sql_join_predicates(
                    condition,
                    f"on:{_joined_table_id(join, context)}",
                    context,
                    expected_kinds,
                )
            )
    actual = _deduplicate_atoms(_coalesce_enum_exclusions(actual))
    between_targets = {(atom[0], atom[1]) for atom in expected if atom[2] == "between"}
    actual = _coalesce_closed_ranges(actual, between_targets)

    remaining = list(actual)
    for expected_atom, kind in zip(expected, kinds, strict=True):
        if expected_atom in remaining:
            remaining.remove(expected_atom)
            continue
        same_target = [item for item in remaining if item[1] == expected_atom[1]]
        if kind == "numeric" and any(
            item[0] != expected_atom[0] for item in same_target
        ):
            code = "numeric_clause_mismatch"
        elif same_target:
            code = f"{kind}_predicate_mismatch"
        else:
            code = f"{kind}_predicate_missing"
        differences.append(
            _difference(code, f"predicates.{kind}", expected_atom, same_target or None)
        )
        if same_target:
            remaining.remove(same_target[0])
    for atom in remaining:
        differences.append(_difference("predicate_extra", "predicates", None, atom))


def _coalesce_closed_ranges(atoms, between_targets):
    """Normalize one inclusive lower/upper pair to its BETWEEN equivalent."""

    bounds: dict[tuple[str, str], dict[str, list[int]]] = {}
    for index, atom in enumerate(atoms):
        clause, target, operator, _values = atom
        if (clause, target) in between_targets and operator in {"gte", "lte"}:
            bounds.setdefault((clause, target), {"gte": [], "lte": []})[
                operator
            ].append(index)

    consumed: set[int] = set()
    replacements: dict[int, tuple] = {}
    for (clause, target), operators in bounds.items():
        lower_indexes = operators["gte"]
        upper_indexes = operators["lte"]
        if len(lower_indexes) != 1 or len(upper_indexes) != 1:
            continue
        lower_index = lower_indexes[0]
        upper_index = upper_indexes[0]
        replacements[min(lower_index, upper_index)] = (
            clause,
            target,
            "between",
            (atoms[lower_index][3][0], atoms[upper_index][3][0]),
        )
        consumed.update({lower_index, upper_index})

    normalized = []
    for index, atom in enumerate(atoms):
        replacement = replacements.get(index)
        if replacement is not None:
            normalized.append(replacement)
        if index not in consumed:
            normalized.append(atom)
    return normalized


def _compare_order(expression, plan, context, targets, differences):
    expected = [
        (item.target_type, item.target_id, item.direction) for item in plan.order_by
    ]
    order = expression.args.get("order")
    actual = []
    for item in order.expressions if order else []:
        inner = item.this
        direction = "desc" if item.args.get("desc") else "asc"
        target = None
        if isinstance(inner, exp.Column) and not inner.table:
            target = targets.get(inner.name.casefold())
        if target is None:
            signature = _expression_signature(inner, context)
            for alias, expected_signature in _expected_select(plan)[0].items():
                if signature == expected_signature:
                    target = targets[alias]
                    break
        actual.append(
            (*target, direction) if target else ("unknown", signature, direction)
        )
    if expected == actual:
        return
    if len(expected) == len(actual) and [item[:2] for item in expected] == [
        item[:2] for item in actual
    ]:
        differences.append(
            _difference("order_direction_mismatch", "order_by", expected, actual)
        )
    elif not actual and expected:
        differences.append(_difference("order_missing", "order_by", expected, actual))
    elif actual and not expected:
        differences.append(_difference("order_extra", "order_by", expected, actual))
    else:
        differences.append(
            _difference("order_target_mismatch", "order_by", expected, actual)
        )


def _compare_limit(expression, plan, differences):
    limit = expression.args.get("limit")
    actual = None
    if limit is not None and isinstance(limit.expression, exp.Literal):
        try:
            actual = int(limit.expression.this)
        except TypeError, ValueError:
            actual = str(limit.expression.this)
    if plan.limit != actual:
        differences.append(_difference("limit_mismatch", "limit", plan.limit, actual))


def _compare_offset(expression, differences):
    offset = expression.args.get("offset")
    if offset is not None:
        differences.append(
            _difference(
                "offset_extra",
                "offset",
                None,
                offset.sql(dialect="mysql"),
            )
        )


def _expected_predicate(predicate, plan):
    if isinstance(predicate, EnumPredicate):
        operator = (
            "set_include" if predicate.operator in {"eq", "in"} else "set_exclude"
        )
        return (
            _column_predicate_clause(predicate.column_id, plan),
            f"col:{predicate.column_id.casefold()}",
            operator,
            canonical_set_values(predicate.canonical_values),
        )
    if isinstance(predicate, NumericPredicate):
        target = (
            _measure_signature(
                _measure_by_id(plan, predicate.target_id),
                _plan_context(plan),
            )
            if predicate.target_type == "measure"
            else f"col:{predicate.target_id.casefold()}"
        )
        return (
            (
                _column_predicate_clause(predicate.target_id, plan)
                if predicate.target_type == "column" and predicate.clause == "where"
                else predicate.clause
            ),
            target,
            predicate.operator,
            tuple(_number(value) for value in predicate.values),
        )
    assert isinstance(predicate, TemporalPredicate)
    start = predicate.start_date_id or predicate.start_date
    end = predicate.end_date_id or predicate.end_date
    operator = predicate.operator
    if operator in {"during", "between"}:
        sql_operator, values = "between", (str(start), str(end))
    elif operator == "on":
        sql_operator, values = "eq", (str(start),)
    elif operator == "before":
        sql_operator, values = "lt", (str(start),)
    elif operator == "after":
        sql_operator, values = "gt", (str(end),)
    elif operator == "since":
        sql_operator, values = "gte", (str(start),)
    else:
        sql_operator, values = "lte", (str(end),)
    return (
        _column_predicate_clause(predicate.column_id, plan),
        f"col:{predicate.column_id.casefold()}",
        sql_operator,
        values,
    )


def _column_predicate_clause(column_id, plan):
    """Keep null-rejecting filters on a LEFT JOIN's nullable side inside ON."""

    table_id = column_id.casefold().rsplit(".", 1)[0]
    nullable_table_ids = {
        join.right_column_id.casefold().rsplit(".", 1)[0]
        for join in plan.joins
        if join.join_type == "left"
    }
    return f"on:{table_id}" if table_id in nullable_table_ids else "where"


def _sql_predicates(node, clause, context, expected_kinds):
    atoms = []
    for item in _flatten_and(node):
        folded = _fold_supported_or(item, clause, context, expected_kinds)
        if folded is not None:
            atoms.append(folded)
            continue
        atom = _sql_predicate(_unwrap_parens(item), clause, context)
        atoms.append(_canonicalize_sql_atom(atom, expected_kinds))
    return atoms


def _sql_join_predicates(node, clause, context, expected_kinds):
    atoms = []
    for item in _flatten_and(node):
        item = _unwrap_parens(item)
        if _is_join_endpoint(item, context):
            continue
        atoms.extend(_sql_predicates(item, clause, context, expected_kinds))
    return atoms


def _is_join_endpoint(node, context):
    if not isinstance(node, exp.EQ):
        return False
    left = _column_id(node.left, context)
    right = _column_id(node.right, context)
    if not left or not right:
        return False
    return left.rsplit(".", 1)[0] != right.rsplit(".", 1)[0]


def _fold_supported_or(node, clause, context, expected_kinds):
    inner = _unwrap_parens(node)
    if not isinstance(inner, exp.Or):
        return None
    atoms = [
        _canonicalize_sql_atom(
            _sql_predicate(_unwrap_parens(item), clause, context),
            expected_kinds,
        )
        for item in _flatten_or(inner)
    ]
    first = atoms[0]
    if first[2] != "set_include" or any(atom[:3] != first[:3] for atom in atoms[1:]):
        return None
    return (
        *first[:3],
        canonical_set_values(value for atom in atoms for value in atom[3]),
    )


def _canonicalize_sql_atom(atom, expected_kinds):
    clause, target, operator, values = atom
    if expected_kinds.get((clause, target)) != "enum":
        return atom
    if operator in {"eq", "in"}:
        operator = "set_include"
    elif operator in {"neq", "not_in"}:
        operator = "set_exclude"
    return (clause, target, operator, canonical_set_values(values))


def _coalesce_enum_exclusions(atoms):
    grouped: dict[tuple[str, str, str], list[str]] = {}
    first_indexes: dict[tuple[str, str, str], int] = {}
    passthrough: list[tuple[int, tuple]] = []
    for index, atom in enumerate(atoms):
        if atom[2] != "set_exclude":
            passthrough.append((index, atom))
            continue
        key = atom[:3]
        first_indexes.setdefault(key, index)
        grouped.setdefault(key, []).extend(atom[3])
    passthrough.extend(
        (
            first_indexes[key],
            (*key, canonical_set_values(values)),
        )
        for key, values in grouped.items()
    )
    return [atom for _, atom in sorted(passthrough, key=lambda item: item[0])]


def _deduplicate_atoms(atoms):
    values = []
    for atom in atoms:
        if atom not in values:
            values.append(atom)
    return values


def _sql_predicate(node, clause, context):
    if isinstance(node, exp.Between):
        return (
            clause,
            _expression_signature(node.this, context),
            "between",
            (_literal(node.args.get("low")), _literal(node.args.get("high"))),
        )
    if isinstance(node, exp.Not) and isinstance(node.this, exp.In):
        inner = node.this
        return (
            clause,
            _expression_signature(inner.this, context),
            "not_in",
            tuple(_literal(item) for item in inner.expressions),
        )
    if isinstance(node, exp.In):
        return (
            clause,
            _expression_signature(node.this, context),
            "in",
            tuple(_literal(item) for item in node.expressions),
        )
    operators = {
        exp.EQ: "eq",
        exp.NEQ: "neq",
        exp.GT: "gt",
        exp.GTE: "gte",
        exp.LT: "lt",
        exp.LTE: "lte",
    }
    for expression_type, operator in operators.items():
        if isinstance(node, expression_type):
            left, right = node.left, node.right
            if isinstance(left, exp.Literal) and not isinstance(right, exp.Literal):
                left, right = right, left
                operator = _reverse_operator(operator)
            return (
                clause,
                _expression_signature(left, context),
                operator,
                (_literal(right),),
            )
    return (clause, f"raw:{node.sql(dialect='mysql').casefold()}", "unknown", ())


def _measure_signature(measure, context) -> str:
    columns = [f"col:{column_id.casefold()}" for column_id in measure.source_column_ids]
    if measure.aggregation == "expression":
        parsed = parse_one(str(measure.expression or ""), read="mysql")
        return _expression_signature(parsed, context)
    if measure.aggregation == "count_distinct":
        return f"count_distinct({','.join(columns)})"
    return f"{measure.aggregation}({','.join(columns)})"


def _expression_signature(node, context):
    if isinstance(node, exp.Column):
        column_id = _column_id(node, context)
        return (
            f"col:{column_id}" if column_id else f"col:unknown.{node.name.casefold()}"
        )
    aggregate_names = {
        exp.Sum: "sum",
        exp.Avg: "avg",
        exp.Min: "min",
        exp.Max: "max",
    }
    for expression_type, name in aggregate_names.items():
        if isinstance(node, expression_type):
            return f"{name}({_expression_signature(node.this, context)})"
    if isinstance(node, exp.Count):
        inner = node.this
        if isinstance(inner, exp.Distinct):
            values = ",".join(
                _expression_signature(item, context) for item in inner.expressions
            )
            return f"count_distinct({values})"
        return f"count({_expression_signature(inner, context)})"
    return f"expression:{_normalized_expression_sql(node, context)}"


def _normalized_expression_sql(node, context):
    def normalize_column(item):
        column_id = _column_id(item, context)
        if not column_id:
            return item
        table, column = column_id.rsplit(".", 1)
        return exp.column(column, table=table)

    normalized = node.copy().transform(
        lambda item: normalize_column(item) if isinstance(item, exp.Column) else item
    )
    return " ".join(normalized.sql(dialect="mysql").casefold().split())


def _sql_context(expression, plan):
    aliases = {}
    for table in expression.find_all(exp.Table):
        table_name = table.name.casefold()
        aliases[table_name] = table_name
        if table.alias:
            aliases[table.alias.casefold()] = table_name
    return _SqlContext(
        aliases=aliases,
        column_ids_by_name=_columns_by_name(plan.required_column_ids),
    )


def _plan_context(plan):
    table_ids = {
        column_id.rsplit(".", 1)[0].casefold() for column_id in plan.required_column_ids
    }
    return _SqlContext(
        aliases={table_id: table_id for table_id in table_ids},
        column_ids_by_name=_columns_by_name(plan.required_column_ids),
    )


def _columns_by_name(column_ids):
    by_name: dict[str, list[str]] = {}
    for column_id in column_ids:
        by_name.setdefault(column_id.rsplit(".", 1)[-1].casefold(), []).append(
            column_id.casefold()
        )
    return {key: tuple(values) for key, values in by_name.items()}


def _column_id(node, context):
    if not isinstance(node, exp.Column):
        return None
    name = node.name.casefold()
    if node.table:
        table = context.aliases.get(node.table.casefold(), node.table.casefold())
        return f"{table}.{name}"
    candidates = context.column_ids_by_name.get(name, ())
    return candidates[0] if len(candidates) == 1 else None


def _flatten_and(node):
    if isinstance(node, exp.And):
        return [*_flatten_and(node.left), *_flatten_and(node.right)]
    return [node]


def _flatten_or(node):
    inner = _unwrap_parens(node)
    if isinstance(inner, exp.Or):
        return [*_flatten_or(inner.left), *_flatten_or(inner.right)]
    return [inner]


def _unwrap_parens(node):
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _literal(node) -> str:
    if not isinstance(node, exp.Literal):
        return f"expression:{node.sql(dialect='mysql').casefold()}"
    return str(node.this) if node.is_string else _number(str(node.this))


def _number(value: str) -> str:
    try:
        return canonical_number(value)
    except InvalidOperation, ValueError:
        return str(value)


def _reverse_operator(operator):
    return {"gt": "lt", "gte": "lte", "lt": "gt", "lte": "gte"}.get(operator, operator)


def _measure_by_id(plan, metric_id):
    return next(item for item in plan.measures if item.metric_id == metric_id)


def _difference(code, path, expected, actual):
    return SqlPlanDifference(code=code, path=path, expected=expected, actual=actual)


def _result(differences):
    values = tuple(differences)
    return SqlPlanConsistencyResult(ok=not values, differences=values)


__all__ = [
    "SqlPlanConsistencyResult",
    "SqlPlanDifference",
    "validate_sql_plan_consistency",
]
