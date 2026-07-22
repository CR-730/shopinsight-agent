"""Shared SQL normalization, validation, and deterministic repair helpers."""

import re
from typing import Any

from sqlglot import expressions as exp
from sqlglot import parse
from sqlglot.errors import ParseError

from app.agent.semantic_planning.plan import EnumPredicate, SemanticQueryPlan
from app.conf.policy_config import load_policy_config


def normalize_sql_for_execution(sql: str) -> str:
    normalized = sql.strip()
    fenced_sql = re.search(
        r"```(?:sql)?\s*(.*?)```", normalized, flags=re.IGNORECASE | re.DOTALL
    )
    if fenced_sql:
        normalized = fenced_sql.group(1).strip()
    else:
        select_match = re.search(r"\bselect\b", normalized, flags=re.IGNORECASE)
        if select_match:
            normalized = normalized[select_match.start() :].strip()

    replacements = {
        "，": ",",
        "；": ";",
        "（": "(",
        "）": ")",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)

    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized[:-1].strip() if normalized.endswith(";") else normalized


def validate_sql_before_execution(state: dict[str, Any], sql: str) -> str | None:
    normalized_sql = sql.strip()
    lowered_sql = normalized_sql.lower()
    lowered_query = (state.get("query") or "").lower()
    parsed_sql = _parse_single_select(normalized_sql)
    if isinstance(parsed_sql, str):
        return parsed_sql

    expression = parsed_sql

    deny_keywords = load_policy_config().get("sql", {}).get("deny_keywords", [])
    for keyword in deny_keywords:
        pattern = rf"\b{re.escape(keyword)}\b"
        if re.search(pattern, lowered_sql):
            return f"SQL 包含危险关键字：{keyword}"

    if _has_select_star(expression):
        return "禁止 SELECT *"

    sensitive_column = _sensitive_column(expression)
    if sensitive_column:
        return f"SQL 访问敏感字段或明细标识：{sensitive_column}"

    unknown_value = _unknown_literal_value(state, expression)
    if unknown_value:
        return f"SQL 使用了未授权的枚举值：{unknown_value}"

    if _looks_like_detail_query(expression, lowered_query):
        return "禁止执行用户或订单明细查询"

    return None


def validate_sql_structure_semantics(
    state: dict[str, Any],
    sql: str,
) -> str | None:
    parsed_sql = _parse_single_select(sql.strip())
    if isinstance(parsed_sql, str):
        return parsed_sql
    return _invalid_join_relationship(state, parsed_sql)


def repair_invalid_join_relationship(
    state: dict[str, Any],
    sql: str,
) -> str | None:
    """Repair a join predicate when the trusted plan has one matching edge."""

    parsed_sql = _parse_single_select(sql.strip())
    if isinstance(parsed_sql, str):
        return None

    alias_to_table = _table_aliases(parsed_sql)
    table_refs = _preferred_table_refs(alias_to_table)
    planned_joins = _planned_joins(state)
    if planned_joins is not None:
        for join in parsed_sql.find_all(exp.Join):
            condition = join.args.get("on")
            if condition is None:
                continue
            for predicate in condition.find_all(exp.EQ):
                if not isinstance(predicate.left, exp.Column) or not isinstance(
                    predicate.right, exp.Column
                ):
                    continue
                left_id = _resolved_column_id(predicate.left, alias_to_table)
                right_id = _resolved_column_id(predicate.right, alias_to_table)
                if left_id is None or right_id is None:
                    continue
                if _is_planned_join(left_id, right_id, planned_joins):
                    continue
                candidate = _planned_join_for_tables(left_id, right_id, planned_joins)
                if candidate is None:
                    continue
                left_expected, right_expected = candidate
                left_table, left_name = left_expected.rsplit(".", 1)
                right_table, right_name = right_expected.rsplit(".", 1)
                predicate.set(
                    "this",
                    exp.column(
                        left_name,
                        table=table_refs.get(left_table, left_table),
                    ),
                )
                predicate.set(
                    "expression",
                    exp.column(
                        right_name,
                        table=table_refs.get(right_table, right_table),
                    ),
                )
                return parsed_sql.sql(dialect="mysql")
        return None

    column_catalog = _column_catalog(state)
    if not column_catalog:
        return None

    for join in parsed_sql.find_all(exp.Join):
        condition = join.args.get("on")
        if condition is None:
            continue
        for predicate in condition.find_all(exp.EQ):
            if not isinstance(predicate.left, exp.Column) or not isinstance(
                predicate.right, exp.Column
            ):
                continue
            left = _resolved_column(predicate.left, alias_to_table, column_catalog)
            right = _resolved_column(predicate.right, alias_to_table, column_catalog)
            if left is None or right is None or left["table"] == right["table"]:
                continue
            if _is_valid_join_pair(left, right):
                continue

            candidate = _candidate_join_pair(left, right, column_catalog)
            if candidate is None:
                continue
            foreign_key, primary_key = candidate
            predicate.set(
                "this",
                exp.column(
                    foreign_key["name"],
                    table=table_refs.get(foreign_key["table"], foreign_key["table"]),
                ),
            )
            predicate.set(
                "expression",
                exp.column(
                    primary_key["name"],
                    table=table_refs.get(primary_key["table"], primary_key["table"]),
                ),
            )
            return parsed_sql.sql(dialect="mysql")
    return None


def _parse_single_select(sql: str) -> exp.Expression | str:
    try:
        expressions = parse(sql, read="mysql")
    except ParseError as exc:
        return f"SQL 解析失败：{exc}"

    if len(expressions) != 1:
        return "仅允许执行单条 SELECT 查询"

    expression = expressions[0]
    if not isinstance(expression, exp.Select):
        return "仅允许执行 SELECT 查询"
    return expression


def _invalid_join_relationship(
    state: dict[str, Any],
    expression: exp.Expression,
) -> str | None:
    alias_to_table = _table_aliases(expression)
    planned_joins = _planned_joins(state)
    if planned_joins is not None:
        for join in expression.find_all(exp.Join):
            condition = join.args.get("on")
            if condition is None:
                continue
            for predicate in condition.find_all(exp.EQ):
                if not isinstance(predicate.left, exp.Column) or not isinstance(
                    predicate.right, exp.Column
                ):
                    continue
                left_id = _resolved_column_id(predicate.left, alias_to_table)
                right_id = _resolved_column_id(predicate.right, alias_to_table)
                if left_id is None or right_id is None or left_id == right_id:
                    continue
                if _is_planned_join(left_id, right_id, planned_joins):
                    continue
                candidate = _planned_join_for_tables(left_id, right_id, planned_joins)
                message = f"JOIN 条件不符合查询计划：{left_id} = {right_id}。"
                if candidate:
                    message += f"计划关系：{candidate[0]} = {candidate[1]}。"
                return message
        return None

    column_catalog = _column_catalog(state)
    if not column_catalog:
        return None

    for join in expression.find_all(exp.Join):
        condition = join.args.get("on")
        if condition is None:
            continue
        for predicate in condition.find_all(exp.EQ):
            if not isinstance(predicate.left, exp.Column) or not isinstance(
                predicate.right, exp.Column
            ):
                continue
            left = _resolved_column(predicate.left, alias_to_table, column_catalog)
            right = _resolved_column(predicate.right, alias_to_table, column_catalog)
            if left is None or right is None or left["table"] == right["table"]:
                continue
            if _is_valid_join_pair(left, right):
                continue

            candidate = _candidate_join_relationship(left, right, column_catalog)
            message = f"JOIN 条件不符合元数据关系：{left['id']} = {right['id']}。"
            if candidate:
                message += f"候选正确关系：{candidate}。"
            return message
    return None


def _planned_joins(
    state: dict[str, Any],
) -> list[tuple[str, str]] | None:
    raw_plan = state.get("semantic_plan")
    if not raw_plan:
        return None
    try:
        plan = SemanticQueryPlan.model_validate(raw_plan)
    except ValueError:
        return []
    return [
        (join.left_column_id.casefold(), join.right_column_id.casefold())
        for join in plan.joins
    ]


def _resolved_column_id(
    column: exp.Column,
    alias_to_table: dict[str, str],
) -> str | None:
    if not column.table:
        return None
    table = alias_to_table.get(column.table, column.table)
    return f"{table}.{column.name}".casefold()


def _is_planned_join(
    left_id: str,
    right_id: str,
    planned_joins: list[tuple[str, str]],
) -> bool:
    pair = frozenset((left_id.casefold(), right_id.casefold()))
    return any(frozenset(candidate) == pair for candidate in planned_joins)


def _planned_join_for_tables(
    left_id: str,
    right_id: str,
    planned_joins: list[tuple[str, str]],
) -> tuple[str, str] | None:
    tables = {
        left_id.rsplit(".", 1)[0].casefold(),
        right_id.rsplit(".", 1)[0].casefold(),
    }
    matches = [
        candidate
        for candidate in planned_joins
        if {
            candidate[0].rsplit(".", 1)[0],
            candidate[1].rsplit(".", 1)[0],
        }
        == tables
    ]
    return matches[0] if len(matches) == 1 else None


def _table_aliases(expression: exp.Expression) -> dict[str, str]:
    aliases = {}
    for table in expression.find_all(exp.Table):
        table_name = table.name
        aliases[table_name] = table_name
        if table.alias:
            aliases[table.alias] = table_name
    return aliases


def _preferred_table_refs(alias_to_table: dict[str, str]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for table_ref, table_name in alias_to_table.items():
        refs.setdefault(table_name, table_ref)
        if table_ref != table_name:
            refs[table_name] = table_ref
    return refs


def _column_catalog(state: dict[str, Any]) -> dict[str, dict[str, str]]:
    catalog: dict[str, dict[str, str]] = {}
    for table in (state.get("sql_context") or {}).get("tables") or []:
        table_name = _field_value(table, "name")
        if not table_name:
            continue
        table_role = _field_value(table, "role") or ""
        for column in _field_value(table, "columns") or []:
            column_name = _field_value(column, "name")
            if not column_name:
                continue
            _put_column_catalog(
                catalog,
                table_name=table_name,
                column_name=column_name,
                column_role=_field_value(column, "role") or "",
                table_role=table_role,
            )

    for column in (state.get("retrieval_context") or {}).get("columns") or []:
        table_name = _field_value(column, "table_id")
        column_name = _field_value(column, "name")
        if not table_name or not column_name:
            continue
        _put_column_catalog(
            catalog,
            table_name=table_name,
            column_name=column_name,
            column_role=_field_value(column, "role") or "",
            table_role="",
        )
    return catalog


def _put_column_catalog(
    catalog: dict[str, dict[str, str]],
    *,
    table_name: str,
    column_name: str,
    column_role: str,
    table_role: str,
):
    column_id = f"{table_name}.{column_name}".lower()
    existing = catalog.get(column_id, {})
    catalog[column_id] = {
        "id": f"{table_name}.{column_name}",
        "table": table_name,
        "name": column_name,
        "role": column_role or existing.get("role", ""),
        "table_role": table_role or existing.get("table_role", ""),
    }


def _field_value(item: Any, field: str) -> Any:
    if isinstance(item, dict):
        return item.get(field)
    return getattr(item, field, None)


def _resolved_column(
    column: exp.Column,
    alias_to_table: dict[str, str],
    catalog: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    table = column.table
    if not table:
        return None
    table_name = alias_to_table.get(table, table)
    return catalog.get(f"{table_name}.{column.name}".lower())


def _is_valid_join_pair(left: dict[str, str], right: dict[str, str]) -> bool:
    if left["name"].lower() != right["name"].lower():
        return False
    roles = {left["role"], right["role"]}
    return "foreign_key" in roles and "primary_key" in roles


def _candidate_join_relationship(
    left: dict[str, str],
    right: dict[str, str],
    catalog: dict[str, dict[str, str]],
) -> str | None:
    candidate = _candidate_join_pair(left, right, catalog)
    if candidate is None:
        return None
    foreign_key, primary_key = candidate
    return f"{foreign_key['id']} = {primary_key['id']}"


def _candidate_join_pair(
    left: dict[str, str],
    right: dict[str, str],
    catalog: dict[str, dict[str, str]],
) -> tuple[dict[str, str], dict[str, str]] | None:
    for foreign_key, other in ((left, right), (right, left)):
        if foreign_key["role"] != "foreign_key":
            continue
        primary_key = catalog.get(f"{other['table']}.{foreign_key['name']}".lower())
        if primary_key and primary_key["role"] == "primary_key":
            return foreign_key, primary_key
    return None


def _has_select_star(expression: exp.Expression) -> bool:
    return any(not _is_aggregate_star(star) for star in expression.find_all(exp.Star))


def _is_aggregate_star(star: exp.Star) -> bool:
    parent = star.parent
    return isinstance(parent, exp.Count)


def _sensitive_column(expression: exp.Expression) -> str | None:
    sql_policy = load_policy_config().get("sql", {})
    sensitive_column_ids = set(sql_policy.get("sensitive_columns", []))
    sensitive_names = set(sql_policy.get("sensitive_column_names", []))
    allowed_join_key_names = set(sql_policy.get("allowed_sensitive_join_key_names", []))
    for column in expression.find_all(exp.Column):
        if _is_allowed_sensitive_join_key(column, allowed_join_key_names):
            continue
        sensitive = _sensitive_column_name(
            column, sensitive_column_ids, sensitive_names
        )
        if sensitive:
            return sensitive
    return None


def _is_allowed_sensitive_join_key(
    column: exp.Column,
    allowed_join_key_names: set[str],
) -> bool:
    if column.name.lower() not in allowed_join_key_names:
        return False
    node = column.parent
    while node is not None:
        if isinstance(node, exp.Join):
            condition = node.args.get("on")
            return condition is not None and any(
                candidate is column for candidate in condition.find_all(exp.Column)
            )
        node = node.parent
    return False


def _sensitive_column_name(
    column: exp.Column,
    sensitive_column_ids: set[str],
    sensitive_names: set[str],
) -> str | None:
    column_name = column.name.lower()
    table_name = column.table
    column_id = f"{table_name}.{column.name}" if table_name else column.name
    if column_id in sensitive_column_ids or column_name in sensitive_names:
        return column.name
    return None


def _unknown_literal_value(
    state: dict[str, Any],
    expression: exp.Expression,
) -> str | None:
    known_literals = _known_literal_values(state)
    for literal_expression in expression.find_all(exp.Literal):
        if not literal_expression.is_string:
            continue
        literal = str(literal_expression.this)
        if _looks_like_temporal_literal(literal):
            continue
        if _looks_like_business_literal(literal) and literal not in known_literals:
            return literal
    return None


def _known_literal_values(state: dict[str, Any]) -> set[str]:
    raw_plan = state.get("semantic_plan")
    if not raw_plan:
        return set()
    try:
        plan = SemanticQueryPlan.model_validate(raw_plan)
    except ValueError:
        return set()
    return {
        literal
        for predicate in plan.predicates
        if isinstance(predicate, EnumPredicate)
        for literal in predicate.canonical_values
    }


def _looks_like_temporal_literal(literal: str) -> bool:
    return bool(
        re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", literal)
        or re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2}", literal)
        or re.fullmatch(r"\d{8}", literal)
        or re.fullmatch(r"\d{4}", literal)
        or re.fullmatch(r"q[1-4]", literal.lower())
        or re.fullmatch(r"(?:%[a-zA-Z][\-/:\s]*)+", literal)
    )


def _looks_like_business_literal(literal: str) -> bool:
    if re.fullmatch(r"\d+(\.\d+)?", literal):
        return False
    if _looks_like_temporal_literal(literal):
        return False
    return True


def _looks_like_detail_query(
    expression: exp.Expression,
    lowered_query: str,
) -> bool:
    asks_for_detail = any(
        word in lowered_query
        for word in ("所有", "明细", "列表", "每个用户", "全部", "手机号", "用户id")
    )
    return asks_for_detail and not _has_aggregate(expression)


def _has_aggregate(expression: exp.Expression) -> bool:
    aggregate_nodes = (exp.Sum, exp.Count, exp.Avg, exp.Min, exp.Max)
    return any(True for _ in expression.find_all(*aggregate_nodes))
