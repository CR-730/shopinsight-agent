"""Combined SQL syntax validation and deterministic pre-execution guard."""

import re
from typing import Any

from langgraph.runtime import Runtime
from sqlglot import expressions as exp
from sqlglot import parse
from sqlglot.errors import ParseError

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.conf.policy_config import load_policy_config
from app.core.log import logger
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository

AGGREGATE_PATTERN = re.compile(r"\b(sum|count|avg|min|max)\s*\(", re.IGNORECASE)
ALIAS_PATTERN = re.compile(r"\bas\s+`?([\w\u4e00-\u9fff]+)`?", re.IGNORECASE)
STRING_LITERAL_PATTERN = re.compile(r"'([^']+)'")


async def pre_sql_execution_validation(
    state: DataAgentState, runtime: Runtime[DataAgentContext]
):
    """Return repairable SQL errors, hard safety blocks, or pass before run_sql."""

    writer = runtime.stream_writer
    step = "SQL执行前综合校验"
    writer({"type": "progress", "step": step, "status": "running"})

    sql = normalize_sql_for_execution(state["sql"])
    dw_mysql_repository: DWMySQLRepository = runtime.context["dw_mysql_repository"]

    parse_error = _parse_single_select(sql)
    if isinstance(parse_error, str):
        logger.info(f"{step} repairable SQL parse error: {parse_error}")
        writer({"type": "progress", "step": step, "status": "repairable_error"})
        return {"sql": sql, "error": parse_error, "safety_error": None}

    try:
        await dw_mysql_repository.validate(sql)
    except Exception as exc:
        error = str(exc)
        logger.info(f"{step} repairable SQL error: {error}")
        writer({"type": "progress", "step": step, "status": "repairable_error"})
        return {"sql": sql, "error": error, "safety_error": None}

    safety_error = validate_sql_before_execution(state, sql)
    if safety_error:
        logger.warning(f"{step} blocked SQL: {safety_error}")
        writer(
            {
                "type": "progress",
                "step": step,
                "status": "blocked",
                "error": safety_error,
            }
        )
        return {
            "sql": sql,
            "error": None,
            "safety_error": safety_error,
            "blocked_by": "pre_sql_execution_validation",
        }

    writer({"type": "progress", "step": step, "status": "success"})
    logger.info("SQL 语法和执行前安全校验通过")
    return {"sql": sql, "error": None, "safety_error": None}


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
        return f"SQL 使用了未召回的枚举值：{unknown_value}"

    if _looks_like_detail_query(expression, lowered_query):
        return "禁止执行用户或订单明细查询"

    fabricated_metric = _fabricated_metric_alias(state, expression)
    if fabricated_metric:
        return f"SQL 编造了未注册指标别名：{fabricated_metric}"

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


def _has_select_star(expression: exp.Expression) -> bool:
    return any(not _is_aggregate_star(star) for star in expression.find_all(exp.Star))


def _is_aggregate_star(star: exp.Star) -> bool:
    parent = star.parent
    return isinstance(parent, exp.Count)


def _sensitive_column(expression: exp.Expression) -> str | None:
    sql_policy = load_policy_config().get("sql", {})
    sensitive_column_ids = set(sql_policy.get("sensitive_columns", []))
    sensitive_names = set(sql_policy.get("sensitive_column_names", []))
    for projection in expression.expressions:
        for column in projection.find_all(exp.Column):
            sensitive = _sensitive_column_name(
                column, sensitive_column_ids, sensitive_names
            )
            if sensitive:
                return sensitive
    return None


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
    known_literals = {
        str(getattr(value_info, "value", ""))
        for value_info in state.get("retrieved_value_infos") or []
        if getattr(value_info, "value", None)
    }
    for literal_expression in expression.find_all(exp.Literal):
        if not literal_expression.is_string:
            continue
        literal = str(literal_expression.this)
        if _looks_like_business_literal(literal) and literal not in known_literals:
            return literal
    return None


def _looks_like_business_literal(literal: str) -> bool:
    if re.fullmatch(r"\d+(\.\d+)?", literal):
        return False
    if re.fullmatch(r"q[1-4]", literal.lower()):
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


def _fabricated_metric_alias(
    state: dict[str, Any],
    expression: exp.Expression,
) -> str | None:
    registered_metric_names = {
        metric.get("name") if isinstance(metric, dict) else getattr(metric, "name", None)
        for metric in state.get("metric_infos") or []
    }
    registered_metric_names.discard(None)
    allowed_aliases = set(registered_metric_names)
    allowed_aliases.update(
        load_policy_config().get("sql", {}).get("allowed_metric_aliases", [])
    )

    for alias_expression in expression.find_all(exp.Alias):
        alias = alias_expression.alias
        if "指数" in alias and alias not in allowed_aliases:
            return alias
    return None
