"""Resolve natural-language business objects once before SQL generation."""

from __future__ import annotations

import calendar
import re
from typing import TYPE_CHECKING, Any

from app.agent.state import (
    BindingIssueState,
    DataAgentState,
    MetricBindingState,
    ResolvedFilterState,
    TimeBindingState,
)
from app.core.log import logger

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

    from app.agent.context import DataAgentContext


async def business_binding(
    state: DataAgentState, runtime: Runtime[DataAgentContext]
) -> dict[str, Any]:
    """Bind user language to canonical metrics, filters, and time constraints."""

    writer = runtime.stream_writer
    step = "业务绑定"
    writer({"type": "progress", "step": step, "status": "running"})

    query = state.get("query") or ""
    metric_infos = state.get("metric_infos") or []
    table_infos = state.get("table_infos") or []
    retrieved_value_infos = state.get("retrieved_value_infos") or []
    enum_aliases = _value_alias_map(
        await runtime.context["meta_mysql_repository"].list_value_aliases()
    )

    metric_bindings = resolve_metric_bindings(query, metric_infos)
    resolved_filters, filter_issues = resolve_value_filters(
        query=query,
        table_infos=table_infos,
        retrieved_value_infos=retrieved_value_infos,
        enum_aliases=enum_aliases,
    )
    resolved_filters.extend(
        await resolve_value_filters_from_db(
            query,
            resolved_filters,
            enum_aliases,
            runtime.context["dw_mysql_repository"],
        )
    )
    time_binding = resolve_time_binding(query)
    unresolved_bindings = resolve_unresolved_bindings(
        query=query,
        metric_bindings=metric_bindings,
        filter_issues=filter_issues,
        enum_aliases=enum_aliases,
        resolved_filters=resolved_filters,
    )
    validated_enum_values = [
        literal
        for resolved_filter in resolved_filters
        for literal in resolved_filter["allowed_sql_literals"]
    ]
    binding = {
        "metrics": metric_bindings,
        "filters": resolved_filters,
        "time": time_binding,
        "unresolved": unresolved_bindings,
        "ambiguous": [],
    }

    logger.info(f"业务绑定结果：{binding}")
    writer({"type": "progress", "step": step, "status": "success"})
    return {
        "business_binding": binding,
        "metric_bindings": metric_bindings,
        "resolved_filters": resolved_filters,
        "time_binding": time_binding,
        "validated_enum_values": validated_enum_values,
        "unresolved_bindings": unresolved_bindings,
        "ambiguous_bindings": [],
    }


def resolve_metric_bindings(
    query: str,
    metric_infos: list[dict[str, Any]],
) -> list[MetricBindingState]:
    bindings: list[MetricBindingState] = []
    bound_metrics: set[str] = set()

    for mention, metric_info, matched_by in _metric_match_candidates(query, metric_infos):
        canonical_metric = str(metric_info.get("name") or "")
        if not canonical_metric or canonical_metric in bound_metrics:
            continue
        bound_metrics.add(canonical_metric)
        bindings.append(
            {
                "raw_mention": mention,
                "canonical_metric": canonical_metric,
                "matched_by": matched_by,
                "evidence": _metric_evidence(metric_info, mention, matched_by),
                "relevant_columns": list(metric_info.get("relevant_columns") or []),
                "confidence": "high",
            }
        )
    return bindings


def resolve_value_filters(
    query: str,
    retrieved_value_infos: list[Any],
    enum_aliases: dict[str, dict[str, str]],
    table_infos: list[dict[str, Any]] | None = None,
) -> tuple[list[ResolvedFilterState], list[BindingIssueState]]:
    filters: list[ResolvedFilterState] = []
    issues: list[BindingIssueState] = []
    bound_values: set[tuple[str, str]] = set()
    values_by_column = _values_by_column(retrieved_value_infos)

    for column_id, raw_value, canonical_value in sorted(
        _iter_enum_aliases(enum_aliases), key=lambda item: len(item[1]), reverse=True
    ):
        if raw_value not in query:
            continue
        if canonical_value not in values_by_column.get(column_id, set()):
            continue
        key = (column_id, str(canonical_value))
        if key in bound_values:
            continue
        bound_values.add(key)
        filters.append(
            _resolved_filter(
                raw_value=str(raw_value),
                canonical_value=str(canonical_value),
                column=column_id,
                field_alias="",
                matched_by="enum_alias",
            )
        )

    for candidate in _field_alias_value_candidates(query, table_infos or []):
        column_id = candidate["column"]
        raw_value = candidate["raw_value"]
        if (column_id, raw_value) in bound_values:
            continue
        if raw_value in values_by_column.get(column_id, set()):
            bound_values.add((column_id, raw_value))
            filters.append(
                _resolved_filter(
                    raw_value=raw_value,
                    canonical_value=raw_value,
                    column=column_id,
                    field_alias=candidate["field_alias"],
                    matched_by="retrieved_value",
                )
            )
            continue
        if not any(issue.get("candidate_column") == column_id for issue in issues):
            issues.append(
                {
                    "type": "enum_value",
                    "raw_text": raw_value,
                    "candidate_column": column_id,
                    "reason": "value_not_found",
                }
            )

    return filters, issues


async def resolve_value_filters_from_db(
    query: str,
    resolved_filters: list[ResolvedFilterState],
    enum_aliases: dict[str, dict[str, str]],
    dw_mysql_repository: Any,
) -> list[ResolvedFilterState]:
    """Resolve aliased enum values when RAG missed but DW confirms the value exists."""

    existing = {(item["column"], item["canonical_value"]) for item in resolved_filters}
    filters: list[ResolvedFilterState] = []
    for column_id, raw_value, canonical_value in sorted(
        _iter_enum_aliases(enum_aliases), key=lambda item: len(item[1]), reverse=True
    ):
        if raw_value not in query or (column_id, canonical_value) in existing:
            continue
        table_name, _, column_name = column_id.partition(".")
        if not table_name or not column_name:
            continue
        if not await dw_mysql_repository.column_value_exists(
            table_name, column_name, canonical_value
        ):
            continue
        existing.add((column_id, canonical_value))
        filters.append(
            _resolved_filter(
                raw_value=str(raw_value),
                canonical_value=str(canonical_value),
                column=column_id,
                field_alias="",
                matched_by="enum_alias_db",
            )
        )
    return filters


def resolve_time_binding(query: str) -> TimeBindingState | None:
    quarter = _parse_quarter(query)
    if quarter:
        return quarter
    month = _parse_month(query)
    if month:
        return month
    day = _parse_day(query)
    if day:
        return day
    return None


def resolve_unresolved_bindings(
    query: str,
    metric_bindings: list[MetricBindingState],
    filter_issues: list[BindingIssueState],
    enum_aliases: dict[str, dict[str, str]],
    resolved_filters: list[ResolvedFilterState] | None = None,
) -> list[BindingIssueState]:
    issues: list[BindingIssueState] = []
    if _looks_like_metric_request(query) and not metric_bindings:
        issues.append(
            {
                "type": "metric",
                "raw_text": query,
                "candidate_column": "",
                "reason": "metric_not_bound",
            }
        )

    resolved_filters = resolved_filters or []
    resolved_aliases = {
        item["raw_value"] for item in resolved_filters if item.get("raw_value")
    }
    issues.extend(
        issue
        for issue in filter_issues
        if issue.get("raw_text") not in resolved_aliases
        and not any(
            resolved_raw.startswith(str(issue.get("raw_text") or ""))
            or str(issue.get("raw_text") or "").startswith(resolved_raw)
            for resolved_raw in resolved_aliases
        )
    )
    for raw_value, canonical in _flat_enum_aliases(enum_aliases).items():
        if raw_value not in query:
            continue
        if raw_value in resolved_aliases:
            continue
        if any(
            resolved_raw.startswith(raw_value) or raw_value.startswith(resolved_raw)
            for resolved_raw in resolved_aliases
        ):
            continue
        if all(issue["raw_text"] != raw_value for issue in issues):
            issues.append(
                {
                    "type": "enum_value",
                    "raw_text": raw_value,
                    "candidate_column": "",
                    "reason": f"enum_alias_not_bound:{canonical}",
                }
            )
    return issues


def _metric_match_candidates(query: str, metric_infos: list[dict[str, Any]]):
    candidates = []
    for metric_info in metric_infos:
        name = str(metric_info.get("name") or "")
        if name:
            candidates.append((name, metric_info, "metric_name"))
        for alias in metric_info.get("alias") or []:
            alias = str(alias)
            if alias:
                candidates.append((alias, metric_info, "metric_alias"))

    candidates.sort(key=lambda item: len(item[0]), reverse=True)
    for mention, metric_info, matched_by in candidates:
        if mention and mention in query:
            yield mention, metric_info, matched_by


def _metric_evidence(metric_info: dict[str, Any], mention: str, matched_by: str) -> str:
    metric_name = str(metric_info.get("name") or "")
    if matched_by == "metric_name":
        return f"{metric_name}.name equals {mention}"
    return f"{metric_name}.alias contains {mention}"


def _field_alias_value_candidates(
    query: str, table_infos: list[dict[str, Any]]
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for table_info in table_infos:
        table_name = str(table_info.get("name") or "")
        if not table_name:
            continue
        if table_name == "dim_date":
            continue
        for column_info in table_info.get("columns") or []:
            if str(column_info.get("role") or "") != "dimension":
                continue
            column_name = str(column_info.get("name") or "")
            column_id = f"{table_name}.{column_name}"
            for field_alias in _column_aliases(column_info):
                raw_value = _value_before_alias(query, field_alias)
                if raw_value:
                    candidates.append(
                        {
                            "column": column_id,
                            "field_alias": field_alias,
                            "raw_value": raw_value,
                        }
                    )
    candidates.sort(key=lambda item: len(item["field_alias"]), reverse=True)
    return candidates


def _column_aliases(column_info: dict[str, Any]) -> list[str]:
    aliases = [str(item) for item in column_info.get("alias") or [] if str(item)]
    name = str(column_info.get("name") or "")
    if name:
        aliases.append(name)
    return sorted(set(aliases), key=len, reverse=True)


def _value_before_alias(query: str, field_alias: str) -> str:
    index = query.find(field_alias)
    if index <= 0:
        return ""
    prefix = query[:index]
    match = re.search(r"([\u4e00-\u9fffA-Za-z0-9_-]+)$", prefix)
    if not match:
        return ""
    raw_value = match.group(1)
    raw_value = re.sub(r"^(统计|查询|查看|分析|计算)", "", raw_value)
    raw_value = re.sub(r"(的|各|每个|所有)$", "", raw_value)
    if not raw_value or raw_value in {"各", "每", "所有"}:
        return ""
    if raw_value.startswith(("按", "各", "每", "每个")):
        return ""
    if "各" in raw_value or "每个" in raw_value:
        return ""
    if any(token in raw_value for token in ("年", "季度", "月")):
        return ""
    return raw_value


def _values_by_column(retrieved_value_infos: list[Any]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for value_info in retrieved_value_infos:
        column_id = str(getattr(value_info, "column_id", "") or "")
        value = str(getattr(value_info, "value", "") or "")
        if column_id and value:
            values.setdefault(column_id, set()).add(value)
    return values


def _resolved_filter(
    raw_value: str,
    canonical_value: str,
    column: str,
    field_alias: str,
    matched_by: str,
) -> ResolvedFilterState:
    return {
        "raw_value": raw_value,
        "canonical_value": canonical_value,
        "column": column,
        "field_alias": field_alias,
        "matched_by": matched_by,
        "allowed_sql_literals": [canonical_value],
    }


def _parse_quarter(query: str) -> TimeBindingState | None:
    pattern = re.compile(r"(?P<year>\d{4})\s*年?\s*(?:第?\s*(?P<cn>[一二三四])|Q(?P<num>[1-4]))\s*季度?", re.I)
    match = pattern.search(query)
    if not match:
        return None
    year = int(match.group("year"))
    quarter_num = int(match.group("num") or _cn_quarter_to_number(match.group("cn")))
    start_month = (quarter_num - 1) * 3 + 1
    end_month = start_month + 2
    _, end_day = calendar.monthrange(year, end_month)
    return {
        "raw_text": match.group(0).strip(),
        "grain": "quarter",
        "year": year,
        "quarter": f"Q{quarter_num}",
        "start_date": f"{year:04d}-{start_month:02d}-01",
        "end_date": f"{year:04d}-{end_month:02d}-{end_day:02d}",
        "start_date_id": int(f"{year:04d}{start_month:02d}01"),
        "end_date_id": int(f"{year:04d}{end_month:02d}{end_day:02d}"),
        "strategy": "date_range",
        "required_columns": ["fact_order.date_id"],
    }


def _parse_month(query: str) -> TimeBindingState | None:
    match = re.search(r"(?P<year>\d{4})\s*年\s*(?P<month>\d{1,2})\s*月", query)
    if not match:
        match = re.search(r"(?P<year>\d{4})-(?P<month>\d{1,2})(?!-\d)", query)
    if not match:
        return None
    year = int(match.group("year"))
    month = int(match.group("month"))
    if month < 1 or month > 12:
        return None
    _, end_day = calendar.monthrange(year, month)
    return {
        "raw_text": match.group(0).strip(),
        "grain": "month",
        "year": year,
        "month": month,
        "start_date": f"{year:04d}-{month:02d}-01",
        "end_date": f"{year:04d}-{month:02d}-{end_day:02d}",
        "start_date_id": int(f"{year:04d}{month:02d}01"),
        "end_date_id": int(f"{year:04d}{month:02d}{end_day:02d}"),
        "strategy": "date_range",
        "required_columns": ["fact_order.date_id"],
    }


def _parse_day(query: str) -> TimeBindingState | None:
    match = re.search(r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})", query)
    if not match:
        return None
    year = int(match.group("year"))
    month = int(match.group("month"))
    day = int(match.group("day"))
    if month < 1 or month > 12:
        return None
    _, end_day = calendar.monthrange(year, month)
    if day < 1 or day > end_day:
        return None
    return {
        "raw_text": match.group(0),
        "grain": "day",
        "year": year,
        "start_date": f"{year:04d}-{month:02d}-{day:02d}",
        "end_date": f"{year:04d}-{month:02d}-{day:02d}",
        "start_date_id": int(f"{year:04d}{month:02d}{day:02d}"),
        "end_date_id": int(f"{year:04d}{month:02d}{day:02d}"),
        "strategy": "date_range",
        "required_columns": ["fact_order.date_id"],
    }


def _cn_quarter_to_number(value: str | None) -> int:
    return {"一": 1, "二": 2, "三": 3, "四": 4}[value or "一"]


def _iter_enum_aliases(enum_aliases: dict[str, dict[str, str]]):
    for column_id, aliases in enum_aliases.items():
        for raw_value, canonical_value in aliases.items():
            yield column_id, raw_value, canonical_value


def _looks_like_metric_request(query: str) -> bool:
    metric_indicators = (
        "指标",
        "指数",
        "评分",
        "得分",
        "销售额",
        "成交额",
        "销售金额",
        "订单金额",
        "客单价",
        "忠诚度",
        "达成率",
        "转化率",
        "复购率",
        "心智",
        "GMV",
        "AOV",
    )
    return any(indicator in query for indicator in metric_indicators)


def _value_alias_map(value_aliases: list[Any]) -> dict[str, dict[str, str]]:
    aliases: dict[str, dict[str, str]] = {}
    for value_alias in value_aliases:
        column_id = str(getattr(value_alias, "column_id", "") or "")
        alias = str(getattr(value_alias, "alias", "") or "")
        canonical_value = str(getattr(value_alias, "canonical_value", "") or "")
        if column_id and alias and canonical_value:
            aliases.setdefault(column_id, {})[alias] = canonical_value
    return aliases


def _flat_enum_aliases(enum_aliases: dict[str, dict[str, str]]) -> dict[str, str]:
    return {
        raw: canonical
        for aliases in enum_aliases.values()
        for raw, canonical in aliases.items()
    }
