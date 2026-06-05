"""Deterministic metadata validation for binding candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent.business_binding.candidates import BindingCandidates
from app.agent.business_binding.time_resolver import resolve_time_mentions
from app.agent.state import (
    BindingIssueState,
    BusinessBindingState,
    GroupByBindingState,
    MetricBindingState,
    ResolvedFilterState,
)


@dataclass
class BindingValidationContext:
    metric_infos: list[dict[str, Any]]
    table_infos: list[dict[str, Any]]
    retrieved_value_infos: list[Any]
    enum_aliases: dict[str, dict[str, str]]
    dw_mysql_repository: Any


async def validate_binding_candidates(
    candidates: BindingCandidates,
    context: BindingValidationContext,
) -> BusinessBindingState:
    metrics, metric_issues = resolve_metric_candidates(
        candidates, context.metric_infos, context.table_infos
    )
    filters, filter_issues = await resolve_filter_candidates(candidates, context)
    groups, group_issues = resolve_groupby_candidates(candidates, context.table_infos)
    time_binding = resolve_time_mentions(
        [item.raw_text for item in candidates.time_mentions] + [candidates.source_query]
    )
    unresolved = [
        *metric_issues,
        *filter_issues,
        *group_issues,
        *_extraction_issues(candidates),
    ]
    return {
        "metrics": metrics,
        "filters": filters,
        "groups": groups,
        "time": time_binding,
        "unresolved": unresolved,
        "ambiguous": [],
    }


def resolve_metric_candidates(
    candidates: BindingCandidates,
    metric_infos: list[dict[str, Any]],
    table_infos: list[dict[str, Any]] | None = None,
) -> tuple[list[MetricBindingState], list[BindingIssueState]]:
    bindings: list[MetricBindingState] = []
    issues: list[BindingIssueState] = []
    bound_metrics: set[str] = set()
    catalog = _metric_catalog(metric_infos)
    measure_aliases = _measure_aliases(table_infos or [])
    for mention in candidates.metric_mentions:
        raw_text = _mention_text(mention.raw_text, mention.normalized_text)
        if not raw_text:
            continue
        metric_info, matched_by = catalog.get(raw_text, (None, ""))
        if not metric_info:
            if raw_text not in measure_aliases:
                issues.append(
                    {
                        "type": "metric",
                        "raw_text": raw_text,
                        "candidate_column": "",
                        "reason": "metric_not_bound",
                    }
                )
            continue
        canonical_metric = str(metric_info.get("name") or "")
        if not canonical_metric or canonical_metric in bound_metrics:
            continue
        bound_metrics.add(canonical_metric)
        bindings.append(
            {
                "raw_mention": raw_text,
                "canonical_metric": canonical_metric,
                "matched_by": matched_by,
                "evidence": _metric_evidence(metric_info, raw_text, matched_by),
                "relevant_columns": list(metric_info.get("relevant_columns") or []),
                "confidence": "high",
            }
        )
    return bindings, issues


async def resolve_filter_candidates(
    candidates: BindingCandidates,
    context: BindingValidationContext,
) -> tuple[list[ResolvedFilterState], list[BindingIssueState]]:
    filters: list[ResolvedFilterState] = []
    issues: list[BindingIssueState] = []
    bound_values: set[tuple[str, str]] = set()
    values_by_column = _values_by_column(context.retrieved_value_infos)
    columns_by_hint = _columns_by_hint(context.table_infos)

    for mention in candidates.filter_mentions:
        raw_value = str(mention.raw_text or "").strip()
        field_hint = str(mention.field_hint or "").strip()
        if not raw_value:
            continue
        resolved = None
        normalized_value = raw_value
        for candidate_value in _filter_value_variants(
            raw_value, field_hint, columns_by_hint
        ):
            resolved = await _resolve_single_filter(
                candidate_value, field_hint, values_by_column, columns_by_hint, context
            )
            if resolved:
                normalized_value = candidate_value
                break
        if resolved:
            key = (resolved["column"], resolved["canonical_value"])
            if key not in bound_values:
                bound_values.add(key)
                filters.append(resolved)
            continue
        issue = _filter_issue(normalized_value, field_hint, columns_by_hint)
        if issue:
            issues.append(issue)
    return filters, issues


def resolve_groupby_candidates(
    candidates: BindingCandidates,
    table_infos: list[dict[str, Any]],
) -> tuple[list[GroupByBindingState], list[BindingIssueState]]:
    groups: list[GroupByBindingState] = []
    issues: list[BindingIssueState] = []
    columns_by_hint = _columns_by_hint(table_infos)
    bound_columns: set[str] = set()

    for mention in candidates.groupby_mentions:
        raw_text = str(mention.raw_text or "").strip()
        field_hint = str(mention.field_hint or "").strip()
        lookup_values = [field_hint, raw_text]
        column = ""
        matched_alias = ""
        for lookup in lookup_values:
            normalized = _normalize_groupby_hint(lookup, columns_by_hint)
            if normalized and normalized in columns_by_hint:
                column = columns_by_hint[normalized][0]
                matched_alias = normalized
                break
        if column:
            if column not in bound_columns:
                bound_columns.add(column)
                groups.append(
                    {
                        "raw_mention": raw_text or field_hint,
                        "column": column,
                        "field_alias": matched_alias,
                        "matched_by": "column_alias",
                        "confidence": "high",
                    }
                )
            continue
        if raw_text or field_hint:
            issues.append(
                {
                    "type": "groupby",
                    "raw_text": raw_text or field_hint,
                    "candidate_column": "",
                    "reason": "groupby_not_bound",
                }
            )
    return groups, issues


async def _resolve_single_filter(
    raw_value: str,
    field_hint: str,
    values_by_column: dict[str, set[str]],
    columns_by_hint: dict[str, list[str]],
    context: BindingValidationContext,
) -> ResolvedFilterState | None:
    for column_id, aliases in context.enum_aliases.items():
        canonical_value = aliases.get(raw_value)
        if not canonical_value:
            continue
        if canonical_value in values_by_column.get(column_id, set()):
            return _resolved_filter(raw_value, canonical_value, column_id, "", "enum_alias")
        table_name, _, column_name = column_id.partition(".")
        if await context.dw_mysql_repository.column_value_exists(
            table_name, column_name, canonical_value
        ):
            return _resolved_filter(raw_value, canonical_value, column_id, "", "enum_alias_db")

    candidate_columns = _candidate_columns(field_hint, columns_by_hint, values_by_column)
    for column_id in candidate_columns:
        if raw_value in values_by_column.get(column_id, set()):
            return _resolved_filter(
                raw_value, raw_value, column_id, field_hint, "retrieved_value"
            )
        table_name, _, column_name = column_id.partition(".")
        if await context.dw_mysql_repository.column_value_exists(
            table_name, column_name, raw_value
        ):
            return _resolved_filter(raw_value, raw_value, column_id, field_hint, "dw_value")
    return None


def validate_business_binding_state(state: dict[str, Any]) -> str | None:
    unresolved = state.get("unresolved_bindings") or []
    if unresolved:
        issue = unresolved[0]
        return _binding_issue_message(issue, "unresolved")

    ambiguous = state.get("ambiguous_bindings") or []
    if ambiguous:
        issue = ambiguous[0]
        return _binding_issue_message(issue, "ambiguous")
    return None


def validated_enum_values(filters: list[ResolvedFilterState]) -> list[str]:
    return [literal for item in filters for literal in item["allowed_sql_literals"]]


def _extraction_issues(candidates: BindingCandidates) -> list[BindingIssueState]:
    # Candidate extraction failure is an observability signal, not a business blocker.
    # Retrieval context and SQL generation can still answer from metric/table evidence.
    return []


def _metric_catalog(metric_infos: list[dict[str, Any]]):
    catalog = {}
    for metric_info in metric_infos:
        name = str(metric_info.get("name") or "")
        if name:
            catalog[name] = (metric_info, "metric_name")
        for alias in metric_info.get("alias") or []:
            alias = str(alias)
            if alias:
                catalog[alias] = (metric_info, "metric_alias")
    return catalog


def _measure_aliases(table_infos: list[dict[str, Any]]) -> set[str]:
    aliases: set[str] = set()
    for table_info in table_infos:
        for column_info in table_info.get("columns") or []:
            if str(column_info.get("role") or "") != "measure":
                continue
            aliases.update(_column_aliases(column_info))
    return aliases


def _metric_evidence(metric_info: dict[str, Any], mention: str, matched_by: str) -> str:
    metric_name = str(metric_info.get("name") or "")
    if matched_by == "metric_name":
        return f"{metric_name}.name equals {mention}"
    return f"{metric_name}.alias contains {mention}"


def _mention_text(*values: str) -> str:
    return next((str(value).strip() for value in values if str(value).strip()), "")


def _values_by_column(retrieved_value_infos: list[Any]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for value_info in retrieved_value_infos:
        column_id = str(getattr(value_info, "column_id", "") or "")
        value = str(getattr(value_info, "value", "") or "")
        if column_id and value:
            values.setdefault(column_id, set()).add(value)
    return values


def _columns_by_hint(table_infos: list[dict[str, Any]]) -> dict[str, list[str]]:
    hints: dict[str, list[str]] = {}
    for table_info in table_infos:
        table_name = str(table_info.get("name") or "")
        if not table_name or table_name == "dim_date":
            continue
        for column_info in table_info.get("columns") or []:
            if str(column_info.get("role") or "") != "dimension":
                continue
            column_id = f"{table_name}.{column_info.get('name')}"
            for alias in _column_aliases(column_info):
                hints.setdefault(alias, []).append(column_id)
    return hints


def _column_aliases(column_info: dict[str, Any]) -> list[str]:
    aliases = [str(item) for item in column_info.get("alias") or [] if str(item)]
    name = str(column_info.get("name") or "")
    if name:
        aliases.append(name)
    return sorted(set(aliases), key=len, reverse=True)


def _candidate_columns(
    field_hint: str,
    columns_by_hint: dict[str, list[str]],
    values_by_column: dict[str, set[str]],
) -> list[str]:
    if field_hint and field_hint in columns_by_hint:
        return columns_by_hint[field_hint]
    return list(values_by_column)


def _filter_value_variants(
    raw_value: str,
    field_hint: str,
    columns_by_hint: dict[str, list[str]],
) -> list[str]:
    variants = [raw_value]
    suffixes = [field_hint] if field_hint else list(columns_by_hint)
    for suffix in sorted({item for item in suffixes if item}, key=len, reverse=True):
        if raw_value.endswith(suffix) and len(raw_value) > len(suffix):
            variants.append(raw_value[: -len(suffix)].rstrip("的 "))
    return list(dict.fromkeys(item for item in variants if item))


def _normalize_groupby_hint(
    value: str, columns_by_hint: dict[str, list[str]]
) -> str:
    normalized = str(value or "").strip()
    for prefix in ("各", "每个", "每一", "按照", "按"):
        if normalized.startswith(prefix) and len(normalized) > len(prefix):
            normalized = normalized[len(prefix) :]
            break
    for suffix in ("分组维度", "维度", "分组"):
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            normalized = normalized[: -len(suffix)]
            break
    if normalized in columns_by_hint:
        return normalized
    for alias in sorted(columns_by_hint, key=len, reverse=True):
        if alias and alias in normalized:
            return alias
    return normalized


def _filter_issue(
    raw_value: str,
    field_hint: str,
    columns_by_hint: dict[str, list[str]],
) -> BindingIssueState | None:
    if not field_hint:
        return {
            "type": "enum_value",
            "raw_text": raw_value,
            "candidate_column": "",
            "reason": "field_hint_missing",
        }
    columns = columns_by_hint.get(field_hint, [])
    if not columns:
        return {
            "type": "enum_value",
            "raw_text": raw_value,
            "candidate_column": "",
            "reason": "field_hint_not_found",
        }
    return {
        "type": "enum_value",
        "raw_text": raw_value,
        "candidate_column": columns[0],
        "reason": "value_not_found",
    }


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


def _binding_issue_message(issue: dict[str, Any], status: str) -> str:
    issue_type = issue.get("type") or "business_object"
    raw_text = issue.get("raw_text") or ""
    reason = issue.get("reason") or "unknown"
    return f"business_binding {status}: {issue_type}={raw_text}, reason={reason}"
