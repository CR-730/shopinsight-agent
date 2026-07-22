"""Build the request-scoped catalog used by controlled semantic planning."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from app.agent.schema_relations import (
    build_schema_graph,
    stable_relationship_id,
    unique_join_edges,
)
from app.entities.column_info import ColumnInfo
from app.entities.metric_info import MetricInfo
from app.entities.value_info import ValueInfo


@dataclass(frozen=True)
class MetricCandidate:
    candidate_id: str
    name: str
    aliases: tuple[str, ...]
    relevant_columns: tuple[str, ...]
    aggregation: str | None = None
    expression: str | None = None
    description: str = ""


@dataclass(frozen=True)
class ColumnCandidate:
    candidate_id: str
    table: str
    name: str
    aliases: tuple[str, ...]
    role: str
    projectable: bool
    data_type: str = ""
    description: str = ""


@dataclass(frozen=True)
class TableCandidate:
    candidate_id: str
    name: str
    role: str
    description: str


@dataclass(frozen=True)
class RelationshipCandidate:
    candidate_id: str
    left_table_id: str
    left_column_id: str
    right_table_id: str
    right_column_id: str


@dataclass(frozen=True)
class ValueCandidate:
    candidate_id: str
    canonical_value: str
    aliases: tuple[str, ...]
    column_id: str
    source: str


@dataclass(frozen=True)
class SemanticCandidateCatalog:
    """Versioned request-scoped catalog backed by authoritative metadata."""

    metadata_version: str
    tables: Mapping[str, TableCandidate]
    columns: Mapping[str, ColumnCandidate]
    relationships: Mapping[str, RelationshipCandidate]
    metrics: Mapping[str, MetricCandidate]
    values: Mapping[str, ValueCandidate]

    def metric_by_id(self, candidate_id: str) -> MetricCandidate:
        return self.metrics[candidate_id]

    def column_by_id(self, candidate_id: str) -> ColumnCandidate:
        return self.columns[candidate_id]

    def value_by_id(self, candidate_id: str) -> ValueCandidate:
        return self.values[candidate_id]

    def relationship_by_id(
        self, candidate_id: str
    ) -> RelationshipCandidate:
        return self.relationships[candidate_id]


def build_semantic_candidate_catalog(
    *,
    sql_context: Mapping[str, Any],
    retrieved_value_infos: list[ValueInfo],
    authoritative_columns: list[ColumnInfo],
    authoritative_metrics: list[MetricInfo],
    metadata_version: str,
    policy: Mapping[str, Any],
) -> SemanticCandidateCatalog:
    """Build the LLM-visible slice while keeping Meta objects authoritative."""

    if not metadata_version.strip():
        raise ValueError("metadata_version_required")
    tables = _build_table_candidates(sql_context)
    columns = _build_authoritative_column_candidates(
        sql_context,
        authoritative_columns=authoritative_columns,
        policy=policy,
    )
    metrics = _build_authoritative_metric_candidates(
        sql_context,
        authoritative_metrics=authoritative_metrics,
    )
    relationships = _build_relationship_candidates(
        authoritative_columns,
        exposed_table_ids=set(tables),
    )
    values = _build_value_candidates(columns, retrieved_value_infos)
    return SemanticCandidateCatalog(
        metadata_version=metadata_version,
        tables=MappingProxyType(tables),
        columns=MappingProxyType(columns),
        relationships=MappingProxyType(relationships),
        metrics=MappingProxyType(metrics),
        values=MappingProxyType(values),
    )


def _build_table_candidates(
    sql_context: Mapping[str, Any],
) -> dict[str, TableCandidate]:
    candidates: dict[str, TableCandidate] = {}
    for table_info in sql_context.get("tables", []) or []:
        table_id = str(table_info.get("id") or table_info.get("name") or "")
        if not table_id:
            continue
        candidates[table_id] = TableCandidate(
            candidate_id=table_id,
            name=str(table_info.get("name") or table_id),
            role=str(table_info.get("role") or ""),
            description=str(table_info.get("description") or ""),
        )
    return candidates


def _build_authoritative_column_candidates(
    sql_context: Mapping[str, Any],
    *,
    authoritative_columns: list[ColumnInfo],
    policy: Mapping[str, Any],
) -> dict[str, ColumnCandidate]:
    authoritative = {column.id.casefold(): column for column in authoritative_columns}
    sql_policy = policy.get("sql", {})
    sensitive_ids = {
        str(column_id).casefold()
        for column_id in sql_policy.get("sensitive_columns", [])
    }
    sensitive_names = {
        str(name).casefold()
        for name in sql_policy.get("sensitive_column_names", [])
    }
    candidates: dict[str, ColumnCandidate] = {}
    for table_info in sql_context.get("tables", []) or []:
        table = str(table_info.get("id") or table_info.get("name") or "")
        for column_info in table_info.get("columns", []) or []:
            exposed_name = str(column_info.get("name") or "")
            exposed_id = str(column_info.get("id") or f"{table}.{exposed_name}")
            column = authoritative.get(exposed_id.casefold())
            if column is None:
                raise ValueError(f"authoritative_column_missing: {exposed_id}")
            candidates[column.id] = ColumnCandidate(
                candidate_id=column.id,
                table=column.table_id,
                name=column.name,
                aliases=_unique_strings(column.alias),
                role=column.role,
                projectable=column.id.casefold() not in sensitive_ids
                and column.name.casefold() not in sensitive_names,
                data_type=column.type,
                description=column.description,
            )
    return candidates


# Candidate record shape adapted from Canner/WrenAI's Apache-2.0
# schema_indexer.py `_measure_record()` at commit
# 3dac00a178aa5e78e9d6472fc6e048d17d1f7271. Modified for ShopInsight's
# typed, request-scoped catalog; see docs/third-party-code.md.
def _build_authoritative_metric_candidates(
    sql_context: Mapping[str, Any],
    *,
    authoritative_metrics: list[MetricInfo],
) -> dict[str, MetricCandidate]:
    by_id = {metric.id.casefold(): metric for metric in authoritative_metrics}
    by_name: dict[str, list[MetricInfo]] = {}
    for metric in authoritative_metrics:
        by_name.setdefault(metric.name.casefold(), []).append(metric)

    candidates: dict[str, MetricCandidate] = {}
    for metric_info in sql_context.get("metrics", []) or []:
        exposed_id = str(metric_info.get("id") or "")
        exposed_name = str(metric_info.get("name") or "")
        metric = by_id.get(exposed_id.casefold()) if exposed_id else None
        if metric is None:
            matches = by_name.get(exposed_name.casefold(), [])
            if len(matches) != 1:
                raise ValueError(
                    f"authoritative_metric_missing: {exposed_id or exposed_name}"
                )
            metric = matches[0]
        candidates[metric.id] = MetricCandidate(
            candidate_id=metric.id,
            name=metric.name,
            aliases=_unique_strings(metric.alias),
            relevant_columns=_unique_strings(metric.relevant_columns),
            aggregation=metric.aggregation,
            expression=metric.expression,
            description=metric.description,
        )
    return candidates


def _build_relationship_candidates(
    authoritative_columns: list[ColumnInfo],
    *,
    exposed_table_ids: set[str],
) -> dict[str, RelationshipCandidate]:
    graph = build_schema_graph(authoritative_columns)
    candidates: dict[str, RelationshipCandidate] = {}
    exposed = {table_id.casefold() for table_id in exposed_table_ids}
    for edge in unique_join_edges(graph):
        if edge.left_table not in exposed or edge.right_table not in exposed:
            continue
        candidate_id = stable_relationship_id(edge)
        candidates[candidate_id] = RelationshipCandidate(
            candidate_id=candidate_id,
            left_table_id=edge.left_table,
            left_column_id=edge.left_column,
            right_table_id=edge.right_table,
            right_column_id=edge.right_column,
        )
    return candidates


def _build_value_candidates(
    columns: Mapping[str, ColumnCandidate],
    retrieved_value_infos: list[ValueInfo],
) -> dict[str, ValueCandidate]:
    candidates: dict[str, ValueCandidate] = {}
    for value_info in retrieved_value_infos:
        if value_info.column_id not in columns:
            continue
        existing = candidates.get(value_info.id)
        if existing is not None:
            if (
                existing.column_id != value_info.column_id
                or existing.canonical_value != value_info.value
            ):
                raise ValueError(f"value_candidate_identity_conflict: {value_info.id}")
            continue
        matched_aliases = [
            text
            for text in value_info.matched_texts
            if text and text != value_info.value
        ]
        sources = list(getattr(value_info, "sources", None) or [])
        candidates[value_info.id] = ValueCandidate(
            candidate_id=value_info.id,
            canonical_value=value_info.value,
            aliases=_unique_strings(matched_aliases),
            column_id=value_info.column_id,
            source="_or_".join(sources) if sources else "retrieval",
        )
    return candidates


def _unique_strings(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))
