"""Validate a resolved semantic plan and materialize its unique JOIN closure."""

from __future__ import annotations

from datetime import date
from decimal import InvalidOperation

from app.agent.predicate_normalization import canonical_number
from app.agent.schema_relations import (
    build_relationship_graph,
    find_unique_shortest_join_closure,
)
from app.agent.semantic_planning.catalog import SemanticCandidateCatalog
from app.agent.semantic_planning.issues import (
    PlanningIssue,
    SemanticPlanningResult,
)
from app.agent.semantic_planning.plan import (
    EnumPredicate,
    JoinPlan,
    NumericPredicate,
    RequiredColumnPlan,
    SemanticQueryPlan,
    TemporalPredicate,
)
from app.agent.semantic_planning.predicate_normalization import (
    normalize_plan_predicates,
)
from app.agent.semantic_planning.resolvers.join import ResolvedJoinPreference

_NUMERIC_TYPES = ("int", "decimal", "numeric", "float", "double", "real")


def validate_plan(
    plan: SemanticQueryPlan,
    catalog: SemanticCandidateCatalog,
    *,
    join_preferences: tuple[ResolvedJoinPreference, ...] = (),
) -> SemanticPlanningResult:
    """Return a fully materialized plan or no plan at all."""

    normalization = normalize_plan_predicates(plan.predicates)
    if normalization.issues:
        return SemanticPlanningResult(
            status="unresolved",
            issues=list(normalization.issues),
        )
    plan = plan.model_copy(update={"predicates": list(normalization.predicates)})

    issues: list[PlanningIssue] = []
    if plan.version != "1":
        issues.append(_issue("unsupported_plan_version"))
    if plan.metadata_version != catalog.metadata_version:
        issues.append(
            _issue(
                "metadata_version_mismatch",
                details={
                    "plan": plan.metadata_version,
                    "catalog": catalog.metadata_version,
                },
            )
        )
    if (
        plan.joins
        or plan.required_table_ids
        or plan.required_column_ids
        or plan.required_columns
    ):
        issues.append(_issue("plan_closure_must_be_empty"))

    selected_measures: set[str] = set()
    selected_dimensions: set[str] = set()
    required_columns: set[str] = set()
    has_projection = False

    for measure in plan.measures:
        metric = catalog.metrics.get(measure.metric_id)
        if metric is None:
            issues.append(_issue("invalid_candidate_id", [measure.metric_id]))
            continue
        if (
            measure.name != metric.name
            or measure.aggregation != metric.aggregation
            or measure.expression != metric.expression
            or tuple(measure.source_column_ids) != metric.relevant_columns
        ):
            issues.append(_issue("metric_definition_mismatch", [measure.metric_id]))
            continue
        missing_columns = [
            column_id
            for column_id in measure.source_column_ids
            if column_id not in catalog.columns
        ]
        if missing_columns:
            issues.append(_issue("invalid_candidate_id", missing_columns))
            continue
        selected_measures.add(measure.metric_id)
        required_columns.update(measure.source_column_ids)

    for dimension in plan.dimensions:
        column = catalog.columns.get(dimension.column_id)
        if column is None:
            issues.append(_issue("invalid_candidate_id", [dimension.column_id]))
            continue
        if dimension.role == "group_by" and column.role != "dimension":
            issues.append(_issue("group_by_role_invalid", [dimension.column_id]))
            continue
        if dimension.role == "projection":
            if not column.projectable:
                issues.append(_issue("projection_not_allowed", [dimension.column_id]))
                continue
            has_projection = True
        selected_dimensions.add(dimension.column_id)
        required_columns.add(dimension.column_id)

    if not selected_measures and not has_projection:
        issues.append(_issue("business_object_not_planned"))

    for predicate in plan.predicates:
        if isinstance(predicate, EnumPredicate):
            _validate_enum(predicate, catalog, required_columns, issues)
        elif isinstance(predicate, NumericPredicate):
            _validate_numeric(
                predicate,
                catalog,
                selected_measures,
                required_columns,
                issues,
            )
        elif isinstance(predicate, TemporalPredicate):
            _validate_temporal(predicate, catalog, required_columns, issues)

    for order in plan.order_by:
        selected = (
            selected_measures if order.target_type == "measure" else selected_dimensions
        )
        if order.target_id not in selected:
            issues.append(_issue("order_target_not_selected", [order.target_id]))
    if plan.limit is not None and not 1 <= plan.limit <= 1000:
        issues.append(_issue("limit_out_of_range"))

    _validate_relationship_records(catalog, issues)
    required_relationships = []
    for preference in join_preferences:
        relationship = catalog.relationships.get(preference.relationship_candidate_id)
        if relationship is None:
            issues.append(
                _issue(
                    "invalid_candidate_id",
                    [preference.relationship_candidate_id],
                )
            )
            continue
        required_relationships.append(relationship)
    if issues:
        return SemanticPlanningResult(status="unresolved", issues=issues)

    required_tables = {
        catalog.columns[column_id].table for column_id in required_columns
    }
    for relationship in required_relationships:
        required_tables.update(
            {
                relationship.left_table_id,
                relationship.right_table_id,
            }
        )
    graph = build_relationship_graph(list(catalog.relationships.values()))
    closure = find_unique_shortest_join_closure(graph, required_tables)
    if closure.status == "unresolved":
        return SemanticPlanningResult(
            status="unresolved",
            issues=[_issue("join_path_not_found")],
        )
    if closure.status == "ambiguous":
        return SemanticPlanningResult(
            status="ambiguous",
            issues=[_issue("join_path_ambiguous")],
        )

    relationship_ids_by_endpoints = {
        frozenset(
            {
                relationship.left_column_id.casefold(),
                relationship.right_column_id.casefold(),
            }
        ): relationship.candidate_id
        for relationship in catalog.relationships.values()
    }
    closure_relationship_ids = {
        relationship_ids_by_endpoints[edge.column_ids] for edge in closure.edges
    }
    missing_required_relationships = [
        preference.relationship_candidate_id
        for preference in join_preferences
        if preference.relationship_candidate_id not in closure_relationship_ids
    ]
    if missing_required_relationships:
        return SemanticPlanningResult(
            status="unresolved",
            issues=[
                _issue(
                    "join_required_edge_missing",
                    missing_required_relationships,
                )
            ],
        )

    preferences_by_id = {
        preference.relationship_candidate_id: preference
        for preference in join_preferences
    }
    joins: list[JoinPlan] = []
    for edge in closure.edges:
        relationship_id = relationship_ids_by_endpoints[edge.column_ids]
        preference = preferences_by_id.get(relationship_id)
        join_type = preference.join_type if preference else "inner"
        left_column_id = edge.left_column
        right_column_id = edge.right_column
        if preference and preference.join_type == "left":
            if preference.left_table_candidate_id == edge.right_table:
                left_column_id, right_column_id = right_column_id, left_column_id
            elif preference.left_table_candidate_id != edge.left_table:
                return SemanticPlanningResult(
                    status="unresolved",
                    issues=[
                        _issue(
                            "join_left_table_invalid",
                            [relationship_id],
                        )
                    ],
                )
        joins.append(
            JoinPlan(
                left_column_id=left_column_id,
                right_column_id=right_column_id,
                join_type=join_type,
            )
        )

    all_columns = required_columns | set(closure.column_ids)
    return SemanticPlanningResult(
        status="resolved",
        plan=plan.model_copy(
            update={
                "joins": joins,
                "required_table_ids": sorted(closure.table_ids),
                "required_column_ids": sorted(all_columns),
                "required_columns": [
                    RequiredColumnPlan(
                        column_id=column_id,
                        data_type=catalog.columns[column_id].data_type,
                    )
                    for column_id in sorted(all_columns)
                ],
            }
        ),
    )


def _validate_enum(predicate, catalog, required_columns, issues) -> None:
    if predicate.column_id not in catalog.columns:
        issues.append(_issue("invalid_candidate_id", [predicate.column_id]))
        return
    if not predicate.canonical_values:
        issues.append(_issue("enum_value_required", [predicate.column_id]))
    if predicate.operator in {"eq", "neq"} and len(predicate.canonical_values) != 1:
        issues.append(_issue("enum_value_count_invalid", [predicate.column_id]))
    required_columns.add(predicate.column_id)


def _validate_numeric(
    predicate,
    catalog,
    selected_measures,
    required_columns,
    issues,
) -> None:
    expected_count = 2 if predicate.operator == "between" else 1
    if len(predicate.values) != expected_count:
        issues.append(_issue("numeric_boundary_count_invalid", [predicate.target_id]))
    try:
        [canonical_number(value) for value in predicate.values]
    except (InvalidOperation, ValueError):
        issues.append(_issue("numeric_value_invalid", [predicate.target_id]))

    if predicate.target_type == "measure":
        if predicate.target_id not in selected_measures:
            issues.append(_issue("numeric_target_not_selected", [predicate.target_id]))
            return
        if predicate.clause != "having":
            issues.append(_issue("numeric_clause_mismatch", [predicate.target_id]))
        metric = catalog.metrics[predicate.target_id]
        required_columns.update(metric.relevant_columns)
        return

    column = catalog.columns.get(predicate.target_id)
    if column is None:
        issues.append(_issue("invalid_candidate_id", [predicate.target_id]))
        return
    if not any(
        column.data_type.strip().casefold().startswith(prefix)
        for prefix in _NUMERIC_TYPES
    ):
        issues.append(_issue("numeric_target_type_invalid", [predicate.target_id]))
    if predicate.clause != "where":
        issues.append(_issue("numeric_clause_mismatch", [predicate.target_id]))
    required_columns.add(predicate.target_id)


def _validate_temporal(predicate, catalog, required_columns, issues) -> None:
    if predicate.column_id not in catalog.columns:
        issues.append(_issue("invalid_candidate_id", [predicate.column_id]))
        return
    if not predicate.start_date or not predicate.end_date:
        issues.append(_issue("temporal_boundary_invalid", [predicate.column_id]))
        return
    try:
        start = date.fromisoformat(predicate.start_date)
        end = date.fromisoformat(predicate.end_date)
    except ValueError:
        issues.append(_issue("temporal_boundary_invalid", [predicate.column_id]))
        return
    if start > end:
        issues.append(_issue("temporal_boundary_reversed", [predicate.column_id]))
    required_columns.add(predicate.column_id)


def _validate_relationship_records(catalog, issues) -> None:
    for relationship in catalog.relationships.values():
        endpoint_ids = [
            relationship.left_column_id,
            relationship.right_column_id,
        ]
        if any(column_id not in catalog.columns for column_id in endpoint_ids):
            issues.append(_issue("relationship_endpoint_invalid", endpoint_ids))
            continue
        if (
            catalog.columns[relationship.left_column_id].table
            != relationship.left_table_id
            or catalog.columns[relationship.right_column_id].table
            != relationship.right_table_id
        ):
            issues.append(_issue("relationship_endpoint_mismatch", endpoint_ids))


def _issue(
    code: str,
    candidate_ids: list[str] | None = None,
    *,
    details: dict | None = None,
) -> PlanningIssue:
    return PlanningIssue(
        phase="validation",
        code=code,
        source_span="",
        candidate_ids=candidate_ids or [],
        details=details or {},
    )


__all__ = ["validate_plan"]
