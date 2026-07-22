"""Orchestrate deterministic semantic resolvers without JOIN or SQL work."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

from app.agent.semantic_planning.catalog import SemanticCandidateCatalog
from app.agent.semantic_planning.draft import (
    EnumPredicateMention,
    NumericPredicateMention,
    SemanticDraft,
    TemporalPredicateMention,
)
from app.agent.semantic_planning.issues import (
    PlanningIssue,
    PlanningStatus,
    SemanticPlanningResult,
)
from app.agent.semantic_planning.plan import (
    DimensionPlan,
    MeasurePlan,
    PlanProvenance,
    SemanticQueryPlan,
)
from app.agent.semantic_planning.resolvers.dimension import (
    DimensionResolutionContext,
    resolve_dimension,
)
from app.agent.semantic_planning.resolvers.enum_predicate import (
    EnumResolutionContext,
    resolve_enum_predicate,
)
from app.agent.semantic_planning.resolvers.measure import (
    MeasureResolutionContext,
    resolve_measure,
)
from app.agent.semantic_planning.resolvers.numeric_predicate import (
    NumericResolutionContext,
    resolve_numeric_predicate,
)
from app.agent.semantic_planning.resolvers.order_limit import (
    OrderLimitResolutionContext,
    resolve_order_and_limit,
)
from app.agent.semantic_planning.resolvers.temporal import (
    TemporalResolutionContext,
    resolve_temporal_predicate,
)


@dataclass(frozen=True)
class SemanticResolutionContext:
    catalog: SemanticCandidateCatalog
    trusted_sources: tuple[str, ...]
    reference_date: date
    temporal_column_id: str


async def resolve_semantic_draft(
    draft: SemanticDraft,
    context: SemanticResolutionContext,
) -> SemanticPlanningResult:
    """Resolve every mention or fail closed without exposing a partial plan."""

    issues: list[PlanningIssue] = []
    blocking_statuses: list[PlanningStatus] = []
    measures: list[MeasurePlan] = []
    dimensions: list[DimensionPlan] = []
    predicates = []
    provenance: list[PlanProvenance] = []

    _collect_ambiguity_reports(
        draft,
        context,
        issues=issues,
        blocking_statuses=blocking_statuses,
    )

    measure_context = MeasureResolutionContext(
        catalog=context.catalog,
        trusted_sources=context.trusted_sources,
    )
    for mention in draft.measure_mentions:
        result = resolve_measure(mention, measure_context)
        if result.status != "resolved":
            _record_failure(result, issues, blocking_statuses)
            continue
        _append_unique(measures, result.plan, lambda item: item.metric_id)
        _append_provenance(
            provenance,
            raw_text=mention.raw_text,
            resolved_id=result.plan.metric_id,
            method="authoritative_metric",
            evidence=result.plan.aggregation,
        )

    dimension_context = DimensionResolutionContext(
        catalog=context.catalog,
        trusted_sources=context.trusted_sources,
    )
    for mention in draft.dimension_mentions:
        result = resolve_dimension(mention, dimension_context)
        if result.status != "resolved":
            _record_failure(result, issues, blocking_statuses)
            continue
        _append_unique(
            dimensions,
            result.plan,
            lambda item: (item.column_id, item.role),
        )
        _append_provenance(
            provenance,
            raw_text=mention.raw_text,
            resolved_id=result.plan.column_id,
            method="authoritative_column",
            evidence=result.plan.role,
        )

    temporal_mentions = [
        mention
        for mention in draft.predicate_mentions
        if isinstance(mention, TemporalPredicateMention)
    ]
    if len(temporal_mentions) > 1:
        issues.append(
            PlanningIssue(
                phase="resolution",
                code="multiple_time_turns_unsupported",
                source_span=" ".join(mention.raw_text for mention in temporal_mentions),
                candidate_ids=[],
                details={},
            )
        )
        blocking_statuses.append("unresolved")

    for mention in draft.predicate_mentions:
        if isinstance(mention, EnumPredicateMention):
            result = await resolve_enum_predicate(
                mention,
                EnumResolutionContext(
                    catalog=context.catalog,
                    trusted_sources=context.trusted_sources,
                ),
            )
            method = "controlled_value_candidate"
        elif isinstance(mention, NumericPredicateMention):
            result = resolve_numeric_predicate(
                mention,
                NumericResolutionContext(
                    catalog=context.catalog,
                    trusted_sources=context.trusted_sources,
                ),
            )
            method = "decimal_normalization"
        else:
            if len(temporal_mentions) > 1:
                continue
            result = resolve_temporal_predicate(
                mention,
                TemporalResolutionContext(
                    catalog=context.catalog,
                    trusted_sources=context.trusted_sources,
                    reference_date=context.reference_date,
                    temporal_column_id=context.temporal_column_id,
                ),
            )
            method = "jionlp_exact_interval"
        if result.status != "resolved":
            _record_failure(result, issues, blocking_statuses)
            continue
        _append_unique(predicates, result.plan, _model_key)
        _append_provenance(
            provenance,
            raw_text=mention.raw_text,
            resolved_id=_predicate_resolved_id(result.plan),
            method=method,
            evidence=result.plan.kind,
        )

    order_result = resolve_order_and_limit(
        draft.order_mentions,
        draft.limit_mentions,
        OrderLimitResolutionContext(
            selected_measures=tuple(measures),
            selected_dimensions=tuple(dimensions),
            trusted_sources=context.trusted_sources,
        ),
    )
    order_by = []
    limit = None
    if order_result.status != "resolved":
        _record_failure(order_result, issues, blocking_statuses)
    else:
        order_by = order_result.order_by or []
        limit = order_result.limit
        for mention, order in zip(draft.order_mentions, order_by, strict=True):
            _append_provenance(
                provenance,
                raw_text=mention.raw_text,
                resolved_id=order.target_id,
                method="selected_object_order",
                evidence=order.direction,
            )
        if draft.limit_mentions and limit is not None:
            _append_provenance(
                provenance,
                raw_text=draft.limit_mentions[0].raw_text,
                resolved_id=f"limit:{limit}",
                method="explicit_integer_limit",
                evidence=str(limit),
            )

    if not measures and not dimensions and not issues:
        issues.append(
            PlanningIssue(
                phase="resolution",
                code="business_object_not_planned",
                source_span=(
                    context.trusted_sources[0] if context.trusted_sources else ""
                ),
                candidate_ids=[],
                details={},
            )
        )
        blocking_statuses.append("unresolved")

    if issues:
        return SemanticPlanningResult(
            status=_overall_status(blocking_statuses),
            issues=issues,
        )
    return SemanticPlanningResult(
        status="resolved",
        plan=SemanticQueryPlan(
            version="1",
            metadata_version=context.catalog.metadata_version,
            measures=measures,
            dimensions=dimensions,
            predicates=predicates,
            order_by=order_by,
            limit=limit,
            joins=[],
            required_table_ids=[],
            required_column_ids=[],
            provenance=provenance,
        ),
    )


def _collect_ambiguity_reports(
    draft: SemanticDraft,
    context: SemanticResolutionContext,
    *,
    issues: list[PlanningIssue],
    blocking_statuses: list[PlanningStatus],
) -> None:
    all_candidates = {
        *context.catalog.metrics,
        *context.catalog.columns,
        *context.catalog.values,
    }
    for report in draft.ambiguity_reports:
        invalid_ids = [
            candidate_id
            for candidate_id in dict.fromkeys(report.candidate_ids)
            if candidate_id not in all_candidates
        ]
        if not report.raw_text or not any(
            report.raw_text in source for source in context.trusted_sources
        ):
            code = "untrusted_source_span"
            status: PlanningStatus = "unresolved"
        elif invalid_ids:
            code = "invalid_candidate_id"
            status = "unresolved"
        else:
            code = report.reason or "semantic_ambiguity"
            status = "ambiguous"
        issues.append(
            PlanningIssue(
                phase="interpretation",
                code=code,
                source_span=report.raw_text,
                candidate_ids=invalid_ids or list(dict.fromkeys(report.candidate_ids)),
                details={},
            )
        )
        blocking_statuses.append(status)


def _record_failure(result, issues, statuses) -> None:
    if result.issue is not None:
        issues.append(result.issue)
    statuses.append(result.status)


def _overall_status(statuses: list[PlanningStatus]) -> PlanningStatus:
    if "failed" in statuses:
        return "failed"
    if "ambiguous" in statuses:
        return "ambiguous"
    return "unresolved"


def _append_unique(items: list, item, key) -> None:
    item_key = key(item)
    if all(key(existing) != item_key for existing in items):
        items.append(item)


def _model_key(item) -> str:
    return json.dumps(item.model_dump(mode="json"), sort_keys=True)


def _predicate_resolved_id(predicate) -> str:
    return str(getattr(predicate, "column_id", None) or predicate.target_id)


def _append_provenance(
    items: list[PlanProvenance],
    *,
    raw_text: str,
    resolved_id: str,
    method: str,
    evidence: str,
) -> None:
    item = PlanProvenance(
        raw_text=raw_text,
        resolved_id=resolved_id,
        method=method,
        evidence=evidence,
    )
    _append_unique(
        items,
        item,
        lambda value: (
            value.raw_text,
            value.resolved_id,
            value.method,
            value.evidence,
        ),
    )


__all__ = ["SemanticResolutionContext", "resolve_semantic_draft"]
