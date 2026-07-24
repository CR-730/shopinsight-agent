"""Validate ordering and explicit row limits against selected plan objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.agent.semantic_planning.draft import LimitMention, OrderMention
from app.agent.semantic_planning.issues import PlanningIssue
from app.agent.semantic_planning.plan import (
    DimensionPlan,
    MeasurePlan,
    OrderByPlan,
)
from app.agent.semantic_planning.resolvers.common import select_one_candidate

OrderLimitStatus = Literal["resolved", "unresolved", "ambiguous"]
@dataclass(frozen=True)
class OrderLimitResolutionContext:
    selected_measures: tuple[MeasurePlan, ...]
    selected_dimensions: tuple[DimensionPlan, ...]


@dataclass(frozen=True)
class OrderLimitResolution:
    status: OrderLimitStatus
    order_by: list[OrderByPlan] | None = None
    limit: int | None = None
    issue: PlanningIssue | None = None


def resolve_order_and_limit(
    order_mentions: list[OrderMention],
    limit_mentions: list[LimitMention],
    context: OrderLimitResolutionContext,
) -> OrderLimitResolution:
    selected = {
        measure.metric_id: ("measure", measure) for measure in context.selected_measures
    }
    selected.update(
        {
            dimension.column_id: ("dimension", dimension)
            for dimension in context.selected_dimensions
        }
    )

    order_by: list[OrderByPlan] = []
    for mention in order_mentions:
        selection = select_one_candidate(
            raw_text=mention.raw_text,
            candidate_ids=mention.target_candidate_ids,
            catalog=selected,
            issue_prefix="order_target",
        )
        if selection.status != "resolved":
            issue = selection.issue
            if issue and issue.code == "order_target_not_bound":
                issue = issue.model_copy(update={"code": "order_target_not_selected"})
            if issue and issue.code == "invalid_candidate_id":
                issue = issue.model_copy(update={"code": "order_target_not_selected"})
            return OrderLimitResolution(status=selection.status, issue=issue)
        target_type, _ = selection.candidate
        order_by.append(
            OrderByPlan(
                target_type=target_type,
                target_id=selection.candidate_id,
                direction=mention.direction,
            )
        )

    limit_result = _resolve_limit(limit_mentions)
    if isinstance(limit_result, OrderLimitResolution):
        return limit_result
    limit = limit_result

    top_n = any(
        _looks_like_top_n(mention.raw_text)
        for mention in [*order_mentions, *limit_mentions]
    )
    if top_n and len(order_by) != 1:
        return _blocked(
            "ambiguous",
            "top_n_order_target_ambiguous",
            " ".join(mention.raw_text for mention in limit_mentions),
            [
                item
                for mention in order_mentions
                for item in mention.target_candidate_ids
            ],
        )
    return OrderLimitResolution(
        status="resolved",
        order_by=order_by,
        limit=limit,
    )


def _resolve_limit(
    mentions: list[LimitMention],
) -> int | None | OrderLimitResolution:
    if not mentions:
        return None
    if len(mentions) != 1:
        return _blocked(
            "ambiguous",
            "limit_ambiguous",
            " ".join(mention.raw_text for mention in mentions),
            [],
        )
    mention = mentions[0]
    limit = mention.value
    if not 1 <= limit <= 1000:
        return _blocked("unresolved", "limit_out_of_range", mention.raw_text, [])
    return limit


def _looks_like_top_n(raw_text: str) -> bool:
    folded = raw_text.casefold()
    return "top" in folded or any(
        marker in raw_text for marker in ("前", "最高", "最低")
    )


def _blocked(
    status: Literal["unresolved", "ambiguous"],
    code: str,
    source_span: str,
    candidate_ids: list[str],
) -> OrderLimitResolution:
    return OrderLimitResolution(
        status=status,
        issue=PlanningIssue(
            phase="resolution",
            code=code,
            source_span=source_span,
            candidate_ids=list(dict.fromkeys(candidate_ids)),
            details={},
        ),
    )


__all__ = [
    "OrderLimitResolution",
    "OrderLimitResolutionContext",
    "resolve_order_and_limit",
]
