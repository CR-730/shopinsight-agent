"""Assemble the phase-two semantic planner without activating the Graph."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

from app.agent.failure import build_failure
from app.agent.memory import sliding_conversation_history
from app.agent.semantic_planning.catalog import (
    build_semantic_candidate_catalog,
)
from app.agent.semantic_planning.interpreter import interpret_semantics
from app.agent.semantic_planning.issues import SemanticPlanningResult
from app.agent.semantic_planning.plan_validator import validate_plan
from app.agent.semantic_planning.resolver import (
    SemanticResolutionContext,
    resolve_semantic_draft,
)
from app.agent.semantic_planning.resolvers.join import resolve_join_preferences
from app.conf.policy_config import load_policy_config


async def build_semantic_plan(state, runtime) -> dict[str, Any]:
    """Build, resolve, and validate a plan while keeping Draft request-local."""

    try:
        query = str(state.get("query") or "")
        sql_context = state.get("sql_context") or {}
        retrieval_context = state.get("retrieval_context") or {}
        repository = runtime.context["meta_mysql_repository"]
        columns, metrics, value_aliases = await asyncio.gather(
            repository.list_column_infos(),
            repository.list_metric_infos(),
            repository.list_value_aliases(),
        )
        metadata_version = str(
            runtime.context.get("metadata_cache_version") or ""
        )
        reference_date = _reference_date(
            runtime.context.get("semantic_reference_date")
        )
        policy = load_policy_config()
        temporal_column_id = str(
            (policy.get("semantic") or {}).get("temporal_column_id") or ""
        )
        if not temporal_column_id:
            raise ValueError("temporal_column_id_required")

        catalog = build_semantic_candidate_catalog(
            sql_context=sql_context,
            retrieved_value_infos=retrieval_context.get("values") or [],
            value_aliases=value_aliases,
            authoritative_columns=columns,
            authoritative_metrics=metrics,
            metadata_version=metadata_version,
            policy=policy,
        )
        conversation_history = sliding_conversation_history(
            state.get("conversation_messages") or []
        )
        draft = await interpret_semantics(
            query,
            runtime,
            conversation_history=conversation_history,
            catalog=catalog,
        )
        resolution = await resolve_semantic_draft(
            draft,
            SemanticResolutionContext(
                catalog=catalog,
                dw_repository=runtime.context["dw_mysql_repository"],
                trusted_sources=_trusted_sources(
                    query,
                    conversation_history=conversation_history,
                ),
                reference_date=reference_date,
                temporal_column_id=temporal_column_id,
            ),
        )
        if resolution.status == "resolved":
            join_resolution = resolve_join_preferences(
                draft.join_mentions,
                catalog=catalog,
                trusted_sources=_trusted_sources(
                    query,
                    conversation_history=conversation_history,
                ),
            )
            if join_resolution.status == "resolved":
                result = validate_plan(
                    resolution.plan,
                    catalog,
                    join_preferences=join_resolution.preferences,
                )
            else:
                result = SemanticPlanningResult(
                    status=join_resolution.status,
                    issues=list(join_resolution.issues),
                )
        else:
            result = resolution
    except Exception as exc:
        return _failed_update(state, exc)
    return _result_update(state, result)


def _result_update(state, result: SemanticPlanningResult) -> dict[str, Any]:
    trace = {
        **(state.get("trace") or {}),
        "planning_issues": [_public_issue(issue) for issue in result.issues],
    }
    if result.status == "resolved":
        return {
            "semantic_plan": result.plan.model_dump(mode="json"),
            "trace": trace,
            "failure": None,
        }
    issue = result.issues[0]
    disposition = "failed" if result.status == "failed" else "blocked"
    category = "system" if disposition == "failed" else "semantic_planning"
    return {
        "trace": trace,
        "failure": build_failure(
            category=category,
            stage="semantic_planning",
            code=issue.code,
            message=issue.code,
            disposition=disposition,
            user_message=(
                "当前问题存在无法唯一确定的业务含义，请补充或明确查询条件。"
                if disposition == "blocked"
                else ""
            ),
        ),
    }


def _failed_update(state, exc: Exception) -> dict[str, Any]:
    issue = {
        "phase": "system",
        "code": "semantic_planning_failed",
        "source_span": "",
        "candidate_ids": [],
    }
    return {
        "trace": {
            **(state.get("trace") or {}),
            "planning_issues": [issue],
        },
        "failure": build_failure(
            category="system",
            stage="semantic_planning",
            code="semantic_planning_failed",
            message=f"semantic planning failed: {exc.__class__.__name__}",
            disposition="failed",
        ),
    }


def _public_issue(issue) -> dict[str, Any]:
    return {
        "phase": issue.phase,
        "code": issue.code,
        "source_span": issue.source_span,
        "candidate_ids": list(issue.candidate_ids),
    }


def _reference_date(value) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        return date.fromisoformat(value)
    raise ValueError("semantic_reference_date_required")


def _trusted_sources(
    query: str,
    *,
    conversation_history: str,
) -> tuple[str, ...]:
    if conversation_history and _allows_history_mentions(query):
        return query, conversation_history
    return (query,)


def _allows_history_mentions(query: str) -> bool:
    normalized = "".join(query.split()).strip("，。！？!?；;")
    return normalized in {
        "可以",
        "可以的",
        "好",
        "好的",
        "行",
        "继续",
        "继续吧",
        "继续查",
        "就这个",
        "就查这个",
        "就按这个查",
        "按这个查",
        "按这个来",
        "是",
        "是的",
        "对",
        "对的",
    }


__all__ = ["build_semantic_plan"]
