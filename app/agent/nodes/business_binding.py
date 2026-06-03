"""Business binding node: extract candidates, validate with metadata, block unresolved."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.agent.business_binding.candidates import extract_binding_candidates
from app.agent.business_binding.validator import (
    BindingValidationContext,
    validate_binding_candidates,
    validate_business_binding_state,
    validated_enum_values,
)
from app.core.log import logger

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

    from app.agent.context import DataAgentContext
    from app.agent.state import DataAgentState


async def business_binding(
    state: DataAgentState, runtime: Runtime[DataAgentContext]
) -> dict[str, Any]:
    """Bind user language to canonical metrics, filters, and time constraints."""

    writer = runtime.stream_writer
    step = "业务绑定"
    writer({"type": "progress", "step": step, "status": "running"})

    query = state.get("query") or ""
    enum_aliases = _value_alias_map(
        await runtime.context["meta_mysql_repository"].list_value_aliases()
    )

    candidates = await extract_binding_candidates(
        query,
        runtime,
        metric_infos=state.get("metric_infos") or [],
        retrieved_value_infos=state.get("retrieved_value_infos") or [],
        enum_aliases=enum_aliases,
    )
    binding = await validate_binding_candidates(
        candidates,
        BindingValidationContext(
            metric_infos=state.get("metric_infos") or [],
            table_infos=state.get("table_infos") or [],
            retrieved_value_infos=state.get("retrieved_value_infos") or [],
            enum_aliases=enum_aliases,
            dw_mysql_repository=runtime.context["dw_mysql_repository"],
        ),
    )
    update = {
        "business_binding": binding,
        "metric_bindings": binding["metrics"],
        "resolved_filters": binding["filters"],
        "time_binding": binding["time"],
        "validated_enum_values": validated_enum_values(binding["filters"]),
        "unresolved_bindings": binding["unresolved"],
        "ambiguous_bindings": binding["ambiguous"],
        "safety_error": None,
    }

    logger.info(f"业务绑定结果：{binding}")
    rule_error = validate_business_binding_state(update)
    if rule_error:
        logger.warning(f"{step} blocked query: {rule_error}")
        writer(
            {"type": "progress", "step": step, "status": "blocked", "error": rule_error}
        )
        return {**update, "safety_error": rule_error, "blocked_by": "business_binding"}
    writer({"type": "progress", "step": step, "status": "success"})
    return update


def _value_alias_map(value_aliases: list[Any]) -> dict[str, dict[str, str]]:
    aliases: dict[str, dict[str, str]] = {}
    for value_alias in value_aliases:
        column_id = str(getattr(value_alias, "column_id", "") or "")
        alias = str(getattr(value_alias, "alias", "") or "")
        canonical_value = str(getattr(value_alias, "canonical_value", "") or "")
        if column_id and alias and canonical_value:
            aliases.setdefault(column_id, {})[alias] = canonical_value
    return aliases
