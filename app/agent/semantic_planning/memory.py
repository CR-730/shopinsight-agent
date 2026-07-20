"""Read trusted semantic plans from successful SQL memory records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.agent.semantic_planning.plan import SemanticQueryPlan


def semantic_plan_from_memory_args(
    args: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return only an explicitly stored and complete semantic plan."""

    raw_plan = args.get("semantic_plan")
    if not raw_plan:
        return None
    try:
        plan = SemanticQueryPlan.model_validate(raw_plan)
    except ValueError:
        return None
    return plan.model_dump(mode="json")


__all__ = ["semantic_plan_from_memory_args"]
