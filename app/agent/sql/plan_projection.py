"""Project a trusted semantic plan into an unambiguous SQL compiler input."""

from datetime import date
from typing import Any

from app.agent.semantic_planning.plan import SemanticQueryPlan


def project_plan_for_sql(
    semantic_plan: SemanticQueryPlan | dict[str, Any],
) -> dict[str, Any]:
    """Expose one physical representation for each temporal boundary."""

    plan = (
        semantic_plan
        if isinstance(semantic_plan, SemanticQueryPlan)
        else SemanticQueryPlan.model_validate(semantic_plan)
    )
    payload = plan.model_dump(mode="json")
    for predicate in payload["predicates"]:
        if predicate["kind"] != "temporal":
            continue
        predicate["start_date_id"] = date_id_from_iso(predicate.pop("start_date"))
        predicate["end_date_id"] = date_id_from_iso(predicate.pop("end_date"))
    return payload


def date_id_from_iso(value: str | None) -> int | None:
    if value is None:
        return None
    return int(date.fromisoformat(value).strftime("%Y%m%d"))


__all__ = ["date_id_from_iso", "project_plan_for_sql"]
