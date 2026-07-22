"""Project a trusted semantic plan into an unambiguous SQL compiler input."""

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
        if predicate.get("start_date_id") is not None:
            predicate.pop("start_date", None)
        if predicate.get("end_date_id") is not None:
            predicate.pop("end_date", None)
    return payload


__all__ = ["project_plan_for_sql"]
