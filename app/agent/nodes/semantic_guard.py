"""Post-binding completeness check before SQL generation."""

from typing import Any

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.core.log import logger


async def semantic_guard(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """Block unresolved or ambiguous business bindings."""

    writer = runtime.stream_writer
    step = "RAG后业务语义闸门"
    writer({"type": "progress", "step": step, "status": "running"})

    rule_error = validate_business_binding_state(state)
    if rule_error:
        logger.warning(f"{step} blocked query: {rule_error}")
        writer(
            {"type": "progress", "step": step, "status": "blocked", "error": rule_error}
        )
        return {"safety_error": rule_error, "blocked_by": "semantic_guard"}

    writer({"type": "progress", "step": step, "status": "success"})
    return {
        "safety_error": None,
        "validated_enum_values": [
            literal
            for resolved_filter in state.get("resolved_filters") or []
            for literal in resolved_filter.get("allowed_sql_literals", [])
        ],
    }


def validate_business_binding_state(state: dict[str, Any]) -> str | None:
    """Return a blocking message when binding did not fully resolve business objects."""

    unresolved = state.get("unresolved_bindings") or []
    if unresolved:
        issue = unresolved[0]
        return _binding_issue_message(issue, "未解析")

    ambiguous = state.get("ambiguous_bindings") or []
    if ambiguous:
        issue = ambiguous[0]
        return _binding_issue_message(issue, "存在歧义")

    return None


def _binding_issue_message(issue: dict[str, Any], status: str) -> str:
    issue_type = issue.get("type") or "business_object"
    raw_text = issue.get("raw_text") or ""
    reason = issue.get("reason") or "unknown"
    return f"业务绑定{status}：{issue_type}={raw_text}，原因：{reason}"
