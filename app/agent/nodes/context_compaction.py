"""Compile minimal SQL context from a validated semantic plan."""

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.context_compaction import (
    add_runtime_context,
    compile_context_from_plan,
)
from app.agent.failure import build_failure
from app.agent.state import DataAgentState


async def context_compaction(
    state: DataAgentState, runtime: Runtime[DataAgentContext]
):
    """Compile authoritative metadata and append runtime-only DB context."""

    compiled = await compile_context_from_plan(state, runtime.context)
    issue = compiled.get("issue")
    if issue:
        reason = str(issue.get("reason") or "context_compaction_failed")
        messages = {
            "semantic_plan_missing": "Validated semantic plan is missing",
            "semantic_plan_dependency_invalid": "Semantic plan dependencies are invalid",
            "metadata_column_not_found": "Required column metadata was not found",
            "metadata_table_not_found": "Required table metadata was not found",
            "metadata_metric_not_found": "Required metric metadata was not found",
            "metadata_repository_unavailable": "Metadata repository is unavailable",
        }
        return {
            "failure": build_failure(
                category="system",
                stage="context_compaction",
                code=reason,
                message=messages.get(reason, "SQL context compilation failed"),
                disposition="failed",
            )
        }

    runtime_update = await add_runtime_context(state, runtime.context)
    return {
        "sql_context": {
            "tables": compiled.get("table_infos") or [],
            "metrics": compiled.get("metric_infos") or [],
            "date": runtime_update.get("date_info"),
            "db": runtime_update.get("db_info"),
        }
    }
