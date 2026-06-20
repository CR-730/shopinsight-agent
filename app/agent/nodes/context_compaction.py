"""Compact retrieved context after business binding."""

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.context_compaction import (
    add_runtime_context,
    filter_metric_context,
    filter_table_context,
)
from app.agent.state import DataAgentState


async def context_compaction(
    state: DataAgentState, runtime: Runtime[DataAgentContext]
):
    """Run table/metric compaction and runtime context enrichment."""

    current_state = dict(state)
    accumulated_update = {}

    table_update = await filter_table_context(current_state, runtime.context)
    current_state["sql_context"] = {
        **(current_state.get("sql_context") or {}),
        "tables": table_update.get("table_infos") or [],
    }

    metric_update = filter_metric_context(current_state)
    current_state["sql_context"] = {
        **(current_state.get("sql_context") or {}),
        "metrics": metric_update.get("metric_infos") or [],
    }

    extra_context_update = await add_runtime_context(current_state, runtime.context)
    current_state["sql_context"] = {
        **(current_state.get("sql_context") or {}),
        "date": extra_context_update.get("date_info"),
        "db": extra_context_update.get("db_info"),
    }
    accumulated_update["sql_context"] = current_state["sql_context"]

    return accumulated_update
