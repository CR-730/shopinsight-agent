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
    current_state.update(table_update)
    accumulated_update.update(table_update)

    metric_update = filter_metric_context(current_state)
    current_state.update(metric_update)
    accumulated_update.update(metric_update)

    extra_context_update = await add_runtime_context(current_state, runtime.context)
    current_state.update(extra_context_update)
    accumulated_update.update(extra_context_update)

    return accumulated_update
