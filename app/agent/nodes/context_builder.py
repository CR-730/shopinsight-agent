"""Build retrieval context before business binding."""

import asyncio

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.retrieval_context import (
    extract_retrieval_keywords,
    merge_retrieved_context,
    recall_column_context,
    recall_metric_context,
    recall_sql_memory_context,
    recall_value_context,
)
from app.agent.state import DataAgentState


async def context_builder(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """Run SQL memory recall and RAG retrieval behind one graph node."""

    current_state = dict(state)
    accumulated_update = {}

    sql_memory_update = await recall_sql_memory_context(
        current_state, runtime.context
    )
    current_state.update(sql_memory_update)
    accumulated_update.update(sql_memory_update)

    keyword_update = await extract_retrieval_keywords(current_state)
    current_state.update(keyword_update)
    accumulated_update.update(keyword_update)

    column_update, value_update, metric_update = await asyncio.gather(
        recall_column_context(current_state, runtime.context),
        recall_value_context(current_state, runtime.context),
        recall_metric_context(current_state, runtime.context),
    )
    current_state.update(column_update)
    current_state.update(value_update)
    current_state.update(metric_update)
    accumulated_update.update(column_update)
    accumulated_update.update(value_update)
    accumulated_update.update(metric_update)

    merge_update = await merge_retrieved_context(current_state, runtime.context)
    accumulated_update.update(merge_update)
    return accumulated_update
