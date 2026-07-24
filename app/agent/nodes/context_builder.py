"""Build retrieval context before semantic planning."""

import asyncio

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.retrieval_context import (
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

    column_update, value_update, metric_update = await asyncio.gather(
        recall_column_context(current_state, runtime.context),
        recall_value_context(current_state, runtime.context),
        recall_metric_context(current_state, runtime.context),
    )
    current_state.update(column_update)
    current_state.update(value_update)
    current_state.update(metric_update)

    merge_update = await merge_retrieved_context(current_state, runtime.context)
    current_state.update(merge_update)
    accumulated_update.update(
        {
            "sql_memory_examples": sql_memory_update.get("sql_memory_examples") or [],
            "retrieval_context": {
                "columns": column_update.get("retrieved_column_infos") or [],
                "metrics": metric_update.get("retrieved_metric_infos") or [],
                "values": value_update.get("retrieved_value_infos") or [],
            },
            "sql_context": {
                "tables": merge_update.get("table_infos") or [],
                "metrics": merge_update.get("metric_infos") or [],
            },
            "trace": {
                "keywords": _unique_strings(
                    column_update.get("column_retrieval_queries") or [],
                    value_update.get("value_retrieval_queries") or [],
                    metric_update.get("metric_retrieval_queries") or [],
                ),
                "retrieval_queries": {
                    "columns": column_update.get("column_retrieval_queries") or [],
                    "metrics": metric_update.get("metric_retrieval_queries") or [],
                    "values": value_update.get("value_retrieval_queries") or [],
                },
                "retrieved_columns": _entity_ids(
                    column_update.get("retrieved_column_infos") or []
                ),
                "retrieved_metrics": _metric_names(
                    metric_update.get("retrieved_metric_infos") or []
                ),
                "retrieved_values": _entity_ids(
                    value_update.get("retrieved_value_infos") or []
                ),
            },
        }
    )
    return accumulated_update


def _entity_ids(items) -> list[str]:
    return [
        str(getattr(item, "id", item))
        for item in items
        if str(getattr(item, "id", item))
    ]


def _metric_names(items) -> list[str]:
    return [
        str(getattr(item, "name", item))
        for item in items
        if str(getattr(item, "name", item))
    ]


def _unique_strings(*groups: list[str]) -> list[str]:
    return list(dict.fromkeys(item for group in groups for item in group if item))
