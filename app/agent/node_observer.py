"""LangGraph node observability wrapper."""

import time
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext


def traced_node(name: str, node: Callable[[dict, Runtime[DataAgentContext]], Awaitable]):
    async def wrapper(state: dict, runtime: Runtime[DataAgentContext]) -> Any:
        started_at = time.perf_counter()
        try:
            result = await node(state, runtime)
            _record(runtime, name, started_at)
            return result
        except Exception as exc:
            _record(runtime, name, started_at, exc.__class__.__name__)
            raise

    wrapper.__name__ = getattr(node, "__name__", name)
    return wrapper


def _record(
    runtime: Runtime[DataAgentContext],
    name: str,
    started_at: float,
    error_type: str | None = None,
):
    cost_tracker = runtime.context.get("cost_tracker")
    if cost_tracker is None:
        return
    cost_tracker.add_node_event(
        name,
        latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
        error_type=error_type,
    )
