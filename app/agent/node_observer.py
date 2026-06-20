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
            timing = _record(runtime, name, started_at)
            if isinstance(result, dict):
                result["trace"] = _merge_trace(state, result, timing)
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
) -> dict[str, Any]:
    event = {
        "step": name,
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "error_type": error_type,
    }
    cost_tracker = runtime.context.get("cost_tracker")
    if cost_tracker is not None:
        cost_tracker.add_node_event(
            name,
            latency_ms=event["latency_ms"],
            error_type=error_type,
        )
    return event


def _merge_trace(state: dict, update: dict, timing: dict[str, Any]) -> dict[str, Any]:
    trace = {
        **(state.get("trace") or {}),
        **(update.get("trace") or {}),
    }
    trace["node_timings"] = [
        *((state.get("trace") or {}).get("node_timings") or []),
        timing,
    ]
    return trace
