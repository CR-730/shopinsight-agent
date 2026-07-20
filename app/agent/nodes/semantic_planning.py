"""Production semantic-planning node backed by the validated planner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.agent.semantic_planning.orchestrator import build_semantic_plan

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

    from app.agent.context import DataAgentContext
    from app.agent.state import DataAgentState


async def semantic_planning(
    state: DataAgentState, runtime: Runtime[DataAgentContext]
) -> dict[str, Any]:
    """Emit only a validated semantic plan or a terminal failure."""

    writer = runtime.stream_writer
    step = "语义规划"
    writer({"type": "progress", "step": step, "status": "running"})
    update = await build_semantic_plan(state, runtime)
    failure = update.get("failure")
    if failure:
        user_message = str(failure.get("user_message") or "")
        if user_message:
            _write_answer_delta(writer, "\n\n" + user_message)
        writer(
            {
                "type": "progress",
                "step": step,
                "status": failure.get("disposition") or "failed",
                "error": failure.get("code") or "semantic_planning_failed",
            }
        )
        return update
    writer({"type": "progress", "step": step, "status": "success"})
    return update


def _write_answer_delta(writer, text: str) -> None:
    content = str(text or "")
    if not content.strip():
        return
    for index in range(0, len(content), 12):
        writer({"type": "answer_delta", "delta": content[index : index + 12]})


__all__ = ["semantic_planning"]
