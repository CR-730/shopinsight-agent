"""Prune metric context using resolved business bindings."""

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState, MetricInfoState
from app.core.log import logger


async def filter_metric(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """Keep bound metrics; otherwise preserve retrieved metric context."""

    writer = runtime.stream_writer
    step = "过滤指标信息"
    writer({"type": "progress", "step": step, "status": "running"})

    metric_infos: list[MetricInfoState] = state["metric_infos"]
    bound_metric_names = {
        binding["canonical_metric"] for binding in state.get("metric_bindings") or []
    }

    if bound_metric_names:
        filtered_metric_infos = [
            metric_info
            for metric_info in metric_infos
            if metric_info["name"] in bound_metric_names
        ]
    else:
        filtered_metric_infos = metric_infos

    logger.info(
        f"过滤后的指标信息：{[item['name'] for item in filtered_metric_infos]}"
    )
    writer({"type": "progress", "step": step, "status": "success"})
    return {"metric_infos": filtered_metric_infos}
