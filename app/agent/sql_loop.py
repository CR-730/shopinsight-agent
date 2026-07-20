"""Agent 图中的路由规则。"""

from typing import Mapping

DEFAULT_MAX_SQL_CORRECTION_ATTEMPTS = 2


def route_after_safety_guard(state: Mapping) -> str:
    """Route a state without terminal failure to the next graph step."""

    failure = state.get("failure")
    if isinstance(failure, Mapping) and failure.get("disposition") in {
        "blocked",
        "failed",
    }:
        return "blocked"
    return "continue"
