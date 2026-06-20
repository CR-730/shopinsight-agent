"""Agent 图中的路由规则。"""

from typing import Mapping

DEFAULT_MAX_SQL_CORRECTION_ATTEMPTS = 2


def route_after_safety_guard(state: Mapping) -> str:
    """Route safe state to next step and blocked state to END."""

    failure = state.get("failure")
    if isinstance(failure, Mapping) and failure.get("disposition") == "blocked":
        return "blocked"
    return "continue"
