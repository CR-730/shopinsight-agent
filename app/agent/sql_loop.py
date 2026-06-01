"""SQL 校验与修正循环的路由规则。"""

from typing import Mapping

DEFAULT_MAX_SQL_CORRECTION_ATTEMPTS = 2


def route_after_sql_validation(state: Mapping) -> str:
    """根据校验结果和修正次数决定下一步节点。"""

    if state.get("error") is None:
        return "pre_execution_guard"

    attempts = int(state.get("correction_attempts") or 0)
    max_attempts = int(
        state.get("max_correction_attempts") or DEFAULT_MAX_SQL_CORRECTION_ATTEMPTS
    )
    if attempts >= max_attempts:
        return "fail_sql_correction"
    return "correct_sql"


def route_after_pre_sql_execution_validation(state: Mapping) -> str:
    """Route combined SQL validation result before database execution."""

    if state.get("safety_error") is not None:
        return "blocked"

    if state.get("error") is None:
        return "pass"

    attempts = int(state.get("correction_attempts") or 0)
    max_attempts = int(
        state.get("max_correction_attempts") or DEFAULT_MAX_SQL_CORRECTION_ATTEMPTS
    )
    if attempts >= max_attempts:
        return "fail_sql_correction"
    return "repairable_error"


def route_after_safety_guard(state: Mapping) -> str:
    """Route safe state to next step and blocked state to END."""

    return "continue" if state.get("safety_error") is None else "blocked"
