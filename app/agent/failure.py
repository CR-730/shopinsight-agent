"""统一失败状态的构造与读取辅助函数。"""

from typing import Literal

from app.agent.state import FailureState


def build_failure(
    *,
    category: Literal[
        "input_guard",
        "business_binding",
        "sql_validation",
        "sql_execution",
        "system",
    ],
    stage: str,
    code: str,
    message: str,
    disposition: Literal["blocked", "failed"],
    user_message: str = "",
) -> FailureState:
    failure: FailureState = {
        "category": category,
        "stage": stage,
        "code": code,
        "message": message,
        "disposition": disposition,
    }
    if user_message:
        failure["user_message"] = user_message
    return failure


def is_blocked_failure(failure: FailureState | None) -> bool:
    return bool(failure and failure.get("disposition") == "blocked")
