"""SQL 多次修正失败后的终止节点。"""

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.core.log import logger


async def fail_sql_correction(
    state: DataAgentState, runtime: Runtime[DataAgentContext]
):
    """输出 SQL 修正失败事件，避免继续执行未通过校验的 SQL。"""

    writer = runtime.stream_writer
    error = state.get("safety_error") or state.get("error") or "SQL 校验失败"
    attempts = state.get("correction_attempts", 0)
    logger.error(f"SQL 修正超过最大次数，停止执行：{error}")
    writer(
        {
            "type": "error",
            "message": f"SQL 修正 {attempts} 次后仍未通过校验：{error}",
        }
    )
