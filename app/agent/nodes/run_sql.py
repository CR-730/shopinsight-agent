"""SQL 执行节点。"""

from langgraph.runtime import Runtime

from app.agent.cached_clients import ainvoke_with_timeout
from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.log import logger


async def run_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """执行 SQL 并产出最终问数结果。"""

    writer = runtime.stream_writer
    step = "执行SQL"
    writer({"type": "progress", "step": step, "status": "running"})

    try:
        sql = state["sql"]
        dw_mysql_repository = runtime.context["dw_mysql_repository"]

        try:
            result = await ainvoke_with_timeout(
                dw_mysql_repository.run(sql),
                app_config.agent.sql_execution_timeout_seconds,
            )
        except TimeoutError:
            error = (
                "SQL 执行超时："
                f"{app_config.agent.sql_execution_timeout_seconds} 秒"
            )
            logger.error(error)
            writer({"type": "progress", "step": step, "status": "error"})
            return {
                "error": error,
                "exception_stage": "tool_execution",
                "blocked_by": None,
            }

        logger.info(f"SQL执行结果：{result}")
        writer({"type": "progress", "step": step, "status": "success"})
        writer({"type": "result", "data": result})
        return {"final_answer": result}

    except Exception as e:
        logger.error(f"{step} failed: {e}")
        writer({"type": "progress", "step": step, "status": "error"})
        raise
