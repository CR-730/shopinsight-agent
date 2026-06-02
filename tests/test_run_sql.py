import asyncio
from types import SimpleNamespace

from app.agent.nodes import run_sql as run_sql_module


class SlowDWRepository:
    async def run(self, sql: str):
        await asyncio.sleep(1)
        return [{"sql": sql}]


def test_run_sql_timeout_returns_tool_execution_state(monkeypatch):
    async def run_case():
        return await run_sql_module.run_sql({"sql": "select 1"}, runtime)

    events = []
    monkeypatch.setattr(
        run_sql_module.app_config.agent,
        "sql_execution_timeout_seconds",
        0.01,
    )
    runtime = SimpleNamespace(
        stream_writer=events.append,
        context={"dw_mysql_repository": SlowDWRepository()},
    )

    result = asyncio.run(run_case())

    assert result == {
        "error": "SQL 执行超时：0.01 秒",
        "exception_stage": "tool_execution",
        "blocked_by": None,
    }
    assert events[-1] == {"type": "progress", "step": "执行SQL", "status": "error"}
