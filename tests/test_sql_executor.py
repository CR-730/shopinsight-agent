import asyncio

from app.agent.sql.sql_executor import SqlExecutionRequest, SqlExecutor


class FakeDWRepository:
    def __init__(self):
        self.ran_sql = []
        self.validated_sql = []

    async def validate(self, sql: str):
        self.validated_sql.append(sql)

    async def run(self, sql: str):
        self.ran_sql.append(sql)
        return [{"GMV": 100}]


class SlowDWRepository(FakeDWRepository):
    async def run(self, sql: str):
        await asyncio.sleep(1)
        return [{"sql": sql}]


class FailingDWRepository(FakeDWRepository):
    async def run(self, sql: str):
        raise RuntimeError("database disconnected")


def test_sql_executor_pre_validate_checks_database():
    async def run_case():
        repository = FakeDWRepository()
        executor = SqlExecutor(repository)

        result = await executor.pre_validate(
            {"query": "查询订单"},
            SqlExecutionRequest(sql="SELECT * FROM fact_order;"),
        )

        assert result.ok is True
        assert result.status == "pass"
        assert result.error is None
        assert result.audit["tool_name"] == "run_sql"
        assert result.audit["status"] == "pass"
        assert repository.validated_sql == ["SELECT * FROM fact_order"]

    asyncio.run(run_case())


def test_sql_executor_execute_runs_repository_with_timeout():
    async def run_case():
        repository = FakeDWRepository()
        executor = SqlExecutor(repository, timeout_seconds=5)

        result = await executor.execute(SqlExecutionRequest(sql="select 1 as GMV"))

        assert result.ok is True
        assert result.result == [{"GMV": 100}]
        assert result.audit["tool_name"] == "run_sql"
        assert result.audit["status"] == "success"
        assert result.audit["sql"] == "select 1 as GMV"
        assert result.audit["row_count"] == 1
        assert result.audit["latency_ms"] >= 0
        assert repository.ran_sql == ["select 1 as GMV"]

    asyncio.run(run_case())


def test_sql_executor_execute_returns_timeout_error():
    async def run_case():
        executor = SqlExecutor(SlowDWRepository(), timeout_seconds=0.01)

        result = await executor.execute(SqlExecutionRequest(sql="select 1"))

        assert result.ok is False
        assert result.error == "SQL 执行超时：0.01 秒"
        assert result.audit["tool_name"] == "run_sql"
        assert result.audit["status"] == "error"
        assert result.audit["exception_stage"] == "tool_execution"
        assert result.audit["error_type"] == "timeout"

    asyncio.run(run_case())


def test_sql_executor_execute_returns_regular_exception_as_result():
    async def run_case():
        executor = SqlExecutor(FailingDWRepository(), timeout_seconds=5)

        result = await executor.execute(SqlExecutionRequest(sql="select 1"))

        assert result.ok is False
        assert result.error == "database disconnected"
        assert result.audit["tool_name"] == "run_sql"
        assert result.audit["status"] == "error"
        assert result.audit["exception_stage"] == "tool_execution"
        assert result.audit["error_type"] == "runtime_error"
        assert result.audit["sql"] == "select 1"

    asyncio.run(run_case())
