"""SQL execution boundary with injected database repository."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from app.agent.cached_clients import ainvoke_with_timeout
from app.agent.sql.sql_guard import (
    normalize_sql_for_execution,
    validate_sql_before_execution,
    validate_sql_structure_semantics,
)
from app.conf.app_config import app_config


class SqlExecutionRequest(BaseModel):
    sql: str = Field(description="SQL query to execute")


@dataclass
class SqlExecutionResult:
    ok: bool
    result: Any = None
    error: str | None = None
    status: str | None = None
    audit: dict[str, Any] = field(default_factory=dict)


class SqlExecutor:
    """Owns SQL pre-validation and read-only execution for the agent."""

    name = "run_sql"
    description = "Execute read-only SQL against the configured DW database"

    def __init__(self, dw_repository, timeout_seconds: float | None = None):
        self.dw_repository = dw_repository
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else app_config.agent.sql_execution_timeout_seconds
        )

    async def pre_validate(
        self, state: dict[str, Any], args: SqlExecutionRequest
    ) -> SqlExecutionResult:
        sql = normalize_sql_for_execution(args.sql)
        audit = self._audit(sql=sql, status="running")

        parse_or_structure_error = validate_sql_structure_semantics(state, sql)
        if parse_or_structure_error:
            audit.update(status="repairable_error", error_type="validation_error")
            return SqlExecutionResult(
                ok=False,
                error=parse_or_structure_error,
                status="repairable_error",
                audit=audit,
            )

        try:
            await self.dw_repository.validate(sql)
        except Exception as exc:
            audit.update(status="repairable_error", error_type="database_validation_error")
            return SqlExecutionResult(
                ok=False,
                error=str(exc),
                status="repairable_error",
                audit=audit,
            )

        safety_error = validate_sql_before_execution(state, sql)
        if safety_error:
            audit.update(status="blocked", error_type="safety_error")
            return SqlExecutionResult(
                ok=False,
                error=safety_error,
                status="blocked",
                audit=audit,
            )

        audit.update(status="pass")
        return SqlExecutionResult(ok=True, result=sql, status="pass", audit=audit)

    async def execute(self, args: SqlExecutionRequest) -> SqlExecutionResult:
        started_at = time.perf_counter()
        audit = self._audit(sql=args.sql, status="running")
        try:
            result = await ainvoke_with_timeout(
                self.dw_repository.run(args.sql),
                self.timeout_seconds,
            )
        except TimeoutError:
            audit.update(
                status="error",
                exception_stage="tool_execution",
                error_type="timeout",
                latency_ms=_latency_ms(started_at),
            )
            return SqlExecutionResult(
                ok=False,
                error=f"SQL 执行超时：{self.timeout_seconds} 秒",
                audit=audit,
            )
        except Exception as exc:
            audit.update(
                status="error",
                exception_stage="tool_execution",
                error_type="runtime_error",
                latency_ms=_latency_ms(started_at),
            )
            return SqlExecutionResult(ok=False, error=str(exc), audit=audit)

        audit.update(
            status="success",
            row_count=len(result) if isinstance(result, list) else None,
            latency_ms=_latency_ms(started_at),
        )
        return SqlExecutionResult(
            ok=True,
            result=result,
            audit=audit,
        )

    def _audit(self, *, sql: str, status: str) -> dict[str, Any]:
        return {"tool_name": self.name, "sql": sql, "status": status}


def _latency_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)
