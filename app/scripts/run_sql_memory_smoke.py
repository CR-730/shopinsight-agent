"""Smoke test successful SQL memory persistence and recall with real services."""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.agent.memory import format_tool_memory_results
from app.clients.embedding_client_manager import embedding_client_manager
from app.clients.es_client_manager import es_client_manager
from app.clients.mysql_client_manager import (
    dw_mysql_client_manager,
    meta_mysql_client_manager,
)
from app.clients.qdrant_client_manager import qdrant_client_manager
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
from app.repositories.mysql.meta.agent_memory_repository import AgentMemoryRepository
from app.repositories.mysql.meta.meta_mysql_repository import MetaMySQLRepository
from app.repositories.qdrant.agent_memory_qdrant_repository import (
    AgentMemoryQdrantRepository,
)
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository
from app.repositories.qdrant.value_qdrant_repository import ValueQdrantRepository
from app.services.query_service import QueryService


@dataclass
class SmokeResult:
    user_id: str
    conversation_id: str
    metadata_cache_version: str
    seed_query: str
    recall_query: str
    seed_sql: str
    recall_sql: str
    direct_memory_count: int
    direct_memory_context: str
    trace_sql_memory_examples: list[dict[str, Any]]


def _parse_sse(chunk: str) -> dict[str, Any] | None:
    payload = "".join(
        line[5:].strip() for line in chunk.splitlines() if line.startswith("data:")
    )
    return json.loads(payload) if payload else None


async def _collect_query(
    service: QueryService,
    query: str,
    *,
    user_id: str,
    conversation_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async for chunk in service.query(
        query,
        conversation_id=conversation_id,
        user_id=user_id,
        include_trace=True,
    ):
        event = _parse_sse(chunk)
        if event:
            events.append(event)

    conversation_event = next(
        event for event in events if event.get("type") == "conversation"
    )
    trace = next((event["data"] for event in events if event.get("type") == "trace"), {})
    return str(conversation_event["data"]["conversation_id"]), trace


def _assert_success_trace(trace: dict[str, Any], label: str) -> None:
    if trace.get("failure"):
        raise AssertionError(f"{label} failed before SQL execution: {trace}")
    output = trace.get("output") or {}
    if not trace.get("sql") or not output.get("rows"):
        raise AssertionError(f"{label} did not produce successful SQL result: {trace}")


async def run_smoke(seed_query: str, recall_query: str) -> SmokeResult:
    qdrant_client_manager.init()
    embedding_client_manager.init()
    es_client_manager.init()
    meta_mysql_client_manager.init()
    dw_mysql_client_manager.init()

    try:
        async with (
            meta_mysql_client_manager.session_factory() as meta_session,
            dw_mysql_client_manager.session_factory() as dw_session,
        ):
            meta_repository = MetaMySQLRepository(meta_session)
            agent_memory_store = AgentMemoryQdrantRepository(
                qdrant_client_manager.client,
                embedding_client_manager.client,
            )
            agent_memory_repository = AgentMemoryRepository(
                meta_session,
                agent_memory_store,
            )
            service = QueryService(
                meta_mysql_repository=meta_repository,
                agent_memory_repository=agent_memory_repository,
                embedding_client=embedding_client_manager.client,
                dw_mysql_repository=DWMySQLRepository(dw_session),
                column_qdrant_repository=ColumnQdrantRepository(
                    qdrant_client_manager.client
                ),
                metric_qdrant_repository=MetricQdrantRepository(
                    qdrant_client_manager.client
                ),
                value_es_repository=ValueESRepository(es_client_manager.client),
                value_qdrant_repository=ValueQdrantRepository(qdrant_client_manager.client),
            )
            user_id = f"sql-memory-smoke-{uuid.uuid4().hex[:8]}"
            metadata_cache_version = await meta_repository.get_metadata_cache_version()

            conversation_id, seed_trace = await _collect_query(
                service, seed_query, user_id=user_id
            )
            _assert_success_trace(seed_trace, "seed query")

            direct_results = await agent_memory_repository.search_similar_usage(
                seed_query,
                user_id=user_id,
                metadata_cache_version=metadata_cache_version,
                tool_name_filter="run_sql",
                similarity_threshold=0.1,
                limit=5,
            )
            if not direct_results:
                raise AssertionError("successful SQL query did not persist tool memory")

            _, recall_trace = await _collect_query(
                service,
                recall_query,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            _assert_success_trace(recall_trace, "recall query")
            sql_memory_examples = list(recall_trace.get("sql_memory_examples") or [])
            if not sql_memory_examples:
                raise AssertionError("recall query did not receive sql_memory_examples")

            return SmokeResult(
                user_id=user_id,
                conversation_id=conversation_id,
                metadata_cache_version=metadata_cache_version,
                seed_query=seed_query,
                recall_query=recall_query,
                seed_sql=str(seed_trace.get("sql") or ""),
                recall_sql=str(recall_trace.get("sql") or ""),
                direct_memory_count=len(direct_results),
                direct_memory_context=format_tool_memory_results(direct_results),
                trace_sql_memory_examples=sql_memory_examples,
            )
    finally:
        if qdrant_client_manager.client is not None:
            await qdrant_client_manager.close()
        if es_client_manager.client is not None:
            await es_client_manager.close()


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-query", default="统计华北地区的销售额")
    parser.add_argument("--recall-query", default="统计华北地区的销售额")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = await run_smoke(args.seed_query, args.recall_query)
    payload = asdict(result)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
