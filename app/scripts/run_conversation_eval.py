"""Run multi-turn conversation memory evaluations."""

import argparse
import asyncio
import json
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from app.clients.embedding_client_manager import embedding_client_manager
from app.clients.es_client_manager import es_client_manager
from app.clients.mysql_client_manager import (
    dw_mysql_client_manager,
    meta_mysql_client_manager,
)
from app.clients.qdrant_client_manager import qdrant_client_manager
from app.evaluation.conversation_cases import (
    evaluate_conversation_case,
    load_conversation_eval_cases,
)
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
from app.repositories.mysql.meta.conversation_memory_repository import (
    ConversationMemoryRepository,
)
from app.repositories.mysql.meta.meta_mysql_repository import MetaMySQLRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository
from app.repositories.qdrant.value_qdrant_repository import ValueQdrantRepository
from app.services.query_service import QueryService


async def run_conversation_eval(
    cases_path: Path, output_path: Path | None = None
) -> int:
    """Run all conversation cases and optionally write a JSON report."""

    started_at = _now_iso()
    started = time.perf_counter()
    run_id = started_at.replace(":", "-")
    eval_user_id = build_eval_user_id(run_id)
    cases = load_conversation_eval_cases(cases_path)
    real_cases = [case for case in cases if case.get("runner") != "mock"]
    mock_cases = [case for case in cases if case.get("runner") == "mock"]
    results: list[dict[str, Any]] = []

    if real_cases:
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
                memory_repository = ConversationMemoryRepository(meta_session)
                query_service = _build_query_service(
                    meta_session=meta_session,
                    dw_session=dw_session,
                    embedding_client=embedding_client_manager.client,
                    memory_repository=memory_repository,
                )
                for case in real_cases:
                    results.append(
                        await evaluate_conversation_case(
                            case,
                            query_service,
                            memory_repository,
                            default_user_id=eval_user_id,
                        )
                    )
        finally:
            await qdrant_client_manager.close()
            await es_client_manager.close()
            await meta_mysql_client_manager.close()
            await dw_mysql_client_manager.close()

    for case in mock_cases:
        memory_repository = InMemoryConversationMemoryRepository()
        results.append(
            await evaluate_conversation_case(
                case,
                MockConversationQueryService(memory_repository),
                memory_repository,
                default_user_id=eval_user_id,
            )
        )

    finished_at = _now_iso()
    passed = sum(1 for item in results if item["passed"])
    payload = {
        "started_at": started_at,
        "finished_at": finished_at,
        "run_id": run_id,
        "eval_user_id": eval_user_id,
        "cases_path": str(cases_path),
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": round(passed / len(results), 4) if results else 0,
        },
        "latency_seconds": round(time.perf_counter() - started, 3),
        "results": results,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if passed == len(results) else 1


def _build_query_service(
    meta_session: Any,
    dw_session: Any,
    embedding_client: Embeddings,
    memory_repository: ConversationMemoryRepository,
) -> QueryService:
    return QueryService(
        meta_mysql_repository=MetaMySQLRepository(meta_session),
        embedding_client=embedding_client,
        dw_mysql_repository=DWMySQLRepository(dw_session),
        column_qdrant_repository=ColumnQdrantRepository(qdrant_client_manager.client),
        metric_qdrant_repository=MetricQdrantRepository(qdrant_client_manager.client),
        value_es_repository=ValueESRepository(es_client_manager.client),
        value_qdrant_repository=ValueQdrantRepository(qdrant_client_manager.client),
        conversation_memory_repository=memory_repository,
    )


def build_eval_user_id(run_id: str) -> str:
    """Build a traceable user id for conversation eval data cleanup."""

    return f"conversation-eval:{run_id}"


class InMemoryConversationMemoryRepository:
    """Small memory repository for runner: mock cases."""

    def __init__(self):
        self.snapshots: dict[tuple[str, str | None], dict[str, Any]] = {}
        self.next_id = 1

    async def create_conversation(self, user_id: str | None, first_query: str) -> str:
        conversation_id = f"mock-conv-{self.next_id}"
        self.next_id += 1
        return conversation_id

    async def get_conversation(
        self, conversation_id: str, user_id: str | None
    ) -> dict[str, Any] | None:
        return {"id": conversation_id, "user_id": user_id, "title": conversation_id}

    async def get_snapshot(
        self, conversation_id: str, user_id: str | None
    ) -> dict[str, Any] | None:
        return deepcopy(self.snapshots.get((conversation_id, user_id)))

    async def save_turn(
        self,
        conversation_id: str,
        user_id: str | None,
        user_query: str,
        rewritten_query: str,
        final_state: dict[str, Any],
        final_answer_summary: str | None,
    ):
        return None

    async def upsert_snapshot(
        self, conversation_id: str, user_id: str | None, snapshot: dict[str, Any]
    ):
        self.snapshots[(conversation_id, user_id)] = deepcopy(snapshot)


class MockConversationQueryService:
    """Mock QueryService for synthetic failure cases."""

    def __init__(self, memory_repository: InMemoryConversationMemoryRepository):
        self.memory_repository = memory_repository

    async def query(
        self,
        query: str,
        conversation_id: str | None = None,
        user_id: str | None = None,
        include_trace: bool = False,
    ):
        if not conversation_id:
            conversation_id = await self.memory_repository.create_conversation(
                user_id, query
            )
        snapshot = await self.memory_repository.get_snapshot(conversation_id, user_id)
        rewrite_result = _mock_rewrite_result(query, snapshot)
        rewritten_query = rewrite_result["standalone_query"]
        trace = _mock_trace(rewritten_query)
        yield _sse(
            {
                "type": "conversation",
                "conversation_id": conversation_id,
                "rewritten_query": rewritten_query,
                "rewrite": rewrite_result,
            }
        )
        if trace.get("final_answer") is not None and not trace.get("error"):
            await self.memory_repository.upsert_snapshot(
                conversation_id,
                user_id,
                {
                    "last_metric_bindings": trace.get("metric_bindings") or [],
                    "last_resolved_filters": trace.get("resolved_filters") or [],
                    "last_time_binding": trace.get("time_binding"),
                    "last_sql": "select 1",
                    "last_answer_summary": "返回 1 行",
                    "recent_turns_summary": [],
                },
            )
        if include_trace:
            yield _sse({"type": "trace", "data": trace})
        yield _sse({"type": "usage", "data": {"llm_total_tokens": 0}})


def _mock_trace(query: str) -> dict[str, Any]:
    if "华北" in query:
        return {
            "metric_bindings": [{"canonical_metric": "GMV"}],
            "resolved_filters": [{"canonical_value": "华北"}],
            "blocked_by": None,
            "final_answer": [{"gmv": 100}],
        }
    if "华东" in query:
        return {
            "metric_bindings": [{"canonical_metric": "GMV"}],
            "resolved_filters": [{"canonical_value": "华东"}],
            "exception_stage": "tool_execution",
            "error": "SQL 执行超时",
            "final_answer": None,
        }
    return {"blocked_by": "semantic_guard", "final_answer": None}


def _mock_rewrite_result(
    query: str, snapshot: dict[str, Any] | None
) -> dict[str, Any]:
    if snapshot is None and query.startswith(("那", "改成", "换成")):
        return {
            "mode": "needs_context",
            "standalone_query": query,
            "reason": "缺少上一轮会话上下文，无法解析追问",
            "inherited_slots": {},
            "overridden_slots": {},
        }
    if snapshot and "华东" in query:
        return {
            "mode": "rewritten",
            "standalone_query": "统计华东地区 GMV",
            "reason": "继承上一轮指标，覆盖地区",
            "inherited_slots": {"metric": ["GMV"]},
            "overridden_slots": {"filters": ["华东"]},
        }
    return {
        "mode": "unchanged",
        "standalone_query": query,
        "reason": "完整问题或无需改写",
        "inherited_slots": {},
        "overridden_slots": {},
    }


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--cases",
        default="examples/conversation_eval_cases.yaml",
        help="多轮会话评测 YAML 文件路径",
    )
    parser.add_argument("--output", default=None, help="评测报告 JSON 输出路径")
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            run_conversation_eval(
                cases_path=Path(args.cases),
                output_path=Path(args.output) if args.output else None,
            )
        )
    )
