"""Minimal Vanna-style conversation memory eval."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from app.agent.memory import Conversation, Message, ToolMemorySearchResult
from app.agent.retrieval_context import recall_sql_memory_context
from app.services.query_service import QueryService


@dataclass
class FakeAgentMemoryRepository:
    conversations: dict[str, Conversation] = field(default_factory=dict)
    search_calls: list[dict[str, Any]] = field(default_factory=list)
    saved_tool_usage: list[dict[str, Any]] = field(default_factory=list)
    search_results: list[ToolMemorySearchResult] = field(default_factory=list)

    async def create_conversation(
        self, conversation_id: str, user_id: str, initial_message: str = ""
    ) -> Conversation:
        conversation = Conversation(id=conversation_id, user_id=user_id)
        if initial_message:
            conversation.add_message(Message(role="user", content=initial_message))
        self.conversations[conversation_id] = conversation
        return conversation

    async def get_conversation(
        self, conversation_id: str, user_id: str
    ) -> Conversation | None:
        conversation = self.conversations.get(conversation_id)
        if conversation and conversation.user_id == user_id:
            return conversation
        return None

    async def update_conversation(self, conversation: Conversation) -> None:
        self.conversations[conversation.id] = conversation

    async def search_similar_usage(self, question: str, **kwargs):
        self.search_calls.append({"question": question, **kwargs})
        return self.search_results

    async def save_tool_usage(self, **kwargs) -> None:
        self.saved_tool_usage.append(kwargs)


async def run_minimal_conversation_memory_eval() -> dict[str, Any]:
    checks = [
        await _check_conversation_user_isolation(),
        await _check_recall_after_guard_contract(),
        await _check_anonymous_recall_is_empty(),
        await _check_metadata_version_filter(),
        await _check_followup_memory_question_shape(),
        await _check_success_with_binding_writes_memory(),
        await _check_blocked_and_error_do_not_write_memory(),
    ]
    return {
        "passed": sum(1 for check in checks if check["passed"]),
        "total": len(checks),
        "checks": checks,
    }


async def _check_conversation_user_isolation() -> dict[str, Any]:
    repository = FakeAgentMemoryRepository()
    await repository.create_conversation("conv-1", "user-a")

    service = SimpleNamespace(agent_memory_repository=repository)
    loaded = await QueryService._load_or_create_conversation(
        service, "conv-1", "user-b"
    )

    return _result(
        "conversation_id/user_id 隔离",
        loaded.id != "conv-1" and loaded.user_id == "user-b",
    )


async def _check_recall_after_guard_contract() -> dict[str, Any]:
    repository = FakeAgentMemoryRepository()
    runtime = _context(repository, user_id="user-a", metadata_cache_version="meta-v1")

    update = await recall_sql_memory_context(
        {
            "query": "那华东呢",
            "conversation_messages": [{"role": "user", "content": "统计华北 GMV"}],
        },
        runtime,
    )

    return _result(
        "sql_memory_examples 在安全闸门后由节点召回",
        len(repository.search_calls) == 1
        and repository.search_calls[0]["metadata_cache_version"] == "meta-v1"
        and update["sql_memory_examples"] == [],
    )


async def _check_anonymous_recall_is_empty() -> dict[str, Any]:
    repository = FakeAgentMemoryRepository()
    update = await recall_sql_memory_context(
        {"query": "统计 GMV"},
        _context(repository, user_id="anonymous", metadata_cache_version="meta-v1"),
    )

    return _result(
        "匿名用户 sql_memory_examples 为空",
        update["sql_memory_examples"] == [] and repository.search_calls == [],
    )


async def _check_metadata_version_filter() -> dict[str, Any]:
    repository = FakeAgentMemoryRepository()
    await recall_sql_memory_context(
        {"query": "统计 GMV"},
        _context(repository, user_id="user-a", metadata_cache_version="meta-v2"),
    )

    return _result(
        "元数据版本变化不召回旧 SQL memory",
        repository.search_calls[0]["metadata_cache_version"] == "meta-v2",
    )


async def _check_followup_memory_question_shape() -> dict[str, Any]:
    repository = FakeAgentMemoryRepository()
    service = SimpleNamespace(agent_memory_repository=repository)
    conversation = Conversation(id="conv-1", user_id="user-a")

    await QueryService._save_memory_after_query(
        service,
        conversation=conversation,
        query="那华东呢",
        memory_query="user: 统计华北 GMV\nuser: 那华东呢",
        metadata_cache_version="meta-v1",
        final_state=_successful_state(),
    )

    question = repository.saved_tool_usage[0]["question"]
    return _result(
        "追问 memory question 包含用户上下文且不含 assistant 摘要",
        "统计华北 GMV" in question
        and "那华东呢" in question
        and "查询成功" not in question,
    )


async def _check_success_with_binding_writes_memory() -> dict[str, Any]:
    repository = FakeAgentMemoryRepository()
    service = SimpleNamespace(agent_memory_repository=repository)

    await QueryService._save_memory_after_query(
        service,
        conversation=Conversation(id="conv-1", user_id="user-a"),
        query="统计 GMV",
        memory_query="统计 GMV",
        metadata_cache_version="meta-v1",
        final_state=_successful_state(),
    )

    return _result(
        "成功且有 binding 才写长期 SQL memory",
        len(repository.saved_tool_usage) == 1
        and repository.saved_tool_usage[0]["metadata_cache_version"] == "meta-v1",
    )


async def _check_blocked_and_error_do_not_write_memory() -> dict[str, Any]:
    repository = FakeAgentMemoryRepository()
    service = SimpleNamespace(agent_memory_repository=repository)

    for final_state in [
        {
            **_successful_state(),
            "failure": {
                "category": "input_guard",
                "stage": "pre_rag_guard",
                "code": "blocked",
                "message": "blocked",
                "disposition": "blocked",
            },
        },
        {
            **_successful_state(),
            "failure": {
                "category": "sql_execution",
                "stage": "tool_execution",
                "code": "failed",
                "message": "sql failed",
                "disposition": "failed",
            },
        },
        {"sql": "select 1", "output": {"rows": [{"GMV": 1}]}},
    ]:
        await QueryService._save_memory_after_query(
            service,
            conversation=Conversation(id="conv-1", user_id="user-a"),
            query="统计 GMV",
            memory_query="统计 GMV",
            metadata_cache_version="meta-v1",
            final_state=final_state,
        )

    return _result(
        "失败或无 binding 不写长期 SQL memory", repository.saved_tool_usage == []
    )


def _successful_state() -> dict[str, Any]:
    return {
        "sql": "select 1 as GMV",
        "output": {"rows": [{"GMV": 1}]},
        "semantic_plan": {
            "version": "1",
            "metadata_version": "meta-v1",
            "measures": [],
            "dimensions": [],
            "predicates": [],
            "order_by": [],
            "limit": None,
            "joins": [],
            "required_table_ids": [],
            "required_column_ids": [],
            "provenance": [],
        },
    }


def _context(
    repository: FakeAgentMemoryRepository,
    *,
    user_id: str,
    metadata_cache_version: str,
):
    return {
        "agent_memory_repository": repository,
        "user_id": user_id,
        "metadata_cache_version": metadata_cache_version,
    }


def _result(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": passed}
