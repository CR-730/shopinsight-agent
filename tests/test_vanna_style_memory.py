import asyncio
from types import SimpleNamespace

from qdrant_client.models import Filter, PointStruct

from app.agent.memory import (
    Conversation,
    InMemoryConversationStore,
    Message,
    ToolMemorySearchResult,
    build_retrieval_query,
    build_sql_tool_memory,
    format_conversation_history,
    format_tool_memory_results,
)
from app.repositories.qdrant.agent_memory_qdrant_repository import (
    AgentMemoryQdrantRepository,
)
from app.services.query_service import QueryService, _assistant_message


class FakeEmbeddingClient:
    async def aembed_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if "华北" in text or "GMV" in text else [0.0, 1.0]


class FakeQdrantClient:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.points: list[PointStruct] = []

    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    async def create_collection(self, collection_name: str, vectors_config) -> None:
        self.collections.add(collection_name)

    async def upsert(self, collection_name: str, points: list[PointStruct]) -> None:
        self.collections.add(collection_name)
        self.points.extend(points)

    async def query_points(
        self,
        collection_name: str,
        query: list[float],
        limit: int,
        score_threshold: float,
        query_filter: Filter | None,
    ):
        hits = []
        for point in self.points:
            if not _matches_filter(point.payload or {}, query_filter):
                continue
            score = sum(left * right for left, right in zip(query, point.vector))
            if score >= score_threshold:
                hits.append(SimpleNamespace(id=point.id, payload=point.payload, score=score))
        hits.sort(key=lambda item: item.score, reverse=True)
        return SimpleNamespace(points=hits[:limit])


def _matches_filter(payload: dict, query_filter: Filter | None) -> bool:
    if query_filter is None:
        return True
    for condition in query_filter.must or []:
        if payload.get(condition.key) != condition.match.value:
            return False
    return True


def test_conversation_store_is_scoped_by_user():
    async def run():
        store = InMemoryConversationStore()

        conversation = await store.create_conversation(
            conversation_id="conv-1",
            user_id="user-a",
            initial_message="统计华北 GMV",
        )
        conversation.add_message(Message(role="assistant", content="华北 GMV 是 100"))
        await store.update_conversation(conversation)

        assert await store.get_conversation("conv-1", "user-b") is None

        loaded = await store.get_conversation("conv-1", "user-a")
        assert loaded is not None
        assert [message.role for message in loaded.messages] == ["user", "assistant"]
        assert loaded.messages[0].content == "统计华北 GMV"

    asyncio.run(run())


def test_agent_memory_uses_vector_search_and_user_scope():
    async def run():
        memory = AgentMemoryQdrantRepository(FakeQdrantClient(), FakeEmbeddingClient())

        await memory.save_tool_usage(
            question="统计华北 GMV",
            tool_name="run_sql",
            args={"sql": "select sum(order_amount) from fact_order"},
            user_id="user-a",
            metadata_cache_version="meta-v1",
        )

        user_b_results = await memory.search_similar_usage(
            question="查询华北 GMV",
            limit=3,
            similarity_threshold=0.1,
            tool_name_filter="run_sql",
            user_id="user-b",
            metadata_cache_version="meta-v1",
        )
        user_a_results = await memory.search_similar_usage(
            question="查询华北 GMV",
            limit=3,
            similarity_threshold=0.1,
            tool_name_filter="run_sql",
            user_id="user-a",
            metadata_cache_version="meta-v1",
        )

        assert user_b_results == []
        assert len(user_a_results) == 1
        assert user_a_results[0].memory.question == "统计华北 GMV"
        assert user_a_results[0].memory.args["sql"].startswith("select")

    asyncio.run(run())


def test_agent_memory_filters_unsuccessful_tool_usage():
    async def run():
        memory = AgentMemoryQdrantRepository(FakeQdrantClient(), FakeEmbeddingClient())

        await memory.save_tool_usage(
            question="统计华北 GMV",
            tool_name="run_sql",
            args={"sql": "select sum(order_amount) from fact_order"},
            success=True,
            user_id="user-a",
            metadata_cache_version="meta-v1",
        )
        await memory.save_tool_usage(
            question="删除订单",
            tool_name="run_sql",
            args={"sql": "delete from fact_order"},
            success=False,
            user_id="user-a",
            metadata_cache_version="meta-v1",
        )

        results = await memory.search_similar_usage(
            question="查询华北 GMV",
            limit=3,
            similarity_threshold=0.1,
            tool_name_filter="run_sql",
            user_id="user-a",
            metadata_cache_version="meta-v1",
        )

        assert len(results) == 1
        assert results[0].memory.question == "统计华北 GMV"

    asyncio.run(run())


def test_agent_memory_filters_metadata_cache_version():
    async def run():
        memory = AgentMemoryQdrantRepository(FakeQdrantClient(), FakeEmbeddingClient())

        await memory.save_tool_usage(
            question="统计华北 GMV",
            tool_name="run_sql",
            args={"sql": "select 1 as old_gmv"},
            user_id="user-a",
            metadata_cache_version="meta-v1",
        )
        await memory.save_tool_usage(
            question="统计华北 GMV",
            tool_name="run_sql",
            args={"sql": "select 2 as new_gmv"},
            user_id="user-a",
            metadata_cache_version="meta-v2",
        )

        results = await memory.search_similar_usage(
            question="查询华北 GMV",
            limit=3,
            similarity_threshold=0.1,
            tool_name_filter="run_sql",
            user_id="user-a",
            metadata_cache_version="meta-v2",
        )

        assert len(results) == 1
        assert results[0].memory.args["sql"] == "select 2 as new_gmv"

    asyncio.run(run())


def test_agent_memory_requires_metadata_cache_version_by_default():
    async def run():
        memory = AgentMemoryQdrantRepository(FakeQdrantClient(), FakeEmbeddingClient())

        try:
            await memory.search_similar_usage(
                question="查询华北 GMV",
                limit=3,
                similarity_threshold=0.1,
                tool_name_filter="run_sql",
                user_id="user-a",
            )
        except ValueError as exc:
            assert "metadata_cache_version is required" in str(exc)
        else:
            raise AssertionError("missing metadata_cache_version should fail")

    asyncio.run(run())


def test_agent_memory_searches_text_memories_with_vector_filter():
    async def run():
        memory = AgentMemoryQdrantRepository(FakeQdrantClient(), FakeEmbeddingClient())

        saved = await memory.save_text_memory("GMV 使用订单金额求和", user_id="user-a")
        results = await memory.search_text_memories(
            "GMV 口径", limit=3, similarity_threshold=0.1, user_id="user-a"
        )

        assert saved.memory_id is not None
        assert len(results) == 1
        assert results[0].memory.content == "GMV 使用订单金额求和"

    asyncio.run(run())


def test_memory_context_formatting_is_compact():
    conversation = Conversation(id="conv-1", user_id="user-a")
    conversation.add_message(Message(role="user", content="统计华北 GMV"))
    conversation.add_message(Message(role="assistant", content="华北 GMV 是 100"))
    tool_results = [
        ToolMemorySearchResult(
            memory=build_sql_tool_memory(
                "统计华北 GMV",
                {
                    "sql": "select sum(order_amount) as GMV from fact_order",
                    "output": {"rows": [{"GMV": 100}]},
                    "business_binding": {"metrics": [{"canonical_metric": "GMV"}]},
                },
            ),
            similarity_score=0.9,
            rank=1,
        )
    ]

    assert format_conversation_history(conversation.messages) == (
        "user: 统计华北 GMV\nassistant: 华北 GMV 是 100"
    )
    assert "统计华北 GMV" in format_tool_memory_results(tool_results)
    assert "select sum(order_amount)" in format_tool_memory_results(tool_results)


def test_build_sql_tool_memory_requires_successful_sql_result():
    assert (
        build_sql_tool_memory(
            "统计 GMV",
            {
                "sql": "select 1",
                "output": {"rows": [{"GMV": 1}]},
                    "business_binding": {"metrics": [{"canonical_metric": "GMV"}]},
            },
        )
        is not None
    )
    assert (
        build_sql_tool_memory(
            "统计 GMV",
            {
                "sql": "select 1",
                "output": {"rows": []},
                    "business_binding": {"metrics": [{"canonical_metric": "GMV"}]},
            },
        )
        is None
    )
    assert (
        build_sql_tool_memory(
            "统计 GMV",
            {
                "sql": "select 1",
                "output": {"rows": [{"GMV": 1}]},
                    "failure": {
                        "category": "input_guard",
                        "stage": "pre_rag_guard",
                        "code": "blocked",
                        "message": "blocked",
                        "disposition": "blocked",
                    },
                    "business_binding": {"metrics": [{"canonical_metric": "GMV"}]},
            },
        )
        is None
    )
    assert (
        build_sql_tool_memory(
            "统计 GMV",
            {
                "sql": "select 1",
                "output": {"rows": [{"GMV": 1}]},
                    "failure": {
                        "category": "sql_validation",
                        "stage": "sql_executor",
                        "code": "blocked",
                        "message": "blocked",
                        "disposition": "blocked",
                    },
                    "business_binding": {"metrics": [{"canonical_metric": "GMV"}]},
            },
        )
        is None
    )

    assert (
        build_sql_tool_memory(
            "统计 GMV",
            {"sql": "select 1", "output": {"rows": [{"GMV": 1}]}},
        )
        is None
    )


def test_sql_tool_memory_does_not_store_result_preview():
    memory = build_sql_tool_memory(
        "统计华北 GMV",
        {
            "sql": "select sum(order_amount) as GMV from fact_order",
            "output": {"rows": [{"GMV": 100, "customer_phone": "13800000000"}]},
            "business_binding": {"metrics": [{"canonical_metric": "GMV"}]},
        },
    )

    assert memory is not None
    assert memory.args == {
        "sql": "select sum(order_amount) as GMV from fact_order",
        "business_binding": {"metrics": [{"canonical_metric": "GMV"}]},
    }


def test_followup_memory_question_includes_conversation_context():
    history = "user: 统计华北 GMV\nassistant: 查询成功，返回 1 行，字段：GMV"

    memory_question = build_retrieval_query("那华东呢", history)

    assert "统计华北 GMV" in memory_question
    assert "那华东呢" in memory_question
    assert "查询成功" not in memory_question


def test_assistant_message_stores_summary_not_result_json():
    content = _assistant_message(
        {
            "sql": "select customer_phone, order_amount from fact_order",
            "output": {
                "rows": [
                    {"customer_phone": "13800000000", "order_amount": 100},
                    {"customer_phone": "13900000000", "order_amount": 200},
                ]
            },
        }
    )

    assert content == "查询完成，共返回 2 行结果，字段：customer_phone, order_amount。"
    assert "13800000000" not in content


def test_assistant_message_uses_generic_error_without_llm_message():
    content = _assistant_message(
        {
            "failure": {
                "category": "business_binding",
                "stage": "business_binding",
                "code": "metric_not_bound",
                "message": "business_binding unresolved: metric=订单, reason=metric_not_bound",
                "disposition": "blocked",
            },
        }
    )

    assert content == "出了点问题，请稍后重试。"
    assert "business_binding" not in content
    assert "metric_not_bound" not in content


def test_assistant_message_prefers_llm_user_facing_message():
    content = _assistant_message(
        {
            "failure": {
                "category": "business_binding",
                "stage": "business_binding",
                "code": "metric_not_bound",
                "message": "business_binding unresolved: metric=订单, reason=metric_not_bound",
                "user_message": "订单数这个指标还没配置；销售额可以先查。你可以直接回复先查销售额。",
                "disposition": "blocked",
            },
        }
    )

    assert content == "订单数这个指标还没配置；销售额可以先查。你可以直接回复先查销售额。"


def test_assistant_message_does_not_generate_rule_copy_from_bound_metrics():
    content = _assistant_message(
        {
            "failure": {
                "category": "business_binding",
                "stage": "business_binding",
                "code": "metric_not_bound",
                "message": "business_binding unresolved: metric=订单, reason=metric_not_bound",
                "disposition": "blocked",
            },
            "business_binding": {
                "metrics": [
                    {
                        "raw_mention": "销售额",
                        "canonical_metric": "GMV",
                        "matched_by": "metric_alias",
                        "evidence": "GMV.alias contains 销售额",
                        "relevant_columns": ["fact_order.order_amount"],
                        "confidence": "high",
                    }
                ]
            },
        }
    )

    assert content == "出了点问题，请稍后重试。"
    assert "business_binding" not in content
    assert "metric_not_bound" not in content


def test_anonymous_user_does_not_write_long_term_sql_memory():
    class FakeAgentMemoryRepository:
        def __init__(self) -> None:
            self.saved_tool_usage = []

        async def update_conversation(self, conversation):
            return None

        async def save_tool_usage(self, **kwargs):
            self.saved_tool_usage.append(kwargs)

    async def run():
        repository = FakeAgentMemoryRepository()
        service = SimpleNamespace(agent_memory_repository=repository)
        conversation = Conversation(id="conv-1", user_id="anonymous")

        await QueryService._save_memory_after_query(
            service,
            conversation=conversation,
            query="统计 GMV",
            memory_query="统计 GMV",
            metadata_cache_version="meta-v1",
            final_state={
                "sql": "select 1 as GMV",
                "output": {"rows": [{"GMV": 1}]},
                "business_binding": {"metrics": [{"canonical_metric": "GMV"}]},
            },
        )

        assert repository.saved_tool_usage == []
        assert [message.role for message in conversation.messages] == [
            "user",
            "assistant",
        ]
        assistant_message = conversation.messages[-1]
        assert assistant_message.metadata["result"] == [{"GMV": 1}]
        assert assistant_message.metadata["sql"] == "select 1 as GMV"

    asyncio.run(run())
