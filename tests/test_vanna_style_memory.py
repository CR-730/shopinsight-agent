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
from app.agent.semantic_planning.memory import semantic_plan_from_memory_args
from app.repositories.qdrant.agent_memory_qdrant_repository import (
    AgentMemoryQdrantRepository,
)
from app.services.query_service import QueryService, _assistant_message


def _semantic_plan():
    return {
        "version": "1",
        "metadata_version": "meta-v2",
        "measures": [
            {
                "metric_id": "GMV",
                "name": "GMV",
                "aggregation": "sum",
                "expression": None,
                "source_column_ids": ["fact_order.order_amount"],
                "output_alias": "GMV",
            }
        ],
        "dimensions": [],
        "predicates": [],
        "order_by": [],
        "limit": None,
        "joins": [],
        "required_table_ids": ["fact_order"],
        "required_column_ids": ["fact_order.order_amount"],
        "required_columns": [
            {"column_id": "fact_order.order_amount", "data_type": "decimal"}
        ],
        "provenance": [
            {
                "raw_text": "销售额",
                "resolved_id": "GMV",
                "method": "metric_alias",
                "evidence": "包含用户原始表达，不应进入长期记忆",
            }
        ],
    }


def _successful_memory_state():
    return {
        "sql": "select sum(order_amount) as GMV from fact_order",
        "output": {"rows": [{"GMV": 100}]},
        "semantic_plan": _semantic_plan(),
    }


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
                hits.append(
                    SimpleNamespace(id=point.id, payload=point.payload, score=score)
                )
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
                _successful_memory_state(),
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
    assert build_sql_tool_memory("统计 GMV", _successful_memory_state()) is not None

    empty = _successful_memory_state()
    empty["output"] = {"rows": []}
    assert build_sql_tool_memory("统计 GMV", empty) is None

    blocked = _successful_memory_state()
    blocked["failure"] = {
        "category": "input_guard",
        "stage": "pre_rag_guard",
        "code": "blocked",
        "message": "blocked",
        "disposition": "blocked",
    }
    assert build_sql_tool_memory("统计 GMV", blocked) is None

    assert (
        build_sql_tool_memory(
            "统计 GMV", {"sql": "select 1", "output": {"rows": [{"GMV": 1}]}}
        )
        is None
    )


def test_success_memory_stores_plan_sql_and_metadata_version():
    memory = build_sql_tool_memory("统计 GMV", _successful_memory_state())

    assert memory is not None
    assert memory.args == {
        "sql": "select sum(order_amount) as GMV from fact_order",
        "semantic_plan": {**_semantic_plan(), "provenance": []},
        "metadata_version": "meta-v2",
    }


def test_memory_never_stores_semantic_draft_or_runtime_trace():
    state = _successful_memory_state()
    state["semantic_draft"] = {"candidate_catalog": {"secret": "candidate"}}
    state["trace"] = {"full_prompt": "secret prompt"}

    memory = build_sql_tool_memory("统计 GMV", state)

    assert memory is not None
    assert "semantic_draft" not in memory.args
    assert "trace" not in memory.args
    assert "candidate_catalog" not in str(memory.args)


def test_failed_empty_or_incomplete_plan_is_not_saved():
    failed = _successful_memory_state()
    failed["failure"] = {
        "category": "semantic_planning",
        "stage": "semantic_planning",
        "code": "ambiguous",
        "message": "ambiguous",
        "disposition": "blocked",
    }
    empty = _successful_memory_state()
    empty["output"] = {"rows": []}
    incomplete = _successful_memory_state()
    incomplete["semantic_plan"] = {"version": "1"}

    assert build_sql_tool_memory("统计 GMV", failed) is None
    assert build_sql_tool_memory("统计 GMV", empty) is None
    assert build_sql_tool_memory("统计 GMV", incomplete) is None


def test_memory_plan_loader_rejects_missing_or_incomplete_plan():
    memory = build_sql_tool_memory("统计 GMV", _successful_memory_state())

    assert memory is not None
    assert semantic_plan_from_memory_args(memory.args) == memory.args["semantic_plan"]
    assert (
        semantic_plan_from_memory_args(
            {"sql": "select sum(order_amount) from fact_order"}
        )
        is None
    )
    assert semantic_plan_from_memory_args({"semantic_plan": {"version": "1"}}) is None


def test_sql_tool_memory_does_not_store_result_preview():
    state = _successful_memory_state()
    state["output"] = {"rows": [{"GMV": 100, "customer_phone": "13800000000"}]}
    memory = build_sql_tool_memory(
        "统计华北 GMV",
        state,
    )

    assert memory is not None
    assert memory.args == {
        "sql": "select sum(order_amount) as GMV from fact_order",
        "semantic_plan": {**_semantic_plan(), "provenance": []},
        "metadata_version": "meta-v2",
    }


def test_sql_tool_memory_keeps_trusted_plan_projections():
    plan = _semantic_plan()
    plan["measures"] = []
    plan["dimensions"] = [
        {
            "column_id": "fact_order.order_id",
            "role": "projection",
            "output_alias": "订单编号",
        }
    ]
    plan["required_column_ids"] = ["fact_order.order_id"]
    plan["provenance"] = []
    memory = build_sql_tool_memory(
        "列出订单编号",
        {
            "sql": "select order_id from fact_order",
            "output": {"rows": [{"order_id": 1}]},
            "semantic_plan": plan,
        },
    )

    assert memory is not None
    assert memory.args["semantic_plan"]["dimensions"] == [
        {
            "column_id": "fact_order.order_id",
            "role": "projection",
            "output_alias": "订单编号",
        }
    ]


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
                "category": "semantic_planning",
                "stage": "semantic_planning",
                "code": "metric_not_bound",
                "message": "semantic planning unresolved: metric_not_bound",
                "disposition": "blocked",
            },
        }
    )

    assert content == "出了点问题，请稍后重试。"
    assert "metric_not_bound" not in content


def test_assistant_message_prefers_llm_user_facing_message():
    content = _assistant_message(
        {
            "failure": {
                "category": "semantic_planning",
                "stage": "semantic_planning",
                "code": "metric_not_bound",
                "message": "semantic planning unresolved: metric_not_bound",
                "user_message": "订单数这个指标还没配置；销售额可以先查。你可以直接回复先查销售额。",
                "disposition": "blocked",
            },
        }
    )

    assert (
        content == "订单数这个指标还没配置；销售额可以先查。你可以直接回复先查销售额。"
    )


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
            metadata_cache_version="meta-v1",
            final_state={
                "sql": "select 1 as GMV",
                "output": {"rows": [{"GMV": 1}]},
                "semantic_plan": _semantic_plan(),
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


def test_sql_memory_stores_rewritten_query_but_conversation_keeps_original_text():
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
        conversation = Conversation(id="conv-1", user_id="user-1")

        await QueryService._save_memory_after_query(
            service,
            conversation=conversation,
            query="改成华南",
            metadata_cache_version="meta-v1",
            final_state={
                "query": "统计华南地区的销售额",
                "sql": "select sum(order_amount) from fact_order",
                "output": {"rows": [{"销售额": 1}]},
                "semantic_plan": _semantic_plan(),
            },
        )

        assert conversation.messages[0].content == "改成华南"
        assert repository.saved_tool_usage[0]["question"] == "统计华南地区的销售额"

    asyncio.run(run())
