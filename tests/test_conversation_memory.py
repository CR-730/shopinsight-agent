from app.agent.memory import (
    build_answer_summary,
    build_snapshot_from_state,
)
from app.api.schemas.query_schema import QuerySchema
from app.repositories.mysql.meta.conversation_memory_repository import (
    ConversationMemoryRepository,
)
from app.services.query_service import QueryService


def test_build_answer_summary_counts_rows_and_columns():
    summary = build_answer_summary(
        [
            {"region": "华北", "gmv": 1200},
            {"region": "华东", "gmv": 1800},
        ]
    )

    assert summary == "返回 2 行，字段：region, gmv"


def test_build_snapshot_from_state_uses_successful_state():
    snapshot = build_snapshot_from_state(
        {
            "metric_bindings": [{"canonical_metric": "GMV"}],
            "resolved_filters": [{"canonical_value": "华北"}],
            "time_binding": {"raw_text": "2025 年第一季度"},
            "sql": "select 1",
            "final_answer": [{"gmv": 100}],
        }
    )

    assert snapshot == {
        "last_metric_bindings": [{"canonical_metric": "GMV"}],
        "last_resolved_filters": [{"canonical_value": "华北"}],
        "last_time_binding": {"raw_text": "2025 年第一季度"},
        "last_sql": "select 1",
        "last_answer_summary": "返回 1 行，字段：gmv",
        "recent_turns_summary": [],
    }


def test_build_snapshot_from_state_ignores_blocked_state():
    snapshot = build_snapshot_from_state(
        {
            "blocked_by": "semantic_guard",
            "safety_error": "业务绑定未解析",
            "metric_bindings": [{"canonical_metric": "GMV"}],
            "final_answer": [{"gmv": 100}],
        }
    )

    assert snapshot is None


def test_build_snapshot_from_state_ignores_execution_error_state():
    snapshot = build_snapshot_from_state(
        {
            "sql": "select 1",
            "error": "SQL 执行超时",
            "exception_stage": "tool_execution",
            "final_answer": None,
        }
    )

    assert snapshot is None


def test_build_snapshot_from_state_requires_structured_metric_binding():
    snapshot = build_snapshot_from_state(
        {
            "metric_bindings": [],
            "resolved_filters": [],
            "time_binding": None,
            "sql": "select sum(quantity) from fact_order",
            "final_answer": [{"销量": 100}],
        }
    )

    assert snapshot is None


class FakeResult:
    def __init__(self, scalar_value=None, row=None):
        self.scalar_value = scalar_value
        self.row = row

    def scalar(self):
        return self.scalar_value

    def mappings(self):
        return self

    def fetchone(self):
        return self.row


class FakeSession:
    def __init__(self, results=None, conversation_rows=None):
        self.statements = []
        self.params = []
        self.commits = 0
        self.results = list(results or [])
        self.conversation_rows = (
            None if conversation_rows is None else list(conversation_rows)
        )

    async def execute(self, statement, params=None):
        sql = str(statement)
        self.statements.append(sql)
        self.params.append(params or {})
        if (
            self.conversation_rows is not None
            and "select id, user_id, title" in sql.lower()
        ):
            return FakeResult(row=self.conversation_rows.pop(0))
        if self.results:
            return self.results.pop(0)
        return FakeResult()

    async def commit(self):
        self.commits += 1


def test_conversation_repository_ensure_tables_creates_three_tables():
    import asyncio

    session = FakeSession()
    repository = ConversationMemoryRepository(session)

    asyncio.run(repository.ensure_tables())

    joined_sql = "\n".join(session.statements)
    assert "create table if not exists conversation" in joined_sql
    assert "create table if not exists conversation_turn" in joined_sql
    assert "create table if not exists conversation_snapshot" in joined_sql


def test_conversation_repository_create_conversation_generates_id_and_title():
    import asyncio

    session = FakeSession()
    repository = ConversationMemoryRepository(session)

    conversation_id = asyncio.run(
        repository.create_conversation(user_id="u1", first_query="统计华北 GMV")
    )

    assert conversation_id
    assert session.params[-1]["id"] == conversation_id
    assert session.params[-1]["user_id"] == "u1"
    assert session.params[-1]["title"] == "统计华北 GMV"
    assert session.commits == 1


def test_conversation_repository_get_snapshot_is_scoped_by_user_id():
    import asyncio

    session = FakeSession(
        results=[
            FakeResult(),
            FakeResult(),
            FakeResult(),
            FakeResult(
                row={
                    "last_metric_bindings": '[{"canonical_metric": "GMV"}]',
                    "last_resolved_filters": "[]",
                    "last_time_binding": None,
                    "last_sql": "select 1",
                    "last_answer_summary": "返回 1 行",
                    "recent_turns_summary": "[]",
                }
            ),
        ]
    )
    repository = ConversationMemoryRepository(session)

    snapshot = asyncio.run(repository.get_snapshot("conv-1", user_id="u1"))

    assert snapshot["last_metric_bindings"] == [{"canonical_metric": "GMV"}]
    select_sql = session.statements[-1].lower()
    assert "from conversation_snapshot" in select_sql
    assert "join conversation" in select_sql
    assert "conversation.user_id = :user_id" in select_sql
    assert session.params[-1] == {"conversation_id": "conv-1", "user_id": "u1"}


def test_conversation_repository_access_check_rejects_missing_or_wrong_user():
    import asyncio

    session = FakeSession(conversation_rows=[None])
    repository = ConversationMemoryRepository(session)

    assert asyncio.run(repository.get_conversation("conv-unknown", user_id="u1")) is None

    select_sql = session.statements[-1].lower()
    assert "from conversation" in select_sql
    assert "conversation.user_id = :user_id" in select_sql


def test_conversation_repository_writes_are_scoped_by_user_id():
    import asyncio

    session = FakeSession(
        results=[FakeResult(scalar_value=1)],
        conversation_rows=[
            {"id": "conv-1", "user_id": "u1", "title": "会话"},
            {"id": "conv-1", "user_id": "u1", "title": "会话"},
        ],
    )
    repository = ConversationMemoryRepository(session)

    asyncio.run(
        repository.save_turn(
            conversation_id="conv-1",
            user_id="u1",
            user_query="那上个月呢",
            rewritten_query="统计华北地区 GMV",
            final_state={"sql": "select 1", "final_answer": [{"gmv": 100}]},
            final_answer_summary="返回 1 行，字段：gmv",
        )
    )
    asyncio.run(
        repository.upsert_snapshot(
            conversation_id="conv-1",
            user_id="u1",
            snapshot={
                "last_metric_bindings": [{"canonical_metric": "GMV"}],
                "last_resolved_filters": [],
                "last_time_binding": None,
                "last_sql": "select 1",
                "last_answer_summary": "返回 1 行，字段：gmv",
                "recent_turns_summary": [],
            },
        )
    )

    assert any(params.get("user_id") == "u1" for params in session.params)


def test_query_schema_accepts_optional_conversation_fields():
    schema = QuerySchema(
        query="那上个月呢",
        conversation_id="conv-1",
        user_id="u1",
    )

    assert schema.query == "那上个月呢"
    assert schema.conversation_id == "conv-1"
    assert schema.user_id == "u1"


def test_query_service_emits_conversation_event_and_persists_memory(monkeypatch):
    import asyncio
    import json

    class FakeMetaRepository:
        async def get_active_build_version(self):
            return "v1"

        async def get_metadata_cache_version(self):
            return "cache-v1"

    class FakeMemoryRepository:
        def __init__(self):
            self.saved_turns = []
            self.snapshots = []

        async def create_conversation(self, user_id, first_query):
            return "conv-new"

        async def get_snapshot(self, conversation_id, user_id):
            assert conversation_id == "conv-new"
            assert user_id == "u1"
            return {
                "last_metric_bindings": [{"canonical_metric": "GMV"}],
                "last_resolved_filters": [{"canonical_value": "华北"}],
            }

        async def save_turn(
            self,
            conversation_id,
            user_id,
            user_query,
            rewritten_query,
            final_state,
            final_answer_summary,
        ):
            self.saved_turns.append(
                {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "user_query": user_query,
                    "rewritten_query": rewritten_query,
                    "final_state": final_state,
                    "final_answer_summary": final_answer_summary,
                }
            )

        async def upsert_snapshot(self, conversation_id, user_id, snapshot):
            self.snapshots.append((conversation_id, user_id, snapshot))

    class FakeGraph:
        def __init__(self):
            self.input = None
            self.stream_mode = None

        async def astream(self, input, context, stream_mode):
            self.input = input
            self.stream_mode = stream_mode
            yield ("values", dict(input))
            yield ("custom", {"type": "progress", "step": "测试", "status": "running"})
            yield (
                "values",
                {
                    **dict(input),
                    "metric_bindings": [{"canonical_metric": "GMV"}],
                    "resolved_filters": [{"canonical_value": "华北"}],
                    "time_binding": None,
                    "sql": "select 1",
                    "final_answer": [{"gmv": 100}],
                },
            )

    fake_graph = FakeGraph()
    monkeypatch.setattr("app.services.query_service.graph", fake_graph)

    async def fake_rewrite_query(query, snapshot, cost_tracker):
        from app.agent.rewrite import ConversationRewriteResult

        return ConversationRewriteResult(
            mode="rewritten",
            standalone_query="统计 2025 年第一季度 华北地区 GMV，那上个月呢",
            reason="基于上一轮快照改写为独立问句",
            inherited_slots={"metric": ["GMV"], "filters": ["华北"]},
            overridden_slots={},
        )

    monkeypatch.setattr("app.services.query_service.rewrite_query", fake_rewrite_query)

    memory_repository = FakeMemoryRepository()
    service = QueryService(
        meta_mysql_repository=FakeMetaRepository(),
        embedding_client=object(),
        dw_mysql_repository=object(),
        column_qdrant_repository=object(),
        metric_qdrant_repository=object(),
        value_es_repository=object(),
        value_qdrant_repository=object(),
        conversation_memory_repository=memory_repository,
    )

    async def collect():
        return [
            item
            async for item in service.query(
                query="那上个月呢",
                conversation_id=None,
                user_id="u1",
            )
        ]

    events = [
        json.loads(item.removeprefix("data: ").strip())
        for item in asyncio.run(collect())
    ]

    assert events[0]["type"] == "conversation"
    assert events[0]["conversation_id"] == "conv-new"
    assert events[0]["rewrite"]["mode"] == "rewritten"
    assert events[0]["rewrite"]["inherited_slots"]["metric"] == ["GMV"]
    assert events[0]["rewritten_query"] == "统计 2025 年第一季度 华北地区 GMV，那上个月呢"
    assert fake_graph.input["query"] == events[0]["rewritten_query"]
    assert fake_graph.stream_mode == ["custom", "values"]
    assert memory_repository.saved_turns[0]["user_query"] == "那上个月呢"
    assert memory_repository.saved_turns[0]["user_id"] == "u1"
    assert memory_repository.saved_turns[0]["final_answer_summary"] == (
        "返回 1 行，字段：gmv"
    )
    assert memory_repository.snapshots[0][0] == "conv-new"


def test_query_service_creates_new_conversation_when_supplied_id_is_inaccessible(
    monkeypatch,
):
    import asyncio
    import json

    class FakeMetaRepository:
        async def get_active_build_version(self):
            return "v1"

        async def get_metadata_cache_version(self):
            return "cache-v1"

    class FakeMemoryRepository:
        def __init__(self):
            self.created = []
            self.access_checks = []

        async def create_conversation(self, user_id, first_query):
            self.created.append((user_id, first_query))
            return "conv-new"

        async def get_conversation(self, conversation_id, user_id):
            self.access_checks.append((conversation_id, user_id))
            return None

        async def get_snapshot(self, conversation_id, user_id):
            return None

        async def save_turn(
            self,
            conversation_id,
            user_id,
            user_query,
            rewritten_query,
            final_state,
            final_answer_summary,
        ):
            pass

        async def upsert_snapshot(self, conversation_id, user_id, snapshot):
            pass

    class FakeGraph:
        async def astream(self, input, context, stream_mode):
            yield ("values", {**dict(input), "final_answer": [{"ok": 1}]})

    monkeypatch.setattr("app.services.query_service.graph", FakeGraph())

    async def fake_rewrite_query(query, snapshot, cost_tracker):
        from app.agent.rewrite import ConversationRewriteResult

        return ConversationRewriteResult(
            mode="unchanged",
            standalone_query=query,
            reason="完整新问题",
            inherited_slots={},
            overridden_slots={},
        )

    monkeypatch.setattr("app.services.query_service.rewrite_query", fake_rewrite_query)
    memory_repository = FakeMemoryRepository()
    service = QueryService(
        meta_mysql_repository=FakeMetaRepository(),
        embedding_client=object(),
        dw_mysql_repository=object(),
        column_qdrant_repository=object(),
        metric_qdrant_repository=object(),
        value_es_repository=object(),
        value_qdrant_repository=object(),
        conversation_memory_repository=memory_repository,
    )

    async def collect():
        return [
            item
            async for item in service.query(
                query="继续看",
                conversation_id="foreign-conv",
                user_id="u1",
            )
        ]

    events = [
        json.loads(item.removeprefix("data: ").strip())
        for item in asyncio.run(collect())
    ]

    assert memory_repository.access_checks == [("foreign-conv", "u1")]
    assert memory_repository.created == [("u1", "继续看")]
    assert events[0]["conversation_id"] == "conv-new"


def test_query_service_blocks_when_rewrite_needs_context(monkeypatch):
    import asyncio
    import json

    class FakeMetaRepository:
        async def get_active_build_version(self):
            return "v1"

        async def get_metadata_cache_version(self):
            return "cache-v1"

    class FakeMemoryRepository:
        def __init__(self):
            self.saved_turns = []
            self.snapshots = []

        async def create_conversation(self, user_id, first_query):
            return "conv-new"

        async def get_snapshot(self, conversation_id, user_id):
            return None

        async def save_turn(
            self,
            conversation_id,
            user_id,
            user_query,
            rewritten_query,
            final_state,
            final_answer_summary,
        ):
            self.saved_turns.append(final_state)

        async def upsert_snapshot(self, conversation_id, user_id, snapshot):
            self.snapshots.append(snapshot)

    class FakeGraph:
        called = False

        async def astream(self, input, context, stream_mode):
            self.called = True
            yield ("values", {**dict(input), "final_answer": [{"unexpected": 1}]})

    fake_graph = FakeGraph()
    monkeypatch.setattr("app.services.query_service.graph", fake_graph)

    async def fake_rewrite_query(query, snapshot, cost_tracker):
        from app.agent.rewrite import ConversationRewriteResult

        assert snapshot is None
        return ConversationRewriteResult(
            mode="needs_context",
            standalone_query=query,
            reason="缺少上一轮会话上下文，无法解析追问",
            inherited_slots={},
            overridden_slots={},
        )

    monkeypatch.setattr("app.services.query_service.rewrite_query", fake_rewrite_query)

    memory_repository = FakeMemoryRepository()
    service = QueryService(
        meta_mysql_repository=FakeMetaRepository(),
        embedding_client=object(),
        dw_mysql_repository=object(),
        column_qdrant_repository=object(),
        metric_qdrant_repository=object(),
        value_es_repository=object(),
        value_qdrant_repository=object(),
        conversation_memory_repository=memory_repository,
    )

    async def collect():
        return [
            item
            async for item in service.query(
                query="那华东呢",
                conversation_id=None,
                user_id="u1",
                include_trace=True,
            )
        ]

    events = [
        json.loads(item.removeprefix("data: ").strip())
        for item in asyncio.run(collect())
    ]

    trace = next(event for event in events if event["type"] == "trace")["data"]
    assert events[0]["rewrite"]["mode"] == "needs_context"
    assert fake_graph.called is False
    assert trace["blocked_by"] == "semantic_guard"
    assert trace["final_answer"] is None
    assert memory_repository.saved_turns[0]["blocked_by"] == "semantic_guard"
    assert memory_repository.snapshots == []
