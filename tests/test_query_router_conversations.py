import asyncio
from datetime import UTC, datetime

from app.agent.memory import Conversation, Message
from app.api.routers.query_router import (
    _conversation_summary,
    delete_conversation_handler,
)


def test_conversation_summary_preserves_full_title_and_message_metadata():
    long_question = "统计 2025 年第一季度各大区的 GMV，并按 GMV 从高到低排序，同时保留可恢复的查询结果上下文"
    conversation = Conversation(
        id="conv-1",
        user_id="user-1",
        created_at=datetime(2026, 6, 5, tzinfo=UTC),
        updated_at=datetime(2026, 6, 5, tzinfo=UTC),
    )
    conversation.add_message(Message(role="user", content=long_question))
    conversation.add_message(
        Message(
            role="assistant",
            content="查询完成，共返回 1 行结果，字段：GMV。",
            metadata={
                "result": [{"GMV": 100}],
                "result_meta": {"tables": ["fact_order"]},
                "sql": "select 100 as GMV",
            },
        )
    )

    summary = _conversation_summary(conversation)

    assert summary.title == long_question
    assert summary.messages[1].metadata["result"] == [{"GMV": 100}]
    assert summary.messages[1].metadata["result_meta"] == {"tables": ["fact_order"]}
    assert summary.messages[1].metadata["sql"] == "select 100 as GMV"


def test_delete_conversation_handler_delegates_to_service():
    class StubQueryService:
        def __init__(self):
            self.calls = []

        async def delete_conversation(self, conversation_id: str, user_id: str):
            self.calls.append((conversation_id, user_id))
            return True

    service = StubQueryService()

    response = asyncio.run(
        delete_conversation_handler(
            conversation_id="conv-1",
            query_service=service,  # type: ignore[arg-type]
            user_id="user-1",
        )
    )

    assert response == {"ok": True}
    assert service.calls == [("conv-1", "user-1")]
