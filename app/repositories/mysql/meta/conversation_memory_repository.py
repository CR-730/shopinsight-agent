"""Conversation memory repository backed by Meta MySQL."""

import json
import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.memory import ConversationSnapshot

_USER_ACCESS_PREDICATE = """
conversation.id = :conversation_id
and (
    conversation.user_id = :user_id
    or (conversation.user_id is null and :user_id is null)
)
"""


class ConversationMemoryRepository:
    """Persist conversation turns and compact snapshots for follow-up rewriting."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def ensure_tables(self):
        """Create conversation memory tables when they do not exist."""

        await self.session.execute(
            text(
                """
                create table if not exists conversation
                (
                    id varchar(64) primary key,
                    user_id varchar(128) null,
                    title varchar(255) not null,
                    created_at timestamp default current_timestamp,
                    updated_at timestamp default current_timestamp
                        on update current_timestamp,
                    archived_at timestamp null
                )
                """
            )
        )
        await self.session.execute(
            text(
                """
                create table if not exists conversation_turn
                (
                    id varchar(64) primary key,
                    conversation_id varchar(64) not null,
                    turn_index int not null,
                    user_query text not null,
                    rewritten_query text not null,
                    sql_text text null,
                    final_answer_summary text null,
                    safety_error text null,
                    blocked_by varchar(64) null,
                    created_at timestamp default current_timestamp,
                    index idx_conversation_turn_conversation
                        (conversation_id, turn_index)
                )
                """
            )
        )
        await self.session.execute(
            text(
                """
                create table if not exists conversation_snapshot
                (
                    conversation_id varchar(64) primary key,
                    last_metric_bindings json null,
                    last_resolved_filters json null,
                    last_time_binding json null,
                    last_sql text null,
                    last_answer_summary text null,
                    recent_turns_summary json null,
                    updated_at timestamp default current_timestamp
                        on update current_timestamp
                )
                """
            )
        )

    async def create_conversation(self, user_id: str | None, first_query: str) -> str:
        """Create a conversation and return its generated id."""

        await self.ensure_tables()
        conversation_id = uuid.uuid4().hex
        title = first_query.strip()[:80] or "新会话"
        await self.session.execute(
            text(
                """
                insert into conversation(id, user_id, title)
                values (:id, :user_id, :title)
                """
            ),
            {"id": conversation_id, "user_id": user_id, "title": title},
        )
        await self.session.commit()
        return conversation_id

    async def get_conversation(
        self, conversation_id: str, user_id: str | None
    ) -> dict[str, Any] | None:
        """Return a conversation only when it belongs to the supplied user scope."""

        await self.ensure_tables()
        result = await self.session.execute(
            text(
                f"""
                select id, user_id, title
                from conversation
                where {_USER_ACCESS_PREDICATE}
                """
            ),
            {"conversation_id": conversation_id, "user_id": user_id},
        )
        row = result.mappings().fetchone()
        return dict(row) if row else None

    async def get_snapshot(
        self, conversation_id: str, user_id: str | None
    ) -> ConversationSnapshot | None:
        """Read the compact snapshot for a conversation."""

        await self.ensure_tables()
        result = await self.session.execute(
            text(
                """
                select
                    last_metric_bindings,
                    last_resolved_filters,
                    last_time_binding,
                    last_sql,
                    last_answer_summary,
                    recent_turns_summary
                from conversation_snapshot
                join conversation
                    on conversation.id = conversation_snapshot.conversation_id
                where conversation.id = :conversation_id
                  and (
                    conversation.user_id = :user_id
                    or (conversation.user_id is null and :user_id is null)
                  )
                """
            ),
            {"conversation_id": conversation_id, "user_id": user_id},
        )
        row = result.mappings().fetchone()
        if not row:
            return None
        return {
            "last_metric_bindings": _json_loads(row.get("last_metric_bindings"), []),
            "last_resolved_filters": _json_loads(row.get("last_resolved_filters"), []),
            "last_time_binding": _json_loads(row.get("last_time_binding"), None),
            "last_sql": row.get("last_sql"),
            "last_answer_summary": row.get("last_answer_summary"),
            "recent_turns_summary": _json_loads(row.get("recent_turns_summary"), []),
        }

    async def save_turn(
        self,
        conversation_id: str,
        user_id: str | None,
        user_query: str,
        rewritten_query: str,
        final_state: dict[str, Any],
        final_answer_summary: str | None,
    ):
        """Append one completed user turn."""

        await self.ensure_tables()
        if not await self.get_conversation(conversation_id, user_id):
            raise ValueError("Conversation does not exist or is not accessible")
        turn_index = await self._next_turn_index(conversation_id)
        await self.session.execute(
            text(
                """
                insert into conversation_turn(
                    id,
                    conversation_id,
                    turn_index,
                    user_query,
                    rewritten_query,
                    sql_text,
                    final_answer_summary,
                    safety_error,
                    blocked_by
                )
                values (
                    :id,
                    :conversation_id,
                    :turn_index,
                    :user_query,
                    :rewritten_query,
                    :sql_text,
                    :final_answer_summary,
                    :safety_error,
                    :blocked_by
                )
                """
            ),
            {
                "id": uuid.uuid4().hex,
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "user_query": user_query,
                "rewritten_query": rewritten_query,
                "sql_text": final_state.get("sql"),
                "final_answer_summary": final_answer_summary,
                "safety_error": final_state.get("safety_error") or final_state.get("error"),
                "blocked_by": final_state.get("blocked_by"),
            },
        )
        await self.session.execute(
            text(
                """
                update conversation
                set updated_at = current_timestamp
                where id = :conversation_id
                """
            ),
            {"conversation_id": conversation_id},
        )
        await self.session.commit()

    async def upsert_snapshot(
        self,
        conversation_id: str,
        user_id: str | None,
        snapshot: ConversationSnapshot,
    ):
        """Insert or update the latest successful conversation snapshot."""

        await self.ensure_tables()
        if not await self.get_conversation(conversation_id, user_id):
            raise ValueError("Conversation does not exist or is not accessible")
        await self.session.execute(
            text(
                """
                insert into conversation_snapshot(
                    conversation_id,
                    last_metric_bindings,
                    last_resolved_filters,
                    last_time_binding,
                    last_sql,
                    last_answer_summary,
                    recent_turns_summary
                )
                values (
                    :conversation_id,
                    :last_metric_bindings,
                    :last_resolved_filters,
                    :last_time_binding,
                    :last_sql,
                    :last_answer_summary,
                    :recent_turns_summary
                )
                on duplicate key update
                    last_metric_bindings = values(last_metric_bindings),
                    last_resolved_filters = values(last_resolved_filters),
                    last_time_binding = values(last_time_binding),
                    last_sql = values(last_sql),
                    last_answer_summary = values(last_answer_summary),
                    recent_turns_summary = values(recent_turns_summary),
                    updated_at = current_timestamp
                """
            ),
            {
                "conversation_id": conversation_id,
                "last_metric_bindings": _json_dumps(
                    snapshot.get("last_metric_bindings") or []
                ),
                "last_resolved_filters": _json_dumps(
                    snapshot.get("last_resolved_filters") or []
                ),
                "last_time_binding": _json_dumps(snapshot.get("last_time_binding")),
                "last_sql": snapshot.get("last_sql"),
                "last_answer_summary": snapshot.get("last_answer_summary"),
                "recent_turns_summary": _json_dumps(
                    snapshot.get("recent_turns_summary") or []
                ),
            },
        )
        await self.session.commit()

    async def _next_turn_index(self, conversation_id: str) -> int:
        result = await self.session.execute(
            text(
                """
                select coalesce(max(turn_index), 0) + 1
                from conversation_turn
                where conversation_id = :conversation_id
                """
            ),
            {"conversation_id": conversation_id},
        )
        return int(result.scalar() or 1)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value
