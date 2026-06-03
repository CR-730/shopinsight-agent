"""MySQL-backed Vanna-style conversation store and agent memory."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.memory import (
    Conversation,
    Message,
    TextMemory,
    TextMemorySearchResult,
    ToolMemorySearchResult,
)
from app.repositories.qdrant.agent_memory_qdrant_repository import (
    AgentMemoryQdrantRepository,
)


class AgentMemoryRepository:
    def __init__(
        self, session: AsyncSession, agent_memory_store: AgentMemoryQdrantRepository
    ):
        self.session = session
        self.agent_memory_store = agent_memory_store
        self._tables_ready = False

    async def create_conversation(
        self, conversation_id: str, user_id: str, initial_message: str = ""
    ) -> Conversation:
        await self._ensure_tables()
        conversation = Conversation(id=conversation_id, user_id=user_id)
        if initial_message:
            conversation.add_message(Message(role="user", content=initial_message))
        await self.update_conversation(conversation)
        return conversation

    async def get_conversation(
        self, conversation_id: str, user_id: str
    ) -> Conversation | None:
        await self._ensure_tables()
        result = await self.session.execute(
            text(
                """
                select id, user_id, metadata_json, created_at, updated_at
                from conversation
                where id = :conversation_id and user_id = :user_id
                """
            ),
            {"conversation_id": conversation_id, "user_id": user_id},
        )
        row = result.mappings().first()
        if not row:
            return None

        messages = await self._list_messages(conversation_id)
        return Conversation(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            messages=messages,
            created_at=_as_datetime(row["created_at"]),
            updated_at=_as_datetime(row["updated_at"]),
            metadata=_loads_dict(row["metadata_json"]),
        )

    async def update_conversation(self, conversation: Conversation) -> None:
        await self._ensure_tables()
        owner = await self._conversation_owner(conversation.id)
        if owner and owner != conversation.user_id:
            raise ValueError("conversation_id belongs to another user")
        await self.session.execute(
            text(
                """
                insert into conversation(
                    id, user_id, title, metadata_json, created_at, updated_at
                )
                values (
                    :id, :user_id, :title, :metadata_json, :created_at, :updated_at
                )
                on duplicate key update
                    title = :title,
                    metadata_json = :metadata_json,
                    updated_at = :updated_at
                """
            ),
            {
                "id": conversation.id,
                "user_id": conversation.user_id,
                "title": _conversation_title(conversation),
                "metadata_json": json.dumps(conversation.metadata, ensure_ascii=False),
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
            },
        )
        existing_count = await self._message_count(conversation.id)
        for message in conversation.messages[existing_count:]:
            await self.session.execute(
                text(
                    """
                    insert into conversation_message(
                        conversation_id, role, content, metadata_json, created_at
                    )
                    values (
                        :conversation_id, :role, :content, :metadata_json, :created_at
                    )
                    """
                ),
                {
                    "conversation_id": conversation.id,
                    "role": message.role,
                    "content": message.content,
                    "metadata_json": json.dumps(
                        message.metadata, ensure_ascii=False, default=str
                    ),
                    "created_at": message.timestamp,
                },
            )
        await self.session.commit()

    async def list_conversations(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> list[Conversation]:
        await self._ensure_tables()
        result = await self.session.execute(
            text(
                """
                select id from conversation
                where user_id = :user_id
                order by updated_at desc
                limit :limit offset :offset
                """
            ),
            {"user_id": user_id, "limit": limit, "offset": offset},
        )
        conversations = []
        for row in result.mappings().fetchall():
            conversation = await self.get_conversation(str(row["id"]), user_id)
            if conversation:
                conversations.append(conversation)
        return conversations

    async def save_tool_usage(
        self,
        question: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        user_id: str,
        metadata_cache_version: str | None = None,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.agent_memory_store.save_tool_usage(
            question=question,
            tool_name=tool_name,
            args=args,
            user_id=user_id,
            metadata_cache_version=metadata_cache_version,
            success=success,
            metadata=metadata,
        )

    async def save_text_memory(self, content: str, *, user_id: str) -> TextMemory:
        return await self.agent_memory_store.save_text_memory(content, user_id=user_id)

    async def search_text_memories(
        self,
        query: str,
        *,
        user_id: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> list[TextMemorySearchResult]:
        return await self.agent_memory_store.search_text_memories(
            query,
            user_id=user_id,
            limit=limit,
            similarity_threshold=similarity_threshold,
        )

    async def search_similar_usage(
        self,
        question: str,
        *,
        user_id: str,
        metadata_cache_version: str | None = None,
        allow_cross_metadata_version: bool = False,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        tool_name_filter: str | None = None,
    ) -> list[ToolMemorySearchResult]:
        return await self.agent_memory_store.search_similar_usage(
            question,
            user_id=user_id,
            metadata_cache_version=metadata_cache_version,
            allow_cross_metadata_version=allow_cross_metadata_version,
            limit=limit,
            similarity_threshold=similarity_threshold,
            tool_name_filter=tool_name_filter,
        )

    async def _list_messages(self, conversation_id: str) -> list[Message]:
        result = await self.session.execute(
            text(
                """
                select role, content, metadata_json, created_at
                from conversation_message
                where conversation_id = :conversation_id
                order by id
                """
            ),
            {"conversation_id": conversation_id},
        )
        return [
            Message(
                role=str(row["role"]),
                content=str(row["content"]),
                timestamp=_as_datetime(row["created_at"]),
                metadata=_loads_dict(row["metadata_json"]),
            )
            for row in result.mappings().fetchall()
        ]

    async def _message_count(self, conversation_id: str) -> int:
        result = await self.session.execute(
            text(
                "select count(*) from conversation_message where conversation_id = :id"
            ),
            {"id": conversation_id},
        )
        return int(result.scalar() or 0)

    async def _conversation_owner(self, conversation_id: str) -> str | None:
        result = await self.session.execute(
            text("select user_id from conversation where id = :id"),
            {"id": conversation_id},
        )
        value = result.scalar()
        return str(value) if value else None

    async def _ensure_tables(self) -> None:
        if self._tables_ready:
            return
        await self.session.execute(
            text(
                """
                create table if not exists conversation
                (
                    id varchar(64) primary key,
                    user_id varchar(128) not null,
                    title varchar(255) not null default '',
                    metadata_json json null,
                    created_at timestamp default current_timestamp,
                    updated_at timestamp default current_timestamp,
                    index idx_conversation_user_updated(user_id, updated_at)
                )
                """
            )
        )
        await self.session.execute(
            text(
                """
                create table if not exists conversation_message
                (
                    id bigint primary key auto_increment,
                    conversation_id varchar(64) not null,
                    role varchar(32) not null,
                    content text not null,
                    metadata_json json null,
                    created_at timestamp default current_timestamp,
                    index idx_conversation_message_conversation_id(conversation_id, id),
                    constraint fk_conversation_message_conversation
                        foreign key(conversation_id) references conversation(id)
                        on delete cascade
                )
                """
            )
        )
        await self._ensure_column("conversation", "metadata_json", "json null")
        await self._ensure_column(
            "conversation", "title", "varchar(255) not null default ''"
        )
        await self._ensure_column("conversation_message", "metadata_json", "json null")
        self._tables_ready = True

    async def _ensure_column(
        self, table_name: str, column_name: str, column_definition: str
    ) -> None:
        result = await self.session.execute(
            text(
                """
                select 1
                from information_schema.columns
                where table_schema = database()
                  and table_name = :table_name
                  and column_name = :column_name
                """
            ),
            {"table_name": table_name, "column_name": column_name},
        )
        if result.scalar():
            return
        await self.session.execute(
            text(f"alter table {table_name} add column {column_name} {column_definition}")
        )


def _loads_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    return json.loads(str(value))


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    return datetime.fromisoformat(str(value)).replace(tzinfo=UTC)


def _conversation_title(conversation: Conversation) -> str:
    for message in conversation.messages:
        if message.role == "user" and message.content.strip():
            return message.content.strip()[:255]
    return conversation.id
