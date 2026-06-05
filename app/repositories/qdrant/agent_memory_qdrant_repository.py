"""Qdrant-backed Vanna-style AgentMemory."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from langchain_core.embeddings import Embeddings
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.agent.memory import (
    TextMemory,
    TextMemorySearchResult,
    ToolMemory,
    ToolMemorySearchResult,
)
from app.conf.app_config import app_config


class AgentMemoryQdrantRepository:
    collection_name = "agent_memory_collection"

    def __init__(
        self, client: AsyncQdrantClient, embedding_client: Embeddings
    ) -> None:
        self.client = client
        self.embedding_client = embedding_client

    async def ensure_collection(self) -> None:
        if not await self.client.collection_exists(self.collection_name):
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=app_config.qdrant.embedding_size,
                    distance=Distance.COSINE,
                ),
            )

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
        await self.ensure_collection()
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        embedding = await self.embedding_client.aembed_query(question)
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload={
                        "memory_type": "tool",
                        "user_id": user_id,
                        "metadata_cache_version": metadata_cache_version,
                        "question": question,
                        "tool_name": tool_name,
                        "args": args,
                        "timestamp": timestamp,
                        "success": success,
                        "metadata": metadata or {},
                    },
                )
            ],
        )

    async def search_similar_usage(
        self,
        question: str,
        *,
        user_id: str,
        metadata_cache_version: str | None = None,
        # 业务查询路径必须保持 false；只有管理、迁移或诊断工具
        # 需要主动审计旧版本记忆时，才允许跨 metadata version 检索。
        allow_cross_metadata_version: bool = False,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        tool_name_filter: str | None = None,
    ) -> list[ToolMemorySearchResult]:
        await self.ensure_collection()
        embedding = await self.embedding_client.aembed_query(question)
        result = await self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=limit,
            score_threshold=similarity_threshold,
            query_filter=_tool_filter(
                user_id,
                metadata_cache_version,
                tool_name_filter,
                allow_cross_metadata_version=allow_cross_metadata_version,
            ),
        )
        return [
            ToolMemorySearchResult(
                memory=ToolMemory(
                    memory_id=str(point.id),
                    question=str(point.payload["question"]),
                    tool_name=str(point.payload["tool_name"]),
                    args=dict(point.payload["args"]),
                    timestamp=point.payload.get("timestamp"),
                    success=bool(point.payload.get("success", True)),
                    metadata=dict(point.payload.get("metadata") or {}),
                ),
                similarity_score=float(point.score),
                rank=rank,
            )
            for rank, point in enumerate(result.points, start=1)
        ]

    async def save_text_memory(self, content: str, *, user_id: str) -> TextMemory:
        await self.ensure_collection()
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        embedding = await self.embedding_client.aembed_query(content)
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload={
                        "memory_type": "text",
                        "user_id": user_id,
                        "content": content,
                        "timestamp": timestamp,
                    },
                )
            ],
        )
        return TextMemory(memory_id=memory_id, content=content, timestamp=timestamp)

    async def search_text_memories(
        self,
        query: str,
        *,
        user_id: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> list[TextMemorySearchResult]:
        await self.ensure_collection()
        embedding = await self.embedding_client.aembed_query(query)
        result = await self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=limit,
            score_threshold=similarity_threshold,
            query_filter=_text_filter(user_id),
        )
        return [
            TextMemorySearchResult(
                memory=TextMemory(
                    memory_id=str(point.id),
                    content=str(point.payload["content"]),
                    timestamp=point.payload.get("timestamp"),
                ),
                similarity_score=float(point.score),
                rank=rank,
            )
            for rank, point in enumerate(result.points, start=1)
        ]


def _tool_filter(
    user_id: str,
    metadata_cache_version: str | None,
    tool_name: str | None,
    *,
    allow_cross_metadata_version: bool = False,
) -> Filter:
    if metadata_cache_version is None and not allow_cross_metadata_version:
        raise ValueError("metadata_cache_version is required for SQL tool memory search")
    conditions = [
        FieldCondition(key="memory_type", match=MatchValue(value="tool")),
        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        FieldCondition(key="success", match=MatchValue(value=True)),
    ]
    if metadata_cache_version is not None:
        conditions.append(
            FieldCondition(
                key="metadata_cache_version",
                match=MatchValue(value=metadata_cache_version),
            )
        )
    if tool_name:
        conditions.append(
            FieldCondition(key="tool_name", match=MatchValue(value=tool_name))
        )
    return Filter(must=conditions)


def _text_filter(user_id: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(key="memory_type", match=MatchValue(value="text")),
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ]
    )
