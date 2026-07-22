"""
字段向量仓储

管理字段向量集合并把已经准备好的 point 批量写入 Qdrant

Service 层负责决定一个字段要拆成哪些 point
Repository 只关心集合存在和向量点如何稳定落库
"""

from dataclasses import fields

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import (
    Distance,
    VectorParams,
)

from app.conf.app_config import app_config
from app.entities.column_info import ColumnInfo
from app.repositories.qdrant.grouped_search import (
    ensure_grouped_payload_indexes,
    query_grouped_points,
)

_COLUMN_INFO_FIELDS = {field.name for field in fields(ColumnInfo)}


class ColumnQdrantRepository:
    """负责字段向量集合的创建 写入和基础检索"""

    collection_name = "column_info_collection"

    def __init__(self, client: AsyncQdrantClient):
        self.client = client

    async def ensure_collection(self):
        """确保字段向量集合存在，并按配置中的维度初始化"""
        if not await self.client.collection_exists(self.collection_name):
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=app_config.qdrant.embedding_size, distance=Distance.COSINE
                ),
            )
        await ensure_grouped_payload_indexes(
            self.client,
            collection_name=self.collection_name,
            group_by="id",
        )

    async def recreate_collection(self):
        """重建字段向量集合，避免删除或改名后的旧 point 残留。"""

        if await self.client.collection_exists(self.collection_name):
            await self.client.delete_collection(self.collection_name)
        await self.ensure_collection()

    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        payloads: list[dict],
        batch_size: int = 10,
    ):
        """分批 upsert 字段向量点，避免一次提交过多 point"""
        points: list[PointStruct] = [
            PointStruct(id=id, vector=embedding, payload=payload)
            for id, embedding, payload in zip(ids, embeddings, payloads)
        ]
        for i in range(0, len(points), batch_size):
            await self.client.upsert(
                collection_name=self.collection_name, points=points[i : i + batch_size]
            )

    async def search(
        self,
        embedding: list[float],
        score_threshold: float = 0.6,
        limit: int = 20,
        meta_build_version: str | None = None,
    ) -> list[ColumnInfo]:
        """按向量相似度检索字段元数据，并还原为 ColumnInfo 实体"""
        result = await query_grouped_points(
            self.client,
            collection_name=self.collection_name,
            embedding=embedding,
            group_by="id",
            group_size=1,
            limit=limit,
            score_threshold=score_threshold,
            meta_build_version=meta_build_version,
        )
        # Qdrant 只保存字段元数据 payload，业务层继续使用 ColumnInfo
        return [
            ColumnInfo(
                **{
                    key: value
                    for key, value in point.payload.items()
                    if key in _COLUMN_INFO_FIELDS
                }
            )
            for group in result.groups
            for point in list(group.hits or [])[:1]
        ]
