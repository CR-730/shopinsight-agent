"""字段取值向量仓储。"""

from dataclasses import fields

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.conf.app_config import app_config
from app.entities.value_info import ValueInfo

_VALUE_INFO_FIELDS = {field.name for field in fields(ValueInfo)}


class ValueQdrantRepository:
    """负责字段真实取值向量集合的创建、写入和检索。"""

    collection_name = "value_info_collection"

    def __init__(self, client: AsyncQdrantClient):
        self.client = client

    async def ensure_collection(self):
        if not await self.client.collection_exists(self.collection_name):
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=app_config.qdrant.embedding_size,
                    distance=Distance.COSINE,
                ),
            )

    async def recreate_collection(self):
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
        points = [
            PointStruct(id=id, vector=embedding, payload=payload)
            for id, embedding, payload in zip(ids, embeddings, payloads)
        ]
        for i in range(0, len(points), batch_size):
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
            )

    async def search(
        self, embedding: list[float], score_threshold: float = 0.5, limit: int = 20
    ) -> list[ValueInfo]:
        result = await self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=limit,
            score_threshold=score_threshold,
        )
        return [
            ValueInfo(
                **{
                    key: value
                    for key, value in point.payload.items()
                    if key in _VALUE_INFO_FIELDS
                }
            )
            for point in result.points
        ]
