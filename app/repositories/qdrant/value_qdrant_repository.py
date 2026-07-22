"""字段取值向量仓储。"""

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.conf.app_config import app_config
from app.entities.value_info import ValueInfo
from app.repositories.qdrant.grouped_search import (
    ensure_grouped_payload_indexes,
    query_grouped_points,
)


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
        await ensure_grouped_payload_indexes(
            self.client,
            collection_name=self.collection_name,
            group_by="candidate_id",
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
        self,
        embedding: list[float],
        score_threshold: float = 0.5,
        limit: int = 20,
        meta_build_version: str | None = None,
    ) -> list[ValueInfo]:
        result = await query_grouped_points(
            self.client,
            collection_name=self.collection_name,
            embedding=embedding,
            group_by="candidate_id",
            group_size=3,
            limit=limit,
            score_threshold=score_threshold,
            meta_build_version=meta_build_version,
        )
        candidates = []
        for group in result.groups:
            hits = list(group.hits or [])
            if not hits:
                continue
            payload = hits[0].payload or {}
            matched_texts = list(
                dict.fromkeys(
                    str(hit.payload.get("matched_text") or "")
                    for hit in hits
                    if hit.payload and hit.payload.get("matched_text")
                )
            )
            candidates.append(
                ValueInfo(
                    id=str(payload["candidate_id"]),
                    value=str(payload["value"]),
                    column_id=str(payload["column_id"]),
                    matched_texts=matched_texts,
                )
            )
        return candidates
