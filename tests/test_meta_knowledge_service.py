from pathlib import Path

import pytest

from app.services.meta_knowledge_service import MetaKnowledgeService


class FakeMetaRepository:
    def __init__(self):
        self.cleared = False
        self.saved_build_versions = []

    async def clear_all(self):
        self.cleared = True

    async def save_build_version(self, version: str, config_path: Path):
        self.saved_build_versions.append((version, config_path))


class FakeDWRepository:
    pass


class FakeEmbeddingClient:
    pass


class FakeQdrantRepository:
    def __init__(self):
        self.recreated = False
        self.upserts = []

    async def recreate_collection(self):
        self.recreated = True

    async def upsert(self, ids, embeddings, payloads):
        self.upserts.append((ids, embeddings, payloads))


class FakeValueRepository:
    def __init__(self):
        self.recreated = False
        self.indexed = []

    async def recreate_index(self):
        self.recreated = True

    async def index(self, value_infos):
        self.indexed.append(value_infos)


@pytest.mark.anyio
async def test_empty_meta_config_clears_all_search_indexes(tmp_path):
    config_path = tmp_path / "meta_config.yaml"
    config_path.write_text("tables: []\nmetrics: []\n", encoding="utf-8")
    meta_repository = FakeMetaRepository()
    column_repository = FakeQdrantRepository()
    value_repository = FakeValueRepository()
    value_qdrant_repository = FakeQdrantRepository()
    metric_repository = FakeQdrantRepository()
    service = MetaKnowledgeService(
        meta_mysql_repository=meta_repository,
        dw_mysql_repository=FakeDWRepository(),
        column_qdrant_repository=column_repository,
        embedding_client=FakeEmbeddingClient(),
        value_es_repository=value_repository,
        value_qdrant_repository=value_qdrant_repository,
        metric_qdrant_repository=metric_repository,
    )

    await service.build(Path(config_path))

    assert meta_repository.cleared is True
    assert column_repository.recreated is True
    assert value_repository.recreated is True
    assert value_qdrant_repository.recreated is True
    assert metric_repository.recreated is True
    assert column_repository.upserts == []
    assert value_repository.indexed == []
    assert metric_repository.upserts == []
    assert meta_repository.saved_build_versions == [
        (MetaKnowledgeService._build_version(config_path), config_path)
    ]
