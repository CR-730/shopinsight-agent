from pathlib import Path

import pytest

from app.conf.meta_config import MetaConfig, MetricConfig
from app.entities.metric_info import MetricInfo
from app.services.meta_knowledge_service import MetaKnowledgeService


class FakeMetaRepository:
    def __init__(self):
        self.cleared = False
        self.saved_build_versions = []
        self.saved_value_aliases = []

    async def clear_all(self):
        self.cleared = True

    async def save_build_version(self, version: str, config_path: Path):
        self.saved_build_versions.append((version, config_path))

    def save_value_aliases(self, value_aliases):
        self.saved_value_aliases.extend(value_aliases)


class FakeDWRepository:
    pass


class FakeEmbeddingClient:
    async def aembed_documents(self, texts):
        return [[float(index)] for index, _ in enumerate(texts)]


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
async def test_empty_meta_config_clears_all_search_indexes(tmp_path, monkeypatch):
    config_path = tmp_path / "meta_config.yaml"
    config_path.write_text("tables: []\nmetrics: []\n", encoding="utf-8")
    cleared_llm_cache = False

    def fake_clear_llm_response_cache():
        nonlocal cleared_llm_cache
        cleared_llm_cache = True

    monkeypatch.setattr(
        "app.services.meta_knowledge_service.clear_llm_response_cache",
        fake_clear_llm_response_cache,
    )
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
    assert cleared_llm_cache is True


class _Transaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _Session:
    def begin(self):
        return _Transaction()


class _MetricMetaRepository(FakeMetaRepository):
    def __init__(self):
        super().__init__()
        self.session = _Session()
        self.saved_metrics = []
        self.saved_column_metrics = []

    def save_metric_infos(self, metrics):
        self.saved_metrics.extend(metrics)

    def save_column_metrics(self, column_metrics):
        self.saved_column_metrics.extend(column_metrics)


def _metric_service(meta_repository=None, metric_repository=None):
    return MetaKnowledgeService(
        meta_mysql_repository=meta_repository or _MetricMetaRepository(),
        dw_mysql_repository=FakeDWRepository(),
        column_qdrant_repository=FakeQdrantRepository(),
        embedding_client=FakeEmbeddingClient(),
        value_es_repository=FakeValueRepository(),
        value_qdrant_repository=FakeQdrantRepository(),
        metric_qdrant_repository=metric_repository or FakeQdrantRepository(),
    )


@pytest.mark.anyio
async def test_metric_definition_is_validated_before_meta_write():
    repository = _MetricMetaRepository()
    service = _metric_service(meta_repository=repository)
    config = MetaConfig(
        metrics=[
            MetricConfig(
                name="GMV",
                description="销售额",
                relevant_columns=["fact_order.order_amount"],
                alias=["销售额"],
                aggregation="median",
            )
        ]
    )

    with pytest.raises(ValueError, match="unsupported_metric_aggregation"):
        await service._save_metrics_to_meta_db(config)

    assert repository.saved_metrics == []
    assert repository.saved_column_metrics == []


@pytest.mark.anyio
async def test_metric_qdrant_payload_keeps_authoritative_semantics():
    metric_repository = FakeQdrantRepository()
    service = _metric_service(metric_repository=metric_repository)
    metric = MetricInfo(
        id="GMV",
        name="GMV",
        description="销售额",
        relevant_columns=["fact_order.order_amount"],
        alias=["销售额"],
        aggregation="sum",
        expression=None,
    )

    await service._save_metrics_to_qdrant([metric], "build-v1")

    _, _, payloads = metric_repository.upserts[0]
    assert all(payload["aggregation"] == "sum" for payload in payloads)
    assert all(payload["expression"] is None for payload in payloads)
    assert all(
        payload["relevant_columns"] == ["fact_order.order_amount"]
        for payload in payloads
    )
