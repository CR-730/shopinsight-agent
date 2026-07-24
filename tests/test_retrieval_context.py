import pytest

from app.agent.cost import CostRates, CostTracker
from app.agent.retrieval_context import (
    _search_values_by_vector,
    build_route_retrieval_queries,
    extract_retrieval_keywords,
    recall_sql_memory_context,
    recall_value_context,
)
from app.entities.value_info import ValueInfo


class FakeEmbeddingClient:
    last_cache_hit = False

    async def aembed_query(self, text: str):
        assert text == "华东"
        return [0.1, 0.2, 0.3]


@pytest.mark.anyio
async def test_retrieval_uses_only_the_rewritten_state_query(monkeypatch):
    extracted_from = []

    def fake_extract_tags(text, **kwargs):
        extracted_from.append(text)
        return []

    monkeypatch.setattr(
        "app.agent.retrieval_context.jieba.analyse.extract_tags",
        fake_extract_tags,
    )
    state = {
        "query": "统计华南地区的销售额",
        "conversation_messages": [
            {"role": "user", "content": "按地区统计销售额"},
        ],
    }

    result = await extract_retrieval_keywords(state)

    assert extracted_from == ["统计华南地区的销售额"]
    assert "统计华南地区的销售额" in result["keywords"]


def test_route_retrieval_queries_keep_full_query_without_jieba_keywords():
    queries = build_route_retrieval_queries(
        "统计2025年第一季度华北和华南各自的销售额",
        ["地区", "区域", "地区", "销售额", "  "],
    )

    assert queries == [
        "统计2025年第一季度华北和华南各自的销售额",
        "地区",
        "区域",
        "销售额",
    ]


@pytest.mark.anyio
async def test_sql_memory_recall_uses_only_the_rewritten_state_query():
    class MemoryRepository:
        query = None

        async def search_similar_usage(self, query, **kwargs):
            self.query = query
            return []

    repository = MemoryRepository()
    await recall_sql_memory_context(
        {
            "query": "统计华南地区的销售额",
            "conversation_messages": [
                {"role": "user", "content": "按地区统计销售额"},
            ],
        },
        {
            "user_id": "user-1",
            "agent_memory_repository": repository,
            "metadata_cache_version": "v1",
        },
    )

    assert repository.query == "统计华南地区的销售额"


class FakeValueQdrantRepository:
    async def search(self, embedding, score_threshold: float, meta_build_version: str | None):
        assert embedding == [0.1, 0.2, 0.3]
        assert score_threshold >= 0
        assert meta_build_version == "build-v1"
        return ["value-info"]


@pytest.mark.anyio
async def test_value_vector_recall_records_embedding_latency():
    tracker = CostTracker(CostRates())

    result = await _search_values_by_vector(
        FakeValueQdrantRepository(),
        FakeEmbeddingClient(),
        "华东",
        tracker,
        "build-v1",
    )

    assert result == ["value-info"]
    embedding_calls = [
        call for call in tracker.summary()["calls"] if call["type"] == "embedding"
    ]
    assert len(embedding_calls) == 1
    assert embedding_calls[0]["step"] == "召回字段取值"
    assert embedding_calls[0]["latency_ms"] is not None


class FailingESRepository:
    async def search(self, *args, **kwargs):
        raise AssertionError("ES must not be called when disable_value_es is enabled")


class VectorOnlyValueRepository:
    async def search(self, embedding, score_threshold, meta_build_version):
        return [
            ValueInfo(
                id="value:east",
                value="华东",
                column_id="dim_region.region_name",
                matched_texts=["华东"],
            )
        ]


@pytest.mark.anyio
async def test_value_recall_disable_es_uses_vector_only(monkeypatch):
    async def no_expansion(**kwargs):
        return []

    monkeypatch.setattr(
        "app.agent.retrieval_context._extend_keywords",
        no_expansion,
    )
    result = await recall_value_context(
        {"query": "华东", "keywords": ["华东"]},
        {
            "value_es_repository": FailingESRepository(),
            "value_qdrant_repository": VectorOnlyValueRepository(),
            "embedding_client": FakeEmbeddingClient(),
            "cost_tracker": CostTracker(CostRates()),
            "metadata_build_version": "build-v1",
            "ablation_options": {"disable_value_es": True},
        },
    )

    assert [item.id for item in result["retrieved_value_infos"]] == ["value:east"]
    assert result["retrieved_value_infos"][0].sources == ["vector"]
