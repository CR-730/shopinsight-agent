import pytest

from app.agent.cost import CostRates, CostTracker
from app.agent.retrieval_context import _search_values_by_vector


class FakeEmbeddingClient:
    last_cache_hit = False

    async def aembed_query(self, text: str):
        assert text == "华东"
        return [0.1, 0.2, 0.3]


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
