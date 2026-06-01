import asyncio

import pytest

from app.agent.cached_clients import CachedEmbeddingClient, ainvoke_with_timeout


class FakeEmbeddingClient:
    def __init__(self):
        self.calls = 0

    async def aembed_query(self, text):
        self.calls += 1
        return [float(len(text))]


@pytest.mark.anyio
async def test_cached_embedding_client_reuses_query_embeddings():
    inner = FakeEmbeddingClient()
    cached = CachedEmbeddingClient(inner)

    first = await cached.aembed_query("华北")
    second = await cached.aembed_query("华北")

    assert first == second == [2.0]
    assert inner.calls == 1
    assert cached.last_cache_hit is True


@pytest.mark.anyio
async def test_ainvoke_with_timeout_raises_timeout_error():
    async def slow():
        await asyncio.sleep(1)

    with pytest.raises(TimeoutError):
        await ainvoke_with_timeout(slow(), timeout_seconds=0.01)
