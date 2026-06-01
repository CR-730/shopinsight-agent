"""Agent 调用外部模型和 Embedding 的轻量治理工具。"""

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable


async def ainvoke_with_timeout(awaitable: Awaitable, timeout_seconds: float):
    """给任意异步调用添加超时保护。"""

    return await asyncio.wait_for(awaitable, timeout=timeout_seconds)


class CachedEmbeddingClient:
    """为 Embedding query/document 调用增加进程内 LRU 缓存。"""

    def __init__(self, inner, max_entries: int = 2048):
        self.inner = inner
        self.max_entries = max_entries
        self.last_cache_hit = False
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._documents_cache: OrderedDict[tuple[str, ...], list[list[float]]] = OrderedDict()

    async def aembed_query(self, text: str) -> list[float]:
        if text in self._query_cache:
            self.last_cache_hit = True
            return self._touch(self._query_cache, text)
        self.last_cache_hit = False
        embedding = await self.inner.aembed_query(text)
        self._store(self._query_cache, text, embedding)
        return embedding

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        key = tuple(texts)
        if key in self._documents_cache:
            self.last_cache_hit = True
            return self._touch(self._documents_cache, key)
        self.last_cache_hit = False
        embeddings = await self.inner.aembed_documents(texts)
        self._store(self._documents_cache, key, embeddings)
        return embeddings

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def _store(self, cache: OrderedDict, key, value):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.max_entries:
            cache.popitem(last=False)

    @staticmethod
    def _touch(cache: OrderedDict, key):
        cache.move_to_end(key)
        return cache[key]
