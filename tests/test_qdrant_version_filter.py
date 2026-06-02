import pytest

from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository
from app.repositories.qdrant.value_qdrant_repository import ValueQdrantRepository


class FakeQueryPointsResult:
    points = []


class FakeQdrantClient:
    def __init__(self):
        self.query_kwargs = None

    async def query_points(self, **kwargs):
        self.query_kwargs = kwargs
        return FakeQueryPointsResult()


async def _search_with_version(repository):
    await repository.search([0.1, 0.2], meta_build_version="build-a")
    query_filter = repository.client.query_kwargs["query_filter"]
    condition = query_filter.must[0]
    return condition.key, condition.match.value


@pytest.mark.anyio
async def test_column_qdrant_search_filters_by_meta_build_version():
    repository = ColumnQdrantRepository(FakeQdrantClient())

    key, value = await _search_with_version(repository)

    assert key == "meta_build_version"
    assert value == "build-a"


@pytest.mark.anyio
async def test_metric_qdrant_search_filters_by_meta_build_version():
    repository = MetricQdrantRepository(FakeQdrantClient())

    key, value = await _search_with_version(repository)

    assert key == "meta_build_version"
    assert value == "build-a"


@pytest.mark.anyio
async def test_value_qdrant_search_filters_by_meta_build_version():
    repository = ValueQdrantRepository(FakeQdrantClient())

    key, value = await _search_with_version(repository)

    assert key == "meta_build_version"
    assert value == "build-a"
