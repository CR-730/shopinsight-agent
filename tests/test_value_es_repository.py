import pytest

from app.entities.value_info import ValueInfo
from app.repositories.es.value_es_repository import ValueESRepository


class FakeIndices:
    def __init__(self):
        self.created_mappings = None

    async def exists(self, index):
        return False

    async def create(self, index, mappings):
        self.created_mappings = mappings


class FakeElasticsearch:
    def __init__(self):
        self.indices = FakeIndices()
        self.bulk_operations = None
        self.search_kwargs = None

    async def bulk(self, operations):
        self.bulk_operations = operations

    async def search(self, **kwargs):
        self.search_kwargs = kwargs
        return {"hits": {"hits": []}}


@pytest.mark.anyio
async def test_value_es_mapping_contains_meta_build_version():
    client = FakeElasticsearch()
    repository = ValueESRepository(client)

    await repository.ensure_index()

    assert client.indices.created_mappings["properties"]["meta_build_version"] == {
        "type": "keyword"
    }


@pytest.mark.anyio
async def test_value_es_index_writes_meta_build_version():
    client = FakeElasticsearch()
    repository = ValueESRepository(client)

    await repository.index(
        [ValueInfo(id="dim_region.region_name.华北", value="华北", column_id="dim_region.region_name")],
        meta_build_version="build-a",
    )

    assert client.bulk_operations[1]["meta_build_version"] == "build-a"


@pytest.mark.anyio
async def test_value_es_search_filters_by_meta_build_version():
    client = FakeElasticsearch()
    repository = ValueESRepository(client)

    await repository.search("华北", meta_build_version="build-a")

    assert client.search_kwargs["query"] == {
        "bool": {
            "must": [
                {"match": {"value": "华北"}},
                {"term": {"meta_build_version": "build-a"}},
            ]
        }
    }
