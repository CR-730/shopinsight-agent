import pytest

from app.entities.value_info import ValueInfo, ValueSearchDocument
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


class FakeElasticsearchHit(FakeElasticsearch):
    async def search(self, **kwargs):
        self.search_kwargs = kwargs
        return {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "id": "document:north-alias",
                            "candidate_id": "value:region:north",
                            "value": "华北",
                            "column_id": "dim_region.region_name",
                            "matched_text": "北方区域",
                            "surface_type": "alias",
                        }
                    }
                ]
            }
        }


@pytest.mark.anyio
async def test_value_es_mapping_contains_meta_build_version():
    client = FakeElasticsearch()
    repository = ValueESRepository(client)

    await repository.ensure_index()

    assert client.indices.created_mappings["properties"]["meta_build_version"] == {
        "type": "keyword"
    }
    assert client.indices.created_mappings["properties"]["candidate_id"] == {
        "type": "keyword"
    }
    assert client.indices.created_mappings["properties"]["matched_text"][
        "analyzer"
    ] == "ik_max_word"


@pytest.mark.anyio
async def test_value_es_index_writes_meta_build_version():
    client = FakeElasticsearch()
    repository = ValueESRepository(client)

    await repository.index(
        [
            ValueSearchDocument(
                id="document:north-canonical",
                candidate_id="value:region:north",
                value="华北",
                column_id="dim_region.region_name",
                matched_text="华北",
                surface_type="canonical",
            )
        ],
        meta_build_version="build-a",
    )

    assert client.bulk_operations[1]["meta_build_version"] == "build-a"


@pytest.mark.anyio
async def test_value_es_indexes_surface_document_with_shared_candidate_id():
    client = FakeElasticsearch()
    repository = ValueESRepository(client)

    await repository.index(
        [
            ValueSearchDocument(
                id="document:north-alias",
                candidate_id="value:region:north",
                value="华北",
                column_id="dim_region.region_name",
                matched_text="北方区域",
                surface_type="alias",
            )
        ],
        meta_build_version="build-a",
    )

    assert client.bulk_operations[0]["index"]["_id"] == "document:north-alias"
    assert client.bulk_operations[1]["candidate_id"] == "value:region:north"
    assert client.bulk_operations[1]["matched_text"] == "北方区域"


@pytest.mark.anyio
async def test_value_es_search_filters_by_meta_build_version():
    client = FakeElasticsearch()
    repository = ValueESRepository(client)

    await repository.search("华北", meta_build_version="build-a")

    assert client.search_kwargs["query"] == {
        "bool": {
            "must": [
                {"match": {"matched_text": "华北"}},
                {"term": {"meta_build_version": "build-a"}},
            ]
        }
    }
    assert client.search_kwargs["collapse"] == {"field": "candidate_id"}


@pytest.mark.anyio
async def test_value_es_search_returns_canonical_candidate_from_alias_hit():
    repository = ValueESRepository(FakeElasticsearchHit())

    results = await repository.search("北方区域")

    assert results == [
        ValueInfo(
            id="value:region:north",
            value="华北",
            column_id="dim_region.region_name",
            matched_texts=["北方区域"],
        )
    ]
