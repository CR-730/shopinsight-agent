import pytest

from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository
from app.repositories.qdrant.value_qdrant_repository import ValueQdrantRepository


class FakeQueryPointsResult:
    points = []


class FakeQueryGroupsResult:
    groups = []


class FakeQdrantClient:
    def __init__(self):
        self.query_kwargs = None
        self.query_groups_kwargs = None

    async def query_points(self, **kwargs):
        self.query_kwargs = kwargs
        return FakeQueryPointsResult()

    async def query_points_groups(self, **kwargs):
        self.query_groups_kwargs = kwargs
        return FakeQueryGroupsResult()

    async def get_collection(self, collection_name):
        return type(
            "CollectionInfo",
            (),
            {"points_count": 0, "payload_schema": {}},
        )()


class _Point:
    def __init__(self, payload):
        self.payload = payload


class _Group:
    def __init__(self, hits):
        self.hits = hits


class FakeGroupedValueClient(FakeQdrantClient):
    async def query_points_groups(self, **kwargs):
        self.query_groups_kwargs = kwargs
        return type(
            "GroupedResult",
            (),
            {
                "groups": [
                    _Group(
                        [
                            _Point(
                                {
                                    "candidate_id": "value:region:north",
                                    "value": "华北",
                                    "column_id": "dim_region.region_name",
                                    "matched_text": "北方区域",
                                }
                            ),
                            _Point(
                                {
                                    "candidate_id": "value:region:north",
                                    "value": "华北",
                                    "column_id": "dim_region.region_name",
                                    "matched_text": "华北",
                                }
                            ),
                        ]
                    )
                ]
            },
        )()


class FakeGroupedEntityClient(FakeQdrantClient):
    def __init__(self, payload):
        super().__init__()
        self.payload = payload

    async def query_points_groups(self, **kwargs):
        self.query_groups_kwargs = kwargs
        return type(
            "GroupedResult",
            (),
            {"groups": [_Group([_Point(self.payload)])]},
        )()


class FakeLegacyValueCollectionClient(FakeQdrantClient):
    async def get_collection(self, collection_name):
        return type(
            "CollectionInfo",
            (),
            {
                "points_count": 3,
                "payload_schema": {
                    "meta_build_version": type("IndexInfo", (), {"points": 3})()
                },
            },
        )()


class FakeCreateValueCollectionClient:
    def __init__(self, exists=False):
        self.exists = exists
        self.collection_kwargs = None
        self.payload_index_calls = []

    async def collection_exists(self, collection_name):
        return self.exists

    async def create_collection(self, **kwargs):
        self.collection_kwargs = kwargs

    async def create_payload_index(self, **kwargs):
        self.payload_index_calls.append(kwargs)


async def _search_with_version(repository):
    await repository.search([0.1, 0.2], meta_build_version="build-a")
    query_filter = repository.client.query_groups_kwargs["query_filter"]
    condition = query_filter.must[0]
    return condition.key, condition.match.value


@pytest.mark.anyio
async def test_column_qdrant_search_filters_by_meta_build_version():
    client = FakeQdrantClient()
    repository = ColumnQdrantRepository(client)

    key, value = await _search_with_version(repository)

    assert client.query_kwargs is None
    assert client.query_groups_kwargs["group_by"] == "id"
    assert client.query_groups_kwargs["group_size"] == 1
    assert key == "meta_build_version"
    assert value == "build-a"


@pytest.mark.anyio
async def test_metric_qdrant_search_filters_by_meta_build_version():
    client = FakeQdrantClient()
    repository = MetricQdrantRepository(client)

    key, value = await _search_with_version(repository)

    assert client.query_kwargs is None
    assert client.query_groups_kwargs["group_by"] == "id"
    assert client.query_groups_kwargs["group_size"] == 1
    assert key == "meta_build_version"
    assert value == "build-a"


@pytest.mark.anyio
async def test_value_qdrant_search_filters_by_meta_build_version():
    client = FakeQdrantClient()
    repository = ValueQdrantRepository(client)

    await repository.search([0.1, 0.2], meta_build_version="build-a")
    query_filter = client.query_groups_kwargs["query_filter"]
    condition = query_filter.must[0]

    assert client.query_kwargs is None
    assert client.query_groups_kwargs["group_by"] == "candidate_id"
    assert client.query_groups_kwargs["group_size"] == 3
    assert condition.key == "meta_build_version"
    assert condition.match.value == "build-a"


@pytest.mark.anyio
async def test_value_qdrant_search_returns_one_canonical_candidate_per_group():
    repository = ValueQdrantRepository(FakeGroupedValueClient())

    results = await repository.search([0.1, 0.2])

    assert len(results) == 1
    assert results[0].id == "value:region:north"
    assert results[0].value == "华北"
    assert results[0].column_id == "dim_region.region_name"
    assert results[0].matched_texts == ["北方区域", "华北"]


@pytest.mark.anyio
async def test_value_qdrant_search_rejects_legacy_collection_without_group_key():
    repository = ValueQdrantRepository(FakeLegacyValueCollectionClient())

    with pytest.raises(RuntimeError, match="qdrant_grouping_rebuild_required"):
        await repository.search([0.1, 0.2])


@pytest.mark.anyio
async def test_value_qdrant_collection_indexes_group_and_filter_keys():
    client = FakeCreateValueCollectionClient()
    repository = ValueQdrantRepository(client)

    await repository.ensure_collection()

    assert client.collection_kwargs["collection_name"] == repository.collection_name
    assert [call["field_name"] for call in client.payload_index_calls] == [
        "candidate_id",
        "meta_build_version",
    ]
    assert all(
        str(call["field_schema"]) == "keyword"
        for call in client.payload_index_calls
    )


@pytest.mark.anyio
async def test_value_qdrant_existing_collection_backfills_payload_indexes():
    client = FakeCreateValueCollectionClient(exists=True)
    repository = ValueQdrantRepository(client)

    await repository.ensure_collection()

    assert client.collection_kwargs is None
    assert [call["field_name"] for call in client.payload_index_calls] == [
        "candidate_id",
        "meta_build_version",
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "repository_type",
    [ColumnQdrantRepository, MetricQdrantRepository],
)
async def test_entity_qdrant_collections_index_group_and_filter_keys(repository_type):
    client = FakeCreateValueCollectionClient(exists=True)
    repository = repository_type(client)

    await repository.ensure_collection()

    assert [call["field_name"] for call in client.payload_index_calls] == [
        "id",
        "meta_build_version",
    ]


@pytest.mark.anyio
async def test_column_qdrant_group_returns_one_column_candidate():
    payload = {
        "id": "dim_region.region_name",
        "name": "region_name",
        "type": "varchar(32)",
        "role": "dimension",
        "examples": ["华北"],
        "description": "大区",
        "alias": ["地区", "区域"],
        "table_id": "dim_region",
    }
    repository = ColumnQdrantRepository(FakeGroupedEntityClient(payload))

    results = await repository.search([0.1, 0.2])

    assert [item.id for item in results] == ["dim_region.region_name"]


@pytest.mark.anyio
async def test_metric_qdrant_group_returns_one_metric_candidate():
    payload = {
        "id": "GMV",
        "name": "销售额",
        "description": "订单销售额",
        "relevant_columns": ["fact_order.order_amount"],
        "alias": ["成交额"],
        "aggregation": "sum",
        "expression": None,
    }
    repository = MetricQdrantRepository(FakeGroupedEntityClient(payload))

    results = await repository.search([0.1, 0.2])

    assert [item.id for item in results] == ["GMV"]
