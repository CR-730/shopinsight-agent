import asyncio

from app.repositories.mysql.meta.meta_mysql_repository import (
    MetaMySQLRepository,
    build_metadata_cache_version,
)


class FakeResult:
    def __init__(self, scalar_value=None, rows=None):
        self._scalar_value = scalar_value
        self._rows = rows or []

    def scalar(self):
        return self._scalar_value

    def mappings(self):
        return self

    def fetchall(self):
        return self._rows


class FakeSession:
    def __init__(self):
        self.added_batches = []
        self.executed_sql = []

    def add_all(self, items):
        self.added_batches.append(items)

    async def execute(self, statement):
        sql = str(statement)
        self.executed_sql.append(sql)
        if "select version from metadata_build" in sql:
            return FakeResult("build-a")
        return FakeResult(rows=[])


def test_save_column_infos_invalidates_instance_column_cache():
    repository = MetaMySQLRepository(FakeSession())
    repository._column_infos_cache = [object()]

    repository.save_column_infos([])

    assert repository._column_infos_cache is None


def test_save_metric_infos_invalidates_instance_metric_cache():
    repository = MetaMySQLRepository(FakeSession())
    repository._metric_infos_cache = [object()]

    repository.save_metric_infos([])

    assert repository._metric_infos_cache is None


def test_save_value_aliases_invalidates_instance_value_alias_cache():
    repository = MetaMySQLRepository(FakeSession())
    repository._value_aliases_cache = [object()]

    repository.save_value_aliases([])

    assert repository._value_aliases_cache is None


def test_query_version_reads_do_not_ensure_schema():
    async def run():
        await repository.get_active_build_version()
        await repository.get_metadata_cache_version()
        await repository.list_value_aliases()

    session = FakeSession()
    repository = MetaMySQLRepository(session)

    asyncio.run(run())

    assert all("create table" not in sql.lower() for sql in session.executed_sql)


def test_metadata_cache_version_changes_when_meta_rows_change():
    first = build_metadata_cache_version(
        active_build_version="build-a",
        table_rows={
            "metric_info": [
                {"id": "GMV", "name": "GMV", "alias": ["销售额"]},
            ],
            "column_info": [],
            "table_info": [],
            "column_metric": [],
            "value_alias": [],
        },
    )
    second = build_metadata_cache_version(
        active_build_version="build-a",
        table_rows={
            "metric_info": [
                {"id": "GMV", "name": "GMV", "alias": ["成交额"]},
            ],
            "column_info": [],
            "table_info": [],
            "column_metric": [],
            "value_alias": [],
        },
    )

    assert first != second


def test_metadata_cache_version_changes_when_active_build_changes():
    rows = {
        "metric_info": [{"id": "GMV", "name": "GMV"}],
        "column_info": [],
        "table_info": [],
        "column_metric": [],
        "value_alias": [],
    }

    first = build_metadata_cache_version("build-a", rows)
    second = build_metadata_cache_version("build-b", rows)

    assert first != second


def test_metadata_cache_version_changes_when_value_alias_rows_change():
    first = build_metadata_cache_version(
        active_build_version="build-a",
        table_rows={
            "metric_info": [],
            "column_info": [],
            "table_info": [],
            "column_metric": [],
            "value_alias": [
                {
                    "column_id": "dim_region.region_name",
                    "alias": "北方区域",
                    "canonical_value": "华北",
                }
            ],
        },
    )
    second = build_metadata_cache_version(
        active_build_version="build-a",
        table_rows={
            "metric_info": [],
            "column_info": [],
            "table_info": [],
            "column_metric": [],
            "value_alias": [
                {
                    "column_id": "dim_region.region_name",
                    "alias": "北方区域",
                    "canonical_value": "华东",
                }
            ],
        },
    )

    assert first != second
