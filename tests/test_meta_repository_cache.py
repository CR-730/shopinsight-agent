from app.repositories.mysql.meta.meta_mysql_repository import (
    MetaMySQLRepository,
    build_metadata_cache_version,
)


class FakeSession:
    def __init__(self):
        self.added_batches = []

    def add_all(self, items):
        self.added_batches.append(items)


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
