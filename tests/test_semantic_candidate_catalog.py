from dataclasses import FrozenInstanceError

import pytest

from app.agent.semantic_planning.catalog import build_semantic_candidate_catalog
from app.entities.column_info import ColumnInfo
from app.entities.metric_info import MetricInfo
from app.entities.value_alias import ValueAlias
from app.entities.value_info import ValueInfo


def _column(table: str, name: str, role: str, data_type: str = "bigint"):
    return ColumnInfo(
        id=f"{table}.{name}",
        name=name,
        type=data_type,
        role=role,
        examples=[],
        description=f"{name} description",
        alias=[f"{name} alias"],
        table_id=table,
    )


def _authoritative_columns():
    return [
        _column("fact_order", "region_id", "foreign_key"),
        _column("fact_order", "order_amount", "measure", "decimal"),
        _column("dim_region", "region_id", "primary_key"),
        _column("dim_region", "region_name", "dimension", "varchar"),
        _column("dim_hidden", "region_name", "dimension", "varchar"),
    ]


def _metric():
    return MetricInfo(
        id="GMV",
        name="GMV",
        description="成交金额总和",
        relevant_columns=["fact_order.order_amount"],
        alias=["销售额"],
        aggregation="sum",
        expression=None,
    )


def _sql_context():
    return {
        "tables": [
            {
                "name": "fact_order",
                "role": "fact",
                "description": "订单事实表",
                "columns": [
                    {"name": "region_id", "role": "wrong", "type": "text"},
                    {"name": "order_amount", "alias": ["伪造别名"]},
                ],
            },
            {
                "name": "dim_region",
                "role": "dim",
                "description": "地区维表",
                "columns": [
                    {"name": "region_id"},
                    {"name": "region_name"},
                ],
            },
        ],
        "metrics": [{"name": "GMV", "aggregation": "median"}],
    }


def _catalog(**changes):
    args = {
        "sql_context": _sql_context(),
        "retrieved_value_infos": [
            ValueInfo(
                id="north",
                value="华北",
                column_id="dim_region.region_name",
            ),
            ValueInfo(
                id="hidden",
                value="隐藏值",
                column_id="dim_hidden.region_name",
            ),
        ],
        "value_aliases": [
            ValueAlias(
                column_id="dim_region.region_name",
                alias="北方区域",
                canonical_value="华北",
            )
        ],
        "authoritative_columns": _authoritative_columns(),
        "authoritative_metrics": [_metric()],
        "metadata_version": "meta-v2",
        "policy": {},
    }
    args.update(changes)
    return build_semantic_candidate_catalog(**args)


def test_catalog_keeps_authoritative_metric_and_column_contract():
    catalog = _catalog()

    assert catalog.metadata_version == "meta-v2"
    assert catalog.metrics["GMV"].aggregation == "sum"
    assert catalog.metrics["GMV"].description == "成交金额总和"
    assert catalog.metrics["GMV"].relevant_columns == (
        "fact_order.order_amount",
    )
    assert catalog.columns["fact_order.order_amount"].data_type == "decimal"
    assert catalog.columns["fact_order.region_id"].role == "foreign_key"
    assert catalog.columns["fact_order.order_amount"].aliases == (
        "order_amount alias",
    )
    assert catalog.relationships


def test_values_keep_owning_column_and_hidden_objects_are_not_promoted():
    catalog = _catalog()

    assert {value.column_id for value in catalog.values.values()} == {
        "dim_region.region_name"
    }
    value = next(iter(catalog.values.values()))
    assert value.aliases == ("北方区域",)
    assert "dim_hidden.region_name" not in catalog.columns
    with pytest.raises(KeyError):
        catalog.column_by_id("llm.invented_column")


def test_metric_uses_authoritative_id_not_legacy_prefixed_name():
    catalog = _catalog()

    assert set(catalog.metrics) == {"GMV"}
    assert catalog.metric_by_id("GMV").candidate_id == "GMV"
    assert "metric:GMV" not in catalog.metrics


def test_relationship_id_is_stable_and_ordinary_same_name_columns_do_not_join():
    first = _catalog()
    second = _catalog(authoritative_columns=list(reversed(_authoritative_columns())))

    assert set(first.relationships) == set(second.relationships)
    relationship = next(iter(first.relationships.values()))
    assert {relationship.left_column_id, relationship.right_column_id} == {
        "fact_order.region_id",
        "dim_region.region_id",
    }

    columns = _authoritative_columns() + [
        _column("fact_order", "label", "dimension", "varchar"),
        _column("dim_region", "label", "dimension", "varchar"),
    ]
    with_ordinary_names = _catalog(authoritative_columns=columns)
    assert len(with_ordinary_names.relationships) == 1


def test_catalog_and_records_are_immutable():
    catalog = _catalog()

    with pytest.raises(TypeError):
        catalog.columns["new"] = catalog.columns["fact_order.region_id"]
    with pytest.raises(FrozenInstanceError):
        catalog.columns["fact_order.region_id"].role = "dimension"


def test_exposed_object_without_authoritative_metadata_fails_closed():
    with pytest.raises(ValueError, match="authoritative_metric_missing"):
        _catalog(authoritative_metrics=[])

    missing_column = [
        column
        for column in _authoritative_columns()
        if column.id != "fact_order.order_amount"
    ]
    with pytest.raises(ValueError, match="authoritative_column_missing"):
        _catalog(authoritative_columns=missing_column)


def test_catalog_requires_a_metadata_version_boundary():
    with pytest.raises(ValueError, match="metadata_version_required"):
        _catalog(metadata_version="  ")
