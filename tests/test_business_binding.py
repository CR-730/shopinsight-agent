import asyncio

from app.agent.nodes.business_binding import (
    business_binding,
    resolve_metric_bindings,
    resolve_time_binding,
    resolve_unresolved_bindings,
    resolve_value_filters,
)
from app.entities.value_alias import ValueAlias
from app.entities.value_info import ValueInfo


def test_resolve_metric_binding_maps_sales_amount_to_gmv():
    bindings = resolve_metric_bindings(
        query="销售额最高的商品",
        metric_infos=[
            {
                "name": "GMV",
                "alias": ["销售额", "成交额", "销售金额"],
                "relevant_columns": ["fact_order.order_amount"],
            }
        ],
    )

    assert bindings == [
        {
            "raw_mention": "销售额",
            "canonical_metric": "GMV",
            "matched_by": "metric_alias",
            "evidence": "GMV.alias contains 销售额",
            "relevant_columns": ["fact_order.order_amount"],
            "confidence": "high",
        }
    ]


def test_resolve_value_filter_maps_north_alias_to_canonical_region():
    filters, issues = resolve_value_filters(
        query="北方区域销售额",
        retrieved_value_infos=[
            ValueInfo(
                id="dim_region.region_name.华北",
                value="华北",
                column_id="dim_region.region_name",
            )
        ],
        enum_aliases={
            "dim_region.region_name": {"北方区域": "华北", "北方": "华北"}
        },
    )

    assert issues == []
    assert filters == [
        {
            "raw_value": "北方区域",
            "canonical_value": "华北",
            "column": "dim_region.region_name",
            "field_alias": "",
            "matched_by": "enum_alias",
            "allowed_sql_literals": ["华北"],
        }
    ]


def test_resolve_value_filter_marks_unknown_value_by_field_alias():
    filters, issues = resolve_value_filters(
        query="火星区域的销售额是多少",
        table_infos=[
            {
                "name": "dim_region",
                "role": "dimension",
                "columns": [
                    {
                        "name": "region_name",
                        "role": "dimension",
                        "alias": ["地区", "区域", "大区"],
                        "examples": ["华北", "华东"],
                    }
                ],
            }
        ],
        retrieved_value_infos=[],
        enum_aliases={},
    )

    assert filters == []
    assert issues == [
        {
            "type": "enum_value",
            "raw_text": "火星",
            "candidate_column": "dim_region.region_name",
            "reason": "value_not_found",
        }
    ]


def test_resolve_value_filter_binds_standalone_rewritten_region_override():
    filters, issues = resolve_value_filters(
        query="统计 华东地区 GMV",
        table_infos=[
            {
                "name": "dim_region",
                "role": "dimension",
                "columns": [
                    {
                        "name": "region_name",
                        "role": "dimension",
                        "alias": ["地区", "区域", "大区"],
                        "examples": ["华北", "华东"],
                    }
                ],
            }
        ],
        retrieved_value_infos=[
            ValueInfo(
                id="dim_region.region_name.华东",
                value="华东",
                column_id="dim_region.region_name",
            )
        ],
        enum_aliases={},
    )

    assert issues == []
    assert filters == [
        {
            "raw_value": "华东",
            "canonical_value": "华东",
            "column": "dim_region.region_name",
            "field_alias": "地区",
            "matched_by": "retrieved_value",
            "allowed_sql_literals": ["华东"],
        }
    ]


def test_resolve_value_filter_ignores_time_prefix_before_dimension_value():
    filters, issues = resolve_value_filters(
        query="统计 2025 年第一季度华北地区 GMV",
        table_infos=[
            {
                "name": "dim_region",
                "role": "dimension",
                "columns": [
                    {
                        "name": "region_name",
                        "role": "dimension",
                        "alias": ["地区", "区域", "大区"],
                        "examples": ["华北", "华东"],
                    }
                ],
            }
        ],
        retrieved_value_infos=[
            ValueInfo(
                id="dim_region.region_name.华北",
                value="华北",
                column_id="dim_region.region_name",
            )
        ],
        enum_aliases={},
    )

    assert issues == []
    assert filters[0]["canonical_value"] == "华北"


def test_resolve_value_filter_does_not_treat_group_by_alias_as_value():
    filters, issues = resolve_value_filters(
        query="按大区统计 GMV",
        table_infos=[
            {
                "name": "dim_region",
                "role": "dimension",
                "columns": [
                    {
                        "name": "region_name",
                        "role": "dimension",
                        "alias": ["地区", "区域", "大区"],
                        "examples": ["华北", "华东"],
                    }
                ],
            }
        ],
        retrieved_value_infos=[],
        enum_aliases={},
    )

    assert filters == []
    assert issues == []


def test_resolve_value_filter_does_not_treat_time_before_group_alias_as_value():
    filters, issues = resolve_value_filters(
        query="2025 年第一季度各大区 GMV",
        table_infos=[
            {
                "name": "dim_region",
                "role": "dimension",
                "columns": [
                    {
                        "name": "region_name",
                        "role": "dimension",
                        "alias": ["地区", "区域", "大区"],
                        "examples": ["华北", "华东"],
                    }
                ],
            }
        ],
        retrieved_value_infos=[],
        enum_aliases={},
    )

    assert filters == []
    assert issues == []


def test_resolve_time_binding_parses_quarter_to_date_range():
    assert resolve_time_binding("2025 年第一季度各大区 GMV") == {
        "raw_text": "2025 年第一季度",
        "grain": "quarter",
        "year": 2025,
        "quarter": "Q1",
        "start_date": "2025-01-01",
        "end_date": "2025-03-31",
        "start_date_id": 20250101,
        "end_date_id": 20250331,
        "strategy": "date_range",
        "required_columns": ["fact_order.date_id"],
    }


def test_business_binding_marks_unknown_metric_as_unresolved():
    issues = resolve_unresolved_bindings(
        query="品牌忠诚度是多少",
        metric_bindings=[],
    filter_issues=[],
    enum_aliases={},
    )

    assert issues == [
        {
            "type": "metric",
            "raw_text": "品牌忠诚度是多少",
            "candidate_column": "",
            "reason": "metric_not_bound",
        }
    ]


def test_business_binding_node_returns_metric_and_value_bindings():
    class MetaRepository:
        async def list_value_aliases(self):
            return [
                ValueAlias(
                    column_id="dim_region.region_name",
                    alias="北方区域",
                    canonical_value="华北",
                ),
                ValueAlias(
                    column_id="dim_region.region_name",
                    alias="北方",
                    canonical_value="华北",
                ),
            ]

    class Runtime:
        stream_writer = staticmethod(lambda _: None)
        context = {
            "meta_mysql_repository": MetaRepository(),
            "dw_mysql_repository": type(
                "DWRepository",
                (),
                {"column_value_exists": staticmethod(lambda *args: False)},
            )(),
        }

    result = asyncio.run(
        business_binding(
            {
                "query": "北方区域销售额",
                "metric_infos": [
                    {
                        "name": "GMV",
                        "description": "Gross Merchandise Value",
                        "alias": ["销售额"],
                        "relevant_columns": ["fact_order.order_amount"],
                    }
                ],
            "retrieved_value_infos": [
                    ValueInfo(
                        id="dim_region.region_name.华北",
                        value="华北",
                        column_id="dim_region.region_name",
                    )
                ],
            },
            Runtime(),
        )
    )

    assert result["metric_bindings"][0]["canonical_metric"] == "GMV"
    assert result["business_binding"]["metrics"][0]["canonical_metric"] == "GMV"
    assert result["resolved_filters"][0]["canonical_value"] == "华北"
    assert result["validated_enum_values"] == ["华北"]
    assert result["unresolved_bindings"] == []


def test_filter_metric_prunes_by_bound_metric_without_llm():
    from app.agent.context_compaction import filter_metric_context

    result = filter_metric_context(
        {
            "query": "销售额最高的商品",
            "metric_bindings": [{"canonical_metric": "GMV"}],
            "metric_infos": [
                {
                    "name": "GMV",
                    "description": "Gross Merchandise Value",
                    "alias": ["销售额"],
                    "relevant_columns": ["fact_order.order_amount"],
                },
                {
                    "name": "AOV",
                    "description": "Average Order Value",
                    "alias": ["客单价"],
                    "relevant_columns": ["fact_order.order_amount"],
                },
            ],
        }
    )

    assert [metric["name"] for metric in result["metric_infos"]] == ["GMV"]


def test_business_binding_uses_dw_fallback_when_alias_value_was_not_recalled():
    class MetaRepository:
        async def list_value_aliases(self):
            return [
                ValueAlias(
                    column_id="dim_region.region_name",
                    alias="北方区域",
                    canonical_value="华北",
                )
            ]

    class DWRepository:
        async def column_value_exists(self, table_name, column_name, value):
            return (table_name, column_name, value) == (
                "dim_region",
                "region_name",
                "华北",
            )

    class Runtime:
        stream_writer = staticmethod(lambda _: None)
        context = {
            "meta_mysql_repository": MetaRepository(),
            "dw_mysql_repository": DWRepository(),
        }

    result = asyncio.run(
        business_binding(
            {
                "query": "北方区域销售额",
                "metric_infos": [
                    {
                        "name": "GMV",
                        "alias": ["销售额"],
                        "relevant_columns": ["fact_order.order_amount"],
                    }
                ],
                "retrieved_value_infos": [],
            },
            Runtime(),
        )
    )

    assert result["resolved_filters"] == [
        {
            "raw_value": "北方区域",
            "canonical_value": "华北",
            "column": "dim_region.region_name",
            "field_alias": "",
            "matched_by": "enum_alias_db",
            "allowed_sql_literals": ["华北"],
        }
    ]
    assert result["unresolved_bindings"] == []
