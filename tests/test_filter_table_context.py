from app.agent.context_compaction import compact_table_context_for_filtering


def test_compact_table_context_preserves_candidates_but_drops_heavy_fields():
    table_infos = [
        {
            "name": "fact_order",
            "role": "fact",
            "description": "orders",
            "columns": [
                {
                    "name": "order_amount",
                    "role": "metric",
                    "alias": ["sales"],
                    "description": "total paid amount",
                    "examples": ["100.00"],
                },
                {"name": "region_id", "role": "foreign_key", "alias": []},
                {"name": "date_id", "role": "foreign_key", "alias": []},
                {"name": "customer_id", "role": "foreign_key", "alias": []},
            ],
        },
        {
            "name": "dim_region",
            "role": "dimension",
            "description": "regions",
            "columns": [
                {"name": "region_id", "role": "primary_key", "alias": []},
                {
                    "name": "region_name",
                    "role": "dimension",
                    "alias": ["region"],
                    "description": "region name",
                },
            ],
        },
    ]

    compacted = compact_table_context_for_filtering(
        table_infos,
        business_binding={
            "metrics": [{"relevant_columns": ["fact_order.order_amount"]}],
            "filters": [{"column": "dim_region.region_name"}],
            "time": {"required_columns": ["fact_order.date_id"]},
        },
    )

    assert compacted == [
        {
            "name": "fact_order",
            "role": "fact",
            "columns": [
                {"name": "order_amount", "role": "metric", "alias": ["sales"]},
                {"name": "region_id", "role": "foreign_key", "alias": []},
                {"name": "date_id", "role": "foreign_key", "alias": []},
                {"name": "customer_id", "role": "foreign_key", "alias": []},
            ],
        },
        {
            "name": "dim_region",
            "role": "dimension",
            "columns": [
                {"name": "region_id", "role": "primary_key", "alias": []},
                {"name": "region_name", "role": "dimension", "alias": ["region"]},
            ],
        },
    ]


def test_compact_table_context_keeps_original_when_binding_is_empty():
    table_infos = [
        {
            "name": "fact_order",
            "columns": [{"name": "order_amount", "description": "amount"}],
        }
    ]

    assert compact_table_context_for_filtering(table_infos, {}) == table_infos
