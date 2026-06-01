from app.agent.keyword_expansion import normalize_keyword_list
from app.models.column_info import ColumnInfoMySQL
from app.models.metric_info import MetricInfoMySQL
from app.repositories.mysql.meta.mappers.column_info_mapper import ColumnInfoMapper
from app.repositories.mysql.meta.mappers.metric_info_mapper import MetricInfoMapper


def test_normalize_keyword_list_accepts_common_llm_shapes():
    raw = {
        "keywords": [
            "GMV",
            ["华北", "销售额"],
            {"name": "订单金额"},
            {"value": "家电"},
            "",
            None,
        ]
    }

    assert normalize_keyword_list(raw) == ["GMV", "华北", "销售额", "订单金额", "家电"]


def test_normalize_keyword_list_wraps_plain_string():
    assert normalize_keyword_list("GMV") == ["GMV"]


def test_column_mapper_decodes_json_string_fields():
    entity = ColumnInfoMapper.to_entity(
        ColumnInfoMySQL(
            id="fact_order.order_amount",
            name="order_amount",
            type="decimal",
            role="measure",
            examples='["100.00", "200.00"]',
            description="订单金额",
            alias='["GMV", "销售额"]',
            table_id="fact_order",
        )
    )

    assert entity.examples == ["100.00", "200.00"]
    assert entity.alias == ["GMV", "销售额"]


def test_metric_mapper_decodes_json_string_fields():
    entity = MetricInfoMapper.to_entity(
        MetricInfoMySQL(
            id="gmv",
            name="GMV",
            description="销售额",
            relevant_columns='["fact_order.order_amount"]',
            alias='["销售额"]',
        )
    )

    assert entity.relevant_columns == ["fact_order.order_amount"]
    assert entity.alias == ["销售额"]
