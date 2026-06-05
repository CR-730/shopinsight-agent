import asyncio

from app.agent.business_binding.candidates import (
    BindingCandidates,
    FilterMention,
    GroupByMention,
    MetricMention,
    fallback_binding_candidates,
)
from app.agent.business_binding.time_resolver import resolve_time_binding
from app.agent.business_binding.validator import (
    BindingValidationContext,
    resolve_metric_candidates,
    validate_binding_candidates,
)
from app.agent.nodes import business_binding as business_binding_module
from app.agent.nodes.business_binding import business_binding
from app.entities.value_alias import ValueAlias
from app.entities.value_info import ValueInfo


class DWRepository:
    async def column_value_exists(self, table_name, column_name, value):
        return False


def _context(**overrides):
    context = {
        "metric_infos": [],
        "table_infos": [],
        "retrieved_value_infos": [],
        "enum_aliases": {},
        "dw_mysql_repository": DWRepository(),
    }
    context.update(overrides)
    return BindingValidationContext(**context)


def _region_table():
    return [
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
    ]


def _customer_table():
    return [
        {
            "name": "dim_customer",
            "role": "dimension",
            "columns": [
                {
                    "name": "customer_id",
                    "role": "primary_key",
                    "alias": ["客户ID"],
                    "examples": [],
                },
                {
                    "name": "member_level",
                    "role": "dimension",
                    "alias": ["会员等级", "用户等级"],
                    "examples": ["普通", "黄金"],
                },
            ],
        }
    ]


def test_metric_candidate_maps_sales_amount_to_gmv():
    bindings, issues = resolve_metric_candidates(
        BindingCandidates(
            metric_mentions=[MetricMention(raw_text="销售额", normalized_text="销售额")]
        ),
        [
            {
                "name": "GMV",
                "alias": ["销售额", "成交额", "销售金额"],
                "relevant_columns": ["fact_order.order_amount"],
            }
        ],
    )

    assert issues == []
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


def test_llm_spoofed_canonical_metric_is_ignored_when_catalog_does_not_match():
    candidates = BindingCandidates.model_validate(
        {
            "metric_mentions": [
                {
                    "raw_text": "品牌心智指数",
                    "normalized_text": "品牌心智指数",
                    "canonical_metric": "GMV",
                }
            ]
        }
    )

    bindings, issues = resolve_metric_candidates(
        candidates,
        [{"name": "GMV", "alias": ["销售额"], "relevant_columns": []}],
    )

    assert bindings == []
    assert issues == [
        {
            "type": "metric",
            "raw_text": "品牌心智指数",
            "candidate_column": "",
            "reason": "metric_not_bound",
        }
    ]


def test_filter_candidate_maps_north_alias_to_canonical_region():
    binding = asyncio.run(
        validate_binding_candidates(
            BindingCandidates(
                filter_mentions=[FilterMention(raw_text="北方区域", field_hint="")]
            ),
            _context(
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
            ),
        )
    )

    assert binding["unresolved"] == []
    assert binding["filters"] == [
        {
            "raw_value": "北方区域",
            "canonical_value": "华北",
            "column": "dim_region.region_name",
            "field_alias": "",
            "matched_by": "enum_alias",
            "allowed_sql_literals": ["华北"],
        }
    ]


def test_filter_candidate_strips_field_hint_suffix_before_validation():
    binding = asyncio.run(
        validate_binding_candidates(
            BindingCandidates(
                filter_mentions=[FilterMention(raw_text="华北地区", field_hint="地区")]
            ),
            _context(
                table_infos=_region_table(),
                retrieved_value_infos=[
                    ValueInfo(
                        id="dim_region.region_name.华北",
                        value="华北",
                        column_id="dim_region.region_name",
                    )
                ],
            ),
        )
    )

    assert binding["filters"][0]["raw_value"] == "华北"
    assert binding["filters"][0]["canonical_value"] == "华北"
    assert binding["unresolved"] == []


def test_filter_candidate_marks_unknown_value_with_field_hint():
    binding = asyncio.run(
        validate_binding_candidates(
            BindingCandidates(
                filter_mentions=[FilterMention(raw_text="火星", field_hint="区域")]
            ),
            _context(table_infos=_region_table()),
        )
    )

    assert binding["filters"] == []
    assert binding["unresolved"] == [
        {
            "type": "enum_value",
            "raw_text": "火星",
            "candidate_column": "dim_region.region_name",
            "reason": "value_not_found",
        }
    ]


def test_groupby_candidate_is_not_enum_filter():
    binding = asyncio.run(
        validate_binding_candidates(
            BindingCandidates(
                metric_mentions=[MetricMention(raw_text="GMV", normalized_text="GMV")],
                groupby_mentions=[GroupByMention(raw_text="各大区", field_hint="大区")],
            ),
            _context(
                metric_infos=[
                    {
                        "name": "GMV",
                        "alias": ["销售额"],
                        "relevant_columns": ["fact_order.order_amount"],
                    }
                ],
                table_infos=_region_table(),
            ),
        )
    )

    assert binding["metrics"][0]["canonical_metric"] == "GMV"
    assert binding["filters"] == []
    assert binding["groups"] == [
        {
            "raw_mention": "各大区",
            "column": "dim_region.region_name",
            "field_alias": "大区",
            "matched_by": "column_alias",
            "confidence": "high",
        }
    ]
    assert binding["unresolved"] == []


def test_groupby_candidate_maps_member_level_to_dimension_column():
    binding = asyncio.run(
        validate_binding_candidates(
            BindingCandidates(
                metric_mentions=[MetricMention(raw_text="销售额", normalized_text="销售额")],
                groupby_mentions=[GroupByMention(raw_text="会员等级", field_hint="会员等级")],
            ),
            _context(
                metric_infos=[
                    {
                        "name": "GMV",
                        "alias": ["销售额"],
                        "relevant_columns": ["fact_order.order_amount"],
                    }
                ],
                table_infos=_customer_table(),
            ),
        )
    )

    assert binding["groups"] == [
        {
            "raw_mention": "会员等级",
            "column": "dim_customer.member_level",
            "field_alias": "会员等级",
            "matched_by": "column_alias",
            "confidence": "high",
        }
    ]
    assert binding["unresolved"] == []


def test_business_binding_node_passes_sliding_history_to_candidate_extractor(monkeypatch):
    class MetaRepository:
        async def list_value_aliases(self):
            return []

    class Runtime:
        stream_writer = staticmethod(lambda _: None)
        context = {
            "meta_mysql_repository": MetaRepository(),
            "dw_mysql_repository": DWRepository(),
            "cost_tracker": object(),
        }

    captured = {}

    async def fake_extract_binding_candidates(query, runtime, **kwargs):
        captured.update(kwargs)
        return BindingCandidates(
            metric_mentions=[MetricMention(raw_text="销售额", normalized_text="销售额")],
            groupby_mentions=[GroupByMention(raw_text="会员等级", field_hint="会员等级")],
        )

    monkeypatch.setattr(
        business_binding_module,
        "extract_binding_candidates",
        fake_extract_binding_candidates,
    )

    result = asyncio.run(
        business_binding(
            {
                "query": "可以",
                "conversation_history": (
                    "user: 按会员等级看订单和销售额\n"
                    "assistant: “订单”指标暂未配置，但“销售额”已成功绑定，"
                    "您可以先查看按会员等级的销售额数据。"
                ),
                "metric_infos": [
                    {
                        "name": "GMV",
                        "description": "Gross Merchandise Value",
                        "alias": ["销售额"],
                        "relevant_columns": ["fact_order.order_amount"],
                    }
                ],
                "table_infos": _customer_table(),
            },
            Runtime(),
        )
    )

    assert "按会员等级看订单和销售额" in captured["conversation_history"]
    assert result["business_binding"]["groups"][0]["column"] == "dim_customer.member_level"


def test_time_candidate_parses_quarter_to_date_range():
    assert resolve_time_binding("2025 年第一季度") == {
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


def test_unknown_metric_blocking_is_based_on_candidate_not_keyword_guessing():
    binding = asyncio.run(
        validate_binding_candidates(
            BindingCandidates(
                metric_mentions=[
                    MetricMention(raw_text="品牌心智指数", normalized_text="品牌心智指数")
                ]
            ),
            _context(metric_infos=[{"name": "GMV", "alias": ["销售额"]}]),
        )
    )

    assert binding["unresolved"] == [
        {
            "type": "metric",
            "raw_text": "品牌心智指数",
            "candidate_column": "",
            "reason": "metric_not_bound",
        }
    ]


def test_mixed_known_and_unknown_metrics_keeps_unresolved_metric():
    binding = asyncio.run(
        validate_binding_candidates(
            BindingCandidates(
                metric_mentions=[
                    MetricMention(raw_text="销售额", normalized_text="销售额"),
                    MetricMention(raw_text="品牌心智指数", normalized_text="品牌心智指数"),
                ]
            ),
            _context(metric_infos=[{"name": "GMV", "alias": ["销售额"]}]),
        )
    )

    assert binding["metrics"][0]["canonical_metric"] == "GMV"
    assert binding["unresolved"] == [
        {
            "type": "metric",
            "raw_text": "品牌心智指数",
            "candidate_column": "",
            "reason": "metric_not_bound",
        }
    ]


def test_order_and_sales_amount_bind_to_order_count_and_gmv():
    bindings, issues = resolve_metric_candidates(
        BindingCandidates(
            metric_mentions=[
                MetricMention(raw_text="订单", normalized_text="订单"),
                MetricMention(raw_text="销售额", normalized_text="销售额"),
            ]
        ),
        [
            {
                "name": "ORDER_COUNT",
                "description": "订单数量，使用订单ID计数。",
                "alias": ["订单", "订单数", "订单量", "订单笔数"],
                "relevant_columns": ["fact_order.order_id"],
            },
            {
                "name": "GMV",
                "alias": ["销售额"],
                "relevant_columns": ["fact_order.order_amount"],
            },
        ],
    )

    assert issues == []
    assert [binding["canonical_metric"] for binding in bindings] == [
        "ORDER_COUNT",
        "GMV",
    ]


def test_measure_column_mentions_do_not_become_unknown_metrics_when_metric_is_bound():
    binding = asyncio.run(
        validate_binding_candidates(
            BindingCandidates(
                metric_mentions=[
                    MetricMention(raw_text="销售额", normalized_text="销售额"),
                    MetricMention(raw_text="订单数量", normalized_text="订单数量"),
                ]
            ),
            _context(
                metric_infos=[{"name": "GMV", "alias": ["销售额"]}],
                table_infos=[
                    {
                        "name": "fact_order",
                        "columns": [
                            {
                                "name": "order_quantity",
                                "role": "measure",
                                "alias": ["订单数量"],
                            }
                        ],
                    }
                ],
            ),
        )
    )

    assert binding["metrics"][0]["canonical_metric"] == "GMV"
    assert binding["unresolved"] == []


def test_extraction_failure_does_not_block_explicit_metric_fallback():
    binding = asyncio.run(
        validate_binding_candidates(
            BindingCandidates(
                metric_mentions=[MetricMention(raw_text="销售额", normalized_text="销售额")],
                extraction_failed=True,
            ),
            _context(metric_infos=[{"name": "GMV", "alias": ["销售额"]}]),
        )
    )

    assert binding["metrics"][0]["canonical_metric"] == "GMV"
    assert binding["unresolved"] == []


def test_filter_candidate_without_field_hint_is_unresolved_when_not_bound():
    binding = asyncio.run(
        validate_binding_candidates(
            BindingCandidates(filter_mentions=[FilterMention(raw_text="火星", field_hint="")]),
            _context(table_infos=_region_table()),
        )
    )

    assert binding["unresolved"] == [
        {
            "type": "enum_value",
            "raw_text": "火星",
            "candidate_column": "",
            "reason": "field_hint_missing",
        }
    ]


def test_filter_candidate_with_unknown_field_hint_can_still_use_retrieved_value():
    binding = asyncio.run(
        validate_binding_candidates(
            BindingCandidates(
                filter_mentions=[
                    FilterMention(raw_text="iPhone 15 Pro", field_hint="商品")
                ]
            ),
            _context(
                table_infos=_region_table(),
                retrieved_value_infos=[
                    ValueInfo(
                        id="dim_product.product_name.iPhone 15 Pro",
                        value="iPhone 15 Pro",
                        column_id="dim_product.product_name",
                    )
                ],
            ),
        )
    )

    assert binding["filters"] == [
        {
            "raw_value": "iPhone 15 Pro",
            "canonical_value": "iPhone 15 Pro",
            "column": "dim_product.product_name",
            "field_alias": "商品",
            "matched_by": "retrieved_value",
            "allowed_sql_literals": ["iPhone 15 Pro"],
        }
    ]
    assert binding["unresolved"] == []


def test_time_resolver_uses_source_query_when_time_mentions_are_missing():
    binding = asyncio.run(
        validate_binding_candidates(
            BindingCandidates(
                metric_mentions=[MetricMention(raw_text="GMV", normalized_text="GMV")],
                source_query="2025 年第一季度 GMV",
            ),
            _context(metric_infos=[{"name": "GMV", "alias": []}]),
        )
    )

    assert binding["time"] == {
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


def test_fallback_candidates_only_use_explicit_metadata_and_rag_hits():
    candidates = fallback_binding_candidates(
        query="统计华北地区 GMV 和 品牌心智指数",
        metric_infos=[{"name": "GMV", "alias": ["销售额"]}],
        retrieved_value_infos=[
            ValueInfo(
                id="dim_region.region_name.华北",
                value="华北",
                column_id="dim_region.region_name",
            )
        ],
        enum_aliases={},
    )

    assert [item.raw_text for item in candidates.metric_mentions] == ["GMV"]
    assert [item.raw_text for item in candidates.filter_mentions] == ["华北"]


def test_business_binding_node_uses_candidates_and_metadata(monkeypatch):
    class MetaRepository:
        async def list_value_aliases(self):
            return [
                ValueAlias(
                    column_id="dim_region.region_name",
                    alias="北方区域",
                    canonical_value="华北",
                )
            ]

    class Runtime:
        stream_writer = staticmethod(lambda _: None)
        context = {
            "meta_mysql_repository": MetaRepository(),
            "dw_mysql_repository": DWRepository(),
            "cost_tracker": object(),
        }

    async def fake_extract_binding_candidates(query, runtime, **kwargs):
        return BindingCandidates(
            metric_mentions=[MetricMention(raw_text="销售额", normalized_text="销售额")],
            filter_mentions=[FilterMention(raw_text="北方区域", field_hint="")],
        )

    monkeypatch.setattr(
        business_binding_module,
        "extract_binding_candidates",
        fake_extract_binding_candidates,
    )

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
    assert result["resolved_filters"][0]["canonical_value"] == "华北"
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
