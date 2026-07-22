import asyncio
from types import SimpleNamespace

import pytest
import yaml

from app.agent.nodes import generate_sql as generate_sql_module


def _plan():
    return {
        "version": "1",
        "metadata_version": "meta-v2",
        "measures": [
            {
                "metric_id": "GMV",
                "name": "GMV",
                "aggregation": "sum",
                "expression": None,
                "source_column_ids": ["fact_order.order_amount"],
                "output_alias": "销售额",
            }
        ],
        "dimensions": [
            {
                "column_id": "dim_region.region_name",
                "role": "group_by",
                "output_alias": "地区",
            }
        ],
        "predicates": [
            {
                "kind": "enum",
                "column_id": "dim_region.region_name",
                "operator": "eq",
                "canonical_values": ["华北地区"],
            }
        ],
        "order_by": [
            {"target_type": "measure", "target_id": "GMV", "direction": "desc"}
        ],
        "limit": 5,
        "joins": [
            {
                "left_column_id": "dim_region.region_id",
                "right_column_id": "fact_order.region_id",
                "join_type": "inner",
            }
        ],
        "required_table_ids": ["dim_region", "fact_order"],
        "required_column_ids": [
            "dim_region.region_id",
            "dim_region.region_name",
            "fact_order.order_amount",
            "fact_order.region_id",
        ],
        "required_columns": [
            {"column_id": "dim_region.region_id", "data_type": "bigint"},
            {"column_id": "dim_region.region_name", "data_type": "varchar"},
            {"column_id": "fact_order.order_amount", "data_type": "decimal"},
            {"column_id": "fact_order.region_id", "data_type": "bigint"},
        ],
        "provenance": [],
    }


def _state():
    return {
        "query": "华北销售额最高的前5个地区",
        "semantic_plan": _plan(),
        "sql_memory_examples": [
            {"question": "历史问题", "sql": "SELECT 1", "similarity": 0.9}
        ],
        "sql_context": {
            "tables": [
                {
                    "name": "fact_order",
                    "columns": [
                        {"name": "order_amount", "type": "decimal"},
                        {"name": "region_id", "type": "bigint"},
                    ],
                },
                {
                    "name": "dim_region",
                    "columns": [
                        {"name": "region_id", "type": "bigint"},
                        {"name": "region_name", "type": "varchar"},
                    ],
                },
            ],
            "metrics": [
                {
                    "id": "GMV",
                    "name": "GMV",
                    "aggregation": "sum",
                    "expression": None,
                    "relevant_columns": ["fact_order.order_amount"],
                }
            ],
            "date": {"date": "2026-07-19"},
            "db": {"dialect": "mysql", "version": "8.0"},
        },
    }


def _runtime():
    class FakeDWRepository:
        async def get_db_info(self):
            return {"dialect": "mysql", "version": "8.0"}

    return SimpleNamespace(
        stream_writer=lambda event: None,
        context={
            "cost_tracker": object(),
            "dw_mysql_repository": FakeDWRepository(),
        },
    )


def _capture(monkeypatch):
    captured = {"calls": 0}

    async def fake_ainvoke_llm_with_usage(
        prompt,
        llm,
        parser,
        inputs,
        step,
        cost_tracker,
        timeout_seconds,
        *,
        cacheable,
    ):
        captured.update(
            {
                "calls": captured["calls"] + 1,
                "template": prompt.template,
                "inputs": inputs,
            }
        )
        return generate_sql_module.GeneratedSqlResponse(
            sql="SELECT SUM(order_amount) AS 销售额 FROM fact_order",
            explanation="按计划统计销售额。",
        )

    monkeypatch.setattr(
        generate_sql_module,
        "ainvoke_llm_with_usage",
        fake_ainvoke_llm_with_usage,
    )
    return captured


def test_generate_sql_serializes_only_the_semantic_plan(monkeypatch):
    captured = _capture(monkeypatch)

    asyncio.run(generate_sql_module.generate_sql(_state(), _runtime()))

    assert yaml.safe_load(captured["inputs"]["semantic_plan"]) == _plan()
    assert "conversation_history" not in captured["inputs"]
    assert set(captured["inputs"]) == {
        "semantic_plan",
        "sql_memory_context",
        "db_info",
        "query_for_explanation_only",
    }


def test_generate_sql_exposes_only_the_selected_physical_time_literals(monkeypatch):
    captured = _capture(monkeypatch)
    state = _state()
    state["semantic_plan"]["predicates"].append(
        {
            "kind": "temporal",
            "column_id": "fact_order.date_id",
            "operator": "during",
            "start_date": "2025-01-01",
            "end_date": "2025-03-31",
            "start_date_id": 20250101,
            "end_date_id": 20250331,
            "grain": "quarter",
        }
    )

    asyncio.run(generate_sql_module.generate_sql(state, _runtime()))

    temporal = yaml.safe_load(captured["inputs"]["semantic_plan"])["predicates"][-1]
    assert temporal["start_date_id"] == 20250101
    assert temporal["end_date_id"] == 20250331
    assert "start_date" not in temporal
    assert "end_date" not in temporal
    assert state["semantic_plan"]["predicates"][-1]["start_date"] == "2025-01-01"


def test_generate_sql_fails_closed_without_semantic_plan(monkeypatch):
    captured = _capture(monkeypatch)
    state = _state()
    state.pop("semantic_plan")

    with pytest.raises(ValueError, match="semantic_plan"):
        asyncio.run(generate_sql_module.generate_sql(state, _runtime()))

    assert captured["calls"] == 0


def test_prompt_requires_every_plan_component_and_forbids_new_constraints(monkeypatch):
    captured = _capture(monkeypatch)

    asyncio.run(generate_sql_module.generate_sql(_state(), _runtime()))

    template = captured["template"]
    for component in (
        "measures",
        "dimensions",
        "predicates",
        "joins",
        "order_by",
        "limit",
    ):
        assert component in template
    assert "唯一查询语义来源" in template
    assert "required_columns" in template
    assert "data_type" in template
    assert "不得改写、格式化或单位转换" in template
    assert "不得从用户原问题" in template
    assert "不得从历史成功 SQL" in template
    assert "聚合指标与 group_by 维度同时存在" in template
    assert "GROUP BY 不得留给修复阶段补齐" in template
    assert "join_type" in template
    assert "LEFT JOIN 的保留侧" in template
    assert "所有行级谓词" in template
    assert "外连接的保行语义优先于默认子句位置" in template
    assert "numeric 的 clause=where" not in template
    assert "RIGHT、FULL 或 CROSS JOIN" in template


def test_query_is_available_for_explanation_only(monkeypatch):
    captured = _capture(monkeypatch)

    asyncio.run(generate_sql_module.generate_sql(_state(), _runtime()))

    assert captured["inputs"]["query_for_explanation_only"] == _state()["query"]
    assert "query_for_explanation_only" in captured["template"]
    assert "仅用于 explanation" in captured["template"]


def test_generate_sql_receives_authoritative_metric_aggregation(monkeypatch):
    captured = _capture(monkeypatch)

    asyncio.run(generate_sql_module.generate_sql(_state(), _runtime()))

    semantic_plan = yaml.safe_load(captured["inputs"]["semantic_plan"])
    assert semantic_plan["measures"][0]["aggregation"] == "sum"


def test_prompt_does_not_repeat_compacted_schema_or_date_context(monkeypatch):
    captured = _capture(monkeypatch)

    asyncio.run(generate_sql_module.generate_sql(_state(), _runtime()))

    assert "{table_infos}" not in captured["template"]
    assert "{metric_infos}" not in captured["template"]
    assert "{date_info}" not in captured["template"]
