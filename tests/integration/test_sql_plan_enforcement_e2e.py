import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.agent.nodes import sql_executor as node

PLAN = {
    "version": "1",
    "metadata_version": "integration",
    "measures": [
        {
            "metric_id": "GMV",
            "name": "GMV",
            "aggregation": "sum",
            "expression": None,
            "source_column_ids": ["fact_order.order_amount"],
            "output_alias": "GMV",
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
            "kind": "temporal",
            "column_id": "fact_order.date_id",
            "operator": "during",
            "start_date": "2025-01-01",
            "end_date": "2025-03-31",
            "start_date_id": 20250101,
            "end_date_id": 20250331,
            "grain": "quarter",
        },
        {
            "kind": "numeric",
            "target_type": "measure",
            "target_id": "GMV",
            "operator": "gt",
            "values": ["10000"],
            "clause": "having",
        },
    ],
    "order_by": [{"target_type": "measure", "target_id": "GMV", "direction": "desc"}],
    "limit": 5,
    "joins": [
        {
            "left_column_id": "fact_order.region_id",
            "right_column_id": "dim_region.region_id",
            "join_type": "inner",
        }
    ],
    "required_table_ids": ["fact_order", "dim_region"],
    "required_column_ids": [
        "fact_order.order_amount",
        "fact_order.date_id",
        "fact_order.region_id",
        "dim_region.region_id",
        "dim_region.region_name",
    ],
    "provenance": [],
}

GOOD_SQL = """
SELECT dr.region_name AS 地区, SUM(fo.order_amount) AS GMV
FROM fact_order AS fo
JOIN dim_region AS dr ON fo.region_id = dr.region_id
WHERE fo.date_id BETWEEN 20250101 AND 20250331
GROUP BY dr.region_name
HAVING SUM(fo.order_amount) > 10000
ORDER BY GMV DESC
LIMIT 5
"""


class SpyDWRepository:
    def __init__(self):
        self.validate_calls = []
        self.run_calls = []

    async def validate(self, sql):
        self.validate_calls.append(sql)

    async def run(self, sql):
        self.run_calls.append(sql)
        return []


@pytest.mark.parametrize(
    "sql",
    [
        GOOD_SQL.replace(
            "WHERE fo.date_id BETWEEN 20250101 AND 20250331\n",
            "",
        ),
        GOOD_SQL.replace("ORDER BY GMV DESC", "ORDER BY GMV ASC"),
        GOOD_SQL.replace(
            "WHERE fo.date_id",
            "WHERE fo.order_amount > 0 AND fo.date_id",
        ),
        GOOD_SQL.replace("LIMIT 5", "LIMIT 10"),
        GOOD_SQL.replace("HAVING SUM(fo.order_amount) > 10000\n", ""),
    ],
    ids=[
        "missing_time",
        "reversed_sort",
        "extra_predicate",
        "wrong_limit",
        "missing_having",
    ],
)
def test_plan_inconsistent_sql_never_reaches_explain_or_execution(monkeypatch, sql):
    repository = SpyDWRepository()
    monkeypatch.setattr(node.app_config.agent, "max_sql_correction_attempts", 0)

    result = asyncio.run(
        node.sql_executor(
            {
                "query": "复杂查询",
                "sql": sql,
                "semantic_plan": PLAN,
            },
            SimpleNamespace(
                context={"dw_mysql_repository": repository},
                stream_writer=lambda _: None,
            ),
        )
    )

    assert result["failure"]["category"] == "sql_validation"
    assert result["failure"]["code"] == "correction_exhausted"
    assert repository.validate_calls == []
    assert repository.run_calls == []


def test_left_join_plan_executes_only_with_planned_preserved_side(monkeypatch):
    repository = SpyDWRepository()
    plan = deepcopy(PLAN)
    plan["joins"][0].update(join_type="left")
    sql = GOOD_SQL.replace("JOIN dim_region", "LEFT JOIN dim_region")

    async def analyze(*_args, **_kwargs):
        return ""

    monkeypatch.setattr(node, "_analyze_result", analyze)
    result = asyncio.run(
        node.sql_executor(
            {
                "query": "包括没有订单的地区",
                "sql": sql,
                "semantic_plan": plan,
            },
            SimpleNamespace(
                context={"dw_mysql_repository": repository},
                stream_writer=lambda _: None,
            ),
        )
    )

    assert result["failure"] is None
    assert repository.validate_calls
    assert repository.run_calls
