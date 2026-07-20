import asyncio
from copy import deepcopy

from app.agent import context_compaction as context_module
from app.agent.context_compaction import compile_context_from_plan
from app.entities.column_info import ColumnInfo
from app.entities.metric_info import MetricInfo
from app.entities.table_info import TableInfo


def _column(table, name, role, data_type="bigint"):
    return ColumnInfo(
        id=f"{table}.{name}",
        name=name,
        type=data_type,
        role=role,
        examples=[],
        description=name,
        alias=[],
        table_id=table,
    )


class MetaRepository:
    def __init__(self, *, columns=None, tables=None, metrics=None):
        self.columns = columns or []
        self.tables = {table.id: table for table in (tables or [])}
        self.metrics = metrics or []
        self.requested_tables = []

    async def list_column_infos(self):
        return self.columns

    async def list_metric_infos(self):
        return self.metrics

    async def get_table_info_by_id(self, table_id):
        self.requested_tables.append(table_id)
        return self.tables.get(table_id)


def _repository():
    return MetaRepository(
        columns=[
            _column("fact_order", "order_amount", "measure", "decimal"),
            _column("fact_order", "product_id", "foreign_key"),
            _column("dim_product", "product_id", "primary_key"),
            _column("dim_product", "product_name", "dimension", "varchar"),
            _column("dim_customer", "customer_name", "dimension", "varchar"),
        ],
        tables=[
            TableInfo("fact_order", "fact_order", "fact", "订单"),
            TableInfo("dim_product", "dim_product", "dim", "商品"),
            TableInfo("dim_customer", "dim_customer", "dim", "客户"),
        ],
        metrics=[
            MetricInfo(
                id="GMV",
                name="GMV",
                description="成交金额总和",
                relevant_columns=["fact_order.order_amount"],
                alias=["销售额"],
                aggregation="sum",
                expression=None,
            )
        ],
    )


def _state(**plan_changes):
    plan = {
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
                "column_id": "dim_product.product_name",
                "role": "group_by",
                "output_alias": "商品",
            }
        ],
        "predicates": [],
        "order_by": [],
        "limit": None,
        "joins": [
            {
                "left_column_id": "dim_product.product_id",
                "right_column_id": "fact_order.product_id",
                "join_type": "inner",
            }
        ],
        "required_table_ids": ["dim_product", "fact_order"],
        "required_column_ids": [
            "dim_product.product_id",
            "dim_product.product_name",
            "fact_order.order_amount",
            "fact_order.product_id",
        ],
        "provenance": [],
    }
    plan.update(plan_changes)
    return {
        "semantic_plan": plan,
        "sql_context": {
            "tables": [
                {
                    "name": "dim_customer",
                    "columns": [{"name": "customer_name"}],
                }
            ],
            "metrics": [{"name": "UNTRUSTED"}],
        },
    }


def _column_ids(table_infos):
    return {
        f"{table['name']}.{column['name']}"
        for table in table_infos
        for column in table["columns"]
    }


def test_context_compiler_keeps_only_plan_required_and_join_columns():
    repository = _repository()

    result = asyncio.run(
        compile_context_from_plan(
            _state(), {"meta_mysql_repository": repository}
        )
    )

    assert _column_ids(result["table_infos"]) == {
        "fact_order.order_amount",
        "fact_order.product_id",
        "dim_product.product_id",
        "dim_product.product_name",
    }
    assert {table["name"] for table in result["table_infos"]} == {
        "fact_order",
        "dim_product",
    }
    assert result["metric_infos"] == [
        {
            "id": "GMV",
            "name": "GMV",
            "description": "成交金额总和",
            "relevant_columns": ["fact_order.order_amount"],
            "alias": ["销售额"],
            "aggregation": "sum",
            "expression": None,
        }
    ]


def test_context_compiler_loads_missing_required_tables_from_meta():
    repository = _repository()
    result = asyncio.run(
        compile_context_from_plan(
            _state(),
            {"meta_mysql_repository": repository},
        )
    )

    assert repository.requested_tables == ["dim_product", "fact_order"]
    assert {table["name"] for table in result["table_infos"]} == {
        "fact_order",
        "dim_product",
    }


def test_context_compiler_does_not_recompute_join_closure():
    assert not hasattr(context_module, "find_unique_shortest_join_closure")

    result = asyncio.run(
        compile_context_from_plan(
            _state(), {"meta_mysql_repository": _repository()}
        )
    )

    assert "issue" not in result


def test_context_compiler_does_not_mutate_plan_or_input_context():
    state = _state()
    original = deepcopy(state)

    asyncio.run(
        compile_context_from_plan(
            state, {"meta_mysql_repository": _repository()}
        )
    )

    assert state == original


def test_context_compiler_reports_system_failure_for_missing_column():
    repository = _repository()
    repository.columns = [
        column
        for column in repository.columns
        if column.id != "dim_product.product_name"
    ]

    result = asyncio.run(
        compile_context_from_plan(
            _state(), {"meta_mysql_repository": repository}
        )
    )

    assert result["issue"] == {
        "category": "system",
        "type": "column",
        "reason": "metadata_column_not_found",
        "candidate_ids": ["dim_product.product_name"],
    }


def test_context_compiler_reports_system_failure_for_missing_join_endpoint():
    state = _state()
    state["semantic_plan"]["joins"][0]["left_column_id"] = (
        "dim_product.ghost_id"
    )

    result = asyncio.run(
        compile_context_from_plan(
            state, {"meta_mysql_repository": _repository()}
        )
    )

    assert result["issue"]["reason"] == "metadata_column_not_found"
    assert result["issue"]["candidate_ids"] == ["dim_product.ghost_id"]
