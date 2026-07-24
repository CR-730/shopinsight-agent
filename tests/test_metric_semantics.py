import asyncio

import pytest
from omegaconf import OmegaConf

from app.entities.metric_info import MetricInfo
from app.models.metric_info import MetricInfoMySQL
from app.repositories.mysql.meta.mappers.metric_info_mapper import MetricInfoMapper
from app.repositories.mysql.meta.meta_mysql_repository import MetaMySQLRepository
from app.services.metric_definition_validator import validate_metric_definition


def metric(**changes):
    values = {
        "id": "GMV",
        "name": "GMV",
        "description": "成交金额总和",
        "relevant_columns": ["fact_order.order_amount"],
        "alias": ["销售额"],
        "aggregation": "sum",
        "expression": None,
    }
    values.update(changes)
    return MetricInfo(**values)


def test_normal_aggregation_accepts_authoritative_column():
    validate_metric_definition(metric())


def test_unknown_aggregation_is_rejected():
    with pytest.raises(ValueError, match="unsupported_metric_aggregation"):
        validate_metric_definition(metric(aggregation="median"))


def test_normal_aggregation_rejects_expression():
    with pytest.raises(ValueError, match="expression_not_allowed"):
        validate_metric_definition(metric(expression="SUM(fact_order.order_amount)"))


def test_expression_requires_a_formula():
    with pytest.raises(ValueError, match="expression_required"):
        validate_metric_definition(metric(aggregation="expression", expression=None))


def test_metric_requires_declared_columns():
    with pytest.raises(ValueError, match="metric_columns_required"):
        validate_metric_definition(metric(relevant_columns=[]))


@pytest.mark.parametrize(
    ("expression", "error"),
    [
        ("SUM(fact_order.net_amount)", "undeclared_metric_column"),
        ("SELECT SUM(fact_order.order_amount)", "metric_expression_not_scalar"),
        ("DELETE FROM fact_order", "metric_expression_not_scalar"),
        ("CREATE TABLE leaked(id INT)", "metric_expression_not_scalar"),
        (
            "SUM((SELECT order_amount FROM fact_order))",
            "metric_expression_not_scalar",
        ),
        ("SUM(fact_order.order_amount); SELECT 1", "metric_expression_multiple"),
        (
            "SUM(fact_order.order_amount) OVER ()",
            "metric_expression_window_not_allowed",
        ),
        ("fact_order.order_amount + 1", "metric_expression_aggregate_required"),
    ],
)
def test_expression_rejects_unsafe_or_incomplete_shapes(expression: str, error: str):
    with pytest.raises(ValueError, match=error):
        validate_metric_definition(
            metric(aggregation="expression", expression=expression)
        )


def test_expression_accepts_readonly_aggregate_over_declared_columns():
    validate_metric_definition(
        metric(
            aggregation="expression",
            expression="SUM(fact_order.order_amount)",
        )
    )


def test_metric_mapper_preserves_authoritative_semantics():
    entity = MetricInfoMapper.to_entity(
        MetricInfoMySQL(
            id="GMV",
            name="GMV",
            description="销售额",
            relevant_columns=["fact_order.order_amount"],
            alias=["销售额"],
            aggregation="sum",
            expression=None,
        )
    )

    assert entity.aggregation == "sum"
    assert entity.expression is None
    model = MetricInfoMapper.to_model(entity)
    assert model.aggregation == "sum"
    assert model.expression is None


def test_mapper_fails_closed_for_pre_migration_metric_row():
    with pytest.raises(ValueError, match="metric_aggregation_missing"):
        MetricInfoMapper.to_entity(
            MetricInfoMySQL(
                id="GMV",
                name="GMV",
                description="销售额",
                relevant_columns=["fact_order.order_amount"],
                alias=["销售额"],
                aggregation=None,
                expression=None,
            )
        )


def test_project_metric_config_declares_authoritative_aggregations():
    config = OmegaConf.load("conf/meta_config.yaml")
    aggregations = {item["name"]: item["aggregation"] for item in config.metrics}

    assert aggregations == {
        "ORDER_COUNT": "count_distinct",
        "GMV": "sum",
        "AOV": "avg",
        "TOTAL_QUANTITY": "sum",
        "AVG_QUANTITY": "avg",
        "CUSTOMER_COUNT": "count_distinct",
        "PRODUCT_COUNT": "count_distinct",
        "MAX_ORDER_AMOUNT": "max",
        "MIN_ORDER_AMOUNT": "min",
        "AVG_ITEM_PRICE": "expression",
    }


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def scalars(self):
        return _Rows(
            [
                next(iter(row.values())) if isinstance(row, dict) else row
                for row in self._rows
            ]
        )

    def fetchall(self):
        return self._rows

    def all(self):
        return self._rows


class _SchemaSession:
    def __init__(self, existing):
        self.existing = set(existing)
        self.statements = []

    async def execute(self, statement):
        sql = str(statement)
        self.statements.append(sql)
        if "information_schema.columns" in sql:
            return _Rows([{"column_name": name} for name in self.existing])
        if "add column aggregation" in sql.lower():
            self.existing.add("aggregation")
        if "add column expression" in sql.lower():
            self.existing.add("expression")
        return _Rows([])


def test_metric_semantics_schema_upgrade_is_idempotent():
    session = _SchemaSession(existing=set())
    repository = MetaMySQLRepository(session)

    asyncio.run(repository.ensure_metric_semantics_schema())
    asyncio.run(repository.ensure_metric_semantics_schema())

    alter_statements = [
        sql for sql in session.statements if sql.lower().startswith("alter table")
    ]
    assert len(alter_statements) == 2
    assert any("add column aggregation" in sql.lower() for sql in alter_statements)
    assert any("add column expression" in sql.lower() for sql in alter_statements)
