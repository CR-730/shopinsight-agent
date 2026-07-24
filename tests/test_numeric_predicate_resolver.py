from types import MappingProxyType

import pytest

from app.agent.semantic_planning.catalog import (
    ColumnCandidate,
    MetricCandidate,
    SemanticCandidateCatalog,
)
from app.agent.semantic_planning.draft import NumericPredicateMention
from app.agent.semantic_planning.resolvers.numeric_predicate import (
    NumericResolutionContext,
    resolve_numeric_predicate,
)


def _column(column_id: str, data_type: str) -> ColumnCandidate:
    table, name = column_id.split(".", 1)
    return ColumnCandidate(
        candidate_id=column_id,
        table=table,
        name=name,
        aliases=(),
        role="measure" if data_type != "varchar" else "dimension",
        projectable=True,
        data_type=data_type,
    )


def _catalog() -> SemanticCandidateCatalog:
    return SemanticCandidateCatalog(
        metadata_version="meta-v2",
        tables=MappingProxyType({}),
        columns=MappingProxyType(
            {
                "fact_order.order_amount": _column(
                    "fact_order.order_amount", "decimal(18,2)"
                ),
                "dim_region.region_name": _column("dim_region.region_name", "varchar"),
            }
        ),
        relationships=MappingProxyType({}),
        metrics=MappingProxyType(
            {
                "GMV": MetricCandidate(
                    candidate_id="GMV",
                    name="GMV",
                    aliases=("销售额",),
                    relevant_columns=("fact_order.order_amount",),
                    aggregation="sum",
                )
            }
        ),
        values=MappingProxyType({}),
    )


def _resolve(**changes):
    values = {
        "raw_text": "销售额大于10000元",
        "target_candidate_ids": ["GMV"],
        "operator_intent": "gt",
        "value_texts": ["10000"],
    }
    values.update(changes)
    mention = NumericPredicateMention(**values)
    return resolve_numeric_predicate(
        mention,
        NumericResolutionContext(catalog=_catalog()),
    )


def test_metric_predicate_is_having_and_column_predicate_is_where():
    metric = _resolve()
    column = _resolve(
        raw_text="订单金额大于10000元",
        target_candidate_ids=["fact_order.order_amount"],
    )

    assert metric.status == "resolved"
    assert metric.plan.target_type == "measure"
    assert metric.plan.target_id == "GMV"
    assert metric.plan.clause == "having"
    assert column.status == "resolved"
    assert column.plan.target_type == "column"
    assert column.plan.clause == "where"


@pytest.mark.parametrize("operator", ["eq", "gt", "gte", "lt", "lte"])
def test_single_boundary_numeric_operators_are_preserved(operator):
    result = _resolve(operator_intent=operator)

    assert result.status == "resolved"
    assert result.plan.operator == operator
    assert result.plan.values == ["10000"]


def test_decimal_values_are_canonicalized_and_between_boundaries_are_sorted():
    result = _resolve(
        raw_text="销售额在20.500和10.00之间",
        operator_intent="between",
        value_texts=["20.500", "10.00"],
    )

    assert result.status == "resolved"
    assert result.plan.values == ["10", "20.5"]


def test_between_requires_exactly_two_boundaries():
    result = _resolve(
        raw_text="销售额在10到20或30之间",
        operator_intent="between",
        value_texts=["10", "20", "30"],
    )

    assert result.status == "unresolved"
    assert result.issue.code == "numeric_boundary_count_invalid"


def test_ordering_operator_rejects_string_column():
    result = _resolve(
        raw_text="地区大于100",
        target_candidate_ids=["dim_region.region_name"],
        value_texts=["100"],
    )

    assert result.status == "unresolved"
    assert result.issue.code == "numeric_target_type_invalid"


def test_units_are_not_converted_without_authoritative_unit_metadata():
    ten_thousand = _resolve(
        raw_text="销售额大于1万",
        value_texts=["1万"],
    )
    percentage = _resolve(
        raw_text="销售额大于50%",
        value_texts=["50%"],
    )

    assert ten_thousand.status == "unresolved"
    assert ten_thousand.issue.code == "numeric_unit_not_declared"
    assert percentage.status == "unresolved"
    assert percentage.issue.code == "numeric_unit_not_declared"


def test_plain_numeric_span_inside_text_with_currency_unit_is_allowed():
    result = _resolve()

    assert result.status == "resolved"
    assert result.plan.values == ["10000"]


def test_invented_target_id_is_rejected():
    result = _resolve(target_candidate_ids=["metric:INVENTED"])

    assert result.status == "unresolved"
    assert result.issue.code == "invalid_candidate_id"
