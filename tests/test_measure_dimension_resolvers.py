from types import MappingProxyType

from app.agent.semantic_planning.catalog import (
    ColumnCandidate,
    MetricCandidate,
    SemanticCandidateCatalog,
)
from app.agent.semantic_planning.draft import DimensionMention, MeasureMention
from app.agent.semantic_planning.resolvers.dimension import (
    DimensionResolutionContext,
    resolve_dimension,
)
from app.agent.semantic_planning.resolvers.measure import (
    MeasureResolutionContext,
    resolve_measure,
)


def _column(
    column_id: str,
    *,
    role: str = "dimension",
    projectable: bool = True,
    data_type: str = "varchar",
) -> ColumnCandidate:
    table, name = column_id.split(".", 1)
    return ColumnCandidate(
        candidate_id=column_id,
        table=table,
        name=name,
        aliases=(),
        role=role,
        projectable=projectable,
        data_type=data_type,
    )


def _catalog() -> SemanticCandidateCatalog:
    columns = {
        "dim_region.region_name": _column("dim_region.region_name"),
        "dim_customer.customer_name": _column(
            "dim_customer.customer_name", projectable=False
        ),
        "fact_order.order_amount": _column(
            "fact_order.order_amount", role="measure", data_type="decimal"
        ),
        "fact_order.order_id": _column(
            "fact_order.order_id", role="primary_key", data_type="bigint"
        ),
    }
    metrics = {
        "GMV": MetricCandidate(
            candidate_id="GMV",
            name="GMV",
            aliases=("销售额",),
            relevant_columns=("fact_order.order_amount",),
            aggregation="sum",
            expression=None,
            description="成交金额总和",
        ),
        "BROKEN": MetricCandidate(
            candidate_id="BROKEN",
            name="BROKEN",
            aliases=(),
            relevant_columns=("fact_order.order_amount",),
            aggregation=None,
        ),
    }
    return SemanticCandidateCatalog(
        metadata_version="meta-v2",
        tables=MappingProxyType({}),
        columns=MappingProxyType(columns),
        relationships=MappingProxyType({}),
        metrics=MappingProxyType(metrics),
        values=MappingProxyType({}),
    )


def test_measure_copies_authoritative_definition_without_calculating_it():
    result = resolve_measure(
        MeasureMention(raw_text="销售额", candidate_ids=["GMV"]),
        MeasureResolutionContext(
            catalog=_catalog(), trusted_sources=("统计销售额",)
        ),
    )

    assert result.status == "resolved"
    assert result.plan.metric_id == "GMV"
    assert result.plan.aggregation == "sum"
    assert result.plan.expression is None
    assert result.plan.source_column_ids == ["fact_order.order_amount"]
    assert result.plan.output_alias == "销售额"


def test_measure_requires_one_authoritative_metric_definition():
    ambiguous = resolve_measure(
        MeasureMention(raw_text="销售", candidate_ids=["GMV", "BROKEN"]),
        MeasureResolutionContext(
            catalog=_catalog(), trusted_sources=("统计销售",)
        ),
    )
    missing_definition = resolve_measure(
        MeasureMention(raw_text="坏指标", candidate_ids=["BROKEN"]),
        MeasureResolutionContext(
            catalog=_catalog(), trusted_sources=("统计坏指标",)
        ),
    )

    assert ambiguous.status == "ambiguous"
    assert ambiguous.issue.code == "metric_ambiguous"
    assert missing_definition.status == "unresolved"
    assert missing_definition.issue.code == "metric_definition_missing"


def test_group_by_requires_a_dimension_column():
    result = resolve_dimension(
        DimensionMention(
            raw_text="订单金额",
            candidate_ids=["fact_order.order_amount"],
            role="group_by",
        ),
        DimensionResolutionContext(
            catalog=_catalog(), trusted_sources=("按订单金额分组",)
        ),
    )

    assert result.status == "unresolved"
    assert result.issue.code == "group_by_role_invalid"


def test_projection_accepts_projectable_keys_but_rejects_sensitive_columns():
    allowed = resolve_dimension(
        DimensionMention(
            raw_text="订单编号",
            candidate_ids=["fact_order.order_id"],
            role="projection",
        ),
        DimensionResolutionContext(
            catalog=_catalog(), trusted_sources=("列出订单编号",)
        ),
    )
    blocked = resolve_dimension(
        DimensionMention(
            raw_text="客户名称",
            candidate_ids=["dim_customer.customer_name"],
            role="projection",
        ),
        DimensionResolutionContext(
            catalog=_catalog(), trusted_sources=("列出客户名称",)
        ),
    )

    assert allowed.status == "resolved"
    assert allowed.plan.column_id == "fact_order.order_id"
    assert allowed.plan.role == "projection"
    assert blocked.status == "unresolved"
    assert blocked.issue.code == "projection_not_allowed"
