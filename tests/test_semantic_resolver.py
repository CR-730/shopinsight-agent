import asyncio
from datetime import date
from types import MappingProxyType

from app.agent.semantic_planning.catalog import (
    ColumnCandidate,
    MetricCandidate,
    SemanticCandidateCatalog,
    ValueCandidate,
)
from app.agent.semantic_planning.draft import (
    AmbiguityReport,
    DimensionMention,
    EnumPredicateMention,
    LimitMention,
    MeasureMention,
    NumericPredicateMention,
    OrderMention,
    SemanticDraft,
    TemporalPredicateMention,
)
from app.agent.semantic_planning.resolver import (
    SemanticResolutionContext,
    resolve_semantic_draft,
)


class FakeDWRepository:
    async def column_value_exists(self, table, column, value):
        return False


def _column(column_id: str, role: str, data_type: str):
    table, name = column_id.split(".", 1)
    return ColumnCandidate(
        candidate_id=column_id,
        table=table,
        name=name,
        aliases=(),
        role=role,
        projectable=True,
        data_type=data_type,
    )


def _catalog():
    return SemanticCandidateCatalog(
        metadata_version="meta-v2",
        tables=MappingProxyType({}),
        columns=MappingProxyType(
            {
                "fact_order.order_amount": _column(
                    "fact_order.order_amount", "measure", "decimal"
                ),
                "fact_order.date_id": _column(
                    "fact_order.date_id", "foreign_key", "bigint"
                ),
                "dim_product.product_name": _column(
                    "dim_product.product_name", "dimension", "varchar"
                ),
                "dim_region.region_name": _column(
                    "dim_region.region_name", "dimension", "varchar"
                ),
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
        values=MappingProxyType(
            {
                "v-north": ValueCandidate(
                    candidate_id="v-north",
                    canonical_value="华北地区",
                    aliases=("华北",),
                    column_id="dim_region.region_name",
                    source="retrieval",
                )
            }
        ),
    )


def _draft(**changes):
    values = {
        "source_query": "2025年第一季度华北地区销售额最高的前5个商品",
        "measure_mentions": [
            MeasureMention(raw_text="销售额", candidate_ids=["GMV"])
        ],
        "dimension_mentions": [
            DimensionMention(
                raw_text="商品",
                candidate_ids=["dim_product.product_name"],
                role="group_by",
            )
        ],
        "predicate_mentions": [
            EnumPredicateMention(
                raw_text="华北地区",
                value_candidate_ids=["v-north"],
            ),
            TemporalPredicateMention(
                raw_text="2025年第一季度", relation_intent="during"
            ),
        ],
        "order_mentions": [
            OrderMention(
                raw_text="销售额最高",
                target_candidate_ids=["GMV"],
                direction="desc",
            )
        ],
        "limit_mentions": [LimitMention(raw_text="前5个")],
    }
    values.update(changes)
    return SemanticDraft(**values)


def _context():
    query = "2025年第一季度华北地区销售额最高的前5个商品"
    return SemanticResolutionContext(
        catalog=_catalog(),
        dw_repository=FakeDWRepository(),
        trusted_sources=(query,),
        reference_date=date(2026, 7, 19),
        temporal_column_id="fact_order.date_id",
    )


def test_resolver_builds_internal_plan_and_preserves_provenance():
    result = asyncio.run(resolve_semantic_draft(_draft(), _context()))

    assert result.status == "resolved"
    assert result.plan.version == "1"
    assert result.plan.metadata_version == "meta-v2"
    assert result.plan.measures[0].metric_id == "GMV"
    assert result.plan.dimensions[0].column_id == "dim_product.product_name"
    assert {predicate.kind for predicate in result.plan.predicates} == {
        "enum",
        "temporal",
    }
    assert result.plan.order_by[0].target_id == "GMV"
    assert result.plan.limit == 5
    assert result.plan.joins == []
    assert result.plan.required_column_ids == []
    assert {item.raw_text for item in result.plan.provenance} >= {
        "销售额",
        "商品",
        "华北地区",
        "2025年第一季度",
        "销售额最高",
        "前5个",
    }


def test_any_issue_blocks_the_entire_untrusted_plan():
    draft = _draft(
        measure_mentions=[
            MeasureMention(raw_text="销售额", candidate_ids=["invented"])
        ]
    )

    result = asyncio.run(resolve_semantic_draft(draft, _context()))

    assert result.status == "unresolved"
    assert result.plan is None
    assert result.issues[0].code == "invalid_candidate_id"


def test_missing_enum_value_candidate_blocks_the_entire_plan():
    draft = _draft(
        predicate_mentions=[
            EnumPredicateMention(
                raw_text="华北地区",
                value_candidate_ids=[],
            )
        ]
    )

    result = asyncio.run(resolve_semantic_draft(draft, _context()))

    assert result.status == "unresolved"
    assert result.plan is None
    assert result.issues[0].code == "value_not_bound"


def test_llm_ambiguity_report_blocks_without_selecting_first_candidate():
    draft = _draft(
        ambiguity_reports=[
            AmbiguityReport(
                raw_text="华南",
                candidate_ids=[
                    "dim_region.region_name",
                    "dim_product.product_name",
                ],
                reason="field_ambiguous",
            )
        ]
    )
    context = _context()
    context = SemanticResolutionContext(
        catalog=context.catalog,
        dw_repository=context.dw_repository,
        trusted_sources=(context.trusted_sources[0] + "，华南",),
        reference_date=context.reference_date,
        temporal_column_id=context.temporal_column_id,
    )

    result = asyncio.run(resolve_semantic_draft(draft, context))

    assert result.status == "ambiguous"
    assert result.plan is None
    assert result.issues[0].code == "field_ambiguous"
    assert result.issues[0].candidate_ids == [
        "dim_region.region_name",
        "dim_product.product_name",
    ]


def test_duplicate_mentions_are_stably_deduplicated():
    mention = MeasureMention(raw_text="销售额", candidate_ids=["GMV"])
    draft = _draft(measure_mentions=[mention, mention])

    result = asyncio.run(resolve_semantic_draft(draft, _context()))

    assert result.status == "resolved"
    assert len(result.plan.measures) == 1


def test_empty_business_object_is_unresolved():
    draft = SemanticDraft(source_query="你好")
    context = SemanticResolutionContext(
        catalog=_catalog(),
        dw_repository=FakeDWRepository(),
        trusted_sources=("你好",),
        reference_date=date(2026, 7, 19),
        temporal_column_id="fact_order.date_id",
    )

    result = asyncio.run(resolve_semantic_draft(draft, context))

    assert result.status == "unresolved"
    assert result.plan is None
    assert result.issues[0].code == "business_object_not_planned"


def test_multiple_independent_time_mentions_are_blocked():
    draft = _draft(
        dimension_mentions=[],
        predicate_mentions=[
            TemporalPredicateMention(raw_text="2025年", relation_intent="during"),
            TemporalPredicateMention(raw_text="2026年", relation_intent="during"),
        ],
        order_mentions=[],
        limit_mentions=[],
    )
    context = _context()
    context = SemanticResolutionContext(
        catalog=context.catalog,
        dw_repository=context.dw_repository,
        trusted_sources=("比较2025年和2026年的销售额",),
        reference_date=context.reference_date,
        temporal_column_id=context.temporal_column_id,
    )

    result = asyncio.run(resolve_semantic_draft(draft, context))

    assert result.status == "unresolved"
    assert result.plan is None
    assert result.issues[0].code == "multiple_time_turns_unsupported"


def test_numeric_metric_predicate_is_included_as_having():
    draft = _draft(
        source_query="2025年第一季度销售额大于10000元的地区",
        dimension_mentions=[
            DimensionMention(
                raw_text="地区",
                candidate_ids=["dim_region.region_name"],
                role="group_by",
            )
        ],
        predicate_mentions=[
            TemporalPredicateMention(
                raw_text="2025年第一季度", relation_intent="during"
            ),
            NumericPredicateMention(
                raw_text="销售额大于10000元",
                target_candidate_ids=["GMV"],
                operator_intent="gt",
                value_texts=["10000"],
            ),
        ],
        order_mentions=[],
        limit_mentions=[],
    )
    context = _context()
    context = SemanticResolutionContext(
        catalog=context.catalog,
        dw_repository=context.dw_repository,
        trusted_sources=(draft.source_query,),
        reference_date=context.reference_date,
        temporal_column_id=context.temporal_column_id,
    )

    result = asyncio.run(resolve_semantic_draft(draft, context))

    assert result.status == "resolved"
    numeric = next(item for item in result.plan.predicates if item.kind == "numeric")
    assert numeric.target_id == "GMV"
    assert numeric.clause == "having"
