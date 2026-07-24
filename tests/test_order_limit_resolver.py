from app.agent.semantic_planning.draft import LimitMention, OrderMention
from app.agent.semantic_planning.plan import DimensionPlan, MeasurePlan
from app.agent.semantic_planning.resolvers.order_limit import (
    OrderLimitResolutionContext,
    resolve_order_and_limit,
)


def _measure(metric_id: str = "GMV") -> MeasurePlan:
    return MeasurePlan(
        metric_id=metric_id,
        name=metric_id,
        aggregation="sum",
        source_column_ids=["fact_order.order_amount"],
        output_alias="销售额",
    )


def _dimension(column_id: str = "dim_product.product_name") -> DimensionPlan:
    return DimensionPlan(
        column_id=column_id,
        role="group_by",
        output_alias="商品",
    )


def _context(*, measures=None, dimensions=None):
    return OrderLimitResolutionContext(
        selected_measures=tuple(measures or [_measure()]),
        selected_dimensions=tuple(dimensions or [_dimension()]),
    )


def test_top_five_uses_explicit_desc_order_and_limit_five():
    result = resolve_order_and_limit(
        [
            OrderMention(
                raw_text="销售额最高",
                target_candidate_ids=["GMV"],
                direction="desc",
            )
        ],
        [LimitMention(raw_text="前5个", value=5)],
        _context(),
    )

    assert result.status == "resolved"
    assert result.order_by[0].target_type == "measure"
    assert result.order_by[0].target_id == "GMV"
    assert result.order_by[0].direction == "desc"
    assert result.limit == 5


def test_order_target_must_already_be_selected():
    result = resolve_order_and_limit(
        [
            OrderMention(
                raw_text="按客单价排序",
                target_candidate_ids=["AOV"],
                direction="desc",
            )
        ],
        [],
        _context(),
    )

    assert result.status == "unresolved"
    assert result.issue.code == "order_target_not_selected"


def test_limit_uses_llm_normalized_value_and_validates_range():
    top_one = resolve_order_and_limit(
        [
            OrderMention(
                raw_text="销量最高",
                target_candidate_ids=["GMV"],
                direction="desc",
            )
        ],
        [LimitMention(raw_text="最高的商品", value=1)],
        _context(),
    )
    chinese = resolve_order_and_limit(
        [],
        [LimitMention(raw_text="显示五条", value=5)],
        _context(),
    )
    zero = resolve_order_and_limit(
        [], [LimitMention(raw_text="0条", value=0)], _context()
    )
    too_many = resolve_order_and_limit(
        [],
        [LimitMention(raw_text="1001条", value=1001)],
        _context(),
    )

    assert top_one.status == "resolved"
    assert top_one.limit == 1
    assert chinese.status == "resolved"
    assert chinese.limit == 5
    assert zero.issue.code == "limit_out_of_range"
    assert too_many.issue.code == "limit_out_of_range"


def test_missing_explicit_limit_remains_none():
    result = resolve_order_and_limit([], [], _context())

    assert result.status == "resolved"
    assert result.order_by == []
    assert result.limit is None


def test_top_n_without_one_order_target_is_ambiguous():
    no_order = resolve_order_and_limit(
        [],
        [LimitMention(raw_text="前5个", value=5)],
        _context(),
    )
    multiple_targets = resolve_order_and_limit(
        [
            OrderMention(
                raw_text="最高",
                target_candidate_ids=["GMV", "dim_product.product_name"],
                direction="desc",
            )
        ],
        [LimitMention(raw_text="前5个", value=5)],
        _context(),
    )

    assert no_order.status == "ambiguous"
    assert no_order.issue.code == "top_n_order_target_ambiguous"
    assert multiple_targets.status == "ambiguous"
    assert multiple_targets.issue.code == "order_target_ambiguous"


def test_plain_limit_does_not_require_an_order():
    result = resolve_order_and_limit(
        [],
        [LimitMention(raw_text="5条", value=5)],
        _context(),
    )

    assert result.status == "resolved"
    assert result.order_by == []
    assert result.limit == 5
