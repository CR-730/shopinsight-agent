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


def _context(*, measures=None, dimensions=None, trusted_sources=()):
    return OrderLimitResolutionContext(
        selected_measures=tuple(measures or [_measure()]),
        selected_dimensions=tuple(dimensions or [_dimension()]),
        trusted_sources=trusted_sources,
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
        [LimitMention(raw_text="前5个")],
        _context(trusted_sources=("销售额最高的前5个商品",)),
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
        _context(trusted_sources=("按客单价排序",)),
    )

    assert result.status == "unresolved"
    assert result.issue.code == "order_target_not_selected"


def test_limit_must_be_an_explicit_integer_in_range():
    zero = resolve_order_and_limit(
        [], [LimitMention(raw_text="0条")], _context(trusted_sources=("列出0条",))
    )
    too_many = resolve_order_and_limit(
        [],
        [LimitMention(raw_text="1001条")],
        _context(trusted_sources=("列出1001条",)),
    )
    chinese = resolve_order_and_limit(
        [],
        [LimitMention(raw_text="前五个")],
        _context(trusted_sources=("列出前五个",)),
    )

    assert zero.issue.code == "limit_out_of_range"
    assert too_many.issue.code == "limit_out_of_range"
    assert chinese.issue.code == "limit_not_explicit_integer"


def test_missing_explicit_limit_remains_none():
    result = resolve_order_and_limit(
        [], [], _context(trusted_sources=("列出所有商品",))
    )

    assert result.status == "resolved"
    assert result.order_by == []
    assert result.limit is None


def test_top_n_without_one_order_target_is_ambiguous():
    no_order = resolve_order_and_limit(
        [],
        [LimitMention(raw_text="前5个")],
        _context(trusted_sources=("前5个商品",)),
    )
    multiple_targets = resolve_order_and_limit(
        [
            OrderMention(
                raw_text="最高",
                target_candidate_ids=["GMV", "dim_product.product_name"],
                direction="desc",
            )
        ],
        [LimitMention(raw_text="前5个")],
        _context(trusted_sources=("最高的前5个商品",)),
    )

    assert no_order.status == "ambiguous"
    assert no_order.issue.code == "top_n_order_target_ambiguous"
    assert multiple_targets.status == "ambiguous"
    assert multiple_targets.issue.code == "order_target_ambiguous"


def test_plain_limit_does_not_require_an_order():
    result = resolve_order_and_limit(
        [],
        [LimitMention(raw_text="5条")],
        _context(trusted_sources=("列出5条订单",)),
    )

    assert result.status == "resolved"
    assert result.order_by == []
    assert result.limit == 5
