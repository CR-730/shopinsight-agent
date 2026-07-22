from app.agent.semantic_planning.plan import EnumPredicate, NumericPredicate
from app.agent.semantic_planning.predicate_normalization import (
    normalize_plan_predicates,
)


def _enum(operator, values, *, column_id="dim_region.region_name"):
    return EnumPredicate(
        column_id=column_id,
        operator=operator,
        canonical_values=list(values),
    )


def _numeric(operator, values):
    return NumericPredicate(
        target_type="column",
        target_id="fact_order.order_amount",
        operator=operator,
        values=list(values),
        clause="where",
    )


def test_merges_same_column_in_mentions_as_one_set():
    result = normalize_plan_predicates([_enum("in", ["华北"]), _enum("in", ["华南"])])

    assert result.issues == ()
    assert result.predicates == (_enum("in", ["华北", "华南"]),)


def test_merges_same_column_not_in_mentions_as_one_set():
    result = normalize_plan_predicates(
        [_enum("not_in", ["华南"]), _enum("not_in", ["华北", "华南"])]
    )

    assert result.issues == ()
    assert result.predicates == (_enum("not_in", ["华北", "华南"]),)


def test_deduplicates_identical_eq_predicates():
    result = normalize_plan_predicates([_enum("eq", ["华北"]), _enum("eq", ["华北"])])

    assert result.issues == ()
    assert result.predicates == (_enum("eq", ["华北"]),)


def test_rejects_distinct_eq_values_for_one_column():
    result = normalize_plan_predicates([_enum("eq", ["华北"]), _enum("eq", ["华南"])])

    assert result.predicates == ()
    assert [issue.code for issue in result.issues] == ["predicate_conflict"]


def test_eq_with_containing_in_set_converges_to_eq():
    result = normalize_plan_predicates(
        [_enum("eq", ["华北"]), _enum("in", ["华北", "华南"])]
    )

    assert result.issues == ()
    assert result.predicates == (_enum("eq", ["华北"]),)


def test_eq_outside_in_set_is_rejected():
    result = normalize_plan_predicates([_enum("eq", ["华北"]), _enum("in", ["华南"])])

    assert result.predicates == ()
    assert [issue.code for issue in result.issues] == ["predicate_conflict"]


def test_eq_excluded_by_negative_set_is_rejected():
    result = normalize_plan_predicates(
        [_enum("eq", ["华北"]), _enum("not_in", ["华北", "华南"])]
    )

    assert result.predicates == ()
    assert [issue.code for issue in result.issues] == ["predicate_conflict"]


def test_include_set_fully_excluded_is_rejected():
    result = normalize_plan_predicates(
        [_enum("in", ["华北", "华南"]), _enum("not_in", ["华北", "华南"])]
    )

    assert result.predicates == ()
    assert [issue.code for issue in result.issues] == ["predicate_conflict"]


def test_partially_excluded_include_set_keeps_both_constraints():
    result = normalize_plan_predicates(
        [_enum("in", ["华北", "华南"]), _enum("neq", ["华南"])]
    )

    assert result.issues == ()
    assert result.predicates == (
        _enum("in", ["华北", "华南"]),
        _enum("neq", ["华南"]),
    )


def test_does_not_merge_different_enum_columns():
    result = normalize_plan_predicates(
        [
            _enum("in", ["华北"]),
            _enum("in", ["一级"], column_id="dim_product.category_name"),
        ]
    )

    assert result.issues == ()
    assert result.predicates == (
        _enum("in", ["华北"]),
        _enum("in", ["一级"], column_id="dim_product.category_name"),
    )


def test_normalizes_and_deduplicates_decimal_literals():
    result = normalize_plan_predicates(
        [_numeric("eq", ["100.0"]), _numeric("eq", ["1E2"])]
    )

    assert result.issues == ()
    assert result.predicates == (_numeric("eq", ["100"]),)


def test_rejects_empty_numeric_interval():
    result = normalize_plan_predicates(
        [_numeric("gt", ["1000"]), _numeric("lt", ["100"])]
    )

    assert result.predicates == ()
    assert [issue.code for issue in result.issues] == ["predicate_conflict"]


def test_preserves_invalid_numeric_literal_for_existing_validator():
    predicate = _numeric("eq", ["not-a-number"])

    result = normalize_plan_predicates([predicate])

    assert result.issues == ()
    assert result.predicates == (predicate,)


def test_non_finite_numeric_bound_does_not_crash_normalization():
    predicates = [
        _numeric("gte", ["NaN"]),
        _numeric("lte", ["5"]),
    ]

    result = normalize_plan_predicates(predicates)

    assert result.issues == ()
    assert result.predicates == tuple(predicates)
