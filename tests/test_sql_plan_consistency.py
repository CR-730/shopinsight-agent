from copy import deepcopy

import pytest

from app.agent.sql.plan_consistency import validate_sql_plan_consistency

BASE_SQL = """
SELECT
  dr.region_name AS 地区,
  SUM(fo.order_amount) AS GMV
FROM fact_order AS fo
JOIN dim_region AS dr ON fo.region_id = dr.region_id
WHERE fo.date_id BETWEEN 20250101 AND 20250331
  AND dr.region_name = '华北'
GROUP BY dr.region_name
HAVING SUM(fo.order_amount) > 10000
ORDER BY GMV DESC
LIMIT 5
"""


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
                "kind": "enum",
                "column_id": "dim_region.region_name",
                "operator": "eq",
                "canonical_values": ["华北"],
                "allowed_sql_literals": ["华北"],
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
            "fact_order.date_id",
            "fact_order.order_amount",
            "fact_order.region_id",
        ],
        "provenance": [],
    }


def _codes(sql):
    return {
        difference.code
        for difference in validate_sql_plan_consistency(sql, _plan()).differences
    }


def test_accepts_exact_metric_dimension_predicates_join_order_and_limit():
    result = validate_sql_plan_consistency(BASE_SQL, _plan())

    assert result.ok is True
    assert result.differences == ()


def test_accepts_commutative_join_equality_and_table_aliases():
    sql = BASE_SQL.replace("fo.region_id = dr.region_id", "dr.region_id = fo.region_id")

    assert validate_sql_plan_consistency(sql, _plan()).ok is True


def test_accepts_explicit_inner_join_for_inner_plan():
    sql = BASE_SQL.replace("JOIN dim_region", "INNER JOIN dim_region")

    assert validate_sql_plan_consistency(sql, _plan()).ok is True


def test_rejects_left_join_for_inner_plan():
    sql = BASE_SQL.replace("JOIN dim_region", "LEFT JOIN dim_region")

    assert "join_type_mismatch" in _codes(sql)


def test_accepts_left_join_with_planned_preserved_side():
    plan = deepcopy(_plan())
    plan["joins"] = [
        {
            "left_column_id": "fact_order.region_id",
            "right_column_id": "dim_region.region_id",
            "join_type": "left",
        }
    ]
    sql = BASE_SQL.replace("JOIN dim_region", "LEFT JOIN dim_region")

    assert validate_sql_plan_consistency(sql, plan).ok is True


def test_rejects_reversed_left_join_direction():
    plan = deepcopy(_plan())
    plan["joins"] = [
        {
            "left_column_id": "fact_order.region_id",
            "right_column_id": "dim_region.region_id",
            "join_type": "left",
        }
    ]
    sql = BASE_SQL.replace(
        "FROM fact_order AS fo\nJOIN dim_region AS dr",
        "FROM dim_region AS dr\nLEFT JOIN fact_order AS fo",
    )

    result = validate_sql_plan_consistency(sql, plan)

    assert "join_direction_mismatch" in {
        difference.code for difference in result.differences
    }


@pytest.mark.parametrize("join_keyword", ["RIGHT JOIN", "FULL JOIN", "CROSS JOIN"])
def test_rejects_unsupported_join_types(join_keyword):
    if join_keyword == "CROSS JOIN":
        sql = BASE_SQL.replace(
            "JOIN dim_region AS dr ON fo.region_id = dr.region_id",
            "CROSS JOIN dim_region AS dr",
        )
    else:
        sql = BASE_SQL.replace("JOIN dim_region", f"{join_keyword} dim_region")

    assert "join_type_unsupported" in _codes(sql)


def test_accepts_closed_bounds_equivalent_to_between():
    sql = BASE_SQL.replace(
        "fo.date_id BETWEEN 20250101 AND 20250331",
        "fo.date_id >= 20250101 AND fo.date_id <= 20250331",
    )

    assert validate_sql_plan_consistency(sql, _plan()).ok is True


def test_rejects_offset_not_represented_in_plan():
    result = validate_sql_plan_consistency(BASE_SQL.rstrip() + " OFFSET 5", _plan())

    assert result.ok is False
    assert [item.code for item in result.differences] == ["offset_extra"]


def test_independent_gte_lte_predicates_are_not_coalesced():
    plan = deepcopy(_plan())
    plan.update(
        dimensions=[],
        predicates=[
            {
                "kind": "numeric",
                "target_type": "column",
                "target_id": "fact_order.order_amount",
                "operator": "gte",
                "values": ["100"],
                "clause": "where",
            },
            {
                "kind": "numeric",
                "target_type": "column",
                "target_id": "fact_order.order_amount",
                "operator": "lte",
                "values": ["1000"],
                "clause": "where",
            },
        ],
        order_by=[],
        limit=None,
        joins=[],
        required_table_ids=["fact_order"],
        required_column_ids=["fact_order.order_amount"],
    )
    sql = (
        "SELECT SUM(order_amount) AS GMV FROM fact_order "
        "WHERE order_amount >= 100 AND order_amount <= 1000"
    )

    assert validate_sql_plan_consistency(sql, plan).ok is True


def test_accepts_authoritative_expression_metric_with_table_alias():
    plan = deepcopy(_plan())
    plan["measures"][0]["aggregation"] = "expression"
    plan["measures"][0]["expression"] = "SUM(fact_order.order_amount)"

    assert validate_sql_plan_consistency(BASE_SQL, plan).ok is True


@pytest.mark.parametrize(
    ("old", "new", "code"),
    [
        (
            "SUM(fo.order_amount) AS GMV",
            "AVG(fo.order_amount) AS GMV",
            "metric_aggregation_mismatch",
        ),
        (",\n  SUM(fo.order_amount) AS GMV", "", "measure_missing"),
        ("dr.region_name AS 地区,\n  ", "", "dimension_missing"),
        (
            "SUM(fo.order_amount) AS GMV",
            "SUM(fo.order_amount) AS GMV, fo.order_amount",
            "select_item_extra",
        ),
        ("GROUP BY dr.region_name", "", "group_by_missing"),
        ("JOIN dim_region AS dr ON fo.region_id = dr.region_id", "", "join_missing"),
        (
            "fo.region_id = dr.region_id",
            "fo.order_amount = dr.region_id",
            "join_endpoint_mismatch",
        ),
        (
            "dr.region_name = '华北'",
            "dr.region_name = '华南'",
            "enum_predicate_mismatch",
        ),
        ("  AND dr.region_name = '华北'", "", "enum_predicate_missing"),
        (
            "fo.date_id BETWEEN 20250101 AND 20250331\n  AND ",
            "",
            "temporal_predicate_missing",
        ),
        ("HAVING SUM(fo.order_amount) > 10000", "", "numeric_predicate_missing"),
        ("ORDER BY GMV DESC", "ORDER BY GMV ASC", "order_direction_mismatch"),
        ("LIMIT 5", "LIMIT 10", "limit_mismatch"),
    ],
)
def test_rejects_plan_semantic_drift(old, new, code):
    assert code in _codes(BASE_SQL.replace(old, new))


def test_rejects_numeric_predicate_in_where_instead_of_having():
    sql = BASE_SQL.replace(
        "  AND dr.region_name = '华北'\nGROUP BY",
        "  AND dr.region_name = '华北'\n  AND SUM(fo.order_amount) > 10000\nGROUP BY",
    ).replace("HAVING SUM(fo.order_amount) > 10000\n", "")

    assert "numeric_clause_mismatch" in _codes(sql)


def test_rejects_extra_table_and_join():
    sql = BASE_SQL.replace(
        "JOIN dim_region AS dr ON fo.region_id = dr.region_id",
        "JOIN dim_region AS dr ON fo.region_id = dr.region_id "
        "JOIN dim_product AS dp ON fo.product_id = dp.product_id",
    )

    codes = _codes(sql)
    assert "table_extra" in codes
    assert "join_extra" in codes


def test_rejects_extra_join_condition():
    sql = BASE_SQL.replace(
        "fo.region_id = dr.region_id",
        "fo.region_id = dr.region_id AND dr.region_name <> '鍗庡崡'",
    )

    assert "join_extra" in _codes(sql)


def test_rejects_duplicate_select_alias_that_would_hide_an_extra_item():
    sql = BASE_SQL.replace(
        "SUM(fo.order_amount) AS GMV",
        "SUM(fo.order_amount) AS GMV, SUM(fo.order_amount) AS GMV",
    )

    assert "select_item_count_mismatch" in _codes(sql)


def test_rejects_extra_business_predicate():
    sql = BASE_SQL.replace(
        "dr.region_name = '华北'",
        "dr.region_name = '华北' AND dr.region_name <> '华南'",
    )

    assert "predicate_extra" in _codes(sql)


def test_invalid_or_multi_statement_sql_fails_closed():
    invalid = validate_sql_plan_consistency("SELECT (", _plan())
    multiple = validate_sql_plan_consistency("SELECT 1; SELECT 2", _plan())

    assert invalid.ok is False
    assert invalid.differences[0].code == "sql_parse_failed"
    assert multiple.ok is False
    assert multiple.differences[0].code == "sql_statement_count_invalid"
