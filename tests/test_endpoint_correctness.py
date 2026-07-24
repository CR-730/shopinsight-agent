from app.evaluation.cases import EvalCase
from app.evaluation.endpoint_correctness import score_endpoint_result


def _case(**overrides) -> EvalCase:
    payload = {
        "id": "sales",
        "query": "统计销售额",
        "oracle_sql": "SELECT SUM(order_amount) AS sales_amount FROM fact_order",
    }
    payload.update(overrides)
    return EvalCase(**payload)


def test_endpoint_result_accepts_equivalent_values_with_different_aliases():
    score = score_endpoint_result(
        _case(),
        generated_sql="SELECT SUM(order_amount) AS GMV FROM fact_order",
        actual_rows=[{"GMV": 100}],
        oracle_rows=[{"sales_amount": 100.0}],
    )

    assert score.correct is True
    assert score.reason == "result_match"


def test_expected_safety_block_is_scored_as_endpoint_correctness():
    case = _case(
        id="private-customer-name",
        expected_blocked_by="sql_executor",
    )

    blocked = score_endpoint_result(
        case,
        generated_sql="SELECT customer_name FROM dim_customer",
        actual_rows=None,
        oracle_rows=[],
        blocked_by="sql_executor",
    )
    unblocked = score_endpoint_result(
        case,
        generated_sql="SELECT customer_name FROM dim_customer",
        actual_rows=[{"customer_name": "张三"}],
        oracle_rows=[],
        blocked_by=None,
    )

    assert blocked.correct is True
    assert blocked.reason == "expected_safety_block"
    assert unblocked.correct is False
    assert unblocked.reason == "missing_expected_block"


def test_coincidental_result_rejects_wrong_metric_formula():
    case = _case(
        expected_metrics=["AOV"],
        oracle_sql=(
            "SELECT member_level, AVG(order_amount) AS average_order_value "
            "FROM fact_order GROUP BY member_level"
        ),
    )

    score = score_endpoint_result(
        case,
        generated_sql=(
            "SELECT member_level, "
            "SUM(order_amount) / COUNT(DISTINCT order_id) AS average_order_value "
            "FROM fact_order GROUP BY member_level"
        ),
        actual_rows=[{"member_level": "白银", "average_order_value": 8999}],
        oracle_rows=[{"member_level": "白银", "average_order_value": 8999}],
    )

    assert score.correct is False
    assert score.reason == "metric_formula_mismatch"


def test_empty_result_rejects_semantically_different_sql():
    case = _case(
        oracle_sql=(
            "SELECT SUM(fo.order_amount) AS sales_amount "
            "FROM fact_order fo "
            "WHERE fo.date_id BETWEEN 20250101 AND 20250131"
        )
    )

    score = score_endpoint_result(
        case,
        generated_sql=(
            "SELECT SUM(order_amount) AS sales_amount "
            "FROM fact_order "
            "WHERE date_id BETWEEN 20250201 AND 20250228"
        ),
        actual_rows=[{"sales_amount": None}],
        oracle_rows=[{"sales_amount": None}],
    )

    assert score.correct is False
    assert score.reason == "empty_result_semantic_mismatch"


def test_empty_result_accepts_alias_and_condition_order_differences():
    case = _case(
        oracle_sql=(
            "SELECT SUM(fo.order_amount) AS sales_amount "
            "FROM fact_order fo "
            "JOIN dim_region dr ON fo.region_id = dr.region_id "
            "WHERE fo.date_id BETWEEN 20250101 AND 20250131 "
            "AND dr.region_name = '华北'"
        )
    )

    score = score_endpoint_result(
        case,
        generated_sql=(
            "SELECT SUM(o.order_amount) AS GMV "
            "FROM fact_order o "
            "INNER JOIN dim_region r ON r.region_id = o.region_id "
            "WHERE r.region_name = '华北' "
            "AND o.date_id BETWEEN 20250101 AND 20250131"
        ),
        actual_rows=[{"GMV": None}],
        oracle_rows=[{"sales_amount": None}],
    )

    assert score.correct is True
    assert score.reason == "empty_result_semantic_match"


def test_topn_tie_accepts_an_alternative_member_at_cutoff():
    case = _case(
        id="top-brands",
        query="购买客户数最多的前3个品牌",
        oracle_sql=(
            "SELECT brand, COUNT(DISTINCT customer_id) AS customer_count "
            "FROM orders GROUP BY brand "
            "ORDER BY customer_count DESC LIMIT 3"
        ),
        order_sensitive=True,
    )
    oracle_full_rows = [
        {"brand": "A", "customer_count": 10},
        {"brand": "B", "customer_count": 8},
        {"brand": "C", "customer_count": 5},
        {"brand": "D", "customer_count": 5},
    ]

    score = score_endpoint_result(
        case,
        generated_sql=case.oracle_sql,
        actual_rows=[
            {"brand_name": "A", "customers": 10},
            {"brand_name": "B", "customers": 8},
            {"brand_name": "D", "customers": 5},
        ],
        oracle_rows=oracle_full_rows[:3],
        oracle_full_rows=oracle_full_rows,
    )

    assert score.correct is True
    assert score.reason == "topn_tie_match"


def test_topn_tie_rejects_value_below_cutoff():
    case = _case(
        id="top-brands",
        query="购买客户数最多的前3个品牌",
        oracle_sql=(
            "SELECT brand, COUNT(DISTINCT customer_id) AS customer_count "
            "FROM orders GROUP BY brand "
            "ORDER BY customer_count DESC LIMIT 3"
        ),
        order_sensitive=True,
    )
    oracle_full_rows = [
        {"brand": "A", "customer_count": 10},
        {"brand": "B", "customer_count": 8},
        {"brand": "C", "customer_count": 5},
        {"brand": "D", "customer_count": 5},
        {"brand": "E", "customer_count": 4},
    ]

    score = score_endpoint_result(
        case,
        generated_sql=case.oracle_sql,
        actual_rows=[
            {"brand": "A", "customer_count": 10},
            {"brand": "B", "customer_count": 8},
            {"brand": "E", "customer_count": 4},
        ],
        oracle_rows=oracle_full_rows[:3],
        oracle_full_rows=oracle_full_rows,
    )

    assert score.correct is False
    assert score.reason == "result_mismatch"
