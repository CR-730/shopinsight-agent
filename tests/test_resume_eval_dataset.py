from pathlib import Path

import sqlglot
import yaml
from sqlglot import exp

from app.evaluation.cases import load_eval_cases

DATASET_PATH = Path("examples/eval_resume_80.yaml")
ALLOWED_METRICS = {
    "GMV",
    "ORDER_COUNT",
    "AOV",
    "TOTAL_QUANTITY",
    "AVG_QUANTITY",
    "CUSTOMER_COUNT",
    "PRODUCT_COUNT",
    "MAX_ORDER_AMOUNT",
    "MIN_ORDER_AMOUNT",
    "AVG_ITEM_PRICE",
}


def _load_cases() -> list[dict]:
    return yaml.safe_load(DATASET_PATH.read_text(encoding="utf-8"))


def _referenced_columns(sql: str) -> set[str]:
    statement = sqlglot.parse_one(sql, read="mysql")
    aliases = {
        table.alias_or_name: table.name for table in statement.find_all(exp.Table)
    }
    tables = set(aliases.values())
    output_aliases = {
        projection.alias for projection in statement.expressions if projection.alias
    }
    result: set[str] = set()
    for column in statement.find_all(exp.Column):
        if column.table:
            table = aliases.get(column.table, column.table)
        elif column.name in output_aliases:
            continue
        elif len(tables) == 1:
            table = next(iter(tables))
        else:
            raise AssertionError(f"多表 SQL 中存在未限定字段：{sql}")
        result.add(f"{table}.{column.name}")
    return result


def _metric_contract(sql: str) -> tuple[set[str], list[str]]:
    statement = sqlglot.parse_one(sql, read="mysql")
    metrics: set[str] = set()
    unsupported: list[str] = []
    simple_metrics = {
        (exp.Sum, "order_amount"): "GMV",
        (exp.Sum, "order_quantity"): "TOTAL_QUANTITY",
        (exp.Avg, "order_amount"): "AOV",
        (exp.Avg, "order_quantity"): "AVG_QUANTITY",
        (exp.Max, "order_amount"): "MAX_ORDER_AMOUNT",
        (exp.Min, "order_amount"): "MIN_ORDER_AMOUNT",
    }
    distinct_metrics = {
        "order_id": "ORDER_COUNT",
        "customer_id": "CUSTOMER_COUNT",
        "product_id": "PRODUCT_COUNT",
    }
    for projection in statement.expressions:
        expression = (
            projection.this if isinstance(projection, exp.Alias) else projection
        )
        if not list(expression.find_all(exp.AggFunc)):
            continue
        columns = list(expression.find_all(exp.Column))
        if isinstance(expression, (exp.Sum, exp.Avg, exp.Max, exp.Min)):
            metric = (
                simple_metrics.get((type(expression), columns[0].name))
                if len(columns) == 1
                else None
            )
            if metric:
                metrics.add(metric)
            else:
                unsupported.append(expression.sql())
        elif isinstance(expression, exp.Count) and isinstance(
            expression.this, exp.Distinct
        ):
            metric = (
                distinct_metrics.get(columns[0].name) if len(columns) == 1 else None
            )
            if metric:
                metrics.add(metric)
            else:
                unsupported.append(expression.sql())
        elif isinstance(expression, exp.Div):
            normalized = expression.sql(dialect="mysql").replace(" ", "").lower()
            if (
                "sum(" in normalized
                and "order_amount" in normalized
                and "order_quantity" in normalized
                and "nullif(" in normalized
            ):
                metrics.add("AVG_ITEM_PRICE")
            else:
                unsupported.append(expression.sql())
        else:
            unsupported.append(expression.sql())
    return metrics, unsupported


def test_resume_dataset_has_seventy_seven_answerable_and_three_safety_cases():
    cases = _load_cases()
    safety_ids = {
        "r33_customer_region_avg_quantity",
        "r36_product_customer_quantity",
        "r62_march_product_customer_item_price",
    }
    safety_cases = [case for case in cases if case["id"] in safety_ids]

    assert len(cases) == 80
    assert len({case["id"] for case in cases}) == 80
    assert len({case["query"] for case in cases}) == 80
    assert all(case.get("oracle_sql") for case in cases)
    assert len(safety_cases) == 3
    assert all(case["suite"] == "safety" for case in safety_cases)
    assert all(case["expected_blocked_by"] == "sql_executor" for case in safety_cases)
    assert all("privacy" in case["tags"] for case in safety_cases)
    assert all(
        not case.get("expected_blocked_by")
        for case in cases
        if case["id"] not in safety_ids
    )
    assert all(not case.get("expected_planning_issue") for case in cases)


def test_resume_dataset_oracles_match_declared_retrieval_ground_truth():
    for case in _load_cases():
        statements = sqlglot.parse(case["oracle_sql"], read="mysql")
        assert len(statements) == 1, case["id"]
        assert isinstance(statements[0], exp.Select), case["id"]

        referenced = _referenced_columns(case["oracle_sql"])
        assert referenced == set(case["expected_columns"]), case["id"]

        metrics, unsupported = _metric_contract(case["oracle_sql"])
        assert not unsupported, (case["id"], unsupported)
        assert metrics == set(case["expected_metrics"]), case["id"]
        assert metrics <= ALLOWED_METRICS


def test_resume_dataset_value_labels_are_scoped_and_representative():
    cases = _load_cases()
    value_cases = [case for case in cases if case.get("expected_value_bindings")]

    assert len(value_cases) >= 30
    for case in value_cases:
        sql = case["oracle_sql"]
        for binding in case["expected_value_bindings"]:
            assert binding["column_id"] in case["expected_columns"], case["id"]
            assert f"'{binding['value']}'" in sql, case["id"]


def test_resume_dataset_separates_retrieval_fields_from_physical_join_closure():
    structural_columns = {
        "fact_order.customer_id",
        "fact_order.product_id",
        "fact_order.date_id",
        "fact_order.region_id",
        "dim_customer.customer_id",
        "dim_product.product_id",
        "dim_region.region_id",
    }
    cases = load_eval_cases(DATASET_PATH)

    for case in cases:
        assert case.expected_retrieved_columns or case.expected_metrics, case.id
        assert set(case.expected_retrieved_columns) <= set(case.expected_columns), (
            case.id
        )
        assert not (set(case.expected_retrieved_columns) & structural_columns), case.id


def test_resume_dataset_contains_realistic_difficulty_and_query_mix():
    cases = _load_cases()
    difficulties = {
        level: sum(case["difficulty"] == level for case in cases)
        for level in ("easy", "medium", "hard")
    }

    assert difficulties["easy"] >= 8
    assert difficulties["medium"] >= 20
    assert difficulties["hard"] >= 30
    assert sum("multi_filter" in case.get("tags", []) for case in cases) >= 5
    assert sum("multi_metric" in case.get("tags", []) for case in cases) >= 5
    assert sum("time" in case.get("tags", []) for case in cases) >= 30


def test_resume_dataset_time_scope_and_oracles_are_unambiguous():
    cases = _load_cases()
    time_cases = [case for case in cases if "time" in case.get("tags", [])]

    assert all("2025年" in case["query"] for case in time_cases)
    assert len({case["oracle_sql"] for case in cases}) == len(cases)
    assert all(
        case.get("order_sensitive") is True
        for case in cases
        if "ORDER BY" in case["oracle_sql"].upper()
    )
