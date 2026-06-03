from app.agent.sql.sql_correction import is_same_sql_after_normalization
from app.agent.sql_loop import route_after_pre_sql_execution_validation


def test_sql_validation_routes_to_execution_when_error_is_empty():
    assert (
        route_after_pre_sql_execution_validation({"error": None, "safety_error": None})
        == "pass"
    )


def test_sql_validation_routes_to_correction_before_max_attempts():
    state = {
        "error": "Unknown column",
        "correction_attempts": 1,
        "max_correction_attempts": 2,
    }

    assert route_after_pre_sql_execution_validation(state) == "repairable_error"


def test_sql_validation_routes_to_failure_at_max_attempts():
    state = {
        "error": "Unknown column",
        "correction_attempts": 2,
        "max_correction_attempts": 2,
    }

    assert route_after_pre_sql_execution_validation(state) == "fail_sql_correction"


def test_sql_validation_routes_to_blocked_on_safety_error():
    state = {
        "error": None,
        "safety_error": "SQL 访问敏感字段",
    }

    assert route_after_pre_sql_execution_validation(state) == "blocked"


def test_correct_sql_detects_unchanged_sql_after_formatting():
    original = "SELECT COUNT(*) AS cnt FROM fact_order"
    corrected = "```sql\nSELECT   COUNT(*) AS cnt\nFROM fact_order;\n```"

    assert is_same_sql_after_normalization(original, corrected) is True
