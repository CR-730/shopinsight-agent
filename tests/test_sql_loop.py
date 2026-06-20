from app.agent.sql.sql_correction import is_same_sql_after_normalization


def test_correct_sql_detects_unchanged_sql_after_formatting():
    original = "SELECT COUNT(*) AS cnt FROM fact_order"
    corrected = "```sql\nSELECT   COUNT(*) AS cnt\nFROM fact_order;\n```"

    assert is_same_sql_after_normalization(original, corrected) is True
