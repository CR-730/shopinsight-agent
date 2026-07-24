from app.scripts.compare_endpoint_runs import summarize_endpoint_comparison


def test_summarize_endpoint_comparison_uses_endpoint_correctness_only():
    rows = [
        {
            "case_id": "r01",
            "prototype": {"correct": False, "reason": "result_mismatch"},
            "current": {"correct": True, "reason": "exact_result_match"},
        },
        {
            "case_id": "r02",
            "prototype": {"correct": True, "reason": "exact_result_match"},
            "current": {"correct": True, "reason": "exact_result_match"},
        },
    ]

    result = summarize_endpoint_comparison(rows)

    assert result["prototype"] == {
        "correct": 1,
        "total": 2,
        "accuracy": 0.5,
    }
    assert result["current"]["accuracy"] == 1.0
    assert result["improvement_percentage_points"] == 50.0
