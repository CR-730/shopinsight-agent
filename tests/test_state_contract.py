from typing import get_args

from app.agent.failure import FailureCategory
from app.agent.memory import build_sql_tool_memory
from app.agent.sql_loop import route_after_safety_guard
from app.agent.state import DataAgentState, TraceState
from app.evaluation.cases import EvalCase, evaluate_case
from app.services import query_service


def test_state_does_not_expose_binding_candidates():
    assert "binding_candidates" not in DataAgentState.__optional_keys__


def test_state_exposes_plan_but_never_draft():
    assert "semantic_plan" in DataAgentState.__optional_keys__
    assert "semantic_draft" not in DataAgentState.__optional_keys__


def test_trace_can_expose_sanitized_planning_issues():
    assert "planning_issues" in TraceState.__optional_keys__


def test_runtime_state_and_failure_categories_have_no_legacy_binding_contract():
    legacy_name = "business_" + "binding"
    assert legacy_name not in DataAgentState.__optional_keys__
    assert legacy_name not in get_args(FailureCategory)


def test_query_service_does_not_run_legacy_candidate_extraction():
    assert not hasattr(query_service, "extract_binding_candidates")


def test_state_keeps_final_result_in_output_object_not_top_level_fields():
    optional_keys = DataAgentState.__optional_keys__

    assert "output" in optional_keys
    assert "final_answer" not in optional_keys
    assert "result_analysis" not in optional_keys
    assert "result_meta" not in optional_keys
    assert "sql_explanation" not in optional_keys
    assert "correction_attempts" not in optional_keys
    assert "max_correction_attempts" not in optional_keys


def test_safety_route_uses_unified_failure_state():
    state = {
        "failure": {
            "category": "input_guard",
            "stage": "pre_rag_guard",
            "code": "prompt_injection",
            "message": "检测到提示词注入",
            "disposition": "blocked",
        }
    }

    assert route_after_safety_guard(state) == "blocked"


def test_safety_route_stops_failed_system_state():
    state = {
        "failure": {
            "category": "system",
            "stage": "context_compaction",
            "code": "metadata_column_not_found",
            "message": "missing metadata",
            "disposition": "failed",
        }
    }

    assert route_after_safety_guard(state) == "blocked"


def test_sql_memory_does_not_save_state_with_unified_failure():
    state = {
        "sql": "select sum(order_amount) from fact_order",
        "output": {"rows": [{"GMV": 100}]},
        "failure": {
            "category": "sql_execution",
            "stage": "tool_execution",
            "code": "timeout",
            "message": "SQL 执行超时",
            "disposition": "failed",
        },
    }

    assert build_sql_tool_memory("统计销售额", state) is None


def test_eval_reads_block_information_from_unified_failure():
    case = EvalCase(
        id="unsafe",
        query="导出所有手机号",
        expected_blocked_by="pre_rag_guard",
    )
    state = {
        "trace": {"keywords": []},
        "failure": {
            "category": "input_guard",
            "stage": "pre_rag_guard",
            "code": "privacy_detail",
            "message": "禁止查询隐私明细",
            "disposition": "blocked",
        },
    }

    assert evaluate_case(case, state).passed is True
