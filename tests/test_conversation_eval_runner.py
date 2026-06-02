import asyncio

from app.evaluation.conversation_cases import (
    collect_query_events,
    evaluate_conversation_case,
    load_conversation_eval_cases,
    parse_sse_message,
)
from app.scripts.run_conversation_eval import build_eval_user_id


def test_parse_sse_message_extracts_json_payload():
    event = parse_sse_message('data: {"type": "conversation", "conversation_id": "c1"}\n\n')

    assert event == {"type": "conversation", "conversation_id": "c1"}


def test_load_conversation_eval_cases_reads_multiturn_cases():
    cases = load_conversation_eval_cases("examples/conversation_eval_cases.yaml")

    assert len(cases) == 10
    assert cases[0]["id"] == "conv_metric_and_region_followup"
    assert len(cases[0]["turns"]) == 2


def test_collect_query_events_parses_service_sse_messages():
    class FakeService:
        async def query(self, query, conversation_id=None, user_id=None, include_trace=False):
            yield 'data: {"type": "conversation", "conversation_id": "c1", "rewritten_query": "统计华北 GMV", "rewrite": {"mode": "unchanged", "standalone_query": "统计华北 GMV", "reason": "完整问题", "inherited_slots": {}, "overridden_slots": {}}}\n\n'
            yield 'data: {"type": "trace", "data": {"metric_bindings": [{"canonical_metric": "GMV"}]}}\n\n'
            yield 'data: {"type": "usage", "data": {"llm_total_tokens": 1}}\n\n'

    events = asyncio.run(
        collect_query_events(
            FakeService(),
            query="统计华北 GMV",
            conversation_id=None,
            user_id="u1",
        )
    )

    assert [event["type"] for event in events] == ["conversation", "trace", "usage"]


def test_evaluate_conversation_case_detects_snapshot_not_written():
    class FakeService:
        async def query(self, query, conversation_id=None, user_id=None, include_trace=False):
            yield 'data: {"type": "conversation", "conversation_id": "c1", "rewritten_query": "统计华北地区 GMV", "rewrite": {"mode": "unchanged", "standalone_query": "统计华北地区 GMV", "reason": "完整问题", "inherited_slots": {}, "overridden_slots": {}}}\n\n'
            yield 'data: {"type": "trace", "data": {"metric_bindings": [{"canonical_metric": "GMV"}], "resolved_filters": [{"canonical_value": "华北"}], "blocked_by": null, "final_answer": [{"gmv": 100}]}}\n\n'
            yield 'data: {"type": "usage", "data": {"llm_total_tokens": 1}}\n\n'

    class FakeMemoryRepository:
        async def get_snapshot(self, conversation_id, user_id):
            return None

    case = {
        "id": "conv_missing_snapshot",
        "turns": [
            {
                "query": "统计华北地区 GMV",
                "expected_conversation_id": "same",
                "expected_rewritten": {"mode": "unchanged", "contains": ["华北", "GMV"]},
                "expected_memory": {
                    "snapshot_write": True,
                    "snapshot_metric": "GMV",
                    "snapshot_filters": ["华北"],
                },
                "expected_trace": {
                    "metric_bindings": ["GMV"],
                    "resolved_filters": ["华北"],
                    "blocked_by": None,
                    "final_answer": "required",
                },
            }
        ],
    }

    result = asyncio.run(
        evaluate_conversation_case(case, FakeService(), FakeMemoryRepository())
    )

    assert not result["passed"]
    assert result["failures"][0]["code"] == "snapshot_not_written"


def test_evaluate_conversation_case_accepts_snapshot_unchanged_when_write_disabled():
    class FakeService:
        async def query(self, query, conversation_id=None, user_id=None, include_trace=False):
            yield 'data: {"type": "conversation", "conversation_id": "c1", "rewritten_query": "查询华北地区订单明细和用户手机号", "rewrite": {"mode": "unchanged", "standalone_query": "查询华北地区订单明细和用户手机号", "reason": "完整问题", "inherited_slots": {}, "overridden_slots": {}}}\n\n'
            yield 'data: {"type": "trace", "data": {"blocked_by": "semantic_guard", "final_answer": null}}\n\n'
            yield 'data: {"type": "usage", "data": {"llm_total_tokens": 1}}\n\n'

    class FakeMemoryRepository:
        async def get_snapshot(self, conversation_id, user_id):
            return {
                "last_metric_bindings": [{"canonical_metric": "GMV"}],
                "last_resolved_filters": [{"canonical_value": "华北"}],
                "last_time_binding": None,
                "last_sql": "select 1",
                "last_answer_summary": "返回 1 行",
                "recent_turns_summary": [],
            }

    case = {
        "id": "conv_blocked",
        "turns": [
                {
                    "query": "查询华北地区订单明细和用户手机号",
                    "supplied_conversation_id": "c1",
                    "expected_conversation_id": "same",
                "expected_rewritten": {"mode": "unchanged", "contains": ["订单明细"]},
                "expected_memory": {
                    "snapshot_write": False,
                    "snapshot_unchanged_from_turn": 1,
                },
                "expected_trace": {
                    "blocked_by": "semantic_guard",
                    "final_answer": "absent",
                },
            }
        ],
    }

    result = asyncio.run(
        evaluate_conversation_case(case, FakeService(), FakeMemoryRepository())
    )

    assert result["passed"]


def test_evaluate_conversation_case_fails_when_unchanged_query_is_rewritten():
    class FakeService:
        async def query(
            self, query, conversation_id=None, user_id=None, include_trace=False
        ):
            yield 'data: {"type": "conversation", "conversation_id": "c1", "rewritten_query": "统计华北地区 GMV", "rewrite": {"mode": "rewritten", "standalone_query": "统计华北地区 GMV", "reason": "错误继承上下文", "inherited_slots": {"metric": ["GMV"]}, "overridden_slots": {}}}\n\n'
            yield 'data: {"type": "trace", "data": {"blocked_by": null, "final_answer": [{"销量": 1}]}}\n\n'

    class FakeMemoryRepository:
        async def get_snapshot(self, conversation_id, user_id):
            return None

    case = {
        "id": "conv_rewrite_mode",
        "turns": [
            {
                "query": "统计各品类销量",
                "expected_conversation_id": "same",
                "expected_rewritten": {
                    "mode": "unchanged",
                    "contains": ["各品类", "销量"],
                },
                "expected_memory": {"snapshot_write": False},
                "expected_trace": {"blocked_by": None, "final_answer": "required"},
            }
        ],
    }

    result = asyncio.run(
        evaluate_conversation_case(case, FakeService(), FakeMemoryRepository())
    )

    assert not result["passed"]
    assert any(
        failure["code"] == "rewritten_query_changed" for failure in result["failures"]
    )


def test_evaluate_conversation_case_fails_when_rewritten_query_is_unchanged():
    class FakeService:
        async def query(
            self, query, conversation_id=None, user_id=None, include_trace=False
        ):
            yield 'data: {"type": "conversation", "conversation_id": "c1", "rewritten_query": "那华东呢", "rewrite": {"mode": "rewritten", "standalone_query": "那华东呢", "reason": "错误未改写", "inherited_slots": {}, "overridden_slots": {}}}\n\n'
            yield 'data: {"type": "trace", "data": {"blocked_by": null, "final_answer": [{"gmv": 1}]}}\n\n'

    class FakeMemoryRepository:
        async def get_snapshot(self, conversation_id, user_id):
            return {"last_metric_bindings": [{"canonical_metric": "GMV"}]}

    case = {
        "id": "conv_contextualized_mode",
        "turns": [
            {
                "query": "那华东呢",
                "supplied_conversation_id": "c1",
                "expected_conversation_id": "same",
                "expected_rewritten": {
                    "mode": "rewritten",
                    "contains": ["那华东呢"],
                },
                "expected_memory": {"snapshot_write": False},
                "expected_trace": {"blocked_by": None, "final_answer": "required"},
            }
        ],
    }

    result = asyncio.run(
        evaluate_conversation_case(case, FakeService(), FakeMemoryRepository())
    )

    assert not result["passed"]
    assert result["failures"][0]["code"] == "rewritten_query_not_contextualized"


def test_evaluate_conversation_case_checks_isolated_empty_snapshot_source():
    class FakeService:
        async def query(
            self, query, conversation_id=None, user_id=None, include_trace=False
        ):
            yield 'data: {"type": "conversation", "conversation_id": "c2", "rewritten_query": "那华东呢", "rewrite": {"mode": "needs_context", "standalone_query": "那华东呢", "reason": "缺少上下文", "inherited_slots": {}, "overridden_slots": {}}}\n\n'
            yield 'data: {"type": "trace", "data": {"blocked_by": "semantic_guard", "final_answer": null}}\n\n'

    class FakeMemoryRepository:
        async def get_snapshot(self, conversation_id, user_id):
            return {"last_metric_bindings": [{"canonical_metric": "GMV"}]}

    case = {
        "id": "conv_isolation",
        "turns": [
            {
                "query": "那华东呢",
                "supplied_conversation_id": "foreign",
                "expected_conversation_id": "new",
                "expected_rewritten": {"mode": "needs_context", "contains": ["那华东呢"]},
                "expected_memory": {
                    "snapshot_write": False,
                    "snapshot_source": "isolated_empty",
                },
                "expected_trace": {
                    "blocked_by": "semantic_guard",
                    "final_answer": "absent",
                },
            }
        ],
    }

    result = asyncio.run(
        evaluate_conversation_case(case, FakeService(), FakeMemoryRepository())
    )

    assert not result["passed"]
    assert result["failures"][0]["code"] == "snapshot_source_not_isolated_empty"


def test_evaluate_conversation_case_treats_time_raw_text_whitespace_as_equivalent():
    class FakeService:
        async def query(
            self, query, conversation_id=None, user_id=None, include_trace=False
        ):
            yield 'data: {"type": "conversation", "conversation_id": "c1", "rewritten_query": "统计华东地区2025年第一季度GMV", "rewrite": {"mode": "rewritten", "standalone_query": "统计华东地区2025年第一季度GMV", "reason": "补足时间和指标", "inherited_slots": {"metric": ["GMV"], "time": ["2025年第一季度"]}, "overridden_slots": {"filter": ["华东"]}}}\n\n'
            yield 'data: {"type": "trace", "data": {"metric_bindings": [{"canonical_metric": "GMV"}], "resolved_filters": [{"canonical_value": "华东"}], "time_binding": {"raw_text": "2025年第一季度", "year": 2025, "quarter": "Q1", "start_date": "2025-01-01", "end_date": "2025-03-31"}, "blocked_by": null, "final_answer": [{"GMV": 1}]}}\n\n'

    class FakeMemoryRepository:
        async def get_snapshot(self, conversation_id, user_id):
            if not conversation_id:
                return None
            return {
                "last_metric_bindings": [{"canonical_metric": "GMV"}],
                "last_resolved_filters": [{"canonical_value": "华东"}],
                "last_time_binding": {
                    "raw_text": "2025年第一季度",
                    "year": 2025,
                    "quarter": "Q1",
                    "start_date": "2025-01-01",
                    "end_date": "2025-03-31",
                },
            }

    case = {
        "id": "conv_time_spacing",
        "turns": [
            {
                "query": "那华东呢",
                "supplied_conversation_id": "c1",
                "expected_conversation_id": "same",
                "expected_rewritten": {
                    "mode": "rewritten",
                    "contains": ["2025 年第一季度", "华东", "GMV"],
                },
                "expected_memory": {
                    "snapshot_write": True,
                    "snapshot_time": {
                        "raw_text": "2025 年第一季度",
                        "year": 2025,
                        "quarter": "Q1",
                    },
                },
                "expected_trace": {
                    "metric_bindings": ["GMV"],
                    "resolved_filters": ["华东"],
                    "time_binding": {
                        "raw_text": "2025 年第一季度",
                        "year": 2025,
                        "quarter": "Q1",
                    },
                    "blocked_by": None,
                    "final_answer": "required",
                },
            }
        ],
    }

    result = asyncio.run(
        evaluate_conversation_case(case, FakeService(), FakeMemoryRepository())
    )

    assert result["passed"]


def test_evaluate_conversation_case_keeps_structured_time_fields_strict():
    class FakeService:
        async def query(
            self, query, conversation_id=None, user_id=None, include_trace=False
        ):
            yield 'data: {"type": "conversation", "conversation_id": "c1", "rewritten_query": "统计华东地区2025年第一季度GMV", "rewrite": {"mode": "rewritten", "standalone_query": "统计华东地区2025年第一季度GMV", "reason": "补足时间和指标", "inherited_slots": {}, "overridden_slots": {}}}\n\n'
            yield 'data: {"type": "trace", "data": {"time_binding": {"raw_text": "2025年第一季度", "year": 2025, "quarter": "Q1"}, "blocked_by": null, "final_answer": [{"GMV": 1}]}}\n\n'

    class FakeMemoryRepository:
        async def get_snapshot(self, conversation_id, user_id):
            if not conversation_id:
                return None
            return {
                "last_time_binding": {
                    "raw_text": "2025年第一季度",
                    "year": 2025,
                    "quarter": "Q1",
                }
            }

    case = {
        "id": "conv_time_structured_strict",
        "turns": [
            {
                "query": "那华东呢",
                "supplied_conversation_id": "c1",
                "expected_conversation_id": "same",
                "expected_rewritten": {
                    "mode": "rewritten",
                    "contains": ["2025 年第一季度"],
                },
                "expected_memory": {
                    "snapshot_write": True,
                    "snapshot_time": {"raw_text": "2025 年第一季度", "year": 2024},
                },
                "expected_trace": {
                    "time_binding": {"raw_text": "2025 年第一季度", "year": 2024},
                    "blocked_by": None,
                    "final_answer": "required",
                },
            }
        ],
    }

    result = asyncio.run(
        evaluate_conversation_case(case, FakeService(), FakeMemoryRepository())
    )

    assert not result["passed"]
    assert any(
        failure["code"] in {"time_binding_mismatch", "snapshot_time_mismatch"}
        for failure in result["failures"]
    )


def test_build_eval_user_id_includes_run_id_prefix():
    assert build_eval_user_id("2026-06-02T10-00-00") == (
        "conversation-eval:2026-06-02T10-00-00"
    )
