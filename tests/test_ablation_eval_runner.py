import asyncio

from app.agent.retrieval_context import recall_sql_memory_context
from app.evaluation.cases import EvalCase
from app.scripts.run_ablation_eval import (
    AblationRunSpec,
    _dry_run_validation_off_case,
    _save_seed_sql_memory,
    ablation_specs,
    select_ablation_cases,
    summarize_ablation_results,
)


def test_select_ablation_cases_splits_by_tags_and_seed_positive_cases():
    cases = [
        EvalCase(id="cost", query="q1", tags=["ablation_cost"]),
        EvalCase(
            id="guard_blocked",
            query="q2",
            tags=["ablation_guard"],
            expected_blocked_by="pre_rag_guard",
        ),
        EvalCase(id="guard_positive", query="q3", tags=["ablation_guard"]),
        EvalCase(id="retrieval", query="q4", tags=["ablation_retrieval"]),
    ]

    groups = select_ablation_cases(cases)

    assert [case.id for case in groups["cost"]] == ["cost"]
    assert [case.id for case in groups["guard"]] == ["guard_blocked", "guard_positive"]
    assert [case.id for case in groups["retrieval"]] == ["retrieval"]
    assert [case.id for case in groups["seed"]] == ["cost", "guard_positive"]


def test_ablation_specs_keep_retrieval_basic_without_sql_memory_or_value_recall():
    specs = {(item.phase, item.variant): item for item in ablation_specs()}

    assert specs[("retrieval", "retrieval_basic")].ablation_options == {
        "disable_sql_memory": True,
        "disable_value_recall": True,
    }
    assert specs[("retrieval", "retrieval_full")].ablation_options == {}
    assert specs[("cost", "unoptimized")].ablation_options[
        "disable_non_sql_llm_cache"
    ] is True
    assert specs[("cost", "unoptimized")].ablation_options[
        "disable_embedding_cache"
    ] is True
    assert specs[("cost", "unoptimized")].ablation_options[
        "disable_context_compaction"
    ] is True


def test_validation_off_is_dry_run_and_does_not_emit_sql_or_usage():
    case = EvalCase(
        id="danger",
        query="删除订单",
        tags=["ablation_guard"],
        expected_blocked_by="pre_rag_guard",
    )
    spec = AblationRunSpec(
        phase="guard",
        variant="dry_run_validation_off",
        case_tag="guard",
        ablation_options={},
        dry_run_validation_off=True,
    )

    payload = _dry_run_validation_off_case(case, spec)

    assert payload["passed"] is True
    assert payload["dry_run_validation_off"] is True
    assert payload["would_continue_without_validation"] is True
    assert payload["trace"]["generated_sql"] == ""
    assert payload["usage"]["llm_total_tokens"] == 0


def test_seed_memory_only_saves_successful_bound_sql():
    class FakeMemoryRepository:
        def __init__(self):
            self.saved = []

        async def save_tool_usage(self, **kwargs):
            self.saved.append(kwargs)

    repository = FakeMemoryRepository()
    final_state = {
        "sql": "select sum(order_amount) from fact_order",
        "final_answer": [{"GMV": 1}],
        "business_binding": {
            "metrics": [
                {
                    "raw_mention": "销售额",
                    "canonical_metric": "GMV",
                    "relevant_columns": ["fact_order.order_amount"],
                }
            ]
        },
    }

    saved = asyncio.run(
        _save_seed_sql_memory(
            case=EvalCase(id="ok", query="统计销售额"),
            final_state=final_state,
            repositories={"agent_memory_repository": repository},
            user_id="ablation-seed:test",
            metadata_cache_version="meta-v1",
        )
    )

    assert saved is True
    assert repository.saved[0]["user_id"] == "ablation-seed:test"
    assert repository.saved[0]["metadata_cache_version"] == "meta-v1"
    assert repository.saved[0]["args"]["sql"] == "select sum(order_amount) from fact_order"


def test_seed_memory_rejects_unbound_successful_sql():
    class FakeMemoryRepository:
        async def save_tool_usage(self, **kwargs):  # pragma: no cover
            raise AssertionError("不应该写入长期 SQL 记忆")

    saved = asyncio.run(
        _save_seed_sql_memory(
            case=EvalCase(id="bad", query="统计数据"),
            final_state={"sql": "select 1", "final_answer": [{"x": 1}]},
            repositories={"agent_memory_repository": FakeMemoryRepository()},
            user_id="ablation-seed:test",
            metadata_cache_version="meta-v1",
        )
    )

    assert saved is False


def test_summarize_ablation_results_aggregates_cost_and_memory_hits():
    summary = summarize_ablation_results(
        [
            {
                "phase": "retrieval",
                "variant": "retrieval_full",
                "passed": True,
                "failure_stage": None,
                "latency_seconds": 2,
                "sql_memory_hit": True,
                "usage": _usage(
                    calls=[
                        {"type": "node", "step": "context_builder", "latency_ms": 10},
                        {
                            "type": "llm",
                            "step": "业务候选抽取",
                            "total_tokens": 20,
                            "latency_ms": 30,
                            "cache_hit": True,
                        },
                        {
                            "type": "embedding",
                            "step": "召回字段信息",
                            "tokens": 5,
                            "latency_ms": 7,
                            "cache_hit": False,
                        },
                    ]
                ),
            },
            {
                "phase": "retrieval",
                "variant": "retrieval_full",
                "passed": False,
                "failure_stage": "rag_recall",
                "latency_seconds": 4,
                "sql_memory_hit": False,
                "usage": _usage(llm_total_tokens=10, total_cost=0.1),
            },
        ]
    )

    item = summary["retrieval:retrieval_full"]
    assert item["total"] == 2
    assert item["pass_rate"] == 0.5
    assert item["avg_latency_seconds"] == 3
    assert item["sql_memory_hits"] == 1
    assert item["usage"]["llm_total_tokens"] == 10
    assert item["cost"]["total_cost"] == 0.1
    assert item["node_usage"]["context_builder"]["calls"] == 1
    assert item["llm_usage"]["业务候选抽取"]["cache_hits"] == 1
    assert item["embedding_usage"]["召回字段信息"]["tokens"] == 5


def test_sql_memory_recall_respects_ablation_options_and_metadata_version():
    class FakeMemoryRepository:
        def __init__(self):
            self.calls = []

        async def search_similar_usage(self, question, **kwargs):
            self.calls.append({"question": question, **kwargs})
            return []

    repository = FakeMemoryRepository()
    disabled_update = asyncio.run(
        recall_sql_memory_context(
            {"query": "统计华东销售额"},
            {
                "user_id": "ablation-seed:test",
                "metadata_cache_version": "meta-v1",
                "agent_memory_repository": repository,
                "ablation_options": {"disable_sql_memory": True},
            },
        )
    )
    enabled_update = asyncio.run(
        recall_sql_memory_context(
            {"query": "统计华东销售额"},
            {
                "user_id": "ablation-seed:test",
                "metadata_cache_version": "meta-v1",
                "agent_memory_repository": repository,
                "ablation_options": {},
            },
        )
    )

    assert disabled_update == {"sql_memory_context": ""}
    assert enabled_update == {"sql_memory_context": ""}
    assert len(repository.calls) == 1
    assert repository.calls[0]["user_id"] == "ablation-seed:test"
    assert repository.calls[0]["metadata_cache_version"] == "meta-v1"


def _usage(
    *,
    llm_total_tokens: int = 0,
    total_cost: float = 0,
    calls: list[dict] | None = None,
) -> dict:
    return {
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
        "llm_total_tokens": llm_total_tokens,
        "embedding_tokens": 0,
        "llm_cost": 0,
        "embedding_cost": 0,
        "total_cost": total_cost,
        "currency": "CNY",
        "embedding_estimated": False,
        "calls": calls or [],
    }
