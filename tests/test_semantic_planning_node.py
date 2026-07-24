import asyncio
import inspect
from datetime import date
from types import SimpleNamespace

from app.agent import state as state_module
from app.agent.nodes import semantic_planning as production_node
from app.agent.semantic_planning.catalog import SemanticCandidateCatalog
from app.agent.semantic_planning.draft import (
    AmbiguityReport,
    MeasureMention,
    SemanticDraft,
)
from app.agent.semantic_planning.orchestrator import build_semantic_plan
from app.entities.column_info import ColumnInfo
from app.entities.metric_info import MetricInfo


class FakeMetaRepository:
    def __init__(self, *, error=None):
        self.error = error

    async def list_column_infos(self):
        if self.error:
            raise self.error
        return [
            ColumnInfo(
                id="fact_order.order_amount",
                name="order_amount",
                type="decimal",
                role="measure",
                examples=[],
                description="订单金额",
                alias=["销售额"],
                table_id="fact_order",
            ),
            ColumnInfo(
                id="fact_order.date_id",
                name="date_id",
                type="bigint",
                role="foreign_key",
                examples=[],
                description="日期",
                alias=["日期"],
                table_id="fact_order",
            ),
        ]

    async def list_metric_infos(self):
        return [
            MetricInfo(
                id="GMV",
                name="GMV",
                description="成交金额总和",
                relevant_columns=["fact_order.order_amount"],
                alias=["销售额"],
                aggregation="sum",
                expression=None,
            )
        ]

    async def list_value_aliases(self):
        return []


class FakeDWRepository:
    async def column_value_exists(self, table, column, value):
        return False


def _state():
    return {
        "query": "统计销售额",
        "retrieval_context": {"values": []},
        "sql_context": {
            "tables": [
                {
                    "name": "fact_order",
                    "role": "fact",
                    "description": "订单事实表",
                    "columns": [
                        {"name": "order_amount"},
                        {"name": "date_id"},
                    ],
                }
            ],
            "metrics": [{"id": "GMV", "name": "GMV"}],
        },
        "trace": {"keywords": ["销售额"]},
    }


def _runtime(*, repository=None):
    return SimpleNamespace(
        context={
            "meta_mysql_repository": repository or FakeMetaRepository(),
            "dw_mysql_repository": FakeDWRepository(),
            "metadata_cache_version": "meta-v2",
            "semantic_reference_date": date(2026, 7, 19),
            "cost_tracker": object(),
        }
    )


def test_orchestrator_exposes_only_validated_plan(monkeypatch):
    calls = []

    async def interpret(query, runtime, **kwargs):
        assert isinstance(kwargs["catalog"], SemanticCandidateCatalog)
        calls.append("interpret")
        return SemanticDraft(
            measure_mentions=[MeasureMention(raw_text="销售额", candidate_ids=["GMV"])],
        )

    monkeypatch.setattr(
        "app.agent.semantic_planning.orchestrator.interpret_semantics", interpret
    )
    result = asyncio.run(build_semantic_plan(_state(), _runtime()))

    assert calls == ["interpret"]
    assert result["semantic_plan"]["measures"][0]["metric_id"] == "GMV"
    assert "semantic_draft" not in result
    assert "semantic_draft" not in result
    assert result["failure"] is None
    assert result["trace"]["keywords"] == ["销售额"]
    assert result["trace"]["planning_issues"] == []


def test_blocked_orchestrator_returns_failure_without_plan(monkeypatch):
    async def interpret(query, runtime, **kwargs):
        return SemanticDraft(
            ambiguity_reports=[
                AmbiguityReport(
                    raw_text="销售额",
                    candidate_ids=["GMV"],
                    reason="metric_intent_ambiguous",
                )
            ],
        )

    monkeypatch.setattr(
        "app.agent.semantic_planning.orchestrator.interpret_semantics", interpret
    )
    result = asyncio.run(build_semantic_plan(_state(), _runtime()))

    assert "semantic_plan" not in result
    assert result["failure"]["stage"] == "semantic_planning"
    assert result["failure"]["code"] == "metric_intent_ambiguous"
    assert result["failure"]["disposition"] == "blocked"
    assert result["trace"]["planning_issues"][0] == {
        "phase": "interpretation",
        "code": "metric_intent_ambiguous",
        "source_span": "销售额",
        "candidate_ids": ["GMV"],
    }


def test_catalog_or_repository_failure_is_failed_not_blocked():
    result = asyncio.run(
        build_semantic_plan(
            _state(),
            _runtime(repository=FakeMetaRepository(error=RuntimeError("db down"))),
        )
    )

    assert "semantic_plan" not in result
    assert result["failure"]["category"] == "system"
    assert result["failure"]["stage"] == "semantic_planning"
    assert result["failure"]["code"] == "semantic_planning_failed"
    assert result["failure"]["disposition"] == "failed"


def test_phase_three_activates_new_orchestrator_without_writing_legacy_state():
    source = inspect.getsource(production_node)

    assert "build_semantic_plan" in source
    assert "semantic_draft" not in source
    assert (
        "business_" + "binding"
    ) not in state_module.DataAgentState.__optional_keys__


def test_orchestrator_consumes_the_rewritten_query_without_history(monkeypatch):
    captured = {}

    async def interpret(query, runtime, **kwargs):
        captured["query"] = query
        captured["kwargs"] = kwargs
        return SemanticDraft(
            measure_mentions=[MeasureMention(raw_text="销售额", candidate_ids=["GMV"])],
        )

    monkeypatch.setattr(
        "app.agent.semantic_planning.orchestrator.interpret_semantics", interpret
    )
    state = _state()
    state["query"] = "统计华南地区的销售额"
    state["conversation_messages"] = [
        {"role": "user", "content": "按地区看销售额"},
        {"role": "assistant", "content": "已完成上一轮查询"},
    ]

    result = asyncio.run(build_semantic_plan(state, _runtime()))

    assert result["failure"] is None
    assert result["semantic_plan"]["measures"][0]["metric_id"] == "GMV"
    assert captured["query"] == "统计华南地区的销售额"
    assert "conversation_history" not in captured["kwargs"]


def test_production_node_streams_blocked_plan_without_legacy_state(monkeypatch):
    events = []

    async def blocked(state, runtime):
        return {
            "trace": {"planning_issues": []},
            "failure": {
                "category": "semantic_planning",
                "stage": "semantic_planning",
                "code": "metric_not_bound",
                "message": "metric_not_bound",
                "user_message": "请明确要查询的指标。",
                "disposition": "blocked",
            },
        }

    monkeypatch.setattr(production_node, "build_semantic_plan", blocked)
    result = asyncio.run(
        production_node.semantic_planning(
            {"query": "复购率"},
            SimpleNamespace(stream_writer=events.append, context={}),
        )
    )

    assert "semantic_draft" not in result
    assert result["failure"]["disposition"] == "blocked"
    assert events[0]["status"] == "running"
    assert events[-1]["status"] == "blocked"
    assert any(event.get("type") == "answer_delta" for event in events)
