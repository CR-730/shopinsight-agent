import asyncio
import importlib
from types import SimpleNamespace

import yaml

from app.agent.nodes import sql_executor as node
from app.agent.sql.plan_consistency import (
    SqlPlanConsistencyResult,
    SqlPlanDifference,
)
from app.agent.sql.sql_guard import normalize_sql_for_execution

executor_core = importlib.import_module("app.agent.sql.sql_executor")
correction = importlib.import_module("app.agent.sql.sql_correction")


PLAN = {
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
    "dimensions": [],
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
        }
    ],
    "order_by": [{"target_type": "measure", "target_id": "GMV", "direction": "desc"}],
    "limit": 5,
    "joins": [],
    "required_table_ids": ["fact_order"],
    "required_column_ids": ["fact_order.date_id", "fact_order.order_amount"],
    "provenance": [],
}

GOOD_SQL = """
SELECT SUM(fo.order_amount) AS GMV
FROM fact_order AS fo
WHERE fo.date_id BETWEEN 20250101 AND 20250331
ORDER BY GMV DESC
LIMIT 5
"""


class RecordingRepository:
    def __init__(self, calls):
        self.calls = calls
        self.validated_sql = []
        self.ran_sql = []

    async def validate(self, sql):
        self.calls.append("explain")
        self.validated_sql.append(sql)

    async def run(self, sql):
        self.calls.append("execute")
        self.ran_sql.append(sql)
        return [{"GMV": 100}]


def _state(sql=GOOD_SQL):
    return {
        "query": "2025年第一季度销售额最高的前5项",
        "sql": sql,
        "semantic_plan": PLAN,
        "sql_context": {"tables": [{"name": "fact_order"}]},
    }


def _runtime(repository):
    return SimpleNamespace(
        context={"dw_mysql_repository": repository, "cost_tracker": object()},
        stream_writer=lambda _: None,
    )


def _install_downstream(monkeypatch, calls, *, safety_error=None):
    def structure(state, sql):
        calls.append("structure")
        return None

    def safety(state, sql):
        calls.append("safety")
        return safety_error

    async def analyze(query, rows, runtime):
        return ""

    monkeypatch.setattr(executor_core, "validate_sql_structure_semantics", structure)
    monkeypatch.setattr(executor_core, "validate_sql_before_execution", safety)
    monkeypatch.setattr(node, "_analyze_result", analyze)


def _pass_result():
    return SqlPlanConsistencyResult(ok=True, differences=())


def _mismatch_result(code="temporal_predicate_missing"):
    return SqlPlanConsistencyResult(
        ok=False,
        differences=(
            SqlPlanDifference(
                code=code,
                path="predicates.temporal",
                expected="required",
                actual=None,
            ),
        ),
    )


def test_plan_consistency_runs_before_explain_and_safety(monkeypatch):
    calls = []
    repository = RecordingRepository(calls)
    _install_downstream(monkeypatch, calls)

    def consistency(sql, plan):
        calls.append("plan_consistency")
        return _pass_result()

    monkeypatch.setattr(
        node, "validate_sql_plan_consistency", consistency, raising=False
    )

    result = asyncio.run(node.sql_executor(_state(), _runtime(repository)))

    assert result["failure"] is None
    assert calls == [
        "plan_consistency",
        "structure",
        "explain",
        "safety",
        "execute",
    ]


def test_mismatched_sql_is_never_executed_before_repair(monkeypatch):
    calls = []
    repository = RecordingRepository(calls)
    _install_downstream(monkeypatch, calls)

    def consistency(sql, plan):
        calls.append("plan_consistency")
        return _pass_result() if "BETWEEN" in sql else _mismatch_result()

    async def repair(
        state,
        context,
        validation_error,
        *,
        correction_attempts,
        max_correction_attempts,
        plan_differences,
    ):
        assert repository.validated_sql == []
        assert repository.ran_sql == []
        assert plan_differences[0]["code"] == "temporal_predicate_missing"
        return {"sql": GOOD_SQL, "attempts": 1}

    monkeypatch.setattr(
        node, "validate_sql_plan_consistency", consistency, raising=False
    )
    monkeypatch.setattr(node, "correct_sql_candidate", repair)

    bad_sql = GOOD_SQL.replace("WHERE fo.date_id BETWEEN 20250101 AND 20250331\n", "")
    result = asyncio.run(node.sql_executor(_state(bad_sql), _runtime(repository)))

    assert result["failure"] is None
    assert repository.ran_sql == [normalize_sql_for_execution(GOOD_SQL)]


def test_repaired_sql_rechecks_full_validation_chain(monkeypatch):
    calls = []
    repository = RecordingRepository(calls)
    _install_downstream(monkeypatch, calls)

    async def repair(*args, **kwargs):
        calls.append("repair")
        return {"sql": GOOD_SQL, "attempts": 1}

    monkeypatch.setattr(node, "correct_sql_candidate", repair)
    bad_sql = GOOD_SQL.replace("DESC", "ASC")

    result = asyncio.run(node.sql_executor(_state(bad_sql), _runtime(repository)))

    assert result["failure"] is None
    assert calls == ["repair", "structure", "explain", "safety", "execute"]


def test_repeated_plan_mismatch_exhausts_without_execution(monkeypatch):
    calls = []
    repository = RecordingRepository(calls)
    _install_downstream(monkeypatch, calls)
    corrections = []

    async def repair(state, context, validation_error, **kwargs):
        corrections.append(validation_error)
        return {"sql": state["sql"] + " ", "attempts": len(corrections)}

    monkeypatch.setattr(
        node,
        "validate_sql_plan_consistency",
        lambda *_: _mismatch_result(),
        raising=False,
    )
    monkeypatch.setattr(node, "correct_sql_candidate", repair)

    result = asyncio.run(node.sql_executor(_state("SELECT 1"), _runtime(repository)))

    assert result["failure"]["code"] == "correction_exhausted"
    assert len(corrections) == 2
    assert repository.validated_sql == []
    assert repository.ran_sql == []


def test_missing_time_is_repaired_before_explain(monkeypatch):
    calls = []
    repository = RecordingRepository(calls)
    _install_downstream(monkeypatch, calls)
    seen = []

    async def repair(state, context, validation_error, **kwargs):
        seen.append(kwargs["plan_differences"][0]["code"])
        return {"sql": GOOD_SQL, "attempts": 1}

    monkeypatch.setattr(node, "correct_sql_candidate", repair)
    bad_sql = GOOD_SQL.replace("WHERE fo.date_id BETWEEN 20250101 AND 20250331\n", "")

    result = asyncio.run(node.sql_executor(_state(bad_sql), _runtime(repository)))

    assert result["failure"] is None
    assert seen == ["temporal_predicate_missing"]
    assert calls == ["structure", "explain", "safety", "execute"]


def test_reversed_sort_is_repaired_before_execution(monkeypatch):
    calls = []
    repository = RecordingRepository(calls)
    _install_downstream(monkeypatch, calls)
    seen = []

    async def repair(state, context, validation_error, **kwargs):
        seen.append(kwargs["plan_differences"][0]["code"])
        return {"sql": GOOD_SQL, "attempts": 1}

    monkeypatch.setattr(node, "correct_sql_candidate", repair)

    result = asyncio.run(
        node.sql_executor(_state(GOOD_SQL.replace("DESC", "ASC")), _runtime(repository))
    )

    assert result["failure"] is None
    assert seen == ["order_direction_mismatch"]
    assert repository.ran_sql == [normalize_sql_for_execution(GOOD_SQL)]


def test_extra_offset_is_repaired_before_execution(monkeypatch):
    calls = []
    repository = RecordingRepository(calls)
    _install_downstream(monkeypatch, calls)
    seen = []

    async def repair(state, context, validation_error, **kwargs):
        seen.append(kwargs["plan_differences"][0]["code"])
        return {"sql": GOOD_SQL, "attempts": 1}

    monkeypatch.setattr(node, "correct_sql_candidate", repair)

    result = asyncio.run(
        node.sql_executor(_state(GOOD_SQL.rstrip() + " OFFSET 5"), _runtime(repository))
    )

    assert result["failure"] is None
    assert seen == ["offset_extra"]
    assert repository.ran_sql == [normalize_sql_for_execution(GOOD_SQL)]


def test_join_type_mismatch_is_repaired_before_execution(monkeypatch):
    calls = []
    repository = RecordingRepository(calls)
    _install_downstream(monkeypatch, calls)
    seen = []
    good_sql = GOOD_SQL.replace(
        "FROM fact_order AS fo",
        "FROM fact_order AS fo "
        "INNER JOIN dim_region AS dr ON fo.region_id = dr.region_id",
    )

    async def repair(state, context, validation_error, **kwargs):
        seen.append(kwargs["plan_differences"][0]["code"])
        return {"sql": good_sql, "attempts": 1}

    monkeypatch.setattr(node, "correct_sql_candidate", repair)
    bad_sql = good_sql.replace("INNER JOIN", "LEFT JOIN")
    plan = dict(PLAN)
    plan.update(
        joins=[
            {
                "left_column_id": "fact_order.region_id",
                "right_column_id": "dim_region.region_id",
                "join_type": "inner",
            }
        ],
        required_table_ids=["fact_order", "dim_region"],
        required_column_ids=[
            "fact_order.date_id",
            "fact_order.order_amount",
            "fact_order.region_id",
            "dim_region.region_id",
        ],
    )
    state = _state(bad_sql)
    state["semantic_plan"] = plan

    result = asyncio.run(node.sql_executor(state, _runtime(repository)))

    assert result["failure"] is None
    assert seen == ["join_type_mismatch"]
    assert repository.ran_sql == [normalize_sql_for_execution(good_sql)]


def test_extra_predicate_is_never_executed(monkeypatch):
    calls = []
    repository = RecordingRepository(calls)
    _install_downstream(monkeypatch, calls)
    monkeypatch.setattr(node.app_config.agent, "max_sql_correction_attempts", 0)
    sql = GOOD_SQL.replace(
        "WHERE fo.date_id",
        "WHERE fo.order_amount > 0 AND fo.date_id",
    )

    result = asyncio.run(node.sql_executor(_state(sql), _runtime(repository)))

    assert result["failure"]["code"] == "correction_exhausted"
    assert repository.validated_sql == []
    assert repository.ran_sql == []


def test_safety_block_remains_blocked_not_repairable(monkeypatch):
    calls = []
    repository = RecordingRepository(calls)
    _install_downstream(monkeypatch, calls, safety_error="blocked by safety")
    corrections = []

    async def repair(*args, **kwargs):
        corrections.append(True)
        return {"sql": GOOD_SQL, "attempts": 1}

    monkeypatch.setattr(node, "correct_sql_candidate", repair)

    result = asyncio.run(node.sql_executor(_state(), _runtime(repository)))

    assert result["failure"]["code"] == "sql_safety_blocked"
    assert corrections == []
    assert repository.ran_sql == []


def test_correction_model_receives_only_plan_tables_sql_and_differences(monkeypatch):
    captured = {}

    async def invoke(prompt, llm, parser, inputs, *args, **kwargs):
        captured["template"] = prompt.template
        captured["inputs"] = inputs
        return GOOD_SQL

    monkeypatch.setattr(correction, "repair_invalid_join_relationship", lambda *_: None)
    monkeypatch.setattr(correction, "ainvoke_llm_with_usage", invoke)
    state = _state("SELECT 1")
    state["conversation_history"] = [{"role": "user", "content": "不可信历史"}]
    difference = {
        "code": "temporal_predicate_missing",
        "path": "predicates.temporal",
        "expected": "required",
        "actual": None,
    }

    asyncio.run(
        correction.correct_sql_candidate(
            state,
            {"cost_tracker": object()},
            "missing time",
            correction_attempts=0,
            max_correction_attempts=2,
            plan_differences=[difference],
        )
    )

    assert set(captured["inputs"]) == {
        "semantic_plan",
        "table_infos",
        "sql",
        "differences",
    }
    assert yaml.safe_load(captured["inputs"]["semantic_plan"]) == PLAN
    assert yaml.safe_load(captured["inputs"]["differences"]) == [difference]
    assert "conversation_history" not in captured["inputs"]
    assert "唯一可信的业务语义来源" in captured["template"]
    assert "只修复 differences 指出的错误" in captured["template"]
    assert "join_type" in captured["template"]
    assert "LEFT JOIN 的保留侧" in captured["template"]
