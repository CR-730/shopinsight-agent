import asyncio
from types import SimpleNamespace

from app.agent.nodes import context_builder as context_builder_module
from app.agent.nodes import context_compaction as context_compaction_module
from app.agent.nodes import sql_executor as sql_executor_module


def test_context_builder_returns_all_inner_updates(monkeypatch):
    async def fake_recall_sql_memory_context(state, context):
        return {"sql_memory_context": "historical SQL"}

    async def fake_extract_retrieval_keywords(state):
        return {"keywords": ["GMV"]}

    async def fake_recall_column_context(state, context):
        return {"retrieved_column_infos": ["column"]}

    async def fake_recall_value_context(state, context):
        return {"retrieved_value_infos": ["value"]}

    async def fake_recall_metric_context(state, context):
        return {"retrieved_metric_infos": ["metric"]}

    async def fake_merge_retrieved_context(state, context):
        assert state["sql_memory_context"] == "historical SQL"
        assert state["keywords"] == ["GMV"]
        assert state["retrieved_column_infos"] == ["column"]
        assert state["retrieved_value_infos"] == ["value"]
        assert state["retrieved_metric_infos"] == ["metric"]
        return {"table_infos": ["table"], "metric_infos": ["metric_info"]}

    monkeypatch.setattr(
        context_builder_module,
        "recall_sql_memory_context",
        fake_recall_sql_memory_context,
    )
    monkeypatch.setattr(
        context_builder_module,
        "extract_retrieval_keywords",
        fake_extract_retrieval_keywords,
    )
    monkeypatch.setattr(
        context_builder_module, "recall_column_context", fake_recall_column_context
    )
    monkeypatch.setattr(
        context_builder_module, "recall_value_context", fake_recall_value_context
    )
    monkeypatch.setattr(
        context_builder_module, "recall_metric_context", fake_recall_metric_context
    )
    monkeypatch.setattr(
        context_builder_module, "merge_retrieved_context", fake_merge_retrieved_context
    )

    result = asyncio.run(
        context_builder_module.context_builder(
            {"query": "统计 GMV"}, SimpleNamespace(context={})
        )
    )

    assert result == {
        "sql_memory_context": "historical SQL",
        "keywords": ["GMV"],
        "retrieved_column_infos": ["column"],
        "retrieved_value_infos": ["value"],
        "retrieved_metric_infos": ["metric"],
        "table_infos": ["table"],
        "metric_infos": ["metric_info"],
    }

def test_context_compaction_returns_all_inner_updates(monkeypatch):
    async def fake_filter_table_context(state, context):
        assert state["table_infos"] == ["raw_table"]
        assert context == {"dw_mysql_repository": "repo"}
        return {"table_infos": ["filtered_table"]}

    def fake_filter_metric_context(state):
        assert state["table_infos"] == ["filtered_table"]
        return {"metric_infos": ["filtered_metric"]}

    async def fake_add_runtime_context(state, context):
        assert state["metric_infos"] == ["filtered_metric"]
        assert context == {"dw_mysql_repository": "repo"}
        return {
            "date_info": {"date": "2026-06-03", "weekday": "Wednesday", "quarter": "Q2"},
            "db_info": {"dialect": "mysql", "version": "8.0"},
        }

    monkeypatch.setattr(
        context_compaction_module, "filter_table_context", fake_filter_table_context
    )
    monkeypatch.setattr(
        context_compaction_module, "filter_metric_context", fake_filter_metric_context
    )
    monkeypatch.setattr(
        context_compaction_module, "add_runtime_context", fake_add_runtime_context
    )

    result = asyncio.run(
        context_compaction_module.context_compaction(
            {"query": "统计 GMV", "table_infos": ["raw_table"]},
            SimpleNamespace(context={"dw_mysql_repository": "repo"}),
        )
    )

    assert result == {
        "table_infos": ["filtered_table"],
        "metric_infos": ["filtered_metric"],
        "date_info": {"date": "2026-06-03", "weekday": "Wednesday", "quarter": "Q2"},
        "db_info": {"dialect": "mysql", "version": "8.0"},
    }


def test_sql_executor_returns_corrected_sql_and_execution_state(monkeypatch):
    calls = []

    async def fake_pre_validate_sql(state, executor):
        calls.append(("validate", state["sql"]))
        if len(calls) == 1:
            return {
                "sql": "select bad",
                "error": "Unknown column",
                "safety_error": None,
            }
        return {"sql": "select corrected", "error": None, "safety_error": None}

    async def fake_correct_sql_candidate(state, context):
        assert state["sql"] == "select bad"
        assert context["dw_mysql_repository"] == "repo"
        return {"sql": "select corrected", "correction_attempts": 1}

    async def fake_execute_sql(state, executor, writer, runtime):
        assert state["sql"] == "select corrected"
        assert runtime.context["dw_mysql_repository"] == "repo"
        return {"final_answer": [{"GMV": 100}]}

    monkeypatch.setattr(sql_executor_module, "_pre_validate_sql", fake_pre_validate_sql)
    monkeypatch.setattr(
        sql_executor_module, "correct_sql_candidate", fake_correct_sql_candidate
    )
    monkeypatch.setattr(sql_executor_module, "_execute_sql", fake_execute_sql)

    result = asyncio.run(
        sql_executor_module.sql_executor(
            {
                "query": "统计 GMV",
                "sql": "select bad",
                "max_correction_attempts": 2,
            },
            SimpleNamespace(
                context={"dw_mysql_repository": "repo"},
                stream_writer=lambda _: None,
            ),
        )
    )

    assert result["sql"] == "select corrected"
    assert result["error"] is None
    assert result["safety_error"] is None
    assert result["correction_attempts"] == 1
    assert result["final_answer"] == [{"GMV": 100}]
    assert calls == [("validate", "select bad"), ("validate", "select corrected")]


def test_sql_executor_returns_failed_correction_state(monkeypatch):
    async def fake_pre_validate_sql(state, executor):
        return {"sql": state["sql"], "error": "Unknown column", "safety_error": None}

    def fake_fail_sql_correction(state):
        return {"blocked_by": "sql_correction", "error": state["error"]}

    monkeypatch.setattr(sql_executor_module, "_pre_validate_sql", fake_pre_validate_sql)
    monkeypatch.setattr(
        sql_executor_module, "_fail_sql_correction", fake_fail_sql_correction
    )

    result = asyncio.run(
        sql_executor_module.sql_executor(
            {
                "query": "统计 GMV",
                "sql": "select bad",
                "correction_attempts": 2,
                "max_correction_attempts": 2,
            },
            SimpleNamespace(
                context={"dw_mysql_repository": "repo"},
                stream_writer=lambda _: None,
            ),
        )
    )

    assert result["sql"] == "select bad"
    assert result["error"] == "Unknown column"
    assert result["blocked_by"] == "sql_correction"


def test_sql_executor_has_internal_loop_limit(monkeypatch):
    validations = 0
    corrections = 0

    async def fake_pre_validate_sql(state, executor):
        nonlocal validations
        validations += 1
        return {"sql": state["sql"], "error": "Unknown column", "safety_error": None}

    async def fake_correct_sql_candidate(state, context):
        nonlocal corrections
        corrections += 1
        return {"sql": state["sql"]}

    monkeypatch.setattr(sql_executor_module, "_pre_validate_sql", fake_pre_validate_sql)
    monkeypatch.setattr(
        sql_executor_module, "correct_sql_candidate", fake_correct_sql_candidate
    )

    result = asyncio.run(
        sql_executor_module.sql_executor(
            {
                "query": "统计 GMV",
                "sql": "select bad",
                "max_correction_attempts": 2,
            },
            SimpleNamespace(
                context={"dw_mysql_repository": "repo"},
                stream_writer=lambda _: None,
            ),
        )
    )

    assert validations == 3
    assert corrections == 3
    assert result["error"] == "SQL executor exceeded internal correction loop limit"
    assert result["blocked_by"] == "sql_executor"
