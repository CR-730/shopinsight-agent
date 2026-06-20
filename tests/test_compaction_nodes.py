import asyncio
from types import SimpleNamespace

from app.agent import context_compaction as context_compaction_helpers
from app.agent.node_observer import traced_node
from app.agent.nodes import context_builder as context_builder_module
from app.agent.nodes import context_compaction as context_compaction_module
from app.agent.nodes import sql_executor as sql_executor_module
from app.entities.column_info import ColumnInfo


def test_traced_node_appends_timing_without_losing_trace():
    async def node(state, runtime):
        return {"trace": {"keywords": ["GMV"]}}

    result = asyncio.run(
        traced_node("context_builder", node)(
            {"trace": {"node_timings": [{"step": "pre_rag_guard"}]}},
            SimpleNamespace(context={}),
        )
    )

    assert result["trace"]["keywords"] == ["GMV"]
    assert [item["step"] for item in result["trace"]["node_timings"]] == [
        "pre_rag_guard",
        "context_builder",
    ]


def test_context_builder_returns_structured_context_and_trace(monkeypatch):
    async def fake_recall_sql_memory_context(state, context):
        return {
            "sql_memory_examples": [
                {
                    "rank": 1,
                    "question": "历史问题",
                    "sql": "select 1",
                    "similarity": 0.9,
                }
            ]
        }

    async def fake_extract_retrieval_keywords(state):
        return {"keywords": ["GMV"]}

    async def fake_recall_column_context(state, context):
        return {"retrieved_column_infos": ["column"]}

    async def fake_recall_value_context(state, context):
        return {"retrieved_value_infos": ["value"]}

    async def fake_recall_metric_context(state, context):
        return {"retrieved_metric_infos": ["metric"]}

    async def fake_merge_retrieved_context(state, context):
        assert state["sql_memory_examples"][0]["sql"] == "select 1"
        assert state["keywords"] == ["GMV"]
        assert state["retrieved_column_infos"] == ["column"]
        assert state["retrieved_value_infos"] == ["value"]
        assert state["retrieved_metric_infos"] == ["metric"]
        return {"table_infos": ["table"], "metric_infos": ["metric_info"]}

    monkeypatch.setattr(context_builder_module, "recall_sql_memory_context", fake_recall_sql_memory_context)
    monkeypatch.setattr(context_builder_module, "extract_retrieval_keywords", fake_extract_retrieval_keywords)
    monkeypatch.setattr(context_builder_module, "recall_column_context", fake_recall_column_context)
    monkeypatch.setattr(context_builder_module, "recall_value_context", fake_recall_value_context)
    monkeypatch.setattr(context_builder_module, "recall_metric_context", fake_recall_metric_context)
    monkeypatch.setattr(context_builder_module, "merge_retrieved_context", fake_merge_retrieved_context)

    result = asyncio.run(
        context_builder_module.context_builder(
            {"query": "统计 GMV"}, SimpleNamespace(context={})
        )
    )

    assert result == {
        "sql_memory_examples": [
            {
                "rank": 1,
                "question": "历史问题",
                "sql": "select 1",
                "similarity": 0.9,
            }
        ],
        "retrieval_context": {"columns": ["column"], "metrics": ["metric"], "values": ["value"]},
        "sql_context": {"tables": ["table"], "metrics": ["metric_info"]},
        "trace": {
            "keywords": ["GMV"],
            "retrieved_columns": ["column"],
            "retrieved_metrics": ["metric"],
            "retrieved_values": ["value"],
        },
    }


def test_context_compaction_returns_sql_context(monkeypatch):
    async def fake_filter_table_context(state, context):
        assert state["sql_context"]["tables"] == ["raw_table"]
        assert context == {"dw_mysql_repository": "repo"}
        return {"table_infos": ["filtered_table"]}

    def fake_filter_metric_context(state):
        assert state["sql_context"]["tables"] == ["filtered_table"]
        return {"metric_infos": ["filtered_metric"]}

    async def fake_add_runtime_context(state, context):
        assert state["sql_context"]["metrics"] == ["filtered_metric"]
        assert context == {"dw_mysql_repository": "repo"}
        return {
            "date_info": {"date": "2026-06-03", "weekday": "Wednesday", "quarter": "Q2"},
            "db_info": {"dialect": "mysql", "version": "8.0"},
        }

    monkeypatch.setattr(context_compaction_module, "filter_table_context", fake_filter_table_context)
    monkeypatch.setattr(context_compaction_module, "filter_metric_context", fake_filter_metric_context)
    monkeypatch.setattr(context_compaction_module, "add_runtime_context", fake_add_runtime_context)

    result = asyncio.run(
        context_compaction_module.context_compaction(
            {"query": "统计 GMV", "sql_context": {"tables": ["raw_table"]}},
            SimpleNamespace(context={"dw_mysql_repository": "repo"}),
        )
    )

    assert result == {
        "sql_context": {
            "tables": ["filtered_table"],
            "metrics": ["filtered_metric"],
            "date": {"date": "2026-06-03", "weekday": "Wednesday", "quarter": "Q2"},
            "db": {"dialect": "mysql", "version": "8.0"},
        }
    }


def test_filter_table_context_adds_missing_protected_columns_from_metadata(monkeypatch):
    async def fake_ainvoke_llm_with_usage(*args, **kwargs):
        return {"dim_date": ["date_id", "quarter"]}

    class MetaRepository:
        async def list_column_infos(self):
            return [
                ColumnInfo(
                    id="dim_date.year",
                    name="year",
                    type="int",
                    role="dimension",
                    examples=[2025],
                    description="年份",
                    alias=["年份"],
                    table_id="dim_date",
                )
            ]

    monkeypatch.setattr(
        context_compaction_helpers,
        "ainvoke_llm_with_usage",
        fake_ainvoke_llm_with_usage,
    )

    result = asyncio.run(
        context_compaction_helpers.filter_table_context(
            {
                "query": "2025年第一季度各大区销售额",
                "business_binding": {
                    "time": {
                        "required_columns": [
                            "fact_order.date_id",
                            "dim_date.year",
                            "dim_date.quarter",
                        ]
                    }
                },
                "sql_context": {
                    "tables": [
                        {
                            "name": "dim_date",
                            "role": "dimension",
                            "columns": [
                                {
                                    "name": "date_id",
                                    "role": "primary_key",
                                    "alias": ["日期ID"],
                                    "examples": [20250101],
                                },
                                {
                                    "name": "quarter",
                                    "role": "dimension",
                                    "alias": ["季度"],
                                    "examples": ["Q1"],
                                },
                            ],
                        }
                    ]
                },
            },
            {
                "cost_tracker": None,
                "meta_mysql_repository": MetaRepository(),
            },
        )
    )

    columns = result["table_infos"][0]["columns"]
    assert {column["name"] for column in columns} == {"date_id", "quarter", "year"}


def test_sql_executor_returns_corrected_sql_and_execution_state(monkeypatch):
    calls = []

    async def fake_pre_validate_sql(state, executor):
        calls.append(("validate", state["sql"]))
        if len(calls) == 1:
            return {
                "sql": "select bad",
                "status": "repairable_error",
                "validation_error": "Unknown column",
            }
        return {"sql": "select corrected", "status": "pass", "validation_error": None}

    async def fake_correct_sql_candidate(
        state, context, validation_error, *, correction_attempts, max_correction_attempts
    ):
        assert state["sql"] == "select bad"
        assert validation_error == "Unknown column"
        assert context["dw_mysql_repository"] == "repo"
        assert correction_attempts == 0
        assert max_correction_attempts == 2
        return {"sql": "select corrected", "attempts": 1}

    async def fake_execute_sql(state, executor, writer, runtime):
        assert state["sql"] == "select corrected"
        assert runtime.context["dw_mysql_repository"] == "repo"
        return {"output": {"rows": [{"GMV": 100}]}}

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
            },
            SimpleNamespace(
                context={"dw_mysql_repository": "repo"},
                stream_writer=lambda _: None,
            ),
        )
    )

    assert result["sql"] == "select corrected"
    assert result["failure"] is None
    assert result["trace"]["sql_correction_attempts"] == 1
    assert result["output"]["rows"] == [{"GMV": 100}]
    assert calls == [("validate", "select bad"), ("validate", "select corrected")]


def test_sql_executor_returns_failed_correction_state(monkeypatch):
    async def fake_pre_validate_sql(state, executor):
        return {
            "sql": state["sql"],
            "status": "repairable_error",
            "validation_error": "Unknown column",
        }

    async def fake_correct_sql_candidate(
        state, context, validation_error, *, correction_attempts, max_correction_attempts
    ):
        return {"sql": state["sql"], "attempts": correction_attempts + 1}

    monkeypatch.setattr(sql_executor_module, "_pre_validate_sql", fake_pre_validate_sql)
    monkeypatch.setattr(
        sql_executor_module, "correct_sql_candidate", fake_correct_sql_candidate
    )

    result = asyncio.run(
        sql_executor_module.sql_executor(
            {
                "query": "统计 GMV",
                "sql": "select bad",
            },
            SimpleNamespace(
                context={"dw_mysql_repository": "repo"},
                stream_writer=lambda _: None,
            ),
        )
    )

    assert result["sql"] == "select bad"
    assert result["failure"] == {
        "category": "sql_validation",
        "stage": "sql_correction",
        "code": "correction_exhausted",
        "message": "Unknown column",
        "disposition": "failed",
    }


def test_sql_executor_has_internal_loop_limit(monkeypatch):
    validations = 0
    corrections = 0

    async def fake_pre_validate_sql(state, executor):
        nonlocal validations
        validations += 1
        return {
            "sql": state["sql"],
            "status": "repairable_error",
            "validation_error": "Unknown column",
        }

    async def fake_correct_sql_candidate(
        state, context, validation_error, *, correction_attempts, max_correction_attempts
    ):
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
            },
            SimpleNamespace(
                context={"dw_mysql_repository": "repo"},
                stream_writer=lambda _: None,
            ),
        )
    )

    assert validations == 3
    assert corrections == 2
    assert result["failure"]["code"] == "correction_exhausted"
    assert result["failure"]["message"] == "Unknown column"
