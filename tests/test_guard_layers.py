import asyncio

from app.agent.nodes import intent_recognition as intent_recognition_module
from app.agent.nodes.intent_recognition import (
    _is_block_decision,
    classify_query_intent,
    intent_recognition,
)
from app.agent.schema_relations import is_valid_join_pair
from app.agent.sql.sql_guard import (
    normalize_sql_for_execution,
    validate_sql_before_execution,
    validate_sql_structure_semantics,
)
from app.agent.stop_signal import split_stop_signal
from app.entities.value_info import ValueInfo


def _semantic_plan(**overrides):
    plan = {
        "version": "1.0",
        "metadata_version": "test-v1",
        "measures": [],
        "dimensions": [],
        "predicates": [],
        "order_by": [],
        "limit": None,
        "joins": [],
        "required_table_ids": [],
        "required_column_ids": [],
        "provenance": [],
    }
    plan.update(overrides)
    return plan


def test_split_stop_signal_strips_marker_from_user_visible_text():
    visible, should_stop = split_stop_signal(
        "我理解你是在问天气，但我只能处理电商经营数据查询。find_error"
    )

    assert visible == "我理解你是在问天气，但我只能处理电商经营数据查询。"
    assert should_stop is True


def test_intent_recognition_blocks_when_classifier_fails():
    assert (
        _is_block_decision(
            {
                "decision": "block",
                "category": "classifier_error",
                "user_message": "入口检查暂时不可用。",
            }
        )
        is True
    )


def test_intent_recognition_allows_clear_data_query_without_business_keyword_rules():
    result = {
        "decision": "allow",
        "category": "safe",
        "rewritten_query": "统计销售额",
        "user_message": "",
    }

    assert _is_block_decision(result) is False


def test_intent_recognition_blocks_missing_query_object():
    result = {
        "decision": "block",
        "category": "missing_query_object",
        "rewritten_query": "",
        "user_message": "请补充要查询的指标或业务对象。",
    }

    assert _is_block_decision(result) is True


def test_intent_recognition_blocks_prompt_injection():
    result = {
        "decision": "block",
        "category": "prompt_injection",
        "rewritten_query": "",
        "user_message": "无法处理该请求。",
    }

    assert _is_block_decision(result) is True


def test_intent_recognition_classifier_failure_returns_block_decision(monkeypatch):
    async def fail_llm(*args, **kwargs):
        raise TimeoutError("classifier timeout")

    class Runtime:
        context = {"cost_tracker": None}

    monkeypatch.setattr(intent_recognition_module, "ainvoke_llm_with_usage", fail_llm)

    result = asyncio.run(classify_query_intent("统计销售额", Runtime()))

    assert result["decision"] == "block"
    assert result["category"] == "classifier_error"


def test_intent_recognition_streams_llm_text_without_stop_marker(monkeypatch):
    events = []

    class Runtime:
        stream_writer = staticmethod(events.append)
        context = {"cost_tracker": None}

    async def fake_classify_query_intent(query, runtime, *, conversation_history=""):
        return {
            "decision": "block",
            "category": "clearly_non_data",
            "rewritten_query": "",
            "user_message": "我理解你是在问天气，但我只能处理电商经营数据查询。",
        }

    monkeypatch.setattr(
        intent_recognition_module,
        "classify_query_intent",
        fake_classify_query_intent,
    )

    result = asyncio.run(intent_recognition({"query": "今天天气怎么样"}, Runtime()))
    text = "".join(event.get("delta", "") for event in events)

    failure = result["failure"]
    assert failure["stage"] == "intent_recognition"
    assert failure["disposition"] == "blocked"
    assert (
        failure["user_message"] == "我理解你是在问天气，但我只能处理电商经营数据查询。"
    )
    assert "find_error" not in text


def test_intent_recognition_reports_step_without_generating_user_response(monkeypatch):
    events = []
    captured = {}

    class Runtime:
        stream_writer = staticmethod(events.append)
        context = {"cost_tracker": None}

    async def fake_classify_query_intent(query, runtime, *, conversation_history=""):
        captured["query"] = query
        captured["conversation_history"] = conversation_history
        return {
            "decision": "allow",
            "category": "safe",
            "rewritten_query": "统计华南地区的销售额",
            "user_message": "",
        }

    monkeypatch.setattr(
        intent_recognition_module,
        "classify_query_intent",
        fake_classify_query_intent,
    )

    result = asyncio.run(
        intent_recognition(
            {
                "query": "改成华南",
                "conversation_messages": [
                    {"role": "user", "content": "按地区统计销售额"},
                    {"role": "assistant", "content": "已经查询完成"},
                ],
            },
            Runtime(),
        )
    )
    text = "".join(event.get("delta", "") for event in events)

    assert result == {"query": "统计华南地区的销售额", "failure": None}
    assert captured == {
        "query": "改成华南",
        "conversation_history": "user: 按地区统计销售额",
    }
    assert events[0] == {"type": "progress", "step": "意图识别", "status": "running"}
    assert events[-1] == {"type": "progress", "step": "意图识别", "status": "success"}
    assert text == ""


def test_pre_sql_execution_validation_blocks_sensitive_detail_sql():
    sql = "SELECT customer_id AS 用户ID, order_amount AS 订单金额 FROM fact_order"

    error = validate_sql_before_execution({"query": "查询所有用户ID和订单金额"}, sql)

    assert error is not None
    assert "敏感字段" in error


def test_pre_sql_execution_validation_blocks_sensitive_column_in_where():
    sql = "SELECT COUNT(*) AS cnt FROM fact_order WHERE customer_id = '123'"

    error = validate_sql_before_execution({"query": "统计订单数"}, sql)

    assert error is not None
    assert "敏感字段" in error


def test_pre_sql_execution_validation_allows_sensitive_identifier_count():
    sql = (
        "SELECT region_id, COUNT(DISTINCT customer_id) AS customer_count "
        "FROM fact_order GROUP BY region_id"
    )

    error = validate_sql_before_execution({"query": "按地区统计下单客户数"}, sql)

    assert error is None


def test_pre_sql_execution_validation_blocks_sensitive_column_in_cte():
    sql = "WITH x AS (SELECT customer_id AS cid FROM fact_order) SELECT cid FROM x"

    error = validate_sql_before_execution({"query": "统计用户"}, sql)

    assert error is not None
    assert "敏感字段" in error


def test_pre_sql_execution_validation_blocks_sensitive_column_in_subquery():
    sql = "SELECT cid FROM (SELECT customer_id AS cid FROM fact_order) t"

    error = validate_sql_before_execution({"query": "统计用户"}, sql)

    assert error is not None
    assert "敏感字段" in error


def test_pre_sql_execution_validation_allows_sensitive_join_key_not_projected():
    sql = """
    SELECT dc.member_level AS 会员等级, AVG(fo.order_amount) AS 客单价
    FROM fact_order fo
    JOIN dim_customer dc ON fo.customer_id = dc.customer_id
    GROUP BY dc.member_level
    """

    error = validate_sql_before_execution({"query": "按会员等级统计客单价"}, sql)

    assert error is None


def test_pre_sql_execution_validation_blocks_sensitive_non_key_join_condition():
    sql = """
    SELECT COUNT(*) AS cnt
    FROM dim_customer dc
    JOIN dim_customer dx ON dc.phone = dx.phone
    """

    error = validate_sql_before_execution({"query": "统计用户数量"}, sql)

    assert error is not None
    assert "敏感字段" in error


def test_pre_sql_execution_validation_flags_invalid_join_relationship_as_repairable():
    state = {
        "semantic_plan": _semantic_plan(
            joins=[
                {
                    "left_column_id": "fact_order.region_id",
                    "right_column_id": "dim_region.region_id",
                    "join_type": "inner",
                },
                {
                    "left_column_id": "fact_order.product_id",
                    "right_column_id": "dim_product.product_id",
                    "join_type": "inner",
                },
            ],
            required_table_ids=["fact_order", "dim_region", "dim_product"],
            required_column_ids=[
                "fact_order.region_id",
                "fact_order.product_id",
                "fact_order.order_amount",
                "dim_region.region_id",
                "dim_region.region_name",
                "dim_product.product_id",
                "dim_product.category",
            ],
        )
    }
    sql = """
    SELECT dp.category AS category, SUM(fo.order_amount) AS amount
    FROM fact_order fo
    JOIN dim_region dr ON fo.region_id = dr.region_name
    JOIN dim_product dp ON fo.product_id = dp.product_id
    WHERE dr.region_name = '华北'
    GROUP BY dp.category
    """

    error = validate_sql_structure_semantics(state, sql)

    assert error == (
        "JOIN 条件不符合查询计划：fact_order.region_id = dim_region.region_name。"
        "计划关系：fact_order.region_id = dim_region.region_id。"
    )


def test_pre_sql_execution_validation_allows_valid_join_relationship():
    state = {
        "semantic_plan": _semantic_plan(
            joins=[
                {
                    "left_column_id": "fact_order.region_id",
                    "right_column_id": "dim_region.region_id",
                    "join_type": "inner",
                }
            ],
            required_table_ids=["fact_order", "dim_region"],
            required_column_ids=[
                "fact_order.region_id",
                "fact_order.order_amount",
                "dim_region.region_id",
                "dim_region.region_name",
            ],
        )
    }
    sql = """
    SELECT dr.region_name AS region_name, SUM(fo.order_amount) AS amount
    FROM fact_order fo
    JOIN dim_region dr ON fo.region_id = dr.region_id
    GROUP BY dr.region_name
    """

    error = validate_sql_structure_semantics(state, sql)

    assert error is None


def test_empty_join_plan_never_falls_back_to_retrieval_context_for_validation():
    state = {
        "semantic_plan": _semantic_plan(
            required_table_ids=["fact_order"],
            required_column_ids=["fact_order.order_amount"],
        ),
        "sql_context": {
            "tables": [
                {
                    "name": "fact_order",
                    "columns": [{"name": "region_id", "role": "foreign_key"}],
                },
                {
                    "name": "dim_region",
                    "columns": [{"name": "region_id", "role": "primary_key"}],
                },
            ]
        },
    }
    sql = """
    SELECT SUM(f.order_amount) AS GMV
    FROM fact_order f
    JOIN dim_region r ON f.region_id = r.region_name
    """

    assert "不符合查询计划" in validate_sql_structure_semantics(state, sql)


def test_schema_relation_join_rule_matches_sql_guard_contract():
    assert is_valid_join_pair(
        {"name": "region_id", "role": "foreign_key", "table_id": "fact_order"},
        {"name": "region_id", "role": "primary_key", "table_id": "dim_region"},
    )
    assert not is_valid_join_pair(
        {"name": "region_id", "role": "foreign_key", "table_id": "fact_order"},
        {"name": "region_name", "role": "primary_key", "table_id": "dim_region"},
    )


def test_pre_sql_execution_validation_blocks_multi_statement_sql():
    sql = "SELECT COUNT(*) AS 订单数 FROM fact_order; SELECT * FROM dim_customer"

    error = validate_sql_before_execution({"query": "统计订单数"}, sql)

    assert error == "仅允许执行单条 SELECT 查询"


def test_pre_sql_execution_validation_blocks_projection_star_by_ast():
    sql = "SELECT fact_order.* FROM fact_order"

    error = validate_sql_before_execution({"query": "查询订单"}, sql)

    assert error == "禁止 SELECT *"


def test_sql_guard_accepts_semantic_plan_literal_without_retrieval_value():
    sql = "SELECT SUM(order_amount) AS GMV FROM fact_order WHERE region_name = '华北'"

    error = validate_sql_before_execution(
        {
            "query": "华北 GMV",
            "semantic_plan": _semantic_plan(
                predicates=[
                    {
                        "kind": "enum",
                        "column_id": "dim_region.region_name",
                        "operator": "eq",
                        "canonical_values": ["华北"],
                    }
                ]
            ),
            "retrieval_context": {"values": []},
        },
        sql,
    )

    assert error is None


def test_sql_guard_rejects_retrieved_literal_not_authorized_by_plan():
    sql = "SELECT SUM(order_amount) FROM fact_order WHERE region_name = '华北'"

    error = validate_sql_before_execution(
        {
            "query": "查询销售额",
            "semantic_plan": _semantic_plan(),
            "retrieval_context": {
                "values": [
                    ValueInfo(
                        id="dim_region.region_name.华北",
                        value="华北",
                        column_id="dim_region.region_name",
                    )
                ]
            },
        },
        sql,
    )

    assert error == "SQL 使用了未授权的枚举值：华北"


def test_pre_sql_execution_validation_allows_date_literals():
    sql = "SELECT SUM(order_amount) AS GMV FROM fact_order WHERE dt = '2025-01-01'"

    error = validate_sql_before_execution({"query": "2025-01-01 GMV"}, sql)

    assert error is None


def test_pre_sql_execution_validation_allows_mysql_date_format_literals():
    sql = (
        "SELECT SUM(order_amount) AS GMV FROM fact_order "
        "WHERE QUARTER(STR_TO_DATE(CAST(date_id AS CHAR), '%Y%m%d')) = 1"
    )

    error = validate_sql_before_execution({"query": "一季度 GMV"}, sql)

    assert error is None


def test_pre_sql_execution_validation_normalizes_fullwidth_comma():
    sql = "SELECT region_name AS 大区，SUM(order_amount) AS GMV FROM fact_order"

    assert normalize_sql_for_execution(sql) == (
        "SELECT region_name AS 大区,SUM(order_amount) AS GMV FROM fact_order"
    )


def test_pre_sql_execution_validation_extracts_sql_from_markdown():
    sql = """
    这是生成的 SQL：
    ```sql
    SELECT SUM(order_amount) AS GMV FROM fact_order;
    ```
    """

    assert (
        normalize_sql_for_execution(sql)
        == "SELECT SUM(order_amount) AS GMV FROM fact_order"
    )


def test_pre_sql_execution_validation_normalizes_fullwidth_parentheses():
    sql = "SELECT SUM（order_amount） AS GMV FROM fact_order；"

    assert (
        normalize_sql_for_execution(sql)
        == "SELECT SUM(order_amount) AS GMV FROM fact_order"
    )
