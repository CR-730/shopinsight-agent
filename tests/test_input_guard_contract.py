from pathlib import Path

import pytest
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError

from app.agent.nodes.intent_recognition import InputGuardDecision
from app.evaluation.cases import load_eval_cases
from app.prompt.prompt_loader import load_prompt


def test_input_guard_decision_includes_the_rewritten_query_contract():
    assert set(InputGuardDecision.model_fields) == {
        "decision",
        "category",
        "rewritten_query",
        "user_message",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"decision": "allow", "category": "missing_query_object", "user_message": ""},
        {
            "decision": "allow",
            "category": "safe",
            "rewritten_query": "",
            "user_message": "",
        },
        {"decision": "block", "category": "safe", "user_message": "不能处理"},
        {
            "decision": "block",
            "category": "clearly_non_data",
            "rewritten_query": "统计销售额",
            "user_message": "",
        },
    ],
)
def test_input_guard_decision_rejects_contradictory_outputs(payload):
    with pytest.raises(ValidationError):
        InputGuardDecision.model_validate(payload)


def test_input_guard_prompt_uses_a_grounded_professional_role():
    prompt = load_prompt("pre_rag_guard")

    assert "企业经营数据分析师" in prompt
    assert "电商问数系统检索前的粗粒度输入守卫" not in prompt
    assert "后续流程" not in prompt
    assert "语义规划" not in prompt


def test_input_guard_prompt_distinguishes_missing_target_from_unknown_business_term():
    prompt = load_prompt("pre_rag_guard")

    assert "不要判断术语、指标、字段或取值是否真实存在" in prompt
    assert "按地区统计销售额" in prompt
    assert "查一下数据" in prompt


def test_input_guard_prompt_renders_without_treating_json_examples_as_variables():
    prompt = PromptTemplate(
        template=load_prompt("pre_rag_guard"),
        input_variables=["query", "conversation_history"],
        partial_variables={"format_instructions": "只输出四个字段"},
    )

    rendered = prompt.format(
        query="改成华南",
        conversation_history="user: 按地区统计销售额",
    )

    assert '"decision":"allow"' in rendered
    assert "改成华南" in rendered
    assert "按地区统计销售额" in rendered


def test_real_input_guard_regression_keeps_turnover_query_and_oracle():
    cases = load_eval_cases(Path("examples/eval_input_guard.yaml"))

    assert len(cases) == 1
    assert cases[0].query == "统计成交额"
    assert cases[0].expected_blocked_by is None
    assert cases[0].oracle_sql
