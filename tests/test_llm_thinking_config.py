from app.agent.llm import build_llm_kwargs, semantic_planning_llm
from app.conf.app_config import app_config


def test_structured_llm_disables_thinking():
    kwargs = build_llm_kwargs(enable_thinking=False)

    assert kwargs["extra_body"]["enable_thinking"] is False


def test_reasoning_llm_enables_thinking():
    kwargs = build_llm_kwargs(enable_thinking=True)

    assert kwargs["extra_body"]["enable_thinking"] is True


def test_build_llm_kwargs_accepts_model_override():
    kwargs = build_llm_kwargs(enable_thinking=False, model="fast-model")

    assert kwargs["model"] == "fast-model"


def test_generate_and_correct_sql_thinking_are_configured_separately():
    assert app_config.llm.generate_sql_enable_thinking is False
    assert app_config.llm.correct_sql_enable_thinking is True


def test_semantic_planning_uses_main_model():
    assert semantic_planning_llm.model_name == app_config.llm.model
    assert semantic_planning_llm.extra_body == {
        "enable_thinking": app_config.llm.structured_enable_thinking
    }
