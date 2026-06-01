from app.agent.llm import build_llm_kwargs


def test_structured_llm_disables_thinking():
    kwargs = build_llm_kwargs(enable_thinking=False)

    assert kwargs["extra_body"]["enable_thinking"] is False


def test_reasoning_llm_enables_thinking():
    kwargs = build_llm_kwargs(enable_thinking=True)

    assert kwargs["extra_body"]["enable_thinking"] is True


def test_build_llm_kwargs_accepts_model_override():
    kwargs = build_llm_kwargs(enable_thinking=False, model="fast-model")

    assert kwargs["model"] == "fast-model"
