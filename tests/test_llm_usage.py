from dataclasses import dataclass

import pytest
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from app.agent.cost import CostRates, CostTracker
from app.agent.llm_usage import ainvoke_llm_with_usage, clear_llm_response_cache


@dataclass
class FakeMessage:
    content: str
    usage_metadata: dict


class FakeLLM:
    model_name = "fast-model"

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        return FakeMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            },
        )


@pytest.mark.anyio
async def test_ainvoke_llm_with_usage_records_latency_model_and_cache_hit():
    clear_llm_response_cache()
    prompt = PromptTemplate(template="hello {name}", input_variables=["name"])
    llm = FakeLLM()
    tracker = CostTracker(CostRates())

    first = await ainvoke_llm_with_usage(
        prompt,
        llm,
        StrOutputParser(),
        {"name": "a"},
        "测试节点",
        tracker,
        timeout_seconds=5,
    )
    second = await ainvoke_llm_with_usage(
        prompt,
        llm,
        StrOutputParser(),
        {"name": "a"},
        "测试节点",
        tracker,
        timeout_seconds=5,
    )

    calls = tracker.summary()["calls"]
    assert first == second == "ok"
    assert llm.calls == 1
    assert calls[0]["model"] == "fast-model"
    assert calls[0]["latency_ms"] is not None
    assert calls[0]["cache_hit"] is False
    assert calls[1]["cache_hit"] is True
    assert calls[1]["total_tokens"] == 0


@pytest.mark.anyio
async def test_ainvoke_llm_with_usage_can_disable_cache():
    clear_llm_response_cache()
    prompt = PromptTemplate(template="hello {name}", input_variables=["name"])
    llm = FakeLLM()
    tracker = CostTracker(CostRates())

    for _ in range(2):
        await ainvoke_llm_with_usage(
            prompt,
            llm,
            StrOutputParser(),
            {"name": "a"},
            "生成SQL",
            tracker,
            timeout_seconds=5,
            cacheable=False,
        )

    assert llm.calls == 2
    assert all(call["cache_hit"] is False for call in tracker.summary()["calls"])
