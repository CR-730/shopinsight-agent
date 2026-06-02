from dataclasses import dataclass

from app.agent.cost import (
    CostRates,
    CostTracker,
    estimate_tokens,
    extract_token_usage,
)


@dataclass
class FakeMessage:
    content: str
    usage_metadata: dict | None = None
    response_metadata: dict | None = None


def test_extract_token_usage_prefers_usage_metadata():
    message = FakeMessage(
        content="ok",
        usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
        response_metadata={"token_usage": {"prompt_tokens": 1}},
    )

    usage = extract_token_usage(message)

    assert usage.input_tokens == 10
    assert usage.output_tokens == 3
    assert usage.total_tokens == 13
    assert usage.estimated is False


def test_extract_token_usage_supports_openai_response_metadata():
    message = FakeMessage(
        content="ok",
        response_metadata={
            "token_usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
            }
        },
    )

    usage = extract_token_usage(message)

    assert usage.input_tokens == 12
    assert usage.output_tokens == 4
    assert usage.total_tokens == 16
    assert usage.estimated is False


def test_cost_tracker_calculates_llm_and_embedding_costs():
    tracker = CostTracker(
        CostRates(
            llm_input_per_1m_tokens=1.0,
            llm_output_per_1m_tokens=2.0,
            embedding_per_1m_tokens=0.5,
            currency="CNY",
        )
    )

    tracker.add_llm_usage("生成SQL", input_tokens=1000, output_tokens=2000)
    tracker.add_embedding_usage("召回字段", tokens=4000, estimated=True)
    summary = tracker.summary()

    assert summary["llm_input_tokens"] == 1000
    assert summary["llm_output_tokens"] == 2000
    assert summary["embedding_tokens"] == 4000
    assert summary["total_cost"] == 0.007
    assert summary["currency"] == "CNY"
    assert summary["embedding_estimated"] is True
    assert summary["calls"][0]["latency_ms"] is None
    assert summary["calls"][0]["model"] is None


def test_cost_tracker_records_llm_observability_fields():
    tracker = CostTracker(CostRates())

    tracker.add_llm_usage(
        "过滤表信息",
        input_tokens=10,
        output_tokens=2,
        model="fast-model",
        latency_ms=123.4,
        cached_tokens=5,
        cache_hit=True,
        retry_count=1,
        breaker_state="open",
        retry_after_ms=250,
        throttle_wait_ms=10,
        final_error_type="TimeoutError",
        error_type="TimeoutError",
    )
    call = tracker.summary()["calls"][0]

    assert call["model"] == "fast-model"
    assert call["latency_ms"] == 123.4
    assert call["cached_tokens"] == 5
    assert call["cache_hit"] is True
    assert call["retry_count"] == 1
    assert call["breaker_state"] == "open"
    assert call["retry_after_ms"] == 250
    assert call["throttle_wait_ms"] == 10
    assert call["final_error_type"] == "TimeoutError"
    assert call["error_type"] == "TimeoutError"


def test_estimate_tokens_handles_chinese_text():
    assert estimate_tokens("华北地区销售额") >= 1
