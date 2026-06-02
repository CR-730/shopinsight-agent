import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from app.agent.cost import CostRates, CostTracker
from app.agent.llm_usage import (
    _cache_key,
    _resilience_settings,
    _runtime_cache_namespace,
    ainvoke_llm_with_usage,
    clear_llm_response_cache,
    clear_model_circuit_breakers,
    configure_llm_resilience_for_tests,
    reset_llm_cache_context_namespace,
    reset_llm_request_call_budget,
    reset_llm_resilience_for_tests,
    set_llm_cache_context_namespace,
    set_llm_request_call_budget,
)
from app.conf.app_config import app_config


@dataclass
class FakeMessage:
    content: str
    usage_metadata: dict


@pytest.fixture(autouse=True)
def reset_llm_resilience_state():
    reset_llm_resilience_for_tests()
    clear_model_circuit_breakers()
    budget_token = set_llm_request_call_budget(None)
    try:
        yield
    finally:
        reset_llm_request_call_budget(budget_token)
        reset_llm_resilience_for_tests()
        clear_model_circuit_breakers()


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


class SequentialFakeLLM:
    model_name = "fast-model"

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        return FakeMessage(
            content=f"response-{self.calls}",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            },
        )


class RejectFirstResponseParser(StrOutputParser):
    def parse(self, text: str) -> str:
        if text == "response-1":
            raise ValueError("invalid structured output")
        return text


class TransientLLM:
    model_name = "retry-model"

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("temporary timeout")
        return FakeMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            },
        )


class QuotaError(Exception):
    status_code = 403


class QuotaLLM:
    model_name = "quota-model"

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        raise QuotaError("AllocationQuota.FreeTierOnly")


class EndpointQuotaLLM(QuotaLLM):
    model_name = "same-model"

    def __init__(self, *, base_url: str, provider: str = "openai"):
        super().__init__()
        self.base_url = base_url
        self.model_provider = provider


class RetryAfterError(Exception):
    status_code = 429

    def __init__(self, retry_after: str):
        super().__init__("rate limit")
        self.response = type(
            "Response",
            (),
            {"headers": {"Retry-After": retry_after}},
        )()


class RateLimitError(Exception):
    status_code = 429


class RetryAfterLLM:
    model_name = "retry-after-model"

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        if self.calls == 1:
            raise RetryAfterError("0.25")
        return FakeMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            },
        )


class RateLimitLLM:
    model_name = "rate-limit-model"

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        raise RateLimitError("rate limit")


class HalfOpenLLM:
    model_name = "half-open-model"

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        if self.calls == 1:
            raise QuotaError("AllocationQuota.FreeTierOnly")
        return FakeMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            },
        )


class AlwaysRetryableLLM:
    model_name = "always-retryable-model"

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        raise TimeoutError("temporary timeout")


class SlowConcurrentLLM:
    model_name = "limited-model"

    def __init__(self):
        self.active = 0
        self.max_active = 0

    async def ainvoke(self, prompt):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return FakeMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            },
        )


class NamedLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name


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


@pytest.mark.anyio
async def test_ainvoke_llm_with_usage_does_not_cache_unparseable_response():
    clear_llm_response_cache()
    prompt = PromptTemplate(template="hello {name}", input_variables=["name"])
    llm = SequentialFakeLLM()
    tracker = CostTracker(CostRates())

    with pytest.raises(ValueError, match="invalid structured output"):
        await ainvoke_llm_with_usage(
            prompt,
            llm,
            RejectFirstResponseParser(),
            {"name": "a"},
            "抽取意图",
            tracker,
            timeout_seconds=5,
        )

    result = await ainvoke_llm_with_usage(
        prompt,
        llm,
        RejectFirstResponseParser(),
        {"name": "a"},
        "抽取意图",
        tracker,
        timeout_seconds=5,
    )

    assert result == "response-2"
    assert llm.calls == 2
    assert [call["cache_hit"] for call in tracker.summary()["calls"]] == [False, False]


def test_cache_key_includes_explicit_namespace_boundary():
    first = _cache_key("fast-model", "抽取意图", "prompt", cache_namespace="policy-v1")
    second = _cache_key("fast-model", "抽取意图", "prompt", cache_namespace="policy-v2")

    assert first != second


def test_runtime_cache_namespace_is_memoized(monkeypatch):
    clear_llm_response_cache()
    read_calls = []

    original_read_bytes = Path.read_bytes

    def tracked_read_bytes(self):
        read_calls.append(self)
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)

    first = _runtime_cache_namespace()
    second = _runtime_cache_namespace()

    assert first == second
    assert len(read_calls) == 3


def test_clear_llm_response_cache_clears_namespace_memo(monkeypatch):
    clear_llm_response_cache()
    read_calls = []

    original_read_bytes = Path.read_bytes

    def tracked_read_bytes(self):
        read_calls.append(self)
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)

    _runtime_cache_namespace()
    clear_llm_response_cache()
    _runtime_cache_namespace()

    assert len(read_calls) == 6


def test_cache_key_includes_request_metadata_namespace():
    clear_llm_response_cache()

    first_token = set_llm_cache_context_namespace("metadata:v1")
    try:
        first = _cache_key("fast-model", "过滤表信息", "prompt")
    finally:
        reset_llm_cache_context_namespace(first_token)

    second_token = set_llm_cache_context_namespace("metadata:v2")
    try:
        second = _cache_key("fast-model", "过滤表信息", "prompt")
    finally:
        reset_llm_cache_context_namespace(second_token)

    assert first != second


def test_fast_and_sql_models_use_separate_resilience_settings():
    fast_settings = _resilience_settings(NamedLLM(app_config.llm.fast_model))
    sql_settings = _resilience_settings(NamedLLM(app_config.llm.model))

    assert fast_settings.max_retries == 2
    assert fast_settings.concurrency_limit == 4
    assert fast_settings.quota_circuit_breaker_seconds == 60
    assert sql_settings.max_retries == 1
    assert sql_settings.concurrency_limit == 1
    assert sql_settings.quota_circuit_breaker_seconds == 300


@pytest.mark.anyio
async def test_ainvoke_llm_with_usage_retries_transient_model_error():
    clear_llm_response_cache()
    clear_model_circuit_breakers()
    configure_llm_resilience_for_tests(
        max_retries=1,
        retry_backoff_seconds=0,
        concurrency_limit=2,
        quota_circuit_breaker_seconds=60,
    )
    prompt = PromptTemplate(template="hello {name}", input_variables=["name"])
    llm = TransientLLM()
    tracker = CostTracker(CostRates())

    result = await ainvoke_llm_with_usage(
        prompt,
        llm,
        StrOutputParser(),
        {"name": "a"},
        "抽取意图",
        tracker,
        timeout_seconds=5,
        cacheable=False,
    )

    assert result == "ok"
    assert llm.calls == 2
    assert tracker.summary()["calls"][0]["retry_count"] == 1


@pytest.mark.anyio
async def test_quota_error_opens_model_circuit_and_next_call_fails_fast():
    clear_llm_response_cache()
    clear_model_circuit_breakers()
    configure_llm_resilience_for_tests(
        max_retries=1,
        retry_backoff_seconds=0,
        concurrency_limit=2,
        quota_circuit_breaker_seconds=60,
    )
    prompt = PromptTemplate(template="hello {name}", input_variables=["name"])
    llm = QuotaLLM()
    tracker = CostTracker(CostRates())

    with pytest.raises(QuotaError):
        await ainvoke_llm_with_usage(
            prompt,
            llm,
            StrOutputParser(),
            {"name": "a"},
            "抽取意图",
            tracker,
            timeout_seconds=5,
            cacheable=False,
        )

    with pytest.raises(RuntimeError, match="circuit breaker is open"):
        await ainvoke_llm_with_usage(
            prompt,
            llm,
            StrOutputParser(),
            {"name": "a"},
            "抽取意图",
            tracker,
            timeout_seconds=5,
            cacheable=False,
        )

    assert llm.calls == 1
    calls = tracker.summary()["calls"]
    assert calls[0]["error_type"] == "QuotaError"
    assert calls[1]["error_type"] == "ModelCircuitOpenError"


@pytest.mark.anyio
async def test_model_circuit_key_includes_provider_base_url_and_model():
    clear_llm_response_cache()
    clear_model_circuit_breakers()
    configure_llm_resilience_for_tests(
        max_retries=0,
        retry_backoff_seconds=0,
        concurrency_limit=2,
        quota_circuit_breaker_seconds=60,
    )
    prompt = PromptTemplate(template="hello {name}", input_variables=["name"])
    first_endpoint = EndpointQuotaLLM(base_url="https://a.example/v1")
    second_endpoint = EndpointQuotaLLM(base_url="https://b.example/v1")
    tracker = CostTracker(CostRates())

    with pytest.raises(QuotaError):
        await ainvoke_llm_with_usage(
            prompt,
            first_endpoint,
            StrOutputParser(),
            {"name": "a"},
            "抽取意图",
            tracker,
            timeout_seconds=5,
            cacheable=False,
        )

    with pytest.raises(QuotaError):
        await ainvoke_llm_with_usage(
            prompt,
            second_endpoint,
            StrOutputParser(),
            {"name": "a"},
            "抽取意图",
            tracker,
            timeout_seconds=5,
            cacheable=False,
        )

    assert first_endpoint.calls == 1
    assert second_endpoint.calls == 1


@pytest.mark.anyio
async def test_retry_respects_retry_after_before_backoff(monkeypatch):
    clear_llm_response_cache()
    clear_model_circuit_breakers()
    configure_llm_resilience_for_tests(
        max_retries=1,
        retry_backoff_seconds=10,
        concurrency_limit=2,
        quota_circuit_breaker_seconds=60,
    )
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("app.agent.llm_usage.asyncio.sleep", fake_sleep)
    prompt = PromptTemplate(template="hello {name}", input_variables=["name"])
    llm = RetryAfterLLM()
    tracker = CostTracker(CostRates())

    result = await ainvoke_llm_with_usage(
        prompt,
        llm,
        StrOutputParser(),
        {"name": "a"},
        "抽取意图",
        tracker,
        timeout_seconds=5,
        cacheable=False,
    )

    assert result == "ok"
    assert sleeps == [0.25]


@pytest.mark.anyio
async def test_llm_calls_are_limited_per_model():
    clear_llm_response_cache()
    clear_model_circuit_breakers()
    configure_llm_resilience_for_tests(
        max_retries=0,
        retry_backoff_seconds=0,
        concurrency_limit=1,
        quota_circuit_breaker_seconds=60,
    )
    prompt = PromptTemplate(template="hello {name}", input_variables=["name"])
    llm = SlowConcurrentLLM()

    async def call_once(index: int):
        tracker = CostTracker(CostRates())
        return await ainvoke_llm_with_usage(
            prompt,
            llm,
            StrOutputParser(),
            {"name": str(index)},
            "抽取意图",
            tracker,
            timeout_seconds=5,
            cacheable=False,
        )

    assert await asyncio.gather(call_once(1), call_once(2), call_once(3)) == [
        "ok",
        "ok",
        "ok",
    ]
    assert llm.max_active == 1


@pytest.mark.anyio
async def test_consecutive_rate_limits_open_circuit_breaker():
    clear_llm_response_cache()
    clear_model_circuit_breakers()
    configure_llm_resilience_for_tests(
        max_retries=0,
        retry_backoff_seconds=0,
        concurrency_limit=2,
        quota_circuit_breaker_seconds=60,
        rate_limit_breaker_threshold=2,
        error_window_seconds=60,
        error_window_min_calls=20,
        error_rate_threshold=0.5,
        max_calls_per_request=100,
    )
    prompt = PromptTemplate(template="hello {name}", input_variables=["name"])
    llm = RateLimitLLM()
    tracker = CostTracker(CostRates())

    for _ in range(2):
        with pytest.raises(RateLimitError):
            await ainvoke_llm_with_usage(
                prompt,
                llm,
                StrOutputParser(),
                {"name": "a"},
                "抽取意图",
                tracker,
                timeout_seconds=5,
                cacheable=False,
            )

    with pytest.raises(RuntimeError, match="circuit breaker is open"):
        await ainvoke_llm_with_usage(
            prompt,
            llm,
            StrOutputParser(),
            {"name": "a"},
            "抽取意图",
            tracker,
            timeout_seconds=5,
            cacheable=False,
        )

    assert llm.calls == 2
    assert tracker.summary()["calls"][-1]["breaker_state"] == "open"


@pytest.mark.anyio
async def test_sliding_error_window_counts_one_logical_call_not_each_retry():
    clear_llm_response_cache()
    clear_model_circuit_breakers()
    configure_llm_resilience_for_tests(
        max_retries=2,
        retry_backoff_seconds=0,
        concurrency_limit=2,
        quota_circuit_breaker_seconds=60,
        rate_limit_breaker_threshold=99,
        error_window_seconds=60,
        error_window_min_calls=2,
        error_rate_threshold=0.5,
        max_calls_per_request=100,
    )
    prompt = PromptTemplate(template="hello {name}", input_variables=["name"])
    llm = AlwaysRetryableLLM()
    tracker = CostTracker(CostRates())

    for _ in range(2):
        with pytest.raises(TimeoutError):
            await ainvoke_llm_with_usage(
                prompt,
                llm,
                StrOutputParser(),
                {"name": "a"},
                "鎶藉彇鎰忓浘",
                tracker,
                timeout_seconds=5,
                cacheable=False,
            )

    assert llm.calls == 6


@pytest.mark.anyio
async def test_open_circuit_moves_to_half_open_and_closes_after_success():
    clear_llm_response_cache()
    clear_model_circuit_breakers()
    configure_llm_resilience_for_tests(
        max_retries=0,
        retry_backoff_seconds=0,
        concurrency_limit=2,
        quota_circuit_breaker_seconds=0,
        rate_limit_breaker_threshold=3,
        error_window_seconds=60,
        error_window_min_calls=20,
        error_rate_threshold=0.5,
        max_calls_per_request=100,
    )
    prompt = PromptTemplate(template="hello {name}", input_variables=["name"])
    llm = HalfOpenLLM()
    tracker = CostTracker(CostRates())

    with pytest.raises(QuotaError):
        await ainvoke_llm_with_usage(
            prompt,
            llm,
            StrOutputParser(),
            {"name": "a"},
            "抽取意图",
            tracker,
            timeout_seconds=5,
            cacheable=False,
        )

    result = await ainvoke_llm_with_usage(
        prompt,
        llm,
        StrOutputParser(),
        {"name": "a"},
        "抽取意图",
        tracker,
        timeout_seconds=5,
        cacheable=False,
    )

    assert result == "ok"
    assert tracker.summary()["calls"][-1]["breaker_state"] == "half_open"


@pytest.mark.anyio
async def test_request_llm_call_budget_limits_actual_attempts():
    clear_llm_response_cache()
    clear_model_circuit_breakers()
    configure_llm_resilience_for_tests(
        max_retries=2,
        retry_backoff_seconds=0,
        concurrency_limit=2,
        quota_circuit_breaker_seconds=60,
        rate_limit_breaker_threshold=3,
        error_window_seconds=60,
        error_window_min_calls=20,
        error_rate_threshold=0.5,
        max_calls_per_request=1,
    )
    budget_token = set_llm_request_call_budget(1)
    prompt = PromptTemplate(template="hello {name}", input_variables=["name"])
    llm = TransientLLM()
    tracker = CostTracker(CostRates())

    try:
        with pytest.raises(RuntimeError, match="LLM call budget exceeded"):
            await ainvoke_llm_with_usage(
                prompt,
                llm,
                StrOutputParser(),
                {"name": "a"},
                "抽取意图",
                tracker,
                timeout_seconds=5,
                cacheable=False,
            )
    finally:
        reset_llm_request_call_budget(budget_token)

    assert llm.calls == 1
    assert tracker.summary()["calls"][0]["final_error_type"] == "LLMCallBudgetExceeded"
