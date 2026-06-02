"""LLM invocation helpers with usage, latency, and lightweight caching."""

import asyncio
import hashlib
import os
import random
import time
from collections import OrderedDict, deque
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate

from app.agent.cached_clients import ainvoke_with_timeout
from app.agent.cost import CostTracker, extract_token_usage
from app.conf.app_config import app_config

_LLM_RESPONSE_CACHE: OrderedDict[str, str] = OrderedDict()
_MAX_LLM_RESPONSE_CACHE_ENTRIES = 512
_CACHE_NAMESPACE_MEMO: str | None = None
_LLM_CACHE_CONTEXT_NAMESPACE: ContextVar[str] = ContextVar(
    "llm_cache_context_namespace", default=""
)
_LLM_REQUEST_CALL_BUDGET: ContextVar[tuple[int | None, int]] = ContextVar(
    "llm_request_call_budget", default=(None, 0)
)
_MODEL_CIRCUITS: dict[str, "CircuitState"] = {}
_MODEL_SEMAPHORES: dict[str, asyncio.Semaphore] = {}
_LLM_RESILIENCE_SETTINGS: "LLMResilienceSettings | None" = None


@dataclass
class LLMResilienceSettings:
    max_retries: int
    retry_backoff_seconds: float
    concurrency_limit: int
    quota_circuit_breaker_seconds: int
    rate_limit_breaker_threshold: int = 3
    error_window_seconds: int = 60
    error_window_min_calls: int = 20
    error_rate_threshold: float = 0.5
    max_calls_per_request: int = 40


@dataclass
class CircuitState:
    state: str = "closed"
    open_until: float = 0.0
    half_open_probe_in_flight: bool = False
    consecutive_rate_limit_errors: int = 0
    recent_results: deque[tuple[float, bool]] = None

    def __post_init__(self):
        if self.recent_results is None:
            self.recent_results = deque()


@dataclass
class LLMPolicyResult:
    message: Any
    retry_count: int
    breaker_state: str = "closed"
    retry_after_ms: float | None = None
    throttle_wait_ms: float = 0.0


class ModelCircuitOpenError(RuntimeError):
    pass


class LLMCallBudgetExceeded(RuntimeError):
    pass


async def ainvoke_llm_with_usage(
    prompt: PromptTemplate,
    llm,
    output_parser: BaseOutputParser,
    inputs: dict,
    step: str,
    cost_tracker: CostTracker,
    timeout_seconds: int,
    cacheable: bool = True,
):
    """Invoke an LLM, record observability data, then parse the response."""

    prompt_value = await prompt.ainvoke(inputs)
    prompt_text = prompt_value.to_string()
    model = _model_name(llm)
    policy_key = _model_policy_key(llm)
    cache_key = _cache_key(model, step, prompt_text)
    started_at = time.perf_counter()

    if cacheable and cache_key in _LLM_RESPONSE_CACHE:
        content = _touch_cache(cache_key)
        cost_tracker.add_llm_usage(
            step,
            input_tokens=0,
            output_tokens=0,
            model=model,
            latency_ms=_elapsed_ms(started_at),
            cache_hit=True,
        )
        return output_parser.parse(content)

    try:
        policy_result = await invoke_llm_with_policy(
            llm, prompt_value, timeout_seconds, policy_key
        )
    except Exception as exc:
        cost_tracker.add_llm_usage(
            step,
            input_tokens=0,
            output_tokens=0,
            estimated=True,
            model=model,
            latency_ms=_elapsed_ms(started_at),
            cache_hit=False,
            retry_count=getattr(exc, "retry_count", 0),
            breaker_state=getattr(exc, "breaker_state", "closed"),
            retry_after_ms=getattr(exc, "retry_after_ms", None),
            throttle_wait_ms=getattr(exc, "throttle_wait_ms", None),
            final_error_type=exc.__class__.__name__,
            error_type=exc.__class__.__name__,
        )
        raise

    message = policy_result.message
    retry_count = policy_result.retry_count
    usage = extract_token_usage(message)
    content = message.content

    try:
        parsed = output_parser.parse(content)
    except Exception as exc:
        cost_tracker.add_llm_usage(
            step,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            estimated=usage.estimated,
            model=model,
            latency_ms=_elapsed_ms(started_at),
            cached_tokens=usage.cached_tokens,
            cache_hit=False,
            retry_count=retry_count,
            breaker_state=policy_result.breaker_state,
            retry_after_ms=policy_result.retry_after_ms,
            throttle_wait_ms=policy_result.throttle_wait_ms,
            final_error_type=exc.__class__.__name__,
            error_type=exc.__class__.__name__,
        )
        raise

    cost_tracker.add_llm_usage(
        step,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        estimated=usage.estimated,
        model=model,
        latency_ms=_elapsed_ms(started_at),
        cached_tokens=usage.cached_tokens,
        cache_hit=False,
        retry_count=retry_count,
        breaker_state=policy_result.breaker_state,
        retry_after_ms=policy_result.retry_after_ms,
        throttle_wait_ms=policy_result.throttle_wait_ms,
    )
    if cacheable:
        _store_cache(cache_key, content)
    return parsed


def clear_llm_response_cache():
    global _CACHE_NAMESPACE_MEMO
    _LLM_RESPONSE_CACHE.clear()
    _CACHE_NAMESPACE_MEMO = None


def clear_model_circuit_breakers():
    _MODEL_CIRCUITS.clear()


def configure_llm_resilience_for_tests(
    *,
    max_retries: int,
    retry_backoff_seconds: float,
    concurrency_limit: int,
    quota_circuit_breaker_seconds: int,
    rate_limit_breaker_threshold: int = 3,
    error_window_seconds: int = 60,
    error_window_min_calls: int = 20,
    error_rate_threshold: float = 0.5,
    max_calls_per_request: int = 40,
):
    global _LLM_RESILIENCE_SETTINGS
    _LLM_RESILIENCE_SETTINGS = LLMResilienceSettings(
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        concurrency_limit=concurrency_limit,
        quota_circuit_breaker_seconds=quota_circuit_breaker_seconds,
        rate_limit_breaker_threshold=rate_limit_breaker_threshold,
        error_window_seconds=error_window_seconds,
        error_window_min_calls=error_window_min_calls,
        error_rate_threshold=error_rate_threshold,
        max_calls_per_request=max_calls_per_request,
    )
    _MODEL_SEMAPHORES.clear()
    _MODEL_CIRCUITS.clear()


def reset_llm_resilience_for_tests():
    global _LLM_RESILIENCE_SETTINGS
    _LLM_RESILIENCE_SETTINGS = None
    _MODEL_SEMAPHORES.clear()
    _MODEL_CIRCUITS.clear()


def set_llm_request_call_budget(max_calls: int | None) -> Token[tuple[int | None, int]]:
    return _LLM_REQUEST_CALL_BUDGET.set((max_calls, 0))


def reset_llm_request_call_budget(token: Token[tuple[int | None, int]]):
    _LLM_REQUEST_CALL_BUDGET.reset(token)


def set_llm_cache_context_namespace(namespace: str) -> Token[str]:
    return _LLM_CACHE_CONTEXT_NAMESPACE.set(namespace)


def reset_llm_cache_context_namespace(token: Token[str]):
    _LLM_CACHE_CONTEXT_NAMESPACE.reset(token)


def _model_name(llm: Any) -> str | None:
    return (
        getattr(llm, "model_name", None)
        or getattr(llm, "model", None)
        or getattr(llm, "model_id", None)
    )


async def invoke_llm_with_policy(
    llm,
    prompt_value,
    timeout_seconds: int,
    policy_key: str,
):
    """Apply model-level concurrency, retry, and circuit-breaker policy."""

    settings = _resilience_settings(llm)
    circuit = _circuit_state(policy_key)
    breaker_state = _current_breaker_state(circuit)
    circuit_error = _model_circuit_error(policy_key, circuit)
    if circuit_error is not None:
        raise circuit_error

    retry_count = 0
    retry_after_ms = None
    throttle_wait_ms = 0.0
    async with _model_semaphore(policy_key, settings):
        while True:
            try:
                _consume_llm_call_budget(settings)
                message = await ainvoke_with_timeout(
                    llm.ainvoke(prompt_value), timeout_seconds
                )
                _record_success(policy_key, settings)
                return LLMPolicyResult(
                    message=message,
                    retry_count=retry_count,
                    breaker_state=breaker_state,
                    retry_after_ms=retry_after_ms,
                    throttle_wait_ms=throttle_wait_ms,
                )
            except Exception as exc:
                if isinstance(exc, LLMCallBudgetExceeded):
                    setattr(exc, "retry_count", retry_count)
                    setattr(exc, "breaker_state", breaker_state)
                    setattr(exc, "retry_after_ms", retry_after_ms)
                    setattr(exc, "throttle_wait_ms", throttle_wait_ms)
                    raise
                if _is_quota_error(exc):
                    _open_model_circuit(policy_key, settings)
                    _record_final_failure(policy_key, exc, settings)
                    setattr(exc, "retry_count", retry_count)
                    setattr(exc, "breaker_state", "open")
                    setattr(exc, "retry_after_ms", retry_after_ms)
                    setattr(exc, "throttle_wait_ms", throttle_wait_ms)
                    raise
                _record_failed_attempt(policy_key, exc, settings)
                if _current_breaker_state(circuit) == "open":
                    setattr(exc, "retry_count", retry_count)
                    setattr(exc, "breaker_state", "open")
                    setattr(exc, "retry_after_ms", retry_after_ms)
                    setattr(exc, "throttle_wait_ms", throttle_wait_ms)
                    raise
                if retry_count >= settings.max_retries:
                    _record_final_failure(policy_key, exc, settings)
                    setattr(exc, "retry_count", retry_count)
                    setattr(exc, "breaker_state", _current_breaker_state(circuit))
                    setattr(exc, "retry_after_ms", retry_after_ms)
                    setattr(exc, "throttle_wait_ms", throttle_wait_ms)
                    raise
                if not _is_retryable_error(exc):
                    _record_final_failure(policy_key, exc, settings)
                    setattr(exc, "retry_count", retry_count)
                    setattr(exc, "breaker_state", _current_breaker_state(circuit))
                    setattr(exc, "retry_after_ms", retry_after_ms)
                    setattr(exc, "throttle_wait_ms", throttle_wait_ms)
                    raise
                retry_count += 1
                retry_delay = _retry_delay(exc, retry_count, settings)
                retry_after = _retry_after_seconds(exc)
                if retry_after is not None:
                    retry_after_ms = retry_after * 1000
                throttle_wait_ms += retry_delay * 1000
                await asyncio.sleep(retry_delay)


def _cache_key(
    model: str | None,
    step: str,
    prompt_text: str,
    *,
    cache_namespace: str | None = None,
) -> str:
    namespace = cache_namespace or "\n".join(
        (_runtime_cache_namespace(), _LLM_CACHE_CONTEXT_NAMESPACE.get())
    )
    payload = f"{namespace}\n{model or ''}\n{step}\n{prompt_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _runtime_cache_namespace() -> str:
    """Version boundary for process-local LLM cache.

    Prompt text is already part of the key. These hashes cover policy and metadata
    files that can change guard/classifier behavior without changing the prompt.
    """

    global _CACHE_NAMESPACE_MEMO
    if _CACHE_NAMESPACE_MEMO is not None:
        return _CACHE_NAMESPACE_MEMO

    explicit_namespace = os.getenv("LLM_CACHE_NAMESPACE", "")
    tracked_files = (
        "conf/app_config.yaml",
        "conf/meta_config.yaml",
        "conf/policy_config.yaml",
    )
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256(explicit_namespace.encode("utf-8"))
    for relative_path in tracked_files:
        path = root / relative_path
        digest.update(relative_path.encode("utf-8"))
        if path.exists():
            digest.update(path.read_bytes())
    _CACHE_NAMESPACE_MEMO = digest.hexdigest()
    return _CACHE_NAMESPACE_MEMO


def _touch_cache(key: str) -> str:
    _LLM_RESPONSE_CACHE.move_to_end(key)
    return _LLM_RESPONSE_CACHE[key]


def _store_cache(key: str, content: str):
    _LLM_RESPONSE_CACHE[key] = content
    _LLM_RESPONSE_CACHE.move_to_end(key)
    while len(_LLM_RESPONSE_CACHE) > _MAX_LLM_RESPONSE_CACHE_ENTRIES:
        _LLM_RESPONSE_CACHE.popitem(last=False)


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)


def _model_policy_key(llm) -> str:
    provider = (
        getattr(llm, "model_provider", None)
        or getattr(llm, "provider", None)
        or "__unknown_provider__"
    )
    base_url = (
        getattr(llm, "base_url", None)
        or getattr(llm, "openai_api_base", None)
        or getattr(llm, "api_base", None)
        or "__unknown_base_url__"
    )
    return "|".join([str(provider), str(base_url), str(_model_name(llm) or "")])


def _resilience_settings(llm=None) -> LLMResilienceSettings:
    if _LLM_RESILIENCE_SETTINGS is not None:
        return _LLM_RESILIENCE_SETTINGS
    model = _model_name(llm) if llm is not None else None
    max_retries = app_config.llm.max_retries
    concurrency_limit = app_config.llm.concurrency_limit
    quota_circuit_breaker_seconds = app_config.llm.quota_circuit_breaker_seconds
    if model == app_config.llm.fast_model:
        max_retries = app_config.llm.fast_max_retries
        concurrency_limit = app_config.llm.fast_concurrency_limit
        quota_circuit_breaker_seconds = app_config.llm.fast_quota_circuit_breaker_seconds
    elif model == app_config.llm.model:
        max_retries = app_config.llm.sql_max_retries
        concurrency_limit = app_config.llm.sql_concurrency_limit
        quota_circuit_breaker_seconds = app_config.llm.sql_quota_circuit_breaker_seconds
    return LLMResilienceSettings(
        max_retries=max_retries,
        retry_backoff_seconds=app_config.llm.retry_backoff_seconds,
        concurrency_limit=max(1, concurrency_limit),
        quota_circuit_breaker_seconds=quota_circuit_breaker_seconds,
        rate_limit_breaker_threshold=app_config.llm.rate_limit_breaker_threshold,
        error_window_seconds=app_config.llm.error_window_seconds,
        error_window_min_calls=app_config.llm.error_window_min_calls,
        error_rate_threshold=app_config.llm.error_rate_threshold,
        max_calls_per_request=app_config.llm.max_calls_per_request,
    )


def _model_semaphore(
    policy_key: str, settings: LLMResilienceSettings
) -> asyncio.Semaphore:
    if policy_key not in _MODEL_SEMAPHORES:
        _MODEL_SEMAPHORES[policy_key] = asyncio.Semaphore(
            max(1, settings.concurrency_limit)
        )
    return _MODEL_SEMAPHORES[policy_key]


def _circuit_state(policy_key: str) -> CircuitState:
    return _MODEL_CIRCUITS.setdefault(policy_key, CircuitState())


def _current_breaker_state(circuit: CircuitState) -> str:
    if circuit.state == "open" and time.monotonic() >= circuit.open_until:
        return "half_open"
    return circuit.state


def _model_circuit_error(
    policy_key: str, circuit: CircuitState
) -> ModelCircuitOpenError | None:
    now = time.monotonic()
    if circuit.state == "closed":
        return None
    if circuit.state == "open" and now >= circuit.open_until:
        circuit.state = "half_open"
    if circuit.state == "half_open" and not circuit.half_open_probe_in_flight:
        circuit.half_open_probe_in_flight = True
        return None
    error = ModelCircuitOpenError(f"model {policy_key} circuit breaker is open")
    setattr(error, "breaker_state", circuit.state)
    return error


def _open_model_circuit(policy_key: str, settings: LLMResilienceSettings):
    circuit = _circuit_state(policy_key)
    circuit.state = "open"
    circuit.open_until = time.monotonic() + settings.quota_circuit_breaker_seconds
    circuit.half_open_probe_in_flight = False


def _is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    retryable_markers = (
        "timeout",
        "temporarily",
        "rate limit",
        "too many requests",
        "service unavailable",
        "connection",
    )
    return any(marker in name or marker in message for marker in retryable_markers)


def _is_quota_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    message = str(exc).lower()
    return status_code == 403 and (
        "quota" in message
        or "allocationquota" in message
        or "free tier" in message
        or "insufficient" in message
    )


def _retry_delay(
    exc: Exception, retry_count: int, settings: LLMResilienceSettings
) -> float:
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return retry_after
    base_delay = settings.retry_backoff_seconds
    if base_delay <= 0:
        return 0
    exponential_delay = base_delay * (2 ** (retry_count - 1))
    return exponential_delay + random.uniform(0, base_delay)


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after is None:
        return None
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        return None


def _record_success(policy_key: str, settings: LLMResilienceSettings):
    circuit = _circuit_state(policy_key)
    circuit.consecutive_rate_limit_errors = 0
    circuit.half_open_probe_in_flight = False
    circuit.state = "closed"
    _append_window_result(circuit, True, settings)


def _record_failed_attempt(
    policy_key: str, exc: Exception, settings: LLMResilienceSettings
):
    circuit = _circuit_state(policy_key)
    if _is_rate_limit_error(exc):
        circuit.consecutive_rate_limit_errors += 1
    else:
        circuit.consecutive_rate_limit_errors = 0
    if circuit.consecutive_rate_limit_errors >= settings.rate_limit_breaker_threshold:
        _open_model_circuit(policy_key, settings)


def _record_final_failure(
    policy_key: str, exc: Exception, settings: LLMResilienceSettings
):
    circuit = _circuit_state(policy_key)
    circuit.half_open_probe_in_flight = False
    if circuit.state == "half_open":
        _open_model_circuit(policy_key, settings)
        return
    _append_window_result(circuit, False, settings)
    if _window_error_rate(circuit, settings) > settings.error_rate_threshold:
        _open_model_circuit(policy_key, settings)


def _append_window_result(
    circuit: CircuitState, success: bool, settings: LLMResilienceSettings
):
    now = time.monotonic()
    circuit.recent_results.append((now, success))
    cutoff = now - settings.error_window_seconds
    while circuit.recent_results and circuit.recent_results[0][0] < cutoff:
        circuit.recent_results.popleft()


def _window_error_rate(
    circuit: CircuitState, settings: LLMResilienceSettings
) -> float:
    if len(circuit.recent_results) < settings.error_window_min_calls:
        return 0.0
    failures = sum(1 for _, success in circuit.recent_results if not success)
    return failures / len(circuit.recent_results)


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    message = str(exc).lower()
    return status_code == 429 or "rate limit" in message or "too many requests" in message


def _consume_llm_call_budget(settings: LLMResilienceSettings):
    explicit_limit, used = _LLM_REQUEST_CALL_BUDGET.get()
    limit = explicit_limit
    if limit is None:
        limit = settings.max_calls_per_request
    if limit is not None and used >= limit:
        raise LLMCallBudgetExceeded("LLM call budget exceeded")
    _LLM_REQUEST_CALL_BUDGET.set((explicit_limit, used + 1))
