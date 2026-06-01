"""LLM invocation helpers with usage, latency, and lightweight caching."""

import hashlib
import time
from collections import OrderedDict
from typing import Any

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate

from app.agent.cached_clients import ainvoke_with_timeout
from app.agent.cost import CostTracker, extract_token_usage

_LLM_RESPONSE_CACHE: OrderedDict[str, str] = OrderedDict()
_MAX_LLM_RESPONSE_CACHE_ENTRIES = 512


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
        message = await ainvoke_with_timeout(llm.ainvoke(prompt_value), timeout_seconds)
        usage = extract_token_usage(message)
        content = message.content
        cost_tracker.add_llm_usage(
            step,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            estimated=usage.estimated,
            model=model,
            latency_ms=_elapsed_ms(started_at),
            cached_tokens=usage.cached_tokens,
            cache_hit=False,
        )
        if cacheable:
            _store_cache(cache_key, content)
        return output_parser.parse(content)
    except Exception as exc:
        cost_tracker.add_llm_usage(
            step,
            input_tokens=0,
            output_tokens=0,
            estimated=True,
            model=model,
            latency_ms=_elapsed_ms(started_at),
            cache_hit=False,
            error_type=exc.__class__.__name__,
        )
        raise


def clear_llm_response_cache():
    _LLM_RESPONSE_CACHE.clear()


def _model_name(llm: Any) -> str | None:
    return (
        getattr(llm, "model_name", None)
        or getattr(llm, "model", None)
        or getattr(llm, "model_id", None)
    )


def _cache_key(model: str | None, step: str, prompt_text: str) -> str:
    payload = f"{model or ''}\n{step}\n{prompt_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
