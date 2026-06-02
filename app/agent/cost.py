"""Token 用量和成本统计。"""

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated: bool = False
    cached_tokens: int = 0


@dataclass
class CostRates:
    llm_input_per_1m_tokens: float = 0.0
    llm_output_per_1m_tokens: float = 0.0
    embedding_per_1m_tokens: float = 0.0
    currency: str = "CNY"


@dataclass
class CostTracker:
    rates: CostRates
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_total_tokens: int = 0
    embedding_tokens: int = 0
    embedding_estimated: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)

    def add_node_event(
        self,
        step: str,
        latency_ms: float,
        error_type: str | None = None,
    ):
        self.calls.append(
            {
                "type": "node",
                "step": step,
                "latency_ms": latency_ms,
                "error_type": error_type,
            }
        )

    def add_llm_usage(
        self,
        step: str,
        input_tokens: int,
        output_tokens: int,
        estimated: bool = False,
        model: str | None = None,
        latency_ms: float | None = None,
        cached_tokens: int = 0,
        cache_hit: bool = False,
        retry_count: int = 0,
        breaker_state: str = "closed",
        retry_after_ms: float | None = None,
        throttle_wait_ms: float | None = None,
        final_error_type: str | None = None,
        error_type: str | None = None,
    ):
        total_tokens = input_tokens + output_tokens
        cost = (
            input_tokens * self.rates.llm_input_per_1m_tokens
            + output_tokens * self.rates.llm_output_per_1m_tokens
        ) / 1_000_000
        self.llm_input_tokens += input_tokens
        self.llm_output_tokens += output_tokens
        self.llm_total_tokens += total_tokens
        self.calls.append(
            {
                "type": "llm",
                "step": step,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cached_tokens": cached_tokens,
                "cost": round(cost, 8),
                "estimated": estimated,
                "model": model,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "retry_count": retry_count,
                "breaker_state": breaker_state,
                "retry_after_ms": retry_after_ms,
                "throttle_wait_ms": throttle_wait_ms,
                "final_error_type": final_error_type or error_type,
                "error_type": error_type,
            }
        )

    def add_embedding_usage(
        self,
        step: str,
        tokens: int,
        estimated: bool = True,
        model: str | None = None,
        latency_ms: float | None = None,
        cache_hit: bool = False,
        error_type: str | None = None,
    ):
        cost = tokens * self.rates.embedding_per_1m_tokens / 1_000_000
        self.embedding_tokens += tokens
        self.embedding_estimated = self.embedding_estimated or estimated
        self.calls.append(
            {
                "type": "embedding",
                "step": step,
                "tokens": tokens,
                "cost": round(cost, 8),
                "estimated": estimated,
                "model": model,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "error_type": error_type,
            }
        )

    def summary(self) -> dict[str, Any]:
        llm_cost = (
            self.llm_input_tokens * self.rates.llm_input_per_1m_tokens
            + self.llm_output_tokens * self.rates.llm_output_per_1m_tokens
        ) / 1_000_000
        embedding_cost = (
            self.embedding_tokens * self.rates.embedding_per_1m_tokens / 1_000_000
        )
        return {
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "llm_total_tokens": self.llm_total_tokens,
            "embedding_tokens": self.embedding_tokens,
            "llm_cost": round(llm_cost, 8),
            "embedding_cost": round(embedding_cost, 8),
            "total_cost": round(llm_cost + embedding_cost, 8),
            "currency": self.rates.currency,
            "embedding_estimated": self.embedding_estimated,
            "calls": self.calls,
        }


def extract_token_usage(message: Any) -> TokenUsage:
    usage_metadata = getattr(message, "usage_metadata", None)
    if usage_metadata:
        input_tokens = int(usage_metadata.get("input_tokens") or 0)
        output_tokens = int(usage_metadata.get("output_tokens") or 0)
        total_tokens = int(
            usage_metadata.get("total_tokens") or input_tokens + output_tokens
        )
        cached_tokens = _cached_tokens_from_usage_metadata(usage_metadata)
        return TokenUsage(
            input_tokens,
            output_tokens,
            total_tokens,
            estimated=False,
            cached_tokens=cached_tokens,
        )

    response_metadata = getattr(message, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or {}
    if token_usage:
        input_tokens = int(token_usage.get("prompt_tokens") or 0)
        output_tokens = int(token_usage.get("completion_tokens") or 0)
        total_tokens = int(
            token_usage.get("total_tokens") or input_tokens + output_tokens
        )
        cached_tokens = _cached_tokens_from_response_metadata(response_metadata)
        return TokenUsage(
            input_tokens,
            output_tokens,
            total_tokens,
            estimated=False,
            cached_tokens=cached_tokens,
        )

    return TokenUsage(estimated=True)


def _cached_tokens_from_usage_metadata(usage_metadata: dict) -> int:
    input_token_details = usage_metadata.get("input_token_details") or {}
    return int(
        input_token_details.get("cache_read")
        or input_token_details.get("cached_tokens")
        or 0
    )


def _cached_tokens_from_response_metadata(response_metadata: dict) -> int:
    token_usage = response_metadata.get("token_usage") or {}
    prompt_details = token_usage.get("prompt_tokens_details") or {}
    return int(prompt_details.get("cached_tokens") or 0)


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数，用于 Embedding 服务不返回 usage 的情况。"""

    if not text:
        return 0
    return max(1, math.ceil(len(text) / 2))
