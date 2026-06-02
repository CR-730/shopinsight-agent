"""Pre-RAG guard: intent safety, prompt-injection, and coarse routing."""

import re
from typing import Any, Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.conf.policy_config import load_policy_config
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


class PreRagGuardDecision(BaseModel):
    """Structured decision returned by the LLM classifier."""

    is_prompt_injection: bool = Field(
        description="Whether the user question is a prompt-injection attempt."
    )
    attack_type: Literal[
        "none",
        "direct",
        "indirect",
        "dangerous_operation",
        "privacy_detail",
        "system_leak",
        "out_of_scope",
        "incomplete",
        "classifier_error",
    ] = Field(description="The primary risk category.")
    risk_level: Literal["low", "medium", "high"] = Field(
        description="Risk severity of the detected intent."
    )
    should_block: bool = Field(
        description="Whether the request should be blocked before retrieval."
    )
    reason: str = Field(description="A concise Chinese reason.")


async def pre_rag_guard(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """Block unsafe requests before retrieval and metadata access."""

    writer = runtime.stream_writer
    step = "RAG前安全闸门"
    writer({"type": "progress", "step": step, "status": "running"})

    query = state.get("query") or ""
    rule_error = validate_query_by_rules(query)
    if rule_error:
        logger.warning(f"{step} rule blocked query: {rule_error}")
        writer({"type": "progress", "step": step, "status": "blocked", "error": rule_error})
        return {"safety_error": rule_error, "blocked_by": "pre_rag_guard"}

    classifier_result = await classify_query_intent(query, runtime)
    if _should_block_classifier_result(classifier_result):
        reason = classifier_result.get("reason") or "RAG 前安全分类器判定应拦截"
        logger.warning(f"{step} classifier blocked query: {classifier_result}")
        writer({"type": "progress", "step": step, "status": "blocked", "error": reason})
        return {"safety_error": reason, "blocked_by": "pre_rag_guard"}

    writer({"type": "progress", "step": step, "status": "success"})
    return {"safety_error": None}


def validate_query_by_rules(query: str) -> str | None:
    lowered = query.lower()
    patterns = load_policy_config().get("pre_rag", {}).get("rule_patterns", {})
    for attack_type, pattern in patterns.items():
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return f"RAG 前规则拦截：{attack_type}"
    return None


async def classify_query_intent(
    query: str, runtime: Runtime[DataAgentContext]
) -> dict[str, Any]:
    parser = PydanticOutputParser(pydantic_object=PreRagGuardDecision)
    prompt = PromptTemplate(
        template=load_prompt("pre_rag_guard"),
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    try:
        result = await ainvoke_llm_with_usage(
            prompt,
            llm,
            parser,
            {"query": query},
            "RAG前安全分类",
            runtime.context["cost_tracker"],
            app_config.llm.timeout_seconds,
        )
        return result.model_dump()
    except Exception as exc:
        logger.warning(f"RAG 前安全分类失败，降级为规则结果: {exc}")
        return {
            "is_prompt_injection": False,
            "attack_type": "classifier_error",
            "risk_level": "high",
            "should_block": True,
            "reason": "RAG 前安全分类器失败，已按保守策略阻断",
        }


def _should_block_classifier_result(result: dict[str, Any]) -> bool:
    risk_level = result.get("risk_level")
    attack_type = result.get("attack_type")
    if result.get("is_prompt_injection") is True and risk_level == "high":
        return True
    if attack_type in {"out_of_scope", "incomplete"}:
        return result.get("should_block") is True and risk_level in {"medium", "high"}
    return result.get("should_block") is True
