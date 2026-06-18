"""Pre-RAG guard: LLM intent safety and coarse routing."""

from typing import Any, Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.agent.state import DataAgentState
from app.agent.stop_signal import split_stop_signal
from app.conf.app_config import app_config
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
    """Block unsafe or out-of-scope requests before retrieval."""

    writer = runtime.stream_writer
    step = "RAG前安全闸门"
    writer({"type": "progress", "step": step, "status": "running"})

    query = state.get("query") or ""
    classifier_result = await classify_query_intent(query, runtime)
    if _should_block_classifier_result(classifier_result):
        reason = str(classifier_result.get("reason") or "").strip()
        user_facing_message, _ = split_stop_signal(reason)
        if user_facing_message:
            _write_answer_delta(writer, "\n\n" + user_facing_message)
        logger.warning("%s classifier blocked query: %s", step, classifier_result)
        writer({"type": "progress", "step": step, "status": "blocked", "error": reason})
        return {
            "safety_error": reason or "pre_rag_guard blocked query",
            "blocked_by": "pre_rag_guard",
            "user_facing_message": user_facing_message,
        }

    writer({"type": "progress", "step": step, "status": "success"})
    return {"safety_error": None}


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
            cacheable=not _ablation_options(runtime.context).get(
                "disable_non_sql_llm_cache"
            ),
        )
        return result.model_dump()
    except Exception as exc:
        logger.warning("RAG 前安全分类失败，保守阻断：%s", exc)
        return {
            "is_prompt_injection": False,
            "attack_type": "classifier_error",
            "risk_level": "high",
            "should_block": True,
            "reason": "我现在没能可靠判断这个问题是否可以进入问数流程，请稍后再试。find_error",
        }


def _should_block_classifier_result(result: dict[str, Any]) -> bool:
    risk_level = result.get("risk_level")
    attack_type = result.get("attack_type")
    if result.get("is_prompt_injection") is True and risk_level == "high":
        return True
    if attack_type in {"out_of_scope", "incomplete"}:
        return result.get("should_block") is True and risk_level in {"medium", "high"}
    return result.get("should_block") is True


def _ablation_options(context: dict[str, Any]) -> dict[str, Any]:
    return dict(context.get("ablation_options") or {})


def _write_answer_delta(writer, text: str) -> None:
    content = str(text or "")
    for index in range(0, len(content), 12):
        writer({"type": "answer_delta", "delta": content[index : index + 12]})
