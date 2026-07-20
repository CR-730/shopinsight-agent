"""Intent recognition and coarse input guard before retrieval."""

from typing import Any, Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field, model_validator

from app.agent.context import DataAgentContext
from app.agent.failure import build_failure
from app.agent.llm import llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


class InputGuardDecision(BaseModel):
    """Minimal pre-retrieval routing decision returned by the LLM guard."""

    decision: Literal["allow", "block"] = Field(
        description="Allow the query to continue or block it before retrieval."
    )
    category: Literal[
        "safe",
        "missing_query_object",
        "clearly_non_data",
        "prompt_injection",
        "dangerous_operation",
        "privacy_detail",
        "system_leak",
    ] = Field(description="The single routing category supporting the decision.")
    user_message: str = Field(
        default="",
        description="A concise Chinese clarification or refusal; empty when allowed.",
    )

    @model_validator(mode="after")
    def validate_decision_category(self):
        if self.decision == "allow" and self.category != "safe":
            raise ValueError("allow decision requires category=safe")
        if self.decision == "block" and self.category == "safe":
            raise ValueError("block decision requires a blocking category")
        if self.decision == "block" and not self.user_message.strip():
            raise ValueError("block decision requires user_message")
        return self


# One-version import compatibility for the old public class name.
IntentRecognitionDecision = InputGuardDecision


async def intent_recognition(
    state: DataAgentState, runtime: Runtime[DataAgentContext]
):
    """Recognize user intent and block unsafe or out-of-scope requests."""

    writer = runtime.stream_writer
    step = "意图识别"
    writer({"type": "progress", "step": step, "status": "running"})

    query = state.get("query") or ""
    classifier_result = await classify_query_intent(query, runtime)
    if _is_block_decision(classifier_result):
        category = str(classifier_result.get("category") or "input_blocked")
        user_facing_message = str(
            classifier_result.get("user_message") or ""
        ).strip()
        if user_facing_message:
            _write_answer_delta(writer, "\n\n" + user_facing_message)
        logger.warning("{} classifier blocked query: {}", step, classifier_result)
        writer(
            {
                "type": "progress",
                "step": step,
                "status": "blocked",
                "error": category,
            }
        )
        return {
            "failure": build_failure(
                category="input_guard",
                stage="intent_recognition",
                code=category,
                message=user_facing_message or "intent_recognition blocked query",
                disposition="blocked",
                user_message=user_facing_message,
            )
        }

    writer({"type": "progress", "step": step, "status": "success"})
    return {"failure": None}


async def classify_query_intent(
    query: str, runtime: Runtime[DataAgentContext]
) -> dict[str, Any]:
    parser = PydanticOutputParser(pydantic_object=InputGuardDecision)
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
            "意图识别",
            runtime.context["cost_tracker"],
            app_config.llm.timeout_seconds,
            cacheable=not _ablation_options(runtime.context).get(
                "disable_non_sql_llm_cache"
            ),
        )
        return result.model_dump()
    except Exception as exc:
        logger.warning("意图识别失败，保守阻断：{}", exc)
        return {
            "decision": "block",
            "category": "classifier_error",
            "user_message": (
                "我现在没能可靠判断这个问题是否可以进入问数流程，请稍后再试。"
            ),
        }


def _is_block_decision(result: dict[str, Any]) -> bool:
    return result.get("decision") == "block"


def _ablation_options(context: dict[str, Any]) -> dict[str, Any]:
    return dict(context.get("ablation_options") or {})


def _write_answer_delta(writer, text: str) -> None:
    content = str(text or "")
    for index in range(0, len(content), 12):
        writer({"type": "answer_delta", "delta": content[index : index + 12]})
