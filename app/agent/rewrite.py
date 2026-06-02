"""Structured conversational query rewriting."""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from app.agent.llm import llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.agent.memory import ConversationSnapshot
from app.conf.app_config import app_config
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


class ConversationRewriteResult(BaseModel):
    """Structured result for standalone conversational query rewriting."""

    mode: Literal["unchanged", "rewritten", "needs_context"] = Field(
        description="Whether the query is unchanged, rewritten, or needs prior context."
    )
    standalone_query: str = Field(
        description="The independent user question to send through the normal graph."
    )
    reason: str = Field(description="Concise Chinese explanation for the rewrite mode.")
    inherited_slots: dict[str, Any] = Field(
        default_factory=dict,
        description="Slots the rewrite says it inherited; diagnostic only.",
    )
    overridden_slots: dict[str, Any] = Field(
        default_factory=dict,
        description="Slots the rewrite says the current query overrides; diagnostic only.",
    )


async def rewrite_query(
    query: str,
    snapshot: ConversationSnapshot | dict[str, Any] | None,
    cost_tracker: Any,
) -> ConversationRewriteResult:
    """Use the fast model to rewrite a conversational query into a standalone query."""

    parser = PydanticOutputParser(pydantic_object=ConversationRewriteResult)
    prompt = PromptTemplate(
        template=load_prompt("rewrite_query"),
        input_variables=["query", "snapshot"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    try:
        result = await ainvoke_llm_with_usage(
            prompt,
            llm,
            parser,
            {
                "query": query,
                "snapshot": _snapshot_for_prompt(snapshot),
            },
            "追问改写",
            cost_tracker,
            app_config.llm.timeout_seconds,
            cacheable=True,
        )
        return _enforce_rewrite_invariants(query, result)
    except Exception as exc:
        logger.warning(f"追问改写失败，按保守策略处理: {exc}")
        return ConversationRewriteResult(
            mode="needs_context",
            standalone_query=query,
            reason="追问改写失败，已按保守策略阻断",
            inherited_slots={},
            overridden_slots={},
        )


def _enforce_rewrite_invariants(
    query: str, result: ConversationRewriteResult
) -> ConversationRewriteResult:
    if result.mode in {"unchanged", "needs_context"}:
        result.standalone_query = query
        result.inherited_slots = {}
        result.overridden_slots = {}
    return result


def _snapshot_for_prompt(snapshot: ConversationSnapshot | dict[str, Any] | None) -> str:
    if snapshot is None:
        return "null"
    allowed = {
        "last_metric_bindings": snapshot.get("last_metric_bindings") or [],
        "last_resolved_filters": snapshot.get("last_resolved_filters") or [],
        "last_time_binding": snapshot.get("last_time_binding"),
        "last_answer_summary": snapshot.get("last_answer_summary"),
        "recent_turns_summary": snapshot.get("recent_turns_summary") or [],
    }
    return json.dumps(allowed, ensure_ascii=False, default=str)
