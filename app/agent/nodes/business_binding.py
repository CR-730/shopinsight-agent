"""Business binding node: extract candidates, validate with metadata, block unresolved."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from app.agent.business_binding.candidates import (
    BindingCandidates,
    extract_binding_candidates,
)
from app.agent.business_binding.validator import (
    BindingValidationContext,
    validate_binding_candidates,
    validate_business_binding_state,
    validated_enum_values,
)
from app.agent.llm import llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.agent.memory import sliding_conversation_history
from app.conf.app_config import app_config
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

    from app.agent.context import DataAgentContext
    from app.agent.state import DataAgentState


async def business_binding(
    state: DataAgentState, runtime: Runtime[DataAgentContext]
) -> dict[str, Any]:
    """Bind user language to canonical metrics, filters, and time constraints."""

    writer = runtime.stream_writer
    step = "业务绑定"
    writer({"type": "progress", "step": step, "status": "running"})

    query = state.get("query") or ""
    enum_aliases = _value_alias_map(
        await runtime.context["meta_mysql_repository"].list_value_aliases()
    )

    # QueryService 可能已经提前抽取过候选，用于前端流式展示“理解问题”。
    # 这里复用候选以避免重复 LLM 调用，但下面的绑定结果只信元数据、
    # RAG 召回和 DW 校验，不直接信候选里的自然语言判断。
    if state.get("binding_candidates"):
        candidates = BindingCandidates.model_validate(state["binding_candidates"])
    else:
        candidates = await extract_binding_candidates(
            query,
            runtime,
            conversation_history=sliding_conversation_history(
                state.get("conversation_history") or ""
            ),
            metric_infos=state.get("metric_infos") or [],
            retrieved_value_infos=state.get("retrieved_value_infos") or [],
            enum_aliases=enum_aliases,
        )
        _write_answer_delta(writer, candidates.user_response)
    binding = await validate_binding_candidates(
        candidates,
        BindingValidationContext(
            metric_infos=state.get("metric_infos") or [],
            table_infos=state.get("table_infos") or [],
            retrieved_value_infos=state.get("retrieved_value_infos") or [],
            enum_aliases=enum_aliases,
            dw_mysql_repository=runtime.context["dw_mysql_repository"],
        ),
    )
    update = {
        "business_binding": binding,
        "metric_bindings": binding["metrics"],
        "resolved_filters": binding["filters"],
        "groupby_bindings": binding.get("groups") or [],
        "time_binding": binding["time"],
        "validated_enum_values": validated_enum_values(binding["filters"]),
        "unresolved_bindings": binding["unresolved"],
        "ambiguous_bindings": binding["ambiguous"],
        "safety_error": None,
    }

    logger.info(f"业务绑定结果：{binding}")
    rule_error = validate_business_binding_state(update)
    if rule_error:
        logger.warning(f"{step} blocked query: {rule_error}")
        user_facing_message = await _blocked_binding_response(
            query=query,
            binding=binding,
            rule_error=rule_error,
            runtime=runtime,
        )
        if user_facing_message:
            _write_answer_delta(writer, "\n\n" + user_facing_message)
        writer(
            {"type": "progress", "step": step, "status": "blocked", "error": rule_error}
        )
        blocked_update = {
            **update,
            "safety_error": rule_error,
            "blocked_by": "business_binding",
        }
        if user_facing_message:
            blocked_update["user_facing_message"] = user_facing_message
        return blocked_update
    writer({"type": "progress", "step": step, "status": "success"})
    return update


def _value_alias_map(value_aliases: list[Any]) -> dict[str, dict[str, str]]:
    aliases: dict[str, dict[str, str]] = {}
    for value_alias in value_aliases:
        column_id = str(getattr(value_alias, "column_id", "") or "")
        alias = str(getattr(value_alias, "alias", "") or "")
        canonical_value = str(getattr(value_alias, "canonical_value", "") or "")
        if column_id and alias and canonical_value:
            aliases.setdefault(column_id, {})[alias] = canonical_value
    return aliases


def _write_answer_delta(writer, text: str) -> None:
    content = str(text or "")
    if not content.strip():
        return
    for index in range(0, len(content), 12):
        writer({"type": "answer_delta", "delta": content[index : index + 12]})


async def _blocked_binding_response(
    *,
    query: str,
    binding: dict[str, Any],
    rule_error: str,
    runtime,
) -> str:
    prompt = PromptTemplate.from_template(load_prompt("business_binding_clarification"))
    try:
        response = await ainvoke_llm_with_usage(
            prompt,
            llm,
            StrOutputParser(),
            {
                "query": query,
                "bound": _bound_summary(binding),
                "unresolved": _unresolved_summary(binding),
                "rule_error": rule_error,
            },
            "业务绑定澄清回复",
            runtime.context["cost_tracker"],
            app_config.llm.timeout_seconds,
            cacheable=False,
        )
        return _sanitize_user_facing_response(response)
    except Exception as exc:
        logger.warning(f"业务绑定澄清回复生成失败，交给通用错误态处理: {exc}")
        return ""


def _bound_summary(binding: dict[str, Any]) -> str:
    metrics = [
        str(item.get("raw_mention") or item.get("canonical_metric") or "").strip()
        for item in binding.get("metrics") or []
    ]
    filters = [
        str(item.get("raw_value") or item.get("canonical_value") or "").strip()
        for item in binding.get("filters") or []
    ]
    values = [item for item in [*metrics, *filters] if item]
    return "、".join(values) if values else "无"


def _unresolved_summary(binding: dict[str, Any]) -> str:
    values = [
        str(item.get("raw_text") or "").strip()
        for item in binding.get("unresolved") or []
        if str(item.get("raw_text") or "").strip()
    ]
    return "、".join(values) if values else "无"


def _sanitize_user_facing_response(response: str) -> str:
    text = str(response or "").strip().strip("`")
    banned = ("business_binding", "metric_not_bound", "reason=", "unresolved:")
    if not text:
        raise ValueError("empty business binding clarification response")
    if any(item in text for item in banned):
        raise ValueError("business binding clarification leaked internal details")
    return text
