"""LLM-backed candidate extraction for business binding."""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, ConfigDict, Field

from app.agent.llm import llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.conf.app_config import app_config
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


class MetricMention(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw_text: str = Field(default="")
    normalized_text: str = Field(default="")


class FilterMention(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw_text: str = Field(default="")
    field_hint: str = Field(default="")
    normalized_text: str = Field(default="")


class TimeMention(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw_text: str = Field(default="")
    granularity_hint: str = Field(default="")
    normalized_text: str = Field(default="")


class GroupByMention(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw_text: str = Field(default="")
    field_hint: str = Field(default="")
    normalized_text: str = Field(default="")


class BindingCandidates(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_query: str = Field(default="")
    user_response: str = Field(default="")
    metric_mentions: list[MetricMention] = Field(default_factory=list)
    filter_mentions: list[FilterMention] = Field(default_factory=list)
    time_mentions: list[TimeMention] = Field(default_factory=list)
    groupby_mentions: list[GroupByMention] = Field(default_factory=list)


async def extract_binding_candidates(
    query: str,
    runtime,
    *,
    conversation_history: str = "",
    metric_infos: list[dict[str, Any]],
    retrieved_value_infos: list[Any],
    enum_aliases: dict[str, dict[str, str]],
    table_infos: list[dict[str, Any]] | None = None,
) -> BindingCandidates:
    parser = PydanticOutputParser(pydantic_object=BindingCandidates)
    prompt = PromptTemplate(
        template=load_prompt("binding_candidate_extractor"),
        input_variables=["query", "conversation_history"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    try:
        metric_infos_context = ", ".join(
            [f"{m.get('name')}(别名: {','.join(m.get('alias', []) or [])})" for m in metric_infos]
        )
        
        enum_list = []
        for aliases in enum_aliases.values():
            for alias, canon in aliases.items():
                enum_list.append(f"{alias}->{canon}")
        for v in retrieved_value_infos:
            val = str(getattr(v, "value", "") or "")
            if val:
                enum_list.append(val)
        enum_aliases_context = ", ".join(set(enum_list))
        
        time_examples_context = ""
        if table_infos:
            for t in table_infos:
                if t.get("name") == "dim_date":
                    for col in t.get("columns", []):
                        if col.get("name") in ["quarter", "month", "year"]:
                            time_examples_context += f"{col.get('name')}示例: {col.get('examples', [])}\n"

        result = await ainvoke_llm_with_usage(
            prompt,
            llm,
            parser,
            {
                "query": query,
                "conversation_history": conversation_history or "无",
                "metric_infos_context": metric_infos_context or "无",
                "enum_aliases_context": enum_aliases_context or "无",
                "time_examples_context": time_examples_context or "无",
            },
            "业务候选抽取",
            runtime.context["cost_tracker"],
            app_config.llm.timeout_seconds,
            cacheable=not _ablation_options(runtime.context).get(
                "disable_non_sql_llm_cache"
            ),
        )
        result.source_query = query
        return result
    except Exception as exc:
        logger.warning(f"业务候选抽取失败，降级为显式元数据/RAG命中: {exc}")
        return fallback_binding_candidates(
            query=query,
            metric_infos=metric_infos,
            retrieved_value_infos=retrieved_value_infos,
            enum_aliases=enum_aliases,
        )


def fallback_binding_candidates(
    *,
    query: str,
    metric_infos: list[dict[str, Any]],
    retrieved_value_infos: list[Any],
    enum_aliases: dict[str, dict[str, str]],
) -> BindingCandidates:
    metrics = [
        MetricMention(raw_text=mention, normalized_text=mention)
        for mention in _explicit_metric_mentions(query, metric_infos)
    ]
    filters = [
        FilterMention(raw_text=mention, field_hint="", normalized_text=mention)
        for mention in _explicit_filter_mentions(query, retrieved_value_infos, enum_aliases)
    ]
    return BindingCandidates(
        source_query=query,
        metric_mentions=metrics,
        filter_mentions=filters,
    )


def _explicit_metric_mentions(query: str, metric_infos: list[dict[str, Any]]) -> list[str]:
    mentions: set[str] = set()
    for metric_info in metric_infos:
        candidates = [metric_info.get("name"), *(metric_info.get("alias") or [])]
        for candidate in candidates:
            mention = str(candidate or "")
            if mention and mention in query:
                mentions.add(mention)
    return sorted(mentions, key=len, reverse=True)


def _ablation_options(context: dict[str, Any]) -> dict[str, Any]:
    return dict(context.get("ablation_options") or {})


def _explicit_filter_mentions(
    query: str,
    retrieved_value_infos: list[Any],
    enum_aliases: dict[str, dict[str, str]],
) -> list[str]:
    mentions: set[str] = set()
    for aliases in enum_aliases.values():
        for alias in aliases:
            if alias and alias in query:
                mentions.add(str(alias))
    for value_info in retrieved_value_infos:
        value = str(getattr(value_info, "value", "") or "")
        if value and value in query:
            mentions.add(value)
    return sorted(mentions, key=len, reverse=True)
