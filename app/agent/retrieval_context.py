"""Helpers for SQL memory recall and metadata retrieval context."""

import asyncio
import time
from typing import Any

import jieba.analyse
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from app.agent.cached_clients import ainvoke_with_timeout
from app.agent.cost import estimate_tokens
from app.agent.keyword_expansion import normalize_keyword_list
from app.agent.llm import llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.agent.memory import (
    SQL_TOOL_NAME,
    build_retrieval_query,
    format_tool_memory_results,
)
from app.agent.state import (
    ColumnInfoState,
    DataAgentState,
    MetricInfoState,
    TableInfoState,
)
from app.conf.app_config import app_config
from app.core.log import logger
from app.entities.column_info import ColumnInfo
from app.entities.metric_info import MetricInfo
from app.entities.table_info import TableInfo
from app.entities.value_info import ValueInfo
from app.prompt.prompt_loader import load_prompt
from app.retrieval.fusion import fuse_ranked_value_infos


async def recall_sql_memory_context(
    state: DataAgentState, context: dict[str, Any]
) -> dict[str, str]:
    user_id = context.get("user_id") or "anonymous"
    if user_id == "anonymous":
        return {"sql_memory_context": ""}

    try:
        memory_query = build_retrieval_query(
            state["query"], state.get("conversation_history") or ""
        )
        results = await context["agent_memory_repository"].search_similar_usage(
            memory_query,
            user_id=user_id,
            metadata_cache_version=context.get("metadata_cache_version"),
            limit=3,
            similarity_threshold=0.35,
            tool_name_filter=SQL_TOOL_NAME,
        )
        logger.info(f"SQL 记忆召回成功: {len(results)} 条")
        return {"sql_memory_context": format_tool_memory_results(results)}
    except Exception as exc:
        logger.warning(f"SQL 记忆召回失败，降级为空上下文: {exc}")
        return {"sql_memory_context": ""}


async def extract_retrieval_keywords(state: DataAgentState) -> dict[str, list[str]]:
    query = build_retrieval_query(
        state["query"], state.get("conversation_history") or ""
    )
    allow_pos = (
        "n",
        "nr",
        "ns",
        "nt",
        "nz",
        "v",
        "vn",
        "a",
        "an",
        "eng",
        "i",
        "l",
    )
    keywords = jieba.analyse.extract_tags(query, allowPOS=allow_pos)
    keywords = list(set(keywords + [query]))
    logger.info(f"抽取关键词成功: {keywords}")
    return {"keywords": keywords}


async def recall_column_context(
    state: DataAgentState, context: dict[str, Any]
) -> dict[str, list[ColumnInfo]]:
    step = "召回字段信息"
    query = state["query"]
    keywords = state["keywords"]
    column_qdrant_repository = context["column_qdrant_repository"]
    embedding_client = context["embedding_client"]

    result = await _extend_keywords(
        prompt_name="extend_keywords_for_column_recall",
        query=query,
        step=step,
        context=context,
    )
    keywords = set(normalize_keyword_list(keywords) + result)
    column_info_map: dict[str, ColumnInfo] = {}
    for keyword in keywords:
        embedding = await _embed_keyword(keyword, step, embedding_client, context)
        current_column_infos: list[ColumnInfo] = await ainvoke_with_timeout(
            column_qdrant_repository.search(
                embedding,
                meta_build_version=context.get("metadata_build_version"),
            ),
            app_config.agent.retrieval_timeout_seconds,
        )
        for column_info in current_column_infos:
            column_info_map.setdefault(column_info.id, column_info)

    return {"retrieved_column_infos": list(column_info_map.values())}


async def recall_metric_context(
    state: DataAgentState, context: dict[str, Any]
) -> dict[str, list[MetricInfo]]:
    step = "召回指标信息"
    query = state["query"]
    keywords = state["keywords"]
    embedding_client = context["embedding_client"]
    metric_qdrant_repository = context["metric_qdrant_repository"]

    result = await _extend_keywords(
        prompt_name="extend_keywords_for_metric_recall",
        query=query,
        step=step,
        context=context,
    )
    keywords = set(normalize_keyword_list(keywords) + result)
    metric_info_map: dict[str, MetricInfo] = {}
    for keyword in keywords:
        embedding = await _embed_keyword(keyword, step, embedding_client, context)
        current_metric_infos: list[MetricInfo] = await ainvoke_with_timeout(
            metric_qdrant_repository.search(
                embedding,
                meta_build_version=context.get("metadata_build_version"),
            ),
            app_config.agent.retrieval_timeout_seconds,
        )
        for metric_info in current_metric_infos:
            metric_info_map.setdefault(metric_info.id, metric_info)

    logger.info(f"检索到指标信息：{list(metric_info_map.keys())}")
    return {"retrieved_metric_infos": list(metric_info_map.values())}


async def recall_value_context(
    state: DataAgentState, context: dict[str, Any]
) -> dict[str, list[ValueInfo]]:
    step = "召回字段取值"
    query = state["query"]
    keywords = state["keywords"]
    value_es_repository = context["value_es_repository"]
    value_qdrant_repository = context["value_qdrant_repository"]
    embedding_client = context["embedding_client"]

    result = await _extend_keywords(
        prompt_name="extend_keywords_for_value_recall",
        query=query,
        step=step,
        context=context,
    )
    keywords = set(normalize_keyword_list(keywords) + result)
    value_infos_map: dict[str, ValueInfo] = {}
    for keyword in keywords:
        es_value_infos, vector_value_infos = await asyncio.gather(
            _search_values_by_es(
                value_es_repository,
                keyword,
                context.get("metadata_build_version"),
            ),
            _search_values_by_vector(
                value_qdrant_repository,
                embedding_client,
                keyword,
                context["cost_tracker"],
                context.get("metadata_build_version"),
            ),
        )
        current_value_infos = fuse_ranked_value_infos(
            {"es": es_value_infos, "vector": vector_value_infos},
            weights={
                "es": app_config.agent.value_hybrid_es_weight,
                "vector": app_config.agent.value_hybrid_vector_weight,
            },
        )
        for current_value_info in current_value_infos:
            value_infos_map.setdefault(current_value_info.id, current_value_info)

    logger.info(f"检索到字段取值：{list(value_infos_map.keys())}")
    return {"retrieved_value_infos": list(value_infos_map.values())}


async def merge_retrieved_context(
    state: DataAgentState, context: dict[str, Any]
) -> dict[str, list]:
    retrieved_column_infos: list[ColumnInfo] = state["retrieved_column_infos"]
    retrieved_metric_infos: list[MetricInfo] = state["retrieved_metric_infos"]
    retrieved_value_infos: list[ValueInfo] = state["retrieved_value_infos"]
    meta_mysql_repository = context["meta_mysql_repository"]

    retrieved_column_infos_map: dict[str, ColumnInfo] = {
        retrieved_column_info.id: retrieved_column_info
        for retrieved_column_info in retrieved_column_infos
    }

    for retrieved_metric_info in retrieved_metric_infos:
        for relevant_column in retrieved_metric_info.relevant_columns:
            if relevant_column not in retrieved_column_infos_map:
                column_info: ColumnInfo = (
                    await meta_mysql_repository.get_column_info_by_id(relevant_column)
                )
                retrieved_column_infos_map[relevant_column] = column_info

    for retrieved_value_info in retrieved_value_infos:
        value = retrieved_value_info.value
        column_id = retrieved_value_info.column_id
        if column_id not in retrieved_column_infos_map:
            column_info: ColumnInfo = await meta_mysql_repository.get_column_info_by_id(
                column_id
            )
            retrieved_column_infos_map[column_id] = column_info
        if value not in retrieved_column_infos_map[column_id].examples:
            retrieved_column_infos_map[column_id].examples.append(value)

    table_to_columns_map: dict[str, list[ColumnInfo]] = {}
    for column_info in retrieved_column_infos_map.values():
        table_to_columns_map.setdefault(column_info.table_id, []).append(column_info)

    for table_id in table_to_columns_map.keys():
        key_columns: list[
            ColumnInfo
        ] = await meta_mysql_repository.get_key_columns_by_table_id(table_id)
        column_ids = [
            column_info.id for column_info in table_to_columns_map[table_id]
        ]
        for key_column in key_columns:
            if key_column.id not in column_ids:
                table_to_columns_map[table_id].append(key_column)

    table_infos: list[TableInfoState] = []
    for table_id, column_infos in table_to_columns_map.items():
        table_info: TableInfo = await meta_mysql_repository.get_table_info_by_id(table_id)
        table_infos.append(
            TableInfoState(
                name=table_info.name,
                role=table_info.role,
                description=table_info.description,
                columns=[
                    ColumnInfoState(
                        name=column_info.name,
                        type=column_info.type,
                        role=column_info.role,
                        examples=column_info.examples,
                        description=column_info.description,
                        alias=column_info.alias,
                    )
                    for column_info in column_infos
                ],
            )
        )

    metric_infos: list[MetricInfoState] = [
        MetricInfoState(
            name=retrieved_metric_info.name,
            description=retrieved_metric_info.description,
            relevant_columns=retrieved_metric_info.relevant_columns,
            alias=retrieved_metric_info.alias,
        )
        for retrieved_metric_info in retrieved_metric_infos
    ]

    logger.info(f"合并后的表信息：{[table_info['name'] for table_info in table_infos]}")
    logger.info(f"合并后的指标信息：{[metric_info['name'] for metric_info in metric_infos]}")
    return {"table_infos": table_infos, "metric_infos": metric_infos}


async def _extend_keywords(
    *, prompt_name: str, query: str, step: str, context: dict[str, Any]
) -> list[str]:
    prompt = PromptTemplate(
        template=load_prompt(prompt_name),
        input_variables=["query"],
    )
    try:
        result = await ainvoke_llm_with_usage(
            prompt,
            llm,
            JsonOutputParser(),
            {"query": query},
            step,
            context["cost_tracker"],
            app_config.llm.timeout_seconds,
        )
    except TimeoutError:
        logger.warning(f"{step} LLM 扩展超时，降级使用原始关键词")
        return []
    return normalize_keyword_list(result)


async def _embed_keyword(keyword: str, step: str, embedding_client, context: dict[str, Any]):
    started_at = time.perf_counter()
    embedding = await ainvoke_with_timeout(
        embedding_client.aembed_query(keyword),
        app_config.agent.embedding_timeout_seconds,
    )
    context["cost_tracker"].add_embedding_usage(
        step,
        estimate_tokens(keyword),
        estimated=True,
        model=app_config.embedding.model,
        latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
        cache_hit=bool(getattr(embedding_client, "last_cache_hit", False)),
    )
    return embedding


async def _search_values_by_es(
    value_es_repository, keyword: str, metadata_build_version: str | None
):
    return await ainvoke_with_timeout(
        value_es_repository.search(keyword, meta_build_version=metadata_build_version),
        app_config.agent.retrieval_timeout_seconds,
    )


async def _search_values_by_vector(
    value_qdrant_repository,
    embedding_client,
    keyword: str,
    cost_tracker,
    metadata_build_version: str | None,
):
    embedding = await ainvoke_with_timeout(
        embedding_client.aembed_query(keyword),
        app_config.agent.embedding_timeout_seconds,
    )
    cost_tracker.add_embedding_usage(
        "召回字段取值",
        estimate_tokens(keyword),
        estimated=True,
        model=app_config.embedding.model,
        cache_hit=bool(getattr(embedding_client, "last_cache_hit", False)),
    )
    return await ainvoke_with_timeout(
        value_qdrant_repository.search(
            embedding,
            score_threshold=app_config.agent.value_vector_score_threshold,
            meta_build_version=metadata_build_version,
        ),
        app_config.agent.retrieval_timeout_seconds,
    )
