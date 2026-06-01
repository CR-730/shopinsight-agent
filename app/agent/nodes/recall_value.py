"""字段取值混合召回节点。"""

import asyncio
import time

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.cached_clients import ainvoke_with_timeout
from app.agent.context import DataAgentContext
from app.agent.cost import estimate_tokens
from app.agent.keyword_expansion import normalize_keyword_list
from app.agent.llm import llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.log import logger
from app.entities.value_info import ValueInfo
from app.prompt.prompt_loader import load_prompt
from app.retrieval.fusion import fuse_ranked_value_infos


async def recall_value(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """召回和用户问题相关的字段取值。"""

    writer = runtime.stream_writer
    step = "召回字段取值"
    writer({"type": "progress", "step": step, "status": "running"})

    try:
        query = state["query"]
        keywords = state["keywords"]
        value_es_repository = runtime.context["value_es_repository"]
        value_qdrant_repository = runtime.context["value_qdrant_repository"]
        embedding_client = runtime.context["embedding_client"]

        prompt = PromptTemplate(
            template=load_prompt("extend_keywords_for_value_recall"),
            input_variables=["query"],
        )
        output_parser = JsonOutputParser()

        started_at = time.perf_counter()
        try:
            result = await ainvoke_llm_with_usage(
                prompt,
                llm,
                output_parser,
                {"query": query},
                step,
                runtime.context["cost_tracker"],
                app_config.llm.timeout_seconds,
            )
        except TimeoutError:
            result = []
            logger.warning(f"{step} LLM 扩展超时，降级使用原始关键词")
        llm_ms = round((time.perf_counter() - started_at) * 1000, 2)
        writer(
            {
                "type": "retrieval_debug",
                "step": step,
                "llm_extended_keywords": result,
                "llm_latency_ms": llm_ms,
            }
        )

        result = normalize_keyword_list(result)
        keywords = set(normalize_keyword_list(keywords) + result)
        value_infos_map: dict[str, ValueInfo] = {}
        for keyword in keywords:
            (
                es_value_infos,
                es_latency_ms,
            ), (
                vector_value_infos,
                vector_latency_ms,
            ) = await asyncio.gather(
                _search_values_by_es(value_es_repository, keyword),
                _search_values_by_vector(
                    value_qdrant_repository,
                    embedding_client,
                    keyword,
                    runtime.context["cost_tracker"],
                ),
            )

            current_value_infos = fuse_ranked_value_infos(
                {"es": es_value_infos, "vector": vector_value_infos},
                weights={
                    "es": app_config.agent.value_hybrid_es_weight,
                    "vector": app_config.agent.value_hybrid_vector_weight,
                },
            )
            writer(
                {
                    "type": "retrieval_debug",
                    "step": step,
                    "keyword": keyword,
                    "source": "hybrid",
                    "es_latency_ms": es_latency_ms,
                    "es_hit_count": len(es_value_infos),
                    "vector_latency_ms": vector_latency_ms,
                    "vector_hit_count": len(vector_value_infos),
                    "fused_hit_count": len(current_value_infos),
                }
            )
            for current_value_info in current_value_infos:
                if current_value_info.id not in value_infos_map:
                    value_infos_map[current_value_info.id] = current_value_info

        logger.info(f"检索到字段取值：{list(value_infos_map.keys())}")
        writer({"type": "progress", "step": step, "status": "success"})
        return {"retrieved_value_infos": list(value_infos_map.values())}
    except Exception as e:
        logger.error(f"{step} failed: {e}")
        writer({"type": "progress", "step": step, "status": "error"})
        raise


async def _search_values_by_es(value_es_repository, keyword: str):
    started_at = time.perf_counter()
    value_infos = await ainvoke_with_timeout(
        value_es_repository.search(keyword),
        app_config.agent.retrieval_timeout_seconds,
    )
    return value_infos, round((time.perf_counter() - started_at) * 1000, 2)


async def _search_values_by_vector(
    value_qdrant_repository, embedding_client, keyword: str, cost_tracker
):
    started_at = time.perf_counter()
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
    value_infos = await ainvoke_with_timeout(
        value_qdrant_repository.search(
            embedding,
            score_threshold=app_config.agent.value_vector_score_threshold,
        ),
        app_config.agent.retrieval_timeout_seconds,
    )
    return value_infos, round((time.perf_counter() - started_at) * 1000, 2)
