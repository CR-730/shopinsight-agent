"""指标召回节点。"""

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
from app.entities.metric_info import MetricInfo
from app.prompt.prompt_loader import load_prompt


async def recall_metric(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """召回和用户问题语义相关的业务指标。"""

    writer = runtime.stream_writer
    step = "召回指标信息"
    writer({"type": "progress", "step": step, "status": "running"})

    try:
        query = state["query"]
        keywords = state["keywords"]
        embedding_client = runtime.context["embedding_client"]
        metric_qdrant_repository = runtime.context["metric_qdrant_repository"]

        prompt = PromptTemplate(
            template=load_prompt("extend_keywords_for_metric_recall"),
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
        metric_info_map: dict[str, MetricInfo] = {}
        for keyword in keywords:
            keyword_started_at = time.perf_counter()
            embedding = await ainvoke_with_timeout(
                embedding_client.aembed_query(keyword),
                app_config.agent.embedding_timeout_seconds,
            )
            embedding_latency_ms = round(
                (time.perf_counter() - keyword_started_at) * 1000, 2
            )
            runtime.context["cost_tracker"].add_embedding_usage(
                step,
                estimate_tokens(keyword),
                estimated=True,
                model=app_config.embedding.model,
                latency_ms=embedding_latency_ms,
                cache_hit=bool(getattr(embedding_client, "last_cache_hit", False)),
            )
            current_metric_infos: list[MetricInfo] = await ainvoke_with_timeout(
                metric_qdrant_repository.search(
                    embedding,
                    meta_build_version=runtime.context.get("metadata_build_version"),
                ),
                app_config.agent.retrieval_timeout_seconds,
            )
            latency_ms = round((time.perf_counter() - keyword_started_at) * 1000, 2)
            writer(
                {
                    "type": "retrieval_debug",
                    "step": step,
                    "keyword": keyword,
                    "source": "qdrant",
                    "latency_ms": latency_ms,
                    "hit_count": len(current_metric_infos),
                }
            )
            for metric_info in current_metric_infos:
                if metric_info.id not in metric_info_map:
                    metric_info_map[metric_info.id] = metric_info

        logger.info(f"检索到指标信息：{list(metric_info_map.keys())}")
        writer({"type": "progress", "step": step, "status": "success"})
        return {"retrieved_metric_infos": list(metric_info_map.values())}
    except Exception as e:
        logger.error(f"{step} failed: {e}")
        writer({"type": "progress", "step": step, "status": "error"})
        raise
