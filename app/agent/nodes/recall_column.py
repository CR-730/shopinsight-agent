"""字段召回节点。"""

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
from app.entities.column_info import ColumnInfo
from app.prompt.prompt_loader import load_prompt


async def recall_column(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """召回和用户问题语义相关的字段元数据。"""

    writer = runtime.stream_writer
    step = "召回字段信息"
    writer({"type": "progress", "step": step, "status": "running"})

    try:
        keywords = state["keywords"]
        query = state["query"]
        column_qdrant_repository = runtime.context["column_qdrant_repository"]
        embedding_client = runtime.context["embedding_client"]

        prompt = PromptTemplate(
            template=load_prompt("extend_keywords_for_column_recall"),
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
        column_info_map: dict[str, ColumnInfo] = {}
        for keyword in keywords:
            keyword_started_at = time.perf_counter()
            embedding = await ainvoke_with_timeout(
                embedding_client.aembed_query(keyword),
                app_config.agent.embedding_timeout_seconds,
            )
            runtime.context["cost_tracker"].add_embedding_usage(
                step,
                estimate_tokens(keyword),
                estimated=True,
                model=app_config.embedding.model,
                cache_hit=bool(getattr(embedding_client, "last_cache_hit", False)),
            )
            current_column_infos: list[ColumnInfo] = await ainvoke_with_timeout(
                column_qdrant_repository.search(embedding),
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
                    "hit_count": len(current_column_infos),
                }
            )
            for column_info in current_column_infos:
                if column_info.id not in column_info_map:
                    column_info_map[column_info.id] = column_info

        writer({"type": "progress", "step": step, "status": "success"})
        return {"retrieved_column_infos": list(column_info_map.values())}
    except Exception as e:
        logger.error(f"{step} failed: {e}")
        writer({"type": "progress", "step": step, "status": "error"})
        raise
