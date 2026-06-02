"""表信息过滤节点。"""

import yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.agent.state import DataAgentState, TableInfoState
from app.conf.app_config import app_config
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def filter_table(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """根据用户问题裁剪候选表结构上下文。"""

    writer = runtime.stream_writer
    step = "过滤表信息"
    writer({"type": "progress", "step": step, "status": "running"})

    try:
        query = state["query"]
        table_infos: list[TableInfoState] = state["table_infos"]
        prompt_table_infos = compact_table_context_for_filtering(
            table_infos, state.get("business_binding") or {}
        )

        prompt = PromptTemplate(
            template=load_prompt("filter_table_info"),
            input_variables=["query", "table_infos"],
        )
        output_parser = JsonOutputParser()

        result = await ainvoke_llm_with_usage(
            prompt,
            llm,
            output_parser,
            {
                "query": query,
                "table_infos": yaml.dump(
                    prompt_table_infos, allow_unicode=True, sort_keys=False
                ),
            },
            step,
            runtime.context["cost_tracker"],
            app_config.llm.timeout_seconds,
        )

        filtered_table_infos: list[TableInfoState] = []
        for table_info in table_infos:
            if table_info["name"] in result:
                table_info["columns"] = [
                    column_info
                    for column_info in table_info["columns"]
                    if column_info["name"] in result[table_info["name"]]
                ]
                filtered_table_infos.append(table_info)

        logger.info(
            f"过滤后的表信息：{[item['name'] for item in filtered_table_infos]}"
        )
        writer({"type": "progress", "step": step, "status": "success"})
        return {"table_infos": filtered_table_infos}

    except Exception as e:
        logger.error(f"{step} failed: {e}")
        writer({"type": "progress", "step": step, "status": "error"})
        raise


def compact_table_context_for_filtering(
    table_infos: list[TableInfoState], business_binding: dict
) -> list[dict]:
    """Return a smaller table context without removing candidate columns."""

    if not business_binding:
        return table_infos

    compacted = []
    for table_info in table_infos:
        compacted.append(
            {
                "name": table_info["name"],
                "role": table_info.get("role", ""),
                "columns": [
                    _compact_column(column_info)
                    for column_info in table_info.get("columns") or []
                ],
            }
        )
    return compacted


def _compact_column(column_info: dict) -> dict:
    return {
        "name": column_info["name"],
        "role": column_info.get("role", ""),
        "alias": column_info.get("alias") or [],
    }
