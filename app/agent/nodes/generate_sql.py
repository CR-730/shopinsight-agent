"""SQL 生成节点。"""

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import generate_sql_llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def generate_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """基于已检索和过滤的上下文生成 SQL。"""

    writer = runtime.stream_writer
    step = "生成SQL"
    writer({"type": "progress", "step": step, "status": "running"})

    try:
        table_infos = state["table_infos"]
        metric_infos = state["metric_infos"]
        business_binding = state.get("business_binding") or {
            "metrics": state.get("metric_bindings") or [],
            "filters": state.get("resolved_filters") or [],
            "time": state.get("time_binding"),
            "unresolved": state.get("unresolved_bindings") or [],
            "ambiguous": state.get("ambiguous_bindings") or [],
        }
        validated_enum_values = state.get("validated_enum_values") or []
        date_info = state["date_info"]
        db_info = state["db_info"]
        query = state["query"]
        conversation_history = state.get("conversation_history") or "无"
        sql_memory_context = state.get("sql_memory_context") or "无"

        prompt = PromptTemplate(
            template=load_prompt("generate_sql"),
            input_variables=[
                "table_infos",
                "metric_infos",
                "business_bindings",
                "conversation_history",
                "sql_memory_context",
                "date_info",
                "db_info",
                "query",
            ],
        )
        output_parser = StrOutputParser()

        result = await ainvoke_llm_with_usage(
            prompt,
            generate_sql_llm,
            output_parser,
            {
                "table_infos": yaml.dump(
                    table_infos, allow_unicode=True, sort_keys=False
                ),
                "metric_infos": yaml.dump(
                    metric_infos, allow_unicode=True, sort_keys=False
                ),
                "business_bindings": yaml.dump(
                    {
                        "business_binding": business_binding,
                        "validated_enum_values": validated_enum_values,
                    },
                    allow_unicode=True,
                    sort_keys=False,
                ),
                "conversation_history": conversation_history,
                "sql_memory_context": sql_memory_context,
                "date_info": yaml.dump(date_info, allow_unicode=True, sort_keys=False),
                "db_info": yaml.dump(db_info, allow_unicode=True, sort_keys=False),
                "query": query,
            },
            step,
            runtime.context["cost_tracker"],
            app_config.llm.timeout_seconds,
            cacheable=False,
        )

        logger.info(f"生成的SQL：{result}")
        writer({"type": "progress", "step": step, "status": "success"})
        return {"sql": result}

    except Exception as e:
        logger.error(f"{step} failed: {e}")
        writer({"type": "progress", "step": step, "status": "error"})
        raise
