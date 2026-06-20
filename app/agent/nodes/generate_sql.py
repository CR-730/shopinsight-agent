"""SQL 生成节点。"""

import yaml
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict, Field

from app.agent.context import DataAgentContext
from app.agent.llm import generate_sql_llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.agent.memory import format_conversation_messages, format_sql_memory_examples
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


class GeneratedSqlResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sql: str = Field(default="")
    explanation: str = Field(default="")


async def generate_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """基于已检索和过滤的上下文生成 SQL。"""

    writer = runtime.stream_writer
    step = "生成SQL"
    writer({"type": "progress", "step": step, "status": "running"})

    try:
        sql_context = state["sql_context"]
        table_infos = sql_context.get("tables") or []
        metric_infos = sql_context.get("metrics") or []
        business_binding = state.get("business_binding") or {}
        date_info = sql_context.get("date") or {}
        db_info = sql_context.get("db") or {}
        query = state["query"]
        conversation_history = (
            format_conversation_messages(state.get("conversation_messages") or [])
            or "无"
        )
        sql_memory_context = (
            format_sql_memory_examples(state.get("sql_memory_examples") or []) or "无"
        )

        parser = PydanticOutputParser(pydantic_object=GeneratedSqlResponse)
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
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        result = await ainvoke_llm_with_usage(
            prompt,
            generate_sql_llm,
            parser,
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
                        "validated_enum_values": _validated_enum_values(
                            business_binding
                        ),
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

        sql = result.sql.strip()
        explanation = result.explanation.strip()
        if explanation:
            _write_answer_delta(writer, "\n\n" + explanation)
        logger.info(f"生成的SQL：{sql}")
        writer({"type": "progress", "step": step, "status": "success"})
        return {"sql": sql, "trace": {"sql_explanation": explanation}}

    except Exception as e:
        logger.error(f"{step} failed: {e}")
        writer({"type": "progress", "step": step, "status": "error"})
        raise


def _write_answer_delta(writer, text: str) -> None:
    content = str(text or "")
    for index in range(0, len(content), 12):
        writer({"type": "answer_delta", "delta": content[index : index + 12]})


def _validated_enum_values(business_binding: dict) -> list[str]:
    return [
        str(literal)
        for item in business_binding.get("filters") or []
        for literal in item.get("allowed_sql_literals", [])
    ]
