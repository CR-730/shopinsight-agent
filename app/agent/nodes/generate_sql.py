"""SQL 生成节点。"""

import yaml
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict, Field

from app.agent.context import DataAgentContext
from app.agent.llm import generate_sql_llm
from app.agent.llm_usage import ainvoke_llm_with_usage
from app.agent.memory import format_sql_memory_examples
from app.agent.sql.plan_projection import project_plan_for_sql
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


class GeneratedSqlResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sql: str = Field(default="")
    explanation: str = Field(default="")


async def generate_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """Compile a validated semantic plan into one read-only SQL statement."""

    writer = runtime.stream_writer
    step = "生成SQL"
    writer({"type": "progress", "step": step, "status": "running"})

    try:
        raw_plan = state.get("semantic_plan")
        if raw_plan is None:
            raise ValueError("semantic_plan is required for SQL generation")
        semantic_plan = project_plan_for_sql(raw_plan)
        db_info = await runtime.context["dw_mysql_repository"].get_db_info()
        query = state["query"]
        sql_memory_context = (
            format_sql_memory_examples(state.get("sql_memory_examples") or []) or "无"
        )

        parser = PydanticOutputParser(pydantic_object=GeneratedSqlResponse)
        prompt = PromptTemplate(
            template=load_prompt("generate_sql"),
            input_variables=[
                "semantic_plan",
                "sql_memory_context",
                "db_info",
                "query_for_explanation_only",
            ],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        result = await ainvoke_llm_with_usage(
            prompt,
            generate_sql_llm,
            parser,
            {
                "semantic_plan": yaml.dump(
                    semantic_plan, allow_unicode=True, sort_keys=False
                ),
                "sql_memory_context": sql_memory_context,
                "db_info": yaml.dump(db_info, allow_unicode=True, sort_keys=False),
                "query_for_explanation_only": query,
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
