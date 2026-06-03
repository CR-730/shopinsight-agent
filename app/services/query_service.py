"""
问数查询服务

负责把 API 层传入的自然语言问题转换成一次 LangGraph 工作流执行：
创建初始 State、组装 Runtime Context、消费 graph.astream 的流式输出，
并统一包装成 SSE 文本返回给路由层。
"""

import json
import uuid

from langchain_core.embeddings import Embeddings

from app.agent.context import DataAgentContext
from app.agent.cost import CostRates, CostTracker
from app.agent.graph import graph
from app.agent.llm_usage import (
    reset_llm_cache_context_namespace,
    reset_llm_request_call_budget,
    set_llm_cache_context_namespace,
    set_llm_request_call_budget,
)
from app.agent.memory import (
    Message,
    build_retrieval_query,
    build_sql_tool_memory,
    format_conversation_history,
)
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
from app.repositories.mysql.meta.agent_memory_repository import AgentMemoryRepository
from app.repositories.mysql.meta.meta_mysql_repository import MetaMySQLRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository
from app.repositories.qdrant.value_qdrant_repository import ValueQdrantRepository


class QueryService:
    """封装一次问数查询所需的业务编排逻辑"""

    def __init__(
        self,
        meta_mysql_repository: MetaMySQLRepository,
        agent_memory_repository: AgentMemoryRepository,
        embedding_client: Embeddings,
        dw_mysql_repository: DWMySQLRepository,
        column_qdrant_repository: ColumnQdrantRepository,
        metric_qdrant_repository: MetricQdrantRepository,
        value_es_repository: ValueESRepository,
        value_qdrant_repository: ValueQdrantRepository,
    ):
        # MySQL 仓储分别负责元数据补全和真实数仓环境信息读取
        self.meta_mysql_repository = meta_mysql_repository
        self.agent_memory_repository = agent_memory_repository
        self.dw_mysql_repository = dw_mysql_repository

        # 召回链路依赖的向量检索、Embedding 和全文检索能力由依赖层注入
        self.embedding_client = embedding_client
        self.column_qdrant_repository = column_qdrant_repository
        self.metric_qdrant_repository = metric_qdrant_repository
        self.value_es_repository = value_es_repository
        self.value_qdrant_repository = value_qdrant_repository

    async def query(
        self,
        query: str,
        conversation_id: str | None = None,
        user_id: str | None = None,
        include_trace: bool = False,
    ):
        """执行一次问数工作流，并逐段产出 SSE 消息"""

        cost_tracker = CostTracker(
            CostRates(
                llm_input_per_1m_tokens=app_config.cost.llm_input_per_1m_tokens,
                llm_output_per_1m_tokens=app_config.cost.llm_output_per_1m_tokens,
                embedding_per_1m_tokens=app_config.cost.embedding_per_1m_tokens,
                currency=app_config.cost.currency,
            )
        )

        user_id = user_id or "anonymous"
        conversation = await self._load_or_create_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
        )
        conversation_history = format_conversation_history(conversation.messages)
        memory_query = build_retrieval_query(query, conversation_history)

        yield _sse(
            {
                "type": "conversation",
                "data": {
                    "conversation_id": conversation.id,
                    "user_id": user_id,
                },
            }
        )

        # State 只放会被图节点读写和合并的业务数据，外部工具对象不塞进 State
        state = DataAgentState(
            query=query,
            conversation_history=conversation_history,
            correction_attempts=0,
            max_correction_attempts=app_config.agent.max_sql_correction_attempts,
        )
        # Context 保存本次图执行需要复用的外部依赖，节点通过 runtime.context 读取
        metadata_build_version = (
            await self.meta_mysql_repository.get_active_build_version()
        )
        metadata_cache_version = (
            await self.meta_mysql_repository.get_metadata_cache_version()
        )
        context = DataAgentContext(
            column_qdrant_repository=self.column_qdrant_repository,
            embedding_client=self.embedding_client,
            metric_qdrant_repository=self.metric_qdrant_repository,
            value_es_repository=self.value_es_repository,
            value_qdrant_repository=self.value_qdrant_repository,
            meta_mysql_repository=self.meta_mysql_repository,
            agent_memory_repository=self.agent_memory_repository,
            dw_mysql_repository=self.dw_mysql_repository,
            cost_tracker=cost_tracker,
            metadata_build_version=metadata_build_version,
            metadata_cache_version=metadata_cache_version,
            user_id=user_id,
        )
        cache_namespace_token = set_llm_cache_context_namespace(
            f"metadata:{metadata_cache_version}"
        )
        call_budget_token = set_llm_request_call_budget(
            app_config.llm.max_calls_per_request
        )
        final_state: dict | None = None
        try:
            # stream_mode="custom" 对应节点内部 writer(...) 写出的进度消息
            async for chunk in graph.astream(
                input=state, context=context, stream_mode=["custom", "values"]
            ):
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    mode, payload = chunk
                    if mode == "values":
                        final_state = dict(payload)
                        continue
                    if mode == "custom":
                        chunk = payload
                # SSE 要求每条消息以 data: 开头，并以两个换行符结束
                # ensure_ascii=False 保留中文进度文案，default=str 兜底处理日期等非 JSON 类型
                yield _sse(chunk)
            if final_state is not None:
                await self._save_memory_after_query(
                    conversation=conversation,
                    query=query,
                    memory_query=memory_query,
                    metadata_cache_version=metadata_cache_version,
                    final_state=final_state,
                )
                if include_trace:
                    trace = {"type": "trace", "data": final_state}
                    yield _sse(trace)
            usage = {"type": "usage", "data": cost_tracker.summary()}
            yield _sse(usage)
        except Exception as e:
            # 流式接口已经开始返回后不能再改 HTTP 状态码，因此把异常也包装成一条 SSE 消息
            error = {"type": "error", "message": str(e)}
            yield _sse(error)
            await self._save_memory_after_query(
                conversation=conversation,
                query=query,
                memory_query=memory_query,
                metadata_cache_version=metadata_cache_version,
                final_state={"error": str(e)},
            )
            usage = {"type": "usage", "data": cost_tracker.summary()}
            yield _sse(usage)
        finally:
            reset_llm_cache_context_namespace(cache_namespace_token)
            reset_llm_request_call_budget(call_budget_token)

    async def _load_or_create_conversation(
        self, conversation_id: str | None, user_id: str
    ):
        if conversation_id:
            conversation = await self.agent_memory_repository.get_conversation(
                conversation_id, user_id
            )
            if conversation:
                return conversation
        return await self.agent_memory_repository.create_conversation(
            str(uuid.uuid4()), user_id
        )

    async def _save_memory_after_query(
        self,
        conversation,
        query: str,
        memory_query: str,
        metadata_cache_version: str,
        final_state: dict,
    ) -> None:
        conversation.add_message(Message(role="user", content=query))
        conversation.add_message(
            Message(role="assistant", content=_assistant_message(final_state))
        )
        await self.agent_memory_repository.update_conversation(conversation)

        tool_memory = build_sql_tool_memory(memory_query, final_state)
        if tool_memory and conversation.user_id != "anonymous":
            await self.agent_memory_repository.save_tool_usage(
                question=tool_memory.question,
                tool_name=tool_memory.tool_name,
                args=tool_memory.args,
                user_id=conversation.user_id,
                metadata_cache_version=metadata_cache_version,
                success=tool_memory.success,
                metadata=tool_memory.metadata,
            )


def _assistant_message(final_state: dict) -> str:
    final_answer = final_state.get("final_answer")
    if isinstance(final_answer, list) and final_answer:
        columns = (
            list(final_answer[0].keys()) if isinstance(final_answer[0], dict) else []
        )
        field_text = f"，字段：{', '.join(columns[:8])}" if columns else ""
        return f"查询成功，返回 {len(final_answer)} 行{field_text}"
    return str(final_state.get("safety_error") or final_state.get("error") or "")


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
