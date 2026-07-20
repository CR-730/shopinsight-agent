"""问数查询服务。

把 API 层传入的自然语言问题转换为一次 LangGraph 工作流执行，并统一包装成
SSE 文本返回给前端。流式事件直接使用 LangGraph astream_events v2。
"""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import date

from langchain_core.embeddings import Embeddings

from app.agent.context import DataAgentContext
from app.agent.cost import CostRates, CostTracker
from app.agent.failure import build_failure
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
    messages_to_state,
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

ASSISTANT_ERROR_MESSAGE = "出了点问题，请稍后重试。"
# Match assistant-ui's TextStreamAnimator cap: 5ms per char. Without this
# pacing, buffered LangGraph node output arrives as one backlog and the UI
# cannot resemble token streaming.
ASSISTANT_UI_TIME_PER_CHAR_SECONDS = 0.005


class QueryService:
    """封装一次问数查询所需的业务编排逻辑。"""

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
        self.meta_mysql_repository = meta_mysql_repository
        self.agent_memory_repository = agent_memory_repository
        self.dw_mysql_repository = dw_mysql_repository
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
        """执行一次问数工作流，并逐段产出 SSE 消息。"""

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
        conversation_messages = messages_to_state(conversation.messages)
        memory_query = build_retrieval_query(query, conversation_messages)

        yield _sse(
            {
                "type": "conversation",
                "data": {
                    "conversation_id": conversation.id,
                    "user_id": user_id,
                },
            }
        )

        state = DataAgentState(
            query=query,
            conversation_messages=conversation_messages,
        )
        metadata_build_version = await self.meta_mysql_repository.get_active_build_version()
        metadata_cache_version = await self.meta_mysql_repository.get_metadata_cache_version()
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
            semantic_reference_date=date.today(),
            user_id=user_id,
        )
        cache_namespace_token = set_llm_cache_context_namespace(
            f"metadata:{metadata_cache_version}"
        )
        call_budget_token = set_llm_request_call_budget(
            app_config.llm.max_calls_per_request
        )
        final_state: dict | None = None
        assistant_text_parts: list[str] = []

        try:
            async for event in graph.astream_events(
                input=state,
                context=context,
                version="v2",
                stream_mode=["custom", "values"],
            ):
                payload = _event_payload(event)
                if payload is None:
                    continue
                mode, data = payload
                if mode == "values":
                    final_state = dict(data)
                    continue
                if mode == "custom" and isinstance(data, dict):
                    if data.get("type") == "answer_delta":
                        async for delta_event in _answer_delta_stream(
                            str(data.get("delta") or "")
                        ):
                            assistant_text_parts.append(delta_event["delta"])
                            yield _sse(delta_event)
                        continue
                    yield _sse(data)

            if final_state is not None:
                final_message = _assistant_message(final_state)
                if _should_emit_error_event(final_state):
                    yield _sse({"type": "error", "message": ASSISTANT_ERROR_MESSAGE})
                elif final_message and final_message not in "".join(assistant_text_parts):
                    async for answer_chunk in _answer_stream(final_message):
                        if answer_chunk.get("type") == "answer_delta":
                            assistant_text_parts.append(str(answer_chunk.get("delta") or ""))
                        yield _sse(answer_chunk)
                else:
                    yield _sse({"type": "answer_done"})
                await self._save_memory_after_query(
                    conversation=conversation,
                    query=query,
                    memory_query=memory_query,
                    metadata_cache_version=metadata_cache_version,
                    final_state=final_state,
                    assistant_content="".join(assistant_text_parts).strip(),
                )
                if include_trace:
                    yield _sse({"type": "trace", "data": final_state})

            yield _sse({"type": "usage", "data": cost_tracker.summary()})
        except Exception as e:
            yield _sse({"type": "error", "message": ASSISTANT_ERROR_MESSAGE})
            await self._save_memory_after_query(
                conversation=conversation,
                query=query,
                memory_query=memory_query,
                metadata_cache_version=metadata_cache_version,
                final_state={
                    "failure": build_failure(
                        category="system",
                        stage="query_service",
                        code=e.__class__.__name__,
                        message=str(e),
                        disposition="failed",
                    )
                },
                assistant_content="",
            )
            yield _sse({"type": "usage", "data": cost_tracker.summary()})
        finally:
            reset_llm_cache_context_namespace(cache_namespace_token)
            reset_llm_request_call_budget(call_budget_token)

    async def list_conversations(self, user_id: str, limit: int = 50):
        """返回当前用户的历史会话，供前端侧栏恢复上下文。"""

        return await self.agent_memory_repository.list_conversations(
            user_id=user_id or "anonymous",
            limit=limit,
        )

    async def get_conversation(self, conversation_id: str, user_id: str):
        """按用户隔离读取单个会话及完整消息。"""

        return await self.agent_memory_repository.get_conversation(
            conversation_id=conversation_id,
            user_id=user_id or "anonymous",
        )

    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """按用户隔离删除单个会话，数据库外键会级联删除消息。"""

        return await self.agent_memory_repository.delete_conversation(
            conversation_id=conversation_id,
            user_id=user_id or "anonymous",
        )

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
        assistant_content: str = "",
    ) -> None:
        conversation.add_message(Message(role="user", content=query))
        conversation.add_message(
            Message(
                role="assistant",
                content=assistant_content or _assistant_message(final_state),
                metadata=_assistant_metadata(final_state),
            )
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


def _event_payload(event: dict) -> tuple[str, object] | None:
    if event.get("event") != "on_chain_stream" or event.get("name") != "LangGraph":
        return None
    chunk = (event.get("data") or {}).get("chunk")
    if not (isinstance(chunk, tuple) and len(chunk) == 2):
        return None
    mode, payload = chunk
    if mode not in {"custom", "values"}:
        return None
    return str(mode), payload


async def _answer_stream(message: str) -> AsyncIterator[dict]:
    text = "\n\n" + message
    async for event in _answer_delta_stream(text):
        yield event
    yield {"type": "answer_done"}


async def _answer_delta_stream(text: str, size: int = 4) -> AsyncIterator[dict]:
    for chunk in _split_text(text, size=size):
        yield {"type": "answer_delta", "delta": chunk}
        await asyncio.sleep(len(chunk) * ASSISTANT_UI_TIME_PER_CHAR_SECONDS)


def _split_text(text: str, size: int = 8) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)]


def _assistant_message(final_state: dict) -> str:
    output = final_state.get("output") or {}
    rows = output.get("rows")
    result_meta = output.get("meta") or {}
    tables = result_meta.get("tables") or _table_names(final_state)
    table_text = f"，涉及表：{', '.join(tables[:5])}" if tables else ""

    if isinstance(rows, list):
        if rows:
            columns = (
                list(rows[0].keys())
                if isinstance(rows[0], dict)
                else []
            )
            field_text = f"，字段：{', '.join(columns[:8])}" if columns else ""
            return f"查询完成，共返回 {len(rows)} 行结果{table_text}{field_text}。"
        return "查询完成，结果为空。可以换一个口径或时间范围继续追问。"

    failure = final_state.get("failure")
    if isinstance(failure, dict):
        user_message = str(failure.get("user_message") or "").strip()
        if user_message:
            return user_message
        return ASSISTANT_ERROR_MESSAGE

    return "流程已结束，但没有返回可展示的查询结果。"


def _should_emit_error_event(final_state: dict) -> bool:
    output = final_state.get("output") or {}
    if isinstance(output.get("rows"), list):
        return False
    failure = final_state.get("failure")
    if not isinstance(failure, dict):
        return False
    if str(failure.get("user_message") or "").strip():
        return False
    return True


def _assistant_metadata(final_state: dict) -> dict:
    metadata: dict = {}
    output = final_state.get("output") or {}
    rows = output.get("rows")
    if isinstance(rows, list):
        metadata["result"] = rows
    result_meta = output.get("meta")
    if isinstance(result_meta, dict):
        metadata["result_meta"] = result_meta
    sql = str(final_state.get("sql") or "").strip()
    if sql:
        metadata["sql"] = sql
    return metadata


def _table_names(final_state: dict) -> list[str]:
    names = []
    for table in (final_state.get("sql_context") or {}).get("tables") or []:
        name = str(table.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
