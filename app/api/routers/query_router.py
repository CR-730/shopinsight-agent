"""
问数查询接口路由

负责定义前端访问的 `/api/query` 接口，把 HTTP 请求交给 QueryService，
并把问数智能体执行过程以 SSE 形式持续返回给客户端。
路由层只处理请求体、依赖声明和响应类型，不直接创建 Repository 或执行图节点。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.responses import StreamingResponse

from app.api.dependencies import get_query_service
from app.api.schemas.query_schema import (
    ConversationListSchema,
    ConversationMessageSchema,
    ConversationSummarySchema,
    QuerySchema,
)
from app.services.query_service import QueryService

# 当前模块只维护查询相关接口，避免后续所有 API 都挤在 main.py 中
query_router = APIRouter()


def _conversation_summary(conversation) -> ConversationSummarySchema:
    title = next(
        (
            message.content.strip()
            for message in conversation.messages
            if message.role == "user" and message.content.strip()
        ),
        conversation.id,
    )
    return ConversationSummarySchema(
        id=conversation.id,
        title=title,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
        messages=[
            ConversationMessageSchema(
                role=message.role,
                content=message.content,
                created_at=message.timestamp.isoformat(),
                metadata=message.metadata,
            )
            for message in conversation.messages
        ],
    )


@query_router.post("/api/query")
async def query_handler(
    # 请求体参数：FastAPI 会把前端 JSON 自动解析成 QuerySchema
    query: QuerySchema,
    # 服务依赖：FastAPI 会调用 get_query_service，递归组装它所需的仓储和客户端
    query_service: Annotated[QueryService, Depends(get_query_service)],
):
    """接收用户自然语言问题，并流式返回 LangGraph 工作流输出"""

    return StreamingResponse(
        # QueryService.query 返回异步生成器供响应逐段消费
        query_service.query(
            query=query.query,
            conversation_id=query.conversation_id,
            user_id=query.user_id,
        ),
        media_type="text/event-stream",
    )


@query_router.get("/api/conversations", response_model=ConversationListSchema)
async def list_conversations_handler(
    query_service: Annotated[QueryService, Depends(get_query_service)],
    user_id: str = Query(default="anonymous"),
    limit: int = Query(default=50, ge=1, le=100),
):
    conversations = await query_service.list_conversations(user_id=user_id, limit=limit)
    return ConversationListSchema(
        conversations=[_conversation_summary(item) for item in conversations]
    )


@query_router.get(
    "/api/conversations/{conversation_id}",
    response_model=ConversationSummarySchema,
)
async def get_conversation_handler(
    conversation_id: str,
    query_service: Annotated[QueryService, Depends(get_query_service)],
    user_id: str = Query(default="anonymous"),
):
    conversation = await query_service.get_conversation(
        conversation_id=conversation_id,
        user_id=user_id,
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return _conversation_summary(conversation)


@query_router.delete("/api/conversations/{conversation_id}")
async def delete_conversation_handler(
    conversation_id: str,
    query_service: Annotated[QueryService, Depends(get_query_service)],
    user_id: str = Query(default="anonymous"),
):
    deleted = await query_service.delete_conversation(
        conversation_id=conversation_id,
        user_id=user_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"ok": True}
