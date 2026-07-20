"""Vanna-style conversation store and agent memory primitives."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from app.agent.cost import estimate_tokens

SQL_TOOL_NAME = "run_sql"
CONVERSATION_HISTORY_TOKEN_BUDGET = 1200


class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Conversation(BaseModel):
    id: str
    user_id: str
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        self.updated_at = datetime.now(UTC)


class ToolMemory(BaseModel):
    memory_id: str | None = None
    question: str
    tool_name: str
    args: dict[str, Any]
    timestamp: str | None = None
    success: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextMemory(BaseModel):
    memory_id: str | None = None
    content: str
    timestamp: str | None = None


class ToolMemorySearchResult(BaseModel):
    memory: ToolMemory
    similarity_score: float
    rank: int


class TextMemorySearchResult(BaseModel):
    memory: TextMemory
    similarity_score: float
    rank: int


class InMemoryConversationStore:
    """Small local implementation matching Vanna's MemoryConversationStore."""

    def __init__(self) -> None:
        self._conversations: dict[str, Conversation] = {}

    async def create_conversation(
        self, conversation_id: str, user_id: str, initial_message: str
    ) -> Conversation:
        conversation = Conversation(
            id=conversation_id,
            user_id=user_id,
            messages=[Message(role="user", content=initial_message)],
        )
        self._conversations[conversation_id] = conversation
        return conversation

    async def get_conversation(
        self, conversation_id: str, user_id: str
    ) -> Conversation | None:
        conversation = self._conversations.get(conversation_id)
        if conversation and conversation.user_id == user_id:
            return conversation
        return None

    async def update_conversation(self, conversation: Conversation) -> None:
        self._conversations[conversation.id] = conversation

    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        conversation = await self.get_conversation(conversation_id, user_id)
        if not conversation:
            return False
        del self._conversations[conversation_id]
        return True

    async def list_conversations(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> list[Conversation]:
        conversations = [
            item for item in self._conversations.values() if item.user_id == user_id
        ]
        conversations.sort(key=lambda item: item.updated_at, reverse=True)
        return conversations[offset : offset + limit]


def build_sql_tool_memory(question: str, state: dict[str, Any]) -> ToolMemory | None:
    sql = str(state.get("sql") or "").strip()
    output = state.get("output") or {}
    rows = output.get("rows") or []
    semantic_plan = _trusted_semantic_plan(state)
    if not sql or not rows or semantic_plan is None or state.get("failure"):
        return None
    return ToolMemory(
        question=question,
        tool_name=SQL_TOOL_NAME,
        args={
            "sql": sql,
            "semantic_plan": semantic_plan,
            "metadata_version": semantic_plan["metadata_version"],
        },
        success=True,
    )


def format_conversation_history(messages: list[Message], limit: int = 8) -> str:
    return format_conversation_messages(messages_to_state(messages), limit=limit)


def messages_to_state(messages: list[Message]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def format_conversation_messages(
    messages: list[dict[str, Any]] | str | None, limit: int = 8
) -> str:
    if isinstance(messages, str):
        return messages
    if not messages:
        return ""
    return "\n".join(
        f"{message.get('role')}: {message.get('content')}"
        for message in messages[-limit:]
        if message.get("role") and message.get("content")
    )


def sliding_conversation_history(
    conversation_history: str | list[dict[str, Any]],
    *,
    token_budget: int = CONVERSATION_HISTORY_TOKEN_BUDGET,
) -> str:
    """Keep full history until it exceeds budget, then drop oldest lines."""

    text = format_conversation_messages(conversation_history)
    lines = [line for line in text.splitlines() if line.strip()]
    if estimate_tokens("\n".join(lines)) <= token_budget:
        return "\n".join(lines)

    selected: list[str] = []
    used = 0
    for line in reversed(lines):
        line_tokens = estimate_tokens(line)
        if selected and used + line_tokens > token_budget:
            break
        if not selected and line_tokens > token_budget:
            selected.append(_trim_to_token_budget(line, token_budget))
            break
        selected.append(line)
        used += line_tokens
    return "\n".join(reversed(selected))


def _trim_to_token_budget(text: str, token_budget: int) -> str:
    max_chars = max(1, token_budget * 2)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def format_tool_memory_results(results: list[ToolMemorySearchResult]) -> str:
    return format_sql_memory_examples(tool_memory_results_to_examples(results))


def tool_memory_results_to_examples(
    results: list[ToolMemorySearchResult],
) -> list[dict[str, Any]]:
    return [
        {
            "rank": result.rank,
            "question": result.memory.question,
            "sql": str(result.memory.args.get("sql") or ""),
            "similarity": round(float(result.similarity_score), 4),
        }
        for result in results
    ]


def format_sql_memory_examples(examples: list[dict[str, Any]] | str | None) -> str:
    if isinstance(examples, str):
        return examples
    if not examples:
        return ""
    lines: list[str] = []
    for item in examples:
        lines.append(
            f"{item.get('rank')}. question: {item.get('question')}\n"
            f"   sql: {item.get('sql')}\n"
            f"   similarity: {float(item.get('similarity') or 0):.2f}"
        )
    return "\n".join(lines)


def build_retrieval_query(
    query: str, conversation_messages: str | list[dict[str, Any]] | None
) -> str:
    history = format_conversation_messages(conversation_messages)
    user_history = [line for line in history.splitlines() if line.startswith("user:")]
    context = "\n".join(user_history[-2:])
    return f"{context}\nuser: {query}" if context else query


def _trusted_semantic_plan(state: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and minimize the plan before it crosses the memory boundary."""

    from app.agent.semantic_planning.plan import SemanticQueryPlan

    raw_plan = state.get("semantic_plan")
    if not raw_plan:
        return None
    try:
        plan = SemanticQueryPlan.model_validate(raw_plan)
    except ValueError:
        return None
    sanitized = plan.model_dump(mode="json")
    # Provenance contains user wording and verbose evidence. SQL memory needs
    # canonical semantics, not the original utterance or candidate rationale.
    sanitized["provenance"] = []
    return sanitized
