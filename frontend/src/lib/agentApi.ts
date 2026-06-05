import type { AgentEvent, ConversationSummary } from "../types/agent";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "";
const USER_ID_KEY = "shopkeeper-agent-user-id";

type QueryOptions = {
  signal?: AbortSignal;
  conversationId?: string | null;
  onEvent: (event: AgentEvent) => void;
};

export function getUserId() {
  const existing = localStorage.getItem(USER_ID_KEY);
  if (existing) return existing;

  const userId = crypto.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  localStorage.setItem(USER_ID_KEY, userId);
  return userId;
}

export async function listConversations(signal?: AbortSignal) {
  const url = new URL(`${API_BASE_URL || window.location.origin}/api/conversations`);
  url.searchParams.set("user_id", getUserId());

  const response = await fetch(toRequestUrl(url), { signal });
  if (!response.ok) {
    throw new Error(`加载历史会话失败：HTTP ${response.status}`);
  }

  const payload = (await response.json()) as { conversations: ConversationSummary[] };
  return payload.conversations ?? [];
}

export async function getConversation(conversationId: string, signal?: AbortSignal) {
  const url = new URL(`${API_BASE_URL || window.location.origin}/api/conversations/${conversationId}`);
  url.searchParams.set("user_id", getUserId());

  const response = await fetch(toRequestUrl(url), { signal });
  if (!response.ok) {
    throw new Error(`加载会话失败：HTTP ${response.status}`);
  }

  return (await response.json()) as ConversationSummary;
}

export async function deleteConversation(conversationId: string, signal?: AbortSignal) {
  const url = new URL(`${API_BASE_URL || window.location.origin}/api/conversations/${conversationId}`);
  url.searchParams.set("user_id", getUserId());

  const response = await fetch(toRequestUrl(url), {
    method: "DELETE",
    signal,
  });
  if (!response.ok) {
    throw new Error(`删除会话失败：HTTP ${response.status}`);
  }
}

export async function streamQuery(query: string, options: QueryOptions) {
  const response = await fetch(`${API_BASE_URL}/api/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      query,
      conversation_id: options.conversationId ?? undefined,
      user_id: getUserId(),
    }),
    signal: options.signal,
  });

  if (!response.ok) {
    throw new Error(`接口请求失败：HTTP ${response.status}`);
  }

  if (!response.body) {
    throw new Error("浏览器没有返回可读取的流式响应。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split(/\n\n/);
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const event = parseSseChunk(chunk);
      if (event) options.onEvent(event);
    }
  }

  buffer += decoder.decode();
  const tail = parseSseChunk(buffer);
  if (tail) options.onEvent(tail);
}

function parseSseChunk(chunk: string): AgentEvent | null {
  const payload = chunk
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.replace(/^data:\s?/, ""))
    .join("\n")
    .trim();

  if (!payload) return null;

  try {
    return JSON.parse(payload) as AgentEvent;
  } catch {
    return {
      type: "error",
      message: "出了点问题，请稍后重试。",
    };
  }
}

function toRequestUrl(url: URL) {
  if (API_BASE_URL) return url.toString();
  return `${url.pathname}${url.search}`;
}
