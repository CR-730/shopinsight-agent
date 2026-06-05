import type {
  AgentEvent,
  ChatMessage,
  ConversationSummary,
  MessagePart,
  ProgressEvent,
  StepState,
} from "../types/agent";
import { summarizeResult } from "./format";

export function makeId() {
  return crypto.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function upsertStep(steps: StepState[] = [], event: ProgressEvent) {
  const next = steps.filter((item) => item.step !== event.step);
  next.push({
    step: event.step,
    label: stageLabel(event.step),
    status: event.status,
    updatedAt: Date.now(),
  });
  return next;
}

export function appendStatusPart(
  parts: MessagePart[] = [],
  event: ProgressEvent,
): MessagePart[] {
  if (event.status !== "running") return parts;
  const label = displayStatusLabel(stageLabel(event.step));
  const statusParts = parts.filter((part) => part.type === "status");
  if (statusParts.some((part) => part.label === label)) return parts;
  const lastStatus = statusParts[statusParts.length - 1];
  if (lastStatus && statusRank(label) <= statusRank(lastStatus.label)) return parts;
  return [
    ...settleStatusParts(parts),
    {
      id: makeId(),
      type: "status",
      label,
      status: event.status,
    },
  ];
}

export function appendTextPart(parts: MessagePart[] = [], delta: string): MessagePart[] {
  const settledParts = settleStatusParts(parts);
  const last = settledParts[settledParts.length - 1];
  if (last?.type === "text") {
    return [
      ...settledParts.slice(0, -1),
      {
        ...last,
        content: last.content + delta,
      },
    ];
  }
  return [
    ...settledParts,
    {
      id: makeId(),
      type: "text",
      content: delta,
    },
  ];
}

export function settleStatusParts(parts: MessagePart[] = []): MessagePart[] {
  return parts.map((part) =>
    part.type === "status" && part.status === "running"
      ? { ...part, status: "success" }
      : part,
  );
}

export function messagesFromConversation(conversation: ConversationSummary): ChatMessage[] {
  return conversation.messages
    .filter((message) => message.role === "user" || message.role === "assistant")
    .map((message) => ({
      id: makeId(),
      role: message.role === "user" ? "user" : "assistant",
      content: message.content,
      createdAt: new Date(message.created_at).getTime(),
      status: "done",
      conversationId: conversation.id,
      result: message.metadata?.result,
      resultMeta: message.metadata?.result_meta,
    }));
}

export function titleForConversation(conversation: ConversationSummary) {
  return conversation.title?.trim() || "新会话";
}

export function applyAgentEventToAssistant(
  message: ChatMessage,
  event: AgentEvent,
  fallbackErrorMessage: string,
): ChatMessage {
  if (event.type === "conversation") {
    return { ...message, conversationId: event.data.conversation_id };
  }

  if (event.type === "progress") {
    return {
      ...message,
      steps: upsertStep(message.steps, event),
      parts: appendStatusPart(message.parts, event),
    };
  }

  if (event.type === "result") {
    return {
      ...message,
      result: event.data,
      resultMeta: event.meta,
      steps: upsertStep(message.steps, {
        type: "progress",
        step: "返回结果",
        status: "success",
      }),
    };
  }

  if (event.type === "answer_delta") {
    return {
      ...message,
      content: message.content + event.delta,
      parts: appendTextPart(message.parts, event.delta),
    };
  }

  if (event.type === "answer_done") {
    return finishAssistantMessage(message);
  }

  if (event.type === "usage") {
    return { ...message, usage: event.data };
  }

  return {
    ...message,
    status: "error",
    content: message.content,
    error: event.message || fallbackErrorMessage,
  };
}

export function finishAssistantMessage(message: ChatMessage): ChatMessage {
  return {
    ...message,
    status: "done",
    content: message.content || summarizeResult(message.result),
  };
}

function stageLabel(step: string) {
  const normalized = step.toLowerCase();
  if (/guard|安全|意图|理解|问题|rag/.test(normalized)) return "理解问题";
  if (/context|召回|检索|过滤|字段|指标|绑定|business/.test(normalized)) {
    return "检索上下文";
  }
  if (/generate|生成/.test(normalized)) return "生成查询";
  if (/result|返回|结果/.test(normalized)) return "返回结果";
  if (/sql|执行|校验|executor|validate/.test(normalized)) return "执行查询";
  return "检索上下文";
}

function displayStatusLabel(label: string) {
  if (label === "理解问题") return "正在思考";
  if (label === "检索上下文" || label === "生成查询") return "召回元数据并生成 SQL";
  if (label === "执行查询" || label === "返回结果") return "执行 SQL 并返回结果";
  return label;
}

function statusRank(label: string) {
  if (label === "正在思考") return 0;
  if (label === "召回元数据并生成 SQL") return 1;
  if (label === "执行 SQL 并返回结果") return 2;
  return 1;
}
