/**
 * 智能体类型定义
 * 定义问数智能体前端使用的 SSE 事件、流程步骤和聊天消息类型
 */
export type ProgressStatus = "running" | "success" | "error";

export type ProgressEvent = {
  type: "progress";
  step: string;
  status: ProgressStatus;
};

export type ResultEvent = {
  type: "result";
  data: unknown;
};

export type ErrorEvent = {
  type: "error";
  message: string;
};

export type UsageSummary = {
  llm_input_tokens: number;
  llm_output_tokens: number;
  llm_total_tokens: number;
  embedding_tokens: number;
  llm_cost: number;
  embedding_cost: number;
  total_cost: number;
  currency: string;
  embedding_estimated: boolean;
  calls: Array<Record<string, unknown>>;
};

export type UsageEvent = {
  type: "usage";
  data: UsageSummary;
};

export type ConversationEvent = {
  type: "conversation";
  conversation_id: string;
  rewritten_query: string;
};

export type AgentEvent =
  | ProgressEvent
  | ResultEvent
  | ErrorEvent
  | UsageEvent
  | ConversationEvent;

export type StepState = {
  step: string;
  status: ProgressStatus;
  updatedAt: number;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  status?: "streaming" | "done" | "error";
  steps?: StepState[];
  result?: unknown;
  error?: string;
  usage?: UsageSummary;
  conversationId?: string;
  rewrittenQuery?: string;
};
