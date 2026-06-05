export type ProgressStatus = "running" | "success" | "error" | "blocked";

export type ProgressEvent = {
  type: "progress";
  step: string;
  status: ProgressStatus;
};

export type ResultEvent = {
  type: "result";
  data: unknown;
  meta?: ResultMeta;
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

export type AnswerDeltaEvent = {
  type: "answer_delta";
  delta: string;
};

export type AnswerDoneEvent = {
  type: "answer_done";
};

export type ConversationEvent = {
  type: "conversation";
  data: {
    conversation_id: string;
    user_id: string;
  };
};

export type AgentEvent =
  | ProgressEvent
  | ResultEvent
  | ErrorEvent
  | UsageEvent
  | AnswerDeltaEvent
  | AnswerDoneEvent
  | ConversationEvent;

export type StepState = {
  step: string;
  label: string;
  status: ProgressStatus;
  updatedAt: number;
};

export type MessagePart =
  | {
      id: string;
      type: "status";
      label: string;
      status: ProgressStatus;
    }
  | {
      id: string;
      type: "text";
      content: string;
    };

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  status?: "streaming" | "done" | "error";
  steps?: StepState[];
  parts?: MessagePart[];
  result?: unknown;
  resultMeta?: ResultMeta;
  error?: string;
  usage?: UsageSummary;
  conversationId?: string;
};

export type ResultMeta = {
  tables?: string[];
};

export type ConversationMessage = {
  role: "user" | "assistant" | string;
  content: string;
  created_at: string;
  metadata?: {
    result?: unknown;
    result_meta?: ResultMeta;
    sql?: string;
    [key: string]: unknown;
  };
};

export type ConversationSummary = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ConversationMessage[];
};
