import {
  Menu,
  MessageSquare,
  MessageSquarePlus,
  MoreHorizontal,
  PanelLeftClose,
  Trash2,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Composer } from "./components/Composer";
import { EmptyState } from "./components/EmptyState";
import { MessageBubble } from "./components/MessageBubble";
import {
  deleteConversation,
  getConversation,
  listConversations,
  streamQuery,
} from "./lib/agentApi";
import {
  appendStatusPart,
  appendTextPart,
  makeId,
  messagesFromConversation,
  titleForConversation,
  upsertStep,
} from "./lib/chatMessages";
import { cn, formatDateTime, summarizeResult } from "./lib/format";
import type { AgentEvent, ChatMessage, ConversationSummary } from "./types/agent";

const examples = [
  "一季度各大区 GMV 排名",
  "3 月各品类销量和销售额",
  "华东一季度 TOP5 商品",
  "按会员等级看订单和销售额",
];

const ASSISTANT_ERROR_MESSAGE = "出了点问题，请稍后重试。";

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [isLoadingThreads, setIsLoadingThreads] = useState(true);
  const [threadError, setThreadError] = useState<string | null>(null);
  const [activeController, setActiveController] = useState<AbortController | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const isStreaming = Boolean(activeController);
  const canSubmit = draft.trim().length > 0 && !isStreaming;
  const activeConversation = useMemo(
    () => conversations.find((item) => item.id === conversationId),
    [conversationId, conversations],
  );

  useEffect(() => {
    const controller = new AbortController();
    void refreshConversations(controller.signal);
    return () => controller.abort();
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  async function refreshConversations(signal?: AbortSignal) {
    const timeout = window.setTimeout(() => {
      setIsLoadingThreads(false);
      setThreadError("历史会话暂时不可用，仍可开始新会话。");
    }, 3000);
    try {
      setIsLoadingThreads(true);
      setThreadError(null);
      const items = await listConversations(signal);
      setConversations(items);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setThreadError("历史会话暂时不可用，仍可开始新会话。");
    } finally {
      window.clearTimeout(timeout);
      setIsLoadingThreads(false);
    }
  }

  async function openConversation(id: string) {
    if (isStreaming || id === conversationId) return;
    const controller = new AbortController();
    try {
      const conversation = await getConversation(id, controller.signal);
      setConversationId(conversation.id);
      setMessages(messagesFromConversation(conversation));
      setSidebarOpen(false);
    } catch (error) {
      setThreadError("无法打开这条历史会话，请稍后再试。");
    }
  }

  function startNewConversation() {
    if (isStreaming) return;
    setConversationId(null);
    setMessages([]);
    setDraft("");
    setSidebarOpen(false);
  }

  async function removeConversation(id: string) {
    if (isStreaming) return;
    const controller = new AbortController();
    try {
      await deleteConversation(id, controller.signal);
      setConversations((current) => current.filter((item) => item.id !== id));
      if (id === conversationId) {
        setConversationId(null);
        setMessages([]);
      }
    } catch (error) {
      setThreadError("无法删除这条历史会话，请稍后再试。");
    }
  }

  async function startQuery(rawQuery = draft) {
    const query = rawQuery.trim();
    if (!query || isStreaming) return;

    const userMessage: ChatMessage = {
      id: makeId(),
      role: "user",
      content: query,
      createdAt: Date.now(),
    };

    const assistantId = makeId();
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      createdAt: Date.now(),
      status: "streaming",
      steps: [],
    };

    const controller = new AbortController();
    setActiveController(controller);
    setDraft("");
    setMessages((current) => [...current, userMessage, assistantMessage]);

    const onEvent = (event: AgentEvent) => {
      if (event.type === "conversation") {
        setConversationId(event.data.conversation_id);
      }

      setMessages((current) =>
        current.map((message) => {
          if (message.id !== assistantId) return message;

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
            return {
              ...message,
              status: "done",
              content: message.content || summarizeResult(message.result),
            };
          }

          if (event.type === "usage") {
            return { ...message, usage: event.data };
          }

          return {
            ...message,
            status: "error",
            content: message.content,
            error: event.message || ASSISTANT_ERROR_MESSAGE,
          };
        }),
      );
    };

    try {
      await streamQuery(query, { signal: controller.signal, conversationId, onEvent });
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId && message.status === "streaming"
            ? {
                ...message,
                status: "done",
                content: message.content || summarizeResult(message.result),
              }
            : message,
        ),
      );
      await refreshConversations();
    } catch (error) {
      const isAbort = error instanceof DOMException && error.name === "AbortError";
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                status: isAbort ? "done" : "error",
                content: isAbort ? message.content || "已停止本次查询。" : "无法连接问数接口。",
                error: isAbort ? undefined : ASSISTANT_ERROR_MESSAGE,
              }
            : message,
        ),
      );
    } finally {
      setActiveController(null);
    }
  }

  function stopQuery() {
    activeController?.abort();
  }

  return (
    <div className="flex h-dvh overflow-hidden bg-white text-[#202123]">
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-[280px] max-w-[calc(100vw-24px)] flex-col border-r border-[#e5e5e5] bg-[#f9f9f9] transition-transform lg:static lg:translate-x-0",
          sidebarOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex h-14 items-center justify-between px-3">
          <button
            type="button"
            onClick={startNewConversation}
            disabled={isStreaming}
            className="flex h-10 min-w-0 flex-1 items-center gap-2 rounded-xl px-3 text-sm font-medium text-[#202123] transition hover:bg-[#ececec] disabled:cursor-not-allowed disabled:opacity-50"
          >
            <MessageSquarePlus className="h-4 w-4" aria-hidden="true" />
            新建会话
          </button>
          <button
            type="button"
            onClick={() => setSidebarOpen(false)}
            className="ml-1 grid h-10 w-10 place-items-center rounded-xl text-[#565869] hover:bg-[#ececec] lg:hidden"
            aria-label="关闭侧栏"
          >
            <PanelLeftClose className="h-4 w-4" />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
          <div className="px-2 pb-2 pt-1 text-xs font-medium text-[#8e8ea0]">历史会话</div>
          {isLoadingThreads ? (
            <div className="space-y-2 px-2">
              {Array.from({ length: 6 }).map((_, index) => (
                <div key={index} className="h-9 animate-pulse rounded-xl bg-[#ececec]" />
              ))}
            </div>
          ) : conversations.length > 0 ? (
            <div className="space-y-1">
              {conversations.map((conversation) => (
                <div
                  key={conversation.id}
                  className={cn(
                    "group relative flex h-11 items-center rounded-xl transition",
                    conversation.id === conversationId
                      ? "bg-[#ececec] text-[#202123]"
                      : "text-[#353740] hover:bg-[#ececec]",
                  )}
                >
                  <button
                    type="button"
                    onClick={() => openConversation(conversation.id)}
                    disabled={isStreaming}
                    className="flex h-full min-w-0 flex-1 items-center gap-2 rounded-xl px-3 pr-20 text-left text-sm disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    <MessageSquare className="h-4 w-4 shrink-0 text-[#8e8ea0]" />
                    <span className="min-w-0 flex-1 truncate">{titleForConversation(conversation)}</span>
                  </button>
                  <div className="absolute right-1 top-1/2 flex -translate-y-1/2 items-center gap-0.5 opacity-0 transition group-hover:opacity-100 group-focus-within:opacity-100">
                    <button
                      type="button"
                      onClick={() => removeConversation(conversation.id)}
                      disabled={isStreaming}
                      className="grid h-8 w-8 place-items-center rounded-lg text-[#b42318] hover:bg-[#fff4f2] disabled:cursor-not-allowed disabled:opacity-50"
                      aria-label="删除会话"
                      title="删除会话"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                    <button
                      type="button"
                      disabled={isStreaming}
                      className="grid h-8 w-8 place-items-center rounded-lg text-[#565869] hover:bg-[#dedede] disabled:cursor-not-allowed disabled:opacity-50"
                      aria-label="更多选项"
                      title="更多选项"
                    >
                      <MoreHorizontal className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="rounded-xl px-3 py-3 text-sm leading-6 text-[#8e8ea0]">
              暂无历史。发送第一条问题后，会话会保存在这里。
            </div>
          )}
          {threadError && (
            <div className="mx-2 mt-3 rounded-xl border border-[#ffd7d2] bg-[#fff4f2] px-3 py-2 text-xs leading-5 text-[#b42318]">
              {threadError}
            </div>
          )}
        </div>

        <div className="border-t border-[#e5e5e5] px-4 py-3 text-xs text-[#8e8ea0]">
          Shopkeeper Agent
        </div>
      </aside>

      {sidebarOpen && (
        <button
          type="button"
          className="fixed inset-0 z-30 bg-black/20 lg:hidden"
          aria-label="关闭侧栏遮罩"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <main className="flex min-h-0 min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-[#ececec] bg-white px-3 sm:px-4">
          <div className="flex min-w-0 items-center gap-2">
            <button
              type="button"
              onClick={() => setSidebarOpen(true)}
              className="grid h-10 w-10 place-items-center rounded-xl text-[#565869] hover:bg-[#f4f4f4] lg:hidden"
              aria-label="打开侧栏"
            >
              <Menu className="h-5 w-5" />
            </button>
            <div className="min-w-0">
              <div className="truncate text-sm font-medium text-[#202123]">
                {activeConversation ? titleForConversation(activeConversation) : "新会话"}
              </div>
              <div className="truncate text-xs text-[#8e8ea0]">
                {activeConversation ? formatDateTime(activeConversation.updated_at) : "准备开始问数"}
              </div>
            </div>
          </div>
        </header>

        <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto scroll-smooth">
          {messages.length === 0 ? (
            <EmptyState examples={examples} onUseExample={(example) => setDraft(example)} />
          ) : (
            <div className="mx-auto flex max-w-3xl flex-col gap-7 px-4 py-8">
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
            </div>
          )}
        </div>

        <Composer
          value={draft}
          disabled={!canSubmit}
          isStreaming={isStreaming}
          onChange={setDraft}
          onSubmit={() => startQuery()}
          onStop={stopQuery}
        />
      </main>
    </div>
  );
}
