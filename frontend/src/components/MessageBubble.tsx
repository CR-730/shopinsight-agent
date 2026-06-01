/**
 * 聊天消息气泡组件
 * 组合展示用户问题、智能体回复、执行流程和结果表格
 */
import { Bot, Coins, Copy, UserRound } from "lucide-react";
import { ResultTable } from "./ResultTable";
import { StepRail } from "./StepRail";
import { cn, formatTime, toClipboardText } from "../lib/format";
import type { ChatMessage } from "../types/agent";

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  const copy = async () => {
    const text = message.result ? toClipboardText(message.result) : message.content;
    await navigator.clipboard.writeText(text);
  };

  return (
    <article className={cn("group flex gap-3", isUser && "justify-end")}>
      {!isUser && (
        <div className="mt-1 grid h-9 w-9 shrink-0 place-items-center rounded-full bg-ink text-parchment">
          <Bot className="h-4 w-4" aria-hidden="true" />
        </div>
      )}

      <div className={cn("max-w-[920px] flex-1", isUser && "flex max-w-[760px] justify-end")}>
        <div
          className={cn(
            "relative border px-5 py-4 shadow-line",
            isUser
              ? "border-ink/80 bg-ink text-parchment"
              : "border-ink/10 bg-[#fffaf1]/78 text-ink backdrop-blur",
          )}
        >
          <div className="flex items-start justify-between gap-3">
            <p className="whitespace-pre-wrap text-[15px] leading-7">{message.content}</p>
            {!isUser && message.status !== "streaming" && (
              <button
                type="button"
                onClick={copy}
                className="shrink-0 rounded-full p-1.5 text-ink/45 opacity-0 outline-none transition hover:bg-ink/5 hover:text-ink focus:opacity-100 focus:ring-2 focus:ring-moss/40 group-hover:opacity-100"
                title="复制"
                aria-label="复制"
              >
                <Copy className="h-4 w-4" aria-hidden="true" />
              </button>
            )}
          </div>

          {message.error && (
            <div className="mt-3 border border-tomato/30 bg-tomato/10 px-3 py-2 text-sm text-tomato">
              {message.error}
            </div>
          )}

          {!isUser && <StepRail steps={message.steps} />}
          {!isUser && message.result !== undefined && <ResultTable data={message.result} />}
          {!isUser && message.usage !== undefined && <UsageSummary usage={message.usage} />}

          <div
            className={cn(
              "mt-3 text-xs",
              isUser ? "text-parchment/55" : "text-ink/45",
            )}
          >
            {formatTime(message.createdAt)}
          </div>
        </div>
      </div>

      {isUser && (
        <div className="mt-1 grid h-9 w-9 shrink-0 place-items-center rounded-full bg-moss text-white">
          <UserRound className="h-4 w-4" aria-hidden="true" />
        </div>
      )}
    </article>
  );
}

function UsageSummary({ usage }: { usage: NonNullable<ChatMessage["usage"]> }) {
  const items = [
    ["输入", usage.llm_input_tokens.toLocaleString()],
    ["输出", usage.llm_output_tokens.toLocaleString()],
    ["Embedding", usage.embedding_tokens.toLocaleString()],
    ["成本", formatCost(usage.total_cost, usage.currency)],
  ];

  return (
    <div className="mt-4 border border-ink/10 bg-white/45 px-3 py-3">
      <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-ink/60">
        <Coins className="h-3.5 w-3.5" aria-hidden="true" />
        Token 成本
        {usage.embedding_estimated && <span className="font-normal text-ink/40">Embedding 为估算</span>}
      </div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {items.map(([label, value]) => (
          <div key={label} className="border border-ink/10 bg-parchment/55 px-2 py-2">
            <div className="text-[11px] text-ink/45">{label}</div>
            <div className="mt-1 font-mono text-xs text-ink">{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function formatCost(value: number, currency: string) {
  if (value === 0) return currency === "USD" ? "$0" : `${currency} 0`;
  if (currency === "CNY") return `¥${value.toFixed(6)}`;
  if (currency === "USD") return `$${value.toFixed(6)}`;
  return `${currency} ${value.toFixed(6)}`;
}
