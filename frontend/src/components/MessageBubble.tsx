import { Check, Copy } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import { PrismLight as SyntaxHighlighter } from "react-syntax-highlighter";
import sql from "react-syntax-highlighter/dist/esm/languages/prism/sql";
import { oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import remarkGfm from "remark-gfm";
import { ResultTable } from "./ResultTable";
import { ReasoningStatusLine } from "./StepRail";
import { cn, formatTime, toClipboardText } from "../lib/format";
import type { ChatMessage, MessagePart } from "../types/agent";

SyntaxHighlighter.registerLanguage("sql", sql);

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);
  const smoothContent = useSmoothText(message.content, message.status === "streaming");

  const copy = async () => {
    const text = message.result ? toClipboardText(message.result) : message.content;
    await navigator.clipboard.writeText(text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  };

  return (
    <article className={cn("group w-full", isUser ? "flex justify-end" : "block")}>
      <div className={cn("max-w-3xl", isUser ? "ml-auto" : "mx-auto")}>
        {isUser ? (
          <div className="rounded-3xl bg-[#f4f4f4] px-5 py-3 text-[15px] leading-7 text-[#202123]">
            <p className="whitespace-pre-wrap">{message.content}</p>
          </div>
        ) : (
          <div className="px-2 py-1 text-[15px] leading-7 text-[#202123]">
            <div className="min-w-0 flex-1">
              {message.parts?.length ? (
                <AssistantParts parts={message.parts} />
              ) : (
                <div className="prose-chat">
                  {message.content ? (
                    <AssistantContent content={smoothContent} />
                  ) : null}
                </div>
              )}
              {message.error && (
                <div className="mt-2 line-clamp-2 rounded-md border border-[#ffd7d2] bg-[#fff4f2] p-3 text-sm text-[#b42318]">
                  {message.error}
                </div>
              )}
              {message.result !== undefined && <ResultTable data={message.result} meta={message.resultMeta} />}
              <div className="mt-2 flex min-h-8 items-center gap-1 text-[#8e8ea0] opacity-0 transition group-hover:opacity-100">
                <button
                  type="button"
                  onClick={copy}
                  className="inline-grid h-8 w-8 place-items-center rounded-lg transition hover:bg-[#f4f4f4] hover:text-[#202123] focus:outline-none focus:ring-2 focus:ring-[#10a37f]/25"
                  title="复制"
                  aria-label="复制"
                >
                  {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </button>
                <span className="text-xs">{formatTime(message.createdAt)}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </article>
  );
}

function AssistantParts({ parts }: { parts: MessagePart[] }) {
  return (
    <div className="space-y-3">
      {parts.map((part) => {
        if (part.type === "status") {
          return <ReasoningStatusLine key={part.id} label={part.label} active={part.status === "running"} />;
        }
        return (
          <div key={part.id} className="prose-chat">
            <AssistantContent content={part.content} />
          </div>
        );
      })}
    </div>
  );
}

function useSmoothText(text: string, running: boolean) {
  const [displayedText, setDisplayedText] = useState(running ? "" : text);

  if (!text.startsWith(displayedText)) {
    setDisplayedText(running ? "" : text);
  }

  const animator = useMemo(() => {
    return new TextStreamAnimator(displayedText, setDisplayedText);
    // Match assistant-ui: create one animator for the message part lifetime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!text.startsWith(animator.targetText)) {
      if (running) {
        animator.currentText = "";
        animator.targetText = text;
        animator.start();
      } else {
        animator.currentText = text;
        animator.targetText = text;
        animator.stop();
        setDisplayedText(text);
      }
      return;
    }

    animator.targetText = text;
    animator.start();
  }, [animator, running, text]);

  useEffect(() => () => animator.stop(), [animator]);

  return running || displayedText !== text ? displayedText : text;
}

class TextStreamAnimator {
  private animationFrameId: number | null = null;
  private lastUpdateTime = Date.now();
  public targetText = "";

  constructor(
    public currentText: string,
    private setText: (newText: string) => void,
  ) {}

  start() {
    if (this.animationFrameId !== null) return;
    this.lastUpdateTime = Date.now();
    this.animate();
  }

  stop() {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  private animate = () => {
    const currentTime = Date.now();
    const deltaTime = currentTime - this.lastUpdateTime;
    let timeToConsume = deltaTime;

    const remainingChars = this.targetText.length - this.currentText.length;

    const baseTimePerChar = Math.min(5, 250 / remainingChars);
    let charsToAdd = 0;
    while (timeToConsume >= baseTimePerChar && charsToAdd < remainingChars) {
      charsToAdd += 1;
      timeToConsume -= baseTimePerChar;
    }

    if (charsToAdd !== remainingChars) {
      this.animationFrameId = requestAnimationFrame(this.animate);
    } else {
      this.animationFrameId = null;
    }
    if (charsToAdd === 0) return;

    this.currentText = this.targetText.slice(0, this.currentText.length + charsToAdd);
    this.lastUpdateTime = currentTime - timeToConsume;
    this.setText(this.currentText);
  };
}

function AssistantContent({ content }: { content: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
      {content}
    </ReactMarkdown>
  );
}

const markdownComponents: Components = {
  p: ({ className, ...props }) => (
    <p className={cn("my-2.5 leading-7 first:mt-0 last:mb-0", className)} {...props} />
  ),
  ul: ({ className, ...props }) => (
    <ul className={cn("my-2 ml-5 list-disc space-y-1", className)} {...props} />
  ),
  ol: ({ className, ...props }) => (
    <ol className={cn("my-2 ml-5 list-decimal space-y-1", className)} {...props} />
  ),
  li: ({ className, ...props }) => <li className={cn("leading-7", className)} {...props} />,
  pre: ({ children }) => <>{children}</>,
  code: ({ className, children, ...props }) => {
    const match = /language-(\w+)/.exec(className || "");
    const code = String(children ?? "").replace(/\n$/, "");
    if (!match) {
      return (
        <code
          className={cn("rounded-md border border-[#e5e5e5] bg-[#f7f7f8] px-1.5 py-0.5 font-mono text-[0.86em]", className)}
          {...props}
        >
          {children}
        </code>
      );
    }

    return <CodeBlock language={match[1]} code={code} />;
  },
};

function CodeBlock({ language, code }: { language: string; code: string }) {
  const [copied, setCopied] = useState(false);
  const displayCode = language.toLowerCase() === "sql" ? formatSql(code) : code;

  async function copyCode() {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  }

  return (
    <div className="my-3 overflow-hidden rounded-xl border border-[#d9d9e3] bg-[#f7f7f8]">
      <div className="flex items-center justify-between border-b border-[#e5e5e5] bg-[#f7f7f8] px-3 py-1.5 text-xs">
        <span className="font-medium lowercase text-[#6e6e80]">{language}</span>
        <button
          type="button"
          onClick={copyCode}
          className="inline-flex h-7 items-center gap-1.5 rounded-md px-2 text-[#565869] transition hover:bg-[#ececec] hover:text-[#202123]"
          aria-label="复制代码"
          title="复制代码"
        >
          {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
          <span>{copied ? "已复制" : "复制"}</span>
        </button>
      </div>
      <SyntaxHighlighter
        language={language}
        style={oneLight}
        PreTag="div"
        wrapLongLines
        customStyle={{
          margin: 0,
          padding: "14px 16px",
          background: "#f7f7f8",
          fontSize: "13px",
          lineHeight: "1.7",
          whiteSpace: "pre-wrap",
          overflowX: "auto",
        }}
        codeTagProps={{
          style: {
            fontFamily:
              'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
          },
        }}
      >
        {displayCode}
      </SyntaxHighlighter>
    </div>
  );
}

function formatSql(sqlText: string) {
  return sqlText
    .replace(/\s+/g, " ")
    .replace(/\b(SELECT|FROM|WHERE|JOIN|LEFT JOIN|RIGHT JOIN|INNER JOIN|GROUP BY|ORDER BY|HAVING|LIMIT)\b/gi, "\n$1")
    .replace(/\b(AND|OR)\b/gi, "\n  $1")
    .replace(/,\s*/g, ",\n  ")
    .trim();
}
