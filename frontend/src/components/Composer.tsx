import { ArrowUp, Square } from "lucide-react";
import { FormEvent, KeyboardEvent, useEffect, useRef } from "react";
import { cn } from "../lib/format";

type ComposerProps = {
  value: string;
  disabled: boolean;
  isStreaming: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
};

export function Composer({
  value,
  disabled,
  isStreaming,
  onChange,
  onSubmit,
  onStop,
}: ComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "0px";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
  }, [value]);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!disabled) onSubmit();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!disabled) onSubmit();
    }
  };

  return (
    <form onSubmit={submit} className="bg-white px-3 pb-3 pt-2 sm:px-4 sm:pb-5">
      <div className="mx-auto max-w-3xl">
        <div className="flex w-full items-end gap-2 rounded-[28px] border border-[#d9d9e3] bg-white px-3 py-2 shadow-[0_0_0_1px_rgba(0,0,0,0.02),0_8px_24px_rgba(0,0,0,0.06)] focus-within:border-[#10a37f] focus-within:ring-2 focus-within:ring-[#10a37f]/15">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={onKeyDown}
            rows={1}
            placeholder="询问电商数据..."
            className="max-h-40 min-h-10 flex-1 resize-none bg-transparent px-2 py-2.5 text-[15px] leading-6 text-[#202123] outline-none placeholder:text-[#8e8ea0]"
          />
          <button
            type={isStreaming ? "button" : "submit"}
            onClick={isStreaming ? onStop : undefined}
            disabled={!isStreaming && disabled}
            className={cn(
              "grid h-9 w-9 shrink-0 place-items-center rounded-full text-white transition focus:outline-none focus:ring-2 focus:ring-[#10a37f]/30 focus:ring-offset-2",
              isStreaming
                ? "bg-[#565869] hover:bg-[#353740]"
                : "bg-[#10a37f] hover:bg-[#0e906f] disabled:cursor-not-allowed disabled:bg-[#d9d9e3]",
            )}
            title={isStreaming ? "停止生成" : "发送"}
            aria-label={isStreaming ? "停止生成" : "发送"}
          >
            {isStreaming ? (
              <Square className="h-3.5 w-3.5 fill-current" aria-hidden="true" />
            ) : (
              <ArrowUp className="h-5 w-5" aria-hidden="true" />
            )}
          </button>
        </div>
        <p className="mt-2 text-center text-xs text-[#8e8ea0]">
          数据结果由当前后端链路生成，请以业务口径和数据库结果为准。
        </p>
      </div>
    </form>
  );
}
