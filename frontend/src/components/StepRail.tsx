import { cn } from "../lib/format";

export function ReasoningStatusLine({
  label,
  active,
}: {
  label: string;
  active?: boolean;
}) {
  return (
    <div
      className={cn(
        "aui-reasoning-trigger flex max-w-[75%] items-center gap-2 py-1 text-sm leading-none text-[#6e6e80] transition-colors",
        "animate-in fade-in duration-150",
      )}
      aria-live={active ? "polite" : undefined}
    >
      <span className="relative inline-block leading-none">
        <span>{label}</span>
        {active ? (
          <span aria-hidden className="shimmer pointer-events-none absolute inset-0 motion-reduce:animate-none">
            {label}
          </span>
        ) : null}
      </span>
    </div>
  );
}
