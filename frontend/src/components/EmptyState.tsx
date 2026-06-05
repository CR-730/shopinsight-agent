type EmptyStateProps = {
  examples: string[];
  onUseExample: (example: string) => void;
};

export function EmptyState({ examples, onUseExample }: EmptyStateProps) {
  return (
    <div className="mx-auto flex min-h-full max-w-3xl flex-col justify-center px-4 py-10">
      <div className="mb-7 text-center">
        <h1 className="text-[26px] font-medium leading-8 tracking-normal text-[#202123] sm:text-[28px]">
          今天想看什么数据？
        </h1>
      </div>

      <div className="mx-auto flex max-w-2xl flex-wrap justify-center gap-3">
        {examples.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => onUseExample(example)}
            className="h-11 rounded-full border border-[#d9d9e3] bg-white px-4 text-[14px] leading-none text-[#353740] transition hover:bg-[#f7f7f8] active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-[#10a37f]/25"
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}
