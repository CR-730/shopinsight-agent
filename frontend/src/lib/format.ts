export function cn(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

export function formatTime(timestamp: number) {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(timestamp);
}

export function formatDateTime(value: string | number | Date) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

export function summarizeResult(data: unknown) {
  if (Array.isArray(data)) {
    return data.length > 0 ? `查询完成，共 ${data.length} 行结果。` : "查询完成，结果为空。";
  }

  if (data && typeof data === "object") {
    return "查询完成，已返回结构化结果。";
  }

  if (data === null || data === undefined || data === "") {
    return "查询完成，结果为空。";
  }

  return `查询完成：${String(data)}`;
}

export function toClipboardText(value: unknown) {
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}
