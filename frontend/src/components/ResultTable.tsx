import { Database } from "lucide-react";
import type { ResultMeta } from "../types/agent";

function normalizeRows(data: unknown): Array<Record<string, unknown>> {
  if (Array.isArray(data)) {
    return data.map((item, index) =>
      item && typeof item === "object" && !Array.isArray(item)
        ? (item as Record<string, unknown>)
        : { 序号: index + 1, 值: item },
    );
  }

  if (data && typeof data === "object") {
    return [data as Record<string, unknown>];
  }

  return [{ 值: data ?? "" }];
}

function formatCell(value: unknown) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") return value.toLocaleString("zh-CN");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export function ResultTable({ data, meta }: { data: unknown; meta?: ResultMeta }) {
  void meta;
  const rows = normalizeRows(data);
  const columns = Array.from(
    rows.reduce((keys, row) => {
      Object.keys(row).forEach((key) => keys.add(key));
      return keys;
    }, new Set<string>()),
  );

  if (columns.length === 0) return null;

  return (
    <section className="mt-4 overflow-hidden rounded-2xl border border-[#ececec] bg-white">
      <div className="flex items-center justify-between border-b border-[#ececec] px-4 py-3">
        <div className="flex items-center gap-2 text-sm font-medium text-[#202123]">
          <Database className="h-4 w-4 text-[#10a37f]" aria-hidden="true" />
          查询结果
        </div>
        <div className="min-w-0 text-right text-xs text-[#8e8ea0]">{rows.length} 行</div>
      </div>
      <div className="max-h-[360px] overflow-auto">
        <table className="min-w-full border-separate border-spacing-0 text-left text-sm">
          <thead className="sticky top-0 z-10 bg-[#f7f7f8]">
            <tr>
              {columns.map((column) => (
                <th
                  key={column}
                  scope="col"
                  className="border-b border-[#ececec] px-4 py-3 font-medium text-[#565869]"
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex} className="odd:bg-white even:bg-[#fafafa]">
                {columns.map((column) => (
                  <td key={column} className="border-b border-[#f0f0f0] px-4 py-3 text-[#353740]">
                    {formatCell(row[column])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
