# Eval 性能与成本对比报告

本文记录 2026-06-02 当前工作区的 eval 性能、token 和成本数据。数据来自 `eval/runs/*.json`，只统计完整 21 case 运行。

## 对比结果

| 报告 | 通过率 | 总耗时 | 平均耗时 | LLM 总 token | 平均 token | 总成本 |
|---|---:|---:|---:|---:|---:|---:|
| `guard-p1-20260601-212819.json` | 16/21 | 266.259s | 12.679s | 162863 | 7755.4 | 0.188974 CNY |
| `dashscope-embedding-20260602-015658.json` | 19/21 | 275.488s | 13.118s | 118985 | 5666.0 | 0.153744 CNY |
| `business-binding-final-20260602-122348.json` | 21/21 | 181.756s | 8.655s | 113238 | 5392.3 | 0.128298 CNY |

## 最新通过报告

最新 fresh run：

```text
eval/runs/business-binding-final-20260602-122348.json
passed: 21
failed: 0
pass_rate: 1.0
```

总体指标：

```text
total_latency_seconds: 181.756
avg_latency_seconds: 8.655
llm_total_tokens: 113238
avg_llm_tokens: 5392.3
embedding_tokens: 832
total_cost: 0.128298 CNY
avg_cost: 0.006109 CNY
```

## 节点耗时

最新 run 的主要节点耗时：

| 节点 | 总耗时 | 平均耗时 | 调用次数 |
|---|---:|---:|---:|
| `correct_sql` | 98908.3ms | 98908.3ms | 1 |
| `generate_sql` | 28491.3ms | 1582.8ms | 18 |
| `recall_metric` | 19875.6ms | 993.8ms | 20 |
| `recall_column` | 18566.9ms | 928.3ms | 20 |
| `pre_rag_guard` | 18035.7ms | 858.8ms | 21 |
| `recall_value` | 17880.6ms | 894.0ms | 20 |
| `filter_table` | 12658.6ms | 632.9ms | 20 |
| `extract_keywords` | 992.1ms | 49.6ms | 20 |

## LLM 调用耗时

最新 run 的主要 LLM 调用耗时：

| 步骤 | 总耗时 | 平均耗时 | token | cache hit | retry |
|---|---:|---:|---:|---:|---:|
| `校正SQL` | 98893.7ms | 98893.7ms | 6486 | 0/1 | 0 |
| `生成SQL` | 28382.4ms | 1576.8ms | 22487 | 0/18 | 0 |
| `RAG前安全分类` | 17982.4ms | 899.1ms | 18070 | 0/20 | 0 |
| `过滤表信息` | 12505.2ms | 625.3ms | 36194 | 0/20 | 0 |
| `召回指标信息` | 11120.6ms | 556.0ms | 11635 | 0/20 | 0 |
| `召回字段信息` | 10011.2ms | 500.6ms | 10084 | 0/20 | 0 |
| `召回字段取值` | 8983.2ms | 449.2ms | 8282 | 0/20 | 0 |

## 结论

业务绑定收敛后，质量和成本都有改善：

- 通过率从 `16/21`、`19/21` 提升到 `21/21`。
- LLM token 从 `162863` 降到 `113238`，相对 `guard-p1` 下降约 `30.47%`。
- 总成本从 `0.188974 CNY` 降到 `0.128298 CNY`，相对 `guard-p1` 下降约 `32.11%`。
- 平均耗时从 `12.679s` 降到 `8.655s`，相对 `guard-p1` 下降约 `31.74%`。

需要注意：

- 最新 run 仍有一次 `correct_sql`，单次约 `98.9s`，它是当前最大长尾。
- 非修正链路中，`generate_sql` 平均约 `1.58s`，不是主要长尾。
- 当前 LLM response cache 在 eval 中没有命中，主要收益来自业务绑定收敛、SQL thinking 关闭、fast/sql 模型分流和减少失败重试链路。

下一步性能治理优先级：

1. 降低 `correct_sql` 长尾：保留 thinking，但增加更短 timeout 或更明确的 SQL 修正输入。
2. 减少 `filter_table` token：可考虑让它消费 `business_binding` 和表关系上下文，而不是完整表上下文。
3. 观测 embedding latency：当前 embedding usage 记录没有 latency，无法量化 DashScope embedding 的实际耗时。
