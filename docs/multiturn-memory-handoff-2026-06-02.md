# 多轮问数记忆、追问改写和会话管理任务交接文档

日期：2026-06-02

交接目标：把当前电商问数 Agent 的下一阶段工作交给新对话，重点实现“更复杂的多轮问数记忆、追问改写和会话管理”。当前不要继续扩安全校验、RAG 规则或 SQL 规则，除非多轮能力确实需要最小接口调整。

## 1. 当前仓库状态

当前已经完成并提交的阶段性工作：

```text
36f82c1 fix: report sql execution timeouts accurately
94f8b50 feat: observe embedding latency and compact table context
0dd9f71 docs: document binding architecture and eval performance
a5e1e28 refactor: consolidate business binding decisions
51c65fc feat: version metadata aliases and retrieval indexes
00c9a3d feat: add llm resilience and usage governance
3cfccf1 chore: initialize shopkeeper agent
```

最近验证结果：

```text
uv run pytest -q
103 passed

uv run ruff check .
All checks passed
```

已有关键文档：

- `docs/five-optimizations-report-2026-06-02.md`
- `docs/business-binding-architecture.md`
- `docs/eval-performance-report-2026-06-02.md`

## 2. 当前主链路

当前 LangGraph 主链路大致为：

```text
pre_rag_guard
  -> extract_keywords
  -> recall_column / recall_value / recall_metric
  -> merge_retrieved_info
  -> business_binding
  -> filter_metric / filter_table
  -> add_extra_context
  -> semantic_guard
  -> generate_sql
  -> pre_sql_execution_validation
  -> correct_sql 或 run_sql
```

关键文件：

- Graph：`app/agent/graph.py`
- State：`app/agent/state.py`
- Context：`app/agent/context.py`
- SQL loop：`app/agent/sql_loop.py`
- Eval：`app/scripts/run_eval.py`
- Eval cases：`examples/eval_cases.yaml`

## 3. 当前架构边界

### 3.1 business_binding 是唯一业务语义裁决源

当前已经把业务语义判断收敛到：

- `app/agent/nodes/business_binding.py`

它输出：

```yaml
business_binding:
  metrics: []
  filters: []
  time: null
  unresolved: []
  ambiguous: []
```

后续节点只消费 binding，不再重新解释业务对象：

- `filter_metric`：只按 `metric_bindings` 裁剪；没有 binding 时保留 RAG top-k。
- `semantic_guard`：只检查 `unresolved` / `ambiguous`。
- `generate_sql`：按 binding 生成 SQL。
- `pre_sql_execution_validation`：只做执行前硬校验，不做自然语言业务解析。

下一阶段做多轮时，必须保持这个边界。多轮记忆和追问改写应该产出“改写后的 query / contextual query / memory context”，然后仍然进入 `business_binding` 做业务裁决，不要在会话层直接判断指标、枚举、SQL。

### 3.2 不要继续堆 guard

当前已有：

- `pre_rag_guard`：RAG 前输入安全。
- `semantic_guard`：binding 完整性检查。
- `pre_sql_execution_validation`：SQL 执行前硬闸门。

多轮阶段不要新增第四层业务 guard。否则会重新出现“同一类判断多个节点重复裁决”的问题。

### 3.3 SQL 执行前硬校验必须保留

`pre_sql_execution_validation` 已经承担执行前确定性兜底：

- 单 SELECT。
- deny keywords。
- SELECT *。
- 敏感字段。
- 明细查询。
- 未绑定枚举字面量。
- join relationship。
- MySQL explain。

多轮改写不能绕过它，也不能为了追问体验跳过它。

## 4. 已完成的能力不要重复实现

### 4.1 元数据自动构建和版本边界

已完成：

- `app/services/meta_knowledge_scheduler.py`
- `app/services/meta_knowledge_service.py`
- `metadata_build` active version。
- Qdrant / ES 查询侧版本过滤。
- LLM cache namespace 绑定 `metadata_cache_version`。

多轮会话如果要缓存上下文，也应把 metadata version 纳入边界，避免老会话引用旧元数据。

### 4.2 字段取值混合检索

已完成：

- ES 全文检索。
- Qdrant 向量检索。
- `fuse_ranked_value_infos()` 排名融合。
- value alias 写入 Meta MySQL。
- RAG 漏召回时 `column_value_exists()` fallback。

多轮追问改写不要自己做枚举值推断，应改写 query 后走现有混合检索和 business_binding。

### 4.3 SQL 修正循环

已完成：

```text
generate_sql
  -> pre_sql_execution_validation
       repairable_error -> correct_sql -> pre_sql_execution_validation
       blocked          -> END
       pass             -> run_sql
```

多轮阶段不需要新增 SQL 反思节点。若追问依赖上一轮 SQL，可作为 memory context 输入，但最终 SQL 仍走现有生成和校验链路。

### 4.4 评测体系

已完成：

- 21 个 eval case。
- 结构化字段、能力标签、风险点、预期工具调用。
- usage / cost / latency / failure stage 报告。
- `run_eval` 失败 exit 1。

多轮阶段应新增一组独立 suite，例如：

```yaml
suite: conversation
capabilities:
  - conversation_memory
  - followup_rewrite
  - clarification
```

不要把多轮 case 混进现有单轮 case 后让失败归因变乱。

### 4.5 性能治理

已完成：

- 节点级 latency。
- LLM / Embedding usage。
- Embedding cache。
- LLM response cache。
- metadata cache namespace。
- fast/sql 模型路由。
- retry / rate limit / circuit breaker / half-open。
- per-request LLM call budget。
- SQL execution timeout。

多轮阶段新增 LLM 节点时，必须统一通过 `ainvoke_llm_with_usage()`，不要裸调模型，否则 usage/cost/latency 会断。

## 5. 下一阶段建议目标

建议把“多轮问数记忆、追问改写和会话管理”拆成三个最小闭环，而不是一次性做复杂 Agent 记忆系统。

### 5.1 会话状态模型

目标：保存上一轮足够少、但足以改写追问的信息。

建议新增结构：

```yaml
conversation_state:
  session_id: string
  turn_id: int
  last_user_query: string
  last_rewritten_query: string
  last_business_binding: object
  last_sql: string
  last_final_answer_summary: string
  last_trace:
    metrics: []
    filters: []
    time: null
    group_by_columns: []
```

注意：

- 不要保存完整大 prompt。
- 不要保存完整 result 明细，尤其不要保存敏感明细。
- 不要把 memory 当事实源；事实源仍是 Meta MySQL / DW / RAG。

可能涉及文件：

- `app/agent/state.py`
- `app/agent/context.py`
- 新增 `app/agent/conversation.py` 或 `app/services/conversation_memory.py`

### 5.2 追问改写节点

目标：把“那华东呢”“这个月呢”“按品类拆一下”改写成完整单轮问句。

建议新增节点位置：

```text
pre_rag_guard
  -> rewrite_followup_query
  -> extract_keywords
```

或者如果需要安全优先：

```text
pre_rag_guard
  -> rewrite_followup_query
  -> pre_rag_guard_again_for_rewritten_query
  -> extract_keywords
```

第二种更安全但多一次 guard；第一版可先保持单次 `pre_rag_guard`，但改写后的 query 不能绕过后续 `semantic_guard` 和 `pre_sql_execution_validation`。

建议输出：

```yaml
query: 原始用户输入
rewritten_query: 完整问数问题
rewrite_reason: 使用了上一轮的哪些上下文
conversation_rewrite:
  is_followup: true
  carried_metrics: []
  carried_filters: []
  changed_filters: []
  carried_time: null
  changed_time: null
  ambiguity: null
```

注意：

- 改写节点不做业务对象最终裁决。
- 它只能“补全上下文”，最终是否存在指标/枚举仍交给 `business_binding`。
- 改写失败或信息不足时，应进入追问，而不是猜。

可能涉及文件：

- 新增 `app/agent/nodes/rewrite_followup_query.py`
- 修改 `app/agent/graph.py`
- 修改 `app/agent/state.py`
- 新增 prompt：`app/prompt/templates/rewrite_followup_query.txt` 或同项目现有 prompt 目录命名风格

### 5.3 澄清追问机制

目标：对不完整、歧义、多候选问题返回追问，而不是强行 SQL。

当前已有：

- `business_binding.unresolved`
- `business_binding.ambiguous`
- `semantic_guard` 阻断 unresolved / ambiguous

建议扩展方式：

- 不新增新 guard。
- 在 `semantic_guard` 阻断时，如果是可澄清问题，返回结构化 `clarification_request`。

示例：

```yaml
clarification_request:
  reason: ambiguous_filter
  question: 你想看华北、华东还是全部大区？
  options:
    - 华北
    - 华东
    - 全部大区
```

下一轮用户回答“华北”后，`rewrite_followup_query` 应把它和上轮 pending clarification 合并成完整 query。

可能涉及文件：

- `app/agent/nodes/semantic_guard.py`
- `app/agent/state.py`
- 新增 eval case

## 6. 推荐最小实现顺序

### 第一步：只做内存型会话上下文，不接数据库

目的：先验证链路，不引入存储复杂度。

建议：

- 在 API 层或 graph context 中传入 `session_id`。
- 用进程内 dict 保存最近 N 轮。
- 支持测试注入 memory store。

验收 case：

```text
turn1: 统计华北地区销售额
turn2: 那华东呢
expected rewritten_query: 统计华东地区销售额
```

### 第二步：新增 followup rewrite 节点

只处理三类高频追问：

1. 替换过滤值：那华东呢、苹果呢、手机数码呢。
2. 替换时间：这个月呢、上个月呢、2025 年第一季度呢。
3. 增加分组：按品类拆一下、按大区看。

暂时不要支持复杂跨轮推理。

### 第三步：新增 conversation eval suite

新增文件可选：

- `examples/eval_conversation_cases.yaml`

或扩展现有 eval runner 支持多 turn case：

```yaml
- id: conv_region_followup_replace_filter
  suite: conversation
  turns:
    - query: 统计华北地区销售额
      expected_sql_contains: [华北, order_amount]
    - query: 那华东呢
      expected_rewritten_query_contains: [华东, 销售额]
      expected_sql_contains: [华东, order_amount]
```

建议先单独写 runner，不要一次性改坏现有 `run_eval.py`。

### 第四步：再考虑持久化会话

如果内存型验证通过，再考虑：

- MySQL 表。
- Redis。
- 文件型测试 store。

不要第一步就上持久化。

## 7. 不建议的做法

1. 不要让多轮节点直接生成 SQL。  
   它应该改写 query，然后走现有 SQL 生成链路。

2. 不要让多轮节点直接判断指标和枚举是否合法。  
   这会和 `business_binding` 重复。

3. 不要把上一轮完整 SQL 直接拿来字符串替换。  
   例如把 `华北` 替换成 `华东` 看起来简单，但会绕过 RAG、binding、权限和安全校验。

4. 不要把所有历史轮次塞进 prompt。  
   当前已经做了性能治理，别重新引入长上下文成本。

5. 不要在 `pre_sql_execution_validation` 里继续加自然语言规则。  
   多轮问题应该在 rewrite / binding 层解决。

6. 不要把 conversation memory 纳入 LLM cache key 之外。  
   如果 LLM 节点使用会话上下文，cache key 必须包含稳定的 conversation context 或直接禁用缓存。

## 8. 需要重点看的代码

### Agent graph

- `app/agent/graph.py`

看节点注册和边：

- 当前新增节点应放在 `pre_rag_guard` 后、`extract_keywords` 前。
- 修改 graph 后必须更新测试。

### State

- `app/agent/state.py`

新增字段建议：

- `session_id`
- `turn_id`
- `rewritten_query`
- `conversation_state`
- `conversation_rewrite`
- `clarification_request`

注意 TypedDict 的字段需要和 eval trace / tests 对齐。

### Business binding

- `app/agent/nodes/business_binding.py`

不要拆它。下一阶段只让它消费改写后的 query，或明确传入 `effective_query`。

### Eval

- `app/evaluation/cases.py`
- `app/scripts/run_eval.py`
- `examples/eval_cases.yaml`

建议新增 conversation eval，而不是大改现有单轮 runner。

### LLM usage

- `app/agent/llm_usage.py`

新增 LLM 节点必须通过：

```python
ainvoke_llm_with_usage(...)
```

并明确 `cacheable`：

- 追问改写如果依赖 conversation context，第一版建议 `cacheable=False`。
- 如果要缓存，cache key 必须包含 conversation summary。

## 9. 建议新增测试

### 单元测试

建议新增：

- `tests/test_conversation_memory.py`
- `tests/test_followup_rewrite.py`
- `tests/test_conversation_eval_cases.py`

覆盖：

```text
华北销售额 -> 那华东呢
苹果品牌销售额 -> 那小米呢
2025年第一季度GMV -> 第二季度呢
按大区统计GMV -> 按品类呢
统计销售额 -> 这个月呢（如果当前日期上下文足够）
```

### 负例测试

必须覆盖：

```text
turn1: 统计华北地区销售额
turn2: 把所有用户手机号也列出来
expected: pre_rag_guard 或 pre_sql_execution_validation 阻断
```

```text
turn1: 统计华北地区销售额
turn2: 火星呢
expected: business_binding unresolved enum_value，不生成 SQL
```

```text
turn1: 统计 GMV
turn2: 品牌心智指数呢
expected: business_binding unresolved metric，不生成 SQL
```

## 10. 验收标准

第一阶段合格标准：

- 单轮 21 case 不回退。
- 新增 conversation suite 至少覆盖 8-10 条多轮 case。
- 追问改写 trace 可见：
  - 原始 query。
  - rewritten query。
  - 使用了哪些上一轮上下文。
  - 是否需要澄清。
- 多轮失败能定位：
  - rewrite 失败。
  - binding 失败。
  - RAG 失败。
  - SQL 生成失败。
  - SQL 校验失败。
  - SQL 执行失败。
- 不新增业务规则散落点。
- 不明显增加单轮 case 平均延迟。

建议命令：

```bash
uv run pytest -q
uv run ruff check .
uv run python -m app.scripts.run_eval --cases examples/eval_cases.yaml --output eval/runs/single-turn-regression.json
```

如果新增 conversation runner：

```bash
uv run python -m app.scripts.run_conversation_eval --cases examples/eval_conversation_cases.yaml --output eval/runs/conversation-regression.json
```

## 11. 当前已知风险

1. `correct_sql` 仍有长尾。  
   多轮改写如果导致 SQL 更容易错，会放大这个问题。

2. `filter_table` 不能再做语义裁列。  
   之前尝试过按 binding 裁列，导致 group-by 维度丢失。现在只做字段负载瘦身。

3. `run_sql` 已有 60 秒节点 timeout。  
   如果多轮新增更复杂 SQL，可能更容易触发 timeout。不要把 timeout 当 SQL 失败，要看 `exception_stage=tool_execution`。

4. 缓存是进程内缓存。  
   多轮内存 store 如果也做进程内，只适合第一阶段验证，不是多实例生产方案。

5. 时间表达当前支持有限。  
   `business_binding` 支持季度、月份、日期；更复杂的“去年同期”“环比上月”不建议第一阶段做。

## 12. 建议给下个对话的第一句话

可以直接把下面这段交给下一个对话：

```text
请先阅读 docs/multiturn-memory-handoff-2026-06-02.md、docs/business-binding-architecture.md 和 docs/five-optimizations-report-2026-06-02.md。

目标是实现“多轮问数记忆、追问改写和会话管理”的第一阶段，不要改 SQL/RAG/安全主架构，不要新增重复 guard。

第一阶段只做：
1. 进程内 conversation memory；
2. pre_rag_guard 后、extract_keywords 前的 followup rewrite 节点；
3. 8-10 条 conversation eval case；
4. 保证现有 21 个单轮 eval 不回退。

实现前先读 app/agent/graph.py、app/agent/state.py、app/agent/nodes/business_binding.py、app/scripts/run_eval.py。
```

## 13. 最后提醒

下一阶段的核心不是“让模型记住更多”，而是“把上一轮的少量结构化上下文安全地转成当前轮完整 query”。只要能做到这一点，现有 RAG、business_binding、SQL 生成、SQL 校验和 eval 体系都可以继续复用。
