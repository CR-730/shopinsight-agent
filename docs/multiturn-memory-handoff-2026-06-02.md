# 当前仓库形态交接文档

日期：2026-06-02

用途：给下一个对话快速了解当前电商问数 Agent 仓库形态。本文只描述当前已经存在的代码、配置、文档、评测和已知风险，不包含下一阶段实现方案。

## 1. 仓库当前状态

当前分支：`master`

最近提交：

```text
25de7c3 docs: add multiturn memory handoff
7f1428a docs: summarize five production optimizations
36f82c1 fix: report sql execution timeouts accurately
94f8b50 feat: observe embedding latency and compact table context
0dd9f71 docs: document binding architecture and eval performance
a5e1e28 refactor: consolidate business binding decisions
51c65fc feat: version metadata aliases and retrieval indexes
00c9a3d feat: add llm resilience and usage governance
3cfccf1 chore: initialize shopkeeper agent
```

最近验证：

```text
uv run pytest -q
103 passed

uv run ruff check .
All checks passed
```

关键文档：

- `docs/five-optimizations-report-2026-06-02.md`
- `docs/business-binding-architecture.md`
- `docs/eval-performance-report-2026-06-02.md`
- `docs/multiturn-memory-handoff-2026-06-02.md`

## 2. 当前主链路

当前 LangGraph 主链路定义在：

- `app/agent/graph.py`

链路形态：

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

主要节点文件：

- `app/agent/nodes/pre_rag_guard.py`
- `app/agent/nodes/extract_keywords.py`
- `app/agent/nodes/recall_column.py`
- `app/agent/nodes/recall_value.py`
- `app/agent/nodes/recall_metric.py`
- `app/agent/nodes/merge_retrieved_info.py`
- `app/agent/nodes/business_binding.py`
- `app/agent/nodes/filter_metric.py`
- `app/agent/nodes/filter_table.py`
- `app/agent/nodes/add_extra_context.py`
- `app/agent/nodes/semantic_guard.py`
- `app/agent/nodes/generate_sql.py`
- `app/agent/nodes/pre_sql_execution_validation.py`
- `app/agent/nodes/correct_sql.py`
- `app/agent/nodes/fail_sql_correction.py`
- `app/agent/nodes/run_sql.py`

## 3. 当前 State 和 Context

State 定义：

- `app/agent/state.py`

Context 定义：

- `app/agent/context.py`

当前 context 中包含：

- Qdrant repositories
- ES value repository
- Meta MySQL repository
- DW MySQL repository
- embedding client
- cost tracker
- `metadata_build_version`
- `metadata_cache_version`

当前没有完整的会话记忆模型，也没有持久化 conversation memory。

## 4. 业务语义裁决层

当前业务语义裁决集中在：

- `app/agent/nodes/business_binding.py`

输出结构：

```yaml
business_binding:
  metrics: []
  filters: []
  time: null
  unresolved: []
  ambiguous: []
```

同步写入 state 的字段包括：

- `business_binding`
- `metric_bindings`
- `resolved_filters`
- `time_binding`
- `validated_enum_values`
- `unresolved_bindings`
- `ambiguous_bindings`

当前支持的 binding 类型：

- 指标绑定：基于已召回指标的 name / alias。
- 枚举过滤绑定：基于字段 alias、retrieved value、value alias、DW 存在性确认。
- 时间绑定：支持季度、月份、日期。
- unresolved binding：用于未知指标、未知枚举值等。

相关文档：

- `docs/business-binding-architecture.md`

## 5. RAG 与混合检索

字段召回：

- `app/agent/nodes/recall_column.py`
- `app/repositories/qdrant/column_qdrant_repository.py`

指标召回：

- `app/agent/nodes/recall_metric.py`
- `app/repositories/qdrant/metric_qdrant_repository.py`

字段取值混合召回：

- `app/agent/nodes/recall_value.py`
- `app/repositories/es/value_es_repository.py`
- `app/repositories/qdrant/value_qdrant_repository.py`
- `app/retrieval/fusion.py`

字段取值召回当前是：

```text
ES 全文检索 + Qdrant 向量检索 + 排名融合
```

相关配置：

- `conf/app_config.yaml`

```yaml
agent:
  value_hybrid_es_weight: 1.2
  value_hybrid_vector_weight: 1.0
  value_vector_score_threshold: 0.65
```

## 6. 元数据知识库构建

手动构建入口：

- `app/scripts/build_meta_knowledge.py`

后台轮询：

- `app/services/meta_knowledge_scheduler.py`
- `app/api/lifespan.py`

构建服务：

- `app/services/meta_knowledge_service.py`

Meta MySQL repository：

- `app/repositories/mysql/meta/meta_mysql_repository.py`

配置：

- `conf/app_config.yaml`

```yaml
metadata_build:
  enabled: false
  config_path: conf/meta_config.yaml
  poll_interval_seconds: 300
  build_on_start: false
```

元数据配置：

- `conf/meta_config.yaml`

当前构建产物：

- Meta MySQL：表、字段、指标、字段-指标关系、value alias、metadata build version。
- Qdrant：字段向量、指标向量、字段取值向量。
- Elasticsearch：字段取值全文索引。

版本边界：

- `metadata_build.version`
- Qdrant payload `meta_build_version`
- ES source `meta_build_version`
- 查询侧传入 `metadata_build_version`
- LLM cache namespace 使用 `metadata_cache_version`

## 7. SQL 生成、校验和修正

SQL 生成：

- `app/agent/nodes/generate_sql.py`

SQL 修正：

- `app/agent/nodes/correct_sql.py`

SQL loop 路由：

- `app/agent/sql_loop.py`

执行前综合校验：

- `app/agent/nodes/pre_sql_execution_validation.py`

执行节点：

- `app/agent/nodes/run_sql.py`

当前 SQL 链路：

```text
generate_sql
  -> pre_sql_execution_validation
       pass             -> run_sql
       repairable_error -> correct_sql -> pre_sql_execution_validation
       blocked          -> END
```

`pre_sql_execution_validation` 当前覆盖：

- SQL normalize
- 单 SELECT 解析
- 危险 SQL 关键字
- `SELECT *`
- 敏感字段
- 明细查询
- 未绑定枚举字面量
- join relationship
- MySQL explain

`run_sql` 当前有 SQL 执行 timeout：

- 配置项：`agent.sql_execution_timeout_seconds`
- 默认值：`60`

## 8. 安全与 Guardrail

RAG 前输入安全：

- `app/agent/nodes/pre_rag_guard.py`

业务完整性检查：

- `app/agent/nodes/semantic_guard.py`

SQL 执行前硬校验：

- `app/agent/nodes/pre_sql_execution_validation.py`

策略配置：

- `conf/policy_config.yaml`
- `app/conf/policy_config.py`

当前 `semantic_guard` 主要消费：

- `unresolved_bindings`
- `ambiguous_bindings`

## 9. LLM、模型路由和 Thinking 配置

LLM 初始化：

- `app/agent/llm.py`

LLM 调用治理：

- `app/agent/llm_usage.py`

配置来自 `.env`：

- `LLM_MODEL`
- `LLM_FAST_MODEL`
- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_PROVIDER`
- `LLM_TIMEOUT_SECONDS`
- `LLM_STRUCTURED_ENABLE_THINKING`
- `LLM_GENERATE_SQL_ENABLE_THINKING`
- `LLM_CORRECT_SQL_ENABLE_THINKING`
- `LLM_INPUT_PER_1M_TOKENS`
- `LLM_OUTPUT_PER_1M_TOKENS`

当前模型策略：

- 普通结构化节点使用 fast model。
- `generate_sql` 使用主模型，当前 thinking 可单独关闭。
- `correct_sql` 使用主模型，当前 thinking 可单独开启。

相关测试：

- `tests/test_llm_thinking_config.py`
- `tests/test_app_config.py`

## 10. 缓存、限流和性能观测

节点级观测：

- `app/agent/node_observer.py`
- `app/agent/cost.py`

LLM usage：

- `app/agent/llm_usage.py`

Embedding cache：

- `app/agent/cached_clients.py`
- `app/clients/embedding_client_manager.py`

当前观测字段包括：

- node latency
- LLM latency
- LLM tokens
- LLM cost
- cached tokens
- cache hit
- retry count
- breaker state
- retry after
- throttle wait
- final error type
- embedding latency
- embedding cache hit

当前 LLM 治理能力包括：

- response cache
- cache namespace
- retry
- exponential backoff + jitter
- Retry-After
- model-level concurrency
- circuit breaker
- half-open
- consecutive 429 breaker
- sliding window error-rate breaker
- per-request LLM call budget

## 11. 评测体系

Eval case：

- `examples/eval_cases.yaml`

Eval runner：

- `app/scripts/run_eval.py`

Eval scoring：

- `app/evaluation/cases.py`

当前 eval case 数量：

- 21

当前 suite：

- `smoke`
- `regression`
- `adversarial`
- `realistic`

当前 capability 标签包括：

- `keyword_extraction`
- `rag_column_recall`
- `rag_metric_recall`
- `rag_value_hybrid_recall`
- `context_filter`
- `sql_generation`
- `sql_validation`
- `sql_correction_loop`
- `tool_execution`
- `safety`

当前 `run_eval` 行为：

- 全部通过返回 exit 0。
- 任一失败返回 exit 1。
- 输出 JSON 报告，包含 summary、usage、cost、latency、capability summary、scenario summary、case trace。

## 12. 当前测试文件

主要测试：

- `tests/test_app_config.py`
- `tests/test_business_binding.py`
- `tests/test_cached_clients.py`
- `tests/test_cost_tracking.py`
- `tests/test_eval_case_quality.py`
- `tests/test_eval_cases.py`
- `tests/test_eval_cases_content.py`
- `tests/test_filter_table_context.py`
- `tests/test_guard_layers.py`
- `tests/test_hybrid_retrieval.py`
- `tests/test_llm_thinking_config.py`
- `tests/test_llm_usage.py`
- `tests/test_meta_knowledge_scheduler.py`
- `tests/test_meta_knowledge_service.py`
- `tests/test_meta_point_ids.py`
- `tests/test_meta_repository_cache.py`
- `tests/test_metadata_boundaries.py`
- `tests/test_qdrant_version_filter.py`
- `tests/test_run_sql.py`
- `tests/test_sql_loop.py`
- `tests/test_value_es_repository.py`

## 13. 当前配置文件

应用配置：

- `conf/app_config.yaml`

元数据配置：

- `conf/meta_config.yaml`

安全策略配置：

- `conf/policy_config.yaml`

环境变量：

- `.env`

## 14. 当前外部依赖

当前完整 eval 依赖：

- MySQL
- Qdrant
- Elasticsearch
- DashScope / OpenAI-compatible LLM API
- DashScope embedding model：当前 `.env` 使用 `EMBEDDING_MODEL=text-embedding-v2`

如果 Docker 或外部服务未启动，RAG 和 eval 会在连接阶段失败。

## 15. 当前已知风险

1. `correct_sql` 仍可能出现长尾。历史 eval 中出现过单次接近 98.9s。

2. `run_sql` 曾出现完整 eval 中偶发 180s 外层 timeout。当前已加节点内 SQL execution timeout 和 eval 归因降噪。

3. `filter_table` 当前只做上下文负载瘦身，不做语义裁列。之前尝试语义裁列会丢失 group-by 维度上下文。

4. 当前 eval 是项目内公开回归集，不是隐藏 benchmark。

5. 当前缓存主要是进程内缓存，多实例一致性没有实现。

6. 当前没有完整的多轮会话状态、追问改写节点、会话持久化或 conversation eval suite。

7. 当前时间 binding 支持有限，主要是季度、月份和日期。

8. 当前 `business_binding.time.required_columns` 以 `fact_order.date_id` 为核心事实表时间键。

## 16. 最近可用验证命令

```bash
uv run pytest -q
uv run ruff check .
uv run python -m app.scripts.run_eval --cases examples/eval_cases.yaml --output eval/runs/latest.json
```

## 17. 当前交接结论

当前仓库已经具备：

- 元数据构建和版本边界。
- 字段、指标、字段取值召回。
- 字段取值混合检索。
- 业务绑定层。
- SQL 生成、执行前校验、修正循环、执行节点。
- 安全 guardrail。
- usage / cost / latency 观测。
- LLM cache、embedding cache、限流、retry、熔断、调用预算。
- 单轮系统化 eval。

当前仓库尚未具备：

- 多轮会话记忆。
- 追问改写。
- 会话级状态管理。
- 会话持久化。
- conversation eval runner / conversation eval cases。
