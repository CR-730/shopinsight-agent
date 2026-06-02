# 电商问数 Agent 五项生产化优化报告

日期：2026-06-02

范围：本报告只总结当前仓库已经落地的五个优化方向，不包含“更复杂的多轮问数记忆、追问改写和会话管理”。

## 总览

本轮优化的核心目标，是把项目从学习型 Demo 推向更接近生产可维护形态。最终形成的主链路大致是：

```text
pre_rag_guard
  -> RAG recall
  -> merge_retrieved_info
  -> business_binding
  -> filter_metric / filter_table
  -> add_extra_context
  -> semantic_guard
  -> generate_sql
  -> pre_sql_execution_validation
  -> correct_sql 或 run_sql
```

五个方向的完成状态：

| 方向 | 当前状态 | 核心变化 |
|---|---|---|
| 后台自动轮询元数据知识库 | 已实现 | FastAPI lifespan 中启动后台 scheduler，检测元数据配置变化后自动 rebuild |
| 字段取值混合检索 | 已实现 | ES 全文检索 + Qdrant 向量检索 + 排名融合，支持别名和语义近似 |
| 异常重构后进入循环 | 已实现 | SQL 生成后进入执行前综合校验，repairable error 进入 correct_sql 循环 |
| 系统化评测集与自动化回归评测 | 已实现 | 21 个结构化 eval case、能力标签、失败归因、usage/cost/latency 报告 |
| 查询缓存、限流和性能治理 | 已实现 | 节点级观测、LLM/Embedding 缓存、模型路由、retry、熔断、限流、调用预算、SQL 执行 timeout |

## 1. 后台自动轮询是否更新元数据知识库

### 优化目标

原始学习项目通常需要手动执行元数据构建脚本。生产化后需要做到：

- 服务启动后可以按配置自动检查元数据配置是否变化。
- 配置变化后自动重建 Meta MySQL、Qdrant、ES 索引。
- 构建结果有版本边界，避免查询链路使用旧索引或旧缓存。
- 后台任务可关闭，不能在服务停止时遗留连接和任务。

### 当前实现

新增 `MetaKnowledgeScheduler` 作为轻量后台轮询器：

- 文件：`app/services/meta_knowledge_scheduler.py`
- 关键职责：
  - 对 `conf/meta_config.yaml` 做 sha256 文件签名。
  - `poll_once()` 比较当前签名和上次签名。
  - 文件变化时调用注入的 `build(config_path)`。
  - 使用 `_build_lock` 避免并发重复构建。
  - `start()` 创建后台 task，`stop()` 取消并等待退出。

服务生命周期中接入 scheduler：

- 文件：`app/api/lifespan.py`
- 配置入口：
  - `conf/app_config.yaml`
  - `app/conf/app_config.py`
- 配置项：

```yaml
metadata_build:
  enabled: false
  config_path: conf/meta_config.yaml
  poll_interval_seconds: 300
  build_on_start: false
```

元数据构建主服务：

- 文件：`app/services/meta_knowledge_service.py`
- 构建内容：
  - 表、字段、指标写入 Meta MySQL。
  - 字段向量索引写入 Qdrant。
  - 指标向量索引写入 Qdrant。
  - 字段取值全文索引写入 ES。
  - 字段取值向量索引写入 Qdrant。
  - 枚举别名写入 Meta MySQL 的 `value_alias`。
  - 成功构建后写入 `metadata_build` 版本。

手动构建入口仍保留：

- 文件：`app/scripts/build_meta_knowledge.py`
- 命令：

```bash
uv run python -m app.scripts.build_meta_knowledge -c conf/meta_config.yaml
```

### 版本一致性治理

元数据版本边界是这部分的关键，不只是“能重建索引”。

已落地：

- `MetaMySQLRepository.save_build_version()` 写入 `metadata_build`。
- `MetaMySQLRepository.get_active_build_version()` 查询当前 active build version。
- Qdrant payload 写入 `meta_build_version`。
- ES value index 也写入 `meta_build_version`。
- 查询侧按 `metadata_build_version` 过滤 Qdrant / ES 检索结果。
- LLM cache namespace 使用 `metadata_cache_version`，避免元数据变了仍命中旧 LLM 中间缓存。

相关文件：

- `app/repositories/mysql/meta/meta_mysql_repository.py`
- `app/repositories/qdrant/column_qdrant_repository.py`
- `app/repositories/qdrant/metric_qdrant_repository.py`
- `app/repositories/qdrant/value_qdrant_repository.py`
- `app/repositories/es/value_es_repository.py`
- `app/scripts/run_eval.py`
- `app/agent/llm_usage.py`

### 测试覆盖

相关测试：

- `tests/test_meta_knowledge_scheduler.py`
- `tests/test_meta_knowledge_service.py`
- `tests/test_metadata_boundaries.py`
- `tests/test_qdrant_version_filter.py`
- `tests/test_value_es_repository.py`
- `tests/test_meta_repository_cache.py`

### 踩坑和修正

1. 最初只记录了 Qdrant 的 `meta_build_version`，ES 字段取值索引没有版本过滤。  
   风险是 ES 重建失败或未来改成增量索引后，旧枚举值可能被召回。已修正为 ES mapping、bulk source、search 都带 `meta_build_version`。

2. 业务别名曾经直接从 `conf/meta_config.yaml` 运行时读取。  
   这会绕开 Meta MySQL / active build version / cache namespace。已改成构建时写入 `value_alias`，查询链路只读 Meta MySQL。

3. LLM cache namespace 最初只和配置文件 hash 相关。  
   直接改 Meta MySQL 时旧缓存仍可能命中。已加入 `metadata_cache_version`，由 active build version 和 Meta MySQL 内容共同决定。

## 2. 字段取值混合检索

### 优化目标

字段取值检索不再只依赖全文或只依赖向量。目标是处理：

- 用户表达和数据库真实取值不完全一致。
- 别称、同义表达、拼音或近似表达。
- ES 能命中精准文本，向量能补语义近似。
- 两路召回结果需要融合排序，而不是简单拼接。

### 当前实现

构建阶段：

- 字段真实取值由 `MetaKnowledgeService._build_value_infos()` 从 DW 读取。
- ES 索引用于全文检索：
  - `app/repositories/es/value_es_repository.py`
- Qdrant value collection 用于向量检索：
  - `app/repositories/qdrant/value_qdrant_repository.py`

查询阶段：

- 文件：`app/agent/nodes/recall_value.py`
- 流程：

```text
query / keywords
  -> LLM 扩展字段取值召回关键词
  -> normalize_keyword_list
  -> 每个 keyword 并发调用：
       ES value search
       Qdrant value vector search
  -> fuse_ranked_value_infos
  -> retrieved_value_infos
```

排序融合：

- 文件：`app/retrieval/fusion.py`
- 配置权重：

```yaml
agent:
  value_hybrid_es_weight: 1.2
  value_hybrid_vector_weight: 1.0
  value_vector_score_threshold: 0.65
```

工具调用在 eval trace 中体现为：

- `hybrid.value.search`
- `es.value.search`
- `qdrant.value.search`

### 版本和观测

混合检索同时接入了版本边界和性能观测：

- ES / Qdrant search 都传入 `metadata_build_version`。
- 向量 embedding 调用记录：
  - `model`
  - `tokens`
  - `latency_ms`
  - `cache_hit`
- ES 和 vector 各自的 latency / hit count 通过 stream writer 写入 retrieval debug。

相关文件：

- `app/agent/nodes/recall_value.py`
- `app/agent/nodes/recall_column.py`
- `app/agent/nodes/recall_metric.py`
- `app/agent/cost.py`

### 测试覆盖

相关测试：

- `tests/test_hybrid_retrieval.py`
- `tests/test_value_es_repository.py`
- `tests/test_qdrant_version_filter.py`
- `tests/test_eval_cases.py`
- `examples/eval_cases.yaml` 中多条 case 要求 `rag_value_hybrid_recall`

### 踩坑和修正

1. Docker / 外部依赖没启动时，RAG 全链路挂在连接阶段。  
   这导致最初 eval 只能证明 YAML 标签存在，不能证明真实能力被测到。后续完整验收必须启动 MySQL、Qdrant、ES。

2. 别名不能只做 YAML 规则词。  
   初版把“北方区域 -> 华北”放到 policy 或配置里，容易变成规则补丁。后续迁到 Meta MySQL `value_alias`，由构建链路统一管理。

3. RAG 漏召回会导致合法值误拦。  
   后续在 `business_binding` 中保留 `column_value_exists()` fallback，对别名 canonical value 做 DW 存在性确认。

4. ES 版本边界一开始缺失。  
   已补 `meta_build_version` mapping、写入和查询过滤。

## 3. 异常重构后再进入循环而不是直接提交

### 优化目标

原链路容易出现两类问题：

- SQL 生成后只要语法能过就直接执行，业务语义和安全行为不够稳。
- SQL 出错后直接失败或直接提交结果，没有清晰的修正循环。

目标链路调整为：

```text
generate_sql
  -> pre_sql_execution_validation
       repairable_error -> correct_sql -> pre_sql_execution_validation
       blocked          -> END
       pass             -> run_sql
```

### 当前实现

图路由：

- 文件：`app/agent/graph.py`
- 关键边：
  - `generate_sql -> pre_sql_execution_validation`
  - `pre_sql_execution_validation` 条件路由到 `run_sql` / `correct_sql` / blocked
  - `correct_sql -> pre_sql_execution_validation`

路由规则：

- 文件：`app/agent/sql_loop.py`
- 关键函数：
  - `route_after_pre_sql_execution_validation()`

执行前综合校验：

- 文件：`app/agent/nodes/pre_sql_execution_validation.py`
- 职责：
  - SQL normalize。
  - sqlglot 单 SELECT 解析。
  - 结构语义校验，尤其 join relationship check。
  - MySQL EXPLAIN / validate。
  - 确定性安全闸门。

修正节点：

- 文件：`app/agent/nodes/correct_sql.py`
- 使用独立的 `correct_sql_llm`。
- `cacheable=False`，避免缓存错误 SQL 修正结果。
- 增加无效修正检测：修正后 SQL 和原 SQL normalize 后相同，则停止无效循环。

失败节点：

- 文件：`app/agent/nodes/fail_sql_correction.py`

### 执行前校验能力

`pre_sql_execution_validation` 当前覆盖：

- 只允许单条 SELECT。
- 禁止危险 SQL 关键字。
- 禁止 `SELECT *`。
- 敏感字段检测。
- 明细查询检测。
- 未绑定枚举字面量检测。
- join 条件必须符合元数据关系。
- MySQL EXPLAIN 级语法和字段校验。

### 测试覆盖

相关测试：

- `tests/test_sql_loop.py`
- `tests/test_guard_layers.py`
- `tests/test_llm_thinking_config.py`
- `tests/test_run_sql.py`

### 踩坑和修正

1. 最初只检查 SQL 是否可执行，导致“业务不合法但 SQL 能返回数据”。  
   例如未知指标、未知区域、prompt injection 明细查询。后续拆成输入安全、业务绑定、执行前硬校验。

2. join 错误频率高。  
   模型会生成 `fact_order.region_id = dim_region.region_name` 这类错误。已在 `pre_sql_execution_validation` 中加入 join relationship check，作为 `repairable_error` 进入 `correct_sql`。

3. 敏感字段 JOIN 豁免过宽。  
   初版跳过了所有 JOIN ON 里的敏感字段，导致 `phone = phone` 这类条件可能放行。后续收紧为只有合法 key / 元数据关系通过的 join key 才能豁免。

4. `correct_sql` 长尾明显。  
   eval 中曾出现单次约 98.9s 的 correct_sql。当前已保留观测，未在本轮深入优化，因为触发频率低，且修正任务仍需要相对强模型。

5. 外层 case timeout 会误导报告。  
   `run_sql` 长尾时，eval 曾显示成 `missing_sql / rag_recall / sql_validation`。已给 `run_sql` 加 `sql_execution_timeout_seconds`，并让 eval timeout 归因更准确。

## 4. 系统化评测集与自动化回归评测

### 优化目标

评测不再只是几个简单样例，而要能回答：

- 覆盖了哪些能力？
- 每个 case 的风险点是什么？
- 失败属于召回、过滤、SQL 生成、SQL 校验、安全还是执行？
- 是否记录 token、成本、延迟？
- 是否能在 CI 或命令行中失败即 exit 1？
- 历史报告能否保留用于对比？

### 当前实现

评测集：

- 文件：`examples/eval_cases.yaml`
- 当前 21 个 case。
- 覆盖 suite：
  - `smoke`
  - `regression`
  - `adversarial`
  - `realistic`
- 每个 case 支持字段：
  - `id`
  - `query`
  - `business_source`
  - `suite`
  - `difficulty`
  - `capabilities`
  - `tags`
  - `risk_points`
  - `expected_sql_contains`
  - `expected_columns`
  - `expected_metrics`
  - `expected_time_binding`
  - `expected_unresolved_binding`
  - `expected_result`
  - `expected_blocked_by`
  - `forbidden_sql`
  - `must_call_tools`
  - `forbidden_behavior`
  - `fatal_errors`
  - `timeout_seconds`

评测 runner：

- 文件：`app/scripts/run_eval.py`
- 命令：

```bash
uv run python -m app.scripts.run_eval --cases examples/eval_cases.yaml --output eval/runs/latest.json
```

失败即退出：

- `run_eval()` 返回 `0 if passed == len(results) else 1`。
- 可直接用于 CI 阻断。

评测逻辑：

- 文件：`app/evaluation/cases.py`
- 能力：
  - 加载 YAML case。
  - 构造 trace。
  - 规则化评测 SQL、上下文、工具调用、结果、阻断节点。
  - 按 failure stage 归因。

报告内容：

- `summary`
- `usage`
- `cost`
- `total_latency_seconds`
- `capability_summary`
- `scenario_summary`
- 每个 case 的 trace、usage、latency、failure list

### 当前评测质量

已经覆盖的能力包括：

- keyword extraction
- RAG column recall
- RAG metric recall
- hybrid value recall
- context filter
- SQL generation
- SQL validation
- SQL correction loop
- tool execution
- safety

case 质量优化重点：

- 删除“为了 benchmark 而刁钻”的倾向。
- 每条 case 绑定真实业务来源。
- 增加负例和边界场景：
  - prompt injection
  - unknown metric
  - unknown enum value
  - sensitive detail query
- 增加可验证结构化期望：
  - `expected_time_binding`
  - `expected_unresolved_binding`
  - `must_call_tools`

### 相关文档

- `docs/business-binding-architecture.md`
- `docs/eval-performance-report-2026-06-02.md`

### 测试覆盖

相关测试：

- `tests/test_eval_cases.py`
- `tests/test_eval_cases_content.py`
- `tests/test_eval_case_quality.py`

### 踩坑和修正

1. 只看 YAML capabilities 标签不可信。  
   早期因为 Docker 没启动，所有 case 挂在 RAG 连接阶段，标签不能证明能力被真实测到。后续必须跑通真实依赖后验收。

2. eval 一开始只看当前输出，定位能力不足。  
   已扩展 trace、usage、capability summary、scenario summary。

3. case 不能强绑旧实现细节。  
   例如季度 case 原先要求 `dim_date.year`，但后来业务绑定统一用 `date_id BETWEEN`。已改成检查 `expected_time_binding`。

4. 失败归因曾经失真。  
   外层 timeout 后空 state 会产生一堆 `missing_sql`、`missing_column` 假失败。已调整为 timeout 只报真实 exception stage。

5. 评测集不是隐藏集，也不是防刷题体系。  
   当前是项目内回归集，能支撑开发回归；还没有训练集 / 回归集 / 隐藏集分层。

## 5. 查询缓存、限流和性能治理

### 优化目标

性能治理的目标不是牺牲安全，而是让链路可观测、可降本、可避免服务不稳定时拖垮整条 agent。

本轮治理顺序：

1. 先补观测。
2. 再做缓存和模型路由。
3. 再做 retry、限流、熔断和调用预算。
4. 保留关键安全闸门。

### 节点级观测

节点级 latency：

- 文件：`app/agent/node_observer.py`
- `graph.py` 中所有 LangGraph node 通过 `traced_node()` 包装。
- 每个节点写入 `CostTracker.calls`：
  - `type=node`
  - `step`
  - `latency_ms`
  - `error_type`

LLM usage / latency / cost：

- 文件：`app/agent/cost.py`
- 文件：`app/agent/llm_usage.py`
- 记录字段：
  - `model`
  - `input_tokens`
  - `output_tokens`
  - `total_tokens`
  - `cached_tokens`
  - `cost`
  - `latency_ms`
  - `cache_hit`
  - `retry_count`
  - `breaker_state`
  - `retry_after_ms`
  - `throttle_wait_ms`
  - `final_error_type`

Embedding usage / latency：

- 文件：
  - `app/agent/nodes/recall_column.py`
  - `app/agent/nodes/recall_metric.py`
  - `app/agent/nodes/recall_value.py`
- 记录：
  - `model`
  - `tokens`
  - `latency_ms`
  - `cache_hit`

### 缓存

Embedding 缓存：

- 文件：`app/agent/cached_clients.py`
- `CachedEmbeddingClient` 提供进程内 LRU：
  - query embedding cache
  - documents embedding cache
  - `last_cache_hit`

LLM response cache：

- 文件：`app/agent/llm_usage.py`
- 进程内 LRU，最多 512 条。
- key 包含：
  - model
  - step
  - prompt text
  - runtime cache namespace
  - metadata cache namespace
- 解析成功后才写缓存，避免坏 JSON / 坏结构化输出被缓存。
- `generate_sql` 和 `correct_sql` 设置 `cacheable=False`，避免缓存 SQL 生成和修正结果。

元数据仓储缓存：

- 文件：
  - `app/repositories/mysql/meta/meta_mysql_repository.py`
  - `app/repositories/mysql/dw/dw_mysql_repository.py`
- 缓存：
  - metric infos
  - column infos
  - value aliases
  - DB info
  - column value exists

缓存一致性：

- `MetaKnowledgeService.build()` 完成后清理 LLM response cache。
- `run_eval.py` 为每个请求设置 `metadata:{metadata_cache_version}` namespace。
- Meta MySQL 内容变化会影响 `metadata_cache_version`。

### 模型路由和 thinking 配置

模型入口：

- 文件：`app/agent/llm.py`
- 配置只从 `.env` 读取：
  - `LLM_MODEL`
  - `LLM_FAST_MODEL`
  - `LLM_GENERATE_SQL_ENABLE_THINKING`
  - `LLM_CORRECT_SQL_ENABLE_THINKING`
  - `LLM_STRUCTURED_ENABLE_THINKING`

策略：

- 普通结构化抽取 / 分类 / 过滤使用 fast model。
- `generate_sql` 使用主模型，但关闭 thinking。
- `correct_sql` 使用主模型，并保留 thinking。

原因：

- SQL 生成是高频主链路，thinking 打开会明显增加 latency 和 token。
- SQL 修正是低频兜底，允许更强推理以提高修正质量。

相关测试：

- `tests/test_llm_thinking_config.py`
- `tests/test_app_config.py`

### retry、限流、熔断、调用预算

统一入口：

- 文件：`app/agent/llm_usage.py`
- 核心函数：
  - `ainvoke_llm_with_usage()`
  - `invoke_llm_with_policy()`

已实现：

- model-level concurrency semaphore。
- provider + base_url + model 作为熔断 key，避免同名模型不同供应商互相影响。
- 可重试错误：
  - timeout
  - connection
  - 429
  - 500 / 502 / 503 / 504
- retry 使用 exponential backoff + jitter。
- 尊重 `Retry-After`。
- quota / 403 类错误触发熔断。
- 连续 429 达阈值触发熔断。
- half-open 探测。
- 滑动窗口错误率熔断。
- 失败按一次业务调用记录窗口结果，而不是每次 retry attempt 都记一次。
- 每请求 LLM 调用预算，防异常循环。

配置：

- 文件：`app/conf/app_config.py`
- `.env` 可配置项包括：
  - `LLM_MAX_RETRIES`
  - `LLM_RETRY_BACKOFF_SECONDS`
  - `LLM_CONCURRENCY_LIMIT`
  - `LLM_QUOTA_CIRCUIT_BREAKER_SECONDS`
  - `LLM_RATE_LIMIT_BREAKER_THRESHOLD`
  - `LLM_ERROR_WINDOW_SECONDS`
  - `LLM_ERROR_WINDOW_MIN_CALLS`
  - `LLM_ERROR_RATE_THRESHOLD`
  - `LLM_MAX_CALLS_PER_REQUEST`
  - `LLM_FAST_MAX_RETRIES`
  - `LLM_FAST_CONCURRENCY_LIMIT`
  - `LLM_FAST_QUOTA_CIRCUIT_BREAKER_SECONDS`
  - `LLM_SQL_MAX_RETRIES`
  - `LLM_SQL_CONCURRENCY_LIMIT`
  - `LLM_SQL_QUOTA_CIRCUIT_BREAKER_SECONDS`

### SQL 执行 timeout

新增：

- 文件：`app/agent/nodes/run_sql.py`
- 配置：

```yaml
agent:
  sql_execution_timeout_seconds: 60
```

作用：

- DW SQL 执行超过节点 timeout 时，返回：
  - `exception_stage=tool_execution`
  - `error=SQL 执行超时：60 秒`
- 避免外层 case timeout 后误报为 `missing_sql`、`rag_recall` 或 `sql_validation`。

### filter_table 上下文优化

文件：

- `app/agent/nodes/filter_table.py`
- `tests/test_filter_table_context.py`

最终策略：

- 不裁掉候选表和候选列。
- 只去掉 description、examples 等重字段。
- 保留：
  - table name
  - table role
  - column name
  - column role
  - column alias

实际效果：

- `filter_table` 平均输入 tokens 从 `1771.85` 降到 `809.4`。
- `filter_table` 总 tokens 从 `36194` 降到 `16791`。

### 测试覆盖

相关测试：

- `tests/test_cost_tracking.py`
- `tests/test_llm_usage.py`
- `tests/test_cached_clients.py`
- `tests/test_filter_table_context.py`
- `tests/test_run_sql.py`
- `tests/test_app_config.py`

### 踩坑和修正

1. LLM cache 最初在解析前写入。  
   如果 JSON / Pydantic parser 失败，坏响应会进入缓存。已改成解析成功后再 `_store_cache()`。

2. cache key 最初缺少元数据版本边界。  
   已加入 runtime namespace 和 metadata cache version。

3. `_runtime_cache_namespace()` 每次读配置算 hash 有开销。  
   已做进程内 memo，并在 `clear_llm_response_cache()` 时清掉。

4. retry_count 字段一开始只是观测字段，没有真实 retry。  
   后续补了统一 retry / breaker 逻辑。

5. 熔断 key 最初只按 model。  
   已改为 provider + base_url + model。

6. retry backoff 最初不够标准。  
   已改为指数退避 + jitter，并尊重 `Retry-After`。

7. 滑动窗口最初按每次 attempt 计失败。  
   已改成一次业务调用最终失败才进入窗口，避免过快打开 breaker。

8. `filter_table` 优化第一次做成按 `business_binding` 裁列。  
   eval 发现会误删 group-by 维度和未绑定取值上下文，导致 SQL 质量回退。最终改成只压缩字段负载，不做语义裁剪。

9. 完整 eval 中偶发 `run_sql` 180s 超时，报告却显示 RAG / SQL 缺失。  
   已增加 SQL 执行节点内 timeout 和 eval timeout 降噪。

## 业务语义收敛：避免五项优化互相打架

虽然这不是你最初五项中的独立方向，但它是后续修复过程中形成的关键架构收敛。

问题背景：

最初安全和语义判断分散在多个节点：

- `business_binding`
- `filter_metric`
- `semantic_guard`
- `generate_sql.prompt`
- `pre_sql_execution_validation`

这导致同一类业务判断被重复裁决，复杂度上升，效果反而不稳。

当前收敛方式：

- 文件：`app/agent/nodes/business_binding.py`
- 文档：`docs/business-binding-architecture.md`

职责：

```yaml
business_binding:
  metrics: []
  filters: []
  time: null
  unresolved: []
  ambiguous: []
```

后续节点原则：

- `filter_metric` 不再调用 LLM，只按 binding 裁剪；没有 binding 时保留 RAG top-k。
- `semantic_guard` 只做 binding 完整性检查。
- `generate_sql` 消费结构化 binding。
- `pre_sql_execution_validation` 只做执行前硬安全和结构兜底，不再做自然语言业务解析。

这个收敛是让五项优化可维护的关键，否则缓存、评测、安全和 SQL 修正都会互相影响。

## 当前验收结果

最近一次本地验证：

```text
uv run pytest -q
103 passed

uv run ruff check .
All checks passed
```

已记录的 eval 结果：

- `business-binding-final-20260602-122348.json`
  - `21 passed / 0 failed`
  - `pass_rate = 1.0`
- 后续性能实验中完整 eval 出现过 `20/21`，唯一失败是 `sql_time_range_quarter_region` 外层 180s timeout。
  - 定向复跑该 case 通过，用时约 6-7 秒。
  - 已补 `run_sql` timeout 和 eval 归因降噪。

## 当前剩余风险

1. `correct_sql` 仍有长尾风险。  
   已有观测，但还没有做专门 timeout / prompt 精简 / 特定错误类型 thinking 路由。

2. eval 仍是公开回归集。  
   当前适合项目内回归，不是防刷题 benchmark。还没有隐藏集、语义变体集、线上样本回流。

3. `business_binding.time.required_columns` 当前固定为 `fact_order.date_id`。  
   当前项目只有一个核心事实表，因此合理；未来多事实表需要从表上下文推导时间键。

4. 缓存是进程内缓存。  
   当前适合单进程开发和轻量服务；多实例部署时需要引入外部缓存或接受实例级缓存不一致。

5. SQL 执行 timeout 只是观测和保护。  
   它能避免诊断失真，但没有解决底层 MySQL 偶发长尾根因。后续需要结合连接池、查询计划、慢查询日志继续治理。

## 结论

五个方向已经从“功能想法”落成了当前仓库里的具体机制：

- 元数据知识库有后台轮询、构建版本和缓存一致性边界。
- 字段取值有 ES + Qdrant 混合召回和版本过滤。
- SQL 异常进入执行前综合校验和修正循环，不再直接执行。
- eval 已从样例集合升级为带能力矩阵、风险点、工具调用、成本和延迟记录的回归评测。
- 性能治理已补齐节点观测、LLM/Embedding 缓存、模型路由、retry、限流、熔断、调用预算和 SQL 执行 timeout。

当前更适合进入下一阶段前的状态整理和针对性风险修复，而不是继续堆新的 guard 或规则。
