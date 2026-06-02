# 多轮问数记忆设计

日期：2026-06-02

## 目标

为当前电商问数 Agent 增加轻量的持久化会话能力，支持多轮问数中的追问改写、会话状态管理和记忆写回，同时保持现有 RAG、业务绑定、SQL 校验和执行链路不被重写。

## 调研依据

本方案基于以下公开资料和论文收敛：

- LangGraph Persistence 文档：https://docs.langchain.com/oss/python/langgraph/persistence
- Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers：https://arxiv.org/abs/2603.07670
- LLM Agent Memory: A Survey from a Unified Representation--Management Perspective：https://openreview.net/forum?id=KPs1EgGKcT
- Memory Matters: The Need to Improve Long-Term Memory in LLM-Agents：https://ojs.aaai.org/index.php/AAAI-SS/article/view/27688
- How Memory Management Impacts LLM Agents：https://arxiv.org/abs/2505.16067
- Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions：https://arxiv.org/abs/2507.05257
- SParC: Cross-Domain Semantic Parsing in Context：https://arxiv.org/abs/1906.02285
- CoSQL: A Conversational Text-to-SQL Challenge：https://arxiv.org/abs/1909.05378
- QURG: Question Rewriting Guided Context-Dependent Text-to-SQL Semantic Parsing：https://arxiv.org/abs/2305.06655
- CQR-SQL: Conversational Question Reformulation Enhanced Context-Dependent Text-to-SQL Parsers：https://aclanthology.org/2022.findings-emnlp.150/
- DIR: A Large-Scale Dialogue Rewrite Dataset for Cross-Domain Conversational Text-to-SQL：https://www.mdpi.com/2076-3417/13/4/2262

这些资料的共同结论是：多轮 Text-to-SQL 的关键不是把所有历史塞进上下文，而是显式处理指代、省略和用户焦点变化；Agent 记忆应拆成写入、管理、读取闭环，并对写入质量进行约束，避免错误经验传播。

## 架构选型

第一阶段采用业务表持久化，不直接把 LangGraph checkpointer 作为产品记忆核心。

原因：

- LangGraph checkpoint 更适合图状态恢复、线程断点和中断续跑。
- 当前产品需要的是用户可见的会话历史、追问改写依据和业务绑定快照。
- 问数场景的记忆对象是结构化业务状态，不是开放域长期偏好。
- 项目已经有 Meta MySQL，增加会话表比引入新的向量记忆存储更稳。

## 记忆分层

第一阶段只实现三层：

- 会话元信息：conversation id、标题、创建和更新时间。
- 轮次历史：原始问题、改写问题、SQL、结果摘要、安全错误。
- 会话快照：上一轮指标绑定、过滤绑定、时间绑定、SQL、结果摘要和最近轮次摘要。

暂不实现：

- 跨会话用户画像。
- 自动长期记忆反思。
- 会话向量检索。
- 记忆删除策略以外的复杂遗忘学习。

## 数据流

```text
API: query + conversation_id?
  -> 创建或读取 conversation
  -> 读取 conversation_snapshot
  -> structured rewrite_query
  -> 原有 LangGraph 单轮链路
  -> 收集最终 state
  -> 保存 conversation_turn
  -> 更新 conversation_snapshot
  -> SSE 返回 conversation_id、rewritten_query、进度、结果、usage
```

## 追问改写边界

追问改写只负责把当前问题补全成可独立理解的问题，不直接绑定指标、枚举值或生成 SQL。
改写层采用 fast LLM 结构化输出，遵循 DIR / CQR-SQL / QURG 的独立问句 reformulation 思路，不使用业务词规则拼接。

规则：

- 如果当前问题已经完整，返回 `mode=unchanged`，`standalone_query` 保持原文。
- 如果当前问题依赖上下文且快照为空，返回 `mode=needs_context`，服务层直接阻断，不进入 SQL 链路。
- 如果当前问题依赖上下文且快照可用，返回 `mode=rewritten` 和可独立理解的 `standalone_query`。
- 补全来源只使用最近快照中的指标、过滤条件、时间条件和上一轮结果摘要。
- `inherited_slots` 和 `overridden_slots` 仅用于观测，不作为业务绑定或快照写回依据。
- 改写后的问题仍必须重新走 RAG、business_binding、semantic_guard 和 SQL validation。

## 数据表

```text
conversation
- id varchar(64) primary key
- user_id varchar(128) null
- title varchar(255)
- created_at timestamp
- updated_at timestamp
- archived_at timestamp null

conversation_turn
- id varchar(64) primary key
- conversation_id varchar(64)
- turn_index int
- user_query text
- rewritten_query text
- sql_text text null
- final_answer_summary text null
- safety_error text null
- blocked_by varchar(64) null
- created_at timestamp

conversation_snapshot
- conversation_id varchar(64) primary key
- last_metric_bindings json
- last_resolved_filters json
- last_time_binding json
- last_sql text null
- last_answer_summary text null
- recent_turns_summary json
- updated_at timestamp
```

## 接口变化

`POST /api/query` 请求体增加可选字段：

```json
{
  "query": "那上个月呢",
  "conversation_id": "optional",
  "user_id": "optional"
}
```

返回 SSE 增加会话事件：

```json
{
  "type": "conversation",
  "conversation_id": "...",
  "rewritten_query": "..."
}
```

## 评测

增加 conversation eval case，覆盖：

- 上一轮完整问题，下一轮省略指标。
- 上一轮有地区过滤，下一轮只换时间。
- 上一轮按维度分组，下一轮换指标。
- 被 semantic_guard 拦截的轮次不污染有效快照。

第一阶段优先做单元测试和轻量脚本，不要求马上跑完整外部依赖 eval。

## 风险控制

- 只保存结构化快照和短摘要，不保存大量表结构上下文。
- 快照只来自本轮最终 state，避免从模型文本中反推业务状态。
- blocked 轮次保存历史但不覆盖成功快照。
- 追问改写不跳过任何现有安全闸门。
