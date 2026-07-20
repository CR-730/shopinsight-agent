# ShopInsight 语义规划层重构设计

> **当前实现差异（2026-07-20）：** 枚举筛选已收紧为严格值候选绑定。`EnumPredicateMention` 不再包含字段候选 ID；字段由 `ValueCandidate.column_id` 唯一推导。缺少值候选时返回 `value_not_bound` 并阻断，不再执行 DW MySQL 发现式兜底。DW 精确查询仅用于验证 Meta 别名候选的规范值。

## 1. 背景与结论

当前主链为：

```text
context_builder
  -> business_binding
  -> context_compaction
  -> generate_sql
```

现有 `business_binding` 已具备受控候选 ID、字段值所属列、DW 精确值兜底和阻断澄清等基础，但它仍是指标、筛选、分组、投影和时间五类独立槽位的绑定器，不是完整查询计划。它没有结构化表示操作符、排序、Top-N、HAVING、多时间条件和 JOIN 计划；指标元数据也没有权威聚合公式。`generate_sql` 仍读取原始问题、历史消息和 SQL few-shot，因此可以绕过绑定结果重新解释查询。

本次将节点改名为 `semantic_planning`，并把它升级成查询语义的唯一裁决层：

> 大模型负责需要语义判断的部分；能够由规则、类型、权威元数据或数据库事实确定的部分，由后端完成规范化、反查或校验。

这里的“后端负责确定性部分”不等于把 SQL 和业务计算全部改写为确定性代码。SQL 仍由生成模型根据可信计划产出，聚合结果仍由数据库计算；后端只对已有确定依据的事项负责，例如候选 ID 反查、字段类型、规范值存在性、日期范围、连接可达性、Limit 边界和 SQL 是否忠实于计划。对于必须结合自然语言上下文才能判断的指标选择、查询角色、比较意图和排序意图，仍由 LLM 解释。

重构后的主链为：

```text
context_builder
  -> semantic_planning
  -> context_compaction
  -> generate_sql
  -> sql_executor
```

## 2. 参考架构与复用边界

### 2.1 WrenAI

借鉴：

- 权威语义模型与可重建检索索引分离；
- 模型、字段、关系、指标、计算表达式和版本组成统一语义资产；
- 规划、dry-plan/dry-run 与真实执行分离；
- 从完整语义模型中提取本次查询所需的最小语义切片。

不复制：

- WrenAI 当前没有可直接移植的 `SemanticQueryPlan` 或 NL 到强类型计划生成器；
- 旧版 Agent 代码采用 AGPL-3.0，且自由文本 reasoning 和 SQL 再读原问题不能解决本项目的约束绕过问题；
- 不引入完整 Wren 运行时，避免替换 ShopInsight 的核心架构。

### 2.2 CHESS、DIN-SQL 与 RESDSQL

借鉴：

- CHESS：先选业务字段，再补连接字段；对最终 SQL 做语义级验收；
- DIN-SQL：将 Schema Linking、问题分解、SQL 生成和修复分开；
- RESDSQL：将 Schema Linking 与 SQL 结构生成解耦。

不复制：

- 不复制 CHESS 的硬编码阈值、多候选 SQL 执行和 LLM 单元测试；
- 不采用 DIN-SQL 的单文件实验 Prompt；
- 不引入 RESDSQL 的专用训练模型流水线。

### 2.3 时间解析

采用 Apache-2.0 的 JioNLP 作为中文时间解析核心，通过项目内适配器调用。适配器负责固定参考时间、规范化日期边界、过滤不确定结果和转换为项目契约。当前正则解析器只保留为严格格式的兼容解析，不允许在 JioNLP 明确失败后猜测日期。

## 3. 目标

- 将 LangGraph 节点、失败阶段和评测阻断点统一改为 `semantic_planning`。
- 建立不可信 `SemanticDraft` 与可信 `SemanticQueryPlan` 的明确边界。
- 让 LLM 只能使用本轮候选目录中的 ID，不得创造字段、指标、规范值、日期或 JOIN。
- 把指标聚合口径升级为权威元数据，作为 SQL LLM 的确定依据和后端一致性校验依据，而不是由后端代替数据库执行聚合。
- 结构化表达指标、维度、枚举/数值/时间谓词、排序、Top-N 和连接计划。
- 将 JOIN closure 前移到计划校验，令 `context_compaction` 退化为纯上下文编译器。
- 增加 SQL AST 与语义计划一致性校验，禁止 SQL 增删或改写计划约束。
- 保持 FastAPI、SSE、前端请求格式和 SQL 执行安全闭环不变。
- 兼容读取已有 SQL 记忆中的旧 `business_binding` 数据，但新请求不再双重维护两份真相。

## 4. 非目标

- 不修改混合检索、RRF、Top-K 和 Embedding 算法。
- 不引入完整 WrenAI、Cube 或其他语义引擎。
- 不让语义规划节点直接生成或执行 SQL。
- 不在本次实现任意嵌套查询、任意窗口函数、同比/环比或多阶段分析工作流；无法表示的复杂问题必须明确阻断。
- 不把向量相似度、LLM 置信度或候选顺序当作最终裁决依据。
- 不使用模糊 DW 查询把用户值改写成数据库值。

## 5. 总体架构

`semantic_planning` 节点内部按以下接口组织：

```text
SemanticCandidateCatalogBuilder
  -> SemanticInterpreter (LLM)
  -> DeterministicResolver
  -> PlanValidator
  -> SemanticPlanningResult
```

成功时：

```text
SemanticPlanningResult(status="resolved", plan=SemanticQueryPlan)
  -> state.semantic_plan
  -> context_compaction
```

失败时：

```text
SemanticPlanningResult(status="unresolved|ambiguous", issues=[...])
  -> failure(stage="semantic_planning", disposition="blocked")
  -> 澄清文本
  -> END
```

未经验证的 `SemanticDraft` 不写入跨节点 State，也不进入长期记忆；需要调试时只能记录经过脱敏和限长的 trace 摘要。

## 6. 权威语义元数据

### 6.1 候选目录

`SemanticCandidateCatalog` 包含：

```text
metadata_version
models/tables
columns
relationships
metrics
values
```

来源与权威级别：

- Meta MySQL：表、字段、类型、角色、别名、关系和指标定义的权威来源；
- DW MySQL：字段真实值存在性的权威来源；
- Qdrant/Elasticsearch：候选发现证据，不是事实来源；
- 历史 SQL：写法参考，不得改变本轮语义计划。

候选目录沿用当前稳定 ID 机制：

- 字段：完整 `table.column`；
- 指标：权威 `metric_id`；
- 字段值：`column_id + canonical_value` 的稳定标识；
- 关系：左右字段 ID 组成的稳定标识。

字段候选新增数据类型和业务角色；值候选始终保留所属字段；关系成为正式语义资产，不再只在 SQL 阶段猜测。

### 6.2 指标定义

`MetricConfig`、`MetricInfo`、Meta MySQL `metric_info` 和 Qdrant payload 新增：

```text
aggregation: sum | avg | count | count_distinct | min | max | expression
expression: 受控 SQL 表达式或空
relevant_columns: 依赖字段 ID
```

约束：

- 普通聚合指标由 `aggregation + relevant_columns` 确定；
- 只有 `aggregation=expression` 时允许 `expression`；
- 表达式由元数据构建阶段使用 SQLGlot 解析，必须是只读标量聚合表达式；
- 表达式只能引用 `relevant_columns`，禁止子查询、语句分隔符和未声明字段；
- GMV、ORDER_COUNT、AOV 分别配置为 `sum`、`count_distinct`、`avg`，不再依赖描述文本猜测。

### 6.3 关系定义

第一阶段复用现有 Meta 全量字段和 `schema_relations` 的唯一最短路径算法。规划结果显式保存 JOIN 边。现有“同名字段 + PK/FK 角色”仍属于已知保守启发式；后续 Meta Schema 可增加显式关系表，但不阻塞本次重构。

## 7. 不可信 SemanticDraft

LLM 输出使用严格 Pydantic 模型和判别联合，大致结构为：

```text
SemanticDraft
  source_query
  measure_mentions[]
  dimension_mentions[]
  predicate_mentions[]
  order_mentions[]
  limit_mentions[]
  ambiguity_reports[]
```

### 7.1 MeasureMention

```text
raw_text
candidate_ids[]
```

### 7.2 DimensionMention

```text
raw_text
candidate_ids[]
role: group_by | projection
```

### 7.3 PredicateMention

判别类型：

- 枚举：`raw_text`、值候选 ID、字段候选 ID、操作符意图；
- 数值：`raw_text`、字段或指标候选 ID、操作符意图、原始数值文本；
- 时间：`raw_text`、关系意图。

LLM 不得输出 canonical value、规范日期、SQL 表达式或真实字段名作为自由文本。

### 7.4 排序与 Limit

- 排序只输出目标候选 ID 和 `asc|desc` 意图；
- Limit 只保留原始片段，具体正整数由后端解析；
- “最高的前五个”应同时产生降序意图和 Limit 原文；
- 排序目标缺失、多目标无法消歧或 Limit 非法时阻断。

### 7.5 Prompt 设计

提示词使用“电商分析语义规划师”这一专业角色，而不是流水线节点角色。固定政策包括：

- 任务只限于解释当前问题并引用受控候选；
- 候选目录和当前问题是本轮可用证据；历史消息仅用于明确省略和指代；
- 对无法确定的表达保留全部候选或报告歧义；
- 不做日期计算、规范值改写、公式推导、JOIN 选择和放行判断；
- 输出严格符合结构化契约，不输出解释性正文；
- 在提交前自检所有 ID 来自输入目录、所有 `raw_text` 来自可信原文、未填写后端负责的字段。

使用 2 至 3 个短 few-shot 覆盖：唯一候选、同值跨字段歧义、未召回值但唯一字段候选。澄清 Prompt 只把结构化 issue 改写成用户可见中文，不能改变 issue、候选或阻断决定。

## 8. 后端可确定部分的规范化与校验

### 8.1 通用候选选择

```text
零个有效 ID       -> unresolved
一个有效 ID       -> 继续解析
多个有效 ID       -> ambiguous
存在目录外 ID     -> unresolved/invalid_candidate_id
raw_text 不可信    -> unresolved/invalid_raw_text
```

不按候选顺序、相似度或 LLM 自报置信度自动选第一项。

### 8.2 枚举谓词

1. 唯一值候选 ID：反查规范值与所属字段；
2. 多个值候选 ID：`ambiguous/value_ambiguous`；
3. 没有值候选但只有一个字段候选：在该字段内对用户原文做一次参数化精确查询；
4. 精确值存在：使用用户原文作为 canonical value；
5. 数据库只有“华北地区”而用户原文是“华北”：不得用 LIKE、向量或 LLM 改写，必须 unresolved/澄清；
6. 多个字段候选时不访问 DW，直接 ambiguous。

### 8.3 数值谓词

- LLM 只识别 `eq|gt|gte|lt|lte|between` 意图和原始数字片段；
- 后端使用 `Decimal` 解析并校验边界；
- 字段类型必须为数值类型；
- 目标为原始字段时编译到 `WHERE`；目标为聚合指标时编译到 `HAVING`；
- between 必须恰好有两个按升序规范化的边界；
- 单位换算只有元数据显式声明时才允许，否则阻断。

### 8.4 时间谓词

- JioNLP 接收用户时间原文和显式参考时间；
- 后端将结果规范化为闭区间日期及日期 ID；
- 模糊、不完整或多种解释同样合理时进入 ambiguous；
- 时间字段依据当前事实表和权威元数据选择，不能由 LLM 自由指定；
- 第一版支持一个主时间谓词；出现多个独立时间轴或同比/环比请求时明确阻断。

### 8.5 排序与 Limit

- 排序目标必须是计划中已选择的指标或投影/分组字段；
- Limit 解析后必须为 `1..1000`；
- 有“前 N/最高 N”但无唯一排序目标时 ambiguous；
- 没有显式 Limit 时保持空值，不由模型擅自补默认值。

## 9. 可信 SemanticQueryPlan

Graph State 保存 `semantic_plan`，不再保存 `business_binding` 作为新请求的事实来源：

```text
SemanticQueryPlan
  version
  metadata_version
  measures[]
  dimensions[]
  predicates[]
  order_by[]
  limit
  joins[]
  required_table_ids[]
  required_column_ids[]
  provenance[]
```

### 9.1 MeasurePlan

```text
metric_id
name
aggregation
expression
source_column_ids[]
output_alias
```

### 9.2 DimensionPlan

```text
column_id
role: group_by | projection
output_alias
```

### 9.3 PredicatePlan

判别联合：

- `EnumPredicate`：字段、操作符、规范值、允许 SQL 字面量；
- `NumericPredicate`：字段或指标、操作符、Decimal 字符串边界、`where|having`；
- `TemporalPredicate`：时间字段、操作符、开始/结束日期和日期 ID、粒度。

### 9.4 Provenance

每个计划元素保存：

```text
raw_text
resolved_id
method
evidence
```

用于审计、评测和澄清，不允许 SQL LLM 修改。

## 10. PlanValidator

PlanValidator 在单项解析后执行整体校验：

- 所有 ID 和元数据版本有效；
- 指标公式只引用声明字段；该检查验证权威口径和生成 SQL 的一致性，不在规划节点执行指标计算；
- 字段角色和操作符类型匹配；
- 枚举值属于谓词字段；
- 指标谓词进入 HAVING，原始字段谓词进入 WHERE；
- 排序目标属于当前计划；
- Limit 合法；
- 根据计划依赖表计算唯一 JOIN closure；
- 无连接路径为 unresolved，多条等长路径或带环保守判定为 ambiguous；
- 计划至少包含一个指标或投影，不允许空计划进入 SQL 生成。

成功后由 Validator 计算并写入 `joins`、`required_table_ids` 和 `required_column_ids`。`context_compaction` 不再决定语义，也不再把 JOIN issue 回写进计划。

## 11. ContextCompiler

保留 `context_compaction` 节点名以维持产品级流程清晰，但内部改为纯确定性 ContextCompiler：

1. 按 `semantic_plan.required_*` 读取权威 Meta 元数据；
2. 保留计划字段、指标依赖字段和 JOIN 两侧字段；
3. 按 `semantic_plan.joins` 补桥接表；
4. 删除无关表和列；
5. 补充数据库方言、版本和当前日期；
6. 元数据缺失属于系统失败，不得改写语义计划。

## 12. SQL 生成与计划一致性

`generate_sql` 主要消费：

- `semantic_plan`；
- 由计划编译得到的最小表/字段上下文；
- 指标权威公式；
- 数据库方言；
- 历史成功 SQL 只作为写法参考。

原始问题可以用于生成用户可见说明，但不能成为新增查询约束的依据。Prompt 明确要求 SQL 完整、逐项实现计划且不得添加计划外条件。

SQL 生成后，在现有 SQLGlot、EXPLAIN 和安全校验之前增加 `SqlPlanConsistencyValidator`，使用 SQLGlot AST 验证：

- SELECT 中的指标、投影和别名；
- GROUP BY 维度；
- WHERE/HAVING 谓词的字段、操作符和值；
- 时间边界；
- ORDER BY 目标与方向；
- LIMIT；
- JOIN 边和表集合；
- 不存在计划外谓词、字段和表。

不一致属于可修复错误，修复 LLM 只能读取 SQL、计划和差异列表；每次修复后重新执行完整一致性与安全校验。修复耗尽后 `failed`，绝不能执行不一致 SQL。

## 13. 错误模型

统一问题结构：

```text
PlanningIssue
  phase: schema_linking | semantic_resolution | plan_validation
  code
  source_span
  candidate_ids[]
  details
```

主要错误码：

- `invalid_candidate_id`
- `invalid_raw_text`
- `metric_not_bound`
- `value_not_bound`
- `value_not_found`
- `value_ambiguous`
- `column_ambiguous`
- `invalid_operator_for_type`
- `invalid_numeric_literal`
- `time_not_resolved`
- `time_ambiguous`
- `order_target_ambiguous`
- `invalid_limit`
- `join_path_not_found`
- `join_path_ambiguous`
- `business_object_not_planned`

业务信息不足和歧义使用 `failure.disposition=blocked`；Repository、元数据或解析器异常属于系统 `failed`。澄清 LLM 失败时仍保留确定性默认澄清文本并结束，不允许继续生成 SQL。

## 14. State、记忆与兼容迁移

### 14.1 新 State

- `DataAgentState.semantic_plan`：唯一可信计划；
- `FailureState.category` 和 `stage` 使用 `semantic_planning`；
- Draft 不写 State；
- 成功记忆保存精简后的 `semantic_plan`、SQL 和元数据版本。

### 14.2 迁移阶段

1. 特征测试锁定现有受控 ID、DW 精确兜底、阻断和 JOIN 行为；
2. 纯改节点/package 名，暂时保留旧 State 读取适配；
3. 引入 `SemanticDraft`、指标定义和 `SemanticQueryPlan`；
4. 建立唯一的 `plan -> legacy binding` 单向适配器，只验证可无损表达的迁移子集，不接入生产 Graph；
5. Planner、ContextCompiler 和 SQL 消费者准备完成后原子切换为 `semantic_plan`，避免新计划经过有损 legacy 视图或出现消费者断链；Guard、记忆和评测随后切换；
6. 停止写入 `business_binding`，只为已有历史记忆保留一版读取适配；
7. 后续版本删除旧 import 和旧记忆适配。

旧历史文档保留原节点名并标记其历史属性，不机械篡改。README、架构文档、当前学习导览和最新学习笔记更新为新链路。

## 15. Prompt 验收标准

应用 `agent-system-prompt-architect` 的检查标准：

- 角色是具体的电商分析语义规划专业身份；
- 任务、候选证据、权限边界和输出契约位于上下文示例之前；
- 模型只看到它需要执行的行为，不暴露无意义的后端流水线术语；
- 所有候选 ID 和原文片段可校验；
- 结构化输出字段无重复语义和自由 canonical 文本；
- 包含唯一候选、歧义和 DW 精确兜底三个短例子；
- 不请求展示隐藏思维，只输出结构化结果；
- 自检项具体可观察；
- Prompt 调用或解析失败时只保留可诊断的显式证据，并产生 `semantic_interpretation_failed` 阻断项；不得用不完整语义继续执行。

## 16. 分层验收标准

### 16.1 单元验收

- Draft 只接受候选 ID、原始片段和语义意图；
- 唯一 ID 成功解析，多 ID 歧义，伪造 ID 阻断；
- 字段值始终保留所属字段；
- DW 兜底只在唯一字段内执行一次参数化等值查询；
- “华北”不匹配仅存在的“华北地区”；
- `2025年第一季度` 稳定解析为 `20250101..20250331`；
- 固定参考日 `2026-07-19` 时，“上个月”为 `2026-06-01..2026-06-30`；
- “销售额最高的前5个商品”生成 GMV 降序和 Limit 5；
- 聚合指标 `> 10000` 进入 HAVING，字符串字段使用 `>` 被拒绝；
- 唯一 JOIN closure 成功，无路径和多路径分别 unresolved/ambiguous。

### 16.2 节点与图验收

- 图中存在 `context_builder -> semantic_planning -> context_compaction`；
- `semantic_planning` 的 blocked 边直达 END；
- 图中不存在 `business_binding` 节点；
- 成功 State 只向下游暴露可信 `semantic_plan`；
- blocked 时不得调用 SQL 生成或 DW 执行；
- FastAPI 请求和 SSE event type 不变，前端能显示新阶段名。

### 16.3 SQL 一致性验收

正确 SQL 必须通过指标、投影、分组、谓词、时间、排序、Limit 和 JOIN 全项检查。以下任一反例必须被拒绝并进入受限修复：

- 漏掉筛选或时间；
- 把 canonical value 改为其他值；
- 把 DESC 改为 ASC；
- 改变 Limit；
- 把 WHERE/HAVING 放错；
- 添加计划外条件、字段、表或 JOIN；
- 使用错误指标聚合公式。

### 16.4 真实数据库验收

至少运行：

1. `2025年第一季度华北地区的销售额`；
2. `2025年第一季度销售额最高的前5个商品`；
3. `2025年第一季度销售额大于10000元的地区，按销售额降序`；
4. 枚举别名查询；
5. 清空值召回后的唯一字段 DW 精确兜底；
6. 用户值与数据库值仅模糊相似时阻断；
7. 同一值属于两个字段时歧义阻断；
8. 人工制造漏时间或反向排序 SQL，验证一致性层拒绝执行。

复杂查询必须与人工审核的 Oracle SQL 结果完全一致；结果比较需排序并规范化 Decimal/Float，不能只判断非空。核心 E2E 连续运行三轮，不允许出现错误计划放行或 SQL 与计划不一致。

### 16.5 工程门槛

- 新增单元、节点、图契约和一致性测试全部通过；
- 现有回归测试全部通过；
- `uv run ruff check app tests` 无错误；
- 110 条评测中现有受支持能力不得回退；
- 所有 blocked 场景均不得执行 DW 业务查询；
- 新链路正常查询只使用一次语义解释 LLM 调用，澄清 LLM 仅在阻断时调用；
- 不修改用户当前未提交的 `app/agent/retrieval_context.py`。

## 17. 计划文件划分

为降低一次性重构风险，实施拆成三份可独立验收的计划：

1. 节点改名、契约骨架和兼容层；
2. 权威指标元数据、SemanticDraft/Plan、解析器与 PlanValidator；
3. ContextCompiler、SQL 计划一致性、记忆/评测迁移和真实 E2E。

每份计划均使用 TDD，小步提交；后一计划必须在前一计划验收后开始。

## 18. 参考资料

- WrenAI 架构：<https://docs.getwren.ai/oss/reference/architecture>
- WrenAI 当前仓库与许可证：<https://github.com/Canner/WrenAI>
- CHESS 论文：<https://arxiv.org/abs/2405.16755>
- CHESS 源码：<https://github.com/ShayanTalaei/CHESS>
- DIN-SQL 论文：<https://openreview.net/pdf?id=p53QDxSIc5>
- RESDSQL：<https://arxiv.org/abs/2302.05965>
- JioNLP：<https://pypi.org/project/jionlp/>
