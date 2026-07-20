# ShopInsight 受控业务绑定与确定性上下文裁剪重构设计

## 1. 背景

当前链路在 `context_builder` 后依次执行：

```text
business_binding -> context_compaction -> generate_sql
```

现有实现存在两组相关问题：

1. `business_binding` 已经识别指标、筛选、分组和时间，但 `context_compaction` 又调用 `filter_table_info` LLM 重新判断所需表字段，形成职责重叠和二次语义决策。
2. 召回的 `ValueInfo` 原本包含 `value + column_id`，候选 Prompt 却把它压平成字段值文本，随后要求 LLM 自由输出 `field_hint` 和 `normalized_text`。这会丢失字段值来源，并可能把“值真实存在”误当成“字段和值的语义映射正确”。

本次重构把两处问题统一收敛到一个原则：

> LLM 负责在后端提供的受控候选集中做语义选择；canonical 指标、字段和值必须由后端根据候选 ID 反查，SQL 上下文必须由 canonical binding 确定性裁剪。

## 2. 目标

- 为指标、字段和字段值建立本轮请求级受控候选目录。
- LLM 只能返回候选 ID 和来自用户原文的 `raw_text`，不能自由生成字段名或 canonical 值。
- `business_binding` 新增普通投影字段 `projections`，覆盖“列出订单编号和商品名称”等非指标、非分组字段诉求。
- 保留字段值漏召回时的 DW MySQL 兜底，但只能在一个已选定的受控字段上执行精确存在性查询。
- 多个合法候选一律标记 `ambiguous`，不按相似度、遍历顺序或模型偏好自动选一个。
- 使用 canonical binding 和主外键元数据确定性裁剪 `sql_context`，移除 `filter_table_info` LLM 调用。
- 保持 LangGraph 产品级节点名称、主顺序和外部 API/SSE 协议不变；允许把 `context_compaction -> generate_sql` 调整为带 blocked 分支的条件边。

## 3. 非目标

- 不修改 FastAPI、SSE 或前端协议。
- 不重构混合检索算法、RRF、Top-K 或 Embedding。
- 不修改 SQL 生成、SQL Guard、EXPLAIN、纠错和执行闭环的总体结构。
- 本次不迁移 Meta MySQL Schema，不新增显式外键目标表；第一版沿用当前“同名字段 + FK/PK 角色”的关系推断规则。
- 本次不实现全局 Steiner Tree、带环复杂图规划或最优 JOIN 树搜索；显式 FK 目标和复杂关系图规划留待后续 Meta Schema 演进。
- 不使用模糊 SQL、`LIKE` 或向量检索直接确认漏召回的 canonical 字段值。

## 4. 方案选择

### 4.1 采用方案：完整 canonical IR + 确定性裁剪

后端基于本轮召回结果构造候选目录，LLM 返回受控 ID，Validator 反查并生成 canonical `BusinessBindingState`。上下文裁剪仅消费 canonical binding 和权威元数据，不再调用 LLM。

### 4.2 未采用：只保护更多字段但保留裁剪 LLM

该方案改动较小，但仍保留职责重复、额外模型成本和二次语义判断，不解决根因。

### 4.3 未采用：让 LLM 输出完整 SQL 逻辑计划

该方案会显著扩大中间表示和校验复杂度，并让 business binding 过度接近 SQL 生成，超出本次重构范围。

## 5. 候选目录

候选目录只覆盖“本轮召回候选 + 相关 Meta 元数据补齐”，不加载全部元数据进入 Prompt。

目录是 `business_binding` 节点内部的临时对象，不写入跨节点 State；最终 State 只保存 canonical binding。

### 5.1 字段候选

来源：`state.sql_context.tables[*].columns`。

字段自身的完整 `column_id` 作为受控 ID：

```json
{
  "candidate_id": "dim_region.region_name",
  "table": "dim_region",
  "name": "region_name",
  "aliases": ["地区", "区域"],
  "role": "dimension"
}
```

字段候选还需标记是否允许作为 projection。命中 SQL 安全策略中的敏感字段不得作为投影候选放行；允许仅作为 JOIN key 使用的字段只能由确定性 JOIN closure 加入。

该集合包含字段召回结果、指标依赖字段、字段值依赖字段以及召回合并阶段补充的关键字段。

### 5.2 指标候选

来源：`state.sql_context.metrics`。

在当前 State 未保留数据库指标 ID 的前提下，第一版使用稳定键 `metric:<canonical_name>`：

```json
{
  "candidate_id": "metric:销售额",
  "name": "销售额",
  "aliases": ["成交金额", "GMV"],
  "relevant_columns": ["fact_order.order_amount"]
}
```

后端必须拒绝目录中不存在的指标 ID。

### 5.3 字段值候选

来源：

- `state.retrieval_context.values` 中保留的 `ValueInfo.value + ValueInfo.column_id`；
- 与当前候选字段相关的 Meta MySQL 枚举别名。

字段值候选 ID 由 `column_id + canonical_value` 生成稳定哈希，避免同值跨字段冲突：

```json
{
  "candidate_id": "value:dim_region.region_name:<stable_hash>",
  "value": "华北地区",
  "aliases": ["华北"],
  "column_id": "dim_region.region_name",
  "source": "retrieval_or_meta_alias"
}
```

同一个文本值属于两个字段时必须生成两个不同候选。

## 6. LLM 候选输出契约

保留 `source_query` 和 `user_response`。候选结构调整为：

```text
MetricMention
  raw_text
  candidate_ids[]

FilterMention
  raw_text
  value_candidate_ids[]
  column_candidate_ids[]  # 仅用于值漏召回后的精确查询兜底

GroupByMention
  raw_text
  column_candidate_ids[]

ProjectionMention
  raw_text
  column_candidate_ids[]

TimeMention
  raw_text
  granularity_hint
  normalized_text
```

约束：

- `raw_text` 必须来自当前问题；只有明确承接上一轮时才允许来自可信会话上下文。
- 指标、筛选、分组和投影不得输出目录外 ID。
- 删除筛选和分组中的自由文本 `field_hint`。
- 删除指标、筛选和分组中的自由文本 `normalized_text`；canonical 结果由后端目录反查。
- 时间继续允许规范化文本，但最终仍由确定性时间解析器处理。
- 模型认为多个候选都可能成立时必须返回全部候选 ID，交由后端标记歧义。
- 后端校验 `raw_text` 必须来自当前 `source_query` 或本轮传入的可信滑动会话文本；否则标记 `invalid_raw_text`。

## 7. 确定性校验

### 7.1 通用判定

```text
零个有效 ID       -> unresolved
一个有效 ID       -> canonical binding
多个有效 ID       -> ambiguous
包含目录外伪造 ID -> unresolved（invalid_candidate_id）
```

不允许使用候选顺序、相似度最高项或“第一个真实存在的值”自动消歧。

如果正常问数请求在指标、筛选、分组、投影和时间上全部为空，必须产生 `business_object_not_bound`，不能把空 binding 交给 SQL 生成。

### 7.2 指标

通过指标候选 ID 反查 `canonical_metric`、`relevant_columns` 和证据，不再优先匹配 LLM 自由生成的 `normalized_text`。

### 7.3 筛选值

处理顺序：

1. 一个有效 `value_candidate_id`：直接反查 `canonical_value + column_id`。
2. 多个有效 `value_candidate_id`：标记 `ambiguous`。
3. 没有值候选、只有一个有效 `column_candidate_id`：在该字段下对 `raw_text` 执行 DW MySQL 精确存在性查询。
4. 多个字段候选：标记 `ambiguous`。
5. 精确查询不存在或没有候选字段：标记 `unresolved`。

DW 兜底必须调用现有参数化 `column_value_exists(table_name, column_name, raw_text)` 或等价精确查询；禁止 `LIKE`、前后缀补全或由 LLM 改写原始值。

### 7.4 分组与投影

通过字段候选 ID 反查真实 `column_id`。分组只能绑定允许作为维度的字段；投影字段还必须服从现有敏感字段和明细查询安全策略，最终由 SQL Guard 再次检查。

### 7.5 时间

沿用 `resolve_time_mentions()`，输入包括时间 mentions 和完整 `source_query`。时间不进入候选目录。

## 8. Canonical State

`BusinessBindingState` 新增：

```text
projections: list[ProjectionBindingState]
```

`ProjectionBindingState` 至少包含：

```text
raw_mention
column
field_alias
matched_by
confidence
```

其他 canonical 对象继续保存指标、筛选、分组、时间、`unresolved` 和 `ambiguous`。候选目录和 LLM 临时 ID 不写入最终 State。

## 9. LLM 失败降级

保留确定性 fallback，但 fallback 也只产生目录内 ID：

- 原文唯一命中指标名称或别名：返回对应指标候选 ID。
- 原文唯一命中字段名称或别名：根据句式恢复分组或投影候选 ID。
- 原文唯一命中枚举别名或召回值：返回对应字段值候选 ID。
- 同一原文命中多个候选：保留全部 ID，后端标记 `ambiguous`。
- 无法唯一恢复：`unresolved`，不生成自由文本字段提示。

## 10. 确定性上下文裁剪

删除 `filter_table_context()` 中的 `filter_table_info` LLM 调用。裁剪所需字段集合为：

```text
required_columns =
    metric.relevant_columns
  + filter.column
  + group.column
  + projection.column
  + time.required_columns
```

处理流程：

1. 校验 required column ID 均属于候选目录或可从 Meta MySQL 按 ID 补齐。
2. 计算 required columns 所属表集合。
3. 对多表集合补齐 JOIN 所需字段。
4. 保留 required columns、JOIN keys 及其完整元数据。
5. 删除未使用的表和字段。
6. 继续调用 `add_runtime_context()` 补充日期与数据库方言/版本。

`filter_metric_context()` 继续按 canonical metric 确定性过滤，不调用 `filter_metric_info.prompt`。

`disable_context_compaction=true` 重新定义为：保留全部候选表字段，但仍执行 required column/键字段补齐和 JOIN 可达性校验，并继续补充日期及数据库信息。

## 11. JOIN 字段补齐

当前 Meta MySQL 只记录字段角色，没有显式外键目标。第一版基于 Meta 全量键字段构图，并复用现有 SQL Guard 的合法关系定义：

```text
字段名相同
+ 一侧 role=foreign_key
+ 另一侧 role=primary_key
= 合法候选关系
```

在全量键关系图中连接本轮 required tables：

- 第一版只自动接受树/森林型关系分量中的确定性增量 closure：按稳定顺序逐个连接 required table，每一步使用多源无权 BFS，并且只接受唯一最短路径。
- 唯一最短路径：补入路径上的桥接表和两侧 FK/PK 字段；整体复杂度约为 `O(T * (V + E))`，其中 `T` 是 required table 数量。
- 多条等长最短路径：标记 `ambiguous`，阻断 SQL 生成。
- required tables 所在的确定性关系分量带环时，即使当前两表存在唯一局部最短路，也保守标记 `ambiguous`。
- 局部存在等长路径时，即使全局 Steiner Tree 存在唯一最优解，也不做全局推断，仍标记 `ambiguous`。这是刻意接受的安全假阴性，用于避免自动生成未经显式元数据证明的 JOIN。
- 无路径且需要多表：标记 `unresolved`，阻断 SQL 生成。

显式外键目标、全局 Steiner 求解和复杂带环图规划属于后续 Meta Schema 改进，不在本次重构中实现。

## 12. 节点与数据流

LangGraph 节点保持不变：

```text
context_builder
  -> 原始 retrieval_context + 候选 sql_context
business_binding
  -> 构造请求级候选目录
  -> LLM 选择受控 ID
  -> 后端确定性校验
  -> canonical business_binding（含 projections）
context_compaction
  -> 从 canonical binding 计算 required columns
  -> Meta 补齐字段和 JOIN keys
  -> 确定性生成紧凑 sql_context
  -> JOIN closure 成功：generate_sql
  -> JOIN 无路径/多路径：blocked -> END
generate_sql
  -> 消费 canonical binding + 紧凑 sql_context
sql_executor
```

## 13. 错误与阻断

- 候选 ID 不存在：`unresolved`, reason=`invalid_candidate_id`。
- 没有有效候选：`unresolved`，使用对象类型对应的 `*_not_bound`。
- 多个有效候选：`ambiguous`，附带候选 column/value/metric ID 供澄清层解释。
- DW 精确值不存在：`unresolved`, reason=`value_not_found`。
- JOIN 关系不存在：`unresolved`, reason=`join_path_not_found`。
- JOIN 关系不唯一：`ambiguous`, reason=`join_path_ambiguous`。
- 任何 unresolved/ambiguous 继续沿用当前 `failure.disposition=blocked`，不进入 SQL 生成。

澄清 LLM 只负责把确定性问题转换为用户可见文本，不参与是否阻断的判断。

## 14. Prompt 变化

- 重写 `binding_candidate_extractor.prompt`，输入结构化指标、字段和值候选目录。
- 明确模型只能返回候选 ID，不能发明 ID、字段名或 canonical 值。
- 增加普通投影字段识别规则和示例。
- 保留非问数请求的 `user_response + find_error` 兼容行为。
- `filter_table_info.prompt` 和未使用的 `filter_metric_info.prompt` 不再属于运行链；确认无引用后删除，避免误导维护者。

## 15. 测试策略

所有生产代码修改遵循 TDD：先新增失败测试，再做最小实现。

### 15.1 候选目录

- 同一字段值属于不同字段时生成不同 value candidate ID。
- 目录保留 `ValueInfo.column_id`。
- 只包含本轮候选及相关 Meta 补齐，不加载无关全库字段。

### 15.2 候选输出校验

- 一个合法 ID 成功绑定。
- 多个合法 ID 进入 ambiguous。
- 伪造 ID 进入 unresolved。
- LLM 不再输出 `field_hint`/筛选 `normalized_text`。

### 15.3 DW 精确兜底

- 值候选缺失、唯一字段 ID、精确值存在时成功。
- 精确值不存在时 unresolved。
- 多个字段 ID 时不访问 DW，直接 ambiguous。
- 测试证明没有调用模糊匹配接口。

### 15.4 投影绑定

- “列出订单编号和商品名称”绑定两个 projection column IDs。
- 同一别名对应多个字段时 ambiguous。
- 投影字段进入最终精简 `sql_context`。

### 15.5 确定性裁剪

- 无 LLM 调用。
- 指标、筛选、分组、投影、时间字段均被保留。
- 多表唯一 FK/PK 关系补齐两侧 JOIN keys。
- JOIN 无关系或多关系时阻断。
- 环图即使存在唯一局部最短路径也 ambiguous；局部等长但全局 Steiner 唯一的审查反例同样 ambiguous，证明安全假阴性是稳定契约。
- 无绑定字段的兼容行为必须显式定义并测试，不能静默恢复到 LLM 裁剪。

### 15.6 回归

- 现有 business binding、graph contract、SQL Guard、SQL executor 测试继续通过。
- 更新受旧候选模型和旧裁剪 LLM 行为影响的测试。
- 运行完整 `pytest` 与 Ruff 检查。

## 16. 兼容与迁移

- LangGraph 节点名称、边和 State 顶层字段保持不变。
- `business_binding.projections` 是新增字段；SQL Prompt 同步读取但现有消费者对缺失值按空列表处理。
- 更新评测数据与测试 fixture，使其包含 projections 默认空列表。
- 删除旧 Prompt 前使用全仓搜索确认无运行时引用。
- 不保留旧 `field_hint` 自动遍历全部字段的兼容路径，防止旧缺陷继续存在。

## 17. 预期收益

- 消除一次非必要 LLM 裁剪调用，降低 Token、延迟和模型故障面。
- 指标、字段、字段值的 canonical 来源可审计。
- 不再丢失字段值的 column provenance。
- 多候选冲突能够可靠进入澄清路径，不再接受第一个真实值。
- business binding 成为 SQL 生成前唯一的业务语义决策层，context compaction 回归纯确定性优化职责。

## 18. 已知限制

- 召回没有覆盖正确字段时，受控候选目录无法凭空发现该字段；系统会安全地 unresolved，而不是扩大到全库猜测。
- 目前 JOIN 关系仍依赖同名 FK/PK 推断，复杂 Schema 需要后续增加显式关系元数据。
- 第一版 JOIN closure 只接受树/森林型关系图中的确定性增量唯一最短路径；带环或局部等长路径会保守阻断，可能产生安全假阴性。
- LLM 仍可能漏选候选；后端保证它不能选择目录外对象，但不能保证模型总能找到目录中的正确对象。评测需要单独覆盖候选选择召回率。
