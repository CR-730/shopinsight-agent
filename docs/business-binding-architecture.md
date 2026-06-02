# Business Binding 架构说明

本文记录当前电商问数 Agent 的业务语义裁决边界。目标是降低重复校验和重复裁决，让自然语言中的业务对象只在一个节点被解析一次。

## 当前链路

```text
pre_rag_guard
  -> RAG recall
  -> merge_retrieved_info
  -> business_binding
  -> filter_table / filter_metric
  -> add_extra_context
  -> semantic_guard
  -> generate_sql
  -> pre_sql_execution_validation
  -> run_sql
```

## 职责边界

`business_binding` 是唯一业务语义裁决源，负责把自然语言业务对象绑定为结构化对象：

```yaml
business_binding:
  metrics: []
  filters: []
  time: null
  unresolved: []
  ambiguous: []
```

后续节点只消费这个结构，不再重新解释业务别名、枚举值或时间表达。

## Metrics Binding

指标绑定只基于已召回的指标元数据：

```yaml
metrics:
  - raw_mention: 销售额
    canonical_metric: GMV
    matched_by: metric_alias
    relevant_columns:
      - fact_order.order_amount
    confidence: high
```

`filter_metric` 不再调用 LLM，也不再裁决指标是否存在。它只按 `business_binding.metrics[].canonical_metric` 裁剪上下文；没有 binding 时保留 RAG top-k。

## Filters Binding

枚举过滤绑定由字段 alias 和取值证据共同驱动：

```yaml
filters:
  - raw_value: 北方区域
    canonical_value: 华北
    column: dim_region.region_name
    field_alias: ""
    matched_by: enum_alias
    allowed_sql_literals:
      - 华北
```

当前解析来源包括：

- `value_alias` 元数据表中的别名映射。
- `retrieved_value_infos` 中召回到的真实枚举值。
- `column_value_exists()` 对别名 canonical value 的数仓存在性确认。

字段 alias 只用于定位候选字段，不硬编码“区域 -> dim_region.region_name”。分组表达如“按大区”“各商品品类”不会被当作枚举过滤值。

未知枚举值会进入 `unresolved`：

```yaml
unresolved:
  - type: enum_value
    raw_text: 火星
    candidate_column: dim_region.region_name
    reason: value_not_found
```

## Time Binding

时间表达由 `business_binding` 解析为统一的 date range：

```yaml
time:
  raw_text: 2025 年第一季度
  grain: quarter
  year: 2025
  quarter: Q1
  start_date: 2025-01-01
  end_date: 2025-03-31
  start_date_id: 20250101
  end_date_id: 20250331
  strategy: date_range
  required_columns:
    - fact_order.date_id
```

当前最小支持：

- `YYYY 年第 N 季度`
- `YYYY-QN`
- `YYYY 年 N 月`
- `YYYY-MM-DD`

当前项目只有一个核心事实表，所以 `required_columns` 固定为 `fact_order.date_id`。如果后续增加多事实表，需要从已选表上下文推导事实表时间键。

## Semantic Guard

`semantic_guard` 只做 binding 完整性检查：

- `unresolved` 非空：阻断，不生成 SQL。
- `ambiguous` 非空：阻断或追问。
- 其余情况放行。

它不再抽取指标，不再重新校验枚举，不再维护另一套业务判断。

## SQL 生成

`generate_sql` 消费 `business_binding`：

- 有 `metrics` 时，只能使用绑定指标和 `relevant_columns`。
- 有 `filters` 时，只能使用 `canonical_value` / `allowed_sql_literals`。
- 有 `time` 且 `strategy=date_range` 时，使用 `start_date_id` / `end_date_id` 生成时间条件。

SQL 生成节点不承担业务裁决职责。

## SQL 执行前校验

`pre_sql_execution_validation` 保持执行前硬闸门职责：

- 单 SELECT。
- 危险 SQL。
- `SELECT *`。
- 敏感字段。
- 未绑定枚举值。
- join 关系是否违反元数据。

它不做自然语言业务解析，不再扩展指标别名判断。

## 当前验收

已验证的关键 case：

- `sql_time_range_quarter_region`：通过 `time_binding` 解析季度，不再依赖 `dim_date.year` 召回。
- `adv_unknown_region_value`：通过 `unresolved enum_value` 阻断，不生成 SQL。
- 21 个 eval case 全部通过。
