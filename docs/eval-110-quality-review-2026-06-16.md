# 110 条评测集质量审查

## 目标

新增 `examples/eval_cases_110.yaml`，用于支撑三类轻量消融：

- 召回效果：对比关闭混合召回/历史 SQL 样例召回与开启完整上下文召回。
- 校验成果：对比关闭业务绑定/SQL 安全校验与开启现有校验。
- 成本下降：对比关闭轻量模型分流、缓存、上下文压缩与开启优化策略。

这份评测集不把 110 条样本包装成统计学意义上的大规模 benchmark，而是作为工程回归和消融诊断集使用。

## 生成原则

- 复用现有 23 条核心回归 case，保留历史失败修复轨迹。
- 新增 case 使用模板生成，`expected_columns`、`expected_metrics`、`expected_values` 由元数据映射生成，不让自由文本决定事实标注。
- Query 文本可以多样化，但标注必须能回到 `conf/meta_config.yaml` 中的真实表字段、指标和值域列。
- 通过 tags 标记消融子集：`ablation_retrieval`、`ablation_guard`、`ablation_cost`、`sql_memory`。

## 当前分布

- 总数：110 条。
- suite 分布：`smoke=4`、`regression=48`、`realistic=29`、`adversarial=29`。
- 校验负例：28 条带 `expected_blocked_by`。
- 召回消融：55 条带 `ablation_retrieval`。
- 成本消融：48 条带 `ablation_cost`。
- 历史 SQL 记忆：15 条带 `sql_memory`。
- 值域召回标注：27 条带 `expected_values`。

## 第一轮质量反思

从“按标签切消融子集”的角度检查，发现复用的老 adversarial case 没有统一打 `ablation_guard` 标签。这样会导致后续统计校验消融时漏掉原有安全回归样本。

修正：

- 在 `generate_eval_cases_110.py` 中增加 `_normalize_base_case()`。
- 所有 `suite=adversarial` 的基础 case 统一补 `ablation_guard` 标签。
- 增加质量测试，要求 `ablation_guard` 样本数不少于 25。

同时发现不能把所有 adversarial 都当成“应被拦截”的负例。比如“统计各大区 GMV”是防误拦正例，属于对抗表达但应正常通过。

修正：

- 质量测试按 `expected_blocked_by` 区分负例。
- 只有带 `expected_blocked_by` 的 case 才要求 `safety` 能力和空 SQL 期望。

## 第二轮质量反思

从“实际跑召回消融”的角度检查，发现只有 `expected_columns` 和 `expected_metrics` 不够。值域召回是当前混合召回链路的重要目标，如果没有结构化 `expected_values`，就只能靠最终 SQL 字符串间接判断，无法单独评价 ES/Qdrant/RRF 对枚举值召回的贡献。

修正：

- `EvalCase` 增加 `expected_values`。
- `evaluate_case()` 增加 `missing_expected_value` 失败类型，stage 归为 `rag_recall`。
- 生成脚本对带枚举筛选的 case 写入形如 `dim_region.region_name.华东` 的值域期望。
- 增加测试，要求召回消融样本中至少 25 条带 `expected_values`。

## 仍然保留的边界

- 这份评测集不是线上真实流量分布，不能用于声称“线上准确率”。
- 历史 SQL 记忆样本是单轮 query 形式，真正的记忆写入/召回仍应结合 `run_sql_memory_smoke` 或后续专门 ablation runner 验证。
- 成本消融需要 runner 层提供开关，例如关闭缓存、关闭 Fast Model、关闭上下文压缩；评测集只提供可切分样本，不直接实现开关。
- 安全消融不能真的执行危险 SQL；无校验模式应以 dry-run 或 would-block 方式统计。

## 验收标准

- `examples/eval_cases_110.yaml` 可以被现有 `load_eval_cases()` 正常加载。
- 所有 expected 字段和指标都能在 `conf/meta_config.yaml` 中找到。
- 召回、校验、成本三类消融都有足够样本覆盖。
- 质量测试固定这些约束，避免后续评测集退化。
