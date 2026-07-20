# 语义计划执行约束收口设计

日期：2026-07-20
状态：当前实现说明

## 1. 目标

本轮只收口四个已经确认的缺口：

1. 从运行时、状态契约和当前测试中彻底移除旧业务绑定兼容层；
2. 在 SQL—计划一致性校验中拒绝计划外 `OFFSET`；
3. 允许语义理解模型在受控关系上选择 `INNER JOIN` 或 `LEFT JOIN`，后端验证后写入可信计划；
4. 仅在可信计划要求 `BETWEEN` 时，才把 SQL 中的 `>=` 与 `<=` 规范化成闭区间。

本轮不修改 Meta MySQL 的表结构、元数据 YAML 或元数据构建脚本，不引入 Cube、WrenAI、Calcite 等运行时，也不扩展 `FULL/CROSS/SEMI/ANTI JOIN`。

## 2. 设计原则

- 模型负责理解自然语言语义；后端负责候选真实性、关系合法性、计划闭包和 SQL 一致性等确定性约束。
- JOIN 选择必须绑定到本轮候选目录中的关系 ID，模型不得自由创造表名、字段名或连接条件。
- SQL 生成模型不是事实来源。可执行 SQL 必须与已校验的 `SemanticQueryPlan` 一致。
- 删除兼容层时不保留双写、旧 State 字段或生产路径回退。
- 每个缺口先增加能够复现问题的失败测试，再编写最小实现。

## 3. 旧业务绑定兼容层清理

### 3.1 删除范围

从当前运行时契约中删除：

- `DataAgentState.business_binding`；
- failure category 中的旧业务绑定枚举值；
- `app/agent/business_binding/` 兼容包；
- `app/agent/nodes/business_binding.py` 节点别名；
- `semantic_planning` 节点导出的旧 callable 别名；
- 从旧绑定结果转换语义计划的兼容代码；
- SQL 字面量守卫对旧绑定结果的读取；
- 已废弃的旧澄清 Prompt；
- 只验证旧兼容行为的测试。

仍有价值的纯语义计划能力必须迁移到语义明确的新模块中。例如，长期记忆中读取 `semantic_plan` 的逻辑可以保留，但不得继续依赖旧绑定字段或旧命名。

### 3.2 SQL 枚举字面量来源

`validate_sql_before_execution()` 的字符串字面量白名单只允许来自：

- `SemanticQueryPlan.predicates` 中 `EnumPredicate.allowed_sql_literals`；
- 与安全策略无关的 SQL 固有字面量不在本轮扩展。

召回值不再独立授权 SQL 字面量。召回只负责发现候选；只有被语义规划选为受控值候选并完成必要权威校验后进入可信计划的值，才能进入 SQL。Meta 别名候选需要 DW 精确校验，但 DW 不负责发现目录中缺失的值。

### 3.3 验收

- `app/`、`tests/`、`prompts/`、`examples/`、`conf/` 和 `README.md` 中不存在旧运行时字段、导入路径、节点名或 Prompt；
- 新请求 State 只保存 `semantic_plan`；
- 枚举值候选缺失时语义规划返回 `value_not_bound`，不得通过字段候选和 DW 精确查询补做绑定；
- Meta 别名候选的规范值经过 DW 精确验证后能够进入可信计划；
- 仅存在于召回值、但未进入可信计划的字符串不能进入 SQL。

历史设计文档可以保留迁移背景，但必须标明已废弃，不能作为当前实现说明。

## 4. OFFSET 一致性约束

当前 `SemanticQueryPlan` 只有 `limit`，没有分页起点语义。因此任何生成 SQL 中的 `OFFSET` 都是计划外行为。

SQLGlot 解析后，只要 SELECT AST 存在非空 `offset`：

```text
code = offset_extra
path = offset
expected = null
actual = 实际 OFFSET 整数或 SQL 表达式
```

该差异属于 `repairable_error`，进入现有 SQL 修复回路。修复耗尽后按现有 `correction_exhausted` 失败。

本轮不在计划中增加 `offset` 字段。以后只有在产品明确支持“第几页、跳过前 N 条”时，才扩展计划契约。

验收用例至少包括：

- 计划 `limit=5`，SQL `LIMIT 5`：通过；
- 计划 `limit=5`，SQL `LIMIT 5 OFFSET 5`：`offset_extra`；
- 计划无 limit，SQL 只有 OFFSET：失败；
- 修复模型移除 OFFSET 后重新校验通过。

## 5. 受控 JOIN 类型选择

### 5.1 开源方案取舍

Vanna、LangChain SQL Chain、CHESS 和 DIN-SQL 都允许 LLM 在生成完整 SQL 时决定 JOIN 类型。ShopInsight 采用更窄的受控版本：模型只选择候选关系及 `inner/left`，不生成连接字段；后端仍计算和验证 JOIN closure。

本轮只借鉴职责划分，不复制第三方源码。

### 5.2 不可信草稿契约

新增严格结构：

```python
class JoinMention(_StrictDraftModel):
    raw_text: str
    relationship_candidate_id: str
    join_type: Literal["inner", "left"]
    left_table_candidate_id: str | None = None
```

并在 `SemanticDraft` 增加：

```python
join_mentions: list[JoinMention]
```

语义：

- `relationship_candidate_id` 必须来自本轮 `SemanticCandidateCatalog.relationships`；
- `join_type="inner"` 时，左右顺序不影响行集合，`left_table_candidate_id` 可以省略；
- `join_type="left"` 时，必须提供 `left_table_candidate_id`，明确哪张表位于 LEFT JOIN 左侧；
- `raw_text` 必须能在可信用户输入中定位，防止模型创造不存在的行保留理由；
- 模型不得输出 JOIN ON 表达式、任意表名或候选目录外 ID。

为使模型只能选择现有关系，`interpreter._serialize_catalog_for_prompt()` 必须把候选关系序列化为只读列表，包含关系 ID、两侧表 ID 和两侧字段 ID。不得向模型暴露自由 SQL condition。

### 5.3 后端解析规则

`join_mentions` 不写入 State，也不提前伪装成可信 `JoinPlan`。Orchestrator 在得到 `SemanticDraft` 后，调用独立的确定性解析函数，把合法 mention 转换为仅在本次调用中存在的 `ResolvedJoinPreference`；随后把这些 preference 作为显式参数传给 PlanValidator。

PlanValidator 仍先根据已解析的指标、维度和谓词计算唯一 JOIN closure。随后将 `ResolvedJoinPreference` 应用于 closure：

1. 无效关系 ID：`invalid_candidate_id`；
2. 模型选择了 closure 之外的关系：`join_not_required`；
3. 同一关系出现互相冲突的类型或左表：`join_type_ambiguous`；
4. `left` 未提供左表，或左表不是关系端点：`join_left_table_invalid`；
5. closure 中没有对应 mention：默认 `inner`，保持当前查询行为；
6. 合法 mention：写入可信 `JoinPlan`。

默认 `inner` 是缺省策略，不表示模型判断结果。只有模型从用户表达中识别出需要保留左侧未匹配行时，才显式选择 `left`。

内部 preference 至少保存关系 ID、类型和可选左表 ID。它不是公共 State 契约，不进入长期记忆；最终只有经过 closure 校验后的 `JoinPlan` 可以跨越语义规划节点。

### 5.4 可信计划契约

```python
class JoinPlan(_StrictPlanModel):
    left_column_id: str
    right_column_id: str
    join_type: Literal["inner", "left"]
```

对于 `left`，`left_column_id` 必须属于 `left_table_candidate_id`；后端在写入计划时规范化端点顺序。对于 `inner`，后端仍使用稳定顺序，便于序列化和比较。

### 5.5 SQL 一致性

SQLGlot 校验每个实际 JOIN 的：

- 关系端点；
- JOIN 类型；
- LEFT JOIN 的左右方向；
- 是否缺少 ON；
- 是否出现计划外关系。

规范化规则：

- 裸 `JOIN` 和显式 `INNER JOIN` 都归一为 `inner`；
- `LEFT JOIN` 归一为 `left`；
- `RIGHT JOIN`、`FULL JOIN`、`CROSS JOIN` 和逗号连接不在本轮能力范围，产生 `join_type_unsupported`；
- 端点正确但类型错误，产生 `join_type_mismatch`；
- LEFT JOIN 类型正确但方向相反，产生 `join_direction_mismatch`。

这些差异进入现有修复回路，修复 Prompt 接收可信计划和差异列表。

### 5.6 Prompt 边界

语义理解 Prompt 只需说明：

- 普通匹配关系可以省略 join mention，后端默认 inner；
- 用户明确要求保留没有关联记录的左侧对象时，选择对应关系、`left` 和左表候选 ID；
- 不确定需要保留哪一侧时报告 ambiguity，不猜测；
- 不输出 SQL JOIN 语句。

SQL 生成和修复 Prompt 必须把 `JoinPlan.join_type` 当作硬约束。

所有 Prompt 修改在实施时遵循 `agent-system-prompt-architect` 的角色、运行时隔离和结构化输出规范。

## 6. 闭区间规范化

当前实现会把同一子句、同一目标上唯一的 `gte` 与 `lte` 无条件合并成 `between`，导致以下可信计划被误判：

```text
price >= 100
price <= 1000
```

如果计划明确保存的是两个独立谓词，SQL 中相同的两个谓词必须逐个比较，不能合并。

新规则：

1. 先从可信计划构建 expected predicate atoms；
2. 只收集 expected 中 operator 为 `between` 的 `(clause, target)`；
3. 仅对这些目标，把 actual SQL 中唯一的 `gte + lte` 合并成 `between`；
4. expected 为两个独立谓词时，actual 保持两个 atoms；
5. 多个下界或上界不做猜测性合并。

验收用例至少包括：

- 计划 `between [100,1000]`，SQL `BETWEEN 100 AND 1000`：通过；
- 同一计划，SQL `>=100 AND <=1000`：通过；
- 计划是独立 `gte 100`、`lte 1000`，SQL 同样为两个条件：通过；
- 计划只有 `gte 100`，SQL 额外增加 `<=1000`：`predicate_extra`；
- 时间 during 闭区间继续兼容 `BETWEEN` 与 `>=/<=` 两种 SQL 写法。

## 7. 数据流

```text
用户问题
→ LLM 输出受控 SemanticDraft（可选 JoinMention）
→ 后端解析指标、维度、谓词
→ 后端计算唯一 JOIN closure
→ 后端验证并应用 JoinMention
→ 生成可信 SemanticQueryPlan
→ 从计划编译最小 SQL 上下文
→ LLM 生成 SQL
→ SQLGlot 校验谓词、JOIN 类型、方向、LIMIT、OFFSET
→ 原有结构、EXPLAIN 和安全校验
→ 执行或进入受限修复回路
```

## 8. 测试与验收

采用 TDD，按以下顺序实施：

1. 旧字段和 SQL 字面量信任边界；
2. OFFSET；
3. 闭区间规范化；
4. JOIN Draft、Resolver、Plan、SQL 一致性和修复闭环；
5. 删除兼容模块并清理导入；
6. 完整单元测试、集成测试、Ruff；
7. 只运行小规模核心真实评测，不运行 110 条集合。

最终验收必须满足：

- 全量现有测试和新增测试通过；
- Ruff 通过；
- 核心四条语义规划评测通过；
- “统计成交额”输入审核回归通过；
- 至少增加一条明确 LEFT JOIN 的真实/集成用例；
- OFFSET、JOIN 类型错误和独立范围谓词都能被自动化测试稳定捕获；
- 工作树中不存在无关业务代码改动。

## 9. 非目标

- 不实现通用语义层；
- 不修改 Meta MySQL Schema 或 `conf/meta_config.yaml`；
- 不支持 FULL、CROSS、SEMI、ANTI JOIN；
- 不解决多事实表、多对多 Fanout 或 JOIN 前预聚合；
- 不新增 OFFSET 产品能力；
- 不运行或重新设计 110 条评测集。
