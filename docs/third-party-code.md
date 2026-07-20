# 第三方代码审计与复用记录

本文记录语义规划重构中审查或适配的第三方源码。只有明确列为“已适配”的代码才进入 ShopInsight；仅借鉴设计的条目不构成源码复制。

## WrenAI Schema Indexer

- 上游仓库：`https://github.com/Canner/WrenAI`
- 锁定 commit：`3dac00a178aa5e78e9d6472fc6e048d17d1f7271`
- 审查文件：`core/wren/src/wren/memory/schema_indexer.py`
- 精确范围：`extract_schema_items()`、`_relationship_record()`、`_measure_record()`
- 路径许可证：Apache License 2.0；WrenAI 为按路径多许可证仓库，本记录不把整个仓库视为 Apache-2.0。
- NOTICE：实际复制前必须再次检查锁定 commit 的根 `NOTICE`、路径内许可证声明和源文件版权头。
- 当前状态：已局部适配 Record Builder 的结构；未复制 Wren 运行时或索引实现。
- ShopInsight 目标：`app/agent/semantic_planning/catalog.py::_build_authoritative_metric_candidates()`。
- 修改说明：将 Wren 面向 MDL manifest 的通用字典记录，改写为 ShopInsight 冻结的 `MetricCandidate`；只接受本轮 `sql_context` 暴露的指标，并从 Meta MySQL `MetricInfo` 回填权威 ID、聚合、表达式、依赖字段和别名。存储、时间戳、统一 schema_items collection 均未采用。
- 计划复用边界：阶段 2 候选目录可以适配 `_measure_record()` / `_relationship_record()` 的“小型纯数据转换”结构，把检索文本与 `item_type`、权威 ID、聚合、表达式、依赖字段和元数据版本并存。不得复制 Wren 的 MDL 运行时、统一索引存储层或自由 SQL relationship condition。
- 不采用：`manifest_hash()`。ShopInsight 现有 `build_metadata_cache_version()` 同时覆盖 active build 和 Meta MySQL 内容，语义更完整，不应退回截断 16 位 Hash。

适配源码已保留如下来源注释：

```python
# Adapted from Canner/WrenAI:
# core/wren/src/wren/memory/schema_indexer.py
# commit 3dac00a178aa5e78e9d6472fc6e048d17d1f7271
# Licensed under Apache License 2.0.
# Modified for ShopInsight typed semantic candidates.
```

若后续继续适配关系 Record Builder，必须另行记录目标函数和修改范围。

## CHESS Schema Generator

- 上游仓库：`https://github.com/ShayanTalaei/CHESS`
- 锁定 commit：`3d6e835f858d26885d21d4bc0215aeecf855efbe`
- 审查文件：`src/database_utils/schema_generator.py`
- 精确范围：`get_schema_with_connections()` 及其连接字段收集逻辑
- 许可证：Apache License 2.0。
- 当前状态：只借鉴设计，不复制源码。
- 借鉴点：业务字段选择完成后，必须从权威 Schema 关系补回 JOIN 两侧字段。
- 不复制原因：CHESS 会补入已选表的多组连接字段，不计算本次查询唯一最短连接路径，也不能区分无路径、等长多路径和环。ShopInsight 当前 `find_unique_shortest_join_closure()` 更严格，直接复制会造成能力回退。
- 范围限制：不复制 CHESS 数据集、模型权重或生成产物。

## 禁止来源

- 不复制旧版 WrenAI Agent 的 AGPL 实现。
- 不从许可证不明的博客、回答或代码片段复制。
- WrenAI 文档若被引用，按其 CC BY 4.0 路径许可证署名，不作为源码复制。

本记录用于工程合规追踪，不构成法律意见。商业分发前仍需核对上游完整许可证和 NOTICE。
