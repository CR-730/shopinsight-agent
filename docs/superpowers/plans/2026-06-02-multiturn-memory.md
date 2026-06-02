# 多轮问数记忆实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为电商问数 Agent 增加轻量持久化会话、追问改写和会话快照写回能力。

**架构：** 在 Meta MySQL 中新增会话表，QueryService 在进入原有 LangGraph 前读取快照并改写追问，执行结束后从最终 state 写回 turn 和 snapshot。原有 RAG、业务绑定、SQL 校验和执行节点保持不变。

**技术栈：** FastAPI、LangGraph、SQLAlchemy AsyncSession、MySQL JSON、pytest、ruff。

---

## 文件结构

- 创建 `app/agent/memory.py`：会话快照类型、结果摘要和快照构建。
- 创建 `app/agent/rewrite.py`：基于 fast LLM 的结构化追问改写，输出 `mode`、`standalone_query`、`reason`、`inherited_slots` 和 `overridden_slots`。
- 创建 `app/repositories/mysql/meta/conversation_memory_repository.py`：会话表建表、创建会话、读取快照、保存轮次和更新快照。
- 修改 `app/services/query_service.py`：接收 conversation 参数，流式执行时收集最终 state，写入会话记忆。
- 修改 `app/api/schemas/query_schema.py`：增加 `conversation_id` 和 `user_id`。
- 修改 `app/api/routers/query_router.py`：透传新增字段。
- 修改 `docker/mysql/meta.sql`：新增会话相关 DDL。
- 创建 `tests/test_conversation_memory.py`：覆盖纯逻辑和仓储 SQL 行为。
- 创建 `examples/conversation_eval_cases.yaml`：增加多轮问数评测样例。

## 任务

### 任务 1：文档和纯逻辑测试

**文件：**
- 创建：`docs/superpowers/specs/2026-06-02-multiturn-memory-design.md`
- 创建：`docs/superpowers/plans/2026-06-02-multiturn-memory.md`
- 创建：`tests/test_conversation_memory.py`
- 创建：`app/agent/memory.py`

- [ ] **步骤 1：编写失败的测试**

运行：`uv run pytest tests/test_conversation_memory.py -q`
预期：FAIL，原因是 `app.agent.memory` 尚不存在。

- [ ] **步骤 2：实现最小纯逻辑**

实现 `build_answer_summary`、`build_snapshot_from_state`，并用结构化 LLM rewrite 取代规则型追问拼接。

- [ ] **步骤 3：运行测试验证通过**

运行：`uv run pytest tests/test_conversation_memory.py -q`
预期：PASS。

### 任务 2：会话仓储

**文件：**
- 创建：`app/repositories/mysql/meta/conversation_memory_repository.py`
- 修改：`docker/mysql/meta.sql`
- 测试：`tests/test_conversation_memory.py`

- [ ] **步骤 1：编写失败的仓储测试**

测试 SQL 文本包含三张表的 `create table if not exists`，并测试会话 id 自动生成。

- [ ] **步骤 2：实现仓储**

实现 `ensure_tables`、`create_conversation`、`get_snapshot`、`save_turn`、`upsert_snapshot`。

- [ ] **步骤 3：运行测试验证通过**

运行：`uv run pytest tests/test_conversation_memory.py -q`
预期：PASS。

### 任务 3：QueryService 接入

**文件：**
- 修改：`app/services/query_service.py`
- 修改：`app/api/schemas/query_schema.py`
- 修改：`app/api/routers/query_router.py`
- 测试：`tests/test_conversation_memory.py`

- [ ] **步骤 1：编写失败的服务层测试**

测试 `/api/query` 请求 schema 接受 `conversation_id`，并测试服务层在 `stream_mode=["custom","values"]` 下收集最终 state。

- [ ] **步骤 2：实现服务层接入**

QueryService 先创建/读取会话，发出 conversation SSE 事件，执行图，保存 turn 和 snapshot。

- [ ] **步骤 3：运行测试验证通过**

运行：`uv run pytest tests/test_conversation_memory.py -q`
预期：PASS。

### 任务 4：conversation eval 样例

**文件：**
- 创建：`examples/conversation_eval_cases.yaml`

- [ ] **步骤 1：写入多轮样例**

覆盖省略指标、换时间、换过滤条件、blocked 轮次不覆盖快照。

- [ ] **步骤 2：运行质量测试**

运行：`uv run pytest tests/test_eval_cases_content.py tests/test_conversation_memory.py -q`
预期：PASS。

### 任务 5：整体验证

**文件：**
- 全项目

- [ ] **步骤 1：运行单元测试**

运行：`uv run pytest -q`
预期：全部通过。

- [ ] **步骤 2：运行 lint**

运行：`uv run ruff check .`
预期：All checks passed。
