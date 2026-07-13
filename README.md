<div align="center">

# ShopInsight Agent

面向电商数据仓库的自然语言问数智能体

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

ShopInsight Agent 将自然语言问题转换为可执行 SQL，通过元数据检索、业务语义绑定、安全校验、自动纠错和流式输出，完成从业务提问到数据结果的完整链路。项目提供 FastAPI 后端、LangGraph Agent、React 聊天界面，以及可直接初始化的电商教学数仓。

![ShopInsight Agent 查询示例](docs/images/example.png)

## 核心能力

- **自然语言问数：** 支持销售额、订单量、客单价等电商指标查询，并识别地区、时间、品类和品牌等分析维度。
- **混合元数据检索：** 使用 Qdrant 召回字段和指标，使用 Elasticsearch 与向量检索协同召回字段取值，使用 MySQL 保存权威元数据。
- **业务语义绑定：** 对时间范围、指标口径、枚举别名和候选取值进行解析；语义不完整时返回澄清信息，不盲目生成 SQL。
- **SQL 安全闭环：** 生成 SQL 后执行只读约束、敏感字段检查、`EXPLAIN` 校验、限次纠错和超时控制。
- **多轮会话记忆：** 保存会话与消息，在追问中继承必要上下文，同时控制记忆边界。
- **可观测流式交互：** 通过 Server-Sent Events（SSE）返回思考状态、召回说明、SQL、查询结果、耗时和错误信息。
- **评测与成本治理：** 内置回归、消融和多轮记忆评测，并记录模型 Token、调用预算和估算成本。

## 系统链路

```text
React 前端
    │  POST /api/query（SSE）
    ▼
FastAPI / QueryService
    │
    ▼
意图识别 → 上下文构建 → 业务绑定 → 上下文压缩 → SQL 生成 → SQL 执行
              │              │                         │
              ├─ Qdrant      ├─ 时间/指标/枚举绑定     ├─ SQL Guard
              ├─ Elasticsearch                          ├─ EXPLAIN
              └─ Meta MySQL                             └─ 自动纠错
                                                            │
                                                            ▼
                                                       DW MySQL
```

LangGraph 只保留产品级节点。关键词扩展、混合召回、上下文裁剪、SQL 校验与纠错等细节由节点内部模块承担，使主流程保持清晰。

## 技术栈

| 层级 | 技术 | 职责 |
| --- | --- | --- |
| 前端 | React 19、TypeScript、Vite、Tailwind CSS | 会话管理、Markdown/SQL 展示、结果表格和 SSE 状态流 |
| API | FastAPI、Pydantic | 问数接口、会话接口、依赖注入和生命周期管理 |
| Agent | LangGraph、LangChain | 状态编排、模型调用、业务绑定、上下文管理和 SQL 自动纠错 |
| 数据访问 | SQLAlchemy、asyncmy | 异步访问元数据库与电商数仓，执行 `EXPLAIN` 和查询 |
| 检索 | Qdrant、Elasticsearch、Jieba | 字段、指标和字段取值的语义/关键词召回 |
| SQL | sqlglot | SQL 解析与静态安全约束 |
| 工程 | uv、pytest、Ruff、pnpm | 依赖管理、测试、静态检查和前端构建 |

## 项目结构

```text
.
├── app/
│   ├── agent/                 # LangGraph、状态、记忆、成本治理与 SQL 闭环
│   │   ├── business_binding/  # 时间、指标和候选取值绑定
│   │   ├── nodes/             # 产品级 Agent 节点
│   │   └── sql/               # SQL 生成后的守卫、校验、纠错和执行
│   ├── api/                   # FastAPI 路由、Schema、依赖和生命周期
│   ├── clients/               # MySQL、Qdrant、ES、Embedding 客户端管理
│   ├── evaluation/            # 评测用例加载和结果判定
│   ├── repositories/          # MySQL、Qdrant、Elasticsearch 仓储实现
│   ├── scripts/               # 知识库构建、评测与真实 SQL 冒烟脚本
│   └── services/              # 查询服务与元数据知识服务
├── conf/                      # 应用、元数据和安全策略配置
├── docker/                    # MySQL、Qdrant、Elasticsearch、Kibana
├── examples/                  # 评测集与 Qdrant 示例
├── frontend/                  # React 问数前端
├── prompts/                   # 意图、召回过滤、SQL 与结果分析 Prompt
├── tests/                     # 单元测试和回归测试
├── main.py                    # FastAPI 应用入口
└── pyproject.toml             # Python 依赖与工具配置
```

## 快速开始

### 1. 环境要求

- Python `>= 3.14`
- [uv](https://docs.astral.sh/uv/)
- Docker 与 Docker Compose
- Node.js `>= 20`
- pnpm `10.x`
- 一个兼容 OpenAI API 的大模型与 Embedding 服务

### 2. 安装依赖

```bash
git clone https://github.com/CR-730/shopinsight-agent.git
cd shopinsight-agent
uv sync
cd frontend
pnpm install
cd ..
```

### 3. 配置模型

在项目根目录新建 `.env`。下列变量是当前配置加载器要求的最小集合：

```dotenv
LLM_PROVIDER=openai
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_API_KEY=your_api_key
LLM_MODEL=your_chat_model
EMBEDDING_MODEL=your_embedding_model

LLM_TIMEOUT_SECONDS=60
LLM_STRUCTURED_ENABLE_THINKING=false
LLM_GENERATE_SQL_ENABLE_THINKING=false
LLM_CORRECT_SQL_ENABLE_THINKING=false
LLM_INPUT_PER_1M_TOKENS=0
LLM_OUTPUT_PER_1M_TOKENS=0
```

可选的快速模型、重试、并发、熔断和单请求调用预算配置位于 [`app/conf/app_config.py`](app/conf/app_config.py)。数据库、检索服务、超时和混合召回权重位于 [`conf/app_config.yaml`](conf/app_config.yaml)。

> `.env` 已被 Git 忽略，请勿提交真实 API Key。

### 4. 启动基础设施

```bash
docker compose -f docker/docker-compose.yaml up -d
docker compose -f docker/docker-compose.yaml ps
```

默认端口：

| 服务 | 地址 |
| --- | --- |
| MySQL | `localhost:3307` |
| Elasticsearch | `http://localhost:9200` |
| Kibana | `http://localhost:5601` |
| Qdrant | `http://localhost:6333` |

MySQL 容器首次启动时会执行 [`docker/mysql/meta.sql`](docker/mysql/meta.sql) 和 [`docker/mysql/dw.sql`](docker/mysql/dw.sql)，分别初始化元数据库和电商教学数仓。

### 5. 构建元数据知识库

```bash
uv run python -m app.scripts.build_meta_knowledge -c conf/meta_config.yaml
```

该命令会同步表、字段、指标与别名，并构建 Qdrant 和 Elasticsearch 检索数据。调整数仓结构或 [`conf/meta_config.yaml`](conf/meta_config.yaml) 后应重新执行。

### 6. 启动后端

```bash
uv run fastapi dev main.py
```

服务默认运行于 `http://127.0.0.1:8000`，OpenAPI 页面位于 `http://127.0.0.1:8000/docs`。

### 7. 启动前端

```bash
cd frontend
pnpm dev
```

浏览器访问 `http://127.0.0.1:5173`。开发服务器默认将 `/api` 代理到 `http://127.0.0.1:8000`；如需修改，可复制 [`frontend/.env.example`](frontend/.env.example) 为 `frontend/.env` 并调整 `VITE_DEV_PROXY_TARGET`。

## API 概览

### 流式问数

```http
POST /api/query
Content-Type: application/json
```

```json
{
  "query": "统计华北地区 2025 年第一季度的销售总额",
  "conversation_id": null,
  "user_id": "anonymous"
}
```

响应类型为 `text/event-stream`。前端按事件逐步展示 Agent 状态、SQL、查询结果或错误。

### 会话管理

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/conversations?user_id=anonymous` | 获取会话列表 |
| `GET` | `/api/conversations/{conversation_id}?user_id=anonymous` | 获取会话详情 |
| `DELETE` | `/api/conversations/{conversation_id}?user_id=anonymous` | 删除会话 |

## 测试与评测

配置好 `.env` 后，运行以单元测试和回归测试为主的测试集。多数用例不会实际访问外部服务：

```bash
uv run pytest
```

检查后端代码规范与前端构建：

```bash
uv run ruff check .
cd frontend
pnpm build
```

基础设施和模型服务可用后，可运行真实链路评测：

```bash
uv run python -m app.scripts.run_eval -c examples/eval_cases.yaml
uv run python -m app.scripts.run_ablation_eval -c examples/eval_cases_110.yaml
uv run python -m app.scripts.run_sql_memory_smoke
```

评测覆盖召回、上下文过滤、SQL 生成与校验、安全策略、多轮记忆、时延、Token 使用量和消融对比。部分历史评测报告位于 [`docs/`](docs/)。

## 配置入口

| 文件 | 用途 |
| --- | --- |
| [`conf/app_config.yaml`](conf/app_config.yaml) | 数据库、检索服务、日志、超时、召回权重和后台构建策略 |
| [`conf/meta_config.yaml`](conf/meta_config.yaml) | 电商表、字段、指标、别名和同步范围 |
| [`conf/policy_config.yaml`](conf/policy_config.yaml) | Prompt 注入、危险 SQL、敏感字段和语义规则 |
| [`prompts/`](prompts/) | 各阶段的大模型 Prompt 模板 |
| [`.env`](.gitignore) | 模型端点、模型名称、API Key、价格与韧性策略 |

## 能力边界

当前仓库适合本地开发、课程实践和问数链路验证。用于生产环境前仍需结合实际组织补充：

- 身份认证、角色权限、行列级数据权限与多租户隔离；
- 凭据托管、网络隔离、审计日志与敏感数据脱敏；
- 查询资源配额、缓存、限流、监控告警与高可用部署；
- 与企业指标平台、数据目录和数据血缘系统的持续同步；
- 针对真实业务口径维护的评测集及发布门禁。

## 许可证

本项目基于 [MIT License](LICENSE) 开源。
