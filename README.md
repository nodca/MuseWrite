# Novel Platform

AI 辅助小说写作平台（Monorepo：`apps/api` + `apps/web`）。

> 目标不是“能聊天”，而是让长篇写作在长期迭代中更稳、更快、可回退。

---

## 为什么这个项目值得用

很多写作工具能给灵感，但在长篇里容易出现三类问题：
- 前后设定冲突
- 对话越写越慢
- AI 改动不可控，难以回退

这个项目的设计重点，就是把这三件事做成可持续的工程能力。

### 项目优势（少术语版）

- **信息不容易打架**：回答时先看最可靠、最贴近当前章节的信息，再补充其它线索，尽量减少前后矛盾。
- **速度更稳**：查询会并行进行；外部服务慢时会自动退回本地方案，不至于整段卡住。
- **回答更快**：常用内容会提前整理好，下一轮对话不用从零开始。
- **重复内容不反复计算**：把“长期不变”“阶段常用”“本轮临时”内容分开处理，能省时间也能省成本。
- **改动可控**：AI 只能先提建议，作者确认后才会真正修改；不合适可以拒绝，也可以撤销。
- **迭代有硬标准**：每次调整检索和回答流程，都能通过固定测试看是否变好，避免“改了但更差”。
- **长期一致性可巡检**：支持手动检查 + 每日自动体检，并能查看章节关系变化，便于尽早发现冲突。

---

## 核心能力

### 1) 写作对话与动作闭环
- `POST /api/chat/stream` 实时对话（SSE）
- AI 建议走“提议 -> 应用/拒绝/撤销”
- 全程有审计日志，便于追溯谁在什么时候改了什么

### 2) 章节工作区
- 章节内容保存、历史版本、回滚
- 结构化规划：Volume / Chapter / Scene Beat / Foreshadowing
- 支持 Prompt 模板与知识注入（settings/cards）

### 3) 一致性治理
- 一致性报告支持手动触发
- worker 支持每日自动巡检
- 图谱候选事实审阅、实体合并巡检、章节关系时间线可视化

### 4) 可验证的质量门槛
- API 单元测试 + Web E2E
- 检索相关改动可走 RAGAS 回归门禁（可选）
- CI 可在合并前拦截明显退化

作者使用路径见：[docs/author-manual.md](docs/author-manual.md)

---

## 系统架构（高层）

```text
Web (React/Vite)
   │
   ▼
API (FastAPI)
   ├─ 对话流 / 动作闭环 / 审计
   ├─ 上下文整理与检索编排
   └─ 一致性报告接口
   │
   ├─ PostgreSQL（主数据 + 异步任务）
   ├─ Neo4j（关系事实，quality-first 图检索依赖 GDS）
   └─ LightRAG（可选，语义召回）

Worker
   ├─ 图谱同步任务
   ├─ 一致性巡检任务
   └─ 实体合并与索引生命周期任务
```

---

## 技术栈

- 后端：Python + FastAPI
- 前端：React + Vite + TypeScript
- 数据：PostgreSQL + Neo4j
- 可选召回：LightRAG
- 测试：pytest + Playwright

---

## 快速启动（推荐 Docker）

### 1) 初始化配置

```bash
cp .env.example .env
```

可选：生成随机本地 token（会覆盖 `.env` 中 token 字段）

```bash
python scripts/init_local_env.py --force
```

### 2) 启动服务

```bash
docker compose up -d --build
```

默认启动：`web + api + worker + postgres + neo4j`

说明：
- Docker 默认使用仓库内的 Neo4j GDS 镜像，直接 baked-in 匹配版本的 Graph Data Science（GDS）jar。
- 质量优先的图检索把 GDS 视为硬依赖；API 启动时会主动探测 GDS。
- Neo4j 自身 healthcheck 也会执行 `gds.version()`；若 GDS 缺失或未正确加载，API 会直接报错停止启动，而不是静默降级到低质量图检索。
- 图检索默认使用命名 projection graph（GDS in-memory graph），而不是每次临时投影：
  - 命名规则：`NEO4J_GDS_GRAPH_NAME_PREFIX + project_id + scope + version`
  - 同一 `project + scope + version` 直接复用已有图
  - 图事实写入/删除/状态变更只 bump version，不在写请求内全量重建
  - 下一次图检索按新 version 懒重建，并清理同 scope 的旧版本图

### 交付版（镜像内置 ONNX reranker）

当你要把服务交给别的机器直接部署，而不想再额外挂载 `apps/api/models` 时，使用交付版：

```bash
docker compose -f docker-compose.delivery.yml up -d --build
```

说明：
- 交付版会把 `apps/api/models` 中的 ONNX reranker 一并打进镜像。
- Neo4j 同样使用 baked-in GDS 镜像，不依赖容器启动时在线拉插件。
- `api` 与 `worker` 复用同一份 baked image，避免重复构建同一个大镜像。
- 这更适合“交付/封版/离线部署”，不适合频繁本地迭代。
- 若模型目录很大，首次构建时间和磁盘占用都会明显上升，这是正常现象。

### 3) 验证服务

- Web: `http://localhost:8080`
- API Health: `http://localhost:8000/health`
- API Docs: `http://localhost:8000/docs`
- Neo4j Browser: `http://localhost:7474`

### 4) 启用 LightRAG（可选）

```bash
docker compose --profile rag up -d
```

并确保 `.env` 中：
- `LIGHTRAG_ENABLED=true`
- `LIGHTRAG_BASE_URL=http://lightrag:9621`
- `LIGHTRAG_QUERY_PATH=/query/data`

---

## 本地开发（Hybrid）

推荐先用 Docker 拉起依赖，再本地启动 API/Web。

如果只是本地开发，优先使用默认 `docker-compose.yml`：
- 默认通过 bind mount 挂载 `./apps/api/models:/app/models:ro`
- 不会把大体积 ONNX 模型打进镜像
- 更适合频繁 rebuild 和调试

### 启动依赖

```bash
docker compose up -d postgres neo4j
```

### API

```bash
cd apps/api
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

另开终端启动 worker：

```bash
cd apps/api
python -m app.worker
```

### Web

```bash
cd apps/web
npm install
npm run dev
```

默认端口 `5173`，`/api` 代理到 `http://localhost:8000`

---

## 配置哲学（默认即最佳实践）

配置已从“全量可配”收敛为“**profile 默认 + 少量策略开关**”：

- **核心必需层**：连接与身份（数据库、鉴权、主模型、LightRAG/Neo4j 开关）。
- **策略开关层**：仅保留少数需要产品/工程协商的行为决策。
- **实现细节层**：timeout/retry/top_k/bias/cache 等默认内化到后端 profile，不建议直接调整。

这意味着：
- 新用户可先用默认值直接跑起来；
- 运维能快速识别“必须改什么”。

### 最低可运行配置

```env
CONFIG_PROFILE=local-dev
AUTH_ENABLED=true
AUTH_TOKENS=local-user:local-dev-token

LLM_PROVIDER=stub
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=
LLM_STRUCTURED_MODE=strict

LIGHTRAG_ENABLED=false
LIGHTRAG_BASE_URL=http://lightrag:9621

NEO4J_ENABLED=true
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=neo4jpassword
NEO4J_DATABASE=neo4j
NEO4J_GDS_GRAPH_NAME_PREFIX=novel_ppr
NEO4J_GDS_REQUIRED=true
```

### 可商榷策略开关

```env
RAG_ROUTE_POLICY=mix
QUALITY_GATE_ENFORCE=false
CITATION_POLICY=off
CITATION_MIN_COUNT=1
CONTEXT_WINDOW_PROFILE=balanced
CONTEXT_COMPRESSION_MODE=rerank
```

### 三个典型 profile 场景

- **本地开发（默认）**：`CONFIG_PROFILE=local-dev`
  - 更快反馈，更低资源占用，适合日常迭代。
- **质量优先**：`CONFIG_PROFILE=quality-first`
  - 更严格的质量门控与证据策略，适合回归验证与关键里程碑。

### 鉴权注意事项

- API 需要 `Authorization: Bearer <token>`。
- 前后端 token 必须一致，否则会 `401`：
  - `AUTH_TOKENS=local-user:<token>`
  - `VITE_API_TOKEN=<token>`
- 默认示例 token `local-dev-token` 仅限本地开发使用。
- 浏览器 WebSocket 不能自定义 `Authorization` header；Ghost Text 流式补全会把同一个 `VITE_API_TOKEN` 挂到 `?token=` 查询参数上，API 端会等价校验。

### Structured Outputs 与 Ghost Text 约定

- `LLM_STRUCTURED_MODE` 默认是 `strict`：优先走 `json_schema` / tool calling，拿不到结构化响应就快速失败，不再做正则抠 JSON。
- 只有显式设置为 `compat` 时，`openai_compatible`/兼容 OpenAI 协议但不支持 `json_schema`、tool calling 的供应商，才允许降级到 `response_format={"type":"json_object"}`。
- 即使在 `compat` 模式下，服务端也只接受“纯 JSON”或“单个 fenced JSON block”，再交给 Pydantic 校验，不接受自由文本混排 JSON。
- Ghost Text 自动补全当前走 WebSocket 流式推送；浏览器侧在停顿后请求建议，`Tab` 接受整段，`Ctrl+ArrowRight` 接受一个词。
- Tiptap 润色/扩写不会直接覆盖正文，而是渲染 Git-style Diff：删除为红色删除线，新增为绿色高亮，每条修改都可单独“接受/忽略”。

---

## 测试与质量门禁

### API Unit Tests

```bash
python -m unittest discover -s apps/api/tests -p "test_*_unittest.py"
```

### Web E2E

```bash
cd apps/web
npm run test:e2e
```

### RAGAS 回归 Gate（可选）

```bash
pip install -r apps/api/requirements.txt -r apps/api/evals/requirements.txt
python apps/api/evals/prepare_fixture_docs.py --base-url http://localhost:8000 --project-id 1 --fixture-docs apps/api/evals/data/novel_rag_docs.jsonl --reset-existing
python apps/api/evals/run_ragas_gate.py --base-url http://localhost:8000 --dataset apps/api/evals/data/novel_rag_eval.jsonl --threshold 0.55 --rag-mode mix --baseline-report apps/api/evals/data/novel_rag_baseline.json --output apps/api/evals/out/ragas-report.json
```

CI 工作流：
- `.github/workflows/test-gate.yml`
- `.github/workflows/ragas-regression-gate.yml`

---

## 常用 API

为避免主文档与接口文档重复维护，API 列表与请求示例统一维护在：
- `apps/api/README.md`

---

## 目录结构

```text
.
├─ apps/
│  ├─ api/              # FastAPI + worker + tests + evals
│  └─ web/              # React/Vite + Playwright e2e
├─ docs/                # 使用手册与方案文档
├─ scripts/             # 仓库级脚本（如 init_local_env.py）
├─ docker-compose.yml
└─ .env.example
```

---

## 更多文档

- [作者使用手册](docs/author-manual.md)
- [API 说明](apps/api/README.md)
- [Web 说明](apps/web/README.md)

---

## License

GNU Affero General Public License v3.0 (AGPLv3)
