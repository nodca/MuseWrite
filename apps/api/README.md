# API (Phase A-1)

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# 另开一个终端启动图谱 worker（异步入图）
python -m app.worker
```

## Docker 运行模式

本目录现在有两种 Docker 形态：

### 1) 本地开发版

- 使用仓库根的 `docker-compose.yml`
- 默认通过 volume 挂载宿主机 ONNX 模型目录：`./apps/api/models:/app/models:ro`
- `Dockerfile` 默认不做本地 ONNX 导出，也不把 `models/` 打进镜像
- 优点：CPU 场景更友好、构建更轻、更适合反复改代码

### 2) 交付版

- 使用仓库根的 `docker-compose.delivery.yml`
- 使用 `apps/api/Dockerfile.delivery`
- 直接把 `apps/api/models` baked 进镜像，适合离线交付或单机封版部署
- `api` 与 `worker` 复用同一份 delivery image，避免重复构建同一个大镜像

交付版启动命令：

```bash
docker compose -f docker-compose.delivery.yml up -d --build
```

注意：
- 若 `apps/api/models` 很大，首次 build 会明显更慢，也更吃磁盘。
- 交付版的目标是“部署方便”，不是“开发迭代最快”。

## Minimum runtime env (converged)

```bash
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/novel_platform
CONFIG_PROFILE=local-dev  # local-dev | quality-first

AUTH_ENABLED=true
AUTH_TOKENS=local-user:local-dev-token
AUTH_PROJECT_OWNERS=

LLM_PROVIDER=stub
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=

LIGHTRAG_ENABLED=false
LIGHTRAG_BASE_URL=http://localhost:9621

NEO4J_ENABLED=false
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=neo4j
NEO4J_DATABASE=neo4j
NEO4J_GDS_GRAPH_NAME_PREFIX=novel_ppr
NEO4J_GDS_REQUIRED=true
```

## Optional strategy toggles (recommended editable surface)

```bash
RAG_ROUTE_POLICY=mix
QUALITY_GATE_ENFORCE=false
CITATION_POLICY=off
CITATION_MIN_COUNT=1
CONTEXT_WINDOW_PROFILE=balanced
CONTEXT_COMPRESSION_MODE=rerank

LANGFUSE_ENABLED=false
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

## Configuration philosophy

- 默认即最佳实践：大部分实现细节参数（timeouts/retries/top_k/cache/bias）已经内化为 profile 默认值。
- 只建议调整策略层：上面“Optional strategy toggles”中的变量。
- 向后兼容：历史细粒度变量仍可显式覆盖，便于平滑迁移与 CI 注入。

Typical profile choices:
- `local-dev`: 本地优先速度与反馈。
- `quality-first`: 质量与证据优先（更严格）。
  - Neo4j 图检索要求 GDS 可用；缺失时 API 启动直接失败，不做静默降级。

Tip:
- From repo root, you can run `python scripts/init_local_env.py` to generate a random local token and write `.env`.
- Script now also ensures `CONFIG_PROFILE=local-dev` for one-command local startup.

Auth note:
- API now requires `Authorization: Bearer <token>`.
- `AUTH_PROJECT_OWNERS` can stay empty for local single-user mode.
- server identity comes from Bearer token only; request body no longer accepts `user_id/operator_id`.
- 默认示例 token 为 `local-dev-token`，仅适用于本地自用。
- startup prints `SECURITY WARNING` when unsafe local defaults are detected.

LightRAG 官方 v1.4+ 建议使用 `/query/data` 契约（`status/message/data/metadata`），本项目已优先适配。

Neo4j 运行说明：
- Docker Compose 默认使用仓库内的 Neo4j GDS 镜像，直接 baked-in 匹配版本的 Graph Data Science（GDS）jar。
- 当 `NEO4J_ENABLED=true` 且 `NEO4J_GDS_REQUIRED=true` 时，API 启动阶段会执行一次 GDS 能力探测。
- Neo4j healthcheck 也会执行 `gds.version()`；若 GDS 未安装、未加载或不可调用，启动会直接抛错，避免图检索悄悄退化为低质量模式。
- 图检索使用命名 projection graph（GDS in-memory graph）并带版本号：
  - 命名规则：`NEO4J_GDS_GRAPH_NAME_PREFIX + project_id + scope + version`
  - 同一 `project + scope + version` 直接复用已有图
  - 图事实写入/删除/状态变更只 bump version，不在写请求内重建
  - 下一次检索触发懒重建，并清理同 scope 的旧版本图
  - `scope` 由是否启用时间过滤与 `current_chapter` 决定

`POST /api/chat/stream` now uses structured model output:

```json
{
  "assistant_text": "给用户的话",
  "proposed_actions": [
    {"action_type": "setting.upsert", "payload": {"key": "x", "value": {}}}
  ]
}
```

Prompt design (NovelForge-inspired, adapted):

- Assistant role is a writing copilot (not pipeline auto-writer).
- Output must be strict JSON with:
  - `assistant_text`
  - `proposed_actions`
- `proposed_actions` are **suggestions only** and require explicit user confirm via action APIs.
- Model receives compact workspace context:
  - recent chat messages
  - settings preview
  - cards preview
- Three-way evidence compiler:
  - DSL exact hit
  - GRAPH fact query (Neo4j adapter, quality-first 场景要求 GDS)
  - semantic recall (LightRAG adapter with local fallback)
- Action suggestions are validated server-side:
  - only allowed action types
  - payload schema checks
  - max 3 actions per response

## Endpoints

- `POST /api/chat/stream`
- `POST /api/chat/ghost-text`
- `GET /api/chat/projects/{project_id}/sessions`
- `PUT /api/chat/projects/{project_id}/sessions/{session_id}`
- `DELETE /api/chat/projects/{project_id}/sessions/{session_id}`
- `GET /api/chat/sessions/{session_id}/messages`
- `GET /api/chat/sessions/{session_id}/actions`
- `POST /api/chat/sessions/{session_id}/actions`
- `POST /api/chat/actions/{action_id}/apply`
- `POST /api/chat/actions/{action_id}/reject`
- `POST /api/chat/actions/{action_id}/undo`
- `GET /api/chat/actions/{action_id}/logs`
- `GET /api/chat/projects/{project_id}/settings`
- `GET /api/chat/projects/{project_id}/cards`
- `GET /api/chat/projects/{project_id}/prompt-templates`
- `POST /api/chat/projects/{project_id}/prompt-templates`
- `PUT /api/chat/projects/{project_id}/prompt-templates/{template_id}`
- `DELETE /api/chat/projects/{project_id}/prompt-templates/{template_id}`
- `GET /api/chat/projects/{project_id}/prompt-templates/{template_id}/revisions`
- `POST /api/chat/projects/{project_id}/prompt-templates/{template_id}/rollback`
- `GET /api/chat/projects/{project_id}/volumes`
- `POST /api/chat/projects/{project_id}/volumes`
- `PUT /api/chat/projects/{project_id}/volumes/{volume_id}`
- `DELETE /api/chat/projects/{project_id}/volumes/{volume_id}`
- `GET /api/chat/projects/{project_id}/chapters/{chapter_id}/scene-beats`
- `POST /api/chat/projects/{project_id}/chapters/{chapter_id}/scene-beats`
- `PUT /api/chat/projects/{project_id}/chapters/{chapter_id}/scene-beats/{beat_id}`
- `DELETE /api/chat/projects/{project_id}/chapters/{chapter_id}/scene-beats/{beat_id}`
- `GET /api/chat/projects/{project_id}/foreshadowing-cards`
- `POST /api/chat/projects/{project_id}/foreshadowing-cards`
- `PUT /api/chat/projects/{project_id}/foreshadowing-cards/{card_id}`
- `DELETE /api/chat/projects/{project_id}/foreshadowing-cards/{card_id}`
- `GET /api/chat/projects/{project_id}/chapters/{chapter_id}/revisions?semantic=true`
- `GET /health`

## `POST /api/chat/stream` example

```json
{
  "project_id": 1,
  "session_id": null,
  "content": "帮我续写这一段冲突戏",
  "model": "stub-model"
}
```

SSE event types:

- `meta`
- `evidence`
- `delta`
- `done`
- `error`

Note: `evidence` is persisted per assistant message as `context_xray` and is returned by `GET /api/chat/sessions/{session_id}/messages`.

`POST /api/chat/stream` supports POV sandbox fields:

```json
{
  "pov_mode": "global|character",
  "pov_anchor": "角色名(可空)"
}
```

`POST /api/chat/stream` also supports prompt workshop field:

```json
{
  "prompt_template_id": 1
}
```

`POST /api/chat/stream` and `POST /api/chat/ghost-text` both support outline constraints:

```json
{
  "chapter_id": 12,
  "scene_beat_id": 3
}
```

`POST /api/chat/stream` supports runtime options:

```json
{
  "thinking_enabled": true,
  "reference_project_ids": [2, 3]
}
```

Legacy draft table cleanup script:

```bash
python scripts/drop_legacy_project_draft_tables.py
```

## Action decision payload

```json
{
  "event_payload": {
    "reason": "looks good"
  }
}
```

## Create action payload

```json
{
  "action_type": "setting.upsert",
  "payload": {
    "key": "世界观/货币体系",
    "value": {
      "currency": "银鹿币",
      "rate": "1金狮=10银鹿"
    }
  },
  "idempotency_key": "sess-1-setting-currency-v1"
}
```
