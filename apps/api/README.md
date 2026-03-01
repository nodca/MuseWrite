# API (Phase A-1)

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# 另开一个终端启动图谱 worker（异步入图）
python -m app.worker
```

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
  - GRAPH fact query (Neo4j adapter with local fallback)
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
