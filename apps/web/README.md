# Web Chat Loop (Phase A)

## Run

```bash
cd apps/web
npm install
npm run dev
```

Vite dev server proxies `/api` to `http://localhost:8000` by default.

If your API host is different, set:

```bash
VITE_API_BASE=http://your-api-host:8000
```

If API auth is enabled, also set:

```bash
VITE_API_TOKEN=your-bearer-token
```

Docker compose 下这是 build-time 变量（会注入静态资源构建），请在仓库根目录 `.env` 设置后重新 `docker compose up -d --build`。
仓库默认值是 `local-dev-token`（本地自用）。
前端不再发送 `user_id/operator_id`，身份由 Bearer token 决定。
可在仓库根目录运行 `python scripts/init_local_env.py` 自动生成随机 token 到 `.env`。

## What is included

- Streaming chat loop (`POST /api/chat/stream`, SSE)
- Session management (switch / rename / delete)
- Writing workspace (chapter editor + scene beat context)
- Assistant action loop (`应用并记录 / 拒绝 / 撤销`) with logs
- Tiptap semantic diff layer for polish/expand suggestions (`接受 / 忽略` per change, Git-style inline rendering)
- Ghost Text streaming autocomplete over WebSocket (`Tab` accept all, `Ctrl+ArrowRight` accept one word)
- Context X-Ray (hover/focus assistant replies to inspect per-message `evidence` and project fallback snippets)
- Prompt & knowledge panel (templates + settings/cards injection)

More product-level usage details: `docs/author-manual.md`

## Notes

- `VITE_API_TOKEN` is reused for both HTTP requests and Ghost Text WebSocket auth.
- Because browsers cannot set custom headers for WebSocket handshakes, the frontend sends that token as `?token=` when opening `/api/chat/ghost-text`.
- If your backend uses an `openai_compatible` provider that does not support `json_schema` or tool calling, set `LLM_STRUCTURED_MODE=compat` on the API side. The default remains `strict`.
