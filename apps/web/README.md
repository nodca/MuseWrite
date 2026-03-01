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
- Prompt & knowledge panel (templates + settings/cards injection)

More product-level usage details: `docs/author-manual.md`
