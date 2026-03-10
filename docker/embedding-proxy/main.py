"""Embedding Proxy — 主备切换。

主模型失败时自动 fallback 到备用模型，两者使用相同 endpoint 和 API key。
"""
import logging
import os

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding-proxy")

app = FastAPI(title="embedding-proxy")

# ---------------------------------------------------------------------------
# 配置（从环境变量读取）
# ---------------------------------------------------------------------------
EMBEDDING_BASE_URL = os.environ["EMBEDDING_BASE_URL"].rstrip("/")  # e.g. https://router.tumuer.me/v1
EMBEDDING_API_KEY  = os.environ["EMBEDDING_API_KEY"]
PRIMARY_MODEL      = os.environ["PRIMARY_MODEL"]    # Qwen/Qwen3-Embedding-8B
FALLBACK_MODEL     = os.environ["FALLBACK_MODEL"]   # BAAI/bge-m3
TIMEOUT            = float(os.getenv("PROXY_TIMEOUT", "30"))

HEADERS = {
    "Authorization": f"Bearer {EMBEDDING_API_KEY}",
    "Content-Type": "application/json",
}


def _call_embedding(model: str, body: dict) -> dict:
    payload = {**body, "model": model}
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(
            f"{EMBEDDING_BASE_URL}/embeddings",
            json=payload,
            headers=HEADERS,
        )
        resp.raise_for_status()
        return resp.json()


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    # 尝试主模型
    try:
        result = _call_embedding(PRIMARY_MODEL, body)
        return JSONResponse(content=result)
    except Exception as primary_exc:
        logger.warning("primary embedding failed (%s), trying fallback", primary_exc)

    # 主模型失败 → fallback
    try:
        result = _call_embedding(FALLBACK_MODEL, body)
        logger.info("fallback embedding succeeded")
        return JSONResponse(content=result)
    except Exception as fallback_exc:
        logger.error("fallback embedding also failed: %s", fallback_exc)
        raise HTTPException(status_code=502, detail=str(fallback_exc))


@app.get("/health")
def health():
    return {"ok": True}
