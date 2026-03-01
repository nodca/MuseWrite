from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings
from app.core.database import init_db

logger = logging.getLogger("novel_platform.security")


def _using_default_dev_token() -> bool:
    for chunk in str(settings.auth_tokens or "").split(","):
        item = chunk.strip()
        if not item or ":" not in item:
            continue
        _, token = item.split(":", 1)
        if token.strip() == "local-dev-token":
            return True
    return str(settings.auth_token or "").strip() == "local-dev-token"


def _emit_startup_security_notice() -> None:
    if not settings.auth_enabled:
        logger.warning(
            "SECURITY WARNING: AUTH_ENABLED=false. This deployment is local-only and must not be exposed publicly."
        )
    if _using_default_dev_token():
        logger.warning(
            "SECURITY WARNING: default token 'local-dev-token' is active. Replace it before any non-local exposure."
        )
    if not str(settings.auth_project_owners or "").strip():
        logger.warning(
            "SECURITY NOTICE: AUTH_PROJECT_OWNERS is empty (single-user mode). Do not treat this as multi-tenant isolation."
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    _emit_startup_security_notice()
    yield


app = FastAPI(title="novel-platform-api", lifespan=lifespan)
app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/health")
def health():
    return {"ok": True}
