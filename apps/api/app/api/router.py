from fastapi import APIRouter

from app.api.endpoints.chat import router as chat_router
from app.api.endpoints.writing import router as writing_router

api_router = APIRouter()
api_router.include_router(chat_router)
api_router.include_router(writing_router)
