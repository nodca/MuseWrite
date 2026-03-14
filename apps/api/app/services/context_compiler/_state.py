import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

from app.services.context_compiler._types import ContextPack


_CONTEXT_PACK_CACHE: dict[int, ContextPack] = {}
_CONTEXT_PACK_LOCK = Lock()
_GRAPH_HITS_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_RAG_HITS_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_WEB_SEARCH_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_RETRIEVAL_CACHE_LOCK = Lock()
_RETRIEVAL_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retrieval")
_LOGGER = logging.getLogger(__name__)
_RETRIEVAL_CIRCUIT_BREAKER_LOCK = Lock()
_RETRIEVAL_CIRCUIT_BREAKERS: dict[str, dict[str, Any]] = {
    "graph": {"failures": deque(), "open_until": 0.0, "opened_count": 0},
    "rag": {"failures": deque(), "open_until": 0.0, "opened_count": 0},
    "web_search": {"failures": deque(), "open_until": 0.0, "opened_count": 0},
}
