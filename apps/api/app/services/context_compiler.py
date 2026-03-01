import logging
import hashlib
import json
import math
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable

import httpx
from sqlmodel import Session

from app.core.config import settings
from app.services.chat_service import (
    get_project_chapter,
    get_project_volume,
    get_prompt_template,
    list_cards,
    list_scene_beats,
    list_messages,
    list_settings,
)
from app.services.retrieval_adapters import (
    fetch_lightrag_semantic_hits,
    fetch_neo4j_graph_facts,
)


@dataclass
class CompiledContextBundle:
    model_context: dict[str, Any]
    evidence_event: dict[str, Any]


@dataclass(frozen=True)
class SettingSnapshot:
    id: int
    project_id: int
    key: str
    value: Any
    aliases: list[str]
    value_text: str
    updated_at: datetime | None


@dataclass(frozen=True)
class CardSnapshot:
    id: int
    project_id: int
    title: str
    content: Any
    aliases: list[str]
    content_text: str
    updated_at: datetime | None


@dataclass
class ContextPack:
    project_id: int
    generated_at: float
    settings_rows: list[SettingSnapshot]
    cards_rows: list[CardSnapshot]


@dataclass(frozen=True)
class ContextWindowPolicy:
    profile: str
    recent_messages_limit: int
    retrieval_settings_limit: int
    retrieval_cards_limit: int
    model_settings_limit: int
    model_cards_limit: int
    chapter_content_chars: int
    chapter_preview_chars: int


@dataclass(frozen=True)
class BudgetWeights:
    dsl: float
    graph: float
    rag: float
    history: float


@dataclass(frozen=True)
class DynamicBudgetPlan:
    mode: str
    source: str
    confidence: float
    weights: BudgetWeights
    recent_messages_limit: int
    retrieval_settings_limit: int
    retrieval_cards_limit: int
    model_settings_limit: int
    model_cards_limit: int
    graph_limit: int
    rag_limit: int
    dsl_limit: int


@dataclass(frozen=True)
class SemanticRouteDecision:
    intent: str
    confidence: float
    source: str
    budget_mode: str | None
    rag_mode: str | None
    signals: list[str]


_RAG_MODES = {"local", "global", "hybrid", "mix"}
_CONTEXT_WINDOW_PROFILES: dict[str, ContextWindowPolicy] = {
    "balanced": ContextWindowPolicy(
        profile="balanced",
        recent_messages_limit=12,
        retrieval_settings_limit=256,
        retrieval_cards_limit=256,
        model_settings_limit=40,
        model_cards_limit=32,
        chapter_content_chars=12000,
        chapter_preview_chars=600,
    ),
    "chapter_focus": ContextWindowPolicy(
        profile="chapter_focus",
        recent_messages_limit=8,
        retrieval_settings_limit=128,
        retrieval_cards_limit=128,
        model_settings_limit=24,
        model_cards_limit=20,
        chapter_content_chars=2000,
        chapter_preview_chars=420,
    ),
    "world_focus": ContextWindowPolicy(
        profile="world_focus",
        recent_messages_limit=10,
        retrieval_settings_limit=256,
        retrieval_cards_limit=256,
        model_settings_limit=56,
        model_cards_limit=40,
        chapter_content_chars=1200,
        chapter_preview_chars=320,
    ),
    "minimal": ContextWindowPolicy(
        profile="minimal",
        recent_messages_limit=6,
        retrieval_settings_limit=64,
        retrieval_cards_limit=64,
        model_settings_limit=16,
        model_cards_limit=12,
        chapter_content_chars=1200,
        chapter_preview_chars=260,
    ),
}
_CONTEXT_PACK_CACHE: dict[int, ContextPack] = {}
_CONTEXT_PACK_LOCK = Lock()
_GRAPH_HITS_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_RAG_HITS_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_RETRIEVAL_CACHE_LOCK = Lock()
_CONTEXT_COMPRESSOR_LOCK = Lock()
_RERANKER_MODEL: Any | None = None
_RERANKER_TOKENIZER: Any | None = None
_RERANKER_MODEL_NAME = ""
_RERANKER_DEVICE = "cpu"
_RERANKER_RUNTIME = "transformers"
_RERANKER_UNAVAILABLE = False
_RETRIEVAL_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retrieval")
_LOGGER = logging.getLogger(__name__)
_NEGATIVE_CONSTRAINT_MARKERS = (
    "绝对不可",
    "绝不能",
    "严禁",
    "禁止",
    "禁忌",
    "不可",
    "不能",
    "不得",
    "must not",
    "do not",
    "forbidden",
    "taboo",
    "prohibited",
    "cannot",
)
_NEGATIVE_CONSTRAINT_STRONG_MARKERS = (
    "绝对不可",
    "绝不能",
    "严禁",
    "must not",
    "forbidden",
    "taboo",
    "prohibited",
)
_NEGATIVE_CONSTRAINT_RELATION_MARKERS = (
    "TABOO",
    "FORBIDDEN",
    "PROHIBITED",
    "MUST_NOT",
    "CANNOT",
    "禁忌",
    "禁止",
    "不可",
    "不能",
    "不得",
)


def _truncate_text(text: str, max_chars: int) -> str:
    content = (text or "").strip()
    if len(content) <= max_chars:
        return content
    return content[:max_chars].rstrip() + "..."


def _safe_iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    return None


def _freshness_days(value: Any) -> int | None:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    else:
        return None
    now = datetime.now(timezone.utc)
    delta = now - dt.astimezone(timezone.utc)
    return max(int(delta.total_seconds() // 86400), 0)


def _serialize(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def _extract_query_terms(user_input: str) -> list[str]:
    stop_words = {
        "请",
        "一下",
        "我们",
        "你们",
        "这个",
        "那个",
        "然后",
        "以及",
        "还有",
        "设定",
        "卡片",
        "角色",
        "剧情",
    }
    raw_tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", user_input or "")
    terms: list[str] = []
    for token in raw_tokens:
        if token in stop_words:
            continue
        if token not in terms:
            terms.append(token)
        # 对连续中文短语补充双字词，提升 DSL 命中稳定性（无分词器兜底）
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", token):
            for idx in range(0, len(token) - 1):
                gram = token[idx : idx + 2]
                if gram in stop_words or gram in terms:
                    continue
                terms.append(gram)
    return terms[:10]


def _normalize_pov(pov_mode: str | None, pov_anchor: str | None) -> tuple[str, str | None, list[str]]:
    mode = (pov_mode or "global").strip().lower()
    if mode not in {"global", "character"}:
        mode = "global"
    anchor = (pov_anchor or "").strip() or None
    notes: list[str] = []
    if mode == "character" and not anchor:
        notes.append("pov_mode=character 但未提供 pov_anchor，已退化为全局上下文。")
        mode = "global"
    return mode, anchor, notes


def _normalize_rag_mode(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    mode = value.strip().lower()
    if mode == "auto":
        return "auto"
    if mode in _RAG_MODES:
        return mode
    return None


def _resolve_rag_route(user_input: str, terms: list[str], request_override: str | None) -> tuple[str, str, str]:
    _ = user_input, terms
    request_mode = _normalize_rag_mode(request_override)
    if request_mode and request_mode in _RAG_MODES:
        return request_mode, "forced_by_request", "request_override"

    configured_mode = _normalize_rag_mode(settings.rag_route_policy)
    if configured_mode and configured_mode in _RAG_MODES:
        return configured_mode, "forced_by_config", "config_policy"

    return "mix", "default_mix_policy", "static_default"


def _normalize_context_window_profile(value: str | None) -> str | None:
    profile = str(value or "").strip().lower()
    if profile in _CONTEXT_WINDOW_PROFILES:
        return profile
    return None


def _resolve_context_window_policy(request_profile: str | None) -> tuple[ContextWindowPolicy, str]:
    if request_profile:
        requested = _normalize_context_window_profile(request_profile)
        if requested is not None:
            return _CONTEXT_WINDOW_PROFILES[requested], "request_override"
        return _CONTEXT_WINDOW_PROFILES["balanced"], "request_invalid_fallback"

    configured = _normalize_context_window_profile(getattr(settings, "context_window_profile", "balanced"))
    if configured is not None:
        return _CONTEXT_WINDOW_PROFILES[configured], "config_default"

    return _CONTEXT_WINDOW_PROFILES["balanced"], "fallback_balanced"


_BUDGET_MODE_PRESETS: dict[str, BudgetWeights] = {
    "balanced": BudgetWeights(dsl=0.25, graph=0.25, rag=0.25, history=0.25),
    "action": BudgetWeights(dsl=0.55, graph=0.15, rag=0.10, history=0.20),
    "investigation": BudgetWeights(dsl=0.20, graph=0.45, rag=0.25, history=0.10),
    "world": BudgetWeights(dsl=0.35, graph=0.20, rag=0.30, history=0.15),
    "dialogue": BudgetWeights(dsl=0.18, graph=0.14, rag=0.14, history=0.54),
}
_SEMANTIC_ROUTE_INTENTS = {"writing_help", "world_query", "action_proposal", "brainstorm"}
_SEMANTIC_INTENT_BUDGET_MODE: dict[str, str] = {
    "writing_help": "dialogue",
    "world_query": "world",
    "action_proposal": "investigation",
    "brainstorm": "investigation",
}
_SEMANTIC_INTENT_RAG_MODE: dict[str, str] = {
    "writing_help": "mix",
    "world_query": "local",
    "action_proposal": "global",
    "brainstorm": "hybrid",
}
_SEMANTIC_INTENT_TOKENS: dict[str, tuple[str, ...]] = {
    "writing_help": (
        "续写",
        "扩写",
        "润色",
        "改写",
        "描写",
        "情绪",
        "文风",
        "对话",
        "桥段",
        "台词",
    ),
    "world_query": (
        "设定",
        "世界观",
        "背景",
        "地图",
        "地理",
        "规则",
        "门派",
        "王朝",
        "组织",
        "关系",
        "历史",
    ),
    "action_proposal": (
        "动作",
        "提议",
        "apply",
        "卡片",
        "设定项",
        "新增",
        "删除",
        "修改",
        "更新",
        "合并",
    ),
    "brainstorm": (
        "脑暴",
        "脑洞",
        "推演",
        "分支",
        "可能性",
        "如果",
        "权谋",
        "悬疑",
        "伏笔",
        "下一步",
    ),
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_budget_mode(value: str | None) -> str | None:
    mode = str(value or "").strip().lower()
    return mode if mode in _BUDGET_MODE_PRESETS else None


def _normalize_semantic_intent(value: str | None) -> str | None:
    intent = str(value or "").strip().lower()
    return intent if intent in _SEMANTIC_ROUTE_INTENTS else None


def _normalize_semantic_router_mode(value: str | None) -> str:
    mode = str(value or "auto").strip().lower()
    if mode in {"llm", "heuristic", "auto"}:
        return mode
    return "auto"


def _semantic_intent_from_heuristics(
    user_input: str,
    *,
    chapter_preview: str,
    scene_beat_text: str,
) -> tuple[str, float, list[str]]:
    text = "\n".join([user_input or "", chapter_preview or "", scene_beat_text or ""]).lower()
    text = text.strip()
    if not text:
        return "writing_help", 0.35, []

    intent_scores: dict[str, int] = {intent: 0 for intent in _SEMANTIC_ROUTE_INTENTS}
    intent_signals: dict[str, list[str]] = {intent: [] for intent in _SEMANTIC_ROUTE_INTENTS}
    for intent, tokens in _SEMANTIC_INTENT_TOKENS.items():
        for token in tokens:
            if token and token in text:
                intent_scores[intent] += 1
                if len(intent_signals[intent]) < 8:
                    intent_signals[intent].append(token)

    if ("?" in user_input or "？" in user_input) and intent_scores["world_query"] > 0:
        intent_scores["world_query"] += 1
    if any(marker in text for marker in ("剧情", "下一章", "下一节", "冲突")) and intent_scores["brainstorm"] > 0:
        intent_scores["brainstorm"] += 1

    ordered = sorted(
        intent_scores.items(),
        key=lambda item: (item[1], item[0] == "writing_help"),
        reverse=True,
    )
    top_intent, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0
    if top_score <= 0:
        return "writing_help", 0.33, []

    confidence = min(0.95, 0.42 + 0.16 * max(top_score - second_score, 0) + 0.05 * top_score)
    return top_intent, confidence, intent_signals.get(top_intent, [])


def _normalize_weight_dict(raw: Any) -> BudgetWeights | None:
    if not isinstance(raw, dict):
        return None
    keys = ("dsl", "graph", "rag", "history")
    values: list[float] = []
    for key in keys:
        try:
            parsed = float(raw.get(key))
        except Exception:
            return None
        values.append(max(parsed, 0.0))
    total = sum(values)
    if total <= 0.0:
        return None
    normalized = [item / total for item in values]
    return BudgetWeights(
        dsl=normalized[0],
        graph=normalized[1],
        rag=normalized[2],
        history=normalized[3],
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    content = str(text or "").strip()
    if not content:
        return None
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _call_semantic_router_llm(
    *,
    user_input: str,
    chapter_preview: str,
    scene_beat_text: str,
) -> tuple[str | None, float, list[str]]:
    model = str(settings.lightrag_llm_model or "").strip()
    base_url = str(settings.lightrag_llm_base_url or "").strip()
    api_key = str(settings.lightrag_llm_api_key or "").strip()
    if not (model and base_url and api_key):
        return None, 0.0, []

    endpoint = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是小说助手的语义路由器。"
                    "根据用户输入、章节片段、Scene Beat，识别主要意图。"
                    "intent 仅可取 writing_help|world_query|action_proposal|brainstorm。"
                    "仅输出 JSON："
                    "{\"intent\":\"...\",\"confidence\":0-1,\"signals\":[\"...\"],\"reason\":\"...\"}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_input": _truncate_text(user_input, 900),
                        "chapter_preview": _truncate_text(chapter_preview, 700),
                        "scene_beat": _truncate_text(scene_beat_text, 450),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        timeout = httpx.Timeout(float(settings.semantic_router_llm_timeout_seconds))
        with httpx.Client(timeout=timeout) as client:
            response = client.post(endpoint, json=body, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except Exception:
        return None, 0.0, []

    content = str(payload.get("choices", [{}])[0].get("message", {}).get("content", "") or "")
    parsed = _extract_json_object(content)
    if not parsed:
        return None, 0.0, []

    intent = _normalize_semantic_intent(parsed.get("intent"))
    confidence = 0.0
    try:
        confidence = max(0.0, min(float(parsed.get("confidence", 0.0)), 1.0))
    except Exception:
        confidence = 0.0
    signals_raw = parsed.get("signals")
    signals: list[str] = []
    if isinstance(signals_raw, list):
        for item in signals_raw:
            token = str(item or "").strip()
            if not token or token in signals:
                continue
            signals.append(token[:32])
            if len(signals) >= 8:
                break
    return intent, confidence, signals


def _resolve_semantic_route(
    *,
    user_input: str,
    chapter_preview: str,
    scene_beat_text: str,
) -> SemanticRouteDecision:
    default_intent = "writing_help"
    if not settings.semantic_router_enabled:
        return SemanticRouteDecision(
            intent=default_intent,
            confidence=1.0,
            source="disabled_default",
            budget_mode=None,
            rag_mode=None,
            signals=[],
        )

    mode = _normalize_semantic_router_mode(settings.semantic_router_mode)
    llm_intent: str | None = None
    llm_confidence = 0.0
    llm_signals: list[str] = []
    if mode in {"llm", "auto"}:
        llm_intent, llm_confidence, llm_signals = _call_semantic_router_llm(
            user_input=user_input,
            chapter_preview=chapter_preview,
            scene_beat_text=scene_beat_text,
        )

    threshold = max(min(float(settings.semantic_router_low_confidence_threshold), 1.0), 0.0)
    if llm_intent and llm_confidence >= threshold:
        return SemanticRouteDecision(
            intent=llm_intent,
            confidence=round(llm_confidence, 4),
            source="llm_router",
            budget_mode=_SEMANTIC_INTENT_BUDGET_MODE.get(llm_intent),
            rag_mode=_SEMANTIC_INTENT_RAG_MODE.get(llm_intent),
            signals=llm_signals,
        )

    heuristic_intent, heuristic_confidence, heuristic_signals = _semantic_intent_from_heuristics(
        user_input,
        chapter_preview=chapter_preview,
        scene_beat_text=scene_beat_text,
    )
    source = "heuristic_router"
    confidence = heuristic_confidence
    if llm_intent and llm_confidence > 0.0:
        source = "heuristic_fallback_after_llm"
        confidence = max(confidence, min(llm_confidence, threshold))
    return SemanticRouteDecision(
        intent=heuristic_intent,
        confidence=round(confidence, 4),
        source=source,
        budget_mode=_SEMANTIC_INTENT_BUDGET_MODE.get(heuristic_intent),
        rag_mode=_SEMANTIC_INTENT_RAG_MODE.get(heuristic_intent),
        signals=heuristic_signals,
    )


def _budget_mode_from_heuristics(
    user_input: str,
    *,
    chapter_preview: str,
    scene_beat_text: str,
) -> tuple[str, float]:
    text = "\n".join([user_input or "", chapter_preview or "", scene_beat_text or ""]).lower()
    if not text.strip():
        return "balanced", 0.3

    investigation_tokens = ("推理", "线索", "证据", "谜", "真相", "凶手", "审问", "陷阱")
    action_tokens = ("战", "剑", "刀", "杀", "追", "冲", "搏", "打斗", "逃亡")
    world_tokens = ("设定", "世界观", "历史", "地图", "宗门", "王朝", "地理", "城邦")
    dialogue_tokens = ("对话", "语气", "情绪", "独白", "交流", "辩论")

    def _count(tokens: tuple[str, ...]) -> int:
        return sum(1 for token in tokens if token in text)

    scored = [
        ("investigation", _count(investigation_tokens)),
        ("action", _count(action_tokens)),
        ("world", _count(world_tokens)),
        ("dialogue", _count(dialogue_tokens)),
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    top_mode, top_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0
    if top_score <= 0:
        return "balanced", 0.35
    confidence = min(0.95, 0.45 + 0.18 * max(top_score - second_score, 0) + 0.06 * top_score)
    return top_mode, confidence


def _call_budget_router_llm(
    *,
    user_input: str,
    chapter_preview: str,
    scene_beat_text: str,
) -> tuple[str | None, BudgetWeights | None, float]:
    model = str(settings.lightrag_llm_model or "").strip()
    base_url = str(settings.lightrag_llm_base_url or "").strip()
    api_key = str(settings.lightrag_llm_api_key or "").strip()
    if not (model and base_url and api_key):
        return None, None, 0.0

    endpoint = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是小说写作系统的上下文预算路由器。"
                    "请基于输入判定 mode，并分配 dsl/graph/rag/history 权重。"
                    "只输出 JSON，格式："
                    "{\"mode\":\"action|investigation|world|dialogue|balanced\","
                    "\"weights\":{\"dsl\":0-1,\"graph\":0-1,\"rag\":0-1,\"history\":0-1},"
                    "\"confidence\":0-1}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_input": _truncate_text(user_input, 800),
                        "chapter_preview": _truncate_text(chapter_preview, 700),
                        "scene_beat": _truncate_text(scene_beat_text, 500),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        timeout = httpx.Timeout(float(settings.budget_router_llm_timeout_seconds))
        with httpx.Client(timeout=timeout) as client:
            response = client.post(endpoint, json=body, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except Exception:
        return None, None, 0.0

    content = str(payload.get("choices", [{}])[0].get("message", {}).get("content", "") or "")
    parsed = _extract_json_object(content)
    if not parsed:
        return None, None, 0.0
    mode = _normalize_budget_mode(parsed.get("mode"))
    weights = _normalize_weight_dict(parsed.get("weights"))
    confidence = 0.0
    try:
        confidence = max(0.0, min(float(parsed.get("confidence", 0.0)), 1.0))
    except Exception:
        confidence = 0.0
    return mode, weights, confidence


def _blend_weights(primary: BudgetWeights, secondary: BudgetWeights, ratio: float) -> BudgetWeights:
    alpha = max(0.0, min(float(ratio), 1.0))
    mixed = {
        "dsl": primary.dsl * alpha + secondary.dsl * (1 - alpha),
        "graph": primary.graph * alpha + secondary.graph * (1 - alpha),
        "rag": primary.rag * alpha + secondary.rag * (1 - alpha),
        "history": primary.history * alpha + secondary.history * (1 - alpha),
    }
    normalized = _normalize_weight_dict(mixed) or secondary
    return normalized


def _scale_limit(base: int, *, weight: float, low: float = 0.45, high: float = 1.95, min_value: int = 1) -> int:
    normalized_weight = max(0.0, min(weight, 1.0))
    ratio = low + (high - low) * normalized_weight
    return max(int(round(base * ratio)), min_value)


def _resolve_dynamic_budget_plan(
    *,
    base_policy: ContextWindowPolicy,
    user_input: str,
    request_budget_mode: str | None,
    chapter_preview: str,
    scene_beat_text: str,
) -> DynamicBudgetPlan:
    default_weights = _BUDGET_MODE_PRESETS["balanced"]
    if not settings.budget_router_enabled:
        return DynamicBudgetPlan(
            mode="balanced",
            source="disabled",
            confidence=1.0,
            weights=default_weights,
            recent_messages_limit=base_policy.recent_messages_limit,
            retrieval_settings_limit=base_policy.retrieval_settings_limit,
            retrieval_cards_limit=base_policy.retrieval_cards_limit,
            model_settings_limit=base_policy.model_settings_limit,
            model_cards_limit=base_policy.model_cards_limit,
            graph_limit=10,
            rag_limit=8,
            dsl_limit=8,
        )

    request_mode = _normalize_budget_mode(request_budget_mode)
    if request_mode:
        chosen_mode = request_mode
        chosen_weights = _BUDGET_MODE_PRESETS[request_mode]
        source = "request_override"
        confidence = 1.0
    else:
        router_mode = str(settings.budget_router_mode or "auto").strip().lower()
        llm_mode: str | None = None
        llm_weights: BudgetWeights | None = None
        llm_confidence = 0.0
        if router_mode in {"llm", "auto"}:
            llm_mode, llm_weights, llm_confidence = _call_budget_router_llm(
                user_input=user_input,
                chapter_preview=chapter_preview,
                scene_beat_text=scene_beat_text,
            )

        if llm_mode and llm_weights and llm_confidence >= settings.budget_router_low_confidence_threshold:
            chosen_mode = llm_mode
            chosen_weights = llm_weights
            source = "llm_router"
            confidence = llm_confidence
        else:
            heuristic_mode, heuristic_conf = _budget_mode_from_heuristics(
                user_input,
                chapter_preview=chapter_preview,
                scene_beat_text=scene_beat_text,
            )
            heuristic_weights = _BUDGET_MODE_PRESETS.get(heuristic_mode, default_weights)
            if llm_weights is not None and llm_confidence > 0.0:
                chosen_weights = _blend_weights(llm_weights, heuristic_weights, ratio=0.45)
                chosen_mode = heuristic_mode
                source = "heuristic_blend_with_llm"
                confidence = max(heuristic_conf, llm_confidence)
            else:
                chosen_mode = heuristic_mode
                chosen_weights = heuristic_weights
                source = "heuristic_router"
                confidence = heuristic_conf

    recent_messages_limit = _scale_limit(
        base_policy.recent_messages_limit,
        weight=chosen_weights.history,
        low=0.4,
        high=1.85,
        min_value=4,
    )
    retrieval_settings_limit = _scale_limit(
        base_policy.retrieval_settings_limit,
        weight=min(1.0, chosen_weights.dsl + chosen_weights.graph * 0.35),
        low=0.45,
        high=1.75,
        min_value=16,
    )
    retrieval_cards_limit = _scale_limit(
        base_policy.retrieval_cards_limit,
        weight=min(1.0, chosen_weights.dsl * 0.3 + chosen_weights.rag * 0.7 + chosen_weights.graph * 0.3),
        low=0.45,
        high=1.75,
        min_value=16,
    )
    model_settings_limit = _scale_limit(
        base_policy.model_settings_limit,
        weight=min(1.0, chosen_weights.dsl + chosen_weights.graph * 0.5),
        low=0.45,
        high=1.7,
        min_value=8,
    )
    model_cards_limit = _scale_limit(
        base_policy.model_cards_limit,
        weight=min(1.0, chosen_weights.rag + chosen_weights.graph * 0.45),
        low=0.45,
        high=1.7,
        min_value=8,
    )
    graph_limit = _scale_limit(10, weight=chosen_weights.graph, low=0.45, high=1.9, min_value=3)
    rag_limit = _scale_limit(8, weight=chosen_weights.rag, low=0.45, high=1.9, min_value=2)
    dsl_limit = _scale_limit(8, weight=chosen_weights.dsl, low=0.45, high=1.9, min_value=2)
    return DynamicBudgetPlan(
        mode=chosen_mode,
        source=source,
        confidence=round(confidence, 4),
        weights=chosen_weights,
        recent_messages_limit=recent_messages_limit,
        retrieval_settings_limit=retrieval_settings_limit,
        retrieval_cards_limit=retrieval_cards_limit,
        model_settings_limit=model_settings_limit,
        model_cards_limit=model_cards_limit,
        graph_limit=graph_limit,
        rag_limit=rag_limit,
        dsl_limit=dsl_limit,
    )


def _normalize_timeout(seconds: float | int | None, fallback: float) -> float:
    try:
        value = float(seconds)
    except Exception:
        value = fallback
    return max(value, 0.1)


def _cache_ttl_seconds() -> float:
    try:
        ttl = float(settings.retrieval_cache_ttl_seconds)
    except Exception:
        ttl = 0.0
    return max(ttl, 0.0)


def _cache_enabled() -> bool:
    return _cache_ttl_seconds() > 0.0


def _clone_hits(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in items if isinstance(item, dict)]


def _cache_get(
    cache: dict[str, tuple[float, list[dict[str, Any]]]],
    key: str,
) -> tuple[list[dict[str, Any]] | None, str]:
    if not _cache_enabled():
        return None, "disabled"
    now = time.time()
    with _RETRIEVAL_CACHE_LOCK:
        cached = cache.get(key)
        if cached is None:
            return None, "miss"
        expires_at, rows = cached
        if expires_at <= now:
            cache.pop(key, None)
            return None, "expired"
        return _clone_hits(rows), "hit"


def _cache_set(
    cache: dict[str, tuple[float, list[dict[str, Any]]]],
    key: str,
    rows: list[dict[str, Any]],
) -> None:
    ttl = _cache_ttl_seconds()
    if ttl <= 0:
        return
    normalized = _clone_hits(rows)
    if not normalized:
        return
    max_entries = max(int(settings.retrieval_cache_max_entries), 16)
    expires_at = time.time() + ttl
    with _RETRIEVAL_CACHE_LOCK:
        cache[key] = (expires_at, normalized)
        if len(cache) > max_entries:
            oldest_key = min(cache.items(), key=lambda item: item[1][0])[0]
            cache.pop(oldest_key, None)


def _graph_cache_key(
    project_id: int,
    terms: list[str],
    anchor: str | None,
    limit: int,
    *,
    current_chapter: int | None = None,
) -> str:
    normalized_terms = sorted(
        {
            str(item).strip().lower()
            for item in terms
            if isinstance(item, str) and str(item).strip()
        }
    )
    anchor_norm = str(anchor or "").strip().lower()
    chapter_norm = _safe_int(current_chapter, 0)
    return f"p:{project_id}|a:{anchor_norm}|l:{int(limit)}|c:{chapter_norm}|terms:{','.join(normalized_terms)}"


def _rag_cache_key(
    user_input: str,
    anchor: str | None,
    mode: str | None,
    limit: int,
) -> str:
    query_hash = hashlib.sha1((user_input or "").strip().encode("utf-8")).hexdigest()[:20]
    anchor_norm = str(anchor or "").strip().lower()
    mode_norm = str(mode or "").strip().lower()
    return f"q:{query_hash}|a:{anchor_norm}|m:{mode_norm}|l:{int(limit)}"


def _submit_graph_future(
    project_id: int,
    terms: list[str],
    anchor: str | None,
    limit: int,
    *,
    current_chapter: int | None = None,
) -> Future[list[dict[str, Any]]]:
    return _RETRIEVAL_EXECUTOR.submit(
        fetch_neo4j_graph_facts,
        project_id,
        terms,
        anchor=anchor,
        limit=limit,
        current_chapter=current_chapter,
    )


def _submit_rag_future(
    user_input: str,
    anchor: str | None,
    limit: int,
    rag_mode: str,
) -> Future[list[dict[str, Any]]]:
    return _RETRIEVAL_EXECUTOR.submit(
        fetch_lightrag_semantic_hits,
        user_input,
        anchor=anchor,
        limit=limit,
        mode_override=rag_mode,
    )


def _await_hits_future(
    future: Future[list[dict[str, Any]]],
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], bool, bool]:
    try:
        rows = future.result(timeout=_normalize_timeout(timeout_seconds, 1.0))
    except TimeoutError:
        future.cancel()
        return [], True, False
    except Exception:
        return [], False, True
    if not isinstance(rows, list):
        return [], False, True
    return [item for item in rows if isinstance(item, dict)], False, False


def _to_setting_snapshot(row: Any, project_id: int) -> SettingSnapshot:
    raw_value = getattr(row, "value", {})
    raw_aliases = getattr(row, "aliases", [])
    return SettingSnapshot(
        id=int(getattr(row, "id")),
        project_id=project_id,
        key=str(getattr(row, "key") or ""),
        value=raw_value,
        aliases=[str(item).strip() for item in raw_aliases if str(item).strip()] if isinstance(raw_aliases, list) else [],
        value_text=_serialize(raw_value),
        updated_at=getattr(row, "updated_at", None),
    )


def _to_card_snapshot(row: Any, project_id: int) -> CardSnapshot:
    raw_content = getattr(row, "content", {})
    raw_aliases = getattr(row, "aliases", [])
    return CardSnapshot(
        id=int(getattr(row, "id")),
        project_id=project_id,
        title=str(getattr(row, "title") or ""),
        content=raw_content,
        aliases=[str(item).strip() for item in raw_aliases if str(item).strip()] if isinstance(raw_aliases, list) else [],
        content_text=_serialize(raw_content),
        updated_at=getattr(row, "updated_at", None),
    )


def _load_context_pack(db: Session, project_id: int, *, force_refresh: bool = False) -> tuple[list[Any], list[Any], dict[str, Any]]:
    settings_limit = max(int(settings.context_pack_max_settings), 1)
    cards_limit = max(int(settings.context_pack_max_cards), 1)
    if not settings.context_pack_enabled:
        settings_rows = [
            _to_setting_snapshot(row, project_id)
            for row in list_settings(db, project_id, limit=settings_limit)
        ]
        cards_rows = [
            _to_card_snapshot(row, project_id)
            for row in list_cards(db, project_id, limit=cards_limit)
        ]
        return settings_rows, cards_rows, {"enabled": False, "source": "disabled", "age_ms": 0}

    now = time.time()
    ttl = max(int(settings.context_pack_ttl_seconds), 0)
    with _CONTEXT_PACK_LOCK:
        cached = _CONTEXT_PACK_CACHE.get(project_id)
        if cached and not force_refresh and (now - cached.generated_at) <= ttl:
            return (
                cached.settings_rows,
                cached.cards_rows,
                {
                    "enabled": True,
                    "source": "cache_hit",
                    "age_ms": int((now - cached.generated_at) * 1000),
                },
            )

    settings_rows = [
        _to_setting_snapshot(row, project_id)
        for row in list_settings(db, project_id, limit=settings_limit)
    ]
    cards_rows = [
        _to_card_snapshot(row, project_id)
        for row in list_cards(db, project_id, limit=cards_limit)
    ]
    pack = ContextPack(
        project_id=project_id,
        generated_at=now,
        settings_rows=settings_rows,
        cards_rows=cards_rows,
    )
    with _CONTEXT_PACK_LOCK:
        _CONTEXT_PACK_CACHE[project_id] = pack
        if len(_CONTEXT_PACK_CACHE) > 128:
            oldest_project = min(_CONTEXT_PACK_CACHE.items(), key=lambda item: item[1].generated_at)[0]
            _CONTEXT_PACK_CACHE.pop(oldest_project, None)

    return settings_rows, cards_rows, {"enabled": True, "source": "cache_miss", "age_ms": 0}


def preheat_context_pack(db: Session, project_id: int) -> dict[str, Any]:
    settings_rows, cards_rows, _ = _load_context_pack(db, project_id, force_refresh=True)
    return {
        "project_id": project_id,
        "settings_count": len(settings_rows),
        "cards_count": len(cards_rows),
        "ttl_seconds": max(int(settings.context_pack_ttl_seconds), 0),
    }


def _normalize_reference_project_ids(
    value: list[int] | None,
    *,
    current_project_id: int,
    max_items: int = 5,
) -> list[int]:
    if not isinstance(value, list):
        return []
    normalized: list[int] = []
    for raw in value:
        try:
            project_id = int(raw)
        except Exception:
            continue
        if project_id <= 0 or project_id == current_project_id:
            continue
        if project_id in normalized:
            continue
        normalized.append(project_id)
        if len(normalized) >= max(max_items, 1):
            break
    return normalized


def _load_reference_context(
    db: Session,
    reference_project_ids: list[int],
) -> tuple[list[SettingSnapshot], list[CardSnapshot], list[dict[str, int]]]:
    settings_rows: list[SettingSnapshot] = []
    cards_rows: list[CardSnapshot] = []
    project_meta: list[dict[str, int]] = []
    settings_limit = max(int(settings.context_pack_max_settings), 1)
    cards_limit = max(int(settings.context_pack_max_cards), 1)
    for ref_project_id in reference_project_ids:
        ref_settings = [
            _to_setting_snapshot(row, ref_project_id)
            for row in list_settings(db, ref_project_id, limit=settings_limit)
        ]
        ref_cards = [
            _to_card_snapshot(row, ref_project_id)
            for row in list_cards(db, ref_project_id, limit=cards_limit)
        ]
        settings_rows.extend(ref_settings)
        cards_rows.extend(ref_cards)
        project_meta.append(
            {
                "project_id": ref_project_id,
                "settings": len(ref_settings),
                "cards": len(ref_cards),
            }
        )
    return settings_rows, cards_rows, project_meta


def _extract_citation(hit: dict[str, Any]) -> dict[str, str] | None:
    if not isinstance(hit, dict):
        return None

    raw = hit.get("citation")
    if isinstance(raw, dict):
        source = str(raw.get("source") or "").strip()
        chunk = str(raw.get("chunk") or "").strip()
        if source or chunk:
            result: dict[str, str] = {}
            if source:
                result["source"] = source
            if chunk:
                result["chunk"] = chunk
            return result

    source = str(hit.get("file_path") or hit.get("source") or hit.get("title") or "").strip()
    chunk = str(hit.get("chunk_id") or "").strip()
    if not source and not chunk:
        return None
    result = {}
    if source:
        result["source"] = source
    if chunk:
        result["chunk"] = chunk
    return result


def _citation_required() -> bool:
    policy = str(settings.citation_policy or "off").strip().lower()
    if policy in {"off", "disabled", "none"}:
        return False
    if policy in {"always", "strict", "factual", "on", "true", "1"}:
        return True
    return False


def _build_quality_gate(user_input: str, rag_provider: str, semantic_hits: list[dict[str, Any]]) -> dict[str, Any]:
    _ = user_input
    citations = [item for hit in semantic_hits if (item := _extract_citation(hit))]
    min_required = max(int(settings.citation_min_count), 0)
    citation_required = _citation_required()
    citation_ok = (not citation_required) or len(citations) >= max(min_required, 1)

    reranker_expected = bool(settings.lightrag_rerank_enabled)
    reranker_effective = reranker_expected and rag_provider.startswith("lightrag")
    reranker_ok = (not settings.reranker_required) or reranker_effective

    reasons: list[str] = []
    if not citation_ok:
        reasons.append("missing_citation")
    if not reranker_ok:
        reasons.append("reranker_not_effective")

    unique_sources = sorted({item.get("source", "") for item in citations if item.get("source")})
    return {
        "degraded": bool(reasons),
        "degrade_reasons": reasons,
        "citation_required": citation_required,
        "citation_min_required": max(min_required, 1) if citation_required else 0,
        "citation_count": len(citations),
        "citation_sources": unique_sources[:8],
        "citation_coverage": round(len(citations) / max(len(semantic_hits), 1), 4),
        "reranker_expected": reranker_expected,
        "reranker_effective": reranker_effective,
    }


def _resolve_rag_short_circuit(
    *,
    deterministic_first: bool,
    dsl_hits: list[dict[str, Any]],
    graph_facts: list[dict[str, Any]],
) -> tuple[bool, str]:
    if not deterministic_first:
        return False, "disabled_by_request"
    if not settings.deterministic_short_circuit_enabled:
        return False, "disabled_by_config"

    min_dsl = max(int(settings.deterministic_min_dsl_hits), 0)
    min_graph = max(int(settings.deterministic_min_graph_hits), 0)
    if min_dsl > 0 and len(dsl_hits) >= min_dsl:
        return True, "dsl_hit_threshold"
    if min_graph > 0 and len(graph_facts) >= min_graph:
        return True, "graph_hit_threshold"
    return False, "insufficient_authoritative_hits"


def _score_term_hits(text: str, terms: list[str]) -> int:
    if not text or not terms:
        return 0
    return sum(1 for term in terms if term and term in text)


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    normalized = re.sub(r"\s+", "", text or "")
    if len(normalized) < n:
        return set()
    return {normalized[idx : idx + n] for idx in range(0, len(normalized) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _setting_value_text(row: Any) -> str:
    value_text = getattr(row, "value_text", None)
    if isinstance(value_text, str):
        return value_text
    return _serialize(getattr(row, "value", {}))


def _card_content_text(row: Any) -> str:
    content_text = getattr(row, "content_text", None)
    if isinstance(content_text, str):
        return content_text
    return _serialize(getattr(row, "content", {}))


def _setting_source_text(row: Any) -> str:
    return f"{getattr(row, 'key', '')}\n{_setting_value_text(row)}"


def _card_source_text(row: Any) -> str:
    return f"{getattr(row, 'title', '')}\n{_card_content_text(row)}"


def _window_retrieval_rows(
    rows: list[Any],
    *,
    limit: int,
    terms: list[str],
    source_text_getter: Callable[[Any], str],
) -> list[Any]:
    max_items = max(int(limit), 0)
    if max_items <= 0:
        return []
    if len(rows) <= max_items:
        return rows

    selected_indexes: set[int] = set()
    if terms:
        ranked: list[tuple[int, int, int]] = []
        for idx, row in enumerate(rows):
            score = _score_term_hits(source_text_getter(row), terms)
            if score <= 0:
                continue
            row_id = int(getattr(row, "id", 0) or 0)
            ranked.append((score, row_id, idx))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _, _, idx in ranked:
            selected_indexes.add(idx)
            if len(selected_indexes) >= max_items:
                break

    if len(selected_indexes) < max_items:
        for idx in range(len(rows) - 1, -1, -1):
            selected_indexes.add(idx)
            if len(selected_indexes) >= max_items:
                break

    ordered_indexes = sorted(selected_indexes)
    return [rows[idx] for idx in ordered_indexes]


def _apply_pov_filter(settings_rows: list[Any], cards_rows: list[Any], mode: str, anchor: str | None) -> tuple[list[Any], list[Any]]:
    if mode != "character" or not anchor:
        return settings_rows, cards_rows

    anchor_lower = anchor.lower()

    filtered_settings = []
    for row in settings_rows:
        text = _setting_source_text(row).lower()
        if anchor_lower in text:
            filtered_settings.append(row)

    filtered_cards = []
    for row in cards_rows:
        text = _card_source_text(row).lower()
        if anchor_lower in text:
            filtered_cards.append(row)

    if filtered_settings or filtered_cards:
        return filtered_settings or settings_rows, filtered_cards or cards_rows
    return settings_rows, cards_rows


def _build_dsl_hits(
    terms: list[str],
    settings_rows: list[Any],
    cards_rows: list[Any],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for row in settings_rows:
        source_text = _setting_source_text(row)
        score = _score_term_hits(source_text, terms)
        if score <= 0:
            continue
        hits.append(
            {
                "kind": "setting",
                "id": row.id,
                "project_id": getattr(row, "project_id", None),
                "title": row.key,
                "score": score,
                "snippet": _truncate_text(source_text, 180),
                "confidence": 0.95,
                "updated_at": _safe_iso(getattr(row, "updated_at", None)),
                "freshness_days": _freshness_days(getattr(row, "updated_at", None)),
            }
        )

    for row in cards_rows:
        source_text = _card_source_text(row)
        score = _score_term_hits(source_text, terms)
        if score <= 0:
            continue
        hits.append(
            {
                "kind": "card",
                "id": row.id,
                "project_id": getattr(row, "project_id", None),
                "title": row.title,
                "score": score,
                "snippet": _truncate_text(source_text, 180),
                "confidence": 0.9,
                "updated_at": _safe_iso(getattr(row, "updated_at", None)),
                "freshness_days": _freshness_days(getattr(row, "updated_at", None)),
            }
        )

    hits.sort(
        key=lambda item: (
            int(item.get("score", 0)),
            -(int(item.get("freshness_days")) if isinstance(item.get("freshness_days"), int) else 99999),
        ),
        reverse=True,
    )
    return hits[: max(int(limit), 1)]


def _build_graph_facts(
    cards_rows: list[Any],
    settings_rows: list[Any],
    anchor: str | None,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    relation_keys = {
        "relationship",
        "relationships",
        "relation",
        "faction",
        "affiliation",
        "motivation",
        "goal",
        "status",
        "secret",
    }

    facts: list[dict[str, Any]] = []

    for row in cards_rows:
        content = row.content if isinstance(row.content, dict) else {}
        for key in relation_keys:
            if key not in content:
                continue
            value_text = _serialize(content.get(key))
            if not value_text.strip():
                continue
            line = f"{row.title} / {key}: {value_text}"
            if anchor and anchor not in line:
                continue
            facts.append(
                {
                    "kind": "card_fact",
                    "id": row.id,
                    "project_id": getattr(row, "project_id", None),
                    "title": row.title,
                    "fact": _truncate_text(line, 180),
                    "confidence": 0.6,
                    "updated_at": _safe_iso(getattr(row, "updated_at", None)),
                    "freshness_days": _freshness_days(getattr(row, "updated_at", None)),
                }
            )
            if len(facts) >= max(int(limit), 1):
                return facts

    for row in settings_rows:
        key_text = (row.key or "").strip()
        if not key_text:
            continue
        if not any(marker in key_text for marker in ("关系", "阵营", "势力", "组织", "规则")):
            continue
        line = f"{key_text}: {_setting_value_text(row)}"
        if anchor and anchor not in line:
            continue
        facts.append(
            {
                "kind": "setting_fact",
                "id": row.id,
                "project_id": getattr(row, "project_id", None),
                "title": key_text,
                "fact": _truncate_text(line, 180),
                "confidence": 0.65,
                "updated_at": _safe_iso(getattr(row, "updated_at", None)),
                "freshness_days": _freshness_days(getattr(row, "updated_at", None)),
            }
        )
        if len(facts) >= max(int(limit), 1):
            break

    return facts[: max(int(limit), 1)]


def _build_semantic_hits(
    user_input: str,
    settings_rows: list[Any],
    cards_rows: list[Any],
    anchor: str | None,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    query_ngrams = _char_ngrams(user_input or "")
    if not query_ngrams:
        return []

    ranked: list[dict[str, Any]] = []
    for row in settings_rows:
        source = _setting_source_text(row)
        if anchor and anchor not in source:
            continue
        score = _jaccard(query_ngrams, _char_ngrams(source))
        if score <= 0:
            continue
        ranked.append(
            {
                "kind": "setting",
                "id": row.id,
                "project_id": getattr(row, "project_id", None),
                "title": row.key,
                "score": round(score, 4),
                "snippet": _truncate_text(source, 180),
                "confidence": round(score, 4),
                "updated_at": _safe_iso(getattr(row, "updated_at", None)),
                "freshness_days": _freshness_days(getattr(row, "updated_at", None)),
            }
        )

    for row in cards_rows:
        source = _card_source_text(row)
        if anchor and anchor not in source:
            continue
        score = _jaccard(query_ngrams, _char_ngrams(source))
        if score <= 0:
            continue
        ranked.append(
            {
                "kind": "card",
                "id": row.id,
                "project_id": getattr(row, "project_id", None),
                "title": row.title,
                "score": round(score, 4),
                "snippet": _truncate_text(source, 180),
                "confidence": round(score, 4),
                "updated_at": _safe_iso(getattr(row, "updated_at", None)),
                "freshness_days": _freshness_days(getattr(row, "updated_at", None)),
            }
        )

    ranked.sort(
        key=lambda item: (
            float(item.get("score", 0.0)),
            -(int(item.get("freshness_days")) if isinstance(item.get("freshness_days"), int) else 99999),
        ),
        reverse=True,
    )
    return ranked[: max(int(limit), 1)]


def _normalize_context_compression_mode(value: str | None) -> str:
    mode = str(value or "task_aware").strip().lower()
    if mode in {"off", "heuristic", "task_aware", "llm", "auto", "rerank"}:
        return mode
    return "task_aware"


def _hit_preview_text(hit: dict[str, Any]) -> str:
    if not isinstance(hit, dict):
        return ""
    return str(
        hit.get("snippet")
        or hit.get("fact")
        or hit.get("content")
        or hit.get("text")
        or ""
    ).strip()


def _contains_negative_constraint_marker(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    lower = value.lower()
    for marker in _NEGATIVE_CONSTRAINT_MARKERS:
        token = str(marker or "").strip()
        if not token:
            continue
        if token in value or token.lower() in lower:
            return True
    return False


def _normalize_negative_constraint_text(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"^[\s\-*•#]+", "", value)
    value = re.sub(r"^\d+[.)、:\s]+", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return _truncate_text(value, 160)


def _negative_constraint_identity(text: str) -> str:
    normalized = re.sub(r"[\W_]+", "", str(text or "").lower())
    return normalized[:120]


def _extract_negative_constraints_from_text(
    text: str,
    *,
    relation_hint: bool,
    limit: int,
) -> list[str]:
    normalized = str(text or "").replace("\r", "\n").strip()
    if not normalized:
        return []
    segments = re.split(r"[。！？!?；;\n]+", normalized)
    extracted: list[str] = []
    seen: set[str] = set()
    for raw_segment in segments:
        segment = _normalize_negative_constraint_text(raw_segment)
        if not segment:
            continue
        if not _contains_negative_constraint_marker(segment):
            continue
        key = _negative_constraint_identity(segment)
        if not key or key in seen:
            continue
        seen.add(key)
        extracted.append(segment)
        if len(extracted) >= max(limit, 1):
            return extracted
    if extracted:
        return extracted
    if relation_hint and _contains_negative_constraint_marker(normalized):
        segment = _normalize_negative_constraint_text(normalized)
        if segment:
            return [segment]
    return []


def _negative_constraint_source_weight(source: str) -> float:
    source_name = str(source or "").strip().upper()
    if source_name == "DSL":
        return 0.35
    if source_name == "GRAPH":
        return 0.25
    if source_name == "RAG":
        return 0.15
    return 0.0


def _build_negative_constraints(
    *,
    user_input: str,
    dsl_hits: list[dict[str, Any]],
    graph_facts: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
    limit: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started_at = time.perf_counter()
    terms = _extract_query_terms(user_input)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    for source, rows in (("DSL", dsl_hits), ("GRAPH", graph_facts), ("RAG", semantic_hits)):
        for hit in rows[: max(limit * 4, 12)]:
            if not isinstance(hit, dict):
                continue
            parts = [
                str(hit.get("title") or ""),
                str(hit.get("fact") or ""),
                str(hit.get("snippet") or ""),
            ]
            corpus = " ".join(part for part in parts if part).strip()
            if not corpus:
                continue
            relation_hint = False
            if source == "GRAPH":
                upper_corpus = corpus.upper()
                relation_hint = any(token in upper_corpus for token in _NEGATIVE_CONSTRAINT_RELATION_MARKERS)
            constraints = _extract_negative_constraints_from_text(
                corpus,
                relation_hint=relation_hint,
                limit=3,
            )
            if not constraints:
                continue

            score_raw = hit.get("score", hit.get("confidence"))
            try:
                base_score = max(min(float(score_raw), 1.0), 0.0)
            except Exception:
                base_score = 0.35

            for text in constraints:
                key = _negative_constraint_identity(text)
                if not key or key in seen:
                    continue
                seen.add(key)
                score = base_score + _negative_constraint_source_weight(source)
                lower_text = text.lower()
                for term in terms:
                    token = str(term or "").strip().lower()
                    if token and token in lower_text:
                        score += 0.08
                if any(marker in text or marker in lower_text for marker in _NEGATIVE_CONSTRAINT_STRONG_MARKERS):
                    score += 0.12
                candidates.append(
                    {
                        "text": text,
                        "source": source,
                        "title": _truncate_text(str(hit.get("title") or hit.get("kind") or source), 64),
                        "kind": str(hit.get("kind") or "").strip(),
                        "score": round(max(min(score, 2.0), 0.0), 4),
                        "confidence": round(base_score, 4),
                    }
                )

    candidates.sort(
        key=lambda item: (
            float(item.get("score", 0.0)),
            float(item.get("confidence", 0.0)),
        ),
        reverse=True,
    )
    selected = candidates[: max(limit, 1)]
    source_counts = {"DSL": 0, "GRAPH": 0, "RAG": 0}
    for item in selected:
        source_name = str(item.get("source") or "").strip().upper()
        if source_name in source_counts:
            source_counts[source_name] += 1
    elapsed_ms = max(int((time.perf_counter() - started_at) * 1000), 0)
    metadata = {
        "enabled": True,
        "count": len(selected),
        "sources": source_counts,
        "elapsed_ms": elapsed_ms,
    }
    return selected, metadata


def _build_context_compression_corpus(
    *,
    dsl_hits: list[dict[str, Any]],
    graph_facts: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    lines: list[str] = []
    for source, rows in (("DSL", dsl_hits), ("GRAPH", graph_facts), ("RAG", semantic_hits)):
        for item in rows:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("kind") or source).strip()
            preview = _truncate_text(_hit_preview_text(item), 220)
            score_raw = item.get("score", item.get("confidence"))
            try:
                score = float(score_raw)
            except Exception:
                score = 0.0
            line = f"[{source}] {title} :: {preview} (score={score:.3f})"
            lines.append(line)
    joined = "\n".join(lines)
    return joined, lines


def _line_task_relevance_score(
    line: str,
    *,
    terms: list[str],
    intent: str,
) -> float:
    score = 0.0
    lower_line = line.lower()
    for term in terms:
        if term and term.lower() in lower_line:
            score += 1.0
    if f"[{intent}]".lower() in lower_line:
        score += 0.8
    if "[dsl]" in lower_line:
        score += 0.45
    if "[graph]" in lower_line:
        score += 0.35
    if "[rag]" in lower_line:
        score += 0.25
    return score


def _heuristic_context_compress(
    *,
    user_input: str,
    intent: str,
    lines: list[str],
    max_chars: int,
) -> str:
    if not lines:
        return ""
    terms = _extract_query_terms(user_input)
    ranked: list[tuple[float, int, str]] = []
    for idx, line in enumerate(lines):
        ranked.append((_line_task_relevance_score(line, terms=terms, intent=intent), idx, line))
    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    selected_lines: list[str] = []
    used_indexes: set[int] = set()
    for _, idx, line in ranked:
        if idx in used_indexes:
            continue
        used_indexes.add(idx)
        selected_lines.append(line)
        if len(selected_lines) >= 24:
            break

    chunks = _apply_source_priority_tiering(selected_lines, max_chars=max_chars)
    return "\n".join(chunks)


def _call_context_compressor_llm(
    *,
    user_input: str,
    intent: str,
    corpus: str,
    max_chars: int,
) -> str | None:
    model = str(settings.lightrag_llm_model or "").strip()
    base_url = str(settings.lightrag_llm_base_url or "").strip()
    api_key = str(settings.lightrag_llm_api_key or "").strip()
    if not (model and base_url and api_key):
        return None

    endpoint = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是上下文压缩器。"
                    "仅保留与当前写作任务最相关、可直接用于生成的事实。"
                    "禁止新增事实。"
                    "输出 JSON：{\"compressed_context\":\"...\"}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "intent": intent,
                        "user_input": _truncate_text(user_input, 800),
                        "evidence": _truncate_text(corpus, 9000),
                        "max_chars": max(int(max_chars), 200),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        timeout = httpx.Timeout(float(settings.context_compression_llm_timeout_seconds))
        with httpx.Client(timeout=timeout) as client:
            response = client.post(endpoint, json=body, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except Exception:
        return None

    content = str(payload.get("choices", [{}])[0].get("message", {}).get("content", "") or "")
    parsed = _extract_json_object(content)
    if not parsed:
        return None
    summary = str(
        parsed.get("compressed_context")
        or parsed.get("summary")
        or parsed.get("text")
        or ""
    ).strip()
    if not summary:
        return None
    return _truncate_text(summary, max(int(max_chars), 200))


def _compression_line_source(line: str) -> str:
    match = re.match(r"^\[([A-Za-z]+)\]", str(line or "").strip())
    if not match:
        return "OTHER"
    return str(match.group(1) or "").strip().upper() or "OTHER"


_COMPRESSION_SOURCE_TIER_ORDER = ("DSL", "GRAPH", "RAG", "OTHER")
_COMPRESSION_SECTION_BY_SOURCE = {
    "DSL": "fact_entities",
    "GRAPH": "fact_relations",
    "RAG": "retrieved_events",
}


def _apply_source_priority_tiering(lines: list[str], *, max_chars: int) -> list[str]:
    if max_chars <= 0:
        return []
    buckets: dict[str, list[str]] = {source: [] for source in _COMPRESSION_SOURCE_TIER_ORDER}
    for line in lines:
        piece = str(line or "").strip()
        if not piece:
            continue
        source = _compression_line_source(piece)
        key = source if source in buckets else "OTHER"
        buckets[key].append(piece)

    selected: list[str] = []
    total = 0
    for source in _COMPRESSION_SOURCE_TIER_ORDER:
        for piece in buckets[source]:
            next_total = total + len(piece) + (1 if selected else 0)
            if next_total > max_chars:
                if not selected:
                    selected.append(_truncate_text(piece, max_chars))
                return selected
            selected.append(piece)
            total = next_total
    return selected


def _parse_compression_line(line: str) -> tuple[str, str, str]:
    raw_line = str(line or "").strip()
    if not raw_line:
        return "OTHER", "", ""
    source = _compression_line_source(raw_line)
    body = re.sub(r"^\[[^\]]+\]\s*", "", raw_line)
    body = re.sub(r"\s+\(score=[^)]+\)\s*$", "", body).strip()
    if "::" in body:
        title, preview = body.split("::", 1)
    elif ":" in body:
        title, preview = body.split(":", 1)
    else:
        title, preview = source, body
    return source, str(title or "").strip(), str(preview or "").strip()


def _format_compression_section_item(*, source: str, title: str, preview: str) -> str:
    source_label = {
        "DSL": "设定",
        "GRAPH": "图谱",
        "RAG": "回忆",
        "OTHER": "证据",
    }.get(source, "证据")
    title_text = re.sub(r"\s+", " ", str(title or "").strip())
    preview_text = re.sub(r"\s+", " ", str(preview or "").strip())
    title_text = _truncate_text(title_text, 60)
    preview_text = _truncate_text(preview_text, 180)
    if title_text and preview_text:
        return f"[{source_label}] {title_text}: {preview_text}"
    if preview_text:
        return f"[{source_label}] {preview_text}"
    if title_text:
        return f"[{source_label}] {title_text}"
    return ""


def _build_compression_sections(
    *,
    summary: str,
    dsl_hits: list[dict[str, Any]],
    graph_facts: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {
        "fact_entities": [],
        "fact_relations": [],
        "retrieved_events": [],
    }
    seen: dict[str, set[str]] = {key: set() for key in sections}

    def _append(section_key: str, text: str, *, limit: int = 6) -> None:
        item = str(text or "").strip()
        if not item:
            return
        if section_key not in sections:
            return
        if item in seen[section_key]:
            return
        if len(sections[section_key]) >= max(limit, 1):
            return
        sections[section_key].append(item)
        seen[section_key].add(item)

    for line in str(summary or "").splitlines():
        source, title, preview = _parse_compression_line(line)
        section_key = _COMPRESSION_SECTION_BY_SOURCE.get(source)
        if not section_key:
            continue
        _append(
            section_key,
            _format_compression_section_item(source=source, title=title, preview=preview),
            limit=6,
        )

    source_rows = (
        ("DSL", dsl_hits, "fact_entities"),
        ("GRAPH", graph_facts, "fact_relations"),
        ("RAG", semantic_hits, "retrieved_events"),
    )
    for source, rows, section_key in source_rows:
        for item in rows:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("kind") or source).strip()
            preview = _hit_preview_text(item)
            _append(
                section_key,
                _format_compression_section_item(source=source, title=title, preview=preview),
                limit=4,
            )
            if len(sections[section_key]) >= 4:
                break

    if not any(sections.values()) and str(summary or "").strip():
        _append(
            "retrieved_events",
            _format_compression_section_item(
                source="OTHER",
                title="压缩摘要",
                preview=_truncate_text(str(summary), 220),
            ),
            limit=2,
        )
    return sections


def _context_compression_rerank_bias(source: str) -> float:
    source_name = str(source or "").strip().upper()
    if source_name == "DSL":
        return float(settings.context_compression_reranker_dsl_bias)
    if source_name == "GRAPH":
        return float(settings.context_compression_reranker_graph_bias)
    if source_name == "RAG":
        return float(settings.context_compression_reranker_rag_bias)
    return 0.0


def _resolve_reranker_device(requested_device: str, *, torch_module: Any) -> str:
    requested = str(requested_device or "").strip().lower()
    has_cuda = bool(getattr(getattr(torch_module, "cuda", None), "is_available", lambda: False)())
    if not requested or requested == "auto":
        return "cuda" if has_cuda else "cpu"
    if requested.startswith("cuda") and not has_cuda:
        return "cpu"
    return requested


def _load_context_compression_reranker() -> tuple[Any, Any, str] | None:
    if not bool(settings.context_compression_reranker_enabled):
        return None
    model_name = str(settings.context_compression_reranker_model or "").strip()
    runtime = str(getattr(settings, "context_compression_reranker_runtime", "onnx") or "onnx").strip().lower()
    onnx_path = str(getattr(settings, "context_compression_reranker_onnx_path", "") or "").strip()
    onnx_provider = str(getattr(settings, "context_compression_reranker_onnx_provider", "CPUExecutionProvider") or "").strip()
    requested_device = str(settings.context_compression_reranker_device or "").strip()
    if not model_name:
        return None
    if runtime not in {"onnx", "transformers", "torch", "auto"}:
        runtime = "onnx"
    cache_key = "|".join([model_name, runtime, onnx_path, onnx_provider, requested_device])
    global _RERANKER_MODEL, _RERANKER_TOKENIZER, _RERANKER_MODEL_NAME, _RERANKER_DEVICE, _RERANKER_RUNTIME, _RERANKER_UNAVAILABLE
    with _CONTEXT_COMPRESSOR_LOCK:
        if (
            _RERANKER_MODEL is not None
            and _RERANKER_TOKENIZER is not None
            and _RERANKER_MODEL_NAME == cache_key
        ):
            return _RERANKER_TOKENIZER, _RERANKER_MODEL, _RERANKER_DEVICE
        if _RERANKER_UNAVAILABLE and _RERANKER_MODEL_NAME == cache_key:
            return None

        if runtime in {"onnx", "auto"}:
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer

                model_path = onnx_path or model_name
                provider_name = onnx_provider or "CPUExecutionProvider"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                session = ort.InferenceSession(model_path, providers=[provider_name])
                _RERANKER_TOKENIZER = tokenizer
                _RERANKER_MODEL = {
                    "runtime": "onnx",
                    "session": session,
                }
                _RERANKER_MODEL_NAME = cache_key
                _RERANKER_DEVICE = provider_name
                _RERANKER_RUNTIME = "onnx"
                _RERANKER_UNAVAILABLE = False
                return tokenizer, _RERANKER_MODEL, provider_name
            except Exception as exc:
                _LOGGER.warning(
                    "context compression ONNX reranker load failed: %s",
                    exc,
                )
                if runtime == "onnx":
                    _RERANKER_MODEL = None
                    _RERANKER_TOKENIZER = None
                    _RERANKER_MODEL_NAME = cache_key
                    _RERANKER_DEVICE = "cpu"
                    _RERANKER_RUNTIME = "onnx"
                    _RERANKER_UNAVAILABLE = True
                    return None

        if runtime in {"transformers", "torch", "auto"}:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                resolved_device = _resolve_reranker_device(requested_device, torch_module=torch)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                model.to(resolved_device)
                model.eval()
                _RERANKER_TOKENIZER = tokenizer
                _RERANKER_MODEL = model
                _RERANKER_MODEL_NAME = cache_key
                _RERANKER_DEVICE = resolved_device
                _RERANKER_RUNTIME = "transformers"
                _RERANKER_UNAVAILABLE = False
                return tokenizer, model, resolved_device
            except Exception:
                _RERANKER_MODEL = None
                _RERANKER_TOKENIZER = None
                _RERANKER_MODEL_NAME = cache_key
                _RERANKER_DEVICE = "cpu"
                _RERANKER_RUNTIME = "transformers"
                _RERANKER_UNAVAILABLE = True
                return None

        _RERANKER_MODEL = None
        _RERANKER_TOKENIZER = None
        _RERANKER_MODEL_NAME = cache_key
        _RERANKER_DEVICE = "cpu"
        _RERANKER_UNAVAILABLE = True
        return None


def _sigmoid(x: float) -> float:
    clipped = max(min(float(x), 40.0), -40.0)
    return 1.0 / (1.0 + math.exp(-clipped))


def _score_lines_with_onnx_reranker(
    *,
    query: str,
    lines: list[str],
    tokenizer: Any,
    session: Any,
    batch_size: int,
    max_length: int,
) -> list[float] | None:
    try:
        input_names = {str(item.name) for item in session.get_inputs()}
    except Exception:
        return None
    if not input_names:
        return None

    scores: list[float] = []
    try:
        for start in range(0, len(lines), batch_size):
            batch_lines = lines[start : start + batch_size]
            encoded = tokenizer(
                [query] * len(batch_lines),
                batch_lines,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )
            ort_inputs: dict[str, Any] = {}
            for key, value in encoded.items():
                if key not in input_names:
                    continue
                try:
                    ort_inputs[key] = value.astype("int64")
                except Exception:
                    ort_inputs[key] = value
            if not ort_inputs:
                return None
            outputs = session.run(None, ort_inputs)
            if not outputs:
                return None
            logits = outputs[0]
            if hasattr(logits, "ndim") and int(logits.ndim) == 2:
                logits = logits[:, 0]
            raw_scores = logits.tolist() if hasattr(logits, "tolist") else list(logits)
            for value in raw_scores:
                try:
                    scores.append(_sigmoid(float(value)))
                except Exception:
                    scores.append(0.0)
    except Exception:
        return None

    if len(scores) != len(lines):
        return None
    return scores


def _score_lines_with_reranker(
    *,
    query: str,
    lines: list[str],
) -> list[float] | None:
    bundle = _load_context_compression_reranker()
    if bundle is None or not lines:
        return None
    tokenizer, model, device = bundle
    batch_size = max(int(settings.context_compression_reranker_batch_size), 1)
    max_length = max(int(settings.context_compression_reranker_max_length), 64)
    if isinstance(model, dict) and str(model.get("runtime")) == "onnx":
        session = model.get("session")
        if session is None:
            return None
        return _score_lines_with_onnx_reranker(
            query=query,
            lines=lines,
            tokenizer=tokenizer,
            session=session,
            batch_size=batch_size,
            max_length=max_length,
        )

    try:
        import torch
    except Exception:
        return None

    scores: list[float] = []
    try:
        with torch.no_grad():
            for start in range(0, len(lines), batch_size):
                batch_lines = lines[start : start + batch_size]
                encoded = tokenizer(
                    [query] * len(batch_lines),
                    batch_lines,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                outputs = model(**encoded)
                logits = outputs.logits
                if hasattr(logits, "dim") and int(logits.dim()) == 2:
                    logits = logits[:, 0]
                raw_scores = logits.detach().cpu().tolist()
                for value in raw_scores:
                    try:
                        scores.append(_sigmoid(float(value)))
                    except Exception:
                        scores.append(0.0)
    except Exception:
        return None

    if len(scores) != len(lines):
        return None
    return scores


def _call_context_compressor_reranker(
    *,
    user_input: str,
    intent: str,
    lines: list[str],
    max_chars: int,
    telemetry: dict[str, Any] | None = None,
) -> str | None:
    started_at = time.perf_counter()
    if telemetry is not None:
        telemetry.clear()
        telemetry.update(
            {
                "attempted": True,
                "line_count": len(lines),
                "max_chars": max(int(max_chars), 0),
            }
        )
    if not lines:
        if telemetry is not None:
            telemetry["reason"] = "empty_lines"
            telemetry["elapsed_ms"] = max(int((time.perf_counter() - started_at) * 1000), 0)
        return None
    query = (
        "你是写作检索压缩器，请优先保留实体、设定、动作约束。\n"
        + f"intent={_truncate_text(intent, 64)}\n"
        + f"user_input={_truncate_text(user_input, 600)}"
    )
    raw_scores = _score_lines_with_reranker(query=query, lines=lines)
    if raw_scores is None:
        if telemetry is not None:
            telemetry["reason"] = "reranker_unavailable"
            telemetry["runtime"] = _RERANKER_RUNTIME
            telemetry["device"] = _RERANKER_DEVICE
            telemetry["elapsed_ms"] = max(int((time.perf_counter() - started_at) * 1000), 0)
        return None

    min_score = max(min(float(settings.context_compression_reranker_min_score), 1.0), 0.0)
    top_k = max(int(settings.context_compression_reranker_top_k), 1)
    min_dsl_keep = max(int(settings.context_compression_reranker_min_dsl_keep), 0)
    if telemetry is not None:
        telemetry["min_score"] = round(min_score, 4)
        telemetry["top_k"] = top_k
        telemetry["min_dsl_keep"] = min_dsl_keep
        telemetry["runtime"] = _RERANKER_RUNTIME
        telemetry["device"] = _RERANKER_DEVICE

    ranked: list[tuple[float, float, int, str, str]] = []
    for idx, line in enumerate(lines):
        source = _compression_line_source(line)
        base_score = raw_scores[idx]
        dsl_hard_bias = 0.2 if source == "DSL" else 0.0
        adjusted = max(
            0.0,
            min(base_score + _context_compression_rerank_bias(source) + dsl_hard_bias, 1.5),
        )
        ranked.append((adjusted, base_score, idx, source, line))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    threshold_hit_count = sum(1 for item in ranked if item[0] >= min_score)

    forced: list[tuple[float, float, int, str, str]] = [
        item for item in ranked if item[3] == "DSL"
    ][:min_dsl_keep]
    selected_map: dict[int, tuple[float, float, int, str, str]] = {
        item[2]: item for item in forced
    }
    for item in ranked:
        if item[0] < min_score:
            continue
        selected_map[item[2]] = item
        if len(selected_map) >= top_k:
            break
    if not selected_map:
        for item in ranked[:top_k]:
            selected_map[item[2]] = item

    selected_ranked = sorted(selected_map.values(), key=lambda item: (item[0], item[1]), reverse=True)[:top_k]
    selected = sorted(selected_ranked, key=lambda item: item[2])

    chunks = _apply_source_priority_tiering([item[4] for item in selected_ranked], max_chars=max_chars)
    if not chunks and selected_ranked:
        chunks.append(_truncate_text(str(selected_ranked[0][4]), max_chars))
    result = "\n".join(chunks).strip()
    if telemetry is not None:
        selected_sources: dict[str, int] = {}
        for _, _, _, source, _ in selected_ranked:
            selected_sources[source] = selected_sources.get(source, 0) + 1
        kept_sources: dict[str, int] = {}
        for line in chunks:
            source = _compression_line_source(line)
            kept_sources[source] = kept_sources.get(source, 0) + 1
        kept_count = len(chunks)
        telemetry["threshold_hit_count"] = threshold_hit_count
        telemetry["selected_count"] = len(selected_ranked)
        telemetry["kept_count"] = kept_count
        telemetry["dropped_count"] = max(len(lines) - kept_count, 0)
        telemetry["selected_sources"] = selected_sources
        telemetry["kept_sources"] = kept_sources
        telemetry["output_chars"] = len(result)
        telemetry["reason"] = "ok" if result else "empty_result"
        telemetry["elapsed_ms"] = max(int((time.perf_counter() - started_at) * 1000), 0)
    return result or None


def _build_context_compression(
    *,
    user_input: str,
    intent: str,
    dsl_hits: list[dict[str, Any]],
    graph_facts: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not settings.context_compression_enabled:
        return None, {"enabled": False, "applied": False, "reason": "disabled"}

    mode = _normalize_context_compression_mode(settings.context_compression_mode)
    if mode == "off":
        return None, {"enabled": True, "applied": False, "reason": "mode_off", "mode": mode}

    min_chars = max(int(settings.context_compression_min_chars), 200)
    max_chars = max(int(settings.context_compression_max_chars), 200)
    corpus, lines = _build_context_compression_corpus(
        dsl_hits=dsl_hits,
        graph_facts=graph_facts,
        semantic_hits=semantic_hits,
    )
    chars_before = len(corpus)
    if chars_before < min_chars:
        return None, {
            "enabled": True,
            "applied": False,
            "reason": "below_threshold",
            "mode": mode,
            "chars_before": chars_before,
            "chars_after": 0,
            "min_chars": min_chars,
            "max_chars": max_chars,
        }

    summary = ""
    source = "heuristic"
    reranker_telemetry: dict[str, Any] | None = None
    if mode in {"rerank", "auto"}:
        reranker_telemetry = {}
        rerank_summary = _call_context_compressor_reranker(
            user_input=user_input,
            intent=intent,
            lines=lines,
            max_chars=max_chars,
            telemetry=reranker_telemetry,
        )
        if rerank_summary:
            summary = rerank_summary
            source = "rerank"

    if mode in {"llm", "auto"}:
        if not summary:
            llm_summary = _call_context_compressor_llm(
                user_input=user_input,
                intent=intent,
                corpus=corpus,
                max_chars=max_chars,
            )
            if llm_summary:
                summary = llm_summary
                source = "llm"

    if not summary:
        summary = _heuristic_context_compress(
            user_input=user_input,
            intent=intent,
            lines=lines,
            max_chars=max_chars,
        )
        if mode == "task_aware":
            source = "heuristic_task_aware"
        elif mode == "auto":
            source = "heuristic_auto_fallback"
        elif mode == "rerank":
            source = "heuristic_after_rerank"
        elif mode == "llm":
            source = "heuristic_after_llm"
        else:
            source = "heuristic"

    summary = summary.strip()
    if not summary:
        metadata = {
            "enabled": True,
            "applied": False,
            "reason": "empty_summary",
            "mode": mode,
            "source": source,
            "chars_before": chars_before,
            "chars_after": 0,
            "min_chars": min_chars,
            "max_chars": max_chars,
        }
        if reranker_telemetry:
            metadata["reranker"] = reranker_telemetry
        return None, metadata

    summary_text = _truncate_text(summary, max_chars)
    sections = _build_compression_sections(
        summary=summary_text,
        dsl_hits=dsl_hits,
        graph_facts=graph_facts,
        semantic_hits=semantic_hits,
    )
    compressed = {
        "intent": intent,
        "summary": summary_text,
        "source": source,
        "max_chars": max_chars,
        "resolver_order": ["DSL", "GRAPH", "RAG"],
        "sections": sections,
    }
    metadata = {
        "enabled": True,
        "applied": True,
        "reason": "ok",
        "mode": mode,
        "source": source,
        "chars_before": chars_before,
        "chars_after": len(summary_text),
        "min_chars": min_chars,
        "max_chars": max_chars,
        "line_count": len(lines),
        "priority_tiering": "DSL>GRAPH>RAG",
        "sections": {key: len(value) for key, value in sections.items()},
    }
    if reranker_telemetry:
        metadata["reranker"] = reranker_telemetry
    return compressed, metadata


def _normalize_self_reflective_mode(value: str | None) -> str:
    mode = str(value or "auto").strip().lower()
    if mode in {"off", "heuristic", "llm", "auto"}:
        return mode
    return "auto"


def _normalize_temperature_profile(value: str | None) -> str:
    profile = str(value or "").strip().lower()
    if profile in {"action", "chat", "ghost", "brainstorm"}:
        return profile
    return ""


def _normalize_followup_queries(raw: Any, *, limit: int) -> list[str]:
    if limit <= 0:
        return []
    values: list[str] = []
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = [str(item or "") for item in raw]
    normalized: list[str] = []
    for item in values:
        query = re.sub(r"\s+", " ", str(item or "").strip())
        if not query:
            continue
        if len(query) > 120:
            query = query[:120].rstrip()
        if len(query) < 2 or query in normalized:
            continue
        normalized.append(query)
        if len(normalized) >= limit:
            break
    return normalized


def _detect_negative_constraint_conflicts(
    *,
    user_input: str,
    negative_constraints: list[dict[str, Any]],
    max_items: int = 4,
) -> list[dict[str, Any]]:
    text = str(user_input or "").strip()
    if not text:
        return []
    normalized_text = re.sub(r"\s+", "", text).lower()
    if not normalized_text:
        return []

    # 用户显式“不要写/避免”时，视为主动遵守禁忌，不判定为冲突。
    if any(
        token in normalized_text
        for token in ("不要", "别写", "避免", "禁止", "不可", "不能", "不得", "不写", "别提")
    ):
        return []

    input_terms = {token.lower() for token in _extract_query_terms(text)}
    conflicts: list[dict[str, Any]] = []
    for item in negative_constraints:
        if not isinstance(item, dict):
            continue
        constraint_text = str(item.get("text", "") or "").strip()
        if not constraint_text:
            continue
        constraint_terms = {
            token.lower()
            for token in _extract_query_terms(constraint_text)
            if len(str(token or "").strip()) >= 2
        }
        if not constraint_terms:
            continue
        matched = sorted(term for term in constraint_terms if term in input_terms)
        if not matched:
            continue
        conflicts.append(
            {
                "text": _truncate_text(constraint_text, 160),
                "source": str(item.get("source") or "").strip(),
                "matched_terms": matched[:4],
            }
        )
        if len(conflicts) >= max(max_items, 1):
            break
    return conflicts


def _heuristic_self_reflective_review(
    *,
    user_input: str,
    intent: str,
    dsl_hits: list[dict[str, Any]],
    graph_facts: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
    negative_constraints: list[dict[str, Any]],
    max_queries: int,
) -> dict[str, Any]:
    issues: list[str] = []
    followup_queries: list[str] = []
    text = str(user_input or "").strip()
    if not text:
        return {
            "needs_refine": False,
            "issues": issues,
            "followup_queries": followup_queries,
            "confidence": 0.2,
            "source": "heuristic_empty_input",
        }

    if intent in {"brainstorm", "writing_help"} and len(semantic_hits) <= 1:
        issues.append("missing_semantic_evidence")
    if len(graph_facts) <= 1 and any(token in text for token in ("线索", "真相", "伏笔", "冲突", "推演", "智斗", "权谋")):
        issues.append("missing_graph_links")
    if len(dsl_hits) <= 1 and any(token in text for token in ("设定", "世界观", "规则", "身份", "地点", "物品")):
        issues.append("missing_dsl_constraints")

    conflicts = _detect_negative_constraint_conflicts(
        user_input=text,
        negative_constraints=negative_constraints,
        max_items=3,
    )
    if conflicts:
        issues.append("negative_constraint_conflict")

    if issues:
        base_query = _truncate_text(text, 96)
        if "missing_graph_links" in issues:
            followup_queries.append(base_query + " 关键关系 伏笔")
        if "missing_semantic_evidence" in issues:
            followup_queries.append(base_query + " 相关前情 章节证据")
        if "missing_dsl_constraints" in issues:
            followup_queries.append(base_query + " 世界观设定 规则")
        if "negative_constraint_conflict" in issues:
            followup_queries.append(base_query + " 禁忌约束 校验 重写")

    followup_queries = _normalize_followup_queries(followup_queries, limit=max_queries)
    return {
        "needs_refine": bool(followup_queries) or bool(conflicts),
        "issues": issues[:6],
        "followup_queries": followup_queries,
        "confidence": 0.62 if conflicts else (0.46 if followup_queries else 0.35),
        "negative_conflicts": conflicts[:3],
        "source": "heuristic",
    }


def _call_self_reflective_judge_llm(
    *,
    user_input: str,
    intent: str,
    temperature_profile: str,
    chapter_preview: str,
    scene_beat_text: str,
    dsl_hits: list[dict[str, Any]],
    graph_facts: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
    negative_constraints: list[dict[str, Any]],
    max_queries: int,
) -> dict[str, Any] | None:
    model = str(settings.lightrag_llm_model or "").strip()
    base_url = str(settings.lightrag_llm_base_url or "").strip()
    api_key = str(settings.lightrag_llm_api_key or "").strip()
    if not (model and base_url and api_key):
        return None

    dsl_preview = [
        {
            "title": str(item.get("title") or item.get("kind") or ""),
            "snippet": _truncate_text(str(item.get("snippet") or ""), 120),
        }
        for item in dsl_hits[:4]
        if isinstance(item, dict)
    ]
    graph_preview = [
        {
            "fact": _truncate_text(str(item.get("fact") or ""), 120),
            "confidence": item.get("confidence"),
        }
        for item in graph_facts[:4]
        if isinstance(item, dict)
    ]
    rag_preview = [
        {
            "title": str(item.get("title") or ""),
            "snippet": _truncate_text(str(item.get("snippet") or ""), 120),
            "citation": (
                item.get("citation")
                if isinstance(item.get("citation"), dict)
                else None
            ),
        }
        for item in semantic_hits[:4]
        if isinstance(item, dict)
    ]
    negative_preview = [
        {
            "text": _truncate_text(str(item.get("text") or ""), 140),
            "source": str(item.get("source") or ""),
            "title": str(item.get("title") or ""),
        }
        for item in negative_constraints[:6]
        if isinstance(item, dict)
    ]

    endpoint = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是小说上下文审视器（Judge）。"
                    "任务：检查当前检索是否遗漏关键事实、是否存在时序风险、是否有明显噪声，"
                    "并判断用户请求是否可能违反 negative_constraints 禁忌约束。"
                    "只输出 JSON，格式："
                    "{\"needs_refine\":true|false,"
                    "\"issues\":[\"missing_key_fact|temporal_risk|noisy_context|negative_constraint_conflict\"],"
                    "\"followup_queries\":[\"...\"],"
                    "\"confidence\":0-1,"
                    "\"reason\":\"...\"}。"
                    "若命中 negative_constraint_conflict，needs_refine 必须为 true。"
                    "followup_queries 最多给 2 条，每条短且可直接检索。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "intent": intent,
                        "temperature_profile": temperature_profile,
                        "user_input": _truncate_text(user_input, 820),
                        "chapter_preview": _truncate_text(chapter_preview, 540),
                        "scene_beat": _truncate_text(scene_beat_text, 360),
                        "retrieved": {
                            "dsl": dsl_preview,
                            "graph": graph_preview,
                            "rag": rag_preview,
                            "negative_constraints": negative_preview,
                        },
                        "max_queries": max(max_queries, 1),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        timeout = httpx.Timeout(float(settings.self_reflective_llm_timeout_seconds))
        with httpx.Client(timeout=timeout) as client:
            response = client.post(endpoint, json=body, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except Exception:
        return None

    content = str(payload.get("choices", [{}])[0].get("message", {}).get("content", "") or "")
    parsed = _extract_json_object(content)
    if not parsed:
        return None
    needs_refine = bool(parsed.get("needs_refine"))
    confidence = 0.0
    try:
        confidence = max(0.0, min(float(parsed.get("confidence", 0.0)), 1.0))
    except Exception:
        confidence = 0.0
    issues_raw = parsed.get("issues")
    issues: list[str] = []
    if isinstance(issues_raw, list):
        for item in issues_raw:
            token = str(item or "").strip().lower()
            if not token or token in issues:
                continue
            issues.append(token[:40])
            if len(issues) >= 6:
                break
    followup_queries = _normalize_followup_queries(parsed.get("followup_queries"), limit=max_queries)
    if "negative_constraint_conflict" in issues and not followup_queries:
        followup_queries = _normalize_followup_queries(
            [_truncate_text(str(user_input or ""), 96) + " 禁忌约束 校验 重写"],
            limit=max_queries,
        )
    if not followup_queries and "negative_constraint_conflict" not in issues:
        needs_refine = False
    return {
        "needs_refine": needs_refine,
        "issues": issues,
        "followup_queries": followup_queries,
        "confidence": confidence,
        "source": "llm",
    }


def _hit_identity(item: dict[str, Any], *, kind: str) -> str:
    if kind == "dsl":
        if item.get("id") is not None:
            return f"id:{item.get('project_id')}:{item.get('id')}:{item.get('kind')}"
        return f"title:{item.get('project_id')}:{item.get('title')}:{item.get('kind')}"
    if kind == "graph":
        if item.get("id") is not None:
            return f"id:{item.get('project_id')}:{item.get('id')}"
        fact_key = str(item.get("fact_key") or "").strip()
        if fact_key:
            return f"fact_key:{fact_key}"
        return f"fact:{item.get('project_id')}:{item.get('fact')}"
    if item.get("id") is not None:
        return f"id:{item.get('project_id')}:{item.get('id')}"
    citation = item.get("citation") if isinstance(item.get("citation"), dict) else {}
    citation_key = f"{citation.get('source')}|{citation.get('chunk')}"
    if citation_key != "None|None":
        return f"citation:{citation_key}"
    return f"title:{item.get('project_id')}:{item.get('title')}:{item.get('snippet')}"


def _merge_unique_hits(
    primary: list[dict[str, Any]],
    extra: list[dict[str, Any]],
    *,
    kind: str,
    limit: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for collection in (primary, extra):
        for item in collection:
            if not isinstance(item, dict):
                continue
            key = _hit_identity(item, kind=kind)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= max(limit, 1):
                return merged
    return merged


def _run_reflective_followup_retrieval(
    *,
    project_id: int,
    followup_queries: list[str],
    graph_anchor: str | None,
    rag_anchor: str | None,
    rag_mode: str,
    current_chapter_index: int | None,
    rag_short_circuit_enabled: bool,
    windowed_retrieval_settings: list[Any],
    windowed_retrieval_cards: list[Any],
    dsl_limit: int,
    graph_limit: int,
    rag_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    start = time.perf_counter()
    dsl_extra: list[dict[str, Any]] = []
    graph_extra: list[dict[str, Any]] = []
    rag_extra: list[dict[str, Any]] = []
    graph_timeout_seconds = _normalize_timeout(settings.retrieval_graph_timeout_seconds, 2.0)
    rag_timeout_seconds = _normalize_timeout(settings.retrieval_rag_timeout_seconds, 2.0)
    graph_remote_hits = 0
    rag_remote_hits = 0
    graph_timeouts = 0
    rag_timeouts = 0

    dsl_step_limit = max(min(dsl_limit, 4), 1)
    graph_step_limit = max(min(graph_limit, 4), 1)
    rag_step_limit = max(min(rag_limit, 4), 1)

    for query in followup_queries:
        terms = _extract_query_terms(query)
        dsl_extra.extend(
            _build_dsl_hits(
                terms,
                windowed_retrieval_settings,
                windowed_retrieval_cards,
                limit=dsl_step_limit,
            )
        )

        graph_future = _submit_graph_future(
            project_id,
            terms,
            graph_anchor,
            graph_step_limit,
            current_chapter=current_chapter_index,
        )
        graph_remote, graph_timed_out, graph_failed = _await_hits_future(graph_future, graph_timeout_seconds)
        if graph_remote:
            graph_extra.extend(graph_remote[:graph_step_limit])
            graph_remote_hits += len(graph_remote[:graph_step_limit])
        else:
            graph_extra.extend(
                _build_graph_facts(
                    windowed_retrieval_cards,
                    windowed_retrieval_settings,
                    graph_anchor,
                    limit=graph_step_limit,
                )
            )
        if graph_timed_out:
            graph_timeouts += 1
        if graph_failed:
            continue

        if rag_short_circuit_enabled:
            continue
        rag_future = _submit_rag_future(query, rag_anchor, rag_step_limit, rag_mode)
        rag_remote, rag_timed_out, rag_failed = _await_hits_future(rag_future, rag_timeout_seconds)
        if rag_remote:
            rag_extra.extend(rag_remote[:rag_step_limit])
            rag_remote_hits += len(rag_remote[:rag_step_limit])
        else:
            rag_extra.extend(
                _build_semantic_hits(
                    query,
                    windowed_retrieval_settings,
                    windowed_retrieval_cards,
                    rag_anchor,
                    limit=rag_step_limit,
                )
            )
        if rag_timed_out:
            rag_timeouts += 1
        if rag_failed:
            continue

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return (
        _apply_memory_decay(dsl_extra),
        _apply_memory_decay(graph_extra),
        _apply_memory_decay(rag_extra),
        {
            "elapsed_ms": elapsed_ms,
            "query_count": len(followup_queries),
            "graph_remote_hits": graph_remote_hits,
            "rag_remote_hits": rag_remote_hits,
            "graph_timeouts": graph_timeouts,
            "rag_timeouts": rag_timeouts,
        },
    )


def _build_context_cache_layers(
    *,
    mode: str,
    anchor: str | None,
    prompt_workshop_template: dict[str, Any] | None,
    working_settings: list[Any],
    working_cards: list[Any],
    semantic_settings: list[Any],
    current_chapter: dict[str, Any] | None,
    current_volume: dict[str, Any] | None,
    scene_beat_context: dict[str, Any] | None,
    latest_messages: list[Any],
    dsl_hits: list[dict[str, Any]],
    graph_facts: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
    negative_constraints: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    static_lines: list[str] = [
        f"pov_mode={mode}",
        f"pov_anchor={anchor or ''}",
        "resolver_order=DSL>GRAPH>RAG",
    ]
    if isinstance(prompt_workshop_template, dict):
        static_lines.append(f"template_name={str(prompt_workshop_template.get('name', '') or '')}")
        static_lines.append(
            "template_system_prompt="
            + _truncate_text(str(prompt_workshop_template.get("system_prompt", "") or ""), 1400)
        )

    ordered_working_settings = sorted(
        [row for row in working_settings if getattr(row, "key", None)],
        key=lambda row: str(getattr(row, "key", "") or ""),
    )
    for row in ordered_working_settings[:90]:
        static_lines.append(
            f"[setting] {str(getattr(row, 'key', '') or '')}: "
            + _truncate_text(_setting_value_text(row), 220)
        )
    ordered_working_cards = sorted(
        [row for row in working_cards if getattr(row, "title", None)],
        key=lambda row: str(getattr(row, "title", "") or ""),
    )
    for row in ordered_working_cards[:72]:
        static_lines.append(
            f"[card] {str(getattr(row, 'title', '') or '')}: "
            + _truncate_text(_card_content_text(row), 200)
        )
    static_prefix = _truncate_text("\n".join(static_lines), 32000)

    persistent_lines: list[str] = []
    ordered_semantic_settings = sorted(
        [row for row in semantic_settings if getattr(row, "key", None)],
        key=lambda row: str(getattr(row, "key", "") or ""),
    )
    for row in ordered_semantic_settings[:40]:
        persistent_lines.append(
            f"[semantic] {str(getattr(row, 'key', '') or '')}: "
            + _truncate_text(_setting_value_text(row), 260)
        )
    if isinstance(current_volume, dict):
        persistent_lines.append(
            "volume_outline="
            + _truncate_text(str(current_volume.get("outline", "") or ""), 2000)
        )
    if isinstance(current_chapter, dict):
        persistent_lines.append(
            "chapter_preview="
            + _truncate_text(str(current_chapter.get("content_preview", "") or ""), 2200)
        )
    if isinstance(scene_beat_context, dict):
        active = scene_beat_context.get("active")
        if isinstance(active, dict):
            persistent_lines.append(
                "scene_beat_active="
                + _truncate_text(str(active.get("content", "") or ""), 600)
            )
    persistent_prefix = _truncate_text("\n".join(persistent_lines), 20000)

    session_lines: list[str] = []
    for msg in latest_messages[-12:]:
        role = str(getattr(msg, "role", "") or "")
        content = _truncate_text(str(getattr(msg, "content", "") or ""), 320)
        session_lines.append(f"[{role}] {content}")
    for source, rows in (("DSL", dsl_hits), ("GRAPH", graph_facts), ("RAG", semantic_hits)):
        for row in rows[:8]:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or row.get("kind") or source)
            snippet = _truncate_text(_hit_preview_text(row), 180)
            session_lines.append(f"[{source}] {title}: {snippet}")
    for item in negative_constraints[:8]:
        if not isinstance(item, dict):
            continue
        text = _truncate_text(str(item.get("text", "") or ""), 180)
        if not text:
            continue
        source_name = str(item.get("source") or "NEG").strip().upper()
        session_lines.append(f"[NEGATIVE/{source_name}] {text}")
    session_prefix = _truncate_text("\n".join(session_lines), 12000)

    stable_prefix_hash = hashlib.sha1(
        (static_prefix + "\n\n" + persistent_prefix).encode("utf-8")
    ).hexdigest()
    layers = {
        "static_prefix": static_prefix,
        "persistent_prefix": persistent_prefix,
        "session_prefix": session_prefix,
        "stable_prefix_hash": stable_prefix_hash,
    }
    meta = {
        "enabled": bool(settings.context_cache_enabled),
        "static_chars": len(static_prefix),
        "persistent_chars": len(persistent_prefix),
        "session_chars": len(session_prefix),
        "stable_prefix_hash": stable_prefix_hash,
    }
    return layers, meta


def _memory_decay_factor(freshness_days: Any) -> float:
    try:
        days = max(float(freshness_days), 0.0)
    except Exception:
        return 1.0
    half_life = max(float(settings.memory_decay_half_life_days), 1.0)
    floor = max(min(float(settings.memory_decay_floor), 1.0), 0.0)
    decay = math.exp(-math.log(2.0) * days / half_life)
    return max(floor, min(1.0, decay))


def _apply_memory_decay(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decayed: list[dict[str, Any]] = []
    for item in hits:
        if not isinstance(item, dict):
            continue
        next_item = dict(item)
        freshness = next_item.get("freshness_days")
        factor = _memory_decay_factor(freshness)
        next_item["memory_decay"] = round(factor, 4)
        score_raw = next_item.get("score")
        confidence_raw = next_item.get("confidence")
        if isinstance(score_raw, (int, float)):
            next_item["score"] = round(float(score_raw) * factor, 4)
        if isinstance(confidence_raw, (int, float)):
            next_item["confidence"] = round(float(confidence_raw) * factor, 4)
        decayed.append(next_item)
    decayed.sort(
        key=lambda row: (
            float(row.get("score", row.get("confidence", 0.0)) or 0.0),
            -_safe_int(row.get("freshness_days"), 99999),
        ),
        reverse=True,
    )
    return decayed


def _build_spatial_graph(settings_rows: list[Any], cards_rows: list[Any]) -> tuple[dict[str, set[str]], dict[str, str]]:
    relation_keys = {item.lower() for item in settings.spatial_relation_keys}
    adjacency: dict[str, set[str]] = {}
    canonical_display: dict[str, str] = {}

    def _norm(value: str) -> str:
        return re.sub(r"\s+", "", str(value or "").strip()).lower()

    def _ensure(name: str) -> str:
        display = str(name or "").strip()
        if not display:
            return ""
        key = _norm(display)
        if not key:
            return ""
        canonical_display.setdefault(key, display)
        adjacency.setdefault(key, set())
        return key

    def _link(left: str, right: str) -> None:
        a = _ensure(left)
        b = _ensure(right)
        if not a or not b or a == b:
            return
        adjacency[a].add(b)
        adjacency[b].add(a)

    for row in cards_rows:
        title = str(getattr(row, "title", "") or "").strip()
        content = getattr(row, "content", {}) if isinstance(getattr(row, "content", {}), dict) else {}
        if not title:
            continue
        for key, raw in content.items():
            if str(key or "").strip().lower() not in relation_keys:
                continue
            if isinstance(raw, list):
                for item in raw:
                    _link(title, str(item))
            elif isinstance(raw, dict):
                for target in raw.keys():
                    _link(title, str(target))
            else:
                _link(title, str(raw))

    for row in settings_rows:
        key_text = str(getattr(row, "key", "") or "").strip()
        value = getattr(row, "value", {})
        if not key_text or not isinstance(value, dict):
            continue
        if "地点" not in key_text and "地理" not in key_text and "区域" not in key_text:
            continue
        for source, raw in value.items():
            source_name = str(source or "").strip()
            if not source_name:
                continue
            if isinstance(raw, list):
                for item in raw:
                    _link(source_name, str(item))
            elif isinstance(raw, dict):
                for target in raw.keys():
                    _link(source_name, str(target))
            else:
                _link(source_name, str(raw))

    return adjacency, canonical_display


def _build_spatial_distance_map(
    adjacency: dict[str, set[str]],
    *,
    current_location: str,
    max_hops: int,
) -> dict[str, int]:
    start = re.sub(r"\s+", "", str(current_location or "").strip()).lower()
    if not start or start not in adjacency:
        return {}
    visited: dict[str, int] = {start: 0}
    queue: list[str] = [start]
    while queue:
        node = queue.pop(0)
        depth = visited.get(node, 0)
        if depth >= max_hops:
            continue
        for nxt in adjacency.get(node, set()):
            if nxt in visited:
                continue
            visited[nxt] = depth + 1
            queue.append(nxt)
    return visited


def _hit_spatial_distance(hit: dict[str, Any], distance_map: dict[str, int]) -> int | None:
    if not distance_map:
        return None
    corpus = " ".join(
        [
            str(hit.get("title", "") or ""),
            str(hit.get("snippet", "") or ""),
            str(hit.get("fact", "") or ""),
        ]
    )
    normalized = re.sub(r"\s+", "", corpus).lower()
    if not normalized:
        return None
    distances: list[int] = []
    for token, distance in distance_map.items():
        if token and token in normalized:
            distances.append(distance)
    if not distances:
        return None
    return min(distances)


def _apply_spatial_penalty(
    hits: list[dict[str, Any]],
    *,
    current_location: str | None,
    settings_rows: list[Any],
    cards_rows: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not settings.spatial_enabled:
        return hits, {"enabled": False, "reason": "disabled"}
    location = str(current_location or "").strip()
    if not location:
        return hits, {"enabled": False, "reason": "missing_location"}

    adjacency, canonical_display = _build_spatial_graph(settings_rows, cards_rows)
    if not adjacency:
        return hits, {"enabled": False, "reason": "no_spatial_graph"}
    distance_map = _build_spatial_distance_map(
        adjacency,
        current_location=location,
        max_hops=max(int(settings.spatial_max_hops), 1),
    )
    if not distance_map:
        return hits, {"enabled": False, "reason": "location_not_found", "current_location": location}

    lam = max(float(settings.spatial_penalty_lambda), 0.0)
    adjusted: list[dict[str, Any]] = []
    penalized_count = 0
    for item in hits:
        if not isinstance(item, dict):
            continue
        distance = _hit_spatial_distance(item, distance_map)
        if distance is None:
            adjusted.append(dict(item))
            continue
        factor = math.exp(-lam * float(distance))
        next_item = dict(item)
        next_item["spatial_distance"] = distance
        next_item["spatial_factor"] = round(factor, 4)
        score_raw = next_item.get("score")
        conf_raw = next_item.get("confidence")
        if isinstance(score_raw, (int, float)):
            next_item["score"] = round(float(score_raw) * factor, 4)
        if isinstance(conf_raw, (int, float)):
            next_item["confidence"] = round(float(conf_raw) * factor, 4)
        if distance > 0:
            penalized_count += 1
        adjusted.append(next_item)

    adjusted.sort(
        key=lambda row: (
            float(row.get("score", row.get("confidence", 0.0)) or 0.0),
            -_safe_int(row.get("freshness_days"), 99999),
        ),
        reverse=True,
    )
    return adjusted, {
        "enabled": True,
        "reason": "ok",
        "current_location": location,
        "graph_nodes": len(adjacency),
        "reachable_nodes": len(distance_map),
        "penalized_hits": penalized_count,
        "penalty_lambda": lam,
        "known_locations": [canonical_display.get(key, key) for key in list(distance_map.keys())[:16]],
    }


def _split_memory_layers(settings_rows: list[Any]) -> tuple[list[Any], list[Any]]:
    semantic_prefix = str(settings.memory_semantic_key_prefix or "memory.semantic.volume.").strip()
    if not semantic_prefix:
        return settings_rows, []
    working: list[Any] = []
    semantic: list[Any] = []
    for row in settings_rows:
        key_text = str(getattr(row, "key", "") or "")
        if key_text.startswith(semantic_prefix):
            semantic.append(row)
        else:
            working.append(row)
    return working, semantic


def compile_context_bundle(
    db: Session,
    *,
    session_id: int | None,
    project_id: int,
    chapter_id: int | None,
    scene_beat_id: int | None,
    prompt_template_id: int | None,
    user_input: str,
    pov_mode: str | None,
    pov_anchor: str | None,
    rag_mode_override: str | None = None,
    deterministic_first: bool = False,
    thinking_enabled: bool = False,
    reference_project_ids: list[int] | None = None,
    context_window_profile: str | None = None,
    budget_mode: str | None = None,
    current_location: str | None = None,
    temperature_profile: str | None = None,
) -> CompiledContextBundle:
    compile_started_at = time.perf_counter()
    mode, anchor, notes = _normalize_pov(pov_mode, pov_anchor)
    terms = _extract_query_terms(user_input)
    context_window, context_window_source = _resolve_context_window_policy(context_window_profile)

    chapter_preview_for_router = ""
    scene_beat_text_for_router = ""
    preloaded_chapter = get_project_chapter(db, project_id, int(chapter_id)) if chapter_id is not None else None
    if preloaded_chapter is not None:
        chapter_preview_for_router = str(getattr(preloaded_chapter, "content", "") or "")[:900]
        try:
            beats_for_router = list(
                list_scene_beats(
                    db,
                    project_id=project_id,
                    chapter_id=int(getattr(preloaded_chapter, "id", 0) or 0),
                )
            )
        except ValueError:
            beats_for_router = []
        if beats_for_router:
            selected = None
            if scene_beat_id is not None:
                selected = next(
                    (item for item in beats_for_router if int(getattr(item, "id", 0) or 0) == int(scene_beat_id)),
                    None,
                )
            if selected is None:
                selected = next((item for item in beats_for_router if str(getattr(item, "status", "")) == "pending"), None)
            if selected is None:
                selected = beats_for_router[0]
            scene_beat_text_for_router = str(getattr(selected, "content", "") or "")

    semantic_route = _resolve_semantic_route(
        user_input=user_input,
        chapter_preview=chapter_preview_for_router,
        scene_beat_text=scene_beat_text_for_router,
    )

    effective_rag_mode_override = rag_mode_override
    rag_override_source = "request_override" if rag_mode_override else ""
    if effective_rag_mode_override is None and semantic_route.rag_mode:
        effective_rag_mode_override = semantic_route.rag_mode
        rag_override_source = "semantic_router"

    rag_mode, rag_route_reason, rag_route_source = _resolve_rag_route(user_input, terms, effective_rag_mode_override)
    if rag_override_source == "semantic_router":
        rag_route_reason = f"semantic_router_{semantic_route.intent}"
        rag_route_source = "semantic_router"

    effective_budget_mode = budget_mode
    if effective_budget_mode is None and semantic_route.budget_mode:
        effective_budget_mode = semantic_route.budget_mode
        notes.append(
            f"semantic_router 命中 {semantic_route.intent}，budget_mode 自动路由为 {semantic_route.budget_mode}。"
        )

    dynamic_budget_plan = _resolve_dynamic_budget_plan(
        base_policy=context_window,
        user_input=user_input,
        request_budget_mode=effective_budget_mode,
        chapter_preview=chapter_preview_for_router,
        scene_beat_text=scene_beat_text_for_router,
    )

    recent_messages = (
        list_messages(db, session_id, limit=dynamic_budget_plan.recent_messages_limit) if session_id is not None else []
    )
    all_settings, all_cards, context_pack_meta = _load_context_pack(db, project_id)
    scoped_settings, scoped_cards = _apply_pov_filter(all_settings, all_cards, mode, anchor)
    normalized_reference_project_ids = _normalize_reference_project_ids(
        reference_project_ids,
        current_project_id=project_id,
    )
    referenced_settings, referenced_cards, reference_project_meta = _load_reference_context(
        db,
        normalized_reference_project_ids,
    )
    scoped_reference_settings, scoped_reference_cards = _apply_pov_filter(
        referenced_settings,
        referenced_cards,
        mode,
        anchor,
    )
    project_working_settings, project_semantic_settings = _split_memory_layers(scoped_settings)
    ref_working_settings, ref_semantic_settings = _split_memory_layers(scoped_reference_settings)
    retrieval_settings = [*project_working_settings, *ref_working_settings, *project_semantic_settings, *ref_semantic_settings]
    retrieval_cards = [*scoped_cards, *scoped_reference_cards]
    windowed_retrieval_settings = _window_retrieval_rows(
        retrieval_settings,
        limit=dynamic_budget_plan.retrieval_settings_limit,
        terms=terms,
        source_text_getter=_setting_source_text,
    )
    windowed_retrieval_cards = _window_retrieval_rows(
        retrieval_cards,
        limit=dynamic_budget_plan.retrieval_cards_limit,
        terms=terms,
        source_text_getter=_card_source_text,
    )
    context_rows_meta = {
        "recent_messages": len(recent_messages),
        "project_settings": len(all_settings),
        "project_cards": len(all_cards),
        "project_settings_scoped": len(scoped_settings),
        "project_cards_scoped": len(scoped_cards),
        "reference_settings_scoped": len(scoped_reference_settings),
        "reference_cards_scoped": len(scoped_reference_cards),
        "working_settings": len(project_working_settings) + len(ref_working_settings),
        "semantic_settings": len(project_semantic_settings) + len(ref_semantic_settings),
        "retrieval_settings": len(retrieval_settings),
        "retrieval_cards": len(retrieval_cards),
        "windowed_retrieval_settings": len(windowed_retrieval_settings),
        "windowed_retrieval_cards": len(windowed_retrieval_cards),
    }

    prompt_workshop_template: dict[str, Any] | None = None
    prompt_workshop_knowledge_settings: list[dict[str, Any]] = []
    prompt_workshop_knowledge_cards: list[dict[str, Any]] = []
    prompt_workshop_reason = "not_selected"
    if prompt_template_id is not None:
        prompt_template = get_prompt_template(db, project_id, int(prompt_template_id))
        if prompt_template is None:
            prompt_workshop_reason = "template_not_found"
            notes.append(f"prompt_template_id={prompt_template_id} 未命中，已忽略模板注入。")
        else:
            knowledge_setting_keys = {
                str(item).strip()
                for item in (getattr(prompt_template, "knowledge_setting_keys", []) or [])
                if str(item).strip()
            }
            knowledge_card_ids = {
                int(item)
                for item in (getattr(prompt_template, "knowledge_card_ids", []) or [])
                if isinstance(item, int) and int(item) > 0
            }
            if knowledge_setting_keys:
                for row in scoped_settings:
                    if row.key not in knowledge_setting_keys:
                        continue
                    prompt_workshop_knowledge_settings.append(
                        {
                            "id": row.id,
                            "key": row.key,
                            "value_preview": _truncate_text(_setting_value_text(row), 500),
                            "freshness_days": _freshness_days(row.updated_at),
                        }
                    )
            if knowledge_card_ids:
                for row in scoped_cards:
                    if int(row.id) not in knowledge_card_ids:
                        continue
                    prompt_workshop_knowledge_cards.append(
                        {
                            "id": row.id,
                            "title": row.title,
                            "content_preview": _truncate_text(_card_content_text(row), 500),
                            "freshness_days": _freshness_days(row.updated_at),
                        }
                    )
            prompt_workshop_template = {
                "id": int(getattr(prompt_template, "id")),
                "name": str(getattr(prompt_template, "name", "") or ""),
                "system_prompt": str(getattr(prompt_template, "system_prompt", "") or ""),
                "user_prompt_prefix": str(getattr(prompt_template, "user_prompt_prefix", "") or ""),
            }
            prompt_workshop_reason = "ok"

    prompt_workshop_meta: dict[str, Any] = {
        "enabled": prompt_workshop_template is not None,
        "reason": prompt_workshop_reason,
        "requested_template_id": prompt_template_id,
        "injected_settings": len(prompt_workshop_knowledge_settings),
        "injected_cards": len(prompt_workshop_knowledge_cards),
    }
    if prompt_workshop_template is not None:
        prompt_workshop_meta.update(
            {
                "template_id": prompt_workshop_template.get("id"),
                "template_name": prompt_workshop_template.get("name"),
            }
        )

    chapter_context_reason = "not_requested"
    current_chapter: dict[str, Any] | None = None
    current_chapter_row = None
    current_volume: dict[str, Any] | None = None
    scene_beat_context: dict[str, Any] | None = None
    outline_context_reason = "chapter_not_requested"
    if chapter_id is not None:
        chapter = preloaded_chapter if preloaded_chapter is not None else get_project_chapter(db, project_id, int(chapter_id))
        if chapter is None:
            chapter_context_reason = "chapter_not_found"
            notes.append(f"chapter_id={chapter_id} 未命中，已忽略章节上下文。")
        else:
            current_chapter_row = chapter
            chapter_content = str(getattr(chapter, "content", "") or "")
            current_chapter = {
                "id": int(getattr(chapter, "id")),
                "volume_id": int(getattr(chapter, "volume_id", 0) or 0) or None,
                "chapter_index": int(getattr(chapter, "chapter_index")),
                "title": str(getattr(chapter, "title", "") or ""),
                "version": int(getattr(chapter, "version", 0) or 0),
                "updated_at": _safe_iso(getattr(chapter, "updated_at", None)),
                "content": _truncate_text(chapter_content, context_window.chapter_content_chars),
                "content_preview": _truncate_text(chapter_content, context_window.chapter_preview_chars),
                "total_chars": len(chapter_content),
            }
            chapter_context_reason = "ok"

    chapter_context_meta: dict[str, Any] = {
        "enabled": current_chapter is not None,
        "reason": chapter_context_reason,
        "requested_chapter_id": chapter_id,
    }
    if current_chapter is not None:
        chapter_context_meta.update(
            {
                "chapter_id": current_chapter.get("id"),
                "chapter_index": current_chapter.get("chapter_index"),
                "chapter_version": current_chapter.get("version"),
                "updated_at": current_chapter.get("updated_at"),
                "total_chars": current_chapter.get("total_chars"),
            }
        )

    if current_chapter_row is not None:
        resolved_volume_id = int(getattr(current_chapter_row, "volume_id", 0) or 0)
        if resolved_volume_id > 0:
            volume_row = get_project_volume(db, project_id, resolved_volume_id)
            if volume_row is not None:
                current_volume = {
                    "id": int(getattr(volume_row, "id", 0) or 0),
                    "volume_index": int(getattr(volume_row, "volume_index", 0) or 0),
                    "title": str(getattr(volume_row, "title", "") or ""),
                    "outline": _truncate_text(str(getattr(volume_row, "outline", "") or ""), 1800),
                    "outline_preview": _truncate_text(str(getattr(volume_row, "outline", "") or ""), 420),
                    "updated_at": _safe_iso(getattr(volume_row, "updated_at", None)),
                }
                outline_context_reason = "ok"
            else:
                outline_context_reason = "volume_not_found"
                notes.append(f"volume_id={resolved_volume_id} 未命中，已忽略卷纲注入。")
        else:
            outline_context_reason = "volume_not_bound"

        try:
            beats = list(list_scene_beats(db, project_id=project_id, chapter_id=int(getattr(current_chapter_row, "id"))))
        except ValueError:
            beats = []

        active_beat_row = None
        if scene_beat_id is not None and beats:
            active_beat_row = next((item for item in beats if int(getattr(item, "id", 0) or 0) == int(scene_beat_id)), None)
            if active_beat_row is None:
                notes.append(f"scene_beat_id={scene_beat_id} 未命中，已退回章节默认 Beat。")
        if active_beat_row is None and beats:
            active_beat_row = next((item for item in beats if str(getattr(item, "status", "")) == "pending"), None)
        if active_beat_row is None and beats:
            active_beat_row = beats[0]

        if active_beat_row is not None:
            active_index = next(
                (idx for idx, item in enumerate(beats) if int(getattr(item, "id", 0) or 0) == int(getattr(active_beat_row, "id", 0) or 0)),
                0,
            )
            prev_row = beats[active_index - 1] if active_index > 0 else None
            next_row = beats[active_index + 1] if active_index + 1 < len(beats) else None
            scene_beat_context = {
                "active": {
                    "id": int(getattr(active_beat_row, "id", 0) or 0),
                    "beat_index": int(getattr(active_beat_row, "beat_index", 0) or 0),
                    "content": _truncate_text(str(getattr(active_beat_row, "content", "") or ""), 600),
                    "status": str(getattr(active_beat_row, "status", "") or "pending"),
                },
                "previous": (
                    {
                        "id": int(getattr(prev_row, "id", 0) or 0),
                        "beat_index": int(getattr(prev_row, "beat_index", 0) or 0),
                        "content": _truncate_text(str(getattr(prev_row, "content", "") or ""), 260),
                    }
                    if prev_row is not None
                    else None
                ),
                "next": (
                    {
                        "id": int(getattr(next_row, "id", 0) or 0),
                        "beat_index": int(getattr(next_row, "beat_index", 0) or 0),
                        "content": _truncate_text(str(getattr(next_row, "content", "") or ""), 260),
                    }
                    if next_row is not None
                    else None
                ),
                "total": len(beats),
            }
        elif beats:
            scene_beat_context = None
        else:
            outline_context_reason = "ok_no_beats" if outline_context_reason.startswith("ok") else outline_context_reason

    outline_context_meta: dict[str, Any] = {
        "enabled": bool(current_volume),
        "reason": outline_context_reason,
        "requested_scene_beat_id": scene_beat_id,
        "selected_scene_beat_id": (
            int(scene_beat_context["active"]["id"]) if scene_beat_context and isinstance(scene_beat_context.get("active"), dict) else None
        ),
    }

    dsl_hits = _build_dsl_hits(
        terms,
        windowed_retrieval_settings,
        windowed_retrieval_cards,
        limit=dynamic_budget_plan.dsl_limit,
    )
    dsl_hits = _apply_memory_decay(dsl_hits)[: max(dynamic_budget_plan.dsl_limit, 1)]
    graph_anchor = anchor if mode == "character" else None
    rag_anchor = anchor if mode == "character" else None
    current_chapter_index = int(current_chapter.get("chapter_index", 0) or 0) if isinstance(current_chapter, dict) else None
    graph_limit = max(dynamic_budget_plan.graph_limit, 1)
    rag_limit = max(dynamic_budget_plan.rag_limit, 1)
    parallel_enabled = bool(settings.retrieval_parallel_enabled)
    graph_timeout_seconds = _normalize_timeout(settings.retrieval_graph_timeout_seconds, 2.0)
    rag_timeout_seconds = _normalize_timeout(settings.retrieval_rag_timeout_seconds, 2.0)
    retrieval_cache_ttl = _cache_ttl_seconds()

    graph_cache_key = _graph_cache_key(
        project_id,
        terms,
        graph_anchor,
        graph_limit,
        current_chapter=current_chapter_index,
    )
    graph_hits_remote, graph_cache_status = _cache_get(_GRAPH_HITS_CACHE, graph_cache_key)
    graph_future: Future[list[dict[str, Any]]] | None = None
    graph_timed_out = False
    graph_failed = False
    if graph_hits_remote is None:
        graph_future = _submit_graph_future(
            project_id,
            terms,
            graph_anchor,
            graph_limit,
            current_chapter=current_chapter_index,
        )

    rag_cache_key = _rag_cache_key(user_input, rag_anchor, rag_mode, rag_limit)
    semantic_hits_remote, rag_cache_status = _cache_get(_RAG_HITS_CACHE, rag_cache_key)
    rag_future: Future[list[dict[str, Any]]] | None = None
    rag_future_started_at: float | None = None
    rag_timed_out = False
    rag_failed = False

    if semantic_hits_remote is None and parallel_enabled and not deterministic_first:
        rag_future_started_at = time.monotonic()
        rag_future = _submit_rag_future(user_input, rag_anchor, rag_limit, rag_mode)

    if graph_hits_remote is None and graph_future is not None:
        graph_hits_remote, graph_timed_out, graph_failed = _await_hits_future(
            graph_future,
            graph_timeout_seconds,
        )
        if graph_hits_remote:
            _cache_set(_GRAPH_HITS_CACHE, graph_cache_key, graph_hits_remote)
            graph_cache_status = "set"
        elif graph_timed_out:
            graph_cache_status = "timeout"
        elif graph_failed:
            graph_cache_status = "error"
        else:
            graph_cache_status = "empty"

    if graph_hits_remote:
        graph_facts = _apply_memory_decay(graph_hits_remote)[:graph_limit]
        graph_provider = "neo4j_cache" if graph_cache_status == "hit" else "neo4j"
    else:
        graph_facts = _apply_memory_decay(
            _build_graph_facts(
                windowed_retrieval_cards,
                windowed_retrieval_settings,
                graph_anchor,
                limit=graph_limit,
            )
        )[:graph_limit]
        if graph_timed_out:
            graph_provider = "neo4j_timeout_local_graph_fallback"
        elif graph_failed:
            graph_provider = "neo4j_error_local_graph_fallback"
        else:
            graph_provider = "local_graph_fallback"

    rag_short_circuit_enabled, rag_short_circuit_reason = _resolve_rag_short_circuit(
        deterministic_first=deterministic_first,
        dsl_hits=dsl_hits,
        graph_facts=graph_facts,
    )
    if rag_short_circuit_enabled:
        semantic_hits = []
        rag_provider = "skipped_by_deterministic_short_circuit"
        if rag_future is not None:
            rag_future.cancel()
    else:
        if semantic_hits_remote is None:
            if rag_future is None:
                rag_future_started_at = time.monotonic()
                rag_future = _submit_rag_future(user_input, rag_anchor, rag_limit, rag_mode)
            elapsed = (time.monotonic() - rag_future_started_at) if rag_future_started_at is not None else 0.0
            wait_timeout = max(rag_timeout_seconds - elapsed, 0.05)
            semantic_hits_remote, rag_timed_out, rag_failed = _await_hits_future(rag_future, wait_timeout)
            if semantic_hits_remote:
                _cache_set(_RAG_HITS_CACHE, rag_cache_key, semantic_hits_remote)
                rag_cache_status = "set"
            elif rag_timed_out:
                rag_cache_status = "timeout"
            elif rag_failed:
                rag_cache_status = "error"
            else:
                rag_cache_status = "empty"
        if semantic_hits_remote:
            semantic_hits = _apply_memory_decay(semantic_hits_remote)[:rag_limit]
            rag_provider = "lightrag_cache" if rag_cache_status == "hit" else "lightrag"
        else:
            semantic_hits = _apply_memory_decay(
                _build_semantic_hits(
                    user_input,
                    windowed_retrieval_settings,
                    windowed_retrieval_cards,
                    rag_anchor,
                    limit=rag_limit,
                )
            )[:rag_limit]
            if rag_timed_out:
                rag_provider = "lightrag_timeout_local_semantic_fallback"
            elif rag_failed:
                rag_provider = "lightrag_error_local_semantic_fallback"
            else:
                rag_provider = "local_semantic_fallback"

    # 反思护栏在 followup 前先基于当前证据抽取一次禁忌约束。
    reflective_negative_constraints, _ = _build_negative_constraints(
        user_input=user_input,
        dsl_hits=dsl_hits,
        graph_facts=graph_facts,
        semantic_hits=semantic_hits,
        limit=8,
    )
    reflective_mode = _normalize_self_reflective_mode(settings.self_reflective_mode)
    normalized_temperature_profile = _normalize_temperature_profile(temperature_profile)
    reflective_triggered = False
    reflective_trigger_reason = "not_enabled"
    if settings.self_reflective_enabled and reflective_mode != "off":
        reflective_trigger_reason = "condition_not_met"
        if (
            bool(settings.self_reflective_brainstorm_trigger_enabled)
            and normalized_temperature_profile == "brainstorm"
        ):
            reflective_triggered = True
            reflective_trigger_reason = "brainstorm_temperature_profile"
        elif (
            bool(settings.self_reflective_low_confidence_trigger_enabled)
            and float(semantic_route.confidence) < float(settings.self_reflective_low_confidence_threshold)
        ):
            reflective_triggered = True
            reflective_trigger_reason = "semantic_router_low_confidence"
    elif reflective_mode == "off":
        reflective_trigger_reason = "mode_off"

    reflective_meta: dict[str, Any] = {
        "enabled": bool(settings.self_reflective_enabled),
        "mode": reflective_mode,
        "max_rounds": max(int(settings.self_reflective_max_rounds), 1),
        "triggered": reflective_triggered,
        "trigger_reason": reflective_trigger_reason,
        "source": "none",
        "needs_refine": False,
        "confidence": 0.0,
        "issues": [],
        "followup_queries": [],
        "query_count": 0,
        "applied": False,
        "added": {"dsl": 0, "graph": 0, "rag": 0},
        "negative_constraint_count": len(reflective_negative_constraints),
        "negative_conflicts": [],
        "elapsed_ms": 0,
        "followup_runtime": {
            "elapsed_ms": 0,
            "query_count": 0,
            "graph_remote_hits": 0,
            "rag_remote_hits": 0,
            "graph_timeouts": 0,
            "rag_timeouts": 0,
        },
    }
    if reflective_triggered:
        reflective_started_at = time.perf_counter()
        max_queries = max(int(settings.self_reflective_max_followup_queries), 1)
        judge_result: dict[str, Any] | None = None
        if reflective_mode in {"llm", "auto"}:
            judge_result = _call_self_reflective_judge_llm(
                user_input=user_input,
                intent=semantic_route.intent,
                temperature_profile=normalized_temperature_profile,
                chapter_preview=chapter_preview_for_router,
                scene_beat_text=scene_beat_text_for_router,
                dsl_hits=dsl_hits,
                graph_facts=graph_facts,
                semantic_hits=semantic_hits,
                negative_constraints=reflective_negative_constraints,
                max_queries=max_queries,
            )
            if judge_result:
                reflective_meta["source"] = "llm"
        if judge_result is None:
            judge_result = _heuristic_self_reflective_review(
                user_input=user_input,
                intent=semantic_route.intent,
                dsl_hits=dsl_hits,
                graph_facts=graph_facts,
                semantic_hits=semantic_hits,
                negative_constraints=reflective_negative_constraints,
                max_queries=max_queries,
            )
            reflective_meta["source"] = str(judge_result.get("source") or "heuristic")

        followup_queries = _normalize_followup_queries(
            judge_result.get("followup_queries"),
            limit=max_queries,
        )
        issues = (
            [str(item) for item in judge_result.get("issues", []) if str(item).strip()]
            if isinstance(judge_result.get("issues"), list)
            else []
        )
        has_negative_conflict = "negative_constraint_conflict" in issues
        needs_refine = bool(judge_result.get("needs_refine")) and (
            bool(followup_queries) or has_negative_conflict
        )
        confidence = 0.0
        try:
            confidence = max(0.0, min(float(judge_result.get("confidence", 0.0)), 1.0))
        except Exception:
            confidence = 0.0
        negative_conflicts = (
            [item for item in judge_result.get("negative_conflicts", []) if isinstance(item, dict)]
            if isinstance(judge_result.get("negative_conflicts"), list)
            else []
        )

        reflective_meta.update(
            {
                "needs_refine": needs_refine,
                "confidence": round(confidence, 4),
                "issues": issues[:6],
                "followup_queries": followup_queries,
                "query_count": len(followup_queries),
                "negative_conflicts": negative_conflicts[:4],
            }
        )

        if needs_refine and followup_queries:
            dsl_extra, graph_extra, rag_extra, followup_runtime = _run_reflective_followup_retrieval(
                project_id=project_id,
                followup_queries=followup_queries,
                graph_anchor=graph_anchor,
                rag_anchor=rag_anchor,
                rag_mode=rag_mode,
                current_chapter_index=current_chapter_index,
                rag_short_circuit_enabled=rag_short_circuit_enabled,
                windowed_retrieval_settings=windowed_retrieval_settings,
                windowed_retrieval_cards=windowed_retrieval_cards,
                dsl_limit=max(dynamic_budget_plan.dsl_limit + max_queries * 2, dynamic_budget_plan.dsl_limit),
                graph_limit=max(dynamic_budget_plan.graph_limit + max_queries * 2, dynamic_budget_plan.graph_limit),
                rag_limit=max(dynamic_budget_plan.rag_limit + max_queries * 2, dynamic_budget_plan.rag_limit),
            )
            before_counts = (len(dsl_hits), len(graph_facts), len(semantic_hits))
            dsl_hits = _merge_unique_hits(
                dsl_hits,
                dsl_extra,
                kind="dsl",
                limit=max(dynamic_budget_plan.dsl_limit + max_queries * 2, dynamic_budget_plan.dsl_limit),
            )
            graph_facts = _merge_unique_hits(
                graph_facts,
                graph_extra,
                kind="graph",
                limit=max(dynamic_budget_plan.graph_limit + max_queries * 2, dynamic_budget_plan.graph_limit),
            )
            semantic_hits = _merge_unique_hits(
                semantic_hits,
                rag_extra,
                kind="rag",
                limit=max(dynamic_budget_plan.rag_limit + max_queries * 2, dynamic_budget_plan.rag_limit),
            )
            reflective_meta["applied"] = True
            reflective_meta["added"] = {
                "dsl": max(len(dsl_hits) - before_counts[0], 0),
                "graph": max(len(graph_facts) - before_counts[1], 0),
                "rag": max(len(semantic_hits) - before_counts[2], 0),
            }
            reflective_meta["followup_runtime"] = followup_runtime

        reflective_meta["elapsed_ms"] = int((time.perf_counter() - reflective_started_at) * 1000)

    dsl_hits, spatial_dsl_meta = _apply_spatial_penalty(
        dsl_hits,
        current_location=current_location,
        settings_rows=windowed_retrieval_settings,
        cards_rows=windowed_retrieval_cards,
    )
    graph_facts, spatial_graph_meta = _apply_spatial_penalty(
        graph_facts,
        current_location=current_location,
        settings_rows=windowed_retrieval_settings,
        cards_rows=windowed_retrieval_cards,
    )
    semantic_hits, spatial_rag_meta = _apply_spatial_penalty(
        semantic_hits,
        current_location=current_location,
        settings_rows=windowed_retrieval_settings,
        cards_rows=windowed_retrieval_cards,
    )
    negative_constraints, negative_constraints_meta = _build_negative_constraints(
        user_input=user_input,
        dsl_hits=dsl_hits,
        graph_facts=graph_facts,
        semantic_hits=semantic_hits,
    )

    compressed_context, context_compression_meta = _build_context_compression(
        user_input=user_input,
        intent=semantic_route.intent,
        dsl_hits=dsl_hits,
        graph_facts=graph_facts,
        semantic_hits=semantic_hits,
    )
    context_cache_layers, context_cache_meta = _build_context_cache_layers(
        mode=mode,
        anchor=anchor,
        prompt_workshop_template=prompt_workshop_template,
        working_settings=[*project_working_settings, *ref_working_settings],
        working_cards=[*scoped_cards, *scoped_reference_cards],
        semantic_settings=[*project_semantic_settings, *ref_semantic_settings],
        current_chapter=current_chapter,
        current_volume=current_volume,
        scene_beat_context=scene_beat_context,
        latest_messages=recent_messages,
        dsl_hits=dsl_hits,
        graph_facts=graph_facts,
        semantic_hits=semantic_hits,
        negative_constraints=negative_constraints,
    )

    quality_gate = _build_quality_gate(user_input, rag_provider, semantic_hits)
    retrieval_runtime_meta = {
        "parallel_enabled": parallel_enabled,
        "graph_timeout_seconds": graph_timeout_seconds,
        "rag_timeout_seconds": rag_timeout_seconds,
        "cache_ttl_seconds": retrieval_cache_ttl,
        "graph_cache": graph_cache_status,
        "rag_cache": rag_cache_status,
        "graph_timeout": graph_timed_out,
        "rag_timeout": rag_timed_out,
        "context_window": {
            "profile": context_window.profile,
            "source": context_window_source,
            "recent_messages_limit": dynamic_budget_plan.recent_messages_limit,
            "retrieval_settings_limit": dynamic_budget_plan.retrieval_settings_limit,
            "retrieval_cards_limit": dynamic_budget_plan.retrieval_cards_limit,
            "model_settings_limit": dynamic_budget_plan.model_settings_limit,
            "model_cards_limit": dynamic_budget_plan.model_cards_limit,
            "chapter_content_chars": context_window.chapter_content_chars,
            "chapter_preview_chars": context_window.chapter_preview_chars,
        },
        "dynamic_budget": {
            "mode": dynamic_budget_plan.mode,
            "source": dynamic_budget_plan.source,
            "confidence": dynamic_budget_plan.confidence,
            "weights": {
                "dsl": round(dynamic_budget_plan.weights.dsl, 4),
                "graph": round(dynamic_budget_plan.weights.graph, 4),
                "rag": round(dynamic_budget_plan.weights.rag, 4),
                "history": round(dynamic_budget_plan.weights.history, 4),
            },
            "dsl_limit": dynamic_budget_plan.dsl_limit,
            "graph_limit": dynamic_budget_plan.graph_limit,
            "rag_limit": dynamic_budget_plan.rag_limit,
        },
        "intent_router": {
            "intent": semantic_route.intent,
            "confidence": semantic_route.confidence,
            "source": semantic_route.source,
            "budget_mode": semantic_route.budget_mode,
            "rag_mode": semantic_route.rag_mode,
            "signals": semantic_route.signals[:8],
        },
        "context_compression": context_compression_meta,
        "negative_constraints": negative_constraints_meta,
        "self_reflective": reflective_meta,
        "context_cache": context_cache_meta,
        "spatial": {
            "dsl": spatial_dsl_meta,
            "graph": spatial_graph_meta,
            "rag": spatial_rag_meta,
        },
        "memory_layers": {
            "working_settings": len(project_working_settings) + len(ref_working_settings),
            "semantic_settings": len(project_semantic_settings) + len(ref_semantic_settings),
            "decay_half_life_days": max(int(settings.memory_decay_half_life_days), 1),
        },
        "context_rows": context_rows_meta,
        "compile_elapsed_ms": 0,
    }
    reference_projects_meta = {
        "requested": normalized_reference_project_ids,
        "resolved": reference_project_meta,
        "settings_count": len(scoped_reference_settings),
        "cards_count": len(scoped_reference_cards),
    }
    runtime_options = {
        "thinking_enabled": bool(thinking_enabled),
        "context_window_profile": context_window.profile,
        "scene_beat_id": scene_beat_id,
        "current_chapter_index": current_chapter_index,
        "budget_mode": dynamic_budget_plan.mode,
        "intent": semantic_route.intent,
        "rag_mode": rag_mode,
        "temperature_profile": normalized_temperature_profile or None,
        "current_location": str(current_location or "").strip() or None,
    }

    model_context = {
        "pov": {
            "mode": mode,
            "anchor": anchor,
            "notes": notes,
        },
        "latest_messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "content": _truncate_text(msg.content, 500),
            }
            for msg in recent_messages
            if msg.content
        ],
        "settings": [
            {
                "id": row.id,
                "project_id": row.project_id,
                "key": row.key,
                "value": row.value,
                "aliases": row.aliases,
            }
            for row in windowed_retrieval_settings[: dynamic_budget_plan.model_settings_limit]
        ],
        "cards": [
            {
                "id": row.id,
                "project_id": row.project_id,
                "title": row.title,
                "content": row.content if isinstance(row.content, dict) else {},
                "aliases": row.aliases,
                "content_preview": _truncate_text(_card_content_text(row), 500),
            }
            for row in windowed_retrieval_cards[: dynamic_budget_plan.model_cards_limit]
        ],
        "memory_layers": {
            "l1_working_memory": {
                "settings_count": len(project_working_settings) + len(ref_working_settings),
                "cards_count": len(scoped_cards) + len(scoped_reference_cards),
            },
            "l2_episodic_memory": {
                "semantic_hits_count": len(semantic_hits),
                "provider": rag_provider,
            },
            "l3_semantic_memory": {
                "settings_count": len(project_semantic_settings) + len(ref_semantic_settings),
                "setting_prefix": str(settings.memory_semantic_key_prefix or "memory.semantic.volume."),
            },
        },
        "current_chapter": current_chapter,
        "story_outline": {
            "volume": current_volume,
            "scene_beat": scene_beat_context,
            "meta": outline_context_meta,
        },
        "runtime_options": runtime_options,
        "reference_projects": reference_projects_meta,
        "context_cache": context_cache_layers,
        "compressed_context": compressed_context,
        "negative_constraints": {
            "items": negative_constraints,
            "meta": negative_constraints_meta,
        },
        "prompt_workshop": {
            "template": prompt_workshop_template,
            "knowledge_injection": {
                "settings": prompt_workshop_knowledge_settings[:24],
                "cards": prompt_workshop_knowledge_cards[:20],
            },
            "meta": prompt_workshop_meta,
        },
        "evidence": {
            "resolver_order": ["DSL", "GRAPH", "RAG"],
            "ranking_dimensions": ["freshness", "confidence", "relevance"],
            "providers": {
                "dsl": "local_dsl",
                "graph": graph_provider,
                "rag": rag_provider,
            },
            "rag_route": {
                "mode": rag_mode,
                "reason": rag_route_reason,
                "source": rag_route_source,
            },
            "rag_short_circuit": {
                "enabled": rag_short_circuit_enabled,
                "reason": rag_short_circuit_reason,
            },
            "retrieval_runtime": retrieval_runtime_meta,
            "context_pack": context_pack_meta,
            "reference_projects": reference_projects_meta,
            "runtime_options": runtime_options,
            "prompt_workshop": prompt_workshop_meta,
            "chapter_context": chapter_context_meta,
            "outline_context": outline_context_meta,
            "quality_gate": quality_gate,
            "dsl_hits": dsl_hits,
            "graph_facts": graph_facts,
            "semantic_hits": semantic_hits,
            "negative_constraints": {
                "items": negative_constraints,
                "meta": negative_constraints_meta,
            },
        },
    }

    evidence_event = {
        "type": "evidence",
        "policy": {
            "mode": mode,
            "anchor": anchor,
            "notes": notes,
            "resolver_order": "DSL > GRAPH > RAG",
            "ranking_dimensions": "freshness + confidence + relevance",
            "providers": {
                "dsl": "local_dsl",
                "graph": graph_provider,
                "rag": rag_provider,
            },
            "rag_route": {
                "mode": rag_mode,
                "reason": rag_route_reason,
                "source": rag_route_source,
            },
            "rag_short_circuit": {
                "enabled": rag_short_circuit_enabled,
                "reason": rag_short_circuit_reason,
            },
            "retrieval_runtime": retrieval_runtime_meta,
            "context_pack": context_pack_meta,
            "reference_projects": reference_projects_meta,
            "runtime_options": runtime_options,
            "prompt_workshop": prompt_workshop_meta,
            "chapter_context": chapter_context_meta,
            "outline_context": outline_context_meta,
            "quality_gate": quality_gate,
            "negative_constraints": negative_constraints_meta,
        },
        "summary": {
            "dsl": len(dsl_hits),
            "graph": len(graph_facts),
            "rag": len(semantic_hits),
            "negative_constraints": len(negative_constraints),
        },
        "sources": {
            "dsl": dsl_hits,
            "graph": graph_facts,
            "rag": semantic_hits,
            "negative_constraints": negative_constraints,
        },
    }

    compile_elapsed_ms = int((time.perf_counter() - compile_started_at) * 1000)
    retrieval_runtime_meta["compile_elapsed_ms"] = compile_elapsed_ms
    _LOGGER.info(
        "context_compiled project_id=%s session_id=%s elapsed_ms=%s profile=%s recent_messages=%s retrieval_settings=%s retrieval_cards=%s",
        project_id,
        session_id,
        compile_elapsed_ms,
        context_window.profile,
        context_rows_meta["recent_messages"],
        context_rows_meta["windowed_retrieval_settings"],
        context_rows_meta["windowed_retrieval_cards"],
    )

    return CompiledContextBundle(model_context=model_context, evidence_event=evidence_event)
