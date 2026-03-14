from typing import Any

from app.core.config import settings
from app.services.context_compiler._types import (
    SemanticRouteDecision,
    BudgetWeights,
    ContextWindowPolicy,
    DynamicBudgetPlan,
)
from app.services.context_compiler._constants import (
    _SEMANTIC_ROUTE_INTENTS,
    _SEMANTIC_INTENT_BUDGET_MODE,
    _SEMANTIC_INTENT_RAG_MODE,
    _SEMANTIC_INTENT_TOKENS,
    _BUDGET_MODE_PRESETS,
)
from app.services.context_compiler._utils import _truncate_text
from app.services.context_compiler.normalization import (
    _normalize_budget_mode,
    _normalize_semantic_intent,
    _normalize_weight_dict,
)


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

    heuristic_intent, heuristic_confidence, heuristic_signals = _semantic_intent_from_heuristics(
        user_input,
        chapter_preview=chapter_preview,
        scene_beat_text=scene_beat_text,
    )
    return SemanticRouteDecision(
        intent=heuristic_intent,
        confidence=round(heuristic_confidence, 4),
        source="heuristic_router",
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
        heuristic_mode, heuristic_conf = _budget_mode_from_heuristics(
            user_input,
            chapter_preview=chapter_preview,
            scene_beat_text=scene_beat_text,
        )
        chosen_mode = heuristic_mode
        chosen_weights = _BUDGET_MODE_PRESETS.get(heuristic_mode, default_weights)
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
