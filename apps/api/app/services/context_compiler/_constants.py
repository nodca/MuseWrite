from app.services.context_compiler._types import ContextWindowPolicy, BudgetWeights


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

_COMPRESSION_SOURCE_TIER_ORDER = ("DSL", "GRAPH", "RAG", "OTHER")
_COMPRESSION_SECTION_BY_SOURCE = {
    "DSL": "fact_entities",
    "GRAPH": "fact_relations",
    "RAG": "retrieved_events",
}

