from dataclasses import dataclass
from datetime import datetime
from typing import Any


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

