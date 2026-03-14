"""场景模拟器服务层。

负责：
- SimulationSession / SimulationTurn 的 CRUD
- 角色上下文构建（信息不对称：每个角色只看到自己知道的事实）
- 外部事件注入（写入 SimulationEvent）
- 角色采访（不影响主流程）

说明：
- 运行主链（发言调度、ToM、bidding、referee、actor generation）已迁移到 `app.services.simulation`。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Session, select

from app.models.content import SettingEntry, StoryCard
from app.models.simulation import (
    SimulationDecisionTrace,
    SimulationEvent,
    SimulationSession,
    SimulationSessionState,
    SimulationTurn,
)
from app.services.llm_provider import generate_chat
from app.services.retrieval_adapters import fetch_neo4j_graph_facts
from app.services.simulation import append_event

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

def create_simulation_session(
    db: Session,
    *,
    project_id: int,
    title: str = "未命名模拟",
    scenario: str = "",
    character_card_ids: list[int],
    setting_keys: list[str] | None = None,
    max_turns: int = 10,
) -> SimulationSession:
    sim = SimulationSession(
        project_id=project_id,
        title=title,
        scenario=scenario,
        character_card_ids=character_card_ids,
        setting_keys=setting_keys or [],
        max_turns=max(1, min(50, max_turns)),
        status="idle",
    )
    db.add(sim)
    db.commit()
    db.refresh(sim)
    return sim


def get_simulation_session(
    db: Session,
    session_id: int,
) -> SimulationSession | None:
    return db.get(SimulationSession, session_id)


def list_simulation_sessions(
    db: Session,
    project_id: int,
    *,
    limit: int = 20,
    offset: int = 0,
) -> list[SimulationSession]:
    stmt = (
        select(SimulationSession)
        .where(SimulationSession.project_id == project_id)
        .order_by(SimulationSession.id.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(db.exec(stmt).all())


def delete_simulation_session(db: Session, session_id: int) -> bool:
    sim = db.get(SimulationSession, session_id)
    if not sim:
        return False

    turns = db.exec(
        select(SimulationTurn).where(SimulationTurn.session_id == session_id)
    ).all()
    traces = db.exec(
        select(SimulationDecisionTrace).where(
            SimulationDecisionTrace.session_id == session_id
        )
    ).all()
    events = db.exec(
        select(SimulationEvent).where(SimulationEvent.session_id == session_id)
    ).all()
    state = db.exec(
        select(SimulationSessionState).where(
            SimulationSessionState.session_id == session_id
        )
    ).first()

    for item in turns:
        db.delete(item)
    for item in traces:
        db.delete(item)
    for item in events:
        db.delete(item)
    if state is not None:
        db.delete(state)

    db.delete(sim)
    db.commit()
    return True


def get_simulation_turns(
    db: Session,
    session_id: int,
) -> list[SimulationTurn]:
    stmt = (
        select(SimulationTurn)
        .where(SimulationTurn.session_id == session_id)
        .order_by(SimulationTurn.turn_index)
    )
    return list(db.exec(stmt).all())


def inject_event(
    db: Session,
    sim: SimulationSession,
    event_text: str,
) -> SimulationSession:
    text = str(event_text or "").strip()
    if not text:
        return sim
    append_event(
        db,
        sim.id,
        event_type="external_text",
        payload={"text": text},
        priority=10,
    )
    sim.updated_at = _utc_now()
    db.add(sim)
    db.commit()
    db.refresh(sim)
    return sim


# ---------------------------------------------------------------------------
# 上下文构建（信息不对称）
# ---------------------------------------------------------------------------

def _load_cards(
    db: Session,
    project_id: int,
    card_ids: list[int],
) -> dict[int, StoryCard]:
    if not card_ids:
        return {}
    stmt = select(StoryCard).where(
        StoryCard.project_id == project_id,
        StoryCard.id.in_(card_ids),
    )
    return {card.id: card for card in db.exec(stmt).all()}


def _load_settings(
    db: Session,
    project_id: int,
    keys: list[str],
) -> list[SettingEntry]:
    if not keys:
        return []
    stmt = select(SettingEntry).where(
        SettingEntry.project_id == project_id,
        SettingEntry.key.in_(keys),
    )
    return list(db.exec(stmt).all())


def _format_card_content(card: StoryCard) -> str:
    content = card.content or {}
    if isinstance(content, str):
        return content[:2000]
    parts: list[str] = [f"【{card.title}】"]
    for key, value in content.items():
        if value:
            parts.append(f"{key}: {value}")
    return "\n".join(parts)[:2000]


def _format_settings(entries: list[SettingEntry]) -> str:
    if not entries:
        return ""
    parts: list[str] = []
    for entry in entries:
        val = entry.value
        if isinstance(val, dict):
            text = "\n".join(f"  {k}: {v}" for k, v in val.items() if v)
        else:
            text = str(val)
        parts.append(f"[{entry.key}]\n{text}")
    return "\n\n".join(parts)[:3000]


def _build_neo4j_facts_for_card(
    project_id: int,
    card: StoryCard,
    current_chapter: int | None = None,
) -> str:
    """为特定角色查询 Neo4j 事实（以角色名为锚点，实现信息不对称）。"""
    name = card.title or ""
    aliases: list[str] = list(card.aliases or [])
    terms = [t for t in [name] + aliases if t]
    if not terms:
        return ""
    facts = fetch_neo4j_graph_facts(
        project_id,
        terms,
        anchor=name,
        limit=12,
        current_chapter=current_chapter,
        raise_on_error=False,
    )
    if not facts:
        return ""
    lines: list[str] = []
    for fact in facts:
        src = fact.get("source_entity") or ""
        rel = fact.get("relation") or ""
        tgt = fact.get("target_entity") or ""
        if src and rel and tgt:
            lines.append(f"- {src} {rel} {tgt}")
    return "\n".join(lines)[:2000]


def build_character_context(
    db: Session,
    sim: SimulationSession,
    card_id: int,
    *,
    current_chapter: int | None = None,
) -> str:
    """构建单个角色的完整上下文。"""
    all_cards = _load_cards(db, sim.project_id, list(sim.character_card_ids or []))
    actor_card = all_cards.get(card_id)
    if not actor_card:
        return ""

    settings_entries = _load_settings(db, sim.project_id, list(sim.setting_keys or []))
    card_text = _format_card_content(actor_card)
    settings_text = _format_settings(settings_entries)
    facts_text = _build_neo4j_facts_for_card(
        sim.project_id, actor_card, current_chapter=current_chapter
    )

    parts: list[str] = [f"## 你的身份\n{card_text}"]
    if settings_text:
        parts.append(f"## 世界设定\n{settings_text}")
    if facts_text:
        parts.append(f"## 你所知道的事实\n{facts_text}")
    other_names = [
        c.title for cid, c in all_cards.items() if cid != card_id and c.title
    ]
    if other_names:
        parts.append("## 场景中的其他人\n" + "、".join(other_names))
    return "\n\n".join(parts)


def _format_turns_text(turns: list[SimulationTurn], *, limit: int = 20) -> str:
    recent = turns[-limit:] if len(turns) > limit else turns
    lines: list[str] = []
    for turn in recent:
        if turn.is_injected_event:
            lines.append(f"[事件] {turn.content}")
        else:
            tag = turn.action_type or "say"
            name = turn.actor_name or str(turn.actor_card_id)
            emotion = f"({turn.emotion})" if turn.emotion else ""
            lines.append(f"{name}{emotion}[{tag}]: {turn.content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 角色采访
# ---------------------------------------------------------------------------

async def interview_character(
    db: Session,
    *,
    sim: SimulationSession,
    card_id: int,
    question: str,
    turns: list[SimulationTurn],
    runtime_config: Any = None,
    current_chapter: int | None = None,
) -> str:
    """以角色视角回答问题（不影响主模拟流程）。"""
    all_cards = _load_cards(db, sim.project_id, list(sim.character_card_ids or []))
    actor_card = all_cards.get(card_id)
    if not actor_card:
        return ""

    char_ctx = build_character_context(
        db, sim, card_id, current_chapter=current_chapter
    )
    history = _format_turns_text(turns, limit=10)

    user_input = (
        "你正在扮演一个角色接受采访。只能以该角色视角和已知信息回答，"
        "不得透露角色不知道的内容。用第一人称，简洁真实。\n\n"
        f"{char_ctx}\n\n"
        + (f"## 场景历史\n{history}\n\n" if history else "")
        + f"采访问题：{question.strip()}"
    )

    try:
        gen = await generate_chat(
            user_input,
            context={
                "pov": {
                    "mode": "character",
                    "anchor": actor_card.title or str(card_id),
                    "notes": [],
                }
            },
            model_override=None,
            thinking_enabled=False,
            temperature_profile="chat",
            temperature_override=None,
            runtime_config=runtime_config,
        )
        if gen.proposed_actions:
            return ""
        return (gen.assistant_text or "").strip()
    except Exception as exc:
        logger.warning("interview llm error: %s", exc)
        return ""

