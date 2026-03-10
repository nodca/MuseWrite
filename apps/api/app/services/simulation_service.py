"""场景模拟器服务层。

负责：
- SimulationSession / SimulationTurn 的 CRUD
- 角色上下文构建（信息不对称：每个角色只看到自己知道的事实）
- 单轮行动决策与 LLM 调用
- 角色采访（不影响主流程）
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Session, select

from app.models.content import SettingEntry, StoryCard
from app.models.simulation import SimulationSession, SimulationTurn
from app.services.llm_provider import generate_chat
from app.services.retrieval_adapters import fetch_neo4j_graph_facts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

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
        pending_events=[],
    )
    db.add(sim)
    db.commit()
    db.refresh(sim)
    return sim


def get_simulation_session(db: Session, session_id: int) -> SimulationSession | None:
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
    for t in turns:
        db.delete(t)
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
    """向 pending_events 队列追加一条外部事件。"""
    events: list[str] = list(sim.pending_events or [])
    events.append(event_text.strip())
    sim.pending_events = events
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
    for k, v in content.items():
        if v:
            parts.append(f"{k}: {v}")
    return "\n".join(parts)[:2000]


def _format_settings(entries: list[SettingEntry]) -> str:
    if not entries:
        return ""
    parts: list[str] = []
    for e in entries:
        val = e.value
        if isinstance(val, dict):
            text = "\n".join(f"  {k}: {v}" for k, v in val.items() if v)
        else:
            text = str(val)
        parts.append(f"[{e.key}]\n{text}")
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
    for f in facts:
        src = f.get("source_entity") or ""
        rel = f.get("relation") or ""
        tgt = f.get("target_entity") or ""
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
    facts_text = _build_neo4j_facts_for_card(sim.project_id, actor_card, current_chapter)

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


# ---------------------------------------------------------------------------
# 行动决策
# ---------------------------------------------------------------------------

def decide_next_actor_id(
    sim: SimulationSession,
    turns: list[SimulationTurn],
) -> int:
    """轮询策略：按 character_card_ids 顺序决定下一个行动者。"""
    card_ids = list(sim.character_card_ids or [])
    if not card_ids:
        raise ValueError("simulation has no characters")
    if not turns:
        return card_ids[0]
    last_actor = turns[-1].actor_card_id
    try:
        idx = card_ids.index(last_actor)
    except ValueError:
        idx = -1
    return card_ids[(idx + 1) % len(card_ids)]


def _format_turns_text(turns: list[SimulationTurn], *, limit: int = 20) -> str:
    """将最近 N 条记录格式化为剧情历史文本。"""
    recent = turns[-limit:] if len(turns) > limit else turns
    lines: list[str] = []
    for t in recent:
        if t.is_injected_event:
            lines.append(f"[事件] {t.content}")
        else:
            tag = t.action_type or "say"
            name = t.actor_name or str(t.actor_card_id)
            emotion = f"({t.emotion})" if t.emotion else ""
            lines.append(f"{name}{emotion}[{tag}]: {t.content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM 解析
# ---------------------------------------------------------------------------

_ACTION_TYPES = {"say", "think", "act", "react"}


def _parse_llm_turn_output(raw: str) -> dict[str, Any]:
    """解析 LLM 输出的 JSON，带容错。"""
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    brace = re.search(r"\{[\s\S]*\}", text)
    if brace:
        try:
            data = json.loads(brace.group(0))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
    return {
        "action_type": "say",
        "content": text[:500],
        "emotion": None,
        "target_card_id": None,
    }


# ---------------------------------------------------------------------------
# run_one_turn
# ---------------------------------------------------------------------------

async def run_one_turn(
    db: Session,
    sim: SimulationSession,
    *,
    runtime_config: Any = None,
    current_chapter: int | None = None,
) -> SimulationTurn:
    """执行一轮模拟，返回新增的 SimulationTurn。"""
    turns = get_simulation_turns(db, sim.id)
    turn_index = len(turns) + 1

    # 先处理 pending_events
    pending: list[str] = list(sim.pending_events or [])
    if pending:
        event_text = pending.pop(0)
        sim.pending_events = pending
        sim.updated_at = _utc_now()
        db.add(sim)
        db.flush()
        injected = SimulationTurn(
            session_id=sim.id,
            turn_index=turn_index,
            actor_card_id=0,
            actor_name="[旁白/事件]",
            action_type="act",
            content=event_text,
            is_injected_event=True,
        )
        db.add(injected)
        db.commit()
        db.refresh(injected)
        return injected

    # 决定行动者 & 构建 prompt
    actor_id = decide_next_actor_id(sim, turns)
    all_cards = _load_cards(
        db, sim.project_id, list(sim.character_card_ids or [])
    )
    actor_card = all_cards.get(actor_id)
    actor_name = actor_card.title if actor_card else str(actor_id)
    char_ctx = build_character_context(
        db, sim, actor_id, current_chapter=current_chapter
    )
    history = _format_turns_text(turns)

    sys_prompt = (
        "你正在参与一场角色扮演模拟。你必须完全代入自己的角色，"
        "只能知道角色视角内的信息。\n\n"
        f"{char_ctx}\n\n"
        f"## 当前情境\n{sim.scenario}\n\n"
        "## 输出格式（严格 JSON，不要解释）\n"
        '{"action_type":"say|think|act|react",'
        '"content":"...","emotion":"...","target_card_id":null}'
    )
    usr_prompt = (
        f"## 剧情历史\n{history}\n\n"
        f"现在轮到【{actor_name}】行动。请输出 JSON。"
        if history
        else f"场景刚刚开始。现在轮到【{actor_name}】行动。请输出 JSON。"
    )

    try:
        gen = await generate_chat(
            usr_prompt,
            context={"system": sys_prompt},
            model_override=None,
            thinking_enabled=False,
            temperature_profile="creative",
            temperature_override=None,
            runtime_config=runtime_config,
        )
        raw = gen.assistant_text or ""
    except Exception as exc:
        logger.warning("simulation llm error: %s", exc)
        raw = ""

    parsed = _parse_llm_turn_output(raw)
    act_type = str(parsed.get("action_type") or "say").lower()
    if act_type not in _ACTION_TYPES:
        act_type = "say"
    content = str(parsed.get("content") or "").strip()[:2000]
    emo_raw = parsed.get("emotion")
    emotion = str(emo_raw).strip()[:64] if emo_raw else None
    tgt_raw = parsed.get("target_card_id")
    target_id: int | None = None
    if tgt_raw is not None:
        try:
            target_id = int(tgt_raw)
        except (TypeError, ValueError):
            pass

    new_turn = SimulationTurn(
        session_id=sim.id,
        turn_index=turn_index,
        actor_card_id=actor_id,
        actor_name=actor_name,
        action_type=act_type,
        content=content,
        target_card_id=target_id,
        emotion=emotion,
        is_injected_event=False,
    )
    db.add(new_turn)
    sim.updated_at = _utc_now()
    db.add(sim)
    db.commit()
    db.refresh(new_turn)
    return new_turn


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
    all_cards = _load_cards(
        db, sim.project_id, list(sim.character_card_ids or [])
    )
    actor_card = all_cards.get(card_id)
    if not actor_card:
        return ""

    char_ctx = build_character_context(
        db, sim, card_id, current_chapter=current_chapter
    )
    history = _format_turns_text(turns, limit=10)

    sys_prompt = (
        "你正在扮演一个角色接受采访。只能以该角色视角和已知信息回答，"
        "不得透露角色不知道的内容。用第一人称，简洁真实。\n\n"
        f"{char_ctx}"
    )
    usr_prompt = (
        f"## 场景历史\n{history}\n\n采访问题：{question.strip()}"
        if history
        else f"采访问题：{question.strip()}"
    )

    try:
        gen = await generate_chat(
            usr_prompt,
            context={"system": sys_prompt},
            model_override=None,
            thinking_enabled=False,
            temperature_profile="chat",
            temperature_override=None,
            runtime_config=runtime_config,
        )
        return (gen.assistant_text or "").strip()
    except Exception as exc:
        logger.warning("interview llm error: %s", exc)
        return ""
