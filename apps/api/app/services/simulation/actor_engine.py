from __future__ import annotations

import logging
from typing import Any

from sqlmodel import Session

from app.models.content import StoryCard
from app.models.simulation import SimulationSession, SimulationTurn
from app.services.llm_provider import generate_chat
from app.services.simulation.types import SessionStateRecord, WorldState

logger = logging.getLogger(__name__)


class ActorGenerationError(RuntimeError):
    def __init__(self, code: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(code)
        self.code = str(code or "").strip() or "ACTOR_GENERATION_FAILED"
        self.details = dict(details or {})


def _format_events_for_prompt(events: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in events[:6]:
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        text = str(payload.get("text") or payload.get("motive") or "").strip()
        if not text:
            text = str(item.get("event_type") or "").strip()
        if text:
            lines.append(f"- {text[:200]}")
    return "\n".join(lines)


def _format_recent_turns_for_prompt(turns: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in turns[-12:]:
        name = str(item.get("actor_name") or item.get("actor_id") or "").strip()
        content = str(item.get("content") or "").strip()
        if name and content:
            lines.append(f"{name}: {content[:240]}")
    return "\n".join(lines)


async def generate_turn(
    db: Session,
    *,
    sim: SimulationSession,
    actor_card: StoryCard,
    turn_index: int,
    world_state: WorldState,
    session_state: SessionStateRecord,
    runtime_config: Any = None,
) -> SimulationTurn:
    actor_name = actor_card.title or str(actor_card.id)
    actor_state = session_state.mind_states.get(str(actor_card.id)) or {}
    events_text = _format_events_for_prompt(list(world_state.pending_events or []))
    history_text = _format_recent_turns_for_prompt(list(world_state.recent_turns or []))

    user_input = (
        "你正在参与一场角色扮演模拟。\n"
        "你必须完全代入自己的角色，只能基于角色视角内信息行动。\n\n"
        f"## 你的身份\n{actor_name}\n\n"
        f"## 当前情境\n{(sim.scenario or '').strip()}\n\n"
        "## 当前压力/心智（供你参考，不要逐条复述）\n"
        f"{str(actor_state)[:2000]}\n\n"
    )
    if events_text:
        user_input += f"## 刚发生的事件\n{events_text}\n\n"
    if history_text:
        user_input += f"## 最近对白\n{history_text}\n\n"
    user_input += (
        "现在轮到你发言。要求：\n"
        "- 用第一人称\n"
        "- 只输出你这一句台词，不要解释、不写旁白\n"
        "- 不要提出 setting/card 动作（proposed_actions 必须为空）\n"
    )

    result = await generate_chat(
        user_input,
        context={
            "pov": {
                "mode": "character",
                "anchor": actor_name,
                "notes": [],
            }
        },
        model_override=None,
        thinking_enabled=False,
        temperature_profile="creative",
        temperature_override=None,
        runtime_config=runtime_config,
    )

    if result.usage.get("raw_response_format") == "non_json":
        raise ActorGenerationError(
            "ACTOR_NON_JSON",
            {"provider": result.usage.get("provider")},
        )
    if result.proposed_actions:
        raise ActorGenerationError(
            "ACTOR_ACTIONS_NOT_ALLOWED",
            {"proposed_actions": result.proposed_actions[:3]},
        )
    content = (result.assistant_text or "").strip()
    if not content:
        raise ActorGenerationError("ACTOR_EMPTY")

    return SimulationTurn(
        session_id=sim.id,
        turn_index=turn_index,
        actor_card_id=int(actor_card.id or 0),
        actor_name=actor_name,
        action_type="say",
        content=content[:2000],
        target_card_id=None,
        emotion=None,
        is_injected_event=False,
    )
