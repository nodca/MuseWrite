from __future__ import annotations

import asyncio
from typing import Any

from app.services.llm_provider import generate_chat
from app.services.simulation.mind_engine import _load_mind_state, _upsert_second_order
from app.services.simulation.types import (
    SecondOrderBelief,
    SessionStateRecord,
    WorldState,
)


class ToMUpdateError(RuntimeError):
    def __init__(self, code: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(code)
        self.code = str(code or "").strip() or "TOM_UPDATE_FAILED"
        self.details = dict(details or {})


def _recent_turns_have_content(turns: list[dict[str, Any]]) -> bool:
    for item in turns:
        content = str(item.get("content") or "").strip()
        if content:
            return True
    return False


async def _infer_second_order_deltas_for_holder(
    *,
    holder_id: int,
    allowed_subject_ids: set[int],
    world_state: WorldState,
    session_state: SessionStateRecord,
    runtime_config: Any = None,
) -> list[SecondOrderBelief]:
    mind = _load_mind_state(session_state.mind_states, holder_id)
    holder_facts = list(mind.known_fact_keys or [])[:40]
    holder_beliefs = [
        {
            "fact_key": item.fact_key,
            "stance": item.stance,
            "confidence": item.confidence,
        }
        for item in mind.beliefs[:20]
    ]
    existing = [
        {
            "subject_id": item.subject_id,
            "fact_key": item.fact_key,
            "believed_knowledge_state": item.believed_knowledge_state,
            "confidence": item.confidence,
            "source_turn_id": item.source_turn_id,
        }
        for item in mind.beliefs_about_others[:30]
    ]

    recent_turns = list(world_state.recent_turns or [])[-12:]
    user_input = (
        "你是一个 ToM(Theory of Mind) 更新器。你的任务是：基于“最近对白(公共)”与“该角色当前心智(私有)”，"
        "为该角色生成二阶 belief 增量（只到二阶，严禁三阶）。\n\n"
        "严禁基于规则/关键词匹配来判断知道与否；必须做自然语言理解后再给出结构化结果。\n\n"
        f"## 角色（holder）\nholder_id={holder_id}\n\n"
        "## holder 已知事实键（节选）\n"
        f"{holder_facts}\n\n"
        "## holder 一阶 belief（节选）\n"
        f"{holder_beliefs}\n\n"
        "## holder 当前已有二阶 belief（节选）\n"
        f"{existing}\n\n"
        "## 最近对白（公共）\n"
        f"{recent_turns}\n\n"
        "请输出 proposed_actions：必须且只能输出 1 个 setting.upsert，作为内部结构化 ToM 增量：\n"
        "  - payload.key 固定为 __internal.simulation.tom_delta__\n"
        "  - payload.value 必须是 JSON 对象，结构如下：\n"
        "    {\n"
        "      \"beliefs_about_others\": [\n"
        "        {\n"
        "          \"holder_id\": int,\n"
        "          \"subject_id\": int,\n"
        "          \"fact_key\": \"短而稳定的事实键(建议snake_case, <=80字)\",\n"
        "          \"believed_knowledge_state\": \"knows|denies|uncertain|suspects|unknown\",\n"
        "          \"confidence\": float,\n"
        "          \"source_turn_id\": int|null\n"
        "        }\n"
        "      ]\n"
        "    }\n"
        "约束：\n"
        f"- holder_id 必须为 {holder_id}\n"
        f"- subject_id 必须属于 {sorted(allowed_subject_ids)} 且 subject_id != holder_id\n"
        "- source_turn_id 优先填写最近对白中的 turn_index；不确定则填 null\n"
        "- 每轮最多输出 8 条；没有有效增量就输出空数组\n"
        "assistant_text 可以为空，不要写剧情正文。\n"
    )

    result = await generate_chat(
        user_input,
        context={
            "pov": {
                "mode": "character",
                "anchor": f"character:{holder_id}",
                "notes": [],
            }
        },
        model_override=None,
        thinking_enabled=False,
        temperature_profile="action",
        temperature_override=None,
        runtime_config=runtime_config,
    )

    actions = list(result.proposed_actions or [])
    if len(actions) != 1:
        raise ToMUpdateError(
            "TOM_ACTIONS_NOT_ALLOWED",
            {"holder_id": holder_id, "action_count": len(actions)},
        )
    action = actions[0]
    if action.get("action_type") != "setting.upsert":
        raise ToMUpdateError(
            "TOM_MISSING_DELTA",
            {"holder_id": holder_id, "action_type": action.get("action_type")},
        )
    payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
    if payload.get("key") != "__internal.simulation.tom_delta__":
        raise ToMUpdateError(
            "TOM_MISSING_DELTA",
            {"holder_id": holder_id, "key": payload.get("key")},
        )
    tom_value = payload.get("value")
    if not isinstance(tom_value, dict):
        raise ToMUpdateError(
            "TOM_DELTA_INVALID",
            {"holder_id": holder_id, "tom_value": tom_value},
        )

    raw_items = tom_value.get("beliefs_about_others")
    if not isinstance(raw_items, list):
        raise ToMUpdateError(
            "TOM_DELTA_INVALID",
            {"holder_id": holder_id, "tom_value": tom_value},
        )
    if len(raw_items) > 8:
        raise ToMUpdateError(
            "TOM_DELTA_TOO_MANY",
            {"holder_id": holder_id, "count": len(raw_items)},
        )

    deltas: list[SecondOrderBelief] = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise ToMUpdateError(
                "TOM_DELTA_INVALID",
                {"holder_id": holder_id, "item_type": type(item).__name__},
            )
        belief = SecondOrderBelief.model_validate(item)
        if belief.believed_knowledge_state not in {
            "knows",
            "denies",
            "uncertain",
            "suspects",
            "unknown",
        }:
            raise ToMUpdateError(
                "TOM_INVALID_KNOWLEDGE_STATE",
                {
                    "holder_id": holder_id,
                    "knowledge_state": belief.believed_knowledge_state,
                },
            )
        if int(belief.holder_id) != holder_id:
            raise ToMUpdateError(
                "TOM_HOLDER_MISMATCH",
                {"holder_id": holder_id, "belief_holder_id": belief.holder_id},
            )
        if int(belief.subject_id) == holder_id:
            raise ToMUpdateError(
                "TOM_SUBJECT_SELF",
                {"holder_id": holder_id},
            )
        if int(belief.subject_id) not in allowed_subject_ids:
            raise ToMUpdateError(
                "TOM_SUBJECT_OUT_OF_SCOPE",
                {"holder_id": holder_id, "subject_id": belief.subject_id},
            )
        deltas.append(belief)
    return deltas


async def update_second_order_beliefs_from_turns(
    world_state: WorldState,
    session_state: SessionStateRecord,
    *,
    character_ids: list[int],
    runtime_config: Any = None,
) -> SessionStateRecord:
    """从 recent_turns 自动更新二阶 ToM。

    约束：
    - 只更新 beliefs_about_others
    - 禁止 rule-based 推断；ToM 增量由 LLM 结构化输出
    - 若 recent_turns 无内容，本轮不触发 ToM 更新
    """

    if not _recent_turns_have_content(list(world_state.recent_turns or [])):
        return session_state

    holder_ids: list[int] = []
    seen: set[int] = set()
    for cid in character_ids:
        holder_id = int(cid)
        if holder_id <= 0 or holder_id in seen:
            continue
        seen.add(holder_id)
        holder_ids.append(holder_id)
    allowed_subject_ids = set(holder_ids)

    tasks = [
        _infer_second_order_deltas_for_holder(
            holder_id=int(holder_id),
            allowed_subject_ids=allowed_subject_ids,
            world_state=world_state,
            session_state=session_state,
            runtime_config=runtime_config,
        )
        for holder_id in holder_ids
    ]
    results = await asyncio.gather(*tasks)

    next_states = dict(session_state.mind_states)
    for holder_id, deltas in zip(holder_ids, results, strict=True):
        mind = _load_mind_state(next_states, int(holder_id))
        second_order = list(mind.beliefs_about_others)
        for delta in deltas:
            second_order = _upsert_second_order(second_order, delta)
        mind = mind.model_copy(update={"beliefs_about_others": second_order})
        next_states[str(holder_id)] = mind.model_dump(mode="json")

    return session_state.model_copy(update={"mind_states": next_states})
