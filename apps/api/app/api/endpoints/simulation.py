"""场景模拟器 API 端点。"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.core.auth import AuthPrincipal, get_current_principal
from app.core.database import get_session
from app.core.sse import sse_event
from app.models.simulation import SimulationSession, SimulationTurn
from app.services.simulation import run_turn
from app.services.simulation_service import (
    create_simulation_session,
    delete_simulation_session,
    get_simulation_session,
    get_simulation_turns,
    inject_event,
    interview_character,
    list_simulation_sessions,
)
from .chat_helpers import ensure_project_scope_access as _ensure_project

router = APIRouter(prefix="/simulation", tags=["simulation"])


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class SimulationSessionCreate(BaseModel):
    project_id: int
    title: str = "未命名模拟"
    scenario: str = ""
    character_card_ids: list[int] = Field(default_factory=list)
    setting_keys: list[str] = Field(default_factory=list)
    max_turns: int = Field(default=10, ge=1, le=50)


class SimulationSessionRead(BaseModel):
    id: int
    project_id: int
    title: str
    scenario: str
    character_card_ids: list[int]
    setting_keys: list[str]
    max_turns: int
    status: str

    model_config = {"from_attributes": True}


class SimulationTurnRead(BaseModel):
    id: int
    session_id: int
    turn_index: int
    actor_card_id: int
    actor_name: str
    action_type: str
    content: str
    target_card_id: int | None
    emotion: str | None
    is_injected_event: bool

    model_config = {"from_attributes": True}


class SimulationSessionDetail(BaseModel):
    session: SimulationSessionRead
    turns: list[SimulationTurnRead]


class InjectEventRequest(BaseModel):
    event_text: str


class InterviewRequest(BaseModel):
    card_id: int
    question: str
    current_chapter: int | None = None


class RunRequest(BaseModel):
    current_chapter: int | None = None


# ---------------------------------------------------------------------------
# CRUD 端点
# ---------------------------------------------------------------------------

@router.post("/sessions", response_model=SimulationSessionRead, status_code=201)
def create_session(
    payload: SimulationSessionCreate,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project(payload.project_id, principal)
    sim = create_simulation_session(
        db,
        project_id=payload.project_id,
        title=payload.title,
        scenario=payload.scenario,
        character_card_ids=payload.character_card_ids,
        setting_keys=payload.setting_keys,
        max_turns=payload.max_turns,
    )
    return sim


@router.get("/sessions", response_model=list[SimulationSessionRead])
def list_sessions(
    project_id: int = Query(...),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    _ensure_project(project_id, principal)
    return list_simulation_sessions(db, project_id, limit=limit, offset=offset)


@router.get("/sessions/{session_id}", response_model=SimulationSessionDetail)
def get_session_detail(
    session_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    sim = _get_sim_or_404(db, session_id, principal)
    turns = get_simulation_turns(db, session_id)
    return SimulationSessionDetail(
        session=SimulationSessionRead.model_validate(sim),
        turns=[SimulationTurnRead.model_validate(t) for t in turns],
    )


@router.delete("/sessions/{session_id}", status_code=204)
def remove_session(
    session_id: int,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    sim = _get_sim_or_404(db, session_id, principal)
    delete_simulation_session(db, sim.id)


# ---------------------------------------------------------------------------
# 辅助：鉴权 + 404
# ---------------------------------------------------------------------------

def _get_sim_or_404(
    db: Session,
    session_id: int,
    principal: AuthPrincipal,
) -> SimulationSession:
    sim = get_simulation_session(db, session_id)
    if not sim:
        raise HTTPException(status_code=404, detail="simulation session not found")
    _ensure_project(sim.project_id, principal)
    return sim


# ---------------------------------------------------------------------------
# 注入事件
# ---------------------------------------------------------------------------

@router.post("/sessions/{session_id}/inject", response_model=SimulationSessionRead)
def inject_event_endpoint(
    session_id: int,
    payload: InjectEventRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    sim = _get_sim_or_404(db, session_id, principal)
    if not payload.event_text.strip():
        raise HTTPException(status_code=400, detail="event_text is required")
    sim = inject_event(db, sim, payload.event_text)
    return sim


# ---------------------------------------------------------------------------
# 角色采访
# ---------------------------------------------------------------------------

@router.post("/sessions/{session_id}/interview")
async def interview_endpoint(
    session_id: int,
    payload: InterviewRequest,
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    sim = _get_sim_or_404(db, session_id, principal)
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    turns = get_simulation_turns(db, session_id)
    answer = await interview_character(
        db,
        sim=sim,
        card_id=payload.card_id,
        question=payload.question,
        turns=turns,
        current_chapter=payload.current_chapter,
    )
    return {"card_id": payload.card_id, "answer": answer}


# ---------------------------------------------------------------------------
# SSE 流式运行
# ---------------------------------------------------------------------------

@router.post("/sessions/{session_id}/run")
async def run_simulation(
    session_id: int,
    payload: RunRequest = RunRequest(),
    db: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_principal),
):
    sim = _get_sim_or_404(db, session_id, principal)

    async def _stream():
        nonlocal sim
        # 重新加载避免状态陈旧
        sim = get_simulation_session(db, session_id)
        if sim.status == "running":
            yield sse_event({"type": "error", "detail": "simulation already running"})
            return

        sim.status = "running"
        db.add(sim)
        db.commit()
        db.refresh(sim)

        turns = get_simulation_turns(db, session_id)
        completed = len([t for t in turns if not t.is_injected_event])

        try:
            while completed < sim.max_turns:
                sim = get_simulation_session(db, session_id)
                if sim.status != "running":
                    break
                outcome, turn = await run_turn(
                    db,
                    sim=sim,
                    current_chapter=payload.current_chapter,
                )
                if outcome.status == "spoken" and turn is not None:
                    completed += 1
                    yield sse_event({
                        "type": "turn",
                        "turn": {
                            "id": turn.id,
                            "turn_index": turn.turn_index,
                            "actor_card_id": turn.actor_card_id,
                            "actor_name": turn.actor_name,
                            "action_type": turn.action_type,
                            "content": turn.content,
                            "emotion": turn.emotion,
                            "is_injected_event": turn.is_injected_event,
                        },
                    })
                    continue
                if outcome.status == "paused":
                    yield sse_event({
                        "type": "paused",
                        "turn_index": outcome.turn_index,
                    })
                    break
                yield sse_event({
                    "type": "error",
                    "detail": outcome.error_code or "simulation turn failed",
                    "turn_index": outcome.turn_index,
                    "error_code": outcome.error_code,
                    "details": outcome.details,
                })
                break
            yield sse_event({"type": "done", "completed": completed})
        except Exception as exc:
            yield sse_event({"type": "error", "detail": str(exc)})
        finally:
            sim = get_simulation_session(db, session_id)
            if sim and sim.status == "running":
                sim.status = "idle"
                db.add(sim)
                db.commit()

    return StreamingResponse(_stream(), media_type="text/event-stream")
