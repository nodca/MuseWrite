from typing import Any

from sqlmodel import Session, select

from app.models.chat import ChatAction, ChatSession, ProjectMutationVersion
from app.services.chat_service._common import (
    _utc_now,
    _normalize_graph_entity_token,
    _normalize_aliases_payload,
)


def _project_id_for_action(db: Session, action: ChatAction) -> int:
    session = db.get(ChatSession, action.session_id)
    if not session:
        raise ValueError("session not found for action")
    return session.project_id


def _current_project_mutation_version(db: Session, project_id: int) -> int:
    stmt = select(ProjectMutationVersion).where(ProjectMutationVersion.project_id == project_id)
    row = db.exec(stmt).first()
    if not row:
        return 0
    return int(row.version)


def _bump_project_mutation_version(db: Session, project_id: int) -> int:
    stmt = select(ProjectMutationVersion).where(ProjectMutationVersion.project_id == project_id)
    row = db.exec(stmt).first()
    if row:
        row.version = int(row.version) + 1
        row.updated_at = _utc_now()
        db.add(row)
        return int(row.version)

    row = ProjectMutationVersion(project_id=project_id, version=1, updated_at=_utc_now())
    db.add(row)
    db.flush()
    return int(row.version)


def _graph_sync_meta(action: ChatAction) -> dict:
    if not isinstance(action.apply_result, dict):
        return {}
    meta = action.apply_result.get("graph_sync")
    return meta if isinstance(meta, dict) else {}


def _action_graph_identifiers(action: ChatAction) -> tuple[str, int, str]:
    meta = _graph_sync_meta(action)
    mutation_id = str(meta.get("mutation_id") or "")
    expected_version_raw = meta.get("expected_version")
    expected_version = int(expected_version_raw) if isinstance(expected_version_raw, int) else 0
    idempotency_key = str(meta.get("job_idempotency_key") or "")
    return mutation_id, expected_version, idempotency_key


def _is_graph_job_stale(
    action: ChatAction,
    *,
    mutation_id: str,
    expected_version: int,
) -> tuple[bool, str]:
    meta = _graph_sync_meta(action)
    current_mutation_id = str(meta.get("mutation_id") or "")
    current_expected_raw = meta.get("expected_version")
    current_expected_version = int(current_expected_raw) if isinstance(current_expected_raw, int) else 0

    if mutation_id and current_mutation_id and current_mutation_id != mutation_id:
        return True, "mutation_id_mismatch"
    if (
        expected_version > 0
        and current_expected_version > 0
        and current_expected_version != expected_version
    ):
        return True, "expected_version_mismatch"
    if action.status != "applied":
        return True, f"action_status_{action.status}"
    return False, ""


def _index_lifecycle_key(slot: str) -> str:
    return "index_lifecycle_compensation" if slot == "compensation" else "index_lifecycle"


def _index_lifecycle_meta(action: ChatAction, *, slot: str = "default") -> dict:
    if not isinstance(action.apply_result, dict):
        return {}
    meta = action.apply_result.get(_index_lifecycle_key(slot))
    return meta if isinstance(meta, dict) else {}

