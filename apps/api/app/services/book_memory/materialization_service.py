from sqlmodel import Session, select

from app.models.book_memory import MemoryMaterialization


def get_memory_materialization(
    db: Session,
    *,
    project_id: int,
    materialization_type: str,
    scope_key: str = "global",
) -> MemoryMaterialization | None:
    stmt = (
        select(MemoryMaterialization)
        .where(
            MemoryMaterialization.project_id == project_id,
            MemoryMaterialization.materialization_type == materialization_type,
            MemoryMaterialization.scope_key == scope_key,
        )
        .order_by(MemoryMaterialization.updated_at.desc(), MemoryMaterialization.id.desc())
    )
    return db.exec(stmt).first()


def upsert_memory_materialization(
    db: Session,
    *,
    project_id: int,
    materialization_type: str,
    scope_key: str = "global",
    payload: dict | None = None,
    source_versions: dict | None = None,
) -> MemoryMaterialization:
    materialization = get_memory_materialization(
        db,
        project_id=project_id,
        materialization_type=materialization_type,
        scope_key=scope_key,
    )
    if materialization is None:
        materialization = MemoryMaterialization(
            project_id=project_id,
            materialization_type=str(materialization_type or "").strip(),
            scope_key=str(scope_key or "global").strip() or "global",
        )
    materialization.payload = dict(payload or {})
    materialization.source_versions = dict(source_versions or {})
    db.add(materialization)
    db.commit()
    db.refresh(materialization)
    return materialization
