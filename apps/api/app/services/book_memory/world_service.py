from sqlmodel import Session, select

from app.models.book_memory import WorldRule


def list_world_rules(db: Session, project_id: int, *, status: str | None = "active") -> list[WorldRule]:
    stmt = select(WorldRule).where(WorldRule.project_id == project_id)
    if status is not None:
        stmt = stmt.where(WorldRule.status == status)
    stmt = stmt.order_by(WorldRule.priority.asc(), WorldRule.id.asc())
    return db.exec(stmt).all()


def create_world_rule(
    db: Session,
    *,
    project_id: int,
    title: str,
    statement: str,
    scope: str = "global",
    priority: int = 100,
    tags: list[str] | None = None,
    status: str = "active",
    source_refs: list[dict] | None = None,
) -> WorldRule:
    rule = WorldRule(
        project_id=project_id,
        scope=str(scope or "global").strip() or "global",
        title=str(title or "").strip(),
        statement=str(statement or "").strip(),
        priority=int(priority),
        tags=list(tags or []),
        status=str(status or "active").strip() or "active",
        source_refs=list(source_refs or []),
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return rule


def update_world_rule(
    db: Session,
    *,
    project_id: int,
    rule_id: int,
    title: str | None = None,
    statement: str | None = None,
    scope: str | None = None,
    priority: int | None = None,
    tags: list[str] | None = None,
    status: str | None = None,
    source_refs: list[dict] | None = None,
) -> WorldRule:
    rule = db.get(WorldRule, rule_id)
    if rule is None or int(rule.project_id) != int(project_id):
        raise ValueError("world rule not found")
    if title is not None:
        rule.title = str(title).strip()
    if statement is not None:
        rule.statement = str(statement).strip()
    if scope is not None:
        rule.scope = str(scope).strip() or "global"
    if priority is not None:
        rule.priority = int(priority)
    if tags is not None:
        rule.tags = list(tags)
    if status is not None:
        rule.status = str(status).strip() or "active"
    if source_refs is not None:
        rule.source_refs = list(source_refs)
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return rule
