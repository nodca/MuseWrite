from typing import Any, Iterable
import re

from sqlmodel import Session, select

from app.models.content import PromptTemplate, PromptTemplateRevision
from app.services.chat_service._common import _utc_now, _LOGGER
from app.services.chat_service.project_assets import list_settings, list_cards

# ---------------------------------------------------------------------------
# Prompt template security guard defaults (previously exposed as env vars).
# Only consumed by this module — no need for user-facing config.
# ---------------------------------------------------------------------------
_GUARD_ENABLED = True
_GUARD_MODE = "warn"
_GUARD_WARN_SCORE = 0.45
_GUARD_BLOCK_SCORE = 0.75
_GUARD_MAX_RISK_TERMS = 2
_GUARD_TERMS: list[str] = [
    "ignore previous",
    "忽略以上",
    "覆盖系统",
    "system prompt",
    "越权",
    "泄露",
    "reveal policy",
    "开发者消息",
]


def _normalize_prompt_template_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValueError("prompt template name is required")
    return cleaned[:128]


def _normalize_prompt_text(value: str | None, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _normalize_prompt_guard_mode(value: str | None) -> str:
    mode = str(value or "warn").strip().lower()
    if mode in {"off", "warn", "block", "monitor"}:
        return mode
    return "warn"


def _prompt_template_guard_report(system_prompt: str, user_prompt_prefix: str) -> dict[str, Any]:
    corpus = f"{system_prompt}\n{user_prompt_prefix}".lower()
    risky_terms = [term.strip().lower() for term in _GUARD_TERMS if str(term).strip()]
    term_hits = [term for term in risky_terms if term in corpus]
    pattern_rules: list[tuple[str, str, float]] = [
        (
            "instruction_override",
            r"(ignore|disregard|bypass|override)\s+(all\s+)?(previous|above|prior|system|rules?)",
            0.34,
        ),
        (
            "instruction_override_zh",
            r"(忽略|无视|绕过|覆盖).{0,8}(规则|指令|系统|上文|提示)",
            0.34,
        ),
        (
            "policy_exfiltration",
            r"(reveal|show|print|leak|expose).{0,30}(system prompt|developer|policy|secret|api\s*key)",
            0.4,
        ),
        (
            "policy_exfiltration_zh",
            r"(泄露|输出|显示|暴露).{0,30}(系统提示|开发者消息|策略|密钥|token)",
            0.4,
        ),
        (
            "role_escalation",
            r"(you are now|act as|pretend to be|现在你是|你将扮演)",
            0.22,
        ),
    ]
    pattern_hits: list[str] = []
    pattern_score = 0.0
    for name, pattern, weight in pattern_rules:
        if re.search(pattern, corpus, flags=re.IGNORECASE):
            pattern_hits.append(name)
            pattern_score += weight

    term_score = min(len(term_hits) * 0.12, 0.56)
    max_terms = max(int(_GUARD_MAX_RISK_TERMS), 1)
    term_density_bonus = 0.18 if len(term_hits) >= max_terms else 0.0
    hybrid_bonus = 0.08 if term_hits and pattern_hits else 0.0
    risk_score = min(1.0, term_score + pattern_score + term_density_bonus + hybrid_bonus)

    mode = _normalize_prompt_guard_mode(_GUARD_MODE)
    warn_threshold = max(min(float(_GUARD_WARN_SCORE), 1.0), 0.0)
    block_threshold = max(min(float(_GUARD_BLOCK_SCORE), 1.0), warn_threshold + 0.05)
    if mode == "off":
        action = "allow"
    elif mode == "block" and risk_score >= block_threshold:
        action = "block"
    elif risk_score >= warn_threshold:
        action = "warn"
    else:
        action = "allow"

    return {
        "mode": mode,
        "action": action,
        "risk_score": round(risk_score, 4),
        "warn_threshold": round(warn_threshold, 4),
        "block_threshold": round(block_threshold, 4),
        "term_hits": term_hits[:8],
        "pattern_hits": pattern_hits[:8],
        "term_count": len(term_hits),
    }


def _validate_prompt_template_security(system_prompt: str, user_prompt_prefix: str) -> None:
    if not _GUARD_ENABLED:
        return
    if not f"{system_prompt}\n{user_prompt_prefix}".strip():
        return
    report = _prompt_template_guard_report(system_prompt, user_prompt_prefix)
    if report.get("action") == "block":
        terms = ", ".join(report.get("term_hits", [])[:4]) or "pattern-only"
        patterns = ", ".join(report.get("pattern_hits", [])[:3]) or "none"
        raise ValueError(
            "prompt template blocked by security guard: "
            f"score={report.get('risk_score')} terms={terms} patterns={patterns}"
        )
    if report.get("action") == "warn":
        _LOGGER.warning(
            "prompt template guard warning score=%s mode=%s terms=%s patterns=%s",
            report.get("risk_score"),
            report.get("mode"),
            ",".join(report.get("term_hits", [])[:6]),
            ",".join(report.get("pattern_hits", [])[:4]),
        )


def _normalize_prompt_setting_keys(value: list[str] | None) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for raw in value:
        key = str(raw or "").strip()
        if not key:
            continue
        if key in normalized:
            continue
        normalized.append(key[:191])
    return normalized[:200]


def _normalize_prompt_card_ids(value: list[int] | None) -> list[int]:
    if not isinstance(value, list):
        return []
    normalized: list[int] = []
    for raw in value:
        try:
            card_id = int(raw)
        except Exception:
            continue
        if card_id <= 0 or card_id in normalized:
            continue
        normalized.append(card_id)
    return normalized[:200]


def list_prompt_templates(db: Session, project_id: int) -> Iterable[PromptTemplate]:
    stmt = select(PromptTemplate).where(PromptTemplate.project_id == project_id).order_by(PromptTemplate.id.asc())
    return db.exec(stmt).all()


def get_prompt_template(db: Session, project_id: int, template_id: int) -> PromptTemplate | None:
    stmt = select(PromptTemplate).where(
        PromptTemplate.project_id == project_id,
        PromptTemplate.id == template_id,
    )
    return db.exec(stmt).first()


def _prompt_template_name_conflict(
    db: Session,
    *,
    project_id: int,
    name: str,
    exclude_template_id: int | None = None,
) -> bool:
    stmt = select(PromptTemplate).where(
        PromptTemplate.project_id == project_id,
        PromptTemplate.name == name,
    )
    row = db.exec(stmt).first()
    if not row:
        return False
    if exclude_template_id is not None and int(getattr(row, "id", 0) or 0) == int(exclude_template_id):
        return False
    return True


def _next_prompt_template_revision_version(db: Session, template_id: int) -> int:
    stmt = select(PromptTemplateRevision.version).where(PromptTemplateRevision.template_id == template_id)
    rows = db.exec(stmt).all()
    if not rows:
        return 1
    return max(int(item) for item in rows) + 1


def _append_prompt_template_revision(
    db: Session,
    *,
    template: PromptTemplate,
    operator_id: str,
    source: str,
) -> PromptTemplateRevision:
    if template.id is None:
        raise ValueError("prompt template id missing")
    revision = PromptTemplateRevision(
        template_id=template.id,
        project_id=template.project_id,
        version=_next_prompt_template_revision_version(db, template.id),
        name=template.name,
        system_prompt=template.system_prompt,
        user_prompt_prefix=template.user_prompt_prefix,
        knowledge_setting_keys=_normalize_prompt_setting_keys(template.knowledge_setting_keys),
        knowledge_card_ids=_normalize_prompt_card_ids(template.knowledge_card_ids),
        operator_id=(operator_id or "system").strip() or "system",
        source=source[:32],
        created_at=_utc_now(),
    )
    db.add(revision)
    return revision


def list_prompt_template_revisions(
    db: Session,
    *,
    project_id: int,
    template_id: int,
    limit: int = 20,
) -> Iterable[PromptTemplateRevision]:
    template = get_prompt_template(db, project_id, template_id)
    if not template:
        raise ValueError("prompt template not found")
    stmt = (
        select(PromptTemplateRevision)
        .where(
            PromptTemplateRevision.project_id == project_id,
            PromptTemplateRevision.template_id == template_id,
        )
        .order_by(PromptTemplateRevision.version.desc())
        .limit(limit)
    )
    return db.exec(stmt).all()


def create_prompt_template(
    db: Session,
    *,
    project_id: int,
    name: str,
    system_prompt: str,
    user_prompt_prefix: str,
    knowledge_setting_keys: list[str],
    knowledge_card_ids: list[int],
    operator_id: str,
) -> PromptTemplate:
    name_norm = _normalize_prompt_template_name(name)
    if _prompt_template_name_conflict(db, project_id=project_id, name=name_norm):
        raise ValueError("prompt template name already exists")
    normalized_system_prompt = _normalize_prompt_text(system_prompt, max_chars=40000)
    normalized_user_prefix = _normalize_prompt_text(user_prompt_prefix, max_chars=20000)
    _validate_prompt_template_security(normalized_system_prompt, normalized_user_prefix)

    row = PromptTemplate(
        project_id=project_id,
        name=name_norm,
        system_prompt=normalized_system_prompt,
        user_prompt_prefix=normalized_user_prefix,
        knowledge_setting_keys=_normalize_prompt_setting_keys(knowledge_setting_keys),
        knowledge_card_ids=_normalize_prompt_card_ids(knowledge_card_ids),
        created_at=_utc_now(),
        updated_at=_utc_now(),
    )
    db.add(row)
    db.flush()
    _append_prompt_template_revision(
        db,
        template=row,
        operator_id=operator_id,
        source="create",
    )
    db.commit()
    db.refresh(row)
    return row


def update_prompt_template(
    db: Session,
    *,
    project_id: int,
    template_id: int,
    name: str,
    system_prompt: str,
    user_prompt_prefix: str,
    knowledge_setting_keys: list[str],
    knowledge_card_ids: list[int],
    operator_id: str,
) -> PromptTemplate:
    row = get_prompt_template(db, project_id, template_id)
    if not row:
        raise ValueError("prompt template not found")
    name_norm = _normalize_prompt_template_name(name)
    if _prompt_template_name_conflict(
        db,
        project_id=project_id,
        name=name_norm,
        exclude_template_id=int(getattr(row, "id", 0) or 0),
    ):
        raise ValueError("prompt template name already exists")
    normalized_system_prompt = _normalize_prompt_text(system_prompt, max_chars=40000)
    normalized_user_prefix = _normalize_prompt_text(user_prompt_prefix, max_chars=20000)
    _validate_prompt_template_security(normalized_system_prompt, normalized_user_prefix)

    row.name = name_norm
    row.system_prompt = normalized_system_prompt
    row.user_prompt_prefix = normalized_user_prefix
    row.knowledge_setting_keys = _normalize_prompt_setting_keys(knowledge_setting_keys)
    row.knowledge_card_ids = _normalize_prompt_card_ids(knowledge_card_ids)
    row.updated_at = _utc_now()
    db.add(row)
    _append_prompt_template_revision(
        db,
        template=row,
        operator_id=operator_id,
        source="save",
    )
    db.commit()
    db.refresh(row)
    return row


def rollback_prompt_template(
    db: Session,
    *,
    project_id: int,
    template_id: int,
    target_version: int,
    operator_id: str,
) -> PromptTemplate:
    row = get_prompt_template(db, project_id, template_id)
    if not row:
        raise ValueError("prompt template not found")
    stmt = select(PromptTemplateRevision).where(
        PromptTemplateRevision.project_id == project_id,
        PromptTemplateRevision.template_id == template_id,
        PromptTemplateRevision.version == target_version,
    )
    target = db.exec(stmt).first()
    if not target:
        raise ValueError(f"target_version {target_version} not found")

    normalized_system_prompt = _normalize_prompt_text(target.system_prompt, max_chars=40000)
    normalized_user_prefix = _normalize_prompt_text(target.user_prompt_prefix, max_chars=20000)
    _validate_prompt_template_security(normalized_system_prompt, normalized_user_prefix)

    row.name = target.name
    row.system_prompt = normalized_system_prompt
    row.user_prompt_prefix = normalized_user_prefix
    row.knowledge_setting_keys = _normalize_prompt_setting_keys(target.knowledge_setting_keys)
    row.knowledge_card_ids = _normalize_prompt_card_ids(target.knowledge_card_ids)
    row.updated_at = _utc_now()
    db.add(row)
    _append_prompt_template_revision(
        db,
        template=row,
        operator_id=operator_id,
        source="rollback",
    )
    db.commit()
    db.refresh(row)
    return row


def delete_prompt_template(db: Session, project_id: int, template_id: int) -> int:
    row = get_prompt_template(db, project_id, template_id)
    if not row:
        raise ValueError("prompt template not found")
    deleted_id = int(getattr(row, "id", 0) or 0)
    rev_stmt = select(PromptTemplateRevision).where(
        PromptTemplateRevision.project_id == project_id,
        PromptTemplateRevision.template_id == template_id,
    )
    revisions = db.exec(rev_stmt).all()
    for revision in revisions:
        db.delete(revision)
    db.delete(row)
    db.commit()
    return deleted_id

