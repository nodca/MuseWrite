from sqlmodel import Session, select

from app.models.book_memory import CharacterProfile


def list_character_profiles(
    db: Session,
    project_id: int,
    *,
    status: str | None = "active",
) -> list[CharacterProfile]:
    stmt = select(CharacterProfile).where(CharacterProfile.project_id == project_id)
    if status is not None:
        stmt = stmt.where(CharacterProfile.status == status)
    stmt = stmt.order_by(CharacterProfile.canonical_name.asc(), CharacterProfile.id.asc())
    return db.exec(stmt).all()


def create_character_profile(
    db: Session,
    *,
    project_id: int,
    canonical_name: str,
    aliases: list[str] | None = None,
    public_traits: list[str] | None = None,
    private_traits: list[str] | None = None,
    core_goals: list[str] | None = None,
    fears: list[str] | None = None,
    taboos: list[str] | None = None,
    default_voice_notes: str = "",
    status: str = "active",
    source_refs: list[dict] | None = None,
) -> CharacterProfile:
    profile = CharacterProfile(
        project_id=project_id,
        canonical_name=str(canonical_name or "").strip(),
        aliases=list(aliases or []),
        public_traits=list(public_traits or []),
        private_traits=list(private_traits or []),
        core_goals=list(core_goals or []),
        fears=list(fears or []),
        taboos=list(taboos or []),
        default_voice_notes=str(default_voice_notes or "").strip(),
        status=str(status or "active").strip() or "active",
        source_refs=list(source_refs or []),
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


def update_character_profile(
    db: Session,
    *,
    project_id: int,
    character_id: int,
    canonical_name: str | None = None,
    aliases: list[str] | None = None,
    public_traits: list[str] | None = None,
    private_traits: list[str] | None = None,
    core_goals: list[str] | None = None,
    fears: list[str] | None = None,
    taboos: list[str] | None = None,
    default_voice_notes: str | None = None,
    status: str | None = None,
    source_refs: list[dict] | None = None,
) -> CharacterProfile:
    profile = db.get(CharacterProfile, character_id)
    if profile is None or int(profile.project_id) != int(project_id):
        raise ValueError("character profile not found")
    if canonical_name is not None:
        profile.canonical_name = str(canonical_name).strip()
    if aliases is not None:
        profile.aliases = list(aliases)
    if public_traits is not None:
        profile.public_traits = list(public_traits)
    if private_traits is not None:
        profile.private_traits = list(private_traits)
    if core_goals is not None:
        profile.core_goals = list(core_goals)
    if fears is not None:
        profile.fears = list(fears)
    if taboos is not None:
        profile.taboos = list(taboos)
    if default_voice_notes is not None:
        profile.default_voice_notes = str(default_voice_notes).strip()
    if status is not None:
        profile.status = str(status).strip() or "active"
    if source_refs is not None:
        profile.source_refs = list(source_refs)
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile
