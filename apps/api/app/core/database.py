from sqlalchemy import inspect, text
from sqlmodel import Session, SQLModel, create_engine

from app.core.config import settings
import app.models  # noqa: F401  # ensure model metadata is registered


engine = create_engine(settings.database_url, echo=False, pool_pre_ping=True)


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = inspect(engine)
    try:
        columns = inspector.get_columns(table_name)
    except Exception:
        return False
    return any(str(item.get("name") or "") == column_name for item in columns)


def _ensure_chatmessage_provenance_column() -> None:
    if _has_column("chatmessage", "provenance"):
        return

    dialect_name = str(engine.dialect.name or "").lower()
    statement = "ALTER TABLE chatmessage ADD COLUMN provenance JSON"
    if dialect_name.startswith("postgres"):
        statement = "ALTER TABLE chatmessage ADD COLUMN IF NOT EXISTS provenance JSON"

    with engine.begin() as connection:
        connection.execute(text(statement))


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    _ensure_chatmessage_provenance_column()


def get_session():
    with Session(engine) as session:
        yield session
