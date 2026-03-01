from sqlmodel import Session, SQLModel, create_engine

from app.core.config import settings
import app.models  # noqa: F401  # ensure model metadata is registered


engine = create_engine(settings.database_url, echo=False, pool_pre_ping=True)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
