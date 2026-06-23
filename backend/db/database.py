"""SQLAlchemy engine/session setup for SQLite persistence.

A synchronous engine is intentional: the LangGraph pipeline is sync and runs in a
threadpool (see backend.api.app), so a plain sync Session is the simplest correct
choice. `check_same_thread=False` lets the threadpool worker reuse connections.
"""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(
    f"sqlite:///{settings.SQLITE_DB_PATH}",
    connect_args={"check_same_thread": False},
    future=True,
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create tables if they don't exist. Imports models for side-effect registration."""
    from backend.db import models  # noqa: F401 - registers ORM classes on Base

    Base.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    """FastAPI dependency that yields a session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
