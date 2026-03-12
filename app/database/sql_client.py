"""SQLAlchemy engine, session, and seed helpers."""

import csv
import os
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.database.models import Base, DynamicConfig

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./parking.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_engine():
    """Create and return SQLAlchemy engine based on DATABASE_URL env variable."""
    return engine


def _seed_dynamic_config(session: Session) -> None:
    seed_path = Path("data/dynamic/seed_data.csv")
    if not seed_path.exists():
        return

    with seed_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            session.add(
                DynamicConfig(
                    type=row["type"],
                    key=row["key"],
                    value=row["value"],
                )
            )


def init_db() -> None:
    """Create database tables and seed dynamic configuration if empty."""
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as session:
        existing = session.scalar(select(DynamicConfig.id).limit(1))
        if existing is None:
            _seed_dynamic_config(session)
            session.commit()


def get_dynamic_value(key: str) -> str | None:
    """Fetch dynamic config value by key, returning None if key does not exist."""
    with SessionLocal() as session:
        record = session.scalar(select(DynamicConfig).where(DynamicConfig.key == key))
        return record.value if record else None
