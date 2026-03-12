"""SQLAlchemy engine, session, and seed helpers."""

import csv
import logging
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base, DynamicConfig, Reservation

logger = logging.getLogger(__name__)


def get_engine():
    """Create and return SQLAlchemy engine from DATABASE_URL env var."""
    database_url = os.getenv("DATABASE_URL", "sqlite:///./parking.db")
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args)


SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)


def init_db():
    """
    Create all tables if they do not exist.
    Seed DynamicConfig from data/dynamic/seed_data.csv if the table is empty.
    """
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        if session.query(DynamicConfig).count() == 0:
            csv_path = os.path.join(
                os.path.dirname(__file__),
                "../../data/dynamic/seed_data.csv",
            )
            with open(csv_path, newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    session.add(
                        DynamicConfig(
                            parking_id=row["parking_id"],
                            type=row["type"],
                            key=row["key"],
                            value=row["value"],
                        )
                    )
            session.commit()
            logger.info("Database seeded from seed_data.csv")
    finally:
        session.close()


def get_dynamic_value(parking_id: str, key: str) -> str | None:
    """Return a single dynamic config value for a given parking_id and key."""
    session = SessionLocal()
    try:
        row = session.query(DynamicConfig).filter_by(parking_id=parking_id, key=key).first()
        return row.value if row else None
    finally:
        session.close()


def get_all_dynamic_for_parking(parking_id: str) -> dict:
    """
    Return all dynamic config entries for a parking space as a nested dict.
    """
    session = SessionLocal()
    try:
        rows = session.query(DynamicConfig).filter_by(parking_id=parking_id).all()
        result: dict = {}
        for row in rows:
            result.setdefault(row.type, {})[row.key] = row.value
        return result
    finally:
        session.close()


def get_all_parkings_summary() -> list[dict]:
    """
    Return a summary list of all parking IDs with their availability and price data.
    """
    session = SessionLocal()
    try:
        rows = session.query(DynamicConfig).all()
        summary: dict = {}
        for row in rows:
            summary.setdefault(row.parking_id, {}).setdefault(row.type, {})[row.key] = row.value
        return [{"parking_id": pid, **data} for pid, data in summary.items()]
    finally:
        session.close()
