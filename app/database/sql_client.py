"""SQLAlchemy engine, session, and seed helpers."""

import csv
import logging
import os
from typing import cast

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
        return cast(str, row.value) if row else None
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

def get_all_parking_ids_and_names() -> list[str]:
    """
    Return a list of all distinct parking_id values from DynamicConfig.
    Used to dynamically enumerate all known parking spaces.
    """
    session = SessionLocal()
    try:
        rows = session.query(DynamicConfig.parking_id).distinct().all()
        return [row.parking_id for row in rows]
    finally:
        session.close()


def save_reservation_with_thread(data: dict, thread_id: str) -> bool:
    """
    Save a confirmed reservation to the database including the given thread_id.
    Status is set to "pending" awaiting administrator review.

    Args:
        data: dict with parking_id, name, surname, car_number, start_date, end_date.
        thread_id: LangGraph thread_id to correlate this reservation with its chatbot session.

    Returns:
        True on success, False on error.
    """
    from app.database.models import Reservation
    from datetime import date

    session = SessionLocal()
    try:
        reservation = Reservation(
            parking_id=data["parking_id"],
            name=data["name"],
            surname=data["surname"],
            car_number=data["car_number"],
            start_date=date.fromisoformat(data["start_date"]),
            end_date=date.fromisoformat(data["end_date"]),
            status="pending",
            thread_id=thread_id,
        )
        session.add(reservation)
        session.commit()
        logger.info(
            "Reservation saved for %s %s with thread_id=%s",
            data["name"], data["surname"], thread_id,
        )
        return True
    except Exception as e:
        session.rollback()
        logger.error("Failed to save reservation with thread: %s", e)
        return False
    finally:
        session.close()


def get_pending_reservations() -> list[dict]:
    """Return all reservations with status 'pending' as a list of dicts."""
    session = SessionLocal()
    try:
        rows = session.query(Reservation).filter(Reservation.status == "pending").all()
        return [
            {
                "id": r.id,
                "parking_id": r.parking_id,
                "name": r.name,
                "surname": r.surname,
                "car_number": r.car_number,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "status": r.status,
                "thread_id": r.thread_id,
                "created_at": r.created_at,
            }
            for r in rows
        ]
    finally:
        session.close()


def get_reservation_by_thread_id(thread_id: str) -> dict | None:
    """Return a reservation by thread_id as a dict, or None if not found."""
    session = SessionLocal()
    try:
        r = session.query(Reservation).filter(Reservation.thread_id == thread_id).first()
        if r is None:
            return None
        return {
            "id": r.id,
            "parking_id": r.parking_id,
            "name": r.name,
            "surname": r.surname,
            "car_number": r.car_number,
            "start_date": r.start_date,
            "end_date": r.end_date,
            "status": r.status,
            "thread_id": r.thread_id,
            "created_at": r.created_at,
        }
    finally:
        session.close()


def update_reservation_status(thread_id: str, status: str) -> bool:
    """
    Update the status of a reservation identified by thread_id.

    Valid status values: "pending", "confirmed", "rejected".
    Returns True on success, False if no reservation found or on error.
    """
    session = SessionLocal()
    try:
        r = session.query(Reservation).filter(Reservation.thread_id == thread_id).first()
        if r is None:
            logger.warning("update_reservation_status: no reservation with thread_id=%s", thread_id)
            return False
        r.status = status
        session.commit()
        logger.info("Reservation thread_id=%s status updated to %s", thread_id, status)
        return True
    except Exception as e:
        session.rollback()
        logger.error("Failed to update reservation status: %s", e)
        return False
    finally:
        session.close()
