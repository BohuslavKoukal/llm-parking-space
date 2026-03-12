"""SQLAlchemy ORM models for parking chatbot data."""

from datetime import datetime

from sqlalchemy import Column, Date, DateTime, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class DynamicConfig(Base):
    """Dynamic key-value data scoped by parking ID."""

    __tablename__ = "dynamic_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parking_id = Column(String, nullable=False, index=True)
    type = Column(String, nullable=False)
    key = Column(String, nullable=False)
    value = Column(String, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Reservation(Base):
    """Reservation entity for parking bookings."""

    __tablename__ = "reservations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parking_id = Column(String, nullable=False)
    name = Column(String, nullable=False)
    surname = Column(String, nullable=False)
    car_number = Column(String, nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
