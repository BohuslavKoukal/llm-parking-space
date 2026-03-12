import pytest
from unittest.mock import MagicMock, patch

from app.rag.ingestion import chunk_parking_object


def test_chunk_parking_object_returns_four_sections():
    """chunk_parking_object should return exactly 4 chunks: general, location, booking, contact."""
    sample = {
        "id": "parking_001",
        "parking_name": "Test Park",
        "address": "1 Test St",
        "description": "A test parking.",
        "features": ["CCTV"],
        "location": {
            "nearest_metro": "Test Metro",
            "walking_distance_minutes": 5,
            "latitude": 48.0,
            "longitude": 2.0,
        },
        "booking_process": "Provide your details.",
        "contact": {"phone": "123", "email": "test@test.com"},
    }
    chunks = chunk_parking_object(sample)
    assert len(chunks) == 4
    sections = {chunk["section"] for chunk in chunks}
    assert sections == {"general", "location", "booking", "contact"}


def test_chunk_parking_object_contains_parking_id():
    """Every chunk produced should carry the correct parking_id and parking_name."""
    sample = {
        "id": "parking_003",
        "parking_name": "Mall Park",
        "address": "2 Mall Rd",
        "description": "Mall parking.",
        "features": [],
        "location": {
            "nearest_metro": "Mall Metro",
            "walking_distance_minutes": 2,
            "latitude": 48.1,
            "longitude": 2.1,
        },
        "booking_process": "Book online.",
        "contact": {"phone": "456", "email": "mall@test.com"},
    }
    chunks = chunk_parking_object(sample)
    for chunk in chunks:
        assert chunk["parking_id"] == "parking_003"
        assert chunk["parking_name"] == "Mall Park"
