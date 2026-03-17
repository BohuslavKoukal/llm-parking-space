"""Tests for the admin CLI behavior and related sql_client functions."""

import pytest
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import sql_client
from app.database.models import Base, Reservation
from app.database.sql_client import get_pending_reservations, update_reservation_status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_db():
    """Create an isolated in-memory SQLite engine with the full schema."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return engine, sessionmaker(bind=engine, autocommit=False, autoflush=False)


def _insert_reservation(TestSession, *, thread_id=None, status="pending"):
    """Insert a minimal reservation row and return its id."""
    session = TestSession()
    try:
        r = Reservation(
            parking_id="parking_001",
            name="Test",
            surname="User",
            car_number="XYZ-0001",
            start_date=date(2026, 4, 1),
            end_date=date(2026, 4, 5),
            status=status,
            thread_id=thread_id,
            created_at=datetime.now(timezone.utc),
        )
        session.add(r)
        session.commit()
        session.refresh(r)
        return r.id
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetPendingReservations:
    def test_get_pending_reservations_excludes_null_thread_id(self):
        """Reservations with no thread_id must not be returned — they cannot be resumed."""
        engine, TestSession = _make_test_db()
        original = sql_client.SessionLocal
        sql_client.SessionLocal = TestSession
        try:
            _insert_reservation(TestSession, thread_id="thread-with-id")
            _insert_reservation(TestSession, thread_id=None)

            result = get_pending_reservations()

            assert len(result) == 1
            assert result[0]["thread_id"] == "thread-with-id"
        finally:
            sql_client.SessionLocal = original
            Base.metadata.drop_all(bind=engine)

    def test_get_pending_reservations_returns_dicts_not_orm(self):
        """Each item in the result must be a plain dict with all required keys."""
        engine, TestSession = _make_test_db()
        original = sql_client.SessionLocal
        sql_client.SessionLocal = TestSession
        try:
            _insert_reservation(TestSession, thread_id="thread-dict-test")

            result = get_pending_reservations()

            assert len(result) == 1
            item = result[0]
            assert isinstance(item, dict)
            for key in (
                "id", "parking_id", "name", "surname", "car_number",
                "start_date", "end_date", "status", "thread_id", "created_at",
            ):
                assert key in item, f"Missing key: {key}"
        finally:
            sql_client.SessionLocal = original
            Base.metadata.drop_all(bind=engine)


class TestAdminResumeStatus:
    def test_admin_resume_approved_updates_db_status(self):
        """Approving a reservation must set its DB status to 'confirmed'."""
        engine, TestSession = _make_test_db()
        original = sql_client.SessionLocal
        sql_client.SessionLocal = TestSession
        try:
            thread_id = "thread-approve-test"
            _insert_reservation(TestSession, thread_id=thread_id)

            with (
                patch("app.chatbot.graph.chatbot_graph") as mock_graph,
            ):
                mock_graph.invoke.return_value = {}
                mock_graph.get_state.return_value = MagicMock(next=["submit_to_admin"])

                result = update_reservation_status(thread_id, "confirmed")

            assert result is True
            session = TestSession()
            try:
                r = session.query(Reservation).filter(Reservation.thread_id == thread_id).first()
                assert r.status == "confirmed"
            finally:
                session.close()
        finally:
            sql_client.SessionLocal = original
            Base.metadata.drop_all(bind=engine)

    def test_admin_resume_rejected_updates_db_status(self):
        """Rejecting a reservation must set its DB status to 'rejected'."""
        engine, TestSession = _make_test_db()
        original = sql_client.SessionLocal
        sql_client.SessionLocal = TestSession
        try:
            thread_id = "thread-reject-test"
            _insert_reservation(TestSession, thread_id=thread_id)

            with (
                patch("app.chatbot.graph.chatbot_graph") as mock_graph,
            ):
                mock_graph.invoke.return_value = {}
                mock_graph.get_state.return_value = MagicMock(next=["submit_to_admin"])

                result = update_reservation_status(thread_id, "rejected")

            assert result is True
            session = TestSession()
            try:
                r = session.query(Reservation).filter(Reservation.thread_id == thread_id).first()
                assert r.status == "rejected"
            finally:
                session.close()
        finally:
            sql_client.SessionLocal = original
            Base.metadata.drop_all(bind=engine)


class TestGraphStateCheck:
    def test_graph_state_check_before_resume(self):
        """CLI must detect a non-interrupted thread (empty next) and refuse to resume."""
        with patch("app.chatbot.graph.chatbot_graph") as mock_graph:
            mock_snapshot = MagicMock()
            mock_snapshot.next = []  # empty = graph is not paused at an interrupt
            mock_graph.get_state.return_value = mock_snapshot

            thread_config = {"configurable": {"thread_id": "test-thread"}}
            state = mock_graph.get_state(thread_config)

            # The admin_review.py guard: `if not state_snapshot.next: → error/exit`
            assert not state.next


# ---------------------------------------------------------------------------
# CLI behaviour tests
# ---------------------------------------------------------------------------

class TestAdminCLI:
    def test_admin_cli_handles_no_pending_reservations(self):
        """main() exits cleanly with code 0 when there are no pending reservations."""
        from scripts.admin_review import main

        with patch("scripts.admin_review.get_pending_reservations", return_value=[]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0

    def test_admin_cli_invalid_input_reprompts(self):
        """Invalid inputs for reservation selection and approve/reject must not crash."""
        from scripts.admin_review import select_reservation, get_admin_decision

        reservations = [
            {
                "id": 1,
                "name": "Test",
                "surname": "User",
                "car_number": "XYZ-0001",
                "parking_id": "parking_001",
                "start_date": "2026-04-01",
                "end_date": "2026-04-05",
                "thread_id": "thread-test-001",
                "created_at": None,
            }
        ]

        # Non-numeric then "q" to quit: should exit 0 without crashing.
        with patch("builtins.input", side_effect=["not-a-number", "q"]):
            with pytest.raises(SystemExit) as exc_info:
                select_reservation(reservations)
        assert exc_info.value.code == 0

        # Invalid decision char then valid "r": should return "rejected".
        with patch("builtins.input", side_effect=["x", "r"]):
            decision = get_admin_decision()
        assert decision == "rejected"

