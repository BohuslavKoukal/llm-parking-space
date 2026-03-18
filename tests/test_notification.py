"""Tests for admin decision notification delivery and node state flags."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from app.chatbot.graph import ChatState, notify_rejection_node, record_reservation_node
from app.main import check_and_deliver_admin_decision
from langchain_core.runnables import RunnableConfig


def _sample_state() -> ChatState:
    return {
        "messages": [],
        "user_input": "check status",
        "intent": "reservation",
        "reservation_data": {
            "parking_id": "parking_001",
            "name": "John",
            "surname": "Doe",
            "car_number": "CAR-123",
            "start_date": "2026-04-01",
            "end_date": "2026-04-03",
            "confirmed": "yes",
        },
        "guardrail_triggered": False,
        "response": "",
        "admin_decision": "",
        "awaiting_admin": True,
    }


def _sample_config() -> RunnableConfig:
    return cast(RunnableConfig, {"configurable": {"thread_id": "thread-1234abcd"}})


def test_check_admin_decision_returns_false_when_still_interrupted():
    snapshot = SimpleNamespace(next=("submit_to_admin",), values={})
    with patch("app.main.chatbot_graph.get_state", return_value=snapshot):
        delivered, response = check_and_deliver_admin_decision(_sample_config(), "status?")

    assert delivered is False
    assert "pending" in response.lower()


def test_check_admin_decision_returns_true_when_approved():
    snapshot = SimpleNamespace(
        next=(),
        values={"admin_decision": "approved", "response": "Your reservation is confirmed."},
    )
    with patch("app.main.chatbot_graph.get_state", return_value=snapshot):
        delivered, response = check_and_deliver_admin_decision(_sample_config(), "status?")

    assert delivered is True
    assert "confirmed" in response.lower() or "approved" in response.lower()


def test_check_admin_decision_returns_true_when_rejected():
    snapshot = SimpleNamespace(
        next=(),
        values={"admin_decision": "rejected", "response": "Your reservation was rejected."},
    )
    with patch("app.main.chatbot_graph.get_state", return_value=snapshot):
        delivered, response = check_and_deliver_admin_decision(_sample_config(), "status?")

    assert delivered is True
    assert "rejected" in response.lower()


def test_check_admin_decision_uses_fallback_when_response_empty():
    snapshot = SimpleNamespace(next=(), values={"admin_decision": "approved", "response": ""})
    with patch("app.main.chatbot_graph.get_state", return_value=snapshot):
        delivered, response = check_and_deliver_admin_decision(_sample_config(), "status?")

    assert delivered is True
    assert "approved" in response.lower() or "confirmed" in response.lower()


def test_check_admin_decision_handles_exception_gracefully():
    with patch("app.main.chatbot_graph.get_state", side_effect=RuntimeError("boom")):
        delivered, response = check_and_deliver_admin_decision(_sample_config(), "status?")

    assert delivered is False
    assert "unable to check reservation status" in response.lower()


def test_record_reservation_node_sets_admin_decision_approved():
    state = _sample_state()
    config = _sample_config()

    with (
        patch("app.database.sql_client.update_reservation_status", return_value=True),
        patch(
            "app.database.sql_client.get_reservation_by_thread_id",
            return_value={
                "name": "John",
                "surname": "Doe",
                "car_number": "CAR-123",
                "parking_id": "parking_001",
                "start_date": "2026-04-01",
                "end_date": "2026-04-03",
            },
        ),
        patch("app.chatbot.graph.write_reservation_via_mcp", return_value="Reservation written successfully: ok"),
    ):
        result = record_reservation_node(state, config)

    assert result["admin_decision"] == "approved"
    assert result["awaiting_admin"] is False
    assert bool(result["response"])


def test_notify_rejection_node_sets_admin_decision_rejected():
    state = _sample_state()
    config = _sample_config()

    with patch("app.database.sql_client.update_reservation_status", return_value=True):
        result = notify_rejection_node(state, config)

    assert result["admin_decision"] == "rejected"
    assert result["awaiting_admin"] is False
    assert bool(result["response"])
