"""Tests for LangGraph SqliteSaver checkpointer and thread management."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import date

from app.chatbot.graph import chatbot_graph, checkpointer, get_thread_config
from app.database.models import Base, Reservation
from app.database.sql_client import (
    get_pending_reservations,
    get_reservation_by_thread_id,
    update_reservation_status,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_in_memory_session():
    """Return a sessionmaker bound to a fresh in-memory SQLite database."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ---------------------------------------------------------------------------
# Tests for get_thread_config
# ---------------------------------------------------------------------------

def test_get_thread_config_format():
    """get_thread_config should return the correct LangGraph config dict."""
    result = get_thread_config("test-123")
    assert result == {"configurable": {"thread_id": "test-123"}}


# ---------------------------------------------------------------------------
# Tests for graph compilation with checkpointer
# ---------------------------------------------------------------------------

def test_graph_compiles_with_checkpointer():
    """chatbot_graph should be compiled and have a non-None checkpointer."""
    assert chatbot_graph is not None
    assert chatbot_graph.checkpointer is not None


# ---------------------------------------------------------------------------
# Tests for state persistence across invocations
# ---------------------------------------------------------------------------

def _mock_unknown_node_invoke(state, config=None):
    """Minimal graph invocation mock that exercises unknown intent path."""
    from langchain_core.messages import HumanMessage, AIMessage
    messages = list(state.get("messages", []))
    messages.append(HumanMessage(content=state["user_input"]))
    reply = "I can help with parking info or reservations."
    messages.append(AIMessage(content=reply))
    return {**state, "response": reply, "messages": messages}


@patch("app.chatbot.graph.build_guardrail_chain")
@patch("app.chatbot.graph.build_intent_chain")
@patch("app.chatbot.graph.get_weaviate_client")
@patch("app.chatbot.graph.get_retriever")
@patch("app.chatbot.graph.build_rag_chain")
def test_graph_persists_state_across_invocations(
    mock_rag_chain,
    mock_retriever,
    mock_weaviate,
    mock_intent_chain,
    mock_guardrail_chain,
):
    """
    Invoking the graph twice with the same thread_id should accumulate messages.
    The second invocation's state should contain messages from the first.
    """
    # Guardrail chain returns "allowed" (not blocked)
    guardrail_mock = MagicMock()
    guardrail_mock.invoke.return_value = "allowed"
    mock_guardrail_chain.return_value = guardrail_mock

    # Intent chain always returns "unknown" to avoid RAG/Weaviate calls
    intent_mock = MagicMock()
    intent_mock.invoke.return_value = "unknown"
    mock_intent_chain.return_value = intent_mock

    thread = "persist-test-thread"
    config = get_thread_config(thread)

    state1 = {
        "messages": [],
        "user_input": "Hello, what parking do you have?",
        "intent": "",
        "reservation_data": {},
        "guardrail_triggered": False,
        "response": "",
    }
    result1 = chatbot_graph.invoke(state1, config=config)
    assert len(result1["messages"]) >= 2  # at least one human + one AI message

    state2 = {
        "messages": result1["messages"],
        "user_input": "Tell me more",
        "intent": "",
        "reservation_data": {},
        "guardrail_triggered": False,
        "response": "",
    }
    result2 = chatbot_graph.invoke(state2, config=config)
    # The second result must contain history from both turns
    assert len(result2["messages"]) > len(result1["messages"])


@patch("app.chatbot.graph.build_guardrail_chain")
@patch("app.chatbot.graph.build_intent_chain")
def test_different_thread_ids_have_independent_state(
    mock_intent_chain,
    mock_guardrail_chain,
):
    """
    Invoking the graph with different thread_ids should yield independent states.
    Messages produced for thread-A must not appear in thread-B.
    """
    guardrail_mock = MagicMock()
    guardrail_mock.invoke.return_value = "allowed"
    mock_guardrail_chain.return_value = guardrail_mock

    intent_mock = MagicMock()
    intent_mock.invoke.return_value = "unknown"
    mock_intent_chain.return_value = intent_mock

    state_a = {
        "messages": [],
        "user_input": "Message for thread A",
        "intent": "",
        "reservation_data": {},
        "guardrail_triggered": False,
        "response": "",
    }
    result_a = chatbot_graph.invoke(state_a, config=get_thread_config("thread-A-iso"))

    state_b = {
        "messages": [],
        "user_input": "Message for thread B",
        "intent": "",
        "reservation_data": {},
        "guardrail_triggered": False,
        "response": "",
    }
    result_b = chatbot_graph.invoke(state_b, config=get_thread_config("thread-B-iso"))

    # Collect all human message contents from each result
    from langchain_core.messages import HumanMessage
    msgs_a = {m.content for m in result_a["messages"] if isinstance(m, HumanMessage)}
    msgs_b = {m.content for m in result_b["messages"] if isinstance(m, HumanMessage)}

    assert "Message for thread A" in msgs_a
    assert "Message for thread B" in msgs_b
    # Thread A should NOT contain thread B's message and vice versa
    assert "Message for thread B" not in msgs_a
    assert "Message for thread A" not in msgs_b


# ---------------------------------------------------------------------------
# Tests for sql_client helpers (using in-memory DB)
# ---------------------------------------------------------------------------

def test_update_reservation_status_changes_status():
    """update_reservation_status should change status from pending to confirmed."""
    InMemorySession = _make_in_memory_session()

    # Patch SessionLocal used by sql_client functions
    with patch("app.database.sql_client.SessionLocal", InMemorySession):
        session = InMemorySession()
        try:
            reservation = Reservation(
                parking_id="parking_001",
                name="Test",
                surname="User",
                car_number="XY-9999",
                start_date=date(2026, 4, 1),
                end_date=date(2026, 4, 3),
                status="pending",
                thread_id="status-test-thread",
            )
            session.add(reservation)
            session.commit()
        finally:
            session.close()

        success = update_reservation_status("status-test-thread", "confirmed")
        assert success is True

        result = get_reservation_by_thread_id("status-test-thread")
        assert result is not None
        assert result["status"] == "confirmed"


def test_get_pending_reservations_returns_only_pending():
    """get_pending_reservations should return only reservations with status pending."""
    InMemorySession = _make_in_memory_session()

    with patch("app.database.sql_client.SessionLocal", InMemorySession):
        session = InMemorySession()
        try:
            session.add(Reservation(
                parking_id="parking_002",
                name="Alice",
                surname="Smith",
                car_number="AA-1111",
                start_date=date(2026, 5, 1),
                end_date=date(2026, 5, 2),
                status="pending",
                thread_id="pending-thread-1",
            ))
            session.add(Reservation(
                parking_id="parking_003",
                name="Bob",
                surname="Jones",
                car_number="BB-2222",
                start_date=date(2026, 5, 3),
                end_date=date(2026, 5, 4),
                status="confirmed",
                thread_id="confirmed-thread-1",
            ))
            session.commit()
        finally:
            session.close()

        pending = get_pending_reservations()
        assert len(pending) == 1
        assert pending[0]["thread_id"] == "pending-thread-1"
        assert pending[0]["status"] == "pending"
