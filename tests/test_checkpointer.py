"""Tests for LangGraph checkpointer integration and related SQL helpers."""

import uuid
import pytest
from unittest.mock import patch, MagicMock

from app.chatbot.graph import get_thread_config, chatbot_graph, build_graph
from app.chatbot.graph import ChatState
from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------------
# Helper: context manager that patches all external I/O for a graph invocation
# so tests run without real OpenAI / Weaviate connections.
# ---------------------------------------------------------------------------

def _all_external_patches():
    """Return a list of patch objects that suppress all external I/O in the graph."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Test info response"

    guardrail_chain = MagicMock()
    guardrail_chain.invoke.return_value = "allowed"

    intent_chain = MagicMock()
    intent_chain.invoke.return_value = "info"

    rag_chain = MagicMock()
    rag_chain.invoke.return_value = "Test info response"

    mock_client = MagicMock()
    mock_retriever = MagicMock()

    return [
        patch("app.chatbot.graph.is_sensitive", return_value=False),
        patch("app.chatbot.graph.build_guardrail_chain", return_value=guardrail_chain),
        patch("app.chatbot.graph.build_intent_chain", return_value=intent_chain),
        patch("app.chatbot.graph.build_rag_chain", return_value=rag_chain),
        patch("app.chatbot.graph.get_weaviate_client", return_value=mock_client),
        patch("app.chatbot.graph.get_retriever", return_value=mock_retriever),
        patch("app.chatbot.graph.get_all_parkings_summary", return_value=[]),
        patch("app.chatbot.graph.get_all_parking_ids_and_names", return_value=["parking_001"]),
    ]


# ---------------------------------------------------------------------------
# Fixture: fresh in-memory graph for each test that invokes the graph
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_graph():
    """
    Build a fresh compiled graph backed by an in-memory SQLite checkpointer.

    Using ':memory:' ensures each test gets a clean, isolated checkpoint store
    with no risk of a closed-connection error from a shared module-level saver.
    """
    graph, _ = build_graph(":memory:")
    return graph


# ---------------------------------------------------------------------------
# Test 1 – get_thread_config format
# ---------------------------------------------------------------------------

def test_get_thread_config_format():
    """get_thread_config should return the correct LangGraph configurable dict."""
    result = get_thread_config("test-123")
    assert result == {"configurable": {"thread_id": "test-123"}}


# ---------------------------------------------------------------------------
# Test 2 – graph compiles with checkpointer attached
# ---------------------------------------------------------------------------

def test_graph_compiles_with_checkpointer():
    """The module-level chatbot_graph should be compiled and have a checkpointer."""
    assert chatbot_graph is not None
    assert chatbot_graph.checkpointer is not None


# ---------------------------------------------------------------------------
# Test 3 – state persists (messages accumulate) across two invocations
# ---------------------------------------------------------------------------

def test_graph_persists_state_across_invocations(fresh_graph):
    """
    Messages produced by the first invocation should be passed to the second
    invocation and appear in the final result, demonstrating multi-turn
    conversation continuity.
    """
    thread_id = f"persist-{uuid.uuid4()}"
    config = get_thread_config(thread_id)

    state_1: ChatState = {
        "messages": [],
        "user_input": "Tell me about parking prices.",
        "intent": "",
        "reservation_data": {},
        "guardrail_triggered": False,
        "response": "",
    }

    patches = _all_external_patches()
    for p in patches:
        p.start()

    try:
        result_1 = fresh_graph.invoke(state_1, config=config)
        messages_after_turn_1 = result_1.get("messages", [])
        assert len(messages_after_turn_1) == 2, "Expected one HumanMessage + one AIMessage after turn 1"

        state_2: ChatState = {
            "messages": messages_after_turn_1,
            "user_input": "What about availability?",
            "intent": "",
            "reservation_data": {},
            "guardrail_triggered": False,
            "response": "",
        }
        result_2 = fresh_graph.invoke(state_2, config=config)
        messages_after_turn_2 = result_2.get("messages", [])

        assert len(messages_after_turn_2) == 4, "Expected four messages after turn 2 (two per turn)"
        # The first turn's human message should still be present
        assert messages_after_turn_1[0] in messages_after_turn_2
    finally:
        for p in patches:
            p.stop()


# ---------------------------------------------------------------------------
# Test 4 – different thread IDs have independent state
# ---------------------------------------------------------------------------

def test_different_thread_ids_have_independent_state(fresh_graph):
    """
    Two conversations with different thread IDs must not share message history.
    """
    config_a = get_thread_config(f"thread-A-{uuid.uuid4()}")
    config_b = get_thread_config(f"thread-B-{uuid.uuid4()}")

    state_a: ChatState = {
        "messages": [],
        "user_input": "Question from thread A.",
        "intent": "",
        "reservation_data": {},
        "guardrail_triggered": False,
        "response": "",
    }
    state_b: ChatState = {
        "messages": [],
        "user_input": "Question from thread B.",
        "intent": "",
        "reservation_data": {},
        "guardrail_triggered": False,
        "response": "",
    }

    patches = _all_external_patches()
    for p in patches:
        p.start()

    try:
        result_a = fresh_graph.invoke(state_a, config=config_a)
        result_b = fresh_graph.invoke(state_b, config=config_b)

        msgs_a = result_a.get("messages", [])
        msgs_b = result_b.get("messages", [])

        # Each thread should have exactly its own two messages
        assert len(msgs_a) == 2
        assert len(msgs_b) == 2

        # The human messages should contain the correct content per thread
        assert msgs_a[0].content == "Question from thread A."
        assert msgs_b[0].content == "Question from thread B."

        # The two lists are independent objects
        assert msgs_a is not msgs_b

        # Human message contents must differ between threads
        a_human_contents = [m.content for m in msgs_a if isinstance(m, HumanMessage)]
        b_human_contents = [m.content for m in msgs_b if isinstance(m, HumanMessage)]
        assert a_human_contents != b_human_contents
    finally:
        for p in patches:
            p.stop()


# ---------------------------------------------------------------------------
# Test 5 – update_reservation_status changes status
# ---------------------------------------------------------------------------

def test_update_reservation_status_changes_status():
    """update_reservation_status should flip a reservation's status in the DB."""
    from datetime import date
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database.models import Base, Reservation
    import app.database.sql_client as sql_client

    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    TestSession = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)

    tid = f"tid-{uuid.uuid4()}"

    session = TestSession()
    try:
        reservation = Reservation(
            parking_id="parking_001",
            name="Test",
            surname="User",
            car_number="TEST-001",
            start_date=date(2026, 4, 1),
            end_date=date(2026, 4, 3),
            status="pending",
            thread_id=tid,
        )
        session.add(reservation)
        session.commit()
    finally:
        session.close()

    original_session_local = sql_client.SessionLocal
    sql_client.SessionLocal = TestSession
    try:
        success = sql_client.update_reservation_status(tid, "confirmed")
        assert success is True

        result = sql_client.get_reservation_by_thread_id(tid)
        assert result is not None
        assert result["status"] == "confirmed"
    finally:
        sql_client.SessionLocal = original_session_local
        Base.metadata.drop_all(bind=test_engine)


# ---------------------------------------------------------------------------
# Test 6 – get_pending_reservations returns only pending rows
# ---------------------------------------------------------------------------

def test_get_pending_reservations_returns_only_pending():
    """get_pending_reservations should exclude non-pending reservations."""
    from datetime import date
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database.models import Base, Reservation
    import app.database.sql_client as sql_client

    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    TestSession = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)

    tid_pending = f"tid-pending-{uuid.uuid4()}"
    tid_confirmed = f"tid-confirmed-{uuid.uuid4()}"

    session = TestSession()
    try:
        session.add(Reservation(
            parking_id="parking_001",
            name="Alice",
            surname="Smith",
            car_number="PEND-001",
            start_date=date(2026, 4, 1),
            end_date=date(2026, 4, 3),
            status="pending",
            thread_id=tid_pending,
        ))
        session.add(Reservation(
            parking_id="parking_002",
            name="Bob",
            surname="Jones",
            car_number="CONF-002",
            start_date=date(2026, 4, 5),
            end_date=date(2026, 4, 7),
            status="confirmed",
            thread_id=tid_confirmed,
        ))
        session.commit()
    finally:
        session.close()

    original_session_local = sql_client.SessionLocal
    sql_client.SessionLocal = TestSession
    try:
        pending = sql_client.get_pending_reservations()
        assert len(pending) == 1
        assert pending[0]["thread_id"] == tid_pending
        assert pending[0]["status"] == "pending"
    finally:
        sql_client.SessionLocal = original_session_local
        Base.metadata.drop_all(bind=test_engine)
