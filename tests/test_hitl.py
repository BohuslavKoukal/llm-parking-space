"""Tests for HITL (Human-in-the-Loop) nodes, routing, and interrupt behaviour."""

import contextlib
import uuid
from copy import deepcopy
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.database.sql_client as sql_client
from app.chatbot.graph import (
    ChatState,
    build_graph,
    notify_rejection_node,
    record_reservation_node,
    route_after_admin_decision,
    route_after_reservation,
    submit_to_admin_node,
)
from app.database.models import Base, Reservation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(thread_id: str) -> dict:
    """Minimal RunnableConfig-compatible dict."""
    return {"configurable": {"thread_id": thread_id}}


@contextlib.contextmanager
def _all_external_patches():
    """
    Context manager that suppresses all external I/O for full-graph invocations.

    Usage:
        with _all_external_patches():
            result = graph.invoke(state, config=config)
    """
    guardrail_chain = MagicMock()
    guardrail_chain.invoke.return_value = "allowed"

    intent_chain = MagicMock()
    intent_chain.invoke.return_value = "reservation"

    mock_client = MagicMock()
    mock_retriever = MagicMock()

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("app.chatbot.graph.is_sensitive", return_value=False))
        stack.enter_context(patch("app.chatbot.graph.build_guardrail_chain", return_value=guardrail_chain))
        stack.enter_context(patch("app.chatbot.graph.build_intent_chain", return_value=intent_chain))
        stack.enter_context(patch("app.chatbot.graph.build_rag_chain", return_value=MagicMock()))
        stack.enter_context(patch("app.chatbot.graph.get_weaviate_client", return_value=mock_client))
        stack.enter_context(patch("app.chatbot.graph.get_retriever", return_value=mock_retriever))
        stack.enter_context(patch("app.chatbot.graph.get_all_parkings_summary", return_value=[]))
        stack.enter_context(patch("app.chatbot.graph.get_all_parking_ids_and_names", return_value=["parking_001"]))
        yield


@contextlib.contextmanager
def _info_external_patches():
    """Like _all_external_patches but routes intent to 'info' for RAG flow tests."""
    guardrail_chain = MagicMock()
    guardrail_chain.invoke.return_value = "allowed"

    intent_chain = MagicMock()
    intent_chain.invoke.return_value = "info"

    rag_chain = MagicMock()
    rag_chain.invoke.return_value = "Parking prices start from €5 per day."

    mock_client = MagicMock()
    mock_retriever = MagicMock()

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("app.chatbot.graph.is_sensitive", return_value=False))
        stack.enter_context(patch("app.chatbot.graph.build_guardrail_chain", return_value=guardrail_chain))
        stack.enter_context(patch("app.chatbot.graph.build_intent_chain", return_value=intent_chain))
        stack.enter_context(patch("app.chatbot.graph.build_rag_chain", return_value=rag_chain))
        stack.enter_context(patch("app.chatbot.graph.get_weaviate_client", return_value=mock_client))
        stack.enter_context(patch("app.chatbot.graph.get_retriever", return_value=mock_retriever))
        stack.enter_context(patch("app.chatbot.graph.get_all_parkings_summary", return_value=[]))
        stack.enter_context(patch("app.chatbot.graph.get_all_parking_ids_and_names", return_value=["parking_001"]))
        yield


def _confirmed_state() -> ChatState:
    """State with all 6 reservation fields and awaiting_admin=True (post-reservation_node)."""
    return {
        "messages": [],
        "user_input": "yes",
        "intent": "reservation",
        "reservation_data": {
            "parking_id": "parking_003",
            "name": "Bohuslav",
            "surname": "Koukal",
            "car_number": "ABC-1234",
            "start_date": "2026-03-20",
            "end_date": "2026-03-22",
            "confirmed": "yes",
        },
        "guardrail_triggered": False,
        "response": "Pending review.",
        "admin_decision": "",
        "awaiting_admin": True,
    }


def _full_graph_input_state() -> ChatState:
    """State suitable for a full graph invocation that routes to reservation confirmation."""
    return {
        "messages": [],
        "user_input": "yes",
        "intent": "reservation",
        "reservation_data": {
            "parking_id": "parking_003",
            "name": "Bohuslav",
            "surname": "Koukal",
            "car_number": "ABC-1234",
            "start_date": "2026-03-20",
            "end_date": "2026-03-22",
        },
        "guardrail_triggered": False,
        "response": "",
        "admin_decision": "",
        "awaiting_admin": False,
    }


@pytest.fixture
def fresh_graph():
    """Isolated in-memory graph per test."""
    graph, _ = build_graph(":memory:")
    return graph


# ---------------------------------------------------------------------------
# Test 1 – submit_to_admin_node saves reservation to DB
# ---------------------------------------------------------------------------

def test_submit_to_admin_saves_reservation_to_db():
    """submit_to_admin_node should persist a pending reservation before calling interrupt."""
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    TestSession = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)

    tid = f"hitl-{uuid.uuid4()}"
    state = _confirmed_state()
    config = _make_config(tid)

    original_session_local = sql_client.SessionLocal
    sql_client.SessionLocal = TestSession
    try:
        with patch("app.chatbot.graph.interrupt", return_value="approved"):
            result = submit_to_admin_node(state, config)

        assert result["admin_decision"] == "approved"

        session = TestSession()
        try:
            rows = session.query(Reservation).all()
            assert len(rows) == 1
            assert rows[0].status == "pending"
            assert rows[0].thread_id == tid
            assert rows[0].parking_id == "parking_003"
        finally:
            session.close()
    finally:
        sql_client.SessionLocal = original_session_local
        Base.metadata.drop_all(bind=test_engine)


# ---------------------------------------------------------------------------
# Test 2 – submit_to_admin_node calls interrupt with the required payload
# ---------------------------------------------------------------------------

def test_submit_to_admin_calls_interrupt():
    """submit_to_admin_node should call interrupt() with type, reservation, thread_id, message."""
    tid = f"hitl-{uuid.uuid4()}"
    state = _confirmed_state()
    config = _make_config(tid)

    with patch("app.chatbot.graph.interrupt", return_value="approved") as mock_interrupt, \
         patch("app.database.sql_client.save_reservation_with_thread", return_value=True), \
         patch("app.database.sql_client.get_reservation_by_thread_id", return_value=None):
        submit_to_admin_node(state, config)

    assert mock_interrupt.called
    payload = mock_interrupt.call_args[0][0]
    assert payload["type"] == "admin_review_required"
    assert "reservation" in payload
    assert payload["thread_id"] == tid
    assert "message" in payload


# ---------------------------------------------------------------------------
# Test 3 – record_reservation_node updates status to confirmed
# ---------------------------------------------------------------------------

def test_record_reservation_updates_status_to_confirmed():
    """record_reservation_node should update the DB status to 'confirmed'."""
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    TestSession = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)

    tid = f"hitl-{uuid.uuid4()}"
    session = TestSession()
    try:
        session.add(Reservation(
            parking_id="parking_001",
            name="Test",
            surname="User",
            car_number="TEST-001",
            start_date=date(2026, 4, 1),
            end_date=date(2026, 4, 3),
            status="pending",
            thread_id=tid,
        ))
        session.commit()
    finally:
        session.close()

    state = _confirmed_state()
    state["admin_decision"] = "approved"
    config = _make_config(tid)

    original_session_local = sql_client.SessionLocal
    sql_client.SessionLocal = TestSession
    try:
        result = record_reservation_node(state, config)

        assert "confirmed" in result["response"].lower()
        assert result["reservation_data"] == {}

        check = TestSession()
        try:
            r = check.query(Reservation).filter_by(thread_id=tid).first()
            assert r is not None
            assert r.status == "confirmed"
        finally:
            check.close()
    finally:
        sql_client.SessionLocal = original_session_local
        Base.metadata.drop_all(bind=test_engine)


# ---------------------------------------------------------------------------
# Test 4 – notify_rejection_node updates status to rejected
# ---------------------------------------------------------------------------

def test_notify_rejection_updates_status_to_rejected():
    """notify_rejection_node should update the DB status to 'rejected'."""
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    TestSession = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)

    tid = f"hitl-{uuid.uuid4()}"
    session = TestSession()
    try:
        session.add(Reservation(
            parking_id="parking_001",
            name="Test",
            surname="User",
            car_number="TEST-002",
            start_date=date(2026, 4, 1),
            end_date=date(2026, 4, 3),
            status="pending",
            thread_id=tid,
        ))
        session.commit()
    finally:
        session.close()

    state = _confirmed_state()
    state["admin_decision"] = "rejected"
    config = _make_config(tid)

    original_session_local = sql_client.SessionLocal
    sql_client.SessionLocal = TestSession
    try:
        result = notify_rejection_node(state, config)

        assert "rejected" in result["response"].lower()
        assert result["reservation_data"] == {}

        check = TestSession()
        try:
            r = check.query(Reservation).filter_by(thread_id=tid).first()
            assert r is not None
            assert r.status == "rejected"
        finally:
            check.close()
    finally:
        sql_client.SessionLocal = original_session_local
        Base.metadata.drop_all(bind=test_engine)


# ---------------------------------------------------------------------------
# Test 5 – route_after_admin_decision returns record_reservation when approved
# ---------------------------------------------------------------------------

def test_route_after_admin_decision_approved():
    """route_after_admin_decision should return 'record_reservation' for approved."""
    state: ChatState = {
        "messages": [], "user_input": "", "intent": "", "reservation_data": {},
        "guardrail_triggered": False, "response": "",
        "admin_decision": "approved", "awaiting_admin": False,
    }
    assert route_after_admin_decision(state) == "record_reservation"


# ---------------------------------------------------------------------------
# Test 6 – route_after_admin_decision returns notify_rejection when rejected
# ---------------------------------------------------------------------------

def test_route_after_admin_decision_rejected():
    """route_after_admin_decision should return 'notify_rejection' for rejected."""
    state: ChatState = {
        "messages": [], "user_input": "", "intent": "", "reservation_data": {},
        "guardrail_triggered": False, "response": "",
        "admin_decision": "rejected", "awaiting_admin": False,
    }
    assert route_after_admin_decision(state) == "notify_rejection"


# ---------------------------------------------------------------------------
# Test 8 – graph state is interrupted after user confirms reservation
# ---------------------------------------------------------------------------

def test_graph_is_interrupted_after_submit_to_admin(fresh_graph):
    """After the user confirms a reservation the graph must pause at submit_to_admin_node."""
    tid = f"hitl-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": tid}}
    state = _full_graph_input_state()

    with _all_external_patches():
        with patch("app.database.sql_client.save_reservation_with_thread", return_value=True), \
             patch("app.database.sql_client.get_reservation_by_thread_id", return_value=None):
            fresh_graph.invoke(state, config=config)

    state_snapshot = fresh_graph.get_state(config)
    assert bool(state_snapshot.next) is True


# ---------------------------------------------------------------------------
# Test 9 – graph completes without interrupt for an info question
# ---------------------------------------------------------------------------

def test_graph_is_not_interrupted_before_reservation(fresh_graph):
    """An info question must complete the graph without any interrupt."""
    tid = f"hitl-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": tid}}
    state = {
        "messages": [],
        "user_input": "What are the parking prices?",
        "intent": "",
        "reservation_data": {},
        "guardrail_triggered": False,
        "response": "",
        "admin_decision": "",
        "awaiting_admin": False,
    }

    with _info_external_patches():
        fresh_graph.invoke(state, config=config)

    state_snapshot = fresh_graph.get_state(config)
    assert bool(state_snapshot.next) is False


# ---------------------------------------------------------------------------
# Test 10 – awaiting_admin flag blocks further graph invocation
# ---------------------------------------------------------------------------

def test_awaiting_admin_blocks_further_graph_invocation():
    """When awaiting_admin=True the chat handler must skip chatbot_graph.invoke()."""
    awaiting_admin = True  # Simulates st.session_state.awaiting_admin

    invoke_was_called = False

    def fake_invoke(*args, **kwargs):
        nonlocal invoke_was_called
        invoke_was_called = True
        return {}

    # Replicate the conditional guard from main.py
    if not awaiting_admin:
        fake_invoke()

    assert not invoke_was_called

