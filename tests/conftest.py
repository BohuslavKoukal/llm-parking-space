import uuid
import pytest

from app.chatbot.graph import ChatState, get_thread_config


@pytest.fixture
def base_state() -> ChatState:
    """Return a default empty chat state used as a clean baseline for tests."""
    return {
        "messages": [],
        "user_input": "",
        "intent": "",
        "reservation_data": {},
        "guardrail_triggered": False,
        "response": "",
        "admin_decision": "",
        "awaiting_admin": False,
    }


@pytest.fixture
def mid_reservation_state() -> ChatState:
    """Return a state where reservation flow has started with only parking_id collected."""
    return {
        "messages": [],
        "user_input": "",
        "intent": "",
        "reservation_data": {
            "parking_id": "parking_003",
        },
        "guardrail_triggered": False,
        "response": "",
        "admin_decision": "",
        "awaiting_admin": False,
    }


@pytest.fixture
def all_fields_collected_state() -> ChatState:
    """Return a state with all required reservation fields collected and confirmation input."""
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
def approved_state() -> ChatState:
    """State after admin has approved the reservation."""
    return {
        "messages": [],
        "user_input": "",
        "intent": "reservation",
        "reservation_data": {},
        "guardrail_triggered": False,
        "response": "Your reservation has been confirmed.",
        "admin_decision": "approved",
        "awaiting_admin": False,
    }


@pytest.fixture
def rejected_state() -> ChatState:
    """State after admin has rejected the reservation."""
    return {
        "messages": [],
        "user_input": "",
        "intent": "reservation",
        "reservation_data": {},
        "guardrail_triggered": False,
        "response": "Your reservation has been rejected.",
        "admin_decision": "rejected",
        "awaiting_admin": False,
    }


@pytest.fixture
def thread_id() -> str:
    """A fixed thread_id for use in tests."""
    return "test-thread-001"


@pytest.fixture
def thread_config(thread_id) -> dict:
    """A LangGraph thread config for use in tests."""
    return get_thread_config(thread_id)


@pytest.fixture
def fresh_thread_config() -> dict:
    """Return a unique LangGraph thread config on every call so tests are fully isolated."""
    return get_thread_config(str(uuid.uuid4()))
