import uuid
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from app.chatbot.graph import ChatState, get_thread_config
from app.chatbot.graph import build_graph
from app.database.sql_client import init_db, reset_db_connection
from app.database.models import Reservation
import app.database.sql_client as sql_client


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


@pytest.fixture
def integration_graph():
    """
    Fresh compiled graph with in-memory SqliteSaver checkpointer.
    Used for integration tests that need real graph execution.
    """
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        yield graph


@pytest.fixture
def integration_thread_config():
    """Fresh unique thread config for each integration test."""
    return get_thread_config(str(uuid.uuid4()))


@pytest.fixture
def test_db(monkeypatch):
    """
    Fresh in-memory SQLite database for integration tests.
    Initializes schema and seeds dynamic config data.
    Cleans up after test.
    """
    import os
    import time

    test_db_path = "test_integration.db"
    original_db_url = os.environ.get("DATABASE_URL", "sqlite:///./parking.db")

    # Remove any stale integration DB file from prior runs before creating a fresh test DB.
    for _attempt in range(5):
        try:
            if os.path.exists(test_db_path):
                os.remove(test_db_path)
            break
        except PermissionError:
            time.sleep(0.2)

    os.environ["DATABASE_URL"] = f"sqlite:///{test_db_path}"
    reset_db_connection()
    init_db()

    # Ensure per-test isolation even if a stale DB survived from a previous run.
    session = sql_client.SessionLocal()
    try:
        session.query(Reservation).delete()
        session.commit()
    finally:
        session.close()

    try:
        yield
    finally:
        close_all = getattr(sql_client.SessionLocal, "close_all", None)
        if callable(close_all):
            try:
                close_all()
            except Exception:
                pass

        bound_engine = getattr(sql_client.SessionLocal, "bind", None)
        if bound_engine is not None:
            try:
                bound_engine.dispose()
            except Exception:
                pass

        sql_client.get_engine().dispose()
        time.sleep(0.1)

        for _attempt in range(5):
            try:
                if os.path.exists(test_db_path):
                    os.remove(test_db_path)
                break
            except PermissionError:
                time.sleep(0.2)

        # Fallback for Windows file-handle timing: defer one more cleanup at process exit.
        if os.path.exists(test_db_path):
            import atexit

            def _cleanup_test_db_file() -> None:
                for _attempt in range(5):
                    try:
                        if os.path.exists(test_db_path):
                            os.remove(test_db_path)
                        break
                    except PermissionError:
                        time.sleep(0.2)

            atexit.register(_cleanup_test_db_file)

        os.environ["DATABASE_URL"] = original_db_url
        reset_db_connection()
