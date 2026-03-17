import importlib
from types import SimpleNamespace
from datetime import datetime
from typing import cast
from unittest.mock import patch

import pytest

from app.chatbot.graph import record_reservation_node
from app.chatbot.graph import ChatState
from langchain_core.runnables import RunnableConfig


@pytest.fixture(autouse=True)
def temp_reservations_file(monkeypatch, tmp_path):
    temp_file = tmp_path / "test_reservations.txt"
    monkeypatch.setenv("RESERVATIONS_FILE_PATH", str(temp_file))
    monkeypatch.setattr("mcp_server.file_writer.RESERVATIONS_FILE_PATH", str(temp_file))
    return temp_file


def _sample_state() -> ChatState:
    return {
        "messages": [],
        "user_input": "",
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
        "admin_decision": "approved",
        "awaiting_admin": False,
    }


def _sample_config() -> RunnableConfig:
    return cast(RunnableConfig, {"configurable": {"thread_id": "thread-1234abcd"}})


def test_get_mcp_server_params_returns_correct_command(monkeypatch):
    monkeypatch.setenv("MCP_API_KEY", "test-key")
    monkeypatch.setenv("RESERVATIONS_FILE_PATH", "data/reservations.txt")

    import app.mcp_client.tools as mcp_tools

    mcp_tools = importlib.reload(mcp_tools)
    params = mcp_tools.get_mcp_server_params()

    assert params.command == "python"
    assert "-m" in params.args
    assert "mcp_server.server" in params.args


def test_get_mcp_server_params_includes_api_key(monkeypatch):
    monkeypatch.setenv("MCP_API_KEY", "my-api-key")

    import app.mcp_client.tools as mcp_tools

    mcp_tools = importlib.reload(mcp_tools)
    params = mcp_tools.get_mcp_server_params()

    assert params.env.get("MCP_API_KEY") == "my-api-key"


def test_write_reservation_via_mcp_success(monkeypatch):
    monkeypatch.setenv("MCP_API_KEY", "test-key")

    import app.mcp_client.tools as mcp_tools

    mcp_tools = importlib.reload(mcp_tools)

    call_record = {}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def initialize(self):
            return None

        async def call_tool(self, name, arguments):
            call_record["name"] = name
            call_record["arguments"] = arguments
            return SimpleNamespace(content=[SimpleNamespace(text="Reservation written successfully: ok")])

    class FakeStdioClient:
        async def __aenter__(self):
            return ("read_stream", "write_stream")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    with (
        patch("app.mcp_client.tools.stdio_client", return_value=FakeStdioClient()),
        patch("app.mcp_client.tools.ClientSession", return_value=FakeSession()),
    ):
        result = mcp_tools.write_reservation_via_mcp(
            name="John",
            surname="Doe",
            car_number="CAR-123",
            parking_id="parking_001",
            start_date="2026-04-01",
            end_date="2026-04-03",
            approval_time="2026-04-01T10:00:00",
        )

    assert "success" in result.lower()
    assert call_record["name"] == "write_parking_reservation"


def test_write_reservation_via_mcp_handles_failure(monkeypatch):
    monkeypatch.setenv("MCP_API_KEY", "test-key")

    import app.mcp_client.tools as mcp_tools

    mcp_tools = importlib.reload(mcp_tools)

    class FailingSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def initialize(self):
            return None

        async def call_tool(self, name, arguments):
            raise RuntimeError("boom")

    class FakeStdioClient:
        async def __aenter__(self):
            return ("read_stream", "write_stream")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    with (
        patch("app.mcp_client.tools.stdio_client", return_value=FakeStdioClient()),
        patch("app.mcp_client.tools.ClientSession", return_value=FailingSession()),
    ):
        result = mcp_tools.write_reservation_via_mcp(
            name="John",
            surname="Doe",
            car_number="CAR-123",
            parking_id="parking_001",
            start_date="2026-04-01",
            end_date="2026-04-03",
        )

    assert "error" in result.lower()


def test_write_reservation_via_mcp_sets_approval_time_if_none(monkeypatch):
    monkeypatch.setenv("MCP_API_KEY", "test-key")

    import app.mcp_client.tools as mcp_tools

    mcp_tools = importlib.reload(mcp_tools)

    call_record = {}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def initialize(self):
            return None

        async def call_tool(self, name, arguments):
            call_record["arguments"] = arguments
            return SimpleNamespace(content=[SimpleNamespace(text="Reservation written successfully: ok")])

    class FakeStdioClient:
        async def __aenter__(self):
            return ("read_stream", "write_stream")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    with (
        patch("app.mcp_client.tools.stdio_client", return_value=FakeStdioClient()),
        patch("app.mcp_client.tools.ClientSession", return_value=FakeSession()),
    ):
        mcp_tools.write_reservation_via_mcp(
            name="John",
            surname="Doe",
            car_number="CAR-123",
            parking_id="parking_001",
            start_date="2026-04-01",
            end_date="2026-04-03",
            approval_time=None,
        )

    approval_time = call_record["arguments"]["approval_time"]
    # Ensure a valid ISO datetime string is passed when no explicit value is given.
    assert isinstance(approval_time, str)
    assert "T" in approval_time
    datetime.fromisoformat(approval_time)


def test_record_reservation_node_calls_mcp():
    state = _sample_state()
    config = _sample_config()

    with (
        patch("app.database.sql_client.update_reservation_status", return_value=True),
        patch("app.chatbot.graph.write_reservation_via_mcp", return_value="Reservation written successfully: ok") as mock_mcp,
    ):
        result = record_reservation_node(state, config)

    mock_mcp.assert_called_once()
    assert result["response"]


def test_record_reservation_node_succeeds_even_if_mcp_fails():
    state = _sample_state()
    config = _sample_config()

    with (
        patch("app.database.sql_client.update_reservation_status", return_value=True),
        patch("app.chatbot.graph.write_reservation_via_mcp", side_effect=RuntimeError("mcp down")),
    ):
        result = record_reservation_node(state, config)

    assert result["response"]
    assert "saved in our system" in result["response"].lower()
