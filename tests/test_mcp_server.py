import asyncio
import importlib
import threading
from pathlib import Path


def _load_security_module(monkeypatch, api_key: str):
    monkeypatch.setenv("MCP_API_KEY", api_key)
    import mcp_server.security as security

    return importlib.reload(security)


def _load_file_writer_module(monkeypatch, file_path: Path):
    monkeypatch.setenv("RESERVATIONS_FILE_PATH", str(file_path))
    import mcp_server.file_writer as file_writer

    return importlib.reload(file_writer)


def test_verify_api_key_returns_true_for_correct_key(monkeypatch):
    security = _load_security_module(monkeypatch, "test_mcp_key")

    assert security.verify_api_key("test_mcp_key") is True


def test_verify_api_key_returns_false_for_wrong_key(monkeypatch):
    security = _load_security_module(monkeypatch, "test_mcp_key")

    assert security.verify_api_key("wrong_key") is False


def test_validate_reservation_input_valid(monkeypatch):
    security = _load_security_module(monkeypatch, "test_mcp_key")

    is_valid, reason = security.validate_reservation_input(
        name="John",
        surname="Doe-Smith",
        car_number="ABC-123",
        start_date="2026-04-01",
        end_date="2026-04-03",
    )

    assert is_valid is True
    assert reason == ""


def test_validate_reservation_input_invalid_date_format(monkeypatch):
    security = _load_security_module(monkeypatch, "test_mcp_key")

    is_valid, reason = security.validate_reservation_input(
        name="John",
        surname="Doe",
        car_number="ABC-123",
        start_date="not-a-date",
        end_date="2026-04-03",
    )

    assert is_valid is False
    assert reason


def test_validate_reservation_input_start_after_end(monkeypatch):
    security = _load_security_module(monkeypatch, "test_mcp_key")

    is_valid, reason = security.validate_reservation_input(
        name="John",
        surname="Doe",
        car_number="ABC-123",
        start_date="2026-04-10",
        end_date="2026-04-03",
    )

    assert is_valid is False
    assert reason


def test_write_reservation_creates_file(monkeypatch, tmp_path):
    reservations_file = tmp_path / "reservations.txt"
    file_writer = _load_file_writer_module(monkeypatch, reservations_file)

    written = file_writer.write_reservation(
        name="Name",
        surname="Surname",
        car_number="CAR123",
        parking_id="parking_001",
        start_date="2026-04-01",
        end_date="2026-04-03",
        approval_time="2026-04-01T10:00:00",
    )

    assert reservations_file.exists()
    lines = [line.strip() for line in reservations_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines == [written]


def test_write_reservation_format(monkeypatch, tmp_path):
    reservations_file = tmp_path / "reservations.txt"
    file_writer = _load_file_writer_module(monkeypatch, reservations_file)

    approval_time = "2026-04-01T10:00:00"
    file_writer.write_reservation(
        name="Name",
        surname="Surname",
        car_number="CAR123",
        parking_id="parking_001",
        start_date="2026-04-01",
        end_date="2026-04-03",
        approval_time=approval_time,
    )

    lines = [line.strip() for line in reservations_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines[0] == f"Name Surname | CAR123 | 2026-04-01 to 2026-04-03 | {approval_time}"


def test_read_reservations_returns_empty_list_if_no_file(monkeypatch, tmp_path):
    reservations_file = tmp_path / "missing" / "reservations.txt"
    file_writer = _load_file_writer_module(monkeypatch, reservations_file)

    if reservations_file.exists():
        reservations_file.unlink()

    assert file_writer.read_reservations() == []


def test_concurrent_writes_do_not_corrupt_file(monkeypatch, tmp_path):
    reservations_file = tmp_path / "reservations.txt"
    file_writer = _load_file_writer_module(monkeypatch, reservations_file)

    def _write_item(index: int):
        file_writer.write_reservation(
            name=f"Name{index}",
            surname="User",
            car_number=f"CAR-{index}",
            parking_id="parking_001",
            start_date="2026-04-01",
            end_date="2026-04-03",
            approval_time=f"2026-04-01T10:00:0{index}",
        )

    threads = [threading.Thread(target=_write_item, args=(i,)) for i in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = [line.strip() for line in reservations_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 5
    for line in lines:
        assert line.count("|") == 3
        assert " to " in line


def test_mcp_server_lists_three_tools(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_API_KEY", "test_mcp_key")
    monkeypatch.setenv("RESERVATIONS_FILE_PATH", str(tmp_path / "reservations.txt"))

    import mcp_server.server as server_module

    server_module = importlib.reload(server_module)
    tools = asyncio.run(server_module.list_tools())

    assert len(tools) == 3
    tool_names = {tool.name for tool in tools}
    assert tool_names == {
        "write_parking_reservation",
        "read_parking_reservations",
        "get_reservations_file_stats",
    }
