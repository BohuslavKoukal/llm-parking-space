from dotenv import load_dotenv

load_dotenv()

import logging
import os
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

RESERVATIONS_FILE_PATH: str = os.getenv("RESERVATIONS_FILE_PATH", "data/reservations.txt")
_reservations_path = Path(RESERVATIONS_FILE_PATH)
_reservations_path.parent.mkdir(parents=True, exist_ok=True)

_write_lock = threading.Lock()


def write_reservation(
    name: str,
    surname: str,
    car_number: str,
    parking_id: str,
    start_date: str,
    end_date: str,
    approval_time: str,
) -> str:
    formatted_line = (
        f"{name} {surname} | {car_number} | {start_date} to {end_date} | {approval_time}"
    )

    with _write_lock:
        file_path = Path(RESERVATIONS_FILE_PATH)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("a", encoding="utf-8") as file_handle:
            file_handle.write(f"{formatted_line}\n")

    logger.info(
        "Reservation written to file",
        extra={
            "parking_id": parking_id,
            "file_path": RESERVATIONS_FILE_PATH,
            "entry": formatted_line,
        },
    )
    return formatted_line


def read_reservations() -> list[str]:
    file_path = Path(RESERVATIONS_FILE_PATH)
    if not file_path.exists():
        return []

    with file_path.open("r", encoding="utf-8") as file_handle:
        return [line.strip() for line in file_handle.readlines() if line.strip()]


def get_file_stats() -> dict:
    file_path = Path(RESERVATIONS_FILE_PATH)
    exists = file_path.exists()

    if not exists:
        return {
            "file_path": str(file_path),
            "exists": False,
            "line_count": 0,
            "size_bytes": 0,
            "last_modified": None,
        }

    stat_result = file_path.stat()
    with file_path.open("r", encoding="utf-8") as file_handle:
        line_count = len([line for line in file_handle.readlines() if line.strip()])

    return {
        "file_path": str(file_path),
        "exists": True,
        "line_count": line_count,
        "size_bytes": stat_result.st_size,
        "last_modified": datetime.fromtimestamp(stat_result.st_mtime).isoformat(),
    }
