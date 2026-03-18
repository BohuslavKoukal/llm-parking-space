import hmac
import logging
import os
import re
from datetime import datetime

logger = logging.getLogger(__name__)

_mcp_api_key = os.getenv("MCP_API_KEY")
if not _mcp_api_key:
    raise RuntimeError("MCP_API_KEY environment variable is not set")
MCP_API_KEY: str = _mcp_api_key

_NAME_PATTERN = re.compile(r"^[A-Za-z\- ]+$")
_CAR_PATTERN = re.compile(r"^[A-Za-z0-9\-]+$")


def verify_api_key(provided_key: str) -> bool:
    """Validate provided MCP API key using timing-safe comparison."""
    provided = provided_key or ""
    is_valid = hmac.compare_digest(provided, MCP_API_KEY)
    if not is_valid:
        logger.warning("MCP API key mismatch detected")
    return is_valid


def validate_reservation_input(
    name: str,
    surname: str,
    car_number: str,
    start_date: str,
    end_date: str,
) -> tuple[bool, str]:
    """Validate reservation payload fields and return (is_valid, reason)."""
    if not name or not name.strip():
        return False, "Name must be non-empty"
    if len(name.strip()) > 100:
        return False, "Name must be at most 100 characters"
    if not _NAME_PATTERN.fullmatch(name.strip()):
        return False, "Name can contain only letters, spaces, and hyphens"

    if not surname or not surname.strip():
        return False, "Surname must be non-empty"
    if len(surname.strip()) > 100:
        return False, "Surname must be at most 100 characters"
    if not _NAME_PATTERN.fullmatch(surname.strip()):
        return False, "Surname can contain only letters, spaces, and hyphens"

    if not car_number or not car_number.strip():
        return False, "Car number must be non-empty"
    if len(car_number.strip()) > 20:
        return False, "Car number must be at most 20 characters"
    if not _CAR_PATTERN.fullmatch(car_number.strip()):
        return False, "Car number can contain only alphanumeric characters and hyphens"

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
    except ValueError:
        return False, "Start date must be in YYYY-MM-DD format"

    try:
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        return False, "End date must be in YYYY-MM-DD format"

    if start > end:
        return False, "Start date cannot be after end date"

    return True, ""
