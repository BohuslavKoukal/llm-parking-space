from dotenv import load_dotenv

load_dotenv()

import asyncio
import logging
from datetime import datetime
from typing import Any

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from mcp_server.file_writer import get_file_stats, read_reservations, write_reservation
from mcp_server.security import validate_reservation_input, verify_api_key

logger = logging.getLogger(__name__)

app = Server("parking-reservation-server")


def _text_response(message: str) -> list[types.TextContent]:
    """Wrap plain text into MCP TextContent response payload."""
    return [types.TextContent(type="text", text=message)]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Dispatch MCP tool calls and return formatted text responses."""
    args = arguments or {}

    if name == "write_parking_reservation":
        api_key = str(args.get("api_key", ""))
        if not verify_api_key(api_key):
            return _text_response("Authentication failed: invalid API key")

        name_value = str(args.get("name", ""))
        surname = str(args.get("surname", ""))
        car_number = str(args.get("car_number", ""))
        parking_id = str(args.get("parking_id", ""))
        start_date = str(args.get("start_date", ""))
        end_date = str(args.get("end_date", ""))
        approval_time = str(args.get("approval_time") or datetime.now().isoformat())

        is_valid, reason = validate_reservation_input(
            name=name_value,
            surname=surname,
            car_number=car_number,
            start_date=start_date,
            end_date=end_date,
        )
        if not is_valid:
            return _text_response(f"Validation failed: {reason}")

        written_line = write_reservation(
            name=name_value,
            surname=surname,
            car_number=car_number,
            parking_id=parking_id,
            start_date=start_date,
            end_date=end_date,
            approval_time=approval_time,
        )
        return _text_response(f"Reservation written successfully: {written_line}")

    if name == "read_parking_reservations":
        api_key = str(args.get("api_key", ""))
        if not verify_api_key(api_key):
            return _text_response("Authentication failed: invalid API key")

        lines = read_reservations()
        if not lines:
            return _text_response("No reservations found.")
        return _text_response("\n".join(lines))

    if name == "get_reservations_file_stats":
        api_key = str(args.get("api_key", ""))
        if not verify_api_key(api_key):
            return _text_response("Authentication failed: invalid API key")

        stats = get_file_stats()
        formatted = (
            f"file_path: {stats['file_path']}\n"
            f"exists: {stats['exists']}\n"
            f"line_count: {stats['line_count']}\n"
            f"size_bytes: {stats['size_bytes']}\n"
            f"last_modified: {stats['last_modified']}"
        )
        return _text_response(formatted)

    raise ValueError(f"Unknown tool: {name}")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """Return the MCP tool registry exposed by this server."""
    return [
        types.Tool(
            name="write_parking_reservation",
            description="Write a confirmed parking reservation to the reservations file. Requires a valid API key.",
            inputSchema={
                "type": "object",
                "properties": {
                    "api_key": {
                        "type": "string",
                        "description": "MCP server API key for authentication",
                    },
                    "name": {
                        "type": "string",
                        "description": "Customer first name",
                    },
                    "surname": {
                        "type": "string",
                        "description": "Customer last name",
                    },
                    "car_number": {
                        "type": "string",
                        "description": "Car registration plate",
                    },
                    "parking_id": {
                        "type": "string",
                        "description": "Parking space identifier",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Reservation start date YYYY-MM-DD",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Reservation end date YYYY-MM-DD",
                    },
                    "approval_time": {
                        "type": "string",
                        "description": "Admin approval timestamp, defaults to current time if not provided",
                    },
                },
                "required": [
                    "api_key",
                    "name",
                    "surname",
                    "car_number",
                    "parking_id",
                    "start_date",
                    "end_date",
                ],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="read_parking_reservations",
            description="Read all confirmed reservations from the reservations file. Requires a valid API key.",
            inputSchema={
                "type": "object",
                "properties": {
                    "api_key": {
                        "type": "string",
                        "description": "MCP server API key for authentication",
                    }
                },
                "required": ["api_key"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="get_reservations_file_stats",
            description="Get statistics about the reservations file. Requires a valid API key.",
            inputSchema={
                "type": "object",
                "properties": {
                    "api_key": {
                        "type": "string",
                        "description": "MCP server API key for authentication",
                    }
                },
                "required": ["api_key"],
                "additionalProperties": False,
            },
        ),
    ]


async def main() -> None:
    """Start the MCP stdio server loop."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Parking Reservation MCP Server")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
