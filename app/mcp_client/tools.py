import asyncio
import logging
import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

load_dotenv()
logger = logging.getLogger(__name__)


def get_mcp_server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command="python",
        args=["-m", "mcp_server.server"],
        env={
            "MCP_API_KEY": os.getenv("MCP_API_KEY", ""),
            "RESERVATIONS_FILE_PATH": os.getenv("RESERVATIONS_FILE_PATH", "data/reservations.txt"),
        },
    )


async def call_write_reservation_tool(
    name: str,
    surname: str,
    car_number: str,
    parking_id: str,
    start_date: str,
    end_date: str,
    approval_time: str,
) -> str:
    api_key = os.getenv("MCP_API_KEY", "")
    if not api_key:
        logger.error("MCP_API_KEY is not set")
        return "Error: MCP_API_KEY is not set"

    try:
        server_params = get_mcp_server_params()
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "write_parking_reservation",
                    arguments={
                        "api_key": api_key,
                        "name": name,
                        "surname": surname,
                        "car_number": car_number,
                        "parking_id": parking_id,
                        "start_date": start_date,
                        "end_date": end_date,
                        "approval_time": approval_time,
                    },
                )

                text = ""
                if getattr(result, "content", None):
                    first = result.content[0]
                    text = getattr(first, "text", str(first))

                logger.info("MCP write_parking_reservation call completed")
                return text or "Error: Empty response from MCP tool"
    except Exception as exc:
        logger.exception("Failed calling MCP write_parking_reservation")
        return f"Error calling MCP write_parking_reservation: {exc}"


def write_reservation_via_mcp(
    name: str,
    surname: str,
    car_number: str,
    parking_id: str,
    start_date: str,
    end_date: str,
    approval_time: Optional[str] = None,
) -> str:
    approval_value = approval_time or datetime.now().isoformat()
    safe_name = (name[:1] + "***") if name else "***"
    logger.info("Calling MCP write tool for reservation owner: %s", safe_name)
    return asyncio.run(
        call_write_reservation_tool(
            name=name,
            surname=surname,
            car_number=car_number,
            parking_id=parking_id,
            start_date=start_date,
            end_date=end_date,
            approval_time=approval_value,
        )
    )


@tool(
    "write_parking_reservation",
    description=(
        "Write a confirmed parking reservation to the reservations file via the MCP server. "
        "Use this tool after an administrator has approved a reservation."
    ),
)
def write_reservation_tool(
    name: str,
    surname: str,
    car_number: str,
    parking_id: str,
    start_date: str,
    end_date: str,
    approval_time: Optional[str] = None,
) -> str:
    return write_reservation_via_mcp(
        name=name,
        surname=surname,
        car_number=car_number,
        parking_id=parking_id,
        start_date=start_date,
        end_date=end_date,
        approval_time=approval_time,
    )
