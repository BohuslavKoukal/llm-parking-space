"""Admin CLI for reviewing and processing pending parking reservations."""

import os
import sys
import logging

# Allow running from any directory: add project root to the Python import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from app.database.sql_client import get_pending_reservations
from app.chatbot.graph import chatbot_graph, get_thread_config
from langgraph.types import Command

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def display_reservations(reservations: list[dict]) -> None:
    """Print a numbered list of pending reservations."""
    print(f"\nFound {len(reservations)} pending reservation(s):\n")
    for i, r in enumerate(reservations, start=1):
        thread_display = f"{r['thread_id'][:8]}..." if r["thread_id"] else "N/A"
        print(f"[{i}] ID: {r['id']}")
        print(f"     Name:    {r['name']} {r['surname']}")
        print(f"     Parking: {r['parking_id']}")
        print(f"     Car:     {r['car_number']}")
        print(f"     Period:  {r['start_date']} to {r['end_date']}")
        print(f"     Thread:  {thread_display}")
        print(f"     Created: {r['created_at']}")
        print()


def select_reservation(reservations: list[dict]) -> dict:
    """Prompt the admin to select a reservation by number. Exit on 'q'."""
    while True:
        raw = input("Enter reservation number to review (or 'q' to quit): ").strip()
        if raw.lower() == "q":
            print("Exiting admin console.")
            sys.exit(0)
        try:
            idx = int(raw)
            if 1 <= idx <= len(reservations):
                return reservations[idx - 1]
            print(f"Please enter a number between 1 and {len(reservations)}.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")


def get_admin_decision() -> str:
    """Prompt for approve/reject and return the decision string."""
    while True:
        raw = input("Approve or reject this reservation? (a/r): ").strip()
        if raw.lower() == "a":
            return "approved"
        if raw.lower() == "r":
            return "rejected"
        print("Please enter 'a' to approve or 'r' to reject.")


def process_reservation(reservation: dict) -> None:
    """Display the reservation, collect admin decision, and resume the graph."""
    print("\n--- Reviewing Reservation ---")
    print(f"ID:      {reservation['id']}")
    print(f"Name:    {reservation['name']} {reservation['surname']}")
    print(f"Parking: {reservation['parking_id']}")
    print(f"Car:     {reservation['car_number']}")
    print(f"Period:  {reservation['start_date']} to {reservation['end_date']}")
    thread_id = reservation["thread_id"]
    print(f"Thread:  {thread_id[:8]}...")
    print()

    decision = get_admin_decision()

    thread_config = get_thread_config(thread_id)
    state_snapshot = chatbot_graph.get_state(thread_config)
    if not state_snapshot.next:
        print(
            "Error: This reservation thread is not in an interrupted state. "
            "It may have already been processed."
        )
        sys.exit(1)

    chatbot_graph.invoke(Command(resume=decision), config=thread_config)

    if decision == "approved":
        print("✅ Reservation APPROVED. User has been notified.")
    else:
        print("❌ Reservation REJECTED. User has been notified.")
    print(f"Thread {thread_id[:8]}... has been resumed successfully.")


def main() -> None:
    print("=== Parking Chatbot - Admin Review Console ===")

    while True:
        reservations = get_pending_reservations()
        if not reservations:
            print("No pending reservations found. Exiting.")
            sys.exit(0)

        display_reservations(reservations)
        reservation = select_reservation(reservations)
        process_reservation(reservation)

        raw = input("\nWould you like to review another reservation? (y/n): ").strip()
        if raw.lower() != "y":
            print("Goodbye.")
            sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAdmin console interrupted. Goodbye.")
        sys.exit(0)
    except Exception:
        logger.exception("Unexpected error in admin console")
        sys.exit(1)
