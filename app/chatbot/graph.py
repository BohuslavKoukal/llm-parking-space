import logging
import os
import sqlite3
from threading import Lock
from typing import TypedDict
import re
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from app.chatbot.chains import (
    build_rag_chain,
    build_intent_chain,
    build_reservation_chain,
    build_guardrail_chain
)
from app.rag.weaviate_client import get_weaviate_client, get_retriever
from app.database.sql_client import get_all_parkings_summary, get_all_parking_ids_and_names
from app.guardrails.filters import is_sensitive, get_block_reason

logger = logging.getLogger(__name__)


# -- State schema -------------------------------------------------------------

class ChatState(TypedDict):
    messages: list[BaseMessage]
    user_input: str
    intent: str                  # "info" | "reservation" | "unknown"
    reservation_data: dict       # collected fields so far
    guardrail_triggered: bool
    response: str
    admin_decision: str          # "approved" | "rejected" | ""
    awaiting_admin: bool         # True while graph is paused for admin review


# -- Node implementations -----------------------------------------------------

REQUIRED_FIELDS = ["parking_id", "name", "surname", "car_number", "start_date", "end_date"]

def is_reservation_in_progress(state: ChatState) -> bool:
    """
    Returns True if a reservation is actively being collected or confirmed.
    Covers two phases:
    - Data collection: some fields collected but not all
    - Awaiting confirmation: all fields collected but not yet confirmed
    """
    collected = state.get("reservation_data", {})
    if not collected:
        return False
    already_confirmed = collected.get("confirmed") is not None
    return not already_confirmed

def guardrail_node(state: ChatState) -> ChatState:
    if is_reservation_in_progress(state):
        logger.info("Guardrail skipped: reservation in progress")
        return {**state, "guardrail_triggered": False}

    # Layer 1: Presidio + regex (local, fast, no API call)
    if is_sensitive(state["user_input"]):
        reason = get_block_reason(state["user_input"])
        logger.warning("Guardrail triggered by Presidio layer (%s) for input: %s", reason, state["user_input"][:50])
        return {**state, "guardrail_triggered": True}

    # Layer 2: LLM guardrail (only if Presidio layer passes)
    chain = build_guardrail_chain()
    result = chain.invoke({"user_input": state["user_input"]}).strip().lower()
    triggered = result == "blocked"
    if triggered:
        logger.warning("Guardrail triggered by LLM layer for input: %s", state["user_input"][:50])
    return {**state, "guardrail_triggered": triggered}

def intent_node(state: ChatState) -> ChatState:
    if state["guardrail_triggered"]:
        return {**state, "intent": "unknown"}

    if is_reservation_in_progress(state):
        logger.info("Intent forced to reservation: reservation in progress")
        return {**state, "intent": "reservation"}

    chain = build_intent_chain()
    raw_intent = chain.invoke({"user_input": state["user_input"]})

    # Normalize common LLM formatting variants like quotes, punctuation, or extra words.
    normalized = raw_intent.strip().lower().strip(" \t\n\r\"'`.,:;!?")
    if normalized in ("info", "reservation", "unknown"):
        intent = normalized
    else:
        match = re.search(r"\b(info|reservation|unknown)\b", raw_intent.lower())
        intent = match.group(1) if match else "unknown"

    # Deterministic fallback for obvious reservation wording if model output is malformed.
    if intent == "unknown":
        user_text = state["user_input"].lower()
        if any(word in user_text for word in ["reservation", "reserve", "book", "booking"]):
            intent = "reservation"

    logger.info("Intent raw output: %s", raw_intent)
    logger.info(f"Intent classified as: {intent}")
    return {**state, "intent": intent}



def rag_node(state: ChatState) -> ChatState:
    client = get_weaviate_client()
    try:
        retriever = get_retriever(client, k=20)
        rag_chain = build_rag_chain(retriever)

        # Dynamically load all parking IDs from SQL — no hardcoding
        all_parking_ids = get_all_parking_ids_and_names()
        dynamic_summary = get_all_parkings_summary()

        enriched_input = (
            f"{state['user_input']}\n\n"
            f"[Known parking IDs: {', '.join(all_parking_ids)}]\n"
            f"[Dynamic data per parking: {dynamic_summary}]"
        )
        response = rag_chain.invoke(enriched_input)
    finally:
        client.close()

    return {**state, "response": response}


def extract_reservation_fields(user_input: str, collected: dict, missing: list) -> dict:
    """
    Use GPT-4o to extract the next reservation field from user input.
    Returns updated collected dict.
    """
    from app.chatbot.chains import get_llm
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    import json

    if not missing:
        return collected

    next_field = missing[0]

    extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""Extract the value for the field '{next_field}' from the user message.
Return ONLY a valid JSON object like: {{{{"{next_field}": "extracted_value"}}}}
If the value cannot be extracted, return: {{{{"{next_field}": null}}}}

Field descriptions:
- parking_id: parking identifier like parking_001 to parking_005
- name: first name of the person
- surname: last name of the person
- car_number: car registration plate number
- start_date: date in YYYY-MM-DD format
- end_date: date in YYYY-MM-DD format
"""),
    ("human", "{user_input}")
])

    chain = extraction_prompt | get_llm(temperature=0.0) | StrOutputParser()
    result = chain.invoke({"user_input": user_input})

    try:
        # Strip markdown code fences if present
        clean = result.strip().replace("```json", "").replace("```", "").strip()
        extracted = json.loads(clean)
        value = extracted.get(next_field)
        if value:
            updated = collected.copy()
            updated[next_field] = value
            return updated
    except Exception as e:
        logger.warning(f"Field extraction failed: {e}, raw: {result}")

    return collected


def reservation_node(state: ChatState) -> ChatState:
    collected = state.get("reservation_data", {})
    missing = [f for f in REQUIRED_FIELDS if not collected.get(f)]

    # Phase 1: still collecting fields
    if missing:
        collected = extract_reservation_fields(state["user_input"], collected, missing)
        missing = [f for f in REQUIRED_FIELDS if not collected.get(f)]
        chain = build_reservation_chain()
        response = chain.invoke({
            "user_input": state["user_input"],
            "collected_data": collected,
            "missing_fields": missing
        })
        return {**state, "reservation_data": collected, "response": response}

    # Phase 2: all fields collected, awaiting confirmation
    user_confirmation = state["user_input"].strip().lower()
    # Tokenize to avoid substring matches like "yesterday" (containing "yes")
    confirmation_tokens = set(re.findall(r"\b\w+\b", user_confirmation))
    positive_confirmations = {"yes", "confirm", "correct", "ok", "sure"}
    negative_confirmations = {"no", "cancel", "change", "wrong"}

    if confirmation_tokens & positive_confirmations:
        # Mark as confirmed and flag for admin review.
        # DB save happens in submit_to_admin_node once the graph resumes.
        collected["confirmed"] = "yes"
        response = (
            f"Your reservation has been submitted for administrator review!\n\n"
            f"**Summary:**\n"
            f"- Parking: {collected['parking_id']}\n"
            f"- Name: {collected['name']} {collected['surname']}\n"
            f"- Car: {collected['car_number']}\n"
            f"- Period: {collected['start_date']} to {collected['end_date']}\n\n"
            f"Your reservation is now **pending administrator confirmation**. "
            f"You will be notified once it is approved."
        )
        return {**state, "reservation_data": collected, "response": response, "awaiting_admin": True}

    elif confirmation_tokens & negative_confirmations:
        # Reset and start over
        response = "No problem! Let's start over. Which parking space would you like to reserve?"
        return {**state, "reservation_data": {}, "response": response}

    else:
        # Unclear confirmation response
        chain = build_reservation_chain()
        response = chain.invoke({
            "user_input": state["user_input"],
            "collected_data": collected,
            "missing_fields": []
        })
        return {**state, "reservation_data": collected, "response": response}


def save_reservation(data: dict) -> bool:
    """Save confirmed reservation to SQL database with status pending."""
    from app.database.sql_client import SessionLocal
    from app.database.models import Reservation
    from datetime import date

    session = SessionLocal()
    try:
        reservation = Reservation(
            parking_id=data["parking_id"],
            name=data["name"],
            surname=data["surname"],
            car_number=data["car_number"],
            start_date=date.fromisoformat(data["start_date"]),
            end_date=date.fromisoformat(data["end_date"]),
            status="pending"
        )
        session.add(reservation)
        session.commit()
        logger.info(f"Reservation saved for {data['name']} {data['surname']}")
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to save reservation: {e}")
        return False
    finally:
        session.close()


def submit_to_admin_node(state: ChatState, config: RunnableConfig) -> ChatState:
    """
    Save reservation to DB then pause execution for admin review via interrupt().

    On resume (when admin calls invoke(Command(resume=decision))), the node
    re-runs from the top, so the DB save is idempotent — guarded by an existence
    check to prevent duplicate inserts.
    """
    from app.database.sql_client import save_reservation_with_thread, get_reservation_by_thread_id

    thread_id = config["configurable"]["thread_id"]
    collected = state.get("reservation_data", {})

    # Idempotent save: only insert if no reservation exists for this thread_id yet.
    existing = get_reservation_by_thread_id(thread_id)
    if not existing:
        success = save_reservation_with_thread(collected, thread_id)
        if not success:
            logger.error("Failed to save reservation for thread_id=%s", thread_id)

    # Pause graph execution; resumes when admin sends Command(resume=decision).
    decision = interrupt({
        "type": "admin_review_required",
        "reservation": collected,
        "thread_id": thread_id,
        "message": (
            f"Reservation from {collected.get('name', '')} {collected.get('surname', '')} "
            f"for {collected.get('parking_id', '')} "
            f"({collected.get('start_date', '')} to {collected.get('end_date', '')})"
        ),
    })

    logger.info("Admin decision received: %s for thread_id=%s", decision, thread_id)
    return {**state, "admin_decision": decision, "awaiting_admin": False}


def record_reservation_node(state: ChatState, config: RunnableConfig) -> ChatState:
    """Update reservation status to 'confirmed' after admin approval."""
    from app.database.sql_client import update_reservation_status

    thread_id = config["configurable"]["thread_id"]
    update_reservation_status(thread_id, "confirmed")
    logger.info("Reservation confirmed for thread_id=%s", thread_id)

    response = (
        "Your reservation has been **confirmed** by the administrator!\n\n"
        "We look forward to seeing you. Enjoy your parking experience."
    )
    return {**state, "reservation_data": {}, "response": response}


def notify_rejection_node(state: ChatState, config: RunnableConfig) -> ChatState:
    """Update reservation status to 'rejected' after admin rejection."""
    from app.database.sql_client import update_reservation_status

    thread_id = config["configurable"]["thread_id"]
    update_reservation_status(thread_id, "rejected")
    logger.info("Reservation rejected for thread_id=%s", thread_id)

    response = (
        "We're sorry, your reservation request has been rejected by the administrator. "
        "Please try again or contact us for more information."
    )
    return {**state, "reservation_data": {}, "response": response}


def response_node(state: ChatState) -> ChatState:
    """
    Finalize the response. If guardrail was triggered, return a safe refusal message.
    Otherwise pass through the response from rag_node or reservation_node.
    """
    if state["guardrail_triggered"]:
        response = (
            "I'm sorry, I can't help with that request. "
            "I'm only able to provide information about our parking spaces "
            "and assist with reservations."
        )
        # Append the guarded turn to message history so the UI can display it
        messages = state.get("messages", [])
        messages = messages + [
            HumanMessage(content=state.get("user_input", "")),
            AIMessage(content=response),
        ]
        return {**state, "response": response, "messages": messages}

    # Append response to message history
    messages = state.get("messages", [])
    messages = messages + [
        HumanMessage(content=state["user_input"]),
        AIMessage(content=state["response"])
    ]
    return {**state, "messages": messages}


def unknown_node(state: ChatState) -> ChatState:
    """Handle unknown or unclassified intents with a helpful fallback message."""
    response = (
        "I'm not sure I understood that. I can help you with:\n"
        "- Information about our 5 parking spaces (prices, hours, location, availability)\n"
        "- Making a parking reservation\n\n"
        "What would you like to know?"
    )
    return {**state, "response": response}


# -- Routing logic ------------------------------------------------------------

def route_after_intent(state: ChatState) -> str:
    """Route to the correct node based on classified intent."""
    if state["guardrail_triggered"]:
        return "response_node"
    intent = state.get("intent", "unknown")
    if intent == "info":
        return "rag_node"
    elif intent == "reservation":
        return "reservation_node"
    else:
        return "unknown_node"


def route_after_reservation(state: ChatState) -> str:
    """Route from reservation_node: if awaiting admin review go to submit_to_admin, else finish."""
    if state.get("awaiting_admin", False):
        return "submit_to_admin"
    return "response_node"


def route_after_admin_decision(state: ChatState) -> str:
    """Route after interrupt resumes: approved goes to record_reservation, anything else to notify_rejection."""
    if state.get("admin_decision", "") == "approved":
        return "record_reservation"
    return "notify_rejection"


# -- Graph assembly -----------------------------------------------------------

def get_thread_config(thread_id: str) -> dict:
    """
    Return a LangGraph config dict for the given thread_id.

    This dict is passed as the `config` argument to every graph invocation
    so that the SqliteSaver checkpointer can persist and resume the correct
    conversation state across turns.

    Usage:
        result = chatbot_graph.invoke(state, config=get_thread_config(thread_id))
    """
    return {"configurable": {"thread_id": thread_id}}


def build_graph(conn_string: str | None = None):
    """
    Assemble and compile the LangGraph chatbot graph.

    Flow:
    START -> guardrail_node -> intent_node -> (conditional routing)
         |- rag_node -> response_node -> END
         |- reservation_node -> (route_after_reservation)
         |    |- response_node -> END
         |    '- submit_to_admin [interrupt] -> (route_after_admin_decision)
         |         |- record_reservation -> response_node -> END
         |         '- notify_rejection   -> response_node -> END
         '- unknown_node -> response_node -> END

    Args:
        conn_string: Optional SQLite connection string. Defaults to the
            CHECKPOINT_DB_URL env var (or "checkpoints.db"). Pass ":memory:"
            in tests to get a fresh, isolated in-memory checkpointer.
    """
    checkpoint_db_url = conn_string if conn_string is not None else os.getenv("CHECKPOINT_DB_URL", "checkpoints.db")
    conn = sqlite3.connect(checkpoint_db_url, check_same_thread=False)
    saver = SqliteSaver(conn)

    graph = StateGraph(ChatState)

    # Add nodes
    graph.add_node("guardrail_node", guardrail_node)
    graph.add_node("intent_node", intent_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("reservation_node", reservation_node)
    graph.add_node("unknown_node", unknown_node)
    graph.add_node("response_node", response_node)
    graph.add_node("submit_to_admin", submit_to_admin_node)
    graph.add_node("record_reservation", record_reservation_node)
    graph.add_node("notify_rejection", notify_rejection_node)

    # Add edges
    graph.add_edge(START, "guardrail_node")
    graph.add_edge("guardrail_node", "intent_node")
    graph.add_conditional_edges("intent_node", route_after_intent, {
        "rag_node": "rag_node",
        "reservation_node": "reservation_node",
        "response_node": "response_node",
        "unknown_node": "unknown_node"
    })
    graph.add_edge("rag_node", "response_node")
    graph.add_conditional_edges("reservation_node", route_after_reservation, {
        "submit_to_admin": "submit_to_admin",
        "response_node": "response_node",
    })
    graph.add_conditional_edges("submit_to_admin", route_after_admin_decision, {
        "record_reservation": "record_reservation",
        "notify_rejection": "notify_rejection",
    })
    graph.add_edge("record_reservation", "response_node")
    graph.add_edge("notify_rejection", "response_node")
    graph.add_edge("unknown_node", "response_node")
    graph.add_edge("response_node", END)

    compiled = graph.compile(checkpointer=saver)
    # Anchor the connection to the compiled graph so it is not garbage-collected
    # as long as the graph is alive, keeping the underlying SQLite connection open.
    compiled._sqlite_conn = conn
    return compiled, saver


# Compiled graph and checkpointer instances - imported by main.py and other modules
chatbot_graph, checkpointer = build_graph()
