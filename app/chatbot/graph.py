"""LangGraph orchestration for the parking assistant.

Flow overview:
1. Guardrail checks user input safety (except during active reservation collection).
2. Intent classification routes to information retrieval, reservation flow, or fallback.
3. Reservation flow collects required fields, asks for user confirmation, then pauses
   for human-in-the-loop administrator review.
4. After admin resume, reservation is finalized as approved/rejected in SQL and,
   on approval, recorded via MCP file-writing integration.
5. A final response node appends AI/user messages to persistent conversation history.
"""

import logging
import os
import re
import sqlite3
from typing import Any, cast, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.chatbot.chains import (
    build_guardrail_chain,
    build_intent_chain,
    build_rag_chain,
    build_reservation_chain,
)
from app.database.sql_client import get_all_parking_ids_and_names, get_all_parkings_summary
from app.guardrails.filters import get_block_reason, is_sensitive
from app.mcp_client.tools import write_reservation_via_mcp
from app.rag.weaviate_client import get_retriever, get_weaviate_client

logger = logging.getLogger(__name__)


# -- State schema -------------------------------------------------------------

REQUIRED_FIELDS = [
    "parking_id",
    "name",
    "surname",
    "car_number",
    "start_date",
    "end_date",
]
VALID_INTENTS: frozenset[str] = frozenset({"info", "reservation", "unknown"})
APPROVAL_KEYWORDS: frozenset[str] = frozenset({"yes", "confirm", "correct", "ok", "sure"})
REJECTION_KEYWORDS: frozenset[str] = frozenset({"no", "cancel", "change", "wrong"})


class ChatState(TypedDict):
    """Graph state carried across nodes.

    Field semantics and sensible defaults used by caller initialization:
    - messages: conversation history as LangChain message objects. Default: [].
    - user_input: current user message being processed. Default: "".
    - intent: classified intent - "info", "reservation", or "unknown". Default: "" before classification.
    - reservation_data: collected reservation fields keyed by parking_id, name, surname,
      car_number, start_date, end_date, confirmed. Default: {}.
    - guardrail_triggered: True if input was blocked by guardrails. Default: False.
    - response: final response string to show the user. Default: "".
    - admin_decision: "approved", "rejected", "pending", or "". Default: "".
    - awaiting_admin: True when graph is interrupted at submit_to_admin. Default: False.
    """

    messages: list[BaseMessage]
    user_input: str
    intent: str
    reservation_data: dict[str, str]
    guardrail_triggered: bool
    response: str
    admin_decision: str
    awaiting_admin: bool


# -- Node implementations -----------------------------------------------------


def is_reservation_in_progress(state: ChatState) -> bool:
    """Return whether reservation flow is active.

    Covers two phases:
    1. Collection phase: some required fields are still missing.
    2. Confirmation phase: all fields exist but explicit confirmation is not final yet.
    """
    collected = state.get("reservation_data", {})
    if not collected:
        return False
    already_confirmed = collected.get("confirmed") is not None
    return not already_confirmed


def guardrail_node(state: ChatState) -> ChatState:
    """Run safety checks on user input.

    Reads: user_input, reservation_data.
    Updates: guardrail_triggered.
    Side effects: optional LLM call via guardrail chain.
    """
    try:
        if is_reservation_in_progress(state):
            logger.info("Guardrail skipped: reservation in progress")
            return {**state, "guardrail_triggered": False}

        # Layer 1: Presidio/regex local filters.
        if is_sensitive(state.get("user_input", "")):
            reason = get_block_reason(state.get("user_input", ""))
            logger.warning(
                "Guardrail triggered by Presidio layer (%s) for input: %s",
                reason,
                state.get("user_input", "")[:50],
            )
            return {**state, "guardrail_triggered": True}

        # Layer 2: LLM guardrail.
        chain = build_guardrail_chain()
        result = chain.invoke({"user_input": state.get("user_input", "")}).strip().lower()
        triggered = result == "blocked"
        if triggered:
            logger.warning("Guardrail triggered by LLM layer for input: %s", state.get("user_input", "")[:50])
        return {**state, "guardrail_triggered": triggered}
    except Exception as exc:
        logger.error("guardrail_node failed: %s", exc)
        return {**state, "guardrail_triggered": True}


def intent_node(state: ChatState) -> ChatState:
    """Classify user intent unless flow should remain in reservation mode.

    Reads: guardrail_triggered, reservation_data, user_input.
    Updates: intent.
    Side effects: LLM call via intent chain.
    """
    try:
        if state.get("guardrail_triggered", False):
            return {**state, "intent": "unknown"}

        if is_reservation_in_progress(state):
            logger.info("Intent forced to reservation: reservation in progress")
            return {**state, "intent": "reservation"}

        chain = build_intent_chain()
        raw_intent = chain.invoke({"user_input": state.get("user_input", "")})

        normalized = raw_intent.strip().lower().strip(" \t\n\r\"'`.,:;!?")
        if normalized in VALID_INTENTS:
            intent = normalized
        else:
            match = re.search(r"\b(info|reservation|unknown)\b", raw_intent.lower())
            intent = match.group(1) if match else "unknown"

        if intent == "unknown":
            user_text = state.get("user_input", "").lower()
            if any(word in user_text for word in ["reservation", "reserve", "book", "booking"]):
                intent = "reservation"

        logger.info("Intent raw output: %s", raw_intent)
        logger.info("Intent classified as: %s", intent)
        return {**state, "intent": intent}
    except Exception as exc:
        logger.error("intent_node failed: %s", exc)
        return {**state, "intent": "unknown"}


def rag_node(state: ChatState) -> ChatState:
    """Answer informational queries through retrieval-augmented generation.

    Reads: user_input.
    Updates: response.
    Side effects: Weaviate retrieval and LLM RAG invocation.
    """
    client = None
    try:
        client = get_weaviate_client()
        retriever = get_retriever(client, k=20)
        rag_chain = build_rag_chain(retriever)

        all_parking_ids = get_all_parking_ids_and_names()
        dynamic_summary = get_all_parkings_summary()

        enriched_input = (
            f"{state.get('user_input', '')}\n\n"
            f"[Known parking IDs: {', '.join(all_parking_ids)}]\n"
            f"[Dynamic data per parking: {dynamic_summary}]"
        )
        response = rag_chain.invoke(enriched_input)
        return {**state, "response": response}
    except Exception as exc:
        logger.error("rag_node failed: %s", exc)
        return {
            **state,
            "response": "I ran into an issue while looking up parking information. Please try again.",
        }
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as close_exc:
                logger.error("rag_node failed to close Weaviate client: %s", close_exc)


def extract_reservation_fields(user_input: str, collected: dict[str, str], missing: list[str]) -> dict[str, str]:
    """Extract the next missing reservation field from user input via LLM."""
    from app.chatbot.chains import get_llm
    import json

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    if not missing:
        return collected

    next_field = missing[0]

    extraction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""Extract the value for the field '{next_field}' from the user message.
Return ONLY a valid JSON object like: {{{{"{next_field}": "extracted_value"}}}}
If the value cannot be extracted, return: {{{{"{next_field}": null}}}}

Field descriptions:
- parking_id: parking identifier like parking_001 to parking_005
- name: first name of the person
- surname: last name of the person
- car_number: car registration plate number
- start_date: date in YYYY-MM-DD format
- end_date: date in YYYY-MM-DD format
""",
            ),
            ("human", "{user_input}"),
        ]
    )

    chain = extraction_prompt | get_llm(temperature=0.0) | StrOutputParser()
    result = chain.invoke({"user_input": user_input})

    try:
        clean = result.strip().replace("```json", "").replace("```", "").strip()
        extracted = json.loads(clean)
        value = extracted.get(next_field)
        if value:
            updated = collected.copy()
            updated[next_field] = value
            return updated
    except Exception as exc:
        logger.warning("Field extraction failed: %s, raw: %s", exc, result)

    return collected


def reservation_node(state: ChatState) -> ChatState:
    """Collect reservation details and handle user confirmation.

    Reads: user_input, reservation_data.
    Updates: reservation_data, response, awaiting_admin.
    Side effects: LLM calls for extraction and reservation assistant messaging.
    """
    try:
        collected = state.get("reservation_data", {})
        missing = [field for field in REQUIRED_FIELDS if not collected.get(field)]

        # Phase 1: still collecting fields.
        if missing:
            collected = extract_reservation_fields(state.get("user_input", ""), collected, missing)
            missing = [field for field in REQUIRED_FIELDS if not collected.get(field)]
            chain = build_reservation_chain()
            response = chain.invoke(
                {
                    "user_input": state.get("user_input", ""),
                    "collected_data": collected,
                    "missing_fields": missing,
                }
            )
            return {**state, "reservation_data": collected, "response": response}

        # Phase 2: all fields collected, awaiting explicit yes/no confirmation.
        user_confirmation = state.get("user_input", "").strip().lower()
        confirmation_tokens = set(re.findall(r"\b\w+\b", user_confirmation))

        if confirmation_tokens & APPROVAL_KEYWORDS:
            collected["confirmed"] = "yes"
            response = (
                "Your reservation has been submitted for administrator review!\n\n"
                "**Summary:**\n"
                f"- Parking: {collected['parking_id']}\n"
                f"- Name: {collected['name']} {collected['surname']}\n"
                f"- Car: {collected['car_number']}\n"
                f"- Period: {collected['start_date']} to {collected['end_date']}\n\n"
                "Your reservation is now **pending administrator confirmation**. "
                "You will be notified once it is approved."
            )
            return {**state, "reservation_data": collected, "response": response, "awaiting_admin": True}

        if confirmation_tokens & REJECTION_KEYWORDS:
            response = "No problem! Let's start over. Which parking space would you like to reserve?"
            return {**state, "reservation_data": {}, "response": response}

        chain = build_reservation_chain()
        response = chain.invoke(
            {
                "user_input": state.get("user_input", ""),
                "collected_data": collected,
                "missing_fields": [],
            }
        )
        return {**state, "reservation_data": collected, "response": response}
    except Exception as exc:
        logger.error("reservation_node failed: %s", exc)
        return {
            **state,
            "response": "I hit an issue while processing your reservation details. Please try again.",
        }


def save_reservation(data: dict) -> bool:
    """Save confirmed reservation to SQL database with status pending.

    Deprecated: use app.database.sql_client.save_reservation_with_thread instead.
    Kept only for backwards-compatible test patching.
    """
    import warnings
    from datetime import date

    from app.database.models import Reservation
    from app.database.sql_client import SessionLocal

    warnings.warn(
        "save_reservation() is deprecated; use save_reservation_with_thread() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    session = SessionLocal()
    try:
        reservation = Reservation(
            parking_id=data["parking_id"],
            name=data["name"],
            surname=data["surname"],
            car_number=data["car_number"],
            start_date=date.fromisoformat(data["start_date"]),
            end_date=date.fromisoformat(data["end_date"]),
            status="pending",
        )
        session.add(reservation)
        session.commit()
        logger.info("Reservation saved for %s %s", data["name"], data["surname"])
        return True
    except Exception as exc:
        session.rollback()
        logger.error("Failed to save reservation: %s", exc)
        return False
    finally:
        session.close()


def submit_to_admin(state: ChatState, config: RunnableConfig) -> ChatState:
    """Persist pending reservation and pause execution for administrator decision.

    Reads: reservation_data, config.configurable.thread_id.
    Updates: admin_decision, awaiting_admin.
    Side effects: SQL read/write and LangGraph interrupt pause.
    """
    from app.database.sql_client import get_reservation_by_thread_id, save_reservation_with_thread

    thread_id = str((config.get("configurable") or {}).get("thread_id", ""))
    if not thread_id:
        logger.error("submit_to_admin missing thread_id in RunnableConfig")
        return {
            **state,
            "awaiting_admin": False,
            "admin_decision": "",
            "response": "We could not submit your reservation for admin review. Please try again.",
        }

    collected = state.get("reservation_data", {})
    try:
        existing = get_reservation_by_thread_id(thread_id)
        if not existing:
            success = save_reservation_with_thread(cast(dict[str, Any], collected), thread_id)
            if not success:
                logger.error("Failed to save reservation for thread_id=%s", thread_id)
    except Exception as exc:
        logger.error("submit_to_admin failed before interrupt: %s", exc)
        return {
            **state,
            "awaiting_admin": False,
            "admin_decision": "",
            "response": "We could not submit your reservation for admin review. Please try again.",
        }

    logger.info("Graph interrupted for admin review. Thread: %s", thread_id[:8])
    decision = interrupt(
        {
            "type": "admin_review_required",
            "reservation": collected,
            "thread_id": thread_id,
            "message": (
                f"Reservation from {collected.get('name', '')} {collected.get('surname', '')} "
                f"for {collected.get('parking_id', '')} "
                f"({collected.get('start_date', '')} to {collected.get('end_date', '')})"
            ),
        }
    )

    logger.info("Admin decision received: %s for thread_id=%s", decision, thread_id)
    return {**state, "admin_decision": decision, "awaiting_admin": False}


def submit_to_admin_node(state: ChatState, config: RunnableConfig) -> ChatState:
    """Backward-compatible alias for submit_to_admin."""
    return submit_to_admin(state, config)


def record_reservation_node(state: ChatState, config: RunnableConfig) -> ChatState:
    """Finalize approved reservation and record it via MCP.

    Reads: config.configurable.thread_id.
    Updates: reservation_data, response.
    Side effects: SQL status update/read and MCP tool call to append reservation file.
    """
    from app.database.sql_client import get_reservation_by_thread_id, update_reservation_status

    try:
        thread_id = str((config.get("configurable") or {}).get("thread_id", ""))
        if not thread_id:
            raise ValueError("Missing thread_id in RunnableConfig")
        result = update_reservation_status(thread_id, "confirmed")

        if not result:
            logger.error("Failed to confirm reservation for thread_id=%s", thread_id)
            response = (
                "Your reservation was approved but we encountered an error confirming it. "
                f"Please contact the administrator with your booking reference: {thread_id[:8]}"
            )
            return {**state, "response": response}

        reservation = get_reservation_by_thread_id(thread_id)
        required_fields = ["name", "surname", "car_number", "parking_id", "start_date", "end_date"]
        missing_fields = [field for field in required_fields if not reservation or not reservation.get(field)]

        if missing_fields:
            logger.error(
                "Canonical reservation data missing for thread_id=%s. Missing: %s",
                thread_id,
                ", ".join(missing_fields),
            )
            response = (
                "Your reservation is confirmed, but we encountered an internal data issue while finalizing "
                "file recording. Please contact the administrator with your booking reference: "
                f"{thread_id[:8]}"
            )
            return {**state, "reservation_data": {}, "response": response}

        response = "Your reservation has been confirmed."
        assert reservation is not None

        try:
            mcp_result = write_reservation_via_mcp(
                name=str(reservation["name"]),
                surname=str(reservation["surname"]),
                car_number=str(reservation["car_number"]),
                parking_id=str(reservation["parking_id"]),
                start_date=str(reservation["start_date"]),
                end_date=str(reservation["end_date"]),
                approval_time=None,
            )
            logger.info("MCP write result for thread_id=%s: %s", thread_id, mcp_result)
            if not mcp_result.lower().startswith("reservation written successfully"):
                logger.error("MCP write failed for thread_id=%s: %s", thread_id, mcp_result)
                response = (
                    "Your reservation is confirmed. Note: file recording encountered an issue "
                    "but your booking is saved in our system."
                )
            else:
                logger.info("Reservation written to file via MCP for thread: %s", thread_id[:8])
        except Exception as exc:
            logger.error("MCP write raised exception for thread_id=%s: %s", thread_id, exc)
            response = (
                "Your reservation is confirmed. Note: file recording encountered an issue "
                "but your booking is saved in our system."
            )

        logger.info("Reservation confirmed for thread: %s", thread_id[:8])
        return {**state, "reservation_data": {}, "response": response}
    except Exception as exc:
        logger.error("record_reservation_node failed: %s", exc)
        return {
            **state,
            "response": "Your reservation was approved, but we hit an internal error finalizing it.",
        }


def notify_rejection_node(state: ChatState, config: RunnableConfig) -> ChatState:
    """Finalize rejected reservation state after administrator decision.

    Reads: config.configurable.thread_id.
    Updates: reservation_data, response.
    Side effects: SQL status update.
    """
    from app.database.sql_client import update_reservation_status

    try:
        thread_id = str((config.get("configurable") or {}).get("thread_id", ""))
        if not thread_id:
            raise ValueError("Missing thread_id in RunnableConfig")
        result = update_reservation_status(thread_id, "rejected")

        if not result:
            logger.error("Failed to reject reservation for thread_id=%s", thread_id)
            response = (
                "We encountered an issue processing your rejection. "
                "Please contact the administrator directly."
            )
        else:
            logger.info("Reservation rejected for thread: %s", thread_id[:8])
            response = "Your reservation request has been rejected."

        return {**state, "reservation_data": {}, "response": response}
    except Exception as exc:
        logger.error("notify_rejection_node failed: %s", exc)
        return {
            **state,
            "response": "We could not finalize the admin rejection result right now. Please try again later.",
        }


def response_node(state: ChatState) -> ChatState:
    """Finalize and append turn messages for UI rendering.

    Reads: guardrail_triggered, user_input, response, messages.
    Updates: response, messages.
    Side effects: none.
    """
    try:
        if state.get("guardrail_triggered", False):
            response = (
                "I'm sorry, I can't help with that request. "
                "I'm only able to provide information about our parking spaces "
                "and assist with reservations."
            )
            messages = state.get("messages", [])
            messages = messages + [
                HumanMessage(content=state.get("user_input", "")),
                AIMessage(content=response),
            ]
            return {**state, "response": response, "messages": messages}

        messages = state.get("messages", [])
        messages = messages + [
            HumanMessage(content=state.get("user_input", "")),
            AIMessage(content=state.get("response", "")),
        ]
        return {**state, "messages": messages}
    except Exception as exc:
        logger.error("response_node failed: %s", exc)
        return {
            **state,
            "response": state.get("response", "I encountered an error preparing the response."),
            "messages": state.get("messages", []),
        }


def unknown_node(state: ChatState) -> ChatState:
    """Return fallback guidance when intent cannot be classified.

    Reads: none.
    Updates: response.
    Side effects: none.
    """
    try:
        response = (
            "I'm not sure I understood that. I can help you with:\n"
            "- Information about our 5 parking spaces (prices, hours, location, availability)\n"
            "- Making a parking reservation\n\n"
            "What would you like to know?"
        )
        return {**state, "response": response}
    except Exception as exc:
        logger.error("unknown_node failed: %s", exc)
        return {
            **state,
            "response": "I did not understand that. You can ask about parking info or make a reservation.",
        }


# -- Routing logic ------------------------------------------------------------


def route_after_intent(state: ChatState) -> str:
    """Route after intent classification.

    Returns one of: response_node, rag_node, reservation_node, unknown_node.
    Guardrail-triggered turns short-circuit to response_node.
    """
    if state.get("guardrail_triggered", False):
        return "response_node"

    intent = state.get("intent", "unknown")
    if intent == "info":
        return "rag_node"
    if intent == "reservation":
        return "reservation_node"
    if intent not in VALID_INTENTS:
        logger.warning("Unexpected intent value '%s'; routing to unknown_node", intent)
    return "unknown_node"


def route_after_reservation(state: ChatState) -> str:
    """Route after reservation handling.

    If awaiting_admin is true, continue to submit_to_admin (HITL interrupt path);
    otherwise continue to response_node.
    """
    if state.get("awaiting_admin", False):
        return "submit_to_admin"
    return "response_node"


def route_after_admin_decision(state: ChatState) -> str:
    """Route after administrator decision resume.

    Approved decisions proceed to record_reservation.
    Rejected and unexpected decisions route to notify_rejection.
    """
    decision = state.get("admin_decision", "")
    if decision == "approved":
        return "record_reservation"
    if decision == "rejected":
        return "notify_rejection"
    logger.warning("Unexpected admin decision '%s'; routing to rejection notifier", decision)
    return "notify_rejection"


# -- Graph assembly -----------------------------------------------------------


def get_thread_config(thread_id: str) -> dict:
    """Build the config object used by the checkpointer to scope thread state."""
    return {"configurable": {"thread_id": thread_id}}


def build_graph(conn_string: str | None = None):
    """Assemble and compile the chatbot LangGraph with SQLite checkpointing."""
    raw = conn_string if conn_string is not None else os.getenv("CHECKPOINT_DB_PATH", "checkpoints.db")
    if raw.startswith("sqlite:///"):
        raw = raw[len("sqlite:///") :]

    conn = sqlite3.connect(raw, check_same_thread=False)
    saver = SqliteSaver(conn)

    graph = StateGraph(ChatState)

    graph.add_node("guardrail_node", guardrail_node)
    graph.add_node("intent_node", intent_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("reservation_node", reservation_node)
    graph.add_node("submit_to_admin", submit_to_admin)
    graph.add_node("record_reservation", record_reservation_node)
    graph.add_node("notify_rejection", notify_rejection_node)
    graph.add_node("response_node", response_node)
    graph.add_node("unknown_node", unknown_node)

    graph.add_edge(START, "guardrail_node")
    graph.add_edge("guardrail_node", "intent_node")

    # Route by classified intent or guardrail short-circuit.
    graph.add_conditional_edges(
        "intent_node",
        route_after_intent,
        {
            "rag_node": "rag_node",
            "reservation_node": "reservation_node",
            "response_node": "response_node",
            "unknown_node": "unknown_node",
        },
    )

    graph.add_edge("rag_node", "response_node")

    # Reservation path either finishes normally or pauses for admin review.
    graph.add_conditional_edges(
        "reservation_node",
        route_after_reservation,
        {
            "submit_to_admin": "submit_to_admin",
            "response_node": "response_node",
        },
    )

    # After admin resume, finalize approval or rejection.
    graph.add_conditional_edges(
        "submit_to_admin",
        route_after_admin_decision,
        {
            "record_reservation": "record_reservation",
            "notify_rejection": "notify_rejection",
        },
    )

    graph.add_edge("record_reservation", "response_node")
    graph.add_edge("notify_rejection", "response_node")
    graph.add_edge("unknown_node", "response_node")
    graph.add_edge("response_node", END)

    compiled = graph.compile(checkpointer=saver)
    setattr(compiled, "_sqlite_conn", conn)
    return compiled, saver


chatbot_graph, checkpointer = build_graph()


def save_graph_diagram(output_path: str = "docs/graph_diagram.png"):
    """
    Save a visual PNG diagram of the LangGraph graph to docs/.
    Requires graphviz to be installed.
    Run this once manually: python -c "from app.chatbot.graph import save_graph_diagram; save_graph_diagram()"
    """
    try:
        from pathlib import Path

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        diagram = chatbot_graph.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(diagram)
        logger.info("Graph diagram saved to %s", output_path)
    except Exception as e:
        logger.warning("Could not save graph diagram: %s", e)
