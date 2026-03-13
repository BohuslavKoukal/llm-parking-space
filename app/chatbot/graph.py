import logging
from typing import TypedDict, Optional
import re
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from app.chatbot.chains import (
    build_rag_chain,
    build_intent_chain,
    build_reservation_chain,
    build_guardrail_chain
)
from app.rag.weaviate_client import get_weaviate_client, get_retriever
from app.database.sql_client import init_db, get_all_parkings_summary, get_all_parking_ids_and_names

logger = logging.getLogger(__name__)


# -- State schema -------------------------------------------------------------

class ChatState(TypedDict):
    messages: list[BaseMessage]
    user_input: str
    intent: str                  # "info" | "reservation" | "unknown"
    reservation_data: dict       # collected fields so far
    guardrail_triggered: bool
    response: str


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

    chain = build_guardrail_chain()
    result = chain.invoke({"user_input": state["user_input"]}).strip().lower()
    triggered = result == "blocked"
    if triggered:
        logger.warning(f"Guardrail triggered for input: {state['user_input'][:50]}")
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
    if any(word in user_confirmation for word in ["yes", "confirm", "correct", "ok", "sure"]):
        # Save to database
        success = save_reservation(collected)
        if success:
            # Mark as confirmed and clear for next reservation
            collected["confirmed"] = "yes"
            response = (
                f"✅ Your reservation has been submitted successfully!\n\n"
                f"**Summary:**\n"
                f"- Parking: {collected['parking_id']}\n"
                f"- Name: {collected['name']} {collected['surname']}\n"
                f"- Car: {collected['car_number']}\n"
                f"- Period: {collected['start_date']} to {collected['end_date']}\n\n"
                f"Your reservation is now **pending administrator confirmation**. "
                f"You will be notified once it is approved."
            )
        else:
            response = "❌ There was an error saving your reservation. Please try again."
        return {**state, "reservation_data": collected, "response": response}

    elif any(word in user_confirmation for word in ["no", "cancel", "change", "wrong"]):
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
        return {**state, "response": response}

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


# -- Graph assembly -----------------------------------------------------------

def build_graph():
    """
    Assemble and compile the LangGraph chatbot graph.

    Flow:
    START -> guardrail_node -> intent_node -> (conditional routing)
         |- rag_node -> response_node -> END
         |- reservation_node -> response_node -> END
         '- unknown_node -> response_node -> END
    """
    graph = StateGraph(ChatState)

    # Add nodes
    graph.add_node("guardrail_node", guardrail_node)
    graph.add_node("intent_node", intent_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("reservation_node", reservation_node)
    graph.add_node("unknown_node", unknown_node)
    graph.add_node("response_node", response_node)

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
    graph.add_edge("reservation_node", "response_node")
    graph.add_edge("unknown_node", "response_node")
    graph.add_edge("response_node", END)

    return graph.compile()


# Compiled graph instance - imported by main.py
chatbot_graph = build_graph()
