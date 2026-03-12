"""LangGraph workflow definition for the parking chatbot."""

from typing import Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph


class ChatState(TypedDict):
    messages: list[BaseMessage]
    user_input: str
    intent: str
    reservation_data: dict[str, Any]
    guardrail_triggered: bool
    response: str


def guardrail_node(state: ChatState) -> ChatState:
    """Stub guardrail node: detects basic sensitive keywords in input."""
    text = state.get("user_input", "").lower()
    sensitive_keywords = ["password", "api_key", "token", "secret"]
    triggered = any(keyword in text for keyword in sensitive_keywords)

    return {
        **state,
        "guardrail_triggered": triggered,
        "response": (
            "I cannot help with sensitive internal data."
            if triggered
            else state.get("response", "")
        ),
    }


def intent_node(state: ChatState) -> ChatState:
    """Stub intent classifier based on simple keyword matching."""
    if state.get("guardrail_triggered"):
        intent = "unknown"
    else:
        text = state.get("user_input", "").lower()
        if any(word in text for word in ["reserve", "reservation", "book", "booking"]):
            intent = "reservation"
        elif text.strip():
            intent = "info"
        else:
            intent = "unknown"

    return {**state, "intent": intent}


def rag_node(state: ChatState) -> ChatState:
    """Stub RAG node returning an informational placeholder response."""
    return {
        **state,
        "response": (
            "Here is what I found about parking information. "
            "(RAG integration placeholder response.)"
        ),
    }


def reservation_node(state: ChatState) -> ChatState:
    """Stub reservation node that asks for missing reservation fields."""
    reservation_data = dict(state.get("reservation_data", {}))
    required = ["name", "surname", "car_number", "start_date", "end_date"]
    missing = [field for field in required if not reservation_data.get(field)]

    response = (
        f"Please provide your {missing[0]} to continue the reservation."
        if missing
        else "Thanks. Your reservation request is complete and pending confirmation."
    )

    return {**state, "reservation_data": reservation_data, "response": response}


def response_node(state: ChatState) -> ChatState:
    """Final response formatter node."""
    response_text = state.get("response") or "How can I help with your parking request today?"
    messages = list(state.get("messages", []))
    if state.get("user_input"):
        messages.append(HumanMessage(content=state["user_input"]))
    messages.append(AIMessage(content=response_text))

    return {**state, "messages": messages, "response": response_text}


def _route_by_intent(state: ChatState) -> str:
    """Route by intent to either info retrieval or reservation flow."""
    return "reservation" if state.get("intent") == "reservation" else "info"


def build_graph():
    """Build and compile the chatbot state graph."""
    graph = StateGraph(ChatState)

    graph.add_node("guardrail_node", guardrail_node)
    graph.add_node("intent_node", intent_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("reservation_node", reservation_node)
    graph.add_node("response_node", response_node)

    graph.add_edge(START, "guardrail_node")
    graph.add_edge("guardrail_node", "intent_node")
    graph.add_conditional_edges(
        "intent_node",
        _route_by_intent,
        {
            "info": "rag_node",
            "reservation": "reservation_node",
        },
    )
    graph.add_edge("rag_node", "response_node")
    graph.add_edge("reservation_node", "response_node")
    graph.add_edge("response_node", END)

    return graph.compile()


chatbot_graph = build_graph()
