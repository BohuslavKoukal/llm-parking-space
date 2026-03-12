"""Streamlit entry point for the Parking Chatbot application."""

import logging
import os
from typing import cast

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import text

from app.chatbot.graph import ChatState, chatbot_graph
from app.database.sql_client import get_engine, init_db
from app.rag.weaviate_client import get_weaviate_client, health_check

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


@st.cache_resource
def _db_status() -> bool:
    try:
        init_db()
        engine = get_engine()
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        LOGGER.exception("Database health check failed")
        return False


@st.cache_resource
def _weaviate_status() -> bool:
    client = None
    try:
        client = get_weaviate_client()
        return health_check(client)
    except Exception:
        LOGGER.exception("Weaviate health check failed")
        return False
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                LOGGER.debug("Unable to close Weaviate client cleanly")


def _render_sidebar() -> None:
    st.sidebar.header("Parking Chatbot")
    st.sidebar.subheader("CityPark Central")

    weaviate_connected = _weaviate_status()
    db_connected = _db_status()

    st.sidebar.write(f"Weaviate connected: {'yes' if weaviate_connected else 'no'}")
    st.sidebar.write(f"DB connected: {'yes' if db_connected else 'no'}")


def _ensure_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _invoke_chatbot(user_input: str) -> str:
    graph_messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            graph_messages.append(HumanMessage(content=msg["content"]))
        else:
            graph_messages.append(AIMessage(content=msg["content"]))

    state = cast(
        ChatState,
        {
        "messages": graph_messages,
        "user_input": user_input,
        "intent": "unknown",
        "reservation_data": {},
        "guardrail_triggered": False,
        "response": "",
        },
    )
    result = chatbot_graph.invoke(state)
    return result.get("response", "I am here to help with parking information and reservations.")


def main() -> None:
    st.set_page_config(page_title="Parking Chatbot", page_icon="🅿️", layout="wide")
    st.title("Parking Chatbot")
    st.caption("RAG-powered assistant for parking information and reservations")

    _ensure_session_state()
    _render_sidebar()
    _render_history()

    user_input = st.chat_input("Ask about parking info or make a reservation")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            assistant_response = _invoke_chatbot(user_input)
        except Exception:
            LOGGER.exception("Chatbot invocation failed")
            assistant_response = (
                "Sorry, I ran into a temporary issue while processing your request. "
                "Please try again in a moment."
            )

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)


if __name__ == "__main__":
    main()
