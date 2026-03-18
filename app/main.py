"""Streamlit entrypoint for the parking assistant UI.

Structure:
1. Load environment and initialize backend services (DB and Weaviate health).
2. Initialize Streamlit session state for chat/thread/processing flags.
3. Render sidebar status and main chat transcript.
4. Invoke LangGraph for each new user message with thread-scoped config.
"""

import sys
import os
import uuid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from dotenv import load_dotenv
import streamlit as st
from app.chatbot.graph import ChatState, chatbot_graph, get_thread_config
from app.database.sql_client import init_db
from app.rag.weaviate_client import get_weaviate_client, health_check
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


# -- Page config --------------------------------------------------------------
st.set_page_config(
    page_title="Parking Assistant",
    page_icon="🅿️",
    layout="centered"
)


# -- Initialize DB on startup -------------------------------------------------
@st.cache_resource
def initialize_services():
    """Initialize database and check Weaviate connection once on startup."""
    init_db()
    client = get_weaviate_client()
    weaviate_ok = False
    try:
        weaviate_ok = health_check(client)
    finally:
        try:
            client.close()
        except Exception as exc:
            logger.warning("Failed to close Weaviate client: %s", exc)
    return weaviate_ok


weaviate_ok = initialize_services()


def initialize_session_state() -> None:
    """Initialize all Streamlit session keys in one place.

    Fields:
    - thread_id: stable conversation identifier for LangGraph checkpointing
    - awaiting_admin: whether current reservation is paused for admin review
    - messages: LangChain chat history shown in the UI
    - reservation_data: in-progress reservation fields across turns
    - is_processing: lock flag preventing concurrent/double submission
    - last_processed_input: audit/debug helper for most recent submitted input
    """
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "awaiting_admin" not in st.session_state:
        st.session_state.awaiting_admin = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "reservation_data" not in st.session_state:
        st.session_state.reservation_data = {}
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "last_processed_input" not in st.session_state:
        st.session_state.last_processed_input = None


initialize_session_state()


# -- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.title("🅿️ Parking Assistant")
    st.markdown("---")
    st.markdown("**System Status**")
    st.markdown(f"Weaviate: {'🟢 Connected' if weaviate_ok else '🔴 Disconnected'}")
    st.markdown("Database: 🟢 Connected")
    if "thread_id" in st.session_state:
        st.markdown(f"Thread: `{st.session_state.thread_id[:8]}...`")
    st.markdown("---")
    st.markdown("**Reservation Status**")
    if st.session_state.get("awaiting_admin", False):
        st.markdown("⏳ Awaiting admin approval")
    else:
        st.markdown("✅ No pending reservation")
    st.markdown("---")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.reservation_data = {}
        st.session_state.awaiting_admin = False
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


# -- Chat UI ------------------------------------------------------------------
st.title("🅿️ Parking Assistant")
st.markdown("Ask me about parking spaces, prices, availability, or make a reservation!")

# Display message history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Waiting notice while admin decision is pending
if st.session_state.get("awaiting_admin", False):
    st.info("⏳ Your reservation is awaiting administrator approval.")

# Handle new input
user_input = st.chat_input(
    "Ask about parking or make a reservation...",
    disabled=st.session_state.is_processing,
)

if user_input and not st.session_state.is_processing:
    # Processing lock prevents duplicate graph invocations from rapid UI reruns.
    st.session_state.is_processing = True
    st.session_state.last_processed_input = user_input

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if st.session_state.awaiting_admin:
                    # CASE 2: Graph is interrupted — show pending notice, no invoke
                    response = (
                        "⏳ Your reservation is currently pending administrator "
                        "approval. Please wait for the administrator to review "
                        "your request. You will be notified of the outcome."
                    )
                else:
                    # CASE 1: Normal flow
                    initial_state: ChatState = {
                        "messages": st.session_state.messages,
                        "user_input": user_input,
                        "intent": "",
                        "reservation_data": st.session_state.reservation_data,
                        "guardrail_triggered": False,
                        "response": "",
                        "admin_decision": "",
                        "awaiting_admin": False,
                    }
                    thread_config = get_thread_config(st.session_state.thread_id)
                    result = chatbot_graph.invoke(initial_state, config=thread_config)
                    response = result["response"]
                    st.session_state.reservation_data = result.get("reservation_data", {})
                    st.session_state.messages = result.get("messages", [])

                    # Check if graph is now paused at an interrupt
                    state_snapshot = chatbot_graph.get_state(thread_config)
                    st.session_state.awaiting_admin = bool(state_snapshot.next)

            except Exception as e:
                logger.error("Graph invocation error: %s", e)
                response = (
                    "Something went wrong while processing your request. "
                    "Please try again in a moment."
                )

            finally:
                # Always release lock so future messages are not blocked after an error.
                st.session_state.is_processing = False

            st.markdown(response)
