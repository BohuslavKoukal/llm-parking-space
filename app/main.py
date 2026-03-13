import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from dotenv import load_dotenv
import streamlit as st
from app.chatbot.graph import ChatState, chatbot_graph
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


# -- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.title("🅿️ Parking Assistant")
    st.markdown("---")
    st.markdown("**System Status**")
    st.markdown(f"Weaviate: {'🟢 Connected' if weaviate_ok else '🔴 Disconnected'}")
    st.markdown("Database: 🟢 Connected")
    st.markdown("---")
    st.markdown("**Available Parkings**")
    st.markdown("- 🏙️ CityPark Central\n- ✈️ AirportPark Express\n- 🛍️ ShoppingMall Park\n- 🏟️ Stadium Park\n- 🏛️ OldTown Garage")
    st.markdown("---")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.reservation_data = {}
        st.rerun()


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "reservation_data" not in st.session_state:
    st.session_state.reservation_data = {}
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "last_processed_input" not in st.session_state:
    st.session_state.last_processed_input = None


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

# Handle new input
user_input = st.chat_input(
    "Ask about parking or make a reservation...",
    disabled=st.session_state.is_processing
)

if user_input and not st.session_state.is_processing:
    # Set lock immediately before anything else
    st.session_state.is_processing = True
    st.session_state.last_processed_input = user_input

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                initial_state: ChatState = {
                    "messages": st.session_state.messages,
                    "user_input": user_input,
                    "intent": "",
                    "reservation_data": st.session_state.reservation_data,
                    "guardrail_triggered": False,
                    "response": ""
                }
                result = chatbot_graph.invoke(initial_state)
                response = result["response"]

                # Save state before any rerun can happen
                st.session_state.reservation_data = result.get("reservation_data", {})
                st.session_state.messages = result.get("messages", [])

            except Exception as e:
                logger.error(f"Graph invocation error: {e}")
                response = "I encountered an error. Please try again."

            finally:
                # Always release lock
                st.session_state.is_processing = False

            st.markdown(response)
