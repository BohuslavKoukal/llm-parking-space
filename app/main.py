import sys
import os
import uuid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from dotenv import load_dotenv
import streamlit as st
from app.chatbot.graph import ChatState, chatbot_graph, get_thread_config
from app.database.sql_client import init_db, get_reservation_by_thread_id
from app.rag.weaviate_client import get_weaviate_client, health_check
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

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

# Initialize thread_id and awaiting_admin before the sidebar so they are always available when rendered
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "awaiting_admin" not in st.session_state:
    st.session_state.awaiting_admin = False


# -- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.title("🅿️ Parking Assistant")
    st.markdown("---")
    st.markdown("**System Status**")
    st.markdown(f"Weaviate: {'🟢 Connected' if weaviate_ok else '🔴 Disconnected'}")
    st.markdown("Database: 🟢 Connected")
    if "thread_id" in st.session_state:
        st.markdown(f"Thread: `{st.session_state.thread_id[:8]}...`")
    if st.session_state.get("awaiting_admin", False):
        st.markdown("Reservation: ⏳ Awaiting admin approval")
    else:
        st.markdown("Reservation: ✅ No pending reservation")
    st.markdown("---")
    st.markdown("**Available Parkings**")
    st.markdown("- 🏙️ CityPark Central\n- ✈️ AirportPark Express\n- 🛍️ ShoppingMall Park\n- 🏟️ Stadium Park\n- 🏛️ OldTown Garage")

    # Admin review panel — only shown when a reservation is awaiting approval
    if st.session_state.get("awaiting_admin", False):
        st.markdown("---")
        st.markdown("**🔐 Admin Review**")
        res = get_reservation_by_thread_id(st.session_state.thread_id)
        if res:
            st.markdown(f"**{res['name']} {res['surname']}**")
            st.markdown(f"Parking: `{res['parking_id']}`")
            st.markdown(f"Car: `{res['car_number']}`")
            st.markdown(f"Dates: {res['start_date']} → {res['end_date']}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Approve", key="btn_approve"):
                try:
                    thread_config = get_thread_config(st.session_state.thread_id)
                    resume_result = chatbot_graph.invoke(
                        Command(resume="approved"), config=thread_config
                    )
                    approval_response = resume_result.get(
                        "response", "Your reservation has been approved!"
                    )
                    st.session_state.messages = st.session_state.messages + [
                        AIMessage(content=approval_response)
                    ]
                    st.session_state.reservation_data = resume_result.get("reservation_data", {})
                    st.session_state.awaiting_admin = False
                except Exception as e:
                    logger.error("Resume error (approve): %s", e)
                st.rerun()
        with col2:
            if st.button("❌ Reject", key="btn_reject"):
                try:
                    thread_config = get_thread_config(st.session_state.thread_id)
                    resume_result = chatbot_graph.invoke(
                        Command(resume="rejected"), config=thread_config
                    )
                    rejection_response = resume_result.get(
                        "response", "Your reservation has been rejected."
                    )
                    st.session_state.messages = st.session_state.messages + [
                        AIMessage(content=rejection_response)
                    ]
                    st.session_state.reservation_data = resume_result.get("reservation_data", {})
                    st.session_state.awaiting_admin = False
                except Exception as e:
                    logger.error("Resume error (reject): %s", e)
                st.rerun()

    st.markdown("---")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.reservation_data = {}
        st.session_state.awaiting_admin = False
        st.session_state.thread_id = str(uuid.uuid4())
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

# Waiting notice while admin decision is pending
if st.session_state.get("awaiting_admin", False):
    st.info("⏳ Your reservation is awaiting administrator approval. Use the sidebar to approve or reject.")

# Handle new input
user_input = st.chat_input(
    "Ask about parking or make a reservation...",
    disabled=st.session_state.is_processing or st.session_state.get("awaiting_admin", False)
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
                    "response": "",
                    "admin_decision": "",
                    "awaiting_admin": False,
                }
                thread_config = get_thread_config(st.session_state.thread_id)
                result = chatbot_graph.invoke(initial_state, config=thread_config)

                if "__interrupt__" in result:
                    # Graph paused at submit_to_admin_node — response_node did not run.
                    response = result.get(
                        "response",
                        "⏳ Your reservation has been submitted and is awaiting administrator approval. "
                        "You will be notified of the decision.",
                    )
                    st.session_state.awaiting_admin = True
                    st.session_state.reservation_data = result.get("reservation_data", {})
                    st.session_state.messages = st.session_state.messages + [
                        HumanMessage(content=user_input),
                        AIMessage(content=response),
                    ]
                else:
                    response = result["response"]
                    st.session_state.reservation_data = result.get("reservation_data", {})
                    st.session_state.messages = result.get("messages", [])

            except Exception as e:
                logger.error(f"Graph invocation error: {e}")
                response = "I encountered an error. Please try again."

            finally:
                # Always release lock
                st.session_state.is_processing = False

            st.markdown(response)
