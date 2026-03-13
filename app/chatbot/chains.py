import os
import logging
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.chatbot.prompts import (
    RAG_PROMPT,
    INTENT_CLASSIFICATION_PROMPT,
    RESERVATION_PROMPT,
    GUARDRAIL_PROMPT
)

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Return a ChatOpenAI instance using GPT-4o."""
    api_key = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(
        model="gpt-4o",
        temperature=temperature,
        api_key=SecretStr(api_key) if api_key else None
    )


def build_rag_chain(retriever):
    """
    Build and return a RAG chain that:
    1. Retrieves relevant documents from Weaviate using the retriever
    2. Formats context from retrieved docs
    3. Passes context + question to GPT-4o via RAG_PROMPT
    4. Returns a string response
    """
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | get_llm(temperature=0.0)
        | StrOutputParser()
    )


def build_intent_chain():
    """
    Build and return a chain that classifies user intent.
    Returns one of: "info", "reservation", "unknown"
    """
    return (
        INTENT_CLASSIFICATION_PROMPT
        | get_llm(temperature=0.0)
        | StrOutputParser()
    )


def build_reservation_chain():
    """
    Build and return a chain that handles reservation data collection.
    Guides the user through providing all required reservation fields.
    """
    return (
        RESERVATION_PROMPT
        | get_llm(temperature=0.3)
        | StrOutputParser()
    )


def build_guardrail_chain():
    """
    Build and return a chain that checks if user input is safe.
    Returns "safe" or "blocked".
    """
    return (
        GUARDRAIL_PROMPT
        | get_llm(temperature=0.0)
        | StrOutputParser()
    )
