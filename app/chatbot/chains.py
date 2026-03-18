"""Chain builders for the parking assistant.

This module centralizes LangChain pipeline construction so graph nodes can
re-use consistent prompts and model settings for intent classification,
guardrails, reservation assistance, and RAG responses.
"""

import logging
import os
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
    """Return a GPT-4o chat model configured for this project.

    Temperature guidance:
    - 0.0 for deterministic tasks (classification/guardrails/extraction)
    - 0.3 for more conversational reservation assistant responses
    """
    api_key = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(
        model="gpt-4o",
        temperature=temperature,
        api_key=SecretStr(api_key) if api_key else None
    )


def build_rag_chain(retriever):
    """
    Build a RAG chain using GPT-4o with temperature 0.0 and RAG_PROMPT.

    Prompt template: RAG_PROMPT from app.chatbot.prompts.
    Returns: string answer generated from retrieved context + user question.
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
    Build an intent classification chain using GPT-4o (temperature 0.0).

    Prompt template: INTENT_CLASSIFICATION_PROMPT.
    Returns: plain text category "info", "reservation", or "unknown".
    """
    return (
        INTENT_CLASSIFICATION_PROMPT
        | get_llm(temperature=0.0)
        | StrOutputParser()
    )


def build_reservation_chain():
    """
    Build a reservation assistant chain using GPT-4o (temperature 0.3).

    Prompt template: RESERVATION_PROMPT.
    Returns: string response guiding field collection or confirmation.
    """
    return (
        RESERVATION_PROMPT
        | get_llm(temperature=0.3)
        | StrOutputParser()
    )


def build_guardrail_chain():
    """
    Build a guardrail classifier chain using GPT-4o (temperature 0.0).

    Prompt template: GUARDRAIL_PROMPT.
    Returns: string label "safe" or "blocked".
    """
    return (
        GUARDRAIL_PROMPT
        | get_llm(temperature=0.0)
        | StrOutputParser()
    )
