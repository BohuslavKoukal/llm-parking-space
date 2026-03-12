"""Embedding utilities for RAG."""

import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def get_embeddings() -> OpenAIEmbeddings:
    """
    Return OpenAI embeddings model.
    Uses text-embedding-3-small for cost efficiency and good performance.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or .env file before running ingestion."
        )

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )
