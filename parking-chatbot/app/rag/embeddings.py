"""Embedding utilities for RAG."""

from langchain_openai import OpenAIEmbeddings


def get_embeddings() -> OpenAIEmbeddings:
    """Create OpenAI embeddings client for Weaviate indexing and retrieval."""
    return OpenAIEmbeddings(model="text-embedding-3-small")
