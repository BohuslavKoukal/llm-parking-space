"""Weaviate connectivity and retriever utilities."""

import logging
import os

import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore

LOGGER = logging.getLogger(__name__)


def get_weaviate_client() -> weaviate.WeaviateClient:
    """Connect to Weaviate using environment variables and log status."""
    url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    api_key = os.getenv("WEAVIATE_API_KEY")

    if api_key:
        client = weaviate.connect_to_custom(
            http_host=url.replace("http://", "").replace("https://", ""),
            http_port=443 if url.startswith("https") else 8080,
            http_secure=url.startswith("https"),
            grpc_host=url.replace("http://", "").replace("https://", ""),
            grpc_port=50051,
            grpc_secure=url.startswith("https"),
            auth_credentials=weaviate.auth.AuthApiKey(api_key),
        )
    else:
        client = weaviate.connect_to_local(host="localhost", port=8080)

    LOGGER.info("Connected to Weaviate at %s", url)
    return client


def get_retriever(client, embeddings=None):
    """Create a LangChain retriever for the ParkingInfo collection."""
    if embeddings is None:
        raise ValueError("embeddings must be provided to initialize WeaviateVectorStore")

    vector_store = WeaviateVectorStore(
        client=client,
        index_name="ParkingInfo",
        text_key="text",
        embedding=embeddings,
    )
    return vector_store.as_retriever(search_kwargs={"k": 4})


def health_check(client) -> bool:
    """Return True when Weaviate is reachable."""
    try:
        return bool(client.is_ready())
    except Exception:
        return False
