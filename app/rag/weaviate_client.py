"""Weaviate connectivity and retriever utilities."""

import logging
import os
from urllib.parse import urlparse

import weaviate
from dotenv import load_dotenv
from weaviate.classes.config import Configure, DataType, Property
from langchain_weaviate.vectorstores import WeaviateVectorStore

from app.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "ParkingInfo"

load_dotenv()


def _parse_local_endpoint(weaviate_url: str) -> tuple[str, int]:
    """Parse and validate local Weaviate URL into host and port values."""
    parsed = urlparse(weaviate_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError(
            "WEAVIATE_URL is invalid. Expected format like http://localhost:8080"
        )

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise RuntimeError(
            f"WEAVIATE_URL has an invalid port: '{weaviate_url}'. "
            "Use a single numeric port, e.g. http://localhost:8080"
        ) from exc

    return parsed.hostname, port


def get_weaviate_client() -> weaviate.WeaviateClient:
    """
    Connect to Weaviate using environment variables.
    Supports both local (anonymous) and cloud (API key) connections.
    Returns a connected WeaviateClient instance.
    """
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080").strip()
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY", "")

    if weaviate_api_key:  # empty string = local mode
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
        )
    else:
        host, port = _parse_local_endpoint(weaviate_url)
        client = weaviate.connect_to_local(host=host, port=port)

    logger.info("Weaviate client connected: %s", client.is_ready())
    return client


def ensure_collection_exists(client: weaviate.WeaviateClient):
    """
    Create the ParkingInfo collection in Weaviate if it does not already exist.
    Uses no built-in vectorizer - vectors are provided externally via OpenAI embeddings.
    """
    if not client.collections.exists(COLLECTION_NAME):
        client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="parking_id", data_type=DataType.TEXT),
                Property(name="parking_name", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="section", data_type=DataType.TEXT),
            ],
        )
        logger.info("Collection '%s' created in Weaviate.", COLLECTION_NAME)
    else:
        logger.info("Collection '%s' already exists.", COLLECTION_NAME)


def get_retriever(client: weaviate.WeaviateClient, k: int = 20):
    """
    Return a LangChain Weaviate vector store retriever for the ParkingInfo collection.
    k: number of documents to retrieve per query.
    """
    embeddings = get_embeddings()

    vector_store = WeaviateVectorStore(
        client=client,
        index_name=COLLECTION_NAME,
        text_key="content",
        embedding=embeddings,
    )
    return vector_store.as_retriever(search_kwargs={"k": k})


def health_check(client: weaviate.WeaviateClient) -> bool:
    """Return True if Weaviate is reachable and ready."""
    try:
        return client.is_ready()
    except Exception as e:
        logger.error("Weaviate health check failed: %s", e)
        return False
