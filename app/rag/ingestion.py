"""Data ingestion pipeline for ParkingInfo in Weaviate."""

import json
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_weaviate.vectorstores import WeaviateVectorStore

from app.rag.embeddings import get_embeddings
from app.rag.weaviate_client import get_weaviate_client

LOGGER = logging.getLogger(__name__)


def _chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list[str]:
    """Split text into overlapping chunks without external splitter dependencies."""
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def ingest_static_data(client, embeddings) -> None:
    """Load static parking JSON, chunk into logical sections, and upsert into ParkingInfo."""
    data_path = Path("data/static/parking_info.json")
    with data_path.open("r", encoding="utf-8") as file:
        parking_info = json.load(file)

    sections = [
        ("overview", {
            "parking_name": parking_info.get("parking_name"),
            "address": parking_info.get("address"),
            "description": parking_info.get("description"),
        }),
        ("capacity", {
            "total_spaces": parking_info.get("total_spaces"),
            "levels": parking_info.get("levels"),
            "features": parking_info.get("features", []),
        }),
        ("location", parking_info.get("location", {})),
        ("booking_process", {"booking_process": parking_info.get("booking_process")}),
        ("contact", parking_info.get("contact", {})),
    ]

    docs = [
        Document(
            page_content=json.dumps(section_content, ensure_ascii=False, indent=2),
            metadata={"section": section_name},
        )
        for section_name, section_content in sections
    ]

    split_docs: list[Document] = []
    for doc in docs:
        for chunk in _chunk_text(doc.page_content, chunk_size=600, overlap=80):
            split_docs.append(Document(page_content=chunk, metadata=doc.metadata))

    vector_store = WeaviateVectorStore(
        client=client,
        index_name="ParkingInfo",
        text_key="text",
        embedding=embeddings,
    )
    vector_store.add_documents(split_docs)

    LOGGER.info("Ingested %s document chunks into ParkingInfo", len(split_docs))


def run_ingestion() -> None:
    """Initialize dependencies and execute static data ingestion."""
    logging.basicConfig(level=logging.INFO)
    client = get_weaviate_client()
    try:
        embeddings = get_embeddings()
        ingest_static_data(client, embeddings)
        LOGGER.info("Ingestion completed successfully")
    finally:
        try:
            client.close()
        except Exception:
            LOGGER.debug("Weaviate client close skipped")
