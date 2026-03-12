"""Data ingestion pipeline for ParkingInfo in Weaviate."""

import json
import logging
import os

import weaviate

from app.rag.embeddings import get_embeddings
from app.rag.weaviate_client import COLLECTION_NAME, ensure_collection_exists, get_weaviate_client

logger = logging.getLogger(__name__)


def chunk_parking_object(parking: dict) -> list[dict]:
    """
    Split a single parking JSON object into multiple text chunks by section.
    Returns list of dicts with keys: content, parking_id, parking_name, section
    """
    parking_id = parking["id"]
    name = parking["parking_name"]
    chunks = []

    features = ", ".join(parking.get("features", []))
    chunks.append(
        {
            "content": f"{name} is located at {parking['address']}. {parking['description']} Features: {features}.",
            "parking_id": parking_id,
            "parking_name": name,
            "section": "general",
        }
    )

    loc = parking["location"]
    chunks.append(
        {
            "content": f"{name} location: nearest metro is {loc['nearest_metro']}, "
            f"{loc['walking_distance_minutes']} minutes walk. "
            f"Coordinates: {loc['latitude']}, {loc['longitude']}.",
            "parking_id": parking_id,
            "parking_name": name,
            "section": "location",
        }
    )

    chunks.append(
        {
            "content": f"Booking process for {name}: {parking['booking_process']}",
            "parking_id": parking_id,
            "parking_name": name,
            "section": "booking",
        }
    )

    contact = parking["contact"]
    chunks.append(
        {
            "content": f"Contact for {name}: phone {contact['phone']}, email {contact['email']}.",
            "parking_id": parking_id,
            "parking_name": name,
            "section": "contact",
        }
    )

    return chunks


def ingest_static_data(client, embeddings):
    """Load static parking JSON, chunk into logical sections, and upsert into ParkingInfo."""
    json_path = os.path.join(os.path.dirname(__file__), "../../data/static/parking_info.json")
    with open(json_path, "r", encoding="utf-8") as file:
        parkings = json.load(file)

    ensure_collection_exists(client)
    collection = client.collections.get(COLLECTION_NAME)

    for parking in parkings:
        parking_id = parking["id"]

        collection.data.delete_many(
            where=weaviate.classes.query.Filter.by_property("parking_id").equal(parking_id)
        )

        chunks = chunk_parking_object(parking)
        texts = [chunk["content"] for chunk in chunks]
        embeddings_list = embeddings.embed_documents(texts)

        for chunk, vector in zip(chunks, embeddings_list):
            collection.data.insert(
                properties={
                    "parking_id": chunk["parking_id"],
                    "parking_name": chunk["parking_name"],
                    "content": chunk["content"],
                    "section": chunk["section"],
                },
                vector=vector,
            )

        logger.info("Ingested %s chunks for %s", len(chunks), parking_id)

    logger.info("Static data ingestion complete.")


def run_ingestion():
    """Initialize dependencies and execute static data ingestion."""
    client = get_weaviate_client()
    embeddings = get_embeddings()
    try:
        ingest_static_data(client, embeddings)
    finally:
        client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ingestion()
