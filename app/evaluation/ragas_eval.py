"""RAGAS evaluation pipeline for parking chatbot RAG quality measurements."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from app.chatbot.chains import build_rag_chain
from app.database.sql_client import get_all_parking_ids_and_names, get_all_parkings_summary, init_db
from app.rag.weaviate_client import get_retriever, get_weaviate_client

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "eval_questions.json"
RAGAS_METRICS = [faithfulness, answer_relevancy, context_recall, context_precision]
RAGAS_METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def load_eval_dataset(path: str) -> list[dict]:
    """Load and return the evaluation dataset from JSON file path."""
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("Evaluation dataset must be a list of entries.")

    return data


def build_eval_sample(
    question: str,
    ground_truth: str,
    contexts: list[str],
    all_parking_ids: list[str] | None = None,
    dynamic_summary: list[dict] | None = None,
) -> dict:
    """Build one evaluation sample by generating answer via RAG chain and measuring latency."""
    if all_parking_ids is None:
        all_parking_ids = get_all_parking_ids_and_names()
    if dynamic_summary is None:
        dynamic_summary = get_all_parkings_summary()

    start = time.perf_counter()
    client = get_weaviate_client()
    try:
        retriever = get_retriever(client)
        rag_chain = build_rag_chain(retriever)

        # Match production rag_node behavior by enriching input with dynamic SQL data.

        all_parking_ids = get_all_parking_ids_and_names()
        dynamic_summary = get_all_parkings_summary()
        enriched_question = (
            f"{question}\n\n"
            f"[Known parking IDs: {', '.join(all_parking_ids)}]\n"
            f"[Dynamic data per parking: {dynamic_summary}]"
        )

        answer = rag_chain.invoke(enriched_question)
    finally:
        client.close()

    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "question": question,
        "answer": str(answer),
        "contexts": contexts,
        "ground_truth": ground_truth,
        "latency_ms": round(latency_ms, 2),
    }


def _extract_metric_scores(eval_result: Any) -> dict[str, float]:
    """Extract aggregate metric scores from ragas evaluate() result object."""
    scores: dict[str, float] = {}

    source_dict: dict[str, Any] = {}
    if isinstance(eval_result, dict):
        source_dict = eval_result
    elif hasattr(eval_result, "to_dict"):
        source_dict = eval_result.to_dict()  # type: ignore[assignment]

    for name in RAGAS_METRIC_NAMES:
        value: Any = source_dict.get(name)

        if value is None:
            try:
                value = eval_result[name]
            except Exception as exc:
                logger.warning(
                    "Metric '%s' is missing or unreadable in evaluation result; "
                    "defaulting score to 0.0. Error: %s",
                    name,
                    exc,
                )
                value = 0.0

        if isinstance(value, list):
            numeric_values = [float(v) for v in value if isinstance(v, (int, float))]
            score = sum(numeric_values) / len(numeric_values) if numeric_values else 0.0
        elif isinstance(value, (int, float)):
            score = float(value)
        else:
            logger.warning(
                "Metric '%s' has unsupported type %s; defaulting score to 0.0.",
                name,
                type(value).__name__,
            )
            score = 0.0

        scores[name] = round(score, 4)

    return scores


def run_evaluation(dataset_path: str | None = None) -> dict:
    """Run RAGAS evaluation for all questions and return aggregate + per-question results."""
    effective_path = dataset_path or str(DEFAULT_DATASET_PATH)
    dataset_entries = load_eval_dataset(effective_path)

    # Ensure dynamic SQL data exists because evaluation depends on enriched runtime context.
    init_db()

    # Fetch invariant dynamic SQL data once per evaluation run.
    all_parking_ids = get_all_parking_ids_and_names()
    dynamic_summary = get_all_parkings_summary()

    total_start = time.perf_counter()
    samples: list[dict] = []

    for entry in dataset_entries:
        sample = build_eval_sample(
            question=entry["question"],
            ground_truth=entry["ground_truth"],
            contexts=entry["contexts"],
            all_parking_ids=all_parking_ids,
            dynamic_summary=dynamic_summary,
        )
        samples.append(sample)

    eval_dataset = Dataset.from_list(
        [
            {
                "question": sample["question"],
                "answer": sample["answer"],
                "contexts": sample["contexts"],
                "ground_truth": sample["ground_truth"],
            }
            for sample in samples
        ]
    )

    eval_result = evaluate(dataset=eval_dataset, metrics=RAGAS_METRICS)
    ragas_scores = _extract_metric_scores(eval_result)

    total_latency_ms = round((time.perf_counter() - total_start) * 1000, 2)
    avg_latency_ms = round(
        sum(sample["latency_ms"] for sample in samples) / len(samples),
        2,
    ) if samples else 0.0

    return {
        "ragas_scores": ragas_scores,
        "avg_latency_ms": avg_latency_ms,
        "total_latency_ms": total_latency_ms,
        "num_questions": len(samples),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "per_question_results": [
            {
                "question": sample["question"],
                "latency_ms": sample["latency_ms"],
                "answer": sample["answer"],
            }
            for sample in samples
        ],
    }


def run_evaluation_with_retry(retries: int = 3) -> dict:
    """Run evaluation with retry to tolerate transient upstream/API failures."""
    if retries < 1:
        raise ValueError("retries must be >= 1")

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            logger.info("Starting evaluation attempt %s/%s", attempt, retries)
            return run_evaluation()
        except Exception as exc:
            last_exc = exc
            logger.warning("Evaluation attempt %s/%s failed: %s", attempt, retries, exc)
            if attempt < retries:
                backoff_seconds = min(2 ** (attempt - 1), 5)
                time.sleep(backoff_seconds)

    raise RuntimeError("Evaluation failed after all retry attempts") from last_exc
