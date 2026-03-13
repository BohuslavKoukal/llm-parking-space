import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.ragas_eval import run_evaluation_with_retry
from app.evaluation.report import METRIC_THRESHOLDS, save_report
from app.rag.weaviate_client import get_weaviate_client, health_check

logger = logging.getLogger(__name__)


def metric_status(metric: str, score: float) -> str:
    """Return pass/warn/fail status for one RAGAS metric."""
    threshold = METRIC_THRESHOLDS[metric]
    if metric == "faithfulness":
        return "passed" if score >= threshold else "failed"
    return "passed" if score >= threshold else "warning"


def main() -> int:
    """Run evaluation pipeline and save Markdown report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    client = None
    try:
        client = get_weaviate_client()
        if not health_check(client):
            print("ERROR: Weaviate is not reachable. Start Weaviate and ingest data before evaluation.")
            return 1
    except Exception as exc:
        print(f"ERROR: Could not connect to Weaviate: {exc}")
        return 1
    finally:
        if client is not None:
            client.close()

    results = run_evaluation_with_retry()
    report_path = save_report(results)

    print(f"Evaluation report saved to: {report_path}")
    print("RAGAS scores:")

    has_failed_metric = False
    scores = results.get("ragas_scores", {})
    for metric, threshold in METRIC_THRESHOLDS.items():
        score = float(scores.get(metric, 0.0))
        status = metric_status(metric, score)
        print(f"- {metric}: {score:.4f} (threshold {threshold:.2f}) -> {status}")
        if status == "failed":
            has_failed_metric = True

    return 1 if has_failed_metric else 0


if __name__ == "__main__":
    raise SystemExit(main())
