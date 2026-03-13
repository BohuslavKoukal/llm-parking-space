"""Markdown reporting helpers for RAGAS evaluation results."""

from __future__ import annotations

import json
from pathlib import Path

METRIC_THRESHOLDS = {
    "faithfulness": 0.7,
    "answer_relevancy": 0.7,
    "context_recall": 0.6,
    "context_precision": 0.6,
}


def _metric_status(metric: str, score: float) -> str:
    threshold = METRIC_THRESHOLDS[metric]
    if metric == "faithfulness":
        return "passed" if score >= threshold else "failed"
    return "passed" if score >= threshold else "warning"


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def generate_markdown_report(results: dict) -> str:
    """Generate a Markdown evaluation report from results dict."""
    scores = results.get("ragas_scores", {})

    statuses = [_metric_status(metric, float(scores.get(metric, 0.0))) for metric in METRIC_THRESHOLDS]
    if "failed" in statuses:
        executive_status = "failed"
    elif "warning" in statuses:
        executive_status = "warning"
    else:
        executive_status = "passed"

    lines: list[str] = []
    lines.append("# RAGAS Evaluation Report")
    lines.append("")
    lines.append(f"Timestamp: {results.get('timestamp', 'unknown')}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"Overall status: **{executive_status}**")
    lines.append("")
    lines.append("## RAGAS Metrics")
    lines.append("")
    lines.append("| Metric | Score | Threshold | Status |")
    lines.append("| --- | ---: | ---: | --- |")

    for metric, threshold in METRIC_THRESHOLDS.items():
        score = float(scores.get(metric, 0.0))
        status = _metric_status(metric, score)
        lines.append(f"| {metric} | {score:.4f} | {threshold:.2f} | {status} |")

    lines.append("")
    lines.append("## Performance Metrics")
    lines.append("")
    lines.append(f"- Average latency (ms): {results.get('avg_latency_ms', 0)}")
    lines.append(f"- Total latency (ms): {results.get('total_latency_ms', 0)}")
    lines.append(f"- Number of questions: {results.get('num_questions', 0)}")

    lines.append("")
    lines.append("## Per-question Results")
    lines.append("")
    lines.append("| Question | Latency (ms) | Answer Preview |")
    lines.append("| --- | ---: | --- |")

    for item in results.get("per_question_results", []):
        question = _escape_cell(str(item.get("question", "")))
        latency = item.get("latency_ms", 0)
        answer = _escape_cell(str(item.get("answer", "")))
        preview = answer[:120] + ("..." if len(answer) > 120 else "")
        lines.append(f"| {question} | {latency} | {preview} |")

    lines.append("")
    lines.append("## Conclusions and Recommendations")
    lines.append("")
    if executive_status == "passed":
        lines.append("- RAG quality is healthy across measured dimensions.")
        lines.append("- Continue periodic evaluation to detect regressions.")
    elif executive_status == "warning":
        lines.append("- Some metrics are below preferred targets but not critically failing.")
        lines.append("- Improve retrieval relevance and context coverage for weaker questions.")
    else:
        lines.append("- At least one critical metric failed and needs immediate attention.")
        lines.append("- Investigate retrieval grounding and answer faithfulness before release.")

    return "\n".join(lines)


def save_report(results: dict, output_path: str | None = None) -> str:
    """Save markdown report and latest results JSON, returning saved report path."""
    report_md = generate_markdown_report(results)

    if output_path:
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        reports_dir = Path(__file__).resolve().parents[2] / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        ts = str(results.get("timestamp", "unknown"))
        safe_ts = ts.replace(":", "-").replace("+", "_")
        report_path = reports_dir / f"eval_report_{safe_ts}.md"

    report_path.write_text(report_md, encoding="utf-8")

    latest_results_path = report_path.parent / "latest_results.json"
    latest_results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return str(report_path)
