from pathlib import Path
from unittest.mock import MagicMock

from app.evaluation.ragas_eval import build_eval_sample, load_eval_dataset
from app.evaluation.report import generate_markdown_report, save_report


def test_load_eval_dataset_returns_list():
    """Loading the real evaluation dataset should return non-empty list with required keys."""
    dataset = load_eval_dataset("data/evaluation/eval_questions.json")
    assert isinstance(dataset, list)
    assert len(dataset) > 0
    for item in dataset:
        assert "question" in item
        assert "ground_truth" in item
        assert "contexts" in item


def test_load_eval_dataset_has_ten_questions():
    """Evaluation dataset should contain exactly 10 question entries."""
    dataset = load_eval_dataset("data/evaluation/eval_questions.json")
    assert len(dataset) == 10


def test_generate_markdown_report_contains_required_sections():
    """Generated Markdown report should include all required section headings."""
    mock_results = {
        "ragas_scores": {
            "faithfulness": 0.82,
            "answer_relevancy": 0.79,
            "context_recall": 0.74,
            "context_precision": 0.71,
        },
        "avg_latency_ms": 123.4,
        "total_latency_ms": 1234.0,
        "num_questions": 10,
        "timestamp": "2026-03-13T12:00:00+00:00",
        "per_question_results": [
            {
                "question": "Sample question",
                "latency_ms": 100.0,
                "answer": "Sample answer",
            }
        ],
    }

    report = generate_markdown_report(mock_results)
    assert "RAGAS Metrics" in report
    assert "Performance Metrics" in report
    assert "Per-question Results" in report
    assert "Executive Summary" in report
    assert "Conclusions" in report


def test_generate_markdown_report_shows_pass_fail_status():
    """Report should display both passed and failed status words when metrics cross thresholds."""
    mock_results = {
        "ragas_scores": {
            "faithfulness": 0.3,
            "answer_relevancy": 0.9,
            "context_recall": 0.7,
            "context_precision": 0.7,
        },
        "avg_latency_ms": 120,
        "total_latency_ms": 1200,
        "num_questions": 10,
        "timestamp": "2026-03-13T12:00:00+00:00",
        "per_question_results": [],
    }

    report = generate_markdown_report(mock_results)
    assert "passed" in report
    assert "failed" in report


def test_save_report_creates_file(tmp_path):
    """save_report should create a markdown report file and latest_results JSON file."""
    mock_results = {
        "ragas_scores": {
            "faithfulness": 0.8,
            "answer_relevancy": 0.8,
            "context_recall": 0.8,
            "context_precision": 0.8,
        },
        "avg_latency_ms": 100,
        "total_latency_ms": 1000,
        "num_questions": 10,
        "timestamp": "2026-03-13T12:00:00+00:00",
        "per_question_results": [],
    }

    output = tmp_path / "eval_report_test.md"
    report_path = save_report(mock_results, output_path=str(output))

    report_file = Path(report_path)
    assert report_file.exists()
    assert (tmp_path / "latest_results.json").exists()

    report_file.unlink()


def test_build_eval_sample_measures_latency(monkeypatch):
    """build_eval_sample should include positive latency_ms while using mocked RAG dependencies."""
    fake_client = MagicMock()
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = "Mocked RAG answer"

    monkeypatch.setattr("app.evaluation.ragas_eval.get_weaviate_client", lambda: fake_client)
    monkeypatch.setattr("app.evaluation.ragas_eval.get_retriever", lambda client: object())
    monkeypatch.setattr("app.evaluation.ragas_eval.build_rag_chain", lambda retriever: fake_chain)
    monkeypatch.setattr("app.evaluation.ragas_eval.get_all_parking_ids_and_names", lambda: ["parking_001", "parking_002"])
    monkeypatch.setattr("app.evaluation.ragas_eval.get_all_parkings_summary", lambda: [{"parking_id": "parking_001", "price": {"hourly": "3.50"}}])

    sample = build_eval_sample(
        question="What is the hourly price at parking_001?",
        ground_truth="2.5 EUR per hour",
        contexts=["parking_001 hourly price is 2.5 EUR"],
    )

    assert "latency_ms" in sample
    assert isinstance(sample["latency_ms"], (int, float))
    assert sample["latency_ms"] >= 0
    assert sample["answer"] == "Mocked RAG answer"
    assert fake_client.close.called
    invoked_input = fake_chain.invoke.call_args.args[0]
    assert "Known parking IDs" in invoked_input
    assert "Dynamic data per parking" in invoked_input
