from scripts.load_test import compute_stats, generate_load_test_report


def test_compute_stats_correct_values():
    result = compute_stats([100, 200, 300, 400, 500])
    assert result["min_ms"] == 100.0
    assert result["max_ms"] == 500.0
    assert result["avg_ms"] == 300.0
    assert result["p95_ms"] >= 400.0


def test_compute_stats_single_value():
    result = compute_stats([250.0])
    assert result["min_ms"] == 250.0
    assert result["max_ms"] == 250.0
    assert result["avg_ms"] == 250.0
    assert result["p95_ms"] == 250.0


def test_generate_load_test_report_contains_all_components():
    results = [
        {"component": "RAG Query", "n_requests": 10, "avg_ms": 100, "p95_ms": 150, "min_ms": 80, "max_ms": 200, "errors": 0},
        {"component": "MCP Write", "n_requests": 10, "avg_ms": 120, "p95_ms": 180, "min_ms": 90, "max_ms": 220, "errors": 0},
        {"component": "Graph Invocation", "n_requests": 5, "avg_ms": 200, "p95_ms": 280, "min_ms": 160, "max_ms": 320, "errors": 0},
        {"component": "Concurrent Graph Invocations", "n_concurrent": 3, "avg_per_request_ms": 300, "total_wall_time_ms": 900, "errors": 0},
    ]

    report = generate_load_test_report(results)
    assert "RAG Query" in report
    assert "MCP" in report
    assert "Graph" in report
    assert "Concurrent" in report


def test_generate_load_test_report_flags_slow_component():
    results = [
        {"component": "RAG Query", "n_requests": 10, "avg_ms": 6000, "p95_ms": 7000, "min_ms": 5000, "max_ms": 8000, "errors": 0}
    ]

    report = generate_load_test_report(results)
    assert "⚠️ Slow" in report


def test_generate_load_test_report_flags_errors():
    results = [
        {"component": "MCP Write", "n_requests": 10, "avg_ms": 100, "p95_ms": 120, "min_ms": 90, "max_ms": 140, "errors": 2}
    ]

    report = generate_load_test_report(results)
    assert "❌ Errors detected" in report


def test_generate_load_test_report_shows_ok_for_good_results():
    results = [
        {"component": "Graph Invocation", "n_requests": 5, "avg_ms": 500, "p95_ms": 650, "min_ms": 300, "max_ms": 700, "errors": 0}
    ]

    report = generate_load_test_report(results)
    assert "✅ OK" in report
