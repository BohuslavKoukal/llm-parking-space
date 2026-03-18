import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from app.chatbot.graph import ChatState, chatbot_graph, get_thread_config
from app.chatbot.chains import build_rag_chain
from app.mcp_client.tools import write_reservation_via_mcp
from app.rag.weaviate_client import get_retriever, get_weaviate_client, health_check

load_dotenv()


def compute_stats(latencies: list[float]) -> dict:
    """Compute min, max, average, and p95 latency values in milliseconds."""
    if not latencies:
        return {"min_ms": 0.0, "max_ms": 0.0, "avg_ms": 0.0, "p95_ms": 0.0}

    sorted_latencies = sorted(latencies)
    p95_index = int(len(sorted_latencies) * 0.95)
    p95_index = min(p95_index, len(sorted_latencies) - 1)

    min_ms = round(sorted_latencies[0], 2)
    max_ms = round(sorted_latencies[-1], 2)
    avg_ms = round(sum(sorted_latencies) / len(sorted_latencies), 2)
    p95_ms = round(sorted_latencies[p95_index], 2)

    return {
        "min_ms": min_ms,
        "max_ms": max_ms,
        "avg_ms": avg_ms,
        "p95_ms": p95_ms,
    }


def test_rag_latency(n_requests: int = 5) -> dict:
    """Measure latency for direct RAG chain invocations."""
    questions = [
        "What are the prices at CityPark Central?",
        "What are the working hours at AirportPark Express?",
        "Where is OldTown Garage located?",
        "What features does Stadium Park have?",
        "How do I make a reservation?",
    ]

    latencies_ms: list[float] = []
    errors = 0

    for i in range(n_requests):
        question = questions[i % len(questions)]
        start = time.perf_counter()
        client = None
        try:
            client = get_weaviate_client()
            retriever = get_retriever(client, k=5)
            chain = build_rag_chain(retriever)
            _ = chain.invoke(question)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(latency_ms)
        except Exception:
            errors += 1
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    stats = compute_stats(latencies_ms)
    return {
        "component": "RAG Query",
        "n_requests": n_requests,
        "latencies_ms": latencies_ms,
        **stats,
        "errors": errors,
    }


def test_mcp_write_latency(n_requests: int = 5) -> dict:
    """Measure latency for MCP-backed reservation write calls."""
    latencies_ms: list[float] = []
    errors = 0

    for i in range(1, n_requests + 1):
        start = time.perf_counter()
        try:
            _ = write_reservation_via_mcp(
                name="Load",
                surname="Test",
                car_number=f"LT-{i:04d}",
                parking_id="parking_001",
                start_date="2026-05-01",
                end_date="2026-05-02",
                approval_time=datetime.now().isoformat(),
            )
            latency_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(latency_ms)
        except Exception:
            errors += 1

    reservations_path = Path(os.getenv("RESERVATIONS_FILE_PATH", "data/reservations.txt"))
    if reservations_path.exists():
        try:
            with reservations_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            filtered = [line for line in lines if "Load Test" not in line]
            with reservations_path.open("w", encoding="utf-8") as f:
                f.writelines(filtered)
        except Exception:
            # Cleanup failure should not crash reporting.
            pass

    stats = compute_stats(latencies_ms)
    return {
        "component": "MCP Write",
        "n_requests": n_requests,
        "latencies_ms": latencies_ms,
        **stats,
        "errors": errors,
    }


def test_graph_latency(n_requests: int = 5) -> dict:
    """Measure latency for full graph execution of simple info queries."""
    latencies_ms: list[float] = []
    errors = 0

    for i in range(n_requests):
        thread_config = cast(RunnableConfig, get_thread_config(f"load-graph-{i}-{datetime.now().timestamp()}"))
        state: ChatState = {
            "messages": [],
            "user_input": "What are the parking prices?",
            "intent": "",
            "reservation_data": {},
            "guardrail_triggered": False,
            "response": "",
            "admin_decision": "",
            "awaiting_admin": False,
        }

        start = time.perf_counter()
        try:
            _ = chatbot_graph.invoke(state, config=thread_config)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(latency_ms)
        except Exception:
            errors += 1

    stats = compute_stats(latencies_ms)
    return {
        "component": "Graph Invocation",
        "n_requests": n_requests,
        "latencies_ms": latencies_ms,
        **stats,
        "errors": errors,
    }


def test_concurrent_graph_invocations(n_concurrent: int = 3) -> dict:
    """Measure wall-clock time for multiple concurrent graph invocations."""
    questions = [
        "What are the parking prices?",
        "What are the working hours?",
        "Where is the nearest parking?",
        "Do you have EV charging?",
        "How can I reserve?",
    ]

    errors = 0

    def _invoke_once(i: int) -> None:
        thread_config = cast(RunnableConfig, get_thread_config(f"load-concurrent-{i}-{datetime.now().timestamp()}"))
        state: ChatState = {
            "messages": [],
            "user_input": questions[i % len(questions)],
            "intent": "",
            "reservation_data": {},
            "guardrail_triggered": False,
            "response": "",
            "admin_decision": "",
            "awaiting_admin": False,
        }
        chatbot_graph.invoke(state, config=thread_config)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
        futures = [executor.submit(_invoke_once, i) for i in range(n_concurrent)]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                errors += 1
    total_wall_time_ms = (time.perf_counter() - start) * 1000

    avg_per_request_ms = total_wall_time_ms / n_concurrent if n_concurrent else 0.0

    return {
        "component": "Concurrent Graph Invocations",
        "n_concurrent": n_concurrent,
        "total_wall_time_ms": round(total_wall_time_ms, 2),
        "avg_per_request_ms": round(avg_per_request_ms, 2),
        "errors": errors,
    }


def generate_load_test_report(results: list[dict]) -> str:
    """Generate markdown report for load-test results."""
    timestamp = datetime.now().isoformat()

    lines: list[str] = []
    lines.append("# Load Test Report")
    lines.append("")
    lines.append(f"Timestamp: {timestamp}")
    lines.append("")
    lines.append("| Component | Requests | Avg (ms) | P95 (ms) | Min (ms) | Max (ms) | Errors |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for result in results:
        component = result.get("component", "Unknown")
        requests = result.get("n_requests", result.get("n_concurrent", 0))
        avg_ms = result.get("avg_ms", result.get("avg_per_request_ms", 0.0))
        p95_ms = result.get("p95_ms", "-")
        min_ms = result.get("min_ms", "-")
        max_ms = result.get("max_ms", result.get("total_wall_time_ms", "-"))
        errors = result.get("errors", 0)

        lines.append(
            f"| {component} | {requests} | {avg_ms} | {p95_ms} | {min_ms} | {max_ms} | {errors} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    for result in results:
        component = result.get("component", "Unknown")
        avg_ms = float(result.get("avg_ms", result.get("avg_per_request_ms", 0.0)))
        errors = int(result.get("errors", 0))

        if errors > 0:
            status = "❌ Errors detected"
        elif avg_ms > 5000:
            status = "⚠️ Slow"
        else:
            status = "✅ OK"

        lines.append(f"- {component}: {status}")

    lines.append("")
    lines.append("## Recommendations")
    lines.append("")

    any_errors = any(int(result.get("errors", 0)) > 0 for result in results)
    any_slow = any(float(result.get("avg_ms", result.get("avg_per_request_ms", 0.0))) > 5000 for result in results)

    if any_errors:
        lines.append("- Investigate failing component logs first; prioritize fixing request errors before latency tuning.")
    if any_slow:
        lines.append("- Optimize slow components by reducing prompt size, retrieval depth, or external call overhead.")
    if not any_errors and not any_slow:
        lines.append("- Performance is within current thresholds; continue periodic load testing as data volume grows.")

    lines.append("- Re-run load testing after major prompt, retrieval, or orchestration changes.")

    return "\n".join(lines)


def main():
    print("=== Parking Chatbot Load Test ===\n")

    try:
        weaviate_client = get_weaviate_client()
        weaviate_ok = health_check(weaviate_client)
        weaviate_client.close()
    except Exception:
        weaviate_ok = False

    if not weaviate_ok:
        print("Error: Weaviate is not reachable. Start Weaviate and retry.")
        sys.exit(1)

    if not os.getenv("MCP_API_KEY"):
        print("Error: MCP_API_KEY is not set.")
        sys.exit(1)

    n_requests = int(os.getenv("LOAD_TEST_N_REQUESTS", "10"))
    print(f"Running {n_requests} requests per component...\n")

    results = []

    print("Testing RAG query latency...")
    results.append(test_rag_latency(n_requests))
    print(f"  Done. Avg: {results[-1]['avg_ms']}ms\n")

    print("Testing MCP write latency...")
    results.append(test_mcp_write_latency(n_requests))
    print(f"  Done. Avg: {results[-1]['avg_ms']}ms\n")

    print("Testing full graph invocation latency...")
    results.append(test_graph_latency(min(n_requests, 5)))
    print(f"  Done. Avg: {results[-1]['avg_ms']}ms\n")

    print("Testing concurrent graph invocations...")
    results.append(test_concurrent_graph_invocations(3))
    print("  Done.\n")

    report = generate_load_test_report(results)

    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/load_test_report_{timestamp}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\nReport saved to: {report_path}")

    total_errors = sum(int(r.get("errors", 0)) for r in results)
    sys.exit(1 if total_errors > 0 else 0)


if __name__ == "__main__":
    main()
