"""Quick integration test for the observability module (no LLM calls)."""
import os
os.environ["METRICS_DB"] = ":memory:"

import time
from src.observability import (
    trace_context, record_latency, record_cost, record_quality,
    get_latency_percentiles, get_cost_summary, get_quality_summary,
    run_regression_gate
)

def test_latency_and_traces():
    for i in range(5):
        with trace_context("rag_pipeline", {"user": "test", "q": f"q{i}"}) as span:
            record_latency("rag_e2e", 0.5 + i * 0.1)
            record_latency("retrieval", 0.2 + i * 0.05)
            record_latency("llm_generate", 0.3 + i * 0.08)
            record_cost("llama-3.1-8b-instant", 500 + i * 50, 200 + i * 20, user="test")

    pct = get_latency_percentiles("rag_e2e", window_hours=1)
    assert pct["count"] == 5, f"Expected 5 latency samples, got {pct['count']}"
    assert pct["p50"] > 0, "p50 should be > 0"
    assert pct["p95"] >= pct["p50"], "p95 >= p50"
    print(f"Latency -> p50={pct['p50']}s, p95={pct['p95']}s, count={pct['count']}")


def test_cost_tracking():
    cost = get_cost_summary(window_hours=1)
    assert cost["total_requests"] > 0, "Should have recorded cost entries"
    assert cost["total_cost_usd"] > 0, "Total cost should be > 0"
    print(f"Cost -> ${cost['total_cost_usd']:.6f} across {cost['total_requests']} requests")


def test_quality_metrics():
    record_quality(
        question="What is RAG?",
        faithfulness=0.82,
        answer_relevancy=0.78,
        context_precision=0.81,
        num_sources=3,
    )
    q = get_quality_summary(window_hours=1)
    assert q["count"] >= 1
    assert 0.8 <= q["faithfulness"] <= 0.9
    print(f"Quality -> faithfulness={q['faithfulness']}, count={q['count']}")


def test_regression_gate_pass():
    gate = run_regression_gate({
        "faithfulness": 0.82,
        "answer_relevancy": 0.78,
        "context_precision": 0.81,
        "p95_latency_s": 3.2,
    })
    assert gate["passed"], f"Gate should pass, got: {gate}"
    print("Gate PASSED as expected")


def test_regression_gate_fail():
    gate = run_regression_gate({
        "faithfulness": 0.50,  # below 0.75 threshold
        "answer_relevancy": 0.90,
        "context_precision": 0.90,
        "p95_latency_s": 3.2,
    })
    assert not gate["passed"], "Gate should fail when faithfulness < threshold"
    print("Gate correctly FAILED on low faithfulness")


if __name__ == "__main__":
    test_latency_and_traces()
    test_cost_tracking()
    test_quality_metrics()
    test_regression_gate_pass()
    test_regression_gate_fail()
    print("\nAll observability tests passed!")
