"""
Observability & Tracing for the Enterprise RAG System
======================================================
Provides:
  - Span-based distributed tracing (no external dependency required)
  - Latency recording (p50 / p95 percentiles)
  - Cost-per-request estimation based on token usage
  - Quality metric storage (faithfulness, relevancy, etc.)
  - Thread-safe SQLite persistence via MetricsStore
"""

from __future__ import annotations

import os
import time
import uuid
import threading
import functools
import statistics
import contextlib
from typing import Optional, Dict, Any, List, Callable

from src.metrics_store import MetricsStore

# ---------------------------------------------------------------------------
# Groq token pricing (USD per 1 000 tokens, as of 2025-Q1)
# Adjust if you switch models.
# ---------------------------------------------------------------------------
_COST_TABLE: Dict[str, Dict[str, float]] = {
    "llama-3.1-8b-instant":   {"input": 0.00005,  "output": 0.00008},
    "llama-3.3-70b-versatile":{"input": 0.00059,  "output": 0.00079},
    "mixtral-8x7b-32768":     {"input": 0.00027,  "output": 0.00027},
    "gemma2-9b-it":            {"input": 0.00020,  "output": 0.00020},
}
_DEFAULT_COST = {"input": 0.00005, "output": 0.00008}

# ---------------------------------------------------------------------------
# Module-level store (singleton, lazy-init)
# ---------------------------------------------------------------------------
_store: Optional[MetricsStore] = None
_store_lock = threading.Lock()


def get_store() -> MetricsStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = MetricsStore()
    return _store


# ---------------------------------------------------------------------------
# Span / Trace context
# ---------------------------------------------------------------------------

class Span:
    """
    A lightweight tracing span.  Create via :func:`start_span` or use the
    :func:`trace` decorator / :func:`trace_context` context-manager.
    """

    def __init__(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.span_id   = uuid.uuid4().hex
        self.trace_id  = trace_id or uuid.uuid4().hex
        self.parent_id = parent_id
        self.name      = name
        self.metadata  = metadata or {}
        self.start_ts  = time.perf_counter()
        self._wall_start = time.time()
        self._ended    = False

    # -- context-manager support --------------------------------------------

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "error" if exc_type else "ok"
        self.end(status=status)
        return False          # don't suppress exceptions

    # -- public API ---------------------------------------------------------

    def set_metadata(self, **kwargs):
        self.metadata.update(kwargs)

    def end(
        self,
        *,
        status: str = "ok",
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Finish the span, persist to DB, return latency in seconds."""
        if self._ended:
            return 0.0
        self._ended = True

        latency_s = time.perf_counter() - self.start_ts
        cost_usd  = _estimate_cost(model or "", input_tokens, output_tokens)

        meta = dict(self.metadata)
        if extra:
            meta.update(extra)

        get_store().record_span(
            span_id       = self.span_id,
            trace_id      = self.trace_id,
            parent_id     = self.parent_id,
            name          = self.name,
            status        = status,
            latency_s     = latency_s,
            input_tokens  = input_tokens,
            output_tokens = output_tokens,
            cost_usd      = cost_usd,
            metadata      = meta,
            wall_time     = self._wall_start,
        )
        return latency_s


# ---------------------------------------------------------------------------
# Thread-local active trace / span
# ---------------------------------------------------------------------------

_local = threading.local()


def current_trace_id() -> Optional[str]:
    return getattr(_local, "trace_id", None)


def current_span_id() -> Optional[str]:
    return getattr(_local, "span_id", None)


@contextlib.contextmanager
def trace_context(name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Context-manager that creates a root span and sets the thread-local
    trace_id so nested calls can attach as children.

    Usage::

        with trace_context("rag_query", {"user": user}) as span:
            docs = retrieve(...)
            span.set_metadata(num_docs=len(docs))
    """
    parent_id = current_span_id()
    trace_id  = current_trace_id() or uuid.uuid4().hex

    span = Span(name, trace_id=trace_id, parent_id=parent_id, metadata=metadata)
    prev_span_id  = getattr(_local, "span_id",  None)
    prev_trace_id = getattr(_local, "trace_id", None)

    _local.span_id  = span.span_id
    _local.trace_id = trace_id

    try:
        yield span
    finally:
        span.end()
        _local.span_id  = prev_span_id
        _local.trace_id = prev_trace_id


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def trace(
    name: Optional[str] = None,
    capture_args: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Decorator that wraps a function in a tracing span.

    Usage::

        @trace("llm_generate")
        def generate_answer(...):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            meta = dict(metadata or {})
            if capture_args:
                meta["args"]   = str(args)[:200]
                meta["kwargs"] = str(kwargs)[:200]

            with trace_context(span_name, meta):
                return fn(*args, **kwargs)

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Latency helpers
# ---------------------------------------------------------------------------

def record_latency(operation: str, latency_s: float, user: Optional[str] = None):
    """Persist a standalone latency observation (not tied to a span)."""
    get_store().record_latency(operation=operation, latency_s=latency_s, user=user)


def get_latency_percentiles(
    operation: str,
    window_hours: int = 24,
) -> Dict[str, float]:
    """Return p50, p95, p99, mean, count for *operation* over the last window."""
    samples = get_store().get_latency_samples(operation, window_hours)
    if not samples:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "count": 0}
    s = sorted(samples)
    n = len(s)

    def pct(p):
        idx = int(p / 100 * n)
        return s[min(idx, n - 1)]

    return {
        "p50":   round(pct(50),  3),
        "p95":   round(pct(95),  3),
        "p99":   round(pct(99),  3),
        "mean":  round(statistics.mean(s), 3),
        "count": n,
    }


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = _COST_TABLE.get(model, _DEFAULT_COST)
    return (
        input_tokens  / 1000 * rates["input"]
        + output_tokens / 1000 * rates["output"]
    )


def record_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    user: Optional[str] = None,
    trace_id: Optional[str] = None,
):
    """Persist a cost record independently of a span."""
    cost = _estimate_cost(model, input_tokens, output_tokens)
    get_store().record_cost(
        model         = model,
        input_tokens  = input_tokens,
        output_tokens = output_tokens,
        cost_usd      = cost,
        user          = user,
        trace_id      = trace_id or current_trace_id(),
    )
    return cost


def get_cost_summary(window_hours: int = 24) -> Dict[str, Any]:
    """Aggregate cost stats over the last *window_hours*."""
    return get_store().get_cost_summary(window_hours)


def get_user_usage(user: str) -> Dict[str, Any]:
    """Return total tokens and cost_usd for a given user."""
    return get_store().get_user_usage(user)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def record_quality(
    *,
    trace_id: Optional[str] = None,
    user: Optional[str] = None,
    question: str,
    faithfulness: Optional[float] = None,
    answer_relevancy: Optional[float] = None,
    context_precision: Optional[float] = None,
    context_recall: Optional[float] = None,
    latency_s: Optional[float] = None,
    model: Optional[str] = None,
    num_sources: int = 0,
):
    """Persist quality evaluation scores for a single query."""
    get_store().record_quality(
        trace_id          = trace_id or current_trace_id() or uuid.uuid4().hex,
        user              = user,
        question          = question[:300],
        faithfulness      = faithfulness,
        answer_relevancy  = answer_relevancy,
        context_precision = context_precision,
        context_recall    = context_recall,
        latency_s         = latency_s,
        model             = model,
        num_sources       = num_sources,
    )


def get_quality_summary(window_hours: int = 168) -> Dict[str, Any]:
    """Rolling averages of quality metrics."""
    return get_store().get_quality_summary(window_hours)


# ---------------------------------------------------------------------------
# Regression gate (used by CI)
# ---------------------------------------------------------------------------

# Metrics where HIGHER is better (quality scores)
_HIGHER_IS_BETTER = {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
# Metrics where LOWER is better (latency)
_LOWER_IS_BETTER  = {"p95_latency_s", "p50_latency_s", "mean_latency_s"}

QUALITY_THRESHOLDS = {
    "faithfulness":      float(os.getenv("GATE_FAITHFULNESS",      "0.75")),
    "answer_relevancy":  float(os.getenv("GATE_ANSWER_RELEVANCY",  "0.75")),
    "context_precision": float(os.getenv("GATE_CONTEXT_PRECISION", "0.70")),
    "p95_latency_s":     float(os.getenv("GATE_P95_LATENCY_S",     "15.0")),
}


def run_regression_gate(results: Dict[str, float]) -> Dict[str, Any]:
    """
    Check *results* against QUALITY_THRESHOLDS.

    - Quality metrics (faithfulness etc.): pass when value >= threshold
    - Latency metrics (p95_latency_s etc.): pass when value <= threshold

    Returns a report dict.  Exit code guidance for CI:
      passed=True  → all checks green
      passed=False → at least one check failed (block merge)
    """
    passed = True
    report: List[Dict] = []
    for metric, threshold in QUALITY_THRESHOLDS.items():
        value = results.get(metric)
        if value is None:
            continue
        # Latency: lower is better → pass when value <= threshold
        # Quality: higher is better → pass when value >= threshold
        if metric in _LOWER_IS_BETTER:
            ok = value <= threshold
        else:
            ok = value >= threshold
        if not ok:
            passed = False
        report.append({
            "metric":    metric,
            "value":     round(value, 4),
            "threshold": threshold,
            "passed":    ok,
            "direction": "≤" if metric in _LOWER_IS_BETTER else "≥",
        })
    return {"passed": passed, "checks": report}
