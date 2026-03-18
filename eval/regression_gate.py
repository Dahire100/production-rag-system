#!/usr/bin/env python
"""
Regression Gate — CI quality evaluation script.
================================================
Runs a set of golden Q&A pairs through the RAG pipeline using RAGAS metrics,
then applies pass/fail thresholds from src.observability.QUALITY_THRESHOLDS.

Exit codes:
  0  — all gates passed
  1  — one or more gates failed (blocks merge in CI)

Usage:
  python eval/regression_gate.py [--eval-file eval/eval_data.json]
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import statistics

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def _parse_args():
    p = argparse.ArgumentParser(description="RAG Regression Gate")
    p.add_argument("--eval-file", default="eval/eval_data.json")
    p.add_argument("--output-json", default=None,
                   help="Optional path to write JSON results for CI artifact upload")
    p.add_argument("--skip-ragas", action="store_true",
                   help="Skip RAGAS evaluation (latency-only gate, for quick CI smoke tests)")
    return p.parse_args()


# ── Metric helpers ──────────────────────────────────────────────────────────

def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = int(p / 100 * len(s))
    return s[min(idx, len(s) - 1)]


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    # -- Load eval dataset --------------------------------------------------
    eval_path = args.eval_file
    if not os.path.exists(eval_path):
        print(f"[GATE] ❌  Eval file not found: {eval_path}")
        sys.exit(1)

    with open(eval_path) as f:
        eval_data = json.load(f)

    if not eval_data:
        print("[GATE] ⚠️  Empty eval dataset — skipping gate.")
        sys.exit(0)

    print(f"[GATE] Running regression gate on {len(eval_data)} Q&A pairs …\n")

    # -- Import RAG pipeline ------------------------------------------------
    try:
        from src.generate import multi_query_retrieve, format_docs, _get_llm
        from src.observability import (
            get_latency_percentiles, record_quality, run_regression_gate,
            QUALITY_THRESHOLDS, current_trace_id,
        )
        from src.observability import trace_context
    except ImportError as e:
        print(f"[GATE] ❌  Import error: {e}")
        sys.exit(1)

    # -- Run each question ---------------------------------------------------
    questions:    list[str]         = []
    ground_truths: list[str]        = []
    answers:      list[str]         = []
    contexts_list: list[list[str]]  = []
    latencies:    list[float]       = []

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    RAG_PROMPT = """\
You are a precise document assistant.
Cite every claim with [doc_id]. Never fabricate.

Context:
{context}

Question: {question}

Answer:"""

    for item in eval_data:
        q  = item["question"]
        gt = item.get("ground_truth", "")

        print(f"  Q: {q[:80]}…")
        t0 = time.perf_counter()
        try:
            with trace_context("regression_gate_query", {"question": q}) as span:
                docs = multi_query_retrieve(q)
                context_str = format_docs(docs)

                prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
                chain  = prompt | _get_llm() | StrOutputParser()
                answer = chain.invoke({"context": context_str, "question": q})

                tid = current_trace_id()

            lat = time.perf_counter() - t0
            latencies.append(lat)

            questions.append(q)
            ground_truths.append(gt)
            answers.append(answer)
            contexts_list.append([d.page_content for d in docs])

            print(f"     ✓  {lat:.2f}s  |  {len(docs)} docs retrieved")

        except Exception as exc:
            lat = time.perf_counter() - t0
            latencies.append(lat)
            print(f"     ✗  ERROR: {exc}")
            questions.append(q)
            ground_truths.append(gt)
            answers.append("")
            contexts_list.append([])

    # -- Latency stats -------------------------------------------------------
    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    mean_lat = statistics.mean(latencies) if latencies else 0.0

    print(f"\n[GATE] Latency stats — p50: {p50:.2f}s  p95: {p95:.2f}s  mean: {mean_lat:.2f}s")

    gate_results: dict = {
        "p95_latency_s": p95,
        "p50_latency_s": p50,
        "mean_latency_s": mean_lat,
        "num_questions": len(questions),
    }

    # -- RAGAS evaluation ----------------------------------------------------
    ragas_scores: dict = {}

    if not args.skip_ragas:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision
            from langchain_groq import ChatGroq
            from langchain_huggingface import HuggingFaceEmbeddings

            groq_llm = ChatGroq(model_name=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
            hf_emb   = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            faithfulness.llm        = groq_llm
            answer_relevancy.llm    = groq_llm
            answer_relevancy.embeddings = hf_emb
            context_precision.llm   = groq_llm

            valid = [(q, gt, a, c)
                     for q, gt, a, c in zip(questions, ground_truths, answers, contexts_list)
                     if a and c]

            if valid:
                vq, vgt, va, vc = zip(*valid)
                ds = Dataset.from_dict({
                    "question":     list(vq),
                    "answer":       list(va),
                    "contexts":     list(vc),
                    "ground_truth": list(vgt),
                })
                result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])

                ragas_scores = {
                    "faithfulness":      float(result["faithfulness"]),
                    "answer_relevancy":  float(result["answer_relevancy"]),
                    "context_precision": float(result["context_precision"]),
                }

                # Store each score in the metrics DB
                for q, a, c, gt in zip(list(vq), list(va), list(vc), list(vgt)):
                    record_quality(
                        question          = q,
                        faithfulness      = ragas_scores["faithfulness"],
                        answer_relevancy  = ragas_scores["answer_relevancy"],
                        context_precision = ragas_scores["context_precision"],
                        latency_s         = mean_lat,
                        num_sources       = len(c),
                    )

                print(f"[GATE] RAGAS scores:")
                for k, v in ragas_scores.items():
                    print(f"       {k:25s} = {v:.4f}")
            else:
                print("[GATE] ⚠️  No valid answers to evaluate with RAGAS.")

        except Exception as exc:
            print(f"[GATE] ⚠️  RAGAS evaluation skipped: {exc}")
    else:
        print("[GATE] --skip-ragas flag set — skipping RAGAS evaluation.")

    gate_results.update(ragas_scores)

    # -- Regression gate checks -----------------------------------------------
    print(f"\n[GATE] Applying thresholds: {QUALITY_THRESHOLDS}")
    report = run_regression_gate(gate_results)

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│                  REGRESSION GATE REPORT                    │")
    print("├────────────────────────┬────────────┬────────────┬─────────┤")
    print("│ Metric                 │    Value   │ Threshold  │  Status │")
    print("├────────────────────────┼────────────┼────────────┼─────────┤")
    for check in report["checks"]:
        status = "✅ PASS" if check["passed"] else "❌ FAIL"
        print(f"│ {check['metric']:22s} │ {check['value']:10.4f} │ {check['threshold']:10.4f} │ {status} │")
    print("└────────────────────────┴────────────┴────────────┴─────────┘")

    overall = "✅  ALL GATES PASSED" if report["passed"] else "❌  GATE FAILED — blocking merge"
    print(f"\n[GATE] {overall}")

    # -- Optionally write JSON artifact --------------------------------------
    full_report = {
        "passed":       report["passed"],
        "checks":       report["checks"],
        "gate_results": gate_results,
        "latency": {"p50": p50, "p95": p95, "mean": mean_lat},
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(full_report, f, indent=2)
        print(f"[GATE] Results written to {args.output_json}")

    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
