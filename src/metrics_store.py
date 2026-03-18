"""
SQLite-backed metrics store for the RAG observability system.
All tables live in a single file: ./metrics.db  (configurable via METRICS_DB env).
"""

from __future__ import annotations

import os
import json
import time
import threading
import sqlite3
from typing import Any, Dict, List, Optional


DB_PATH = os.getenv("METRICS_DB", "./metrics.db")


class MetricsStore:
    """Thread-safe SQLite wrapper for all telemetry data."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._local  = threading.local()
        self._init_db()

    # ── Connection (per-thread) ──────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
            self._apply_schema(conn)
        return self._local.conn

    # ── Schema ──────────────────────────────────────────────────────────

    def _init_db(self):
        """Trigger schema creation on the calling thread's connection."""
        self._conn()  # creates connection + applies schema on first call

    def _apply_schema(self, conn: sqlite3.Connection):
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS spans (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            span_id       TEXT NOT NULL,
            trace_id      TEXT NOT NULL,
            parent_id     TEXT,
            name          TEXT NOT NULL,
            status        TEXT DEFAULT 'ok',
            latency_s     REAL,
            input_tokens  INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cost_usd      REAL DEFAULT 0,
            metadata      TEXT,
            wall_time     REAL
        );

        CREATE TABLE IF NOT EXISTS latency_samples (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            operation   TEXT NOT NULL,
            latency_s   REAL NOT NULL,
            user        TEXT,
            recorded_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cost_records (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            model         TEXT NOT NULL,
            input_tokens  INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cost_usd      REAL DEFAULT 0,
            user          TEXT,
            trace_id      TEXT,
            recorded_at   REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS quality_scores (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id          TEXT,
            user              TEXT,
            question          TEXT,
            faithfulness      REAL,
            answer_relevancy  REAL,
            context_precision REAL,
            context_recall    REAL,
            latency_s         REAL,
            model             TEXT,
            num_sources       INTEGER DEFAULT 0,
            recorded_at       REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_spans_trace   ON spans(trace_id);
        CREATE INDEX IF NOT EXISTS idx_latency_op    ON latency_samples(operation, recorded_at);
        CREATE INDEX IF NOT EXISTS idx_cost_time     ON cost_records(recorded_at);
        CREATE INDEX IF NOT EXISTS idx_quality_time  ON quality_scores(recorded_at);
        """)
        conn.commit()


    # ── Span ────────────────────────────────────────────────────────────

    def record_span(
        self,
        *,
        span_id: str,
        trace_id: str,
        parent_id: Optional[str],
        name: str,
        status: str,
        latency_s: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        metadata: Dict[str, Any],
        wall_time: float,
    ):
        conn = self._conn()
        conn.execute(
            """INSERT INTO spans
               (span_id,trace_id,parent_id,name,status,latency_s,
                input_tokens,output_tokens,cost_usd,metadata,wall_time)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                span_id, trace_id, parent_id, name, status, latency_s,
                input_tokens, output_tokens, cost_usd,
                json.dumps(metadata), wall_time,
            ),
        )
        conn.commit()

    def get_spans(self, limit: int = 200) -> List[Dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM spans ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_trace(self, trace_id: str) -> List[Dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM spans WHERE trace_id=? ORDER BY id", (trace_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Latency ─────────────────────────────────────────────────────────

    def record_latency(self, *, operation: str, latency_s: float, user: Optional[str]):
        conn = self._conn()
        conn.execute(
            "INSERT INTO latency_samples(operation,latency_s,user,recorded_at)"
            " VALUES(?,?,?,?)",
            (operation, latency_s, user, time.time()),
        )
        conn.commit()

    def get_latency_samples(self, operation: str, window_hours: int) -> List[float]:
        cutoff = time.time() - window_hours * 3600
        conn   = self._conn()
        rows   = conn.execute(
            "SELECT latency_s FROM latency_samples"
            " WHERE operation=? AND recorded_at>=?",
            (operation, cutoff),
        ).fetchall()
        return [r[0] for r in rows]

    def get_all_latency_stats(self, window_hours: int = 24) -> List[Dict]:
        """Return per-operation latency stats within window."""
        cutoff = time.time() - window_hours * 3600
        conn   = self._conn()
        rows   = conn.execute(
            "SELECT operation, latency_s FROM latency_samples WHERE recorded_at>=?",
            (cutoff,),
        ).fetchall()

        import statistics as st
        ops: Dict[str, List[float]] = {}
        for op, lat in rows:
            ops.setdefault(op, []).append(lat)

        results = []
        for op, samples in ops.items():
            s = sorted(samples)
            n = len(s)
            def p(pct):
                idx = int(pct / 100 * n)
                return round(s[min(idx, n - 1)], 3)
            results.append({
                "operation": op,
                "count": n,
                "mean":  round(st.mean(s), 3),
                "p50":   p(50),
                "p95":   p(95),
                "p99":   p(99),
            })
        return results

    # ── Cost ────────────────────────────────────────────────────────────

    def record_cost(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        user: Optional[str],
        trace_id: Optional[str],
    ):
        conn = self._conn()
        conn.execute(
            """INSERT INTO cost_records
               (model,input_tokens,output_tokens,cost_usd,user,trace_id,recorded_at)
               VALUES(?,?,?,?,?,?,?)""",
            (model, input_tokens, output_tokens, cost_usd, user, trace_id, time.time()),
        )
        conn.commit()

    def get_cost_summary(self, window_hours: int = 24) -> Dict[str, Any]:
        cutoff = time.time() - window_hours * 3600
        conn   = self._conn()
        rows = conn.execute(
            """SELECT count(*) as cnt,
                      sum(cost_usd)      as total_cost,
                      sum(input_tokens)  as total_input,
                      sum(output_tokens) as total_output
               FROM cost_records WHERE recorded_at>=?""",
            (cutoff,),
        ).fetchone()
        by_model = conn.execute(
            """SELECT model, count(*) as cnt, sum(cost_usd) as cost
               FROM cost_records WHERE recorded_at>=? GROUP BY model""",
            (cutoff,),
        ).fetchall()
        return {
            "window_hours":  window_hours,
            "total_requests": rows["cnt"] or 0,
            "total_cost_usd": round(rows["total_cost"] or 0, 6),
            "total_input_tokens":  rows["total_input"] or 0,
            "total_output_tokens": rows["total_output"] or 0,
            "by_model": [dict(r) for r in by_model],
        }

    def get_user_usage(self, user: str) -> Dict[str, Any]:
        """Return total tokens and cost for a specific user lifetime."""
        conn = self._conn()
        row = conn.execute(
            """SELECT sum(input_tokens + output_tokens) as total_tokens,
                      sum(cost_usd) as total_cost
               FROM cost_records WHERE user=?""",
            (user,),
        ).fetchone()
        return {
            "tokens": row["total_tokens"] or 0,
            "cost_usd": row["total_cost"] or 0.0,
        }

    def get_cost_timeseries(self, window_hours: int = 24, bucket_minutes: int = 60) -> List[Dict]:
        """Hourly cost buckets for the given window."""
        cutoff     = time.time() - window_hours * 3600
        bucket_s   = bucket_minutes * 60
        conn       = self._conn()
        rows = conn.execute(
            """SELECT CAST(recorded_at / ? AS INTEGER) * ? AS bucket,
                      sum(cost_usd) as cost, count(*) as cnt
               FROM cost_records WHERE recorded_at>=?
               GROUP BY bucket ORDER BY bucket""",
            (bucket_s, bucket_s, cutoff),
        ).fetchall()
        return [{"bucket_ts": r["bucket"], "cost": round(r["cost"], 6), "requests": r["cnt"]} for r in rows]

    # ── Quality ─────────────────────────────────────────────────────────

    def record_quality(
        self,
        *,
        trace_id: str,
        user: Optional[str],
        question: str,
        faithfulness: Optional[float],
        answer_relevancy: Optional[float],
        context_precision: Optional[float],
        context_recall: Optional[float],
        latency_s: Optional[float],
        model: Optional[str],
        num_sources: int,
    ):
        conn = self._conn()
        conn.execute(
            """INSERT INTO quality_scores
               (trace_id,user,question,faithfulness,answer_relevancy,
                context_precision,context_recall,latency_s,model,num_sources,recorded_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                trace_id, user, question,
                faithfulness, answer_relevancy, context_precision, context_recall,
                latency_s, model, num_sources, time.time(),
            ),
        )
        conn.commit()

    def get_quality_rows(self, window_hours: int = 168, limit: int = 200) -> List[Dict]:
        cutoff = time.time() - window_hours * 3600
        conn   = self._conn()
        rows = conn.execute(
            "SELECT * FROM quality_scores WHERE recorded_at>=? ORDER BY id DESC LIMIT ?",
            (cutoff, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_quality_summary(self, window_hours: int = 168) -> Dict[str, Any]:
        cutoff = time.time() - window_hours * 3600
        conn   = self._conn()
        row = conn.execute(
            """SELECT count(*) as cnt,
                      avg(faithfulness)      as avg_faith,
                      avg(answer_relevancy)  as avg_rel,
                      avg(context_precision) as avg_prec,
                      avg(context_recall)    as avg_recall,
                      avg(latency_s)         as avg_lat
               FROM quality_scores WHERE recorded_at>=?""",
            (cutoff,),
        ).fetchone()
        return {
            "window_hours":       window_hours,
            "count":              row["cnt"] or 0,
            "faithfulness":       round(row["avg_faith"]  or 0, 4),
            "answer_relevancy":   round(row["avg_rel"]    or 0, 4),
            "context_precision":  round(row["avg_prec"]   or 0, 4),
            "context_recall":     round(row["avg_recall"] or 0, 4),
            "avg_latency_s":      round(row["avg_lat"]    or 0, 3),
        }

    # ── Recent trace overview ────────────────────────────────────────────

    def get_recent_traces(self, limit: int = 100) -> List[Dict]:
        """Root-level spans (parent_id IS NULL), most recent first."""
        conn = self._conn()
        rows = conn.execute(
            """SELECT s.*, 
                      (SELECT sum(s2.cost_usd) FROM spans s2 WHERE s2.trace_id=s.trace_id) as trace_cost,
                      (SELECT sum(s2.latency_s) FROM spans s2 WHERE s2.trace_id=s.trace_id) as trace_latency
               FROM spans s WHERE s.parent_id IS NULL
               ORDER BY s.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
