"""
Audit & feedback logging for Enterprise AI Assistant.
All queries, answers, and user feedback are persisted to CSV files.
"""
import csv
import os
import datetime
from src.config import AUDIT_CSV, FEEDBACK_CSV

_AUDIT_HEADERS   = ["timestamp", "user", "question", "answer_snippet", "num_sources", "session_id"]
_FEEDBACK_HEADERS = ["timestamp", "user", "question", "answer_snippet", "rating", "comment"]


def _ensure_csv(path: str, headers: list):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)


def log_query(user: str, question: str, answer: str, num_sources: int, session_id: str = ""):
    _ensure_csv(AUDIT_CSV, _AUDIT_HEADERS)
    with open(AUDIT_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.datetime.now().isoformat(),
            user,
            question[:200],
            answer[:300],
            num_sources,
            session_id,
        ])


def log_feedback(user: str, question: str, answer: str, rating: str, comment: str = ""):
    _ensure_csv(FEEDBACK_CSV, _FEEDBACK_HEADERS)
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.datetime.now().isoformat(),
            user,
            question[:200],
            answer[:300],
            rating,
            comment,
        ])


def read_audit_log() -> list:
    if not os.path.exists(AUDIT_CSV):
        return []
    with open(AUDIT_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_feedback_log() -> list:
    if not os.path.exists(FEEDBACK_CSV):
        return []
    with open(FEEDBACK_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))
