"""
================================================================================
   HALCYON CREDIT — Decision Record Tool (SQLite Persistence)
   Stage 3 | Author: Harshit
   Persists DecisionRecord to SQLite (dev) or Postgres (prod).
   Records are immutable after write — no UPDATE operations.
================================================================================
"""
from __future__ import annotations
import os
import json
import sqlite3
from datetime import datetime
from typing import Optional


DB_URL = os.getenv("DATABASE_URL", "sqlite:///./halcyon_decisions.db")
_DB_PATH = DB_URL.replace("sqlite:///", "")


def _get_connection() -> sqlite3.Connection:
    """Get a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table(conn: sqlite3.Connection) -> None:
    """Create decision_records table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decision_records (
            audit_id            TEXT PRIMARY KEY,
            application_id      TEXT NOT NULL,
            recommendation      TEXT NOT NULL,
            risk_score          REAL,
            risk_band           TEXT,
            faithfulness        REAL,
            retry_count         INTEGER DEFAULT 0,
            cost_usd            REAL DEFAULT 0.0,
            full_state_trace    TEXT,
            escalated           INTEGER DEFAULT 0,
            created_at          TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_application_id
        ON decision_records (application_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_created_at
        ON decision_records (created_at)
    """)
    conn.commit()


def persist_record(record) -> None:
    """
    Persist a DecisionRecord to the database.
    Immutable — INSERT only, no UPDATE on existing audit_ids.

    Args:
        record: DecisionRecord instance
    """
    conn = _get_connection()
    try:
        _ensure_table(conn)

        # Serialize the full trace as JSON
        trace_json = json.dumps([
            {
                "agent":      e.agent,
                "action":     e.action,
                "timestamp":  e.timestamp,
                "retry":      e.retry,
                "latency_ms": e.latency_ms,
                "cost_usd":   e.cost_usd,
            }
            for e in (record.full_trace or [])
        ])

        conn.execute("""
            INSERT OR IGNORE INTO decision_records
            (audit_id, application_id, recommendation, risk_score, risk_band,
             faithfulness, retry_count, cost_usd, full_state_trace, escalated, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.audit_id,
            record.application_id,
            record.final_decision.recommendation if record.final_decision else "UNKNOWN",
            record.risk_result.risk_score if record.risk_result else None,
            record.risk_result.risk_band  if record.risk_result else None,
            record.eval_result.faithfulness if record.eval_result else None,
            len(record.full_trace or []),
            record.cost_usd_total,
            trace_json,
            1 if record.escalated else 0,
            record.created_at,
        ))
        conn.commit()
    finally:
        conn.close()


def fetch_record(audit_id: str) -> Optional[dict]:
    """
    Retrieve a decision record by audit_id.

    Returns:
        dict with all record fields, or None if not found
    """
    conn = _get_connection()
    try:
        _ensure_table(conn)
        row = conn.execute(
            "SELECT * FROM decision_records WHERE audit_id = ?", (audit_id,)
        ).fetchone()

        if row is None:
            return None

        result = dict(row)
        if result.get("full_state_trace"):
            result["full_state_trace"] = json.loads(result["full_state_trace"])
        return result
    finally:
        conn.close()


def list_records(limit: int = 50) -> list[dict]:
    """
    List recent decision records ordered by creation date desc.

    Returns:
        List of dicts with record fields (without full trace)
    """
    conn = _get_connection()
    try:
        _ensure_table(conn)
        rows = conn.execute("""
            SELECT audit_id, application_id, recommendation, risk_score,
                   risk_band, faithfulness, retry_count, cost_usd,
                   escalated, created_at
            FROM decision_records
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
