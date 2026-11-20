# -----------------------------------------------------------------------------
# storage.py
# Laboratory: Lab1, NLP Coursework
# Date: 2025-11-19
#
# This module implements a lightweight persistence layer for the feedback
# analysis system using SQLite. It stores:
#   - Raw feedback entries (review-level)
#   - Aspect-level sentiment opinions associated with each feedback
#
# The design focuses on:
#   - Simplicity and reproducibility (single-file SQLite database)
#   - Safe schema initialization and reset
#   - Minimal, clear CRUD operations for experiments and evaluation
# -----------------------------------------------------------------------------

import sqlite3, os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

# -----------------------------------------------------------------------------
# Database configuration
# -----------------------------------------------------------------------------
# Database file path can be overridden via environment variable FEEDBACK_DB_PATH.
DB_PATH = os.environ.get("FEEDBACK_DB_PATH", "data/feedback.db")

# SQLite schema for feedback and aspect-level opinions.
# foreign_keys pragma is enabled to support cascade deletion.
SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  review_id TEXT,
  text TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS aspect_opinion (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  feedback_id INTEGER NOT NULL,
  aspect TEXT NOT NULL,
  sentiment TEXT NOT NULL,
  score REAL DEFAULT 0.0,
  source TEXT,
  FOREIGN KEY (feedback_id) REFERENCES feedback(id) ON DELETE CASCADE
);
"""

# -----------------------------------------------------------------------------
# Connection management
# -----------------------------------------------------------------------------
def get_conn():
    """
    Create and return a new SQLite connection.

    Notes
    -----
    - Ensures that the directory containing the database file exists.
    - Does not enable row factory; callers convert rows explicitly.
    """
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

# -----------------------------------------------------------------------------
# Schema initialization and reset
# -----------------------------------------------------------------------------
def init_db():
    """
    Initialize the SQLite database schema if it does not already exist.

    Effects
    -------
    - Creates feedback and aspect_opinion tables.
    - Enables foreign key constraints for the connection used.
    """
    conn = get_conn()
    with conn:
        conn.executescript(SCHEMA)
    conn.close()

def reset_db():
    """
    Reset the database by dropping all tables and recreating the schema.

    Warning
    -------
    This operation is destructive and will remove ALL stored feedback and
    aspect-level opinion records. It is intended for experiments and tests.
    """
    conn = get_conn()
    with conn:
        conn.executescript(
            "DROP TABLE IF EXISTS aspect_opinion; "
            "DROP TABLE IF EXISTS feedback;"
        )
    conn.close()
    init_db()

# -----------------------------------------------------------------------------
# Insert operations
# -----------------------------------------------------------------------------
def insert_feedback(text: str, review_id: Optional[str] = None) -> int:
    """
    Insert a new feedback entry.

    Parameters
    ----------
    text : str
        Raw feedback text (non-empty).
    review_id : Optional[str]
        Optional external review identifier (from original dataset).

    Returns
    -------
    int
        The auto-incremented primary key (feedback.id) of the inserted row.

    Notes
    -----
    - created_at is stored in UTC using ISO-8601 format (second-level precision).
    """
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn = get_conn()
    with conn:
        cur = conn.execute(
            "INSERT INTO feedback(review_id,text,created_at) VALUES (?,?,?)",
            (review_id, text, created)
        )
        fid = cur.lastrowid
    conn.close()
    return int(fid)

def insert_aspect_opinions(feedback_id: int, opinions: List[Dict[str, Any]]):
    """
    Insert multiple aspect-level opinions linked to a single feedback entry.

    Parameters
    ----------
    feedback_id : int
        Foreign key referencing the feedback.id to which opinions belong.
    opinions : List[Dict[str, Any]]
        Each opinion dict is expected to contain:
            - aspect   : str
            - sentiment: str
            - score    : float-like (confidence or probability)
            - source   : str (e.g., 'gold', 'user_auto', 'model')

    Notes
    -----
    - Uses executemany for batch insertion.
    - If feedback_id does not exist, the foreign-key constraint will fail.
    """
    conn = get_conn()
    rows = [
        (
            feedback_id,
            op["aspect"],
            op["sentiment"],
            float(op.get("score", 0.0)),
            op.get("source", "")
        )
        for op in opinions
    ]
    with conn:
        conn.executemany(
            "INSERT INTO aspect_opinion(feedback_id,aspect,sentiment,score,source) "
            "VALUES (?,?,?,?,?)",
            rows
        )
    conn.close()

# -----------------------------------------------------------------------------
# Query operations
# -----------------------------------------------------------------------------
def fetch_recent(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch the most recent feedback entries.

    Parameters
    ----------
    limit : int
        Maximum number of feedback rows to fetch (default: 20).

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with keys:
            - id
            - review_id
            - text
            - created_at
    """
    conn = get_conn()
    cur = conn.execute(
        """
        SELECT f.id, f.review_id, f.text, f.created_at
        FROM feedback f
        ORDER BY f.id DESC
        LIMIT ?
        """,
        (limit,)
    )
    rows = cur.fetchall()
    conn.close()

    keys = ["id", "review_id", "text", "created_at"]
    return [dict(zip(keys, r)) for r in rows]

def fetch_all_aspects() -> List[Dict[str, Any]]:
    """
    Fetch all aspect-level opinions joined with their parent feedback.

    Returns
    -------
    List[Dict[str, Any]]
        Each entry is a dictionary with keys:
            - feedback_id
            - review_id
            - text
            - aspect
            - sentiment
            - score
            - source

    Notes
    -----
    - The join ensures that opinions can be analyzed together with their
      original feedback text and review identifiers.
    - Ordered by aspect_opinion.id to preserve insertion sequence.
    """
    conn = get_conn()
    cur = conn.execute(
        """
        SELECT ao.feedback_id, f.review_id, f.text,
               ao.aspect, ao.sentiment, ao.score, ao.source
        FROM aspect_opinion ao
        JOIN feedback f ON f.id = ao.feedback_id
        ORDER BY ao.id ASC
        """
    )
    rows = cur.fetchall()
    conn.close()

    keys = [
        "feedback_id",
        "review_id",
        "text",
        "aspect",
        "sentiment",
        "score",
        "source"
    ]
    return [dict(zip(keys, r)) for r in rows]
