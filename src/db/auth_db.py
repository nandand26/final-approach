"""SQLite helpers for authentication-aware chat thread isolation.

This module creates and maintains two relational tables:
1) users: one row per authenticated Google identity
2) user_threads: one row per chat thread owned by a user

WHY this exists:
- Streamlit authentication tells us *who* is logged in.
- LangGraph checkpointer uses a `thread_id` to store conversation state.
- We need a durable mapping from (Google user) -> (their thread IDs)
  so users never read/write each other's memory.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Resolve project root from src/db/auth_db.py -> src -> project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dedicated SQLite file for auth/thread ownership metadata.
# Keeping this separate from flight-history DB avoids schema coupling.
AUTH_DB_PATH = PROJECT_ROOT / "auth_sessions.db"


def _now_utc_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format for audit fields."""
    return datetime.now(tz=timezone.utc).isoformat()


def _connect(db_path: Path = AUTH_DB_PATH) -> sqlite3.Connection:
    """Open SQLite connection with FK checks enabled.

    WHY enable PRAGMA foreign_keys:
    - Ensures user_threads.user_id must exist in users.user_id
    - Prevents orphan thread rows if malformed writes are attempted
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON") # Enable foreign key constraints to prevent orphaned rows
    return conn


def init_auth_db(db_path: Path = AUTH_DB_PATH) -> Path:
    """Create auth/session mapping tables if missing.

    Tables:
    - users(user_id PK, email, display_name, first_seen)
    - user_threads(thread_id PK, user_id FK, title, created_at)
    """
    conn = _connect(db_path)
    try:
        cursor = conn.cursor()

        # One row per Google identity. user_id is stable (prefer Google `sub`).
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT,
                display_name TEXT,
                first_seen TEXT
            )
            """
        )

        # One row per chat thread, explicitly owned by one user.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_threads (
                thread_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
            """
        )

        # Helpful index when listing threads for sidebar rendering.
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_threads_user_id
            ON user_threads(user_id)
            """
        )

        conn.commit()
    finally:
        conn.close()
    return db_path


def register_or_get_user(
    user_info: dict[str, Any],
    db_path: Path = AUTH_DB_PATH,
) -> dict[str, str]:
    """Upsert a user identity and return canonical user payload.

    Expected user_info keys:
    - user_id: stable unique ID (Google `sub` preferred, fallback `email`)
    - email: user email
    - display_name: friendly name shown in sidebar

    WHY upsert:
    - First login inserts first_seen timestamp.
    - Future logins refresh email/display name if provider changed values.
    """
    user_id = str(user_info.get("user_id", "")).strip()
    email = str(user_info.get("email", "")).strip()
    display_name = str(user_info.get("display_name", "")).strip() or email or user_id

    if not user_id:
        raise ValueError("register_or_get_user requires non-empty user_id")

    conn = _connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (user_id, email, display_name, first_seen)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                email = excluded.email,
                display_name = excluded.display_name
            """,
            (user_id, email, display_name, _now_utc_iso()),
        )
        conn.commit()

        cursor.execute(
            """
            SELECT user_id, email, display_name, first_seen
            FROM users
            WHERE user_id = ?
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError("Failed to read user after upsert")
        return {
            "user_id": str(row[0]),
            "email": str(row[1] or ""),
            "display_name": str(row[2] or ""),
            "first_seen": str(row[3] or ""),
        }
    finally:
        conn.close()


def get_user_threads(
    user_id: str,
    db_path: Path = AUTH_DB_PATH,
) -> list[dict[str, str]]:
    """Return all threads owned by a given user, newest first.

    WHY filter by user_id:
    - This is the primary guardrail preventing cross-user thread visibility.
    """
    conn = _connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT thread_id, user_id, title, created_at
            FROM user_threads
            WHERE user_id = ?
            ORDER BY datetime(created_at) DESC
            """,
            (user_id,),
        )
        rows = cursor.fetchall()
        return [
            {
                "thread_id": str(row[0]),
                "user_id": str(row[1]),
                "title": str(row[2] or "New Chat"),
                "created_at": str(row[3] or ""),
            }
            for row in rows
        ]
    finally:
        conn.close()


def create_new_thread(
    user_id: str,
    title: str,
    db_path: Path = AUTH_DB_PATH,
) -> dict[str, str]:
    """Create a new thread row bound to one user.

    UUID generation happens here so the function directly matches app UX:
    "+ New Chat" -> create row -> return new thread_id.
    """
    import uuid

    clean_user_id = str(user_id).strip()
    clean_thread_id = str(uuid.uuid4()) # Generate a new UUID for the thread using the uuid4 function which is a 128-bit universally unique identifier (UUID)
    clean_title = str(title).strip() or "New Chat"
    if not clean_user_id:
        raise ValueError("create_new_thread requires non-empty user_id")

    conn = _connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_threads (thread_id, user_id, title, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (clean_thread_id, clean_user_id, clean_title, _now_utc_iso()),
        )
        conn.commit()
        return {
            "thread_id": clean_thread_id,
            "user_id": clean_user_id,
            "title": clean_title,
            "created_at": _now_utc_iso(),
        }
    finally:
        conn.close()
