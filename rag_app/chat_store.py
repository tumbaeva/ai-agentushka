"""SQLite-backed chat history: sessions and messages survive restarts."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path


class ChatStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    citations TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                """
            )

    def create_session(self, title: str = "Новый диалог") -> dict:
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO sessions (title, created_at, updated_at) VALUES (?, ?, ?)",
                (title, now, now),
            )
            sid = cur.lastrowid
        return {"id": sid, "title": title, "created_at": now, "updated_at": now}

    def list_sessions(self) -> list[dict]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_messages(self, session_id: int) -> list[dict]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT id, role, text, citations, created_at FROM messages "
                "WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["citations"] = json.loads(d["citations"]) if d["citations"] else []
            out.append(d)
        return out

    def add_message(self, session_id: int, role: str, text: str,
                    citations: list | None = None) -> dict:
        now = datetime.now().isoformat(timespec="seconds")
        payload = json.dumps(citations or [], ensure_ascii=False)
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, text, citations, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, text, payload, now),
            )
            mid = cur.lastrowid
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
            # auto-title from the first user message
            if role == "user":
                row = conn.execute(
                    "SELECT title FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
                if row and row["title"] == "Новый диалог":
                    conn.execute("UPDATE sessions SET title = ? WHERE id = ?",
                                 (text[:60], session_id))
        return {"id": mid, "role": role, "text": text, "citations": citations or [],
                "created_at": now}

    def delete_session(self, session_id: int) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
