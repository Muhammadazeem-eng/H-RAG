"""Chat history / session DB (paper Figure 1: 'Chat History / Session DB').

Per-session list of Q/A turns retrieved by the API orchestrator before
query rewriting. SQLite keeps it dependency-free.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import List, Tuple

from .config import settings


class SessionStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or settings.session_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                ts         REAL NOT NULL,
                question   TEXT NOT NULL,
                answer     TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, id)"
        )
        self._conn.commit()

    def append_turn(self, session_id: str, question: str, answer: str) -> None:
        self._conn.execute(
            "INSERT INTO turns(session_id, ts, question, answer) VALUES (?,?,?,?)",
            (session_id, time.time(), question, answer),
        )
        self._conn.commit()

    def recent_turns(self, session_id: str, limit: int = 3) -> List[Tuple[str, str]]:
        """Most recent Q/A pairs in chronological order (oldest first)."""
        rows = self._conn.execute(
            "SELECT question, answer FROM turns WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return list(reversed(rows))

    def close(self) -> None:
        self._conn.close()
