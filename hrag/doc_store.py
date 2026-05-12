"""Parent document store (paper Figure 1: 'Docs Store').

SQLite-backed key-value store mapping parent_id -> full document text.
Used at generation time to reconstruct parent-level context.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

from .chunking import ParentDoc
from .config import settings


class ParentDocStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or settings.docstore_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS parents (
                parent_id TEXT PRIMARY KEY,
                text      TEXT NOT NULL,
                source    TEXT,
                metadata  TEXT
            )
            """
        )
        self._conn.commit()

    def upsert_many(self, parents: Iterable[ParentDoc]) -> None:
        rows = [
            (p.parent_id, p.text, p.source, json.dumps(p.metadata))
            for p in parents
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO parents(parent_id, text, source, metadata) VALUES (?,?,?,?)",
            rows,
        )
        self._conn.commit()

    def get(self, parent_id: str) -> Optional[ParentDoc]:
        row = self._conn.execute(
            "SELECT parent_id, text, source, metadata FROM parents WHERE parent_id=?",
            (parent_id,),
        ).fetchone()
        if not row:
            return None
        return ParentDoc(
            parent_id=row[0],
            text=row[1],
            source=row[2] or "",
            metadata=json.loads(row[3] or "{}"),
        )

    def get_many(self, parent_ids: List[str]) -> List[ParentDoc]:
        if not parent_ids:
            return []
        placeholders = ",".join("?" * len(parent_ids))
        rows = self._conn.execute(
            f"SELECT parent_id, text, source, metadata FROM parents WHERE parent_id IN ({placeholders})",
            parent_ids,
        ).fetchall()
        by_id = {
            r[0]: ParentDoc(
                parent_id=r[0],
                text=r[1],
                source=r[2] or "",
                metadata=json.loads(r[3] or "{}"),
            )
            for r in rows
        }
        return [by_id[pid] for pid in parent_ids if pid in by_id]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM parents").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
