"""SQLite-backed content-addressed cache for persistent, async-safe caching."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"
DEFAULT_DB_PATH = DEFAULT_CACHE_DIR / "calls.db"


class CacheStore:
    """Thread-safe and persistent SQLite cache store."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.saved_latency_ms = 0.0
        self.saved_cost_usd = 0.0
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        endpoint TEXT,
                        model TEXT,
                        response_json TEXT,
                        latency_ms REAL,
                        cost_usd REAL,
                        created_at REAL
                    )
                    """
                )
                conn.commit()

    def get(self, key: str) -> Optional[Tuple[Dict[str, Any], float, float]]:
        """Retrieve cached response by content-addressed key."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT response_json, latency_ms, cost_usd FROM cache_entries WHERE key = ?",
                    (key,),
                )
                row = cursor.fetchone()
                if row:
                    self.hits += 1
                    resp_json, latency_ms, cost_usd = row
                    self.saved_latency_ms += latency_ms
                    self.saved_cost_usd += cost_usd
                    return json.loads(resp_json), latency_ms, cost_usd
                else:
                    self.misses += 1
                    return None

    def set(
        self,
        key: str,
        endpoint: str,
        model: str,
        response_payload: Dict[str, Any],
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
    ) -> None:
        """Store response in cache."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries
                    (key, endpoint, model, response_json, latency_ms, cost_usd, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key,
                        endpoint,
                        model,
                        json.dumps(response_payload),
                        latency_ms,
                        cost_usd,
                        time.time(),
                    ),
                )
                conn.commit()

    def clear(self) -> None:
        """Clear all cache records."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries")
                conn.commit()
            self.hits = 0
            self.misses = 0
            self.saved_latency_ms = 0.0
            self.saved_cost_usd = 0.0

    def stats(self) -> Dict[str, Any]:
        """Return cache hit/miss statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate": hit_rate,
            "saved_latency_ms": self.saved_latency_ms,
            "saved_cost_usd": self.saved_cost_usd,
        }
