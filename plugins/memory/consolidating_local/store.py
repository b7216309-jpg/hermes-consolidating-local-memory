from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "use",
    "uses",
    "using",
    "we",
    "with",
    "you",
    "your",
}


def now_ts() -> float:
    return time.time()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_text(text: str) -> str:
    clean = normalize_whitespace(text).lower()
    clean = re.sub(r"[^a-z0-9\s\-_/.:]", "", clean)
    return normalize_whitespace(clean)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalize_text(text))
    slug = slug.strip("-")
    return slug or "general"


def pretty_topic(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").title()


def fingerprint_text(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def text_signature(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", normalize_text(text))
    keep = [token for token in tokens if token not in STOPWORDS]
    return " ".join(keep[:6])


def _row_to_dict(row: sqlite3.Row | None) -> Dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    for key in list(data.keys()):
        if not key.endswith("_json"):
            continue
        raw = data.pop(key)
        parsed_key = key[:-5]
        if raw in (None, ""):
            data[parsed_key] = {} if parsed_key in {"metadata", "payload", "stats"} else []
            continue
        try:
            data[parsed_key] = json.loads(raw)
        except Exception:
            data[parsed_key] = {} if parsed_key in {"metadata", "payload", "stats"} else []
    return data


def _merge_json_dict(existing: Any, update: Dict[str, Any] | None) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if isinstance(existing, dict):
        merged.update(existing)
    if isinstance(update, dict):
        merged.update(update)
    return merged


class MemoryStore:
    SEARCH_SCOPES = (
        "facts",
        "topics",
        "episodes",
        "summaries",
        "journals",
        "preferences",
        "policies",
    )

    def __init__(self, db_path: str | Path):
        self.db_path = str(Path(db_path).expanduser())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._fts_enabled = False
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur

    def _fetchone(self, sql: str, params: Iterable[Any] = ()) -> Dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(sql, tuple(params)).fetchone()
        return _row_to_dict(row)

    def _fetchall(self, sql: str, params: Iterable[Any] = ()) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [_row_to_dict(row) or {} for row in rows]

    def _init_schema(self) -> None:
        schema = [
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                normalized_content TEXT NOT NULL,
                fingerprint TEXT NOT NULL UNIQUE,
                signature TEXT NOT NULL,
                category TEXT NOT NULL,
                topic TEXT NOT NULL,
                source TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                importance INTEGER NOT NULL DEFAULT 5,
                confidence REAL NOT NULL DEFAULT 0.7,
                salience REAL NOT NULL DEFAULT 0.55,
                active INTEGER NOT NULL DEFAULT 1,
                superseded_by INTEGER,
                subject_key TEXT NOT NULL DEFAULT '',
                value_key TEXT NOT NULL DEFAULT '',
                polarity INTEGER NOT NULL DEFAULT 1,
                exclusive INTEGER NOT NULL DEFAULT 0,
                source_session_id TEXT NOT NULL DEFAULT '',
                last_recalled_at REAL NOT NULL DEFAULT 0,
                decay_half_life_days REAL NOT NULL DEFAULT 45,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_seen_at REAL NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_facts_topic ON facts(topic, active, updated_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_facts_signature ON facts(signature, active)",
            "CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_key, active, updated_at DESC)",
            """
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                summary TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                importance INTEGER NOT NULL DEFAULT 5,
                salience REAL NOT NULL DEFAULT 0.55,
                source_session_id TEXT NOT NULL DEFAULT '',
                last_recalled_at REAL NOT NULL DEFAULT 0,
                decay_half_life_days REAL NOT NULL DEFAULT 60,
                updated_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS topic_membership (
                topic_id INTEGER NOT NULL,
                fact_id INTEGER NOT NULL,
                PRIMARY KEY (topic_id, fact_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_content TEXT NOT NULL,
                assistant_content TEXT NOT NULL,
                digest TEXT NOT NULL,
                topic_hint TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id, created_at DESC)",
            """
            CREATE TABLE IF NOT EXISTS provider_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS consolidation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reason TEXT NOT NULL,
                started_at REAL NOT NULL,
                finished_at REAL NOT NULL,
                source_episode_id INTEGER NOT NULL DEFAULT 0,
                stats_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS contradictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_key TEXT NOT NULL,
                winner_fact_id INTEGER NOT NULL,
                loser_fact_id INTEGER NOT NULL,
                resolution TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_contradictions_subject ON contradictions(subject_key, created_at DESC)",
            """
            CREATE TABLE IF NOT EXISTS memory_sessions (
                session_id TEXT PRIMARY KEY,
                label TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'open',
                started_at REAL NOT NULL,
                ended_at REAL NOT NULL DEFAULT 0,
                last_activity_at REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                label TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                trace_type TEXT NOT NULL DEFAULT 'turn',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                importance INTEGER NOT NULL DEFAULT 4,
                salience REAL NOT NULL DEFAULT 0.45,
                last_recalled_at REAL NOT NULL DEFAULT 0,
                source_episode_id INTEGER NOT NULL DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_memory_traces_session ON memory_traces(session_id, updated_at DESC)",
            """
            CREATE TABLE IF NOT EXISTS memory_journals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL DEFAULT '',
                label TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                journal_type TEXT NOT NULL DEFAULT 'note',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                importance INTEGER NOT NULL DEFAULT 6,
                salience REAL NOT NULL DEFAULT 0.6,
                last_recalled_at REAL NOT NULL DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_memory_journals_session ON memory_journals(session_id, updated_at DESC)",
            """
            CREATE TABLE IF NOT EXISTS memory_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL DEFAULT '',
                label TEXT NOT NULL,
                summary TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                summary_type TEXT NOT NULL DEFAULT 'session',
                source_hash TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                importance INTEGER NOT NULL DEFAULT 7,
                salience REAL NOT NULL DEFAULT 0.65,
                last_recalled_at REAL NOT NULL DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """,
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_summaries_source_hash ON memory_summaries(source_hash)",
            "CREATE INDEX IF NOT EXISTS idx_memory_summaries_session ON memory_summaries(session_id, updated_at DESC)",
            """
            CREATE TABLE IF NOT EXISTS memory_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                preference_key TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                value TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                source_session_id TEXT NOT NULL DEFAULT '',
                importance INTEGER NOT NULL DEFAULT 8,
                salience REAL NOT NULL DEFAULT 0.9,
                last_recalled_at REAL NOT NULL DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_policies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                policy_key TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                source_session_id TEXT NOT NULL DEFAULT '',
                importance INTEGER NOT NULL DEFAULT 9,
                salience REAL NOT NULL DEFAULT 0.95,
                last_recalled_at REAL NOT NULL DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_kind TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                subject_key TEXT NOT NULL DEFAULT '',
                action TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_memory_history_entity ON memory_history(entity_kind, entity_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memory_history_subject ON memory_history(subject_key, created_at DESC)",
            """
            CREATE TABLE IF NOT EXISTS memory_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_kind TEXT NOT NULL,
                source_id TEXT NOT NULL,
                target_kind TEXT NOT NULL,
                target_id TEXT NOT NULL,
                link_type TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            )
            """,
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_links_unique ON memory_links(source_kind, source_id, target_kind, target_id, link_type)",
            "CREATE INDEX IF NOT EXISTS idx_memory_links_target ON memory_links(target_kind, target_id, link_type)",
        ]
        for sql in schema:
            self._execute(sql)

        self._ensure_column("facts", "salience", "REAL NOT NULL DEFAULT 0.55")
        self._ensure_column("facts", "source_session_id", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("facts", "last_recalled_at", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("facts", "decay_half_life_days", "REAL NOT NULL DEFAULT 45")
        self._ensure_column("topics", "salience", "REAL NOT NULL DEFAULT 0.55")
        self._ensure_column("topics", "source_session_id", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("topics", "last_recalled_at", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("topics", "decay_half_life_days", "REAL NOT NULL DEFAULT 60")
        self._ensure_column("memory_preferences", "source_session_id", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("memory_policies", "source_session_id", "TEXT NOT NULL DEFAULT ''")
        self._execute("CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(source_session_id, updated_at DESC)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_memory_preferences_session ON memory_preferences(source_session_id, updated_at DESC)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_memory_policies_session ON memory_policies(source_session_id, updated_at DESC)")
        self._backfill_source_sessions("memory_preferences")
        self._backfill_source_sessions("memory_policies")
        self._backfill_memory_sessions()
        self._init_fts()

    def _init_fts(self) -> None:
        try:
            self._execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(fact_id UNINDEXED, content, topic, category)"
            )
            self._execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS topics_fts USING fts5(topic_id UNINDEXED, title, summary, category)"
            )
            self._execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(episode_id UNINDEXED, digest, user_content, assistant_content)"
            )
            self._execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memory_summaries_fts USING fts5(summary_id UNINDEXED, label, summary, content, summary_type)"
            )
            self._execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memory_journals_fts USING fts5(journal_id UNINDEXED, label, content, journal_type)"
            )
            self._execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memory_preferences_fts USING fts5(preference_id UNINDEXED, preference_key, label, value, content)"
            )
            self._execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memory_policies_fts USING fts5(policy_id UNINDEXED, policy_key, label, content)"
            )
            self._execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memory_traces_fts USING fts5(trace_id UNINDEXED, label, content, trace_type)"
            )
            self._fts_enabled = True
        except sqlite3.OperationalError:
            self._fts_enabled = False

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        with self._lock:
            rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {str(row["name"]) for row in rows}
        if column in existing:
            return
        self._execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def _backfill_source_sessions(self, table: str) -> None:
        rows = self._fetchall(
            f"""
            SELECT id, metadata_json
            FROM {table}
            WHERE COALESCE(source_session_id, '') = ''
            """
        )
        for row in rows:
            metadata = dict(row.get("metadata") or {})
            session_id = normalize_whitespace(str(metadata.get("session_id") or ""))
            if not session_id:
                continue
            self._execute(
                f"UPDATE {table} SET source_session_id = ? WHERE id = ?",
                (session_id, int(row["id"])),
            )

    def _backfill_memory_sessions(self) -> None:
        session_ids = set()
        queries = [
            "SELECT DISTINCT source_session_id AS session_id FROM facts WHERE COALESCE(source_session_id, '') != ''",
            "SELECT DISTINCT source_session_id AS session_id FROM memory_preferences WHERE COALESCE(source_session_id, '') != ''",
            "SELECT DISTINCT source_session_id AS session_id FROM memory_policies WHERE COALESCE(source_session_id, '') != ''",
            "SELECT DISTINCT session_id FROM memory_traces WHERE COALESCE(session_id, '') != ''",
            "SELECT DISTINCT session_id FROM memory_journals WHERE COALESCE(session_id, '') != ''",
            "SELECT DISTINCT session_id FROM memory_summaries WHERE COALESCE(session_id, '') != ''",
            "SELECT DISTINCT session_id FROM episodes WHERE COALESCE(session_id, '') != ''",
        ]
        for sql in queries:
            for row in self._fetchall(sql):
                session_id = normalize_whitespace(str(row.get("session_id") or ""))
                if session_id:
                    session_ids.add(session_id)
        for session_id in sorted(session_ids):
            self.ensure_memory_session(session_id, label=session_id)

    def get_state(self, key: str, default: str = "") -> str:
        row = self._fetchone("SELECT value FROM provider_state WHERE key = ?", (key,))
        if not row:
            return default
        return str(row["value"])

    def set_state(self, key: str, value: Any) -> None:
        self._execute(
            """
            INSERT INTO provider_state(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, str(value)),
        )

    def counts(self) -> Dict[str, int]:
        tables = {
            "facts": "SELECT COUNT(*) AS count FROM facts WHERE active = 1",
            "topics": "SELECT COUNT(*) AS count FROM topics",
            "episodes": "SELECT COUNT(*) AS count FROM episodes",
            "contradictions": "SELECT COUNT(*) AS count FROM contradictions",
            "sessions": "SELECT COUNT(*) AS count FROM memory_sessions",
            "traces": "SELECT COUNT(*) AS count FROM memory_traces WHERE active = 1",
            "journals": "SELECT COUNT(*) AS count FROM memory_journals WHERE active = 1",
            "summaries": "SELECT COUNT(*) AS count FROM memory_summaries WHERE active = 1",
            "preferences": "SELECT COUNT(*) AS count FROM memory_preferences WHERE active = 1",
            "policies": "SELECT COUNT(*) AS count FROM memory_policies WHERE active = 1",
            "history": "SELECT COUNT(*) AS count FROM memory_history",
            "links": "SELECT COUNT(*) AS count FROM memory_links",
        }
        counts: Dict[str, int] = {}
        for key, sql in tables.items():
            row = self._fetchone(sql) or {"count": 0}
            counts[key] = int(row["count"])
        return counts

    def ensure_memory_session(
        self,
        session_id: str,
        *,
        label: str = "",
        summary: str = "",
        status: str = "open",
    ) -> Dict[str, Any]:
        clean_id = normalize_whitespace(session_id)
        if not clean_id:
            raise ValueError("session_id is required")
        now = now_ts()
        existing = self._fetchone("SELECT * FROM memory_sessions WHERE session_id = ?", (clean_id,))
        if existing:
            next_label = label or str(existing.get("label") or "")
            next_summary = summary or str(existing.get("summary") or "")
            next_status = status or str(existing.get("status") or "open")
            self._execute(
                """
                UPDATE memory_sessions
                SET label = ?, summary = ?, status = ?, last_activity_at = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (next_label, next_summary, next_status, now, now, clean_id),
            )
            return self._fetchone("SELECT * FROM memory_sessions WHERE session_id = ?", (clean_id,)) or {}

        self._execute(
            """
            INSERT INTO memory_sessions(session_id, label, summary, status, started_at, ended_at, last_activity_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)
            """,
            (clean_id, label or clean_id, summary or "", status or "open", now, now, now, now),
        )
        session = self._fetchone("SELECT * FROM memory_sessions WHERE session_id = ?", (clean_id,)) or {}
        self.record_history(
            entity_kind="session",
            entity_id=clean_id,
            action="opened",
            reason="ensure_session",
            source="session",
            payload=session,
        )
        return session

    def close_memory_session(self, session_id: str, *, summary: str = "") -> Dict[str, Any]:
        session = self.ensure_memory_session(session_id, summary=summary or "", status="closed")
        now = now_ts()
        self._execute(
            """
            UPDATE memory_sessions
            SET summary = ?, status = 'closed', ended_at = ?, last_activity_at = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (summary or str(session.get("summary") or ""), now, now, now, normalize_whitespace(session_id)),
        )
        closed = self._fetchone("SELECT * FROM memory_sessions WHERE session_id = ?", (normalize_whitespace(session_id),)) or {}
        self.record_history(
            entity_kind="session",
            entity_id=normalize_whitespace(session_id),
            action="closed",
            reason="session_end",
            source="session",
            payload=closed,
        )
        return closed

    def append_episode(
        self,
        *,
        session_id: str,
        user_content: str,
        assistant_content: str,
        topic_hint: str = "",
        created_at: float | None = None,
    ) -> Dict[str, Any]:
        created_at = float(created_at or now_ts())
        clean_session = normalize_whitespace(session_id)
        self.ensure_memory_session(clean_session)
        digest_source = normalize_whitespace(f"{user_content} {assistant_content}")[:240]
        digest = digest_source or "(empty turn)"
        cur = self._execute(
            """
            INSERT INTO episodes(session_id, user_content, assistant_content, digest, topic_hint, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (clean_session, user_content or "", assistant_content or "", digest, topic_hint or "", created_at),
        )
        episode_id = int(cur.lastrowid)
        self._upsert_episode_fts(
            episode_id=episode_id,
            digest=digest,
            user_content=user_content or "",
            assistant_content=assistant_content or "",
        )
        self._execute(
            "UPDATE memory_sessions SET last_activity_at = ?, updated_at = ? WHERE session_id = ?",
            (created_at, created_at, clean_session),
        )
        return {
            "id": episode_id,
            "session_id": clean_session,
            "digest": digest,
            "topic_hint": topic_hint or "",
            "created_at": created_at,
        }

    def purge_episode_buffers(self, *, retention_hours: float, max_episode_id: int | None = None) -> int:
        cutoff = now_ts() - max(float(retention_hours), 0.0) * 3600.0
        sql = "SELECT id FROM episodes WHERE created_at <= ?"
        params: List[Any] = [cutoff]
        if max_episode_id is not None:
            sql += " AND id <= ?"
            params.append(int(max_episode_id))
        rows = self._fetchall(sql, params)
        if not rows:
            return 0
        count = 0
        for row in rows:
            episode_id = int(row["id"])
            self._execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
            self._delete_episode_fts(episode_id)
            count += 1
        return count

    def sessions_since_episode(self, episode_id: int) -> int:
        row = self._fetchone(
            "SELECT COUNT(DISTINCT session_id) AS count FROM episodes WHERE id > ?",
            (int(episode_id),),
        ) or {"count": 0}
        return int(row["count"])

    def episodes_since_episode(self, episode_id: int, limit: int = 500) -> List[Dict[str, Any]]:
        return self._fetchall(
            """
            SELECT id, session_id, user_content, assistant_content, digest, topic_hint, created_at
            FROM episodes
            WHERE id > ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (int(episode_id), int(limit)),
        )

    def latest_episode_id(self) -> int:
        row = self._fetchone("SELECT COALESCE(MAX(id), 0) AS id FROM episodes") or {"id": 0}
        return int(row["id"])

    def get_session_artifacts(self, session_id: str, *, limit: int = 20) -> Dict[str, Any]:
        clean_id = normalize_whitespace(session_id)
        like_session = f'%\"session_id\": \"{clean_id}\"%'
        return {
            "session": self._fetchone("SELECT * FROM memory_sessions WHERE session_id = ?", (clean_id,)) or {},
            "episodes": self._fetchall(
                """
                SELECT id, session_id, digest, topic_hint, created_at
                FROM episodes
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (clean_id, int(limit)),
            ),
            "traces": self._fetchall(
                """
                SELECT id, session_id, label, content, trace_type, importance, salience, updated_at
                FROM memory_traces
                WHERE session_id = ? AND active = 1
                ORDER BY id DESC
                LIMIT ?
                """,
                (clean_id, int(limit)),
            ),
            "journals": self._fetchall(
                """
                SELECT id, session_id, label, content, journal_type, importance, salience, updated_at
                FROM memory_journals
                WHERE session_id = ? AND active = 1
                ORDER BY id DESC
                LIMIT ?
                """,
                (clean_id, int(limit)),
            ),
            "summaries": self._fetchall(
                """
                SELECT id, session_id, label, summary, summary_type, importance, salience, updated_at
                FROM memory_summaries
                WHERE session_id = ? AND active = 1
                ORDER BY id DESC
                LIMIT ?
                """,
                (clean_id, int(limit)),
            ),
            "preferences": self._fetchall(
                """
                SELECT id, preference_key, label, value, content, source_session_id, importance, salience, updated_at
                FROM memory_preferences
                WHERE active = 1
                  AND (source_session_id = ? OR (source_session_id = '' AND metadata_json LIKE ?))
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (clean_id, like_session, int(limit)),
            ),
            "policies": self._fetchall(
                """
                SELECT id, policy_key, label, content, source_session_id, importance, salience, updated_at
                FROM memory_policies
                WHERE active = 1
                  AND (source_session_id = ? OR (source_session_id = '' AND metadata_json LIKE ?))
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (clean_id, like_session, int(limit)),
            ),
            "facts": self._fetchall(
                """
                SELECT id, content, category, topic, importance, salience, updated_at, subject_key, value_key, source_session_id
                FROM facts
                WHERE source_session_id = ? AND active = 1
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (clean_id, int(limit)),
            ),
        }

    def list_sessions(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        return self._fetchall(
            """
            SELECT session_id, label, summary, status, started_at, ended_at, last_activity_at, created_at, updated_at
            FROM memory_sessions
            ORDER BY updated_at DESC, session_id ASC
            LIMIT ?
            """,
            (int(limit),),
        )

    def list_topics(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        return self._fetchall(
            """
            SELECT id, slug, title, summary, category, importance, salience, source_session_id, last_recalled_at, decay_half_life_days, updated_at
            FROM topics
            ORDER BY salience DESC, importance DESC, updated_at DESC, slug ASC
            LIMIT ?
            """,
            (int(limit),),
        )

    def list_preferences(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        return self._fetchall(
            """
            SELECT id, preference_key, label, value, content, metadata_json, source_session_id, importance, salience, updated_at
            FROM memory_preferences
            WHERE active = 1
            ORDER BY salience DESC, importance DESC, updated_at DESC, preference_key ASC
            LIMIT ?
            """,
            (int(limit),),
        )

    def list_policies(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        return self._fetchall(
            """
            SELECT id, policy_key, label, content, metadata_json, source_session_id, importance, salience, updated_at
            FROM memory_policies
            WHERE active = 1
            ORDER BY salience DESC, importance DESC, updated_at DESC, policy_key ASC
            LIMIT ?
            """,
            (int(limit),),
        )

    def topic_supporting_facts(self, topic_id: int, *, limit: int = 12) -> List[Dict[str, Any]]:
        return self._fetchall(
            """
            SELECT f.id, f.content, f.category, f.topic, f.importance, f.confidence, f.salience, f.updated_at, f.subject_key, f.value_key, f.source_session_id
            FROM topic_membership tm
            JOIN facts f ON f.id = tm.fact_id
            WHERE tm.topic_id = ? AND f.active = 1
            ORDER BY f.salience DESC, f.importance DESC, f.updated_at DESC, f.id ASC
            LIMIT ?
            """,
            (int(topic_id), int(limit)),
        )

    def list_links(
        self,
        *,
        source_kind: str = "",
        source_id: Any | None = None,
        target_kind: str = "",
        target_id: Any | None = None,
        link_type: str = "",
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if source_kind:
            clauses.append("source_kind = ?")
            params.append(normalize_whitespace(source_kind))
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(str(source_id))
        if target_kind:
            clauses.append("target_kind = ?")
            params.append(normalize_whitespace(target_kind))
        if target_id is not None:
            clauses.append("target_id = ?")
            params.append(str(target_id))
        if link_type:
            clauses.append("link_type = ?")
            params.append(normalize_whitespace(link_type))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        return self._fetchall(
            f"""
            SELECT id, source_kind, source_id, target_kind, target_id, link_type, metadata_json, created_at
            FROM memory_links
            {where}
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            params,
        )

    def append_trace(
        self,
        *,
        session_id: str,
        label: str,
        content: str,
        trace_type: str = "turn",
        metadata: Dict[str, Any] | None = None,
        importance: int = 4,
        salience: float = 0.45,
        source_episode_id: int = 0,
    ) -> Dict[str, Any]:
        clean_session = normalize_whitespace(session_id)
        clean_content = normalize_whitespace(content)
        if not clean_content:
            raise ValueError("Trace content cannot be empty.")
        self.ensure_memory_session(clean_session)
        now = now_ts()
        cur = self._execute(
            """
            INSERT INTO memory_traces(session_id, label, content, trace_type, metadata_json, importance, salience, last_recalled_at, source_episode_id, active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, 1, ?, ?)
            """,
            (
                clean_session,
                normalize_whitespace(label) or trace_type,
                clean_content,
                trace_type or "turn",
                json.dumps(dict(metadata or {}), sort_keys=True),
                int(importance),
                float(salience),
                int(source_episode_id or 0),
                now,
                now,
            ),
        )
        trace_id = int(cur.lastrowid)
        trace = self._fetchone("SELECT * FROM memory_traces WHERE id = ?", (trace_id,)) or {}
        self._upsert_trace_fts(trace)
        self.add_link("trace", trace_id, "session", clean_session, "captured_in")
        if source_episode_id:
            self.add_link("trace", trace_id, "episode", int(source_episode_id), "derived_from_episode")
        self.record_history(
            entity_kind="trace",
            entity_id=trace_id,
            action="inserted",
            reason=trace_type,
            source="sync_turn",
            payload=trace,
        )
        return trace

    def add_journal(
        self,
        *,
        label: str,
        content: str,
        session_id: str = "",
        journal_type: str = "note",
        metadata: Dict[str, Any] | None = None,
        importance: int = 6,
        salience: float = 0.6,
    ) -> Dict[str, Any]:
        clean_content = normalize_whitespace(content)
        if not clean_content:
            raise ValueError("Journal content cannot be empty.")
        clean_session = normalize_whitespace(session_id)
        if clean_session:
            self.ensure_memory_session(clean_session)
        now = now_ts()
        cur = self._execute(
            """
            INSERT INTO memory_journals(session_id, label, content, journal_type, metadata_json, importance, salience, last_recalled_at, active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1, ?, ?)
            """,
            (
                clean_session,
                normalize_whitespace(label) or "Journal",
                clean_content,
                journal_type or "note",
                json.dumps(dict(metadata or {}), sort_keys=True),
                int(importance),
                float(salience),
                now,
                now,
            ),
        )
        journal_id = int(cur.lastrowid)
        journal = self._fetchone("SELECT * FROM memory_journals WHERE id = ?", (journal_id,)) or {}
        self._upsert_journal_fts(journal)
        if clean_session:
            self.add_link("journal", journal_id, "session", clean_session, "captured_in")
        self.record_history(
            entity_kind="journal",
            entity_id=journal_id,
            action="inserted",
            reason=journal_type,
            source="journal",
            payload=journal,
        )
        return journal

    def _make_source_hash(
        self,
        *,
        session_id: str,
        summary_type: str,
        label: str,
        source_refs: Sequence[Dict[str, Any]] | None,
    ) -> str:
        serialized = [normalize_whitespace(session_id), normalize_whitespace(summary_type), normalize_whitespace(label)]
        if not normalize_whitespace(session_id):
            ordered_refs = sorted(
                (
                    (
                        normalize_whitespace(str(ref.get("kind", ""))),
                        normalize_whitespace(str(ref.get("id", ""))),
                    )
                    for ref in (source_refs or [])
                ),
                key=lambda item: (item[0], item[1]),
            )
            for kind, ref_id in ordered_refs:
                serialized.append(f"{kind}:{ref_id}")
        return fingerprint_text("|".join(serialized))

    def upsert_summary(
        self,
        *,
        label: str,
        summary: str,
        session_id: str = "",
        content: str = "",
        summary_type: str = "session",
        metadata: Dict[str, Any] | None = None,
        importance: int = 7,
        salience: float = 0.65,
        source_refs: Sequence[Dict[str, Any]] | None = None,
        reason: str = "distill",
    ) -> Dict[str, Any]:
        clean_summary = normalize_whitespace(summary)
        if not clean_summary:
            raise ValueError("Summary text cannot be empty.")
        clean_session = normalize_whitespace(session_id)
        if clean_session:
            self.ensure_memory_session(clean_session)
        now = now_ts()
        refs = list(source_refs or [])
        source_hash = self._make_source_hash(
            session_id=clean_session,
            summary_type=summary_type,
            label=label,
            source_refs=refs,
        )
        existing = self._fetchone("SELECT * FROM memory_summaries WHERE source_hash = ?", (source_hash,))
        meta = _merge_json_dict(existing.get("metadata") if existing else {}, metadata)
        if refs:
            meta["source_refs"] = refs
        else:
            meta.pop("source_refs", None)
        if existing:
            self._execute(
                """
                UPDATE memory_summaries
                SET session_id = ?, label = ?, summary = ?, content = ?, summary_type = ?, metadata_json = ?, importance = MAX(importance, ?), salience = MAX(salience, ?), active = 1, updated_at = ?
                WHERE source_hash = ?
                """,
                (
                    clean_session,
                    normalize_whitespace(label) or pretty_topic(summary_type),
                    clean_summary,
                    normalize_whitespace(content),
                    summary_type or "session",
                    json.dumps(meta, sort_keys=True),
                    int(importance),
                    float(salience),
                    now,
                    source_hash,
                ),
            )
            summary_id = int(existing["id"])
            action = "updated"
        else:
            cur = self._execute(
                """
                INSERT INTO memory_summaries(session_id, label, summary, content, summary_type, source_hash, metadata_json, importance, salience, last_recalled_at, active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 1, ?, ?)
                """,
                (
                    clean_session,
                    normalize_whitespace(label) or pretty_topic(summary_type),
                    clean_summary,
                    normalize_whitespace(content),
                    summary_type or "session",
                    source_hash,
                    json.dumps(meta, sort_keys=True),
                    int(importance),
                    float(salience),
                    now,
                    now,
                ),
            )
            summary_id = int(cur.lastrowid)
            action = "inserted"
        row = self._fetchone("SELECT * FROM memory_summaries WHERE id = ?", (summary_id,)) or {}
        self._upsert_summary_fts(row)
        self.delete_links(source_kind="summary", source_id=summary_id, link_types=("captured_in", "summarizes"))
        if clean_session:
            self.add_link("summary", summary_id, "session", clean_session, "captured_in")
        for ref in refs:
            self.add_link("summary", summary_id, str(ref.get("kind") or "memory"), str(ref.get("id") or ""), "summarizes")
        self.record_history(
            entity_kind="summary",
            entity_id=summary_id,
            action=action,
            reason=reason,
            source="summary",
            payload=row,
        )
        return row

    def upsert_preference(
        self,
        *,
        key: str,
        label: str,
        value: str,
        content: str = "",
        metadata: Dict[str, Any] | None = None,
        importance: int = 8,
        salience: float = 0.9,
        reason: str = "remember",
    ) -> Dict[str, Any]:
        pref_key = normalize_whitespace(key) or slugify(label or value)
        pref_label = normalize_whitespace(label) or pref_key
        pref_value = normalize_whitespace(value) or pref_label
        pref_content = normalize_whitespace(content) or f"{pref_label}: {pref_value}"
        now = now_ts()
        existing = self._fetchone("SELECT * FROM memory_preferences WHERE preference_key = ?", (pref_key,))
        meta = _merge_json_dict(existing.get("metadata") if existing else {}, metadata)
        source_session_id = normalize_whitespace(
            str(meta.get("session_id") or (existing.get("source_session_id") if existing else "") or "")
        )
        if source_session_id:
            self.ensure_memory_session(source_session_id)
        if existing:
            self._execute(
                """
                UPDATE memory_preferences
                SET label = ?, value = ?, content = ?, metadata_json = ?, source_session_id = ?, importance = MAX(importance, ?), salience = MAX(salience, ?), active = 1, updated_at = ?
                WHERE preference_key = ?
                """,
                (
                    pref_label,
                    pref_value,
                    pref_content,
                    json.dumps(meta, sort_keys=True),
                    source_session_id,
                    int(importance),
                    float(salience),
                    now,
                    pref_key,
                ),
            )
            pref_id = int(existing["id"])
            action = "updated"
        else:
            cur = self._execute(
                """
                INSERT INTO memory_preferences(preference_key, label, value, content, metadata_json, source_session_id, importance, salience, last_recalled_at, active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 1, ?, ?)
                """,
                (
                    pref_key,
                    pref_label,
                    pref_value,
                    pref_content,
                    json.dumps(meta, sort_keys=True),
                    source_session_id,
                    int(importance),
                    float(salience),
                    now,
                    now,
                ),
            )
            pref_id = int(cur.lastrowid)
            action = "inserted"
        row = self._fetchone("SELECT * FROM memory_preferences WHERE id = ?", (pref_id,)) or {}
        self._upsert_preference_fts(row)
        if source_session_id:
            self.add_link("preference", pref_id, "session", source_session_id, "captured_in")
        self.record_history(
            entity_kind="preference",
            entity_id=pref_id,
            subject_key=str(meta.get("subject_key") or pref_key),
            action=action,
            reason=reason,
            source="preference",
            payload=row,
        )
        return row

    def upsert_policy(
        self,
        *,
        key: str,
        label: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
        importance: int = 9,
        salience: float = 0.95,
        reason: str = "policy",
    ) -> Dict[str, Any]:
        policy_key = normalize_whitespace(key) or slugify(label or content[:40])
        policy_label = normalize_whitespace(label) or policy_key
        clean_content = normalize_whitespace(content)
        if not clean_content:
            raise ValueError("Policy content cannot be empty.")
        now = now_ts()
        existing = self._fetchone("SELECT * FROM memory_policies WHERE policy_key = ?", (policy_key,))
        meta = _merge_json_dict(existing.get("metadata") if existing else {}, metadata)
        source_session_id = normalize_whitespace(
            str(meta.get("session_id") or (existing.get("source_session_id") if existing else "") or "")
        )
        if source_session_id:
            self.ensure_memory_session(source_session_id)
        if existing:
            self._execute(
                """
                UPDATE memory_policies
                SET label = ?, content = ?, metadata_json = ?, source_session_id = ?, importance = MAX(importance, ?), salience = MAX(salience, ?), active = 1, updated_at = ?
                WHERE policy_key = ?
                """,
                (
                    policy_label,
                    clean_content,
                    json.dumps(meta, sort_keys=True),
                    source_session_id,
                    int(importance),
                    float(salience),
                    now,
                    policy_key,
                ),
            )
            policy_id = int(existing["id"])
            action = "updated"
        else:
            cur = self._execute(
                """
                INSERT INTO memory_policies(policy_key, label, content, metadata_json, source_session_id, importance, salience, last_recalled_at, active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1, ?, ?)
                """,
                (
                    policy_key,
                    policy_label,
                    clean_content,
                    json.dumps(meta, sort_keys=True),
                    source_session_id,
                    int(importance),
                    float(salience),
                    now,
                    now,
                ),
            )
            policy_id = int(cur.lastrowid)
            action = "inserted"
        row = self._fetchone("SELECT * FROM memory_policies WHERE id = ?", (policy_id,)) or {}
        self._upsert_policy_fts(row)
        if source_session_id:
            self.add_link("policy", policy_id, "session", source_session_id, "captured_in")
        self.record_history(
            entity_kind="policy",
            entity_id=policy_id,
            subject_key=str(meta.get("subject_key") or policy_key),
            action=action,
            reason=reason,
            source="policy",
            payload=row,
        )
        return row

    def add_link(
        self,
        source_kind: str,
        source_id: Any,
        target_kind: str,
        target_id: Any,
        link_type: str,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        source_kind = normalize_whitespace(source_kind)
        target_kind = normalize_whitespace(target_kind)
        link_type = normalize_whitespace(link_type)
        source_id_text = str(source_id)
        target_id_text = str(target_id)
        if not (source_kind and target_kind and link_type and source_id_text and target_id_text):
            raise ValueError("Link fields cannot be empty.")
        existing = self._fetchone(
            """
            SELECT * FROM memory_links
            WHERE source_kind = ? AND source_id = ? AND target_kind = ? AND target_id = ? AND link_type = ?
            """,
            (source_kind, source_id_text, target_kind, target_id_text, link_type),
        )
        if existing:
            return existing
        cur = self._execute(
            """
            INSERT OR IGNORE INTO memory_links(source_kind, source_id, target_kind, target_id, link_type, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_kind,
                source_id_text,
                target_kind,
                target_id_text,
                link_type,
                json.dumps(dict(metadata or {}), sort_keys=True),
                now_ts(),
            ),
        )
        if int(cur.lastrowid or 0) <= 0:
            return self._fetchone(
                """
                SELECT * FROM memory_links
                WHERE source_kind = ? AND source_id = ? AND target_kind = ? AND target_id = ? AND link_type = ?
                """,
                (source_kind, source_id_text, target_kind, target_id_text, link_type),
            ) or {}
        return self._fetchone("SELECT * FROM memory_links WHERE id = ?", (int(cur.lastrowid),)) or {}

    def delete_links(
        self,
        *,
        source_kind: str = "",
        source_id: Any | None = None,
        target_kind: str = "",
        target_id: Any | None = None,
        link_types: Sequence[str] | None = None,
    ) -> int:
        clauses: List[str] = []
        params: List[Any] = []
        if source_kind:
            clauses.append("source_kind = ?")
            params.append(normalize_whitespace(source_kind))
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(str(source_id))
        if target_kind:
            clauses.append("target_kind = ?")
            params.append(normalize_whitespace(target_kind))
        if target_id is not None:
            clauses.append("target_id = ?")
            params.append(str(target_id))
        if link_types:
            clean_types = [normalize_whitespace(item) for item in link_types if normalize_whitespace(item)]
            if clean_types:
                clauses.append(f"link_type IN ({', '.join('?' for _ in clean_types)})")
                params.extend(clean_types)
        if not clauses:
            return 0
        cur = self._execute(f"DELETE FROM memory_links WHERE {' AND '.join(clauses)}", params)
        return int(cur.rowcount or 0)

    def record_history(
        self,
        *,
        entity_kind: str,
        entity_id: Any,
        action: str,
        reason: str = "",
        source: str = "",
        payload: Dict[str, Any] | None = None,
        subject_key: str = "",
    ) -> Dict[str, Any]:
        created_at = now_ts()
        cur = self._execute(
            """
            INSERT INTO memory_history(entity_kind, entity_id, subject_key, action, reason, source, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalize_whitespace(entity_kind),
                str(entity_id),
                normalize_whitespace(subject_key),
                normalize_whitespace(action),
                normalize_whitespace(reason),
                normalize_whitespace(source),
                json.dumps(dict(payload or {}), sort_keys=True),
                created_at,
            ),
        )
        return self._fetchone("SELECT * FROM memory_history WHERE id = ?", (int(cur.lastrowid),)) or {}

    def list_history(
        self,
        *,
        memory_type: str = "",
        entity_id: Any | None = None,
        subject_key: str = "",
        limit: int = 20,
        since_days: int | None = None,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if memory_type:
            clauses.append("entity_kind = ?")
            params.append(normalize_whitespace(memory_type))
        if entity_id is not None:
            clauses.append("entity_id = ?")
            params.append(str(entity_id))
        if subject_key:
            clauses.append("subject_key = ?")
            params.append(normalize_whitespace(subject_key))
        if since_days is not None:
            clauses.append("created_at >= ?")
            params.append(now_ts() - (int(since_days) * 86400))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        return self._fetchall(
            f"""
            SELECT id, entity_kind, entity_id, subject_key, action, reason, source, payload_json, created_at
            FROM memory_history
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        )

    def _default_fact_salience(self, category: str, metadata: Dict[str, Any]) -> float:
        if str(metadata.get("subject_key") or "").startswith("user:"):
            return 0.9
        if category == "workflow":
            return 0.88
        if category == "project":
            return 0.74
        if category == "environment":
            return 0.68
        return 0.55

    def _default_fact_half_life(self, category: str, metadata: Dict[str, Any]) -> float:
        if str(metadata.get("subject_key") or "").startswith("user:"):
            return 240.0
        if category == "workflow":
            return 180.0
        if category == "project":
            return 120.0
        if category == "environment":
            return 90.0
        return 45.0

    def upsert_fact(
        self,
        *,
        content: str,
        category: str,
        topic: str,
        source: str,
        importance: int = 5,
        confidence: float = 0.7,
        metadata: Dict[str, Any] | None = None,
        observed_at: float | None = None,
        salience: float | None = None,
        source_session_id: str = "",
        decay_half_life_days: float | None = None,
        history_reason: str = "",
    ) -> Dict[str, Any]:
        clean = normalize_whitespace(content)
        if not clean:
            raise ValueError("Fact content cannot be empty.")
        observed_at = float(observed_at or now_ts())
        fingerprint = fingerprint_text(clean)
        signature = text_signature(clean)
        topic_slug = slugify(topic)
        existing = self._fetchone("SELECT * FROM facts WHERE fingerprint = ?", (fingerprint,))
        meta = _merge_json_dict(existing.get("metadata") if existing else {}, metadata)
        subject_key = normalize_whitespace(str(meta.get("subject_key") or ""))
        value_key = normalize_text(str(meta.get("value_key") or "")) if meta.get("value_key") else ""
        polarity = -1 if str(meta.get("polarity", 1)).strip() in {"-1", "false", "neg"} else 1
        exclusive = 1 if subject_key and bool(meta.get("exclusive")) else 0
        source_session = normalize_whitespace(source_session_id or str(meta.get("source_session_id") or ""))
        if source_session:
            self.ensure_memory_session(source_session)
        salience_value = float(salience if salience is not None else self._default_fact_salience(category, meta))
        half_life = float(
            decay_half_life_days if decay_half_life_days is not None else self._default_fact_half_life(category, meta)
        )
        metadata_json = json.dumps(meta, sort_keys=True)
        if existing:
            next_subject = subject_key or str(existing.get("subject_key") or "")
            next_value = value_key or str(existing.get("value_key") or "")
            next_polarity = polarity if subject_key else int(existing.get("polarity") or 1)
            next_exclusive = exclusive if subject_key else int(existing.get("exclusive") or 0)
            next_salience = max(float(existing.get("salience") or 0.0), salience_value)
            next_half_life = max(float(existing.get("decay_half_life_days") or 0.0), half_life)
            next_session = source_session or str(existing.get("source_session_id") or "")
            self._execute(
                """
                UPDATE facts
                SET active = 1,
                    importance = MAX(importance, ?),
                    confidence = MAX(confidence, ?),
                    salience = ?,
                    updated_at = ?,
                    last_seen_at = ?,
                    metadata_json = ?,
                    subject_key = ?,
                    value_key = ?,
                    polarity = ?,
                    exclusive = ?,
                    source_session_id = ?,
                    decay_half_life_days = ?
                WHERE id = ?
                """,
                (
                    int(importance),
                    float(confidence),
                    next_salience,
                    observed_at,
                    observed_at,
                    metadata_json,
                    next_subject,
                    next_value,
                    next_polarity,
                    next_exclusive,
                    next_session,
                    next_half_life,
                    int(existing["id"]),
                ),
            )
            updated = self._fetchone("SELECT * FROM facts WHERE id = ?", (int(existing["id"]),)) or {}
            self._upsert_fact_fts(updated)
            contradictions = self._resolve_subject_state(updated)
            if next_session:
                self.add_link("fact", updated["id"], "session", next_session, "captured_in")
            self.record_history(
                entity_kind="fact",
                entity_id=updated["id"],
                subject_key=next_subject,
                action="updated",
                reason=history_reason or source,
                source=source,
                payload=updated,
            )
            return {
                "action": "updated",
                "fact": updated,
                "superseded": contradictions["superseded"],
                "contradictions": contradictions["contradictions"],
            }

        cur = self._execute(
            """
            INSERT INTO facts(
                content,
                normalized_content,
                fingerprint,
                signature,
                category,
                topic,
                source,
                metadata_json,
                importance,
                confidence,
                salience,
                active,
                subject_key,
                value_key,
                polarity,
                exclusive,
                source_session_id,
                last_recalled_at,
                decay_half_life_days,
                created_at,
                updated_at,
                last_seen_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
            """,
            (
                clean,
                normalize_text(clean),
                fingerprint,
                signature,
                category or "general",
                topic_slug,
                source or "manual",
                metadata_json,
                int(importance),
                float(confidence),
                salience_value,
                subject_key,
                value_key,
                int(polarity),
                int(exclusive),
                source_session,
                half_life,
                observed_at,
                observed_at,
                observed_at,
            ),
        )
        fact_id = int(cur.lastrowid)
        inserted = self._fetchone("SELECT * FROM facts WHERE id = ?", (fact_id,)) or {}
        superseded = self._supersede_older_facts(inserted)
        contradictions = self._resolve_subject_state(inserted)
        self._upsert_fact_fts(inserted)
        if source_session:
            self.add_link("fact", fact_id, "session", source_session, "captured_in")
        self.record_history(
            entity_kind="fact",
            entity_id=fact_id,
            subject_key=subject_key,
            action="inserted",
            reason=history_reason or source,
            source=source,
            payload=inserted,
        )
        return {
            "action": "inserted",
            "fact": inserted,
            "superseded": superseded + contradictions["superseded"],
            "contradictions": contradictions["contradictions"],
        }

    def _supersede_older_facts(self, new_fact: Dict[str, Any]) -> List[int]:
        signature = str(new_fact.get("signature", "")).strip()
        if not signature:
            return []
        older = self._fetchall(
            """
            SELECT id
            FROM facts
            WHERE id != ?
              AND active = 1
              AND category = ?
              AND topic = ?
              AND signature = ?
            ORDER BY updated_at DESC
            """,
            (
                int(new_fact["id"]),
                new_fact["category"],
                new_fact["topic"],
                signature,
            ),
        )
        superseded_ids: List[int] = []
        for row in older:
            fact_id = int(row["id"])
            self._soft_supersede_fact(
                fact_id,
                int(new_fact["id"]),
                float(new_fact["updated_at"]),
                subject_key=str(new_fact.get("subject_key") or ""),
                reason="duplicate_signature",
            )
            superseded_ids.append(fact_id)
        return superseded_ids

    def _soft_supersede_fact(
        self,
        fact_id: int,
        winner_id: int,
        updated_at: float,
        *,
        subject_key: str = "",
        reason: str = "superseded",
    ) -> None:
        self._execute(
            "UPDATE facts SET active = 0, superseded_by = ?, updated_at = ? WHERE id = ?",
            (int(winner_id), float(updated_at), int(fact_id)),
        )
        self._delete_fact_fts(int(fact_id))
        self.add_link("fact", winner_id, "fact", fact_id, "supersedes", {"reason": reason})
        self.record_history(
            entity_kind="fact",
            entity_id=fact_id,
            subject_key=subject_key,
            action="superseded",
            reason=reason,
            source="fact",
            payload={"winner_fact_id": winner_id},
        )

    def _resolve_subject_state(self, new_fact: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]] | List[int]]:
        subject_key = normalize_whitespace(str(new_fact.get("subject_key") or ""))
        if not subject_key or int(new_fact.get("exclusive") or 0) != 1:
            return {"superseded": [], "contradictions": []}

        others = self._fetchall(
            """
            SELECT id, content, value_key, polarity
            FROM facts
            WHERE id != ?
              AND active = 1
              AND subject_key = ?
              AND exclusive = 1
            ORDER BY updated_at DESC
            """,
            (int(new_fact["id"]), subject_key),
        )
        superseded: List[int] = []
        contradictions: List[Dict[str, Any]] = []
        new_value = str(new_fact.get("value_key") or "")
        new_polarity = int(new_fact.get("polarity") or 1)
        for row in others:
            row_value = str(row.get("value_key") or "")
            row_polarity = int(row.get("polarity") or 1)
            same_state = row_value == new_value and row_polarity == new_polarity
            contradictory = row_polarity != new_polarity or (row_value and new_value and row_value != new_value)
            if not same_state and not contradictory:
                continue
            self._soft_supersede_fact(
                int(row["id"]),
                int(new_fact["id"]),
                float(new_fact["updated_at"]),
                subject_key=subject_key,
                reason="exclusive_subject",
            )
            superseded.append(int(row["id"]))
            if contradictory:
                contradictions.append(
                    self._record_contradiction(
                        subject_key=subject_key,
                        winner_fact_id=int(new_fact["id"]),
                        loser_fact_id=int(row["id"]),
                        resolution=f"subject={subject_key}; old={row_value or row_polarity}; new={new_value or new_polarity}",
                    )
                )
                self.add_link("fact", int(new_fact["id"]), "fact", int(row["id"]), "contradicts")
        return {"superseded": superseded, "contradictions": contradictions}

    def _record_contradiction(
        self,
        *,
        subject_key: str,
        winner_fact_id: int,
        loser_fact_id: int,
        resolution: str,
    ) -> Dict[str, Any]:
        created_at = now_ts()
        cur = self._execute(
            """
            INSERT INTO contradictions(subject_key, winner_fact_id, loser_fact_id, resolution, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (subject_key, int(winner_fact_id), int(loser_fact_id), resolution, created_at),
        )
        contradiction_id = int(cur.lastrowid)
        row = self._fetchone(
            """
            SELECT c.id,
                   c.subject_key,
                   c.resolution,
                   c.created_at,
                   w.id AS winner_fact_id,
                   w.content AS winner_content,
                   l.id AS loser_fact_id,
                   l.content AS loser_content
            FROM contradictions c
            LEFT JOIN facts w ON w.id = c.winner_fact_id
            LEFT JOIN facts l ON l.id = c.loser_fact_id
            WHERE c.id = ?
            """,
            (contradiction_id,),
        )
        self.record_history(
            entity_kind="contradiction",
            entity_id=contradiction_id,
            subject_key=subject_key,
            action="inserted",
            reason=resolution,
            source="fact",
            payload=row or {},
        )
        return row or {
            "id": contradiction_id,
            "subject_key": subject_key,
            "winner_fact_id": winner_fact_id,
            "loser_fact_id": loser_fact_id,
            "resolution": resolution,
            "created_at": created_at,
        }

    def deactivate_fact(self, fact_id: int, *, reason: str = "manual", source: str = "tool") -> bool:
        row = self._fetchone("SELECT id, subject_key FROM facts WHERE id = ? AND active = 1", (int(fact_id),))
        if not row:
            return False
        self._execute(
            "UPDATE facts SET active = 0, updated_at = ? WHERE id = ?",
            (now_ts(), int(fact_id)),
        )
        self._delete_fact_fts(int(fact_id))
        self.record_history(
            entity_kind="fact",
            entity_id=int(fact_id),
            subject_key=str(row.get("subject_key") or ""),
            action="deactivated",
            reason=reason,
            source=source,
            payload={"fact_id": int(fact_id)},
        )
        return True

    def deactivate_memory_item(
        self,
        memory_type: str,
        entry_id: int,
        *,
        reason: str = "manual",
        source: str = "tool",
    ) -> bool:
        kind = normalize_whitespace(memory_type)
        if kind == "fact":
            return self.deactivate_fact(entry_id, reason=reason, source=source)
        table_map = {
            "journal": ("memory_journals", "journal"),
            "summary": ("memory_summaries", "summary"),
            "preference": ("memory_preferences", "preference"),
            "policy": ("memory_policies", "policy"),
        }
        table_info = table_map.get(kind)
        if not table_info:
            return False
        table, entity_kind = table_info
        row = self._fetchone(f"SELECT id FROM {table} WHERE id = ? AND active = 1", (int(entry_id),))
        if not row:
            return False
        self._execute(f"UPDATE {table} SET active = 0, updated_at = ? WHERE id = ?", (now_ts(), int(entry_id)))
        if kind == "journal":
            self._delete_journal_fts(int(entry_id))
        elif kind == "summary":
            self._delete_summary_fts(int(entry_id))
        elif kind == "preference":
            self._delete_preference_fts(int(entry_id))
        elif kind == "policy":
            self._delete_policy_fts(int(entry_id))
        self.record_history(
            entity_kind=entity_kind,
            entity_id=int(entry_id),
            action="deactivated",
            reason=reason,
            source=source,
            payload={"id": int(entry_id)},
        )
        return True

    def deactivate_matching(self, query: str, limit: int = 10) -> int:
        clean = normalize_whitespace(query)
        if not clean:
            return 0
        rows = self._fetchall(
            """
            SELECT id
            FROM facts
            WHERE active = 1
              AND (content LIKE ? OR normalized_content LIKE ?)
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (f"%{clean}%", f"%{normalize_text(clean)}%", int(limit)),
        )
        count = 0
        for row in rows:
            if self.deactivate_fact(int(row["id"])):
                count += 1
        return count

    def prune_stale_facts(self, max_age_days: int = 90) -> int:
        cutoff = now_ts() - (int(max_age_days) * 86400)
        rows = self._fetchall(
            """
            SELECT id
            FROM facts
            WHERE active = 1
              AND importance <= 4
              AND category = 'general'
              AND source = 'episode_extract'
              AND updated_at < ?
            """,
            (cutoff,),
        )
        count = 0
        for row in rows:
            if self.deactivate_fact(int(row["id"]), reason="prune", source="consolidation"):
                count += 1
        return count

    def rebuild_topics(self, *, max_facts: int = 5, max_chars: int = 650) -> int:
        rows = self._fetchall(
            """
            SELECT *
            FROM facts
            WHERE active = 1
            ORDER BY topic ASC, salience DESC, importance DESC, updated_at DESC
            """
        )
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["topic"])].append(row)

        self._execute("DELETE FROM topic_membership")
        live_slugs = set(grouped.keys())

        for slug, facts in grouped.items():
            top_facts = facts[: max(1, int(max_facts))]
            pieces: List[str] = []
            seen = set()
            for fact in top_facts:
                content = normalize_whitespace(str(fact["content"]))
                if content in seen:
                    continue
                seen.add(content)
                next_summary = " | ".join(pieces + [content])
                if len(next_summary) > int(max_chars):
                    break
                pieces.append(content)
            summary = " | ".join(pieces)[: int(max_chars)]
            category = str(top_facts[0]["category"]) if top_facts else "general"
            importance = max(int(fact["importance"]) for fact in top_facts) if top_facts else 5
            salience = max(float(fact.get("salience") or 0.0) for fact in top_facts) if top_facts else 0.55
            updated_at = max(float(fact["updated_at"]) for fact in top_facts) if top_facts else now_ts()
            source_session_id = next((str(fact.get("source_session_id") or "") for fact in top_facts if fact.get("source_session_id")), "")
            decay_half_life_days = max(float(fact.get("decay_half_life_days") or 0.0) for fact in top_facts) if top_facts else 60.0
            title = pretty_topic(slug)
            existing = self._fetchone("SELECT * FROM topics WHERE slug = ?", (slug,))
            if existing:
                self._execute(
                    """
                    UPDATE topics
                    SET title = ?, category = ?, summary = ?, metadata_json = '{}', importance = ?, salience = ?, source_session_id = ?, decay_half_life_days = ?, updated_at = ?
                    WHERE slug = ?
                    """,
                    (title, category, summary, int(importance), salience, source_session_id, decay_half_life_days, updated_at, slug),
                )
                topic_id = int(existing["id"])
                action = "updated"
            else:
                cur = self._execute(
                    """
                    INSERT INTO topics(slug, title, category, summary, metadata_json, importance, salience, source_session_id, last_recalled_at, decay_half_life_days, updated_at)
                    VALUES (?, ?, ?, ?, '{}', ?, ?, ?, 0, ?, ?)
                    """,
                    (slug, title, category, summary, int(importance), salience, source_session_id, decay_half_life_days, updated_at),
                )
                topic_id = int(cur.lastrowid)
                action = "inserted"
            topic_row = self._fetchone(
                "SELECT id, slug, title, summary, category, importance, salience, source_session_id, last_recalled_at, decay_half_life_days, updated_at FROM topics WHERE slug = ?",
                (slug,),
            ) or {}
            if topic_row:
                self._upsert_topic_fts(topic_row)
                self.delete_links(source_kind="topic", source_id=topic_id, link_types=("supports",))
                for fact in top_facts:
                    self._execute(
                        "INSERT OR IGNORE INTO topic_membership(topic_id, fact_id) VALUES (?, ?)",
                        (int(topic_row["id"]), int(fact["id"])),
                    )
                    self.add_link("topic", topic_row["id"], "fact", fact["id"], "supports")
                self.record_history(
                    entity_kind="topic",
                    entity_id=topic_id,
                    action=action,
                    reason="rebuild_topics",
                    source="consolidation",
                    payload=topic_row,
                )

        stale_topics = self._fetchall("SELECT id, slug FROM topics")
        for row in stale_topics:
            if str(row["slug"]) in live_slugs:
                continue
            self._execute("DELETE FROM topics WHERE id = ?", (int(row["id"]),))
            self._delete_topic_fts(int(row["id"]))
            self.delete_links(source_kind="topic", source_id=int(row["id"]))
            self.delete_links(target_kind="topic", target_id=int(row["id"]))

        return len(grouped)

    def recent_items(self, *, limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "facts": self._fetchall(
                """
                SELECT id, content, category, topic, importance, confidence, salience, updated_at, subject_key, value_key, polarity, exclusive, source_session_id
                FROM facts
                WHERE active = 1
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ),
            "topics": self._fetchall(
                """
                SELECT id, slug, title, summary, category, importance, salience, updated_at, source_session_id
                FROM topics
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ),
            "episodes": self._fetchall(
                """
                SELECT id, session_id, digest, topic_hint, created_at
                FROM episodes
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ),
            "summaries": self._fetchall(
                """
                SELECT id, session_id, label, summary, summary_type, importance, salience, updated_at
                FROM memory_summaries
                WHERE active = 1
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ),
            "journals": self._fetchall(
                """
                SELECT id, session_id, label, content, journal_type, importance, salience, updated_at
                FROM memory_journals
                WHERE active = 1
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ),
            "preferences": self._fetchall(
                """
                SELECT id, preference_key, label, value, content, importance, salience, updated_at
                FROM memory_preferences
                WHERE active = 1
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ),
            "policies": self._fetchall(
                """
                SELECT id, policy_key, label, content, importance, salience, updated_at
                FROM memory_policies
                WHERE active = 1
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ),
            "contradictions": self.recent_contradictions(limit=limit),
        }

    def scoped_recent_items(self, *, scope: str = "all", limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        recent = self.recent_items(limit=limit)
        if scope == "all":
            return {name: recent.get(name, []) for name in self.SEARCH_SCOPES}
        return {name: (recent.get(name, []) if name == scope else []) for name in self.SEARCH_SCOPES}

    def recent_contradictions(self, *, limit: int = 5, max_age_days: int | None = None) -> List[Dict[str, Any]]:
        params: List[Any] = []
        where = ""
        if max_age_days is not None:
            where = "WHERE c.created_at >= ?"
            params.append(now_ts() - (int(max_age_days) * 86400))
        params.append(int(limit))
        return self._fetchall(
            f"""
            SELECT c.id,
                   c.subject_key,
                   c.resolution,
                   c.created_at,
                   w.id AS winner_fact_id,
                   w.content AS winner_content,
                   w.topic AS winner_topic,
                   l.id AS loser_fact_id,
                   l.content AS loser_content,
                   l.topic AS loser_topic
            FROM contradictions c
            LEFT JOIN facts w ON w.id = c.winner_fact_id
            LEFT JOIN facts l ON l.id = c.loser_fact_id
            {where}
            ORDER BY c.id DESC
            LIMIT ?
            """,
            params,
        )

    def search(
        self,
        query: str,
        *,
        scope: str = "all",
        limit: int = 8,
        include_inactive: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        clean = normalize_whitespace(query)
        if not clean:
            return self.scoped_recent_items(scope=scope, limit=limit)
        results = {name: [] for name in self.SEARCH_SCOPES}
        if scope in ("all", "facts"):
            results["facts"] = self._search_facts(clean, limit=limit, include_inactive=include_inactive)
        if scope in ("all", "topics"):
            results["topics"] = self._search_topics(clean, limit=max(1, min(limit, 6)))
        if scope in ("all", "episodes"):
            results["episodes"] = self._search_episodes(clean, limit=max(1, min(limit, 6)))
        if scope in ("all", "summaries"):
            results["summaries"] = self._search_summaries(clean, limit=max(1, min(limit, 6)), include_inactive=include_inactive)
        if scope in ("all", "journals"):
            results["journals"] = self._search_journals(clean, limit=max(1, min(limit, 6)), include_inactive=include_inactive)
        if scope in ("all", "preferences"):
            results["preferences"] = self._search_preferences(clean, limit=max(1, min(limit, 6)), include_inactive=include_inactive)
        if scope in ("all", "policies"):
            results["policies"] = self._search_policies(clean, limit=max(1, min(limit, 6)), include_inactive=include_inactive)
        return results

    def _search_facts(self, query: str, *, limit: int, include_inactive: bool) -> List[Dict[str, Any]]:
        active_clause = "" if include_inactive else "AND f.active = 1"
        if self._fts_enabled:
            try:
                return self._fetchall(
                    f"""
                    SELECT f.id, f.content, f.category, f.topic, f.importance, f.confidence, f.salience, f.updated_at, f.subject_key, f.value_key, f.polarity, f.exclusive, f.source_session_id
                    FROM facts_fts idx
                    JOIN facts f ON f.id = idx.fact_id
                    WHERE facts_fts MATCH ?
                      {active_clause}
                    ORDER BY bm25(facts_fts), f.salience DESC, f.importance DESC, f.updated_at DESC
                    LIMIT ?
                    """,
                    (query, int(limit)),
                )
            except sqlite3.OperationalError:
                pass
        like = f"%{query}%"
        active_sql = "" if include_inactive else "AND active = 1"
        return self._fetchall(
            f"""
            SELECT id, content, category, topic, importance, confidence, salience, updated_at, subject_key, value_key, polarity, exclusive, source_session_id
            FROM facts
            WHERE (content LIKE ? OR topic LIKE ? OR category LIKE ? OR subject_key LIKE ?)
              {active_sql}
            ORDER BY salience DESC, importance DESC, updated_at DESC
            LIMIT ?
            """,
            (like, like, like, like, int(limit)),
        )

    def _search_topics(self, query: str, *, limit: int) -> List[Dict[str, Any]]:
        if self._fts_enabled:
            try:
                return self._fetchall(
                    """
                    SELECT t.id, t.slug, t.title, t.summary, t.category, t.importance, t.salience, t.updated_at, t.source_session_id
                    FROM topics_fts idx
                    JOIN topics t ON t.id = idx.topic_id
                    WHERE topics_fts MATCH ?
                    ORDER BY bm25(topics_fts), t.salience DESC, t.importance DESC, t.updated_at DESC
                    LIMIT ?
                    """,
                    (query, int(limit)),
                )
            except sqlite3.OperationalError:
                pass
        like = f"%{query}%"
        return self._fetchall(
            """
            SELECT id, slug, title, summary, category, importance, salience, updated_at, source_session_id
            FROM topics
            WHERE title LIKE ? OR summary LIKE ? OR slug LIKE ?
            ORDER BY salience DESC, importance DESC, updated_at DESC
            LIMIT ?
            """,
            (like, like, like, int(limit)),
        )

    def _search_episodes(self, query: str, *, limit: int) -> List[Dict[str, Any]]:
        if self._fts_enabled:
            try:
                return self._fetchall(
                    """
                    SELECT e.id, e.session_id, e.digest, e.topic_hint, e.created_at
                    FROM episodes_fts idx
                    JOIN episodes e ON e.id = idx.episode_id
                    WHERE episodes_fts MATCH ?
                    ORDER BY bm25(episodes_fts), e.created_at DESC
                    LIMIT ?
                    """,
                    (query, int(limit)),
                )
            except sqlite3.OperationalError:
                pass
        like = f"%{query}%"
        return self._fetchall(
            """
            SELECT id, session_id, digest, topic_hint, created_at
            FROM episodes
            WHERE digest LIKE ? OR user_content LIKE ? OR assistant_content LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (like, like, like, int(limit)),
        )

    def _search_summaries(self, query: str, *, limit: int, include_inactive: bool) -> List[Dict[str, Any]]:
        active_clause = "" if include_inactive else "AND s.active = 1"
        if self._fts_enabled:
            try:
                return self._fetchall(
                    f"""
                    SELECT s.id, s.session_id, s.label, s.summary, s.summary_type, s.importance, s.salience, s.updated_at
                    FROM memory_summaries_fts idx
                    JOIN memory_summaries s ON s.id = idx.summary_id
                    WHERE memory_summaries_fts MATCH ?
                      {active_clause}
                    ORDER BY bm25(memory_summaries_fts), s.salience DESC, s.importance DESC, s.updated_at DESC
                    LIMIT ?
                    """,
                    (query, int(limit)),
                )
            except sqlite3.OperationalError:
                pass
        like = f"%{query}%"
        active_sql = "" if include_inactive else "AND active = 1"
        return self._fetchall(
            f"""
            SELECT id, session_id, label, summary, summary_type, importance, salience, updated_at
            FROM memory_summaries
            WHERE (label LIKE ? OR summary LIKE ? OR content LIKE ?)
              {active_sql}
            ORDER BY salience DESC, importance DESC, updated_at DESC
            LIMIT ?
            """,
            (like, like, like, int(limit)),
        )

    def _search_journals(self, query: str, *, limit: int, include_inactive: bool) -> List[Dict[str, Any]]:
        active_clause = "" if include_inactive else "AND j.active = 1"
        if self._fts_enabled:
            try:
                return self._fetchall(
                    f"""
                    SELECT j.id, j.session_id, j.label, j.content, j.journal_type, j.importance, j.salience, j.updated_at
                    FROM memory_journals_fts idx
                    JOIN memory_journals j ON j.id = idx.journal_id
                    WHERE memory_journals_fts MATCH ?
                      {active_clause}
                    ORDER BY bm25(memory_journals_fts), j.salience DESC, j.importance DESC, j.updated_at DESC
                    LIMIT ?
                    """,
                    (query, int(limit)),
                )
            except sqlite3.OperationalError:
                pass
        like = f"%{query}%"
        active_sql = "" if include_inactive else "AND active = 1"
        return self._fetchall(
            f"""
            SELECT id, session_id, label, content, journal_type, importance, salience, updated_at
            FROM memory_journals
            WHERE (label LIKE ? OR content LIKE ? OR journal_type LIKE ?)
              {active_sql}
            ORDER BY salience DESC, importance DESC, updated_at DESC
            LIMIT ?
            """,
            (like, like, like, int(limit)),
        )

    def _search_preferences(self, query: str, *, limit: int, include_inactive: bool) -> List[Dict[str, Any]]:
        active_clause = "" if include_inactive else "AND p.active = 1"
        if self._fts_enabled:
            try:
                return self._fetchall(
                    f"""
                    SELECT p.id, p.preference_key, p.label, p.value, p.content, p.source_session_id, p.importance, p.salience, p.updated_at
                    FROM memory_preferences_fts idx
                    JOIN memory_preferences p ON p.id = idx.preference_id
                    WHERE memory_preferences_fts MATCH ?
                      {active_clause}
                    ORDER BY bm25(memory_preferences_fts), p.salience DESC, p.importance DESC, p.updated_at DESC
                    LIMIT ?
                    """,
                    (query, int(limit)),
                )
            except sqlite3.OperationalError:
                pass
        like = f"%{query}%"
        active_sql = "" if include_inactive else "AND active = 1"
        return self._fetchall(
            f"""
            SELECT id, preference_key, label, value, content, source_session_id, importance, salience, updated_at
            FROM memory_preferences
            WHERE (preference_key LIKE ? OR label LIKE ? OR value LIKE ? OR content LIKE ?)
              {active_sql}
            ORDER BY salience DESC, importance DESC, updated_at DESC
            LIMIT ?
            """,
            (like, like, like, like, int(limit)),
        )

    def _search_policies(self, query: str, *, limit: int, include_inactive: bool) -> List[Dict[str, Any]]:
        active_clause = "" if include_inactive else "AND p.active = 1"
        if self._fts_enabled:
            try:
                return self._fetchall(
                    f"""
                    SELECT p.id, p.policy_key, p.label, p.content, p.source_session_id, p.importance, p.salience, p.updated_at
                    FROM memory_policies_fts idx
                    JOIN memory_policies p ON p.id = idx.policy_id
                    WHERE memory_policies_fts MATCH ?
                      {active_clause}
                    ORDER BY bm25(memory_policies_fts), p.salience DESC, p.importance DESC, p.updated_at DESC
                    LIMIT ?
                    """,
                    (query, int(limit)),
                )
            except sqlite3.OperationalError:
                pass
        like = f"%{query}%"
        active_sql = "" if include_inactive else "AND active = 1"
        return self._fetchall(
            f"""
            SELECT id, policy_key, label, content, source_session_id, importance, salience, updated_at
            FROM memory_policies
            WHERE (policy_key LIKE ? OR label LIKE ? OR content LIKE ?)
              {active_sql}
            ORDER BY salience DESC, importance DESC, updated_at DESC
            LIMIT ?
            """,
            (like, like, like, int(limit)),
        )

    def touch_recall(self, kind: str, ids: Sequence[Any], *, session_id: str = "") -> None:
        if not ids:
            return
        now = now_ts()
        clean_kind = normalize_whitespace(kind)
        clean_session = normalize_whitespace(session_id)
        table_map = {
            "fact": ("facts", "id"),
            "topic": ("topics", "id"),
            "summary": ("memory_summaries", "id"),
            "journal": ("memory_journals", "id"),
            "preference": ("memory_preferences", "id"),
            "policy": ("memory_policies", "id"),
            "trace": ("memory_traces", "id"),
        }
        table_info = table_map.get(clean_kind)
        if not table_info:
            return
        table, id_col = table_info
        unique_ids = [str(item) for item in ids if str(item)]
        if not unique_ids:
            return
        for raw_id in unique_ids:
            self._execute(f"UPDATE {table} SET last_recalled_at = ? WHERE {id_col} = ?", (now, raw_id))
            if clean_session:
                self.add_link("session", clean_session, clean_kind, raw_id, "recalls")

    def touch_recall_batch(self, results: Dict[str, List[Dict[str, Any]]], *, session_id: str = "") -> None:
        mapping = {
            "facts": "fact",
            "topics": "topic",
            "summaries": "summary",
            "journals": "journal",
            "preferences": "preference",
            "policies": "policy",
        }
        for section, kind in mapping.items():
            ids = [row.get("id") for row in results.get(section, []) if row.get("id") is not None]
            self.touch_recall(kind, ids, session_id=session_id)

    def last_consolidation(self) -> Dict[str, Any] | None:
        row = self._fetchone(
            """
            SELECT *
            FROM consolidation_runs
            ORDER BY id DESC
            LIMIT 1
            """
        )
        if not row:
            return None
        return row

    def latest_session_summaries(self, limit: int = 5) -> List[Dict[str, Any]]:
        return self._fetchall(
            """
            SELECT id, session_id, label, summary, summary_type, importance, salience, updated_at
            FROM memory_summaries
            WHERE active = 1 AND summary_type = 'session'
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (int(limit),),
        )

    def record_consolidation(
        self,
        *,
        reason: str,
        started_at: float,
        finished_at: float,
        source_episode_id: int,
        stats: Dict[str, Any],
    ) -> None:
        stats_json = json.dumps(stats, sort_keys=True)
        self._execute(
            """
            INSERT INTO consolidation_runs(reason, started_at, finished_at, source_episode_id, stats_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (reason, float(started_at), float(finished_at), int(source_episode_id), stats_json),
        )
        self.set_state("last_consolidated_at", finished_at)
        self.set_state("last_consolidated_episode_id", source_episode_id)
        self.set_state("last_consolidation_stats", stats_json)

    def apply_decay(self, *, half_life_days: float, min_salience: float) -> Dict[str, Any]:
        now = now_ts()
        last_decay_at = float(self.get_state("last_decay_at", "0") or 0)
        half_life = max(float(half_life_days), 0.01)
        threshold = max(float(min_salience), 0.0)
        stats = {
            "facts_decayed": 0,
            "facts_deactivated": 0,
            "topics_decayed": 0,
            "summaries_decayed": 0,
            "summaries_deactivated": 0,
            "journals_decayed": 0,
            "journals_deactivated": 0,
            "traces_decayed": 0,
            "traces_deactivated": 0,
            "preferences_decayed": 0,
            "policies_decayed": 0,
        }

        for row in self._fetchall(
            """
            SELECT id, category, importance, salience, last_recalled_at, last_seen_at, updated_at, decay_half_life_days
            FROM facts
            WHERE active = 1
            """
        ):
            anchor = max(
                float(row.get("updated_at") or 0),
                float(row.get("last_seen_at") or 0),
                float(row.get("last_recalled_at") or 0),
                last_decay_at,
            )
            age_days = max((now - anchor) / 86400.0, 0.0)
            item_half_life = max(float(row.get("decay_half_life_days") or half_life), 0.01)
            next_salience = max(0.01, float(row.get("salience") or 0.0) * math.pow(0.5, age_days / item_half_life))
            self._execute("UPDATE facts SET salience = ? WHERE id = ?", (next_salience, int(row["id"])))
            stats["facts_decayed"] += 1
            if next_salience < threshold and int(row.get("importance") or 0) <= 4 and str(row.get("category") or "") == "general":
                if self.deactivate_fact(int(row["id"]), reason="decay", source="decay"):
                    stats["facts_deactivated"] += 1

        stats["topics_decayed"] = self._decay_table("topics", now=now, half_life=half_life, threshold=threshold, last_decay_at=last_decay_at)
        journal_stats = self._decay_table("memory_journals", now=now, half_life=half_life, threshold=threshold, last_decay_at=last_decay_at, deactivate=True, max_keep_importance=5)
        trace_stats = self._decay_table("memory_traces", now=now, half_life=half_life, threshold=threshold, last_decay_at=last_decay_at, deactivate=True, max_keep_importance=4)
        summary_stats = self._decay_table("memory_summaries", now=now, half_life=half_life, threshold=threshold, last_decay_at=last_decay_at, deactivate=True, max_keep_importance=5)
        stats["journals_decayed"], stats["journals_deactivated"] = journal_stats
        stats["traces_decayed"], stats["traces_deactivated"] = trace_stats
        stats["summaries_decayed"], stats["summaries_deactivated"] = summary_stats
        stats["preferences_decayed"] = self._decay_table("memory_preferences", now=now, half_life=max(half_life * 2.0, 1.0), threshold=0.0, last_decay_at=last_decay_at)
        stats["policies_decayed"] = self._decay_table("memory_policies", now=now, half_life=max(half_life * 3.0, 1.0), threshold=0.0, last_decay_at=last_decay_at)
        self.set_state("last_decay_at", now)
        self.set_state("last_decay_stats", json.dumps(stats, sort_keys=True))
        return stats

    def _decay_table(
        self,
        table: str,
        *,
        now: float,
        half_life: float,
        threshold: float,
        last_decay_at: float,
        deactivate: bool = False,
        max_keep_importance: int = 0,
    ) -> int | tuple[int, int]:
        if table == "topics":
            rows = self._fetchall("SELECT id, salience, last_recalled_at, updated_at FROM topics")
        else:
            rows = self._fetchall(
                f"""
                SELECT id, salience, importance, last_recalled_at, updated_at
                FROM {table}
                WHERE active = 1
                """
            )
        changed = 0
        deactivated = 0
        for row in rows:
            anchor = max(float(row.get("updated_at") or 0), float(row.get("last_recalled_at") or 0), float(last_decay_at or 0))
            age_days = max((now - anchor) / 86400.0, 0.0)
            next_salience = max(0.01, float(row.get("salience") or 0.0) * math.pow(0.5, age_days / max(half_life, 0.01)))
            self._execute(f"UPDATE {table} SET salience = ? WHERE id = ?", (next_salience, int(row["id"])))
            changed += 1
            if deactivate and next_salience < threshold and int(row.get("importance") or 0) <= max_keep_importance:
                self._execute(f"UPDATE {table} SET active = 0, updated_at = ? WHERE id = ?", (now, int(row["id"])))
                deactivated += 1
        if deactivate:
            return changed, deactivated
        return changed

    def _upsert_fact_fts(self, fact: Dict[str, Any]) -> None:
        if not self._fts_enabled:
            return
        self._delete_fact_fts(int(fact["id"]))
        self._execute(
            "INSERT INTO facts_fts(fact_id, content, topic, category) VALUES (?, ?, ?, ?)",
            (int(fact["id"]), fact["content"], fact["topic"], fact["category"]),
        )

    def _delete_fact_fts(self, fact_id: int) -> None:
        if self._fts_enabled:
            self._execute("DELETE FROM facts_fts WHERE fact_id = ?", (int(fact_id),))

    def _upsert_topic_fts(self, topic: Dict[str, Any]) -> None:
        if not self._fts_enabled:
            return
        self._delete_topic_fts(int(topic["id"]))
        self._execute(
            "INSERT INTO topics_fts(topic_id, title, summary, category) VALUES (?, ?, ?, ?)",
            (int(topic["id"]), topic["title"], topic["summary"], topic["category"]),
        )

    def _delete_topic_fts(self, topic_id: int) -> None:
        if self._fts_enabled:
            self._execute("DELETE FROM topics_fts WHERE topic_id = ?", (int(topic_id),))

    def _upsert_episode_fts(self, *, episode_id: int, digest: str, user_content: str, assistant_content: str) -> None:
        if not self._fts_enabled:
            return
        self._delete_episode_fts(int(episode_id))
        self._execute(
            """
            INSERT INTO episodes_fts(episode_id, digest, user_content, assistant_content)
            VALUES (?, ?, ?, ?)
            """,
            (int(episode_id), digest, user_content, assistant_content),
        )

    def _delete_episode_fts(self, episode_id: int) -> None:
        if self._fts_enabled:
            self._execute("DELETE FROM episodes_fts WHERE episode_id = ?", (int(episode_id),))

    def _upsert_summary_fts(self, summary: Dict[str, Any]) -> None:
        if not self._fts_enabled:
            return
        self._delete_summary_fts(int(summary["id"]))
        self._execute(
            """
            INSERT INTO memory_summaries_fts(summary_id, label, summary, content, summary_type)
            VALUES (?, ?, ?, ?, ?)
            """,
            (int(summary["id"]), summary["label"], summary["summary"], summary.get("content", ""), summary["summary_type"]),
        )

    def _delete_summary_fts(self, summary_id: int) -> None:
        if self._fts_enabled:
            self._execute("DELETE FROM memory_summaries_fts WHERE summary_id = ?", (int(summary_id),))

    def _upsert_journal_fts(self, journal: Dict[str, Any]) -> None:
        if not self._fts_enabled:
            return
        self._delete_journal_fts(int(journal["id"]))
        self._execute(
            """
            INSERT INTO memory_journals_fts(journal_id, label, content, journal_type)
            VALUES (?, ?, ?, ?)
            """,
            (int(journal["id"]), journal["label"], journal["content"], journal["journal_type"]),
        )

    def _delete_journal_fts(self, journal_id: int) -> None:
        if self._fts_enabled:
            self._execute("DELETE FROM memory_journals_fts WHERE journal_id = ?", (int(journal_id),))

    def _upsert_preference_fts(self, preference: Dict[str, Any]) -> None:
        if not self._fts_enabled:
            return
        self._delete_preference_fts(int(preference["id"]))
        self._execute(
            """
            INSERT INTO memory_preferences_fts(preference_id, preference_key, label, value, content)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                int(preference["id"]),
                preference["preference_key"],
                preference["label"],
                preference["value"],
                preference["content"],
            ),
        )

    def _delete_preference_fts(self, preference_id: int) -> None:
        if self._fts_enabled:
            self._execute("DELETE FROM memory_preferences_fts WHERE preference_id = ?", (int(preference_id),))

    def _upsert_policy_fts(self, policy: Dict[str, Any]) -> None:
        if not self._fts_enabled:
            return
        self._delete_policy_fts(int(policy["id"]))
        self._execute(
            """
            INSERT INTO memory_policies_fts(policy_id, policy_key, label, content)
            VALUES (?, ?, ?, ?)
            """,
            (int(policy["id"]), policy["policy_key"], policy["label"], policy["content"]),
        )

    def _delete_policy_fts(self, policy_id: int) -> None:
        if self._fts_enabled:
            self._execute("DELETE FROM memory_policies_fts WHERE policy_id = ?", (int(policy_id),))

    def _upsert_trace_fts(self, trace: Dict[str, Any]) -> None:
        if not self._fts_enabled:
            return
        self._delete_trace_fts(int(trace["id"]))
        self._execute(
            """
            INSERT INTO memory_traces_fts(trace_id, label, content, trace_type)
            VALUES (?, ?, ?, ?)
            """,
            (int(trace["id"]), trace["label"], trace["content"], trace["trace_type"]),
        )

    def _delete_trace_fts(self, trace_id: int) -> None:
        if self._fts_enabled:
            self._execute("DELETE FROM memory_traces_fts WHERE trace_id = ?", (int(trace_id),))
