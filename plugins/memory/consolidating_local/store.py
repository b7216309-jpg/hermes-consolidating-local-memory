from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
    metadata_json = data.pop("metadata_json", None)
    if metadata_json is not None:
        try:
            data["metadata"] = json.loads(metadata_json)
        except Exception:
            data["metadata"] = {}
    return data


class MemoryStore:
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
                active INTEGER NOT NULL DEFAULT 1,
                superseded_by INTEGER,
                subject_key TEXT NOT NULL DEFAULT '',
                value_key TEXT NOT NULL DEFAULT '',
                polarity INTEGER NOT NULL DEFAULT 1,
                exclusive INTEGER NOT NULL DEFAULT 0,
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
        ]
        for sql in schema:
            self._execute(sql)
        self._ensure_column("facts", "subject_key", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("facts", "value_key", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("facts", "polarity", "INTEGER NOT NULL DEFAULT 1")
        self._ensure_column("facts", "exclusive", "INTEGER NOT NULL DEFAULT 0")
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
        facts = self._fetchone(
            "SELECT COUNT(*) AS count FROM facts WHERE active = 1"
        ) or {"count": 0}
        topics = self._fetchone("SELECT COUNT(*) AS count FROM topics") or {"count": 0}
        episodes = self._fetchone("SELECT COUNT(*) AS count FROM episodes") or {"count": 0}
        contradictions = self._fetchone("SELECT COUNT(*) AS count FROM contradictions") or {"count": 0}
        return {
            "facts": int(facts["count"]),
            "topics": int(topics["count"]),
            "episodes": int(episodes["count"]),
            "contradictions": int(contradictions["count"]),
        }

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
        digest_source = normalize_whitespace(f"{user_content} {assistant_content}")[:240]
        digest = digest_source or "(empty turn)"
        cur = self._execute(
            """
            INSERT INTO episodes(session_id, user_content, assistant_content, digest, topic_hint, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, user_content or "", assistant_content or "", digest, topic_hint or "", created_at),
        )
        episode_id = int(cur.lastrowid)
        self._upsert_episode_fts(
            episode_id=episode_id,
            digest=digest,
            user_content=user_content or "",
            assistant_content=assistant_content or "",
        )
        return {
            "id": episode_id,
            "session_id": session_id,
            "digest": digest,
            "topic_hint": topic_hint or "",
            "created_at": created_at,
        }

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
    ) -> Dict[str, Any]:
        clean = normalize_whitespace(content)
        if not clean:
            raise ValueError("Fact content cannot be empty.")
        observed_at = float(observed_at or now_ts())
        meta = dict(metadata or {})
        subject_key = normalize_whitespace(str(meta.get("subject_key") or ""))
        value_key = normalize_text(str(meta.get("value_key") or "")) if meta.get("value_key") else ""
        polarity = -1 if str(meta.get("polarity", 1)).strip() in {"-1", "false", "neg"} else 1
        exclusive = 1 if subject_key and bool(meta.get("exclusive")) else 0
        fingerprint = fingerprint_text(clean)
        signature = text_signature(clean)
        topic_slug = slugify(topic)
        metadata_json = json.dumps(meta, sort_keys=True)

        existing = self._fetchone(
            """
            SELECT *
            FROM facts
            WHERE fingerprint = ?
            """,
            (fingerprint,),
        )
        if existing:
            self._execute(
                """
                UPDATE facts
                SET active = 1,
                    importance = MAX(importance, ?),
                    confidence = MAX(confidence, ?),
                    updated_at = ?,
                    last_seen_at = ?,
                    metadata_json = ?,
                    subject_key = ?,
                    value_key = ?,
                    polarity = ?,
                    exclusive = ?
                WHERE id = ?
                """,
                (
                    int(importance),
                    float(confidence),
                    observed_at,
                    observed_at,
                    metadata_json,
                    subject_key or str(existing.get("subject_key") or ""),
                    value_key or str(existing.get("value_key") or ""),
                    polarity if subject_key else int(existing.get("polarity") or 1),
                    exclusive if subject_key else int(existing.get("exclusive") or 0),
                    int(existing["id"]),
                ),
            )
            updated = self._fetchone("SELECT * FROM facts WHERE id = ?", (int(existing["id"]),)) or {}
            self._upsert_fact_fts(updated)
            contradictions = self._resolve_subject_state(updated)
            return {"action": "updated", "fact": updated, "superseded": contradictions["superseded"], "contradictions": contradictions["contradictions"]}

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
                active,
                subject_key,
                value_key,
                polarity,
                exclusive,
                created_at,
                updated_at,
                last_seen_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
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
                subject_key,
                value_key,
                int(polarity),
                int(exclusive),
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
            self._soft_supersede_fact(fact_id, int(new_fact["id"]), float(new_fact["updated_at"]))
            superseded_ids.append(fact_id)
        return superseded_ids

    def _soft_supersede_fact(self, fact_id: int, winner_id: int, updated_at: float) -> None:
        self._execute(
            "UPDATE facts SET active = 0, superseded_by = ?, updated_at = ? WHERE id = ?",
            (int(winner_id), float(updated_at), int(fact_id)),
        )
        self._delete_fact_fts(int(fact_id))

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
            self._soft_supersede_fact(int(row["id"]), int(new_fact["id"]), float(new_fact["updated_at"]))
            superseded.append(int(row["id"]))
            if contradictory:
                contradictions.append(
                    self._record_contradiction(
                        subject_key=subject_key,
                        winner_fact_id=int(new_fact["id"]),
                        loser_fact_id=int(row["id"]),
                        resolution=(
                            f"subject={subject_key}; old={row_value or row_polarity}; "
                            f"new={new_value or new_polarity}"
                        ),
                    )
                )
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
        return row or {
            "id": contradiction_id,
            "subject_key": subject_key,
            "winner_fact_id": winner_fact_id,
            "loser_fact_id": loser_fact_id,
            "resolution": resolution,
            "created_at": created_at,
        }

    def deactivate_fact(self, fact_id: int) -> bool:
        row = self._fetchone("SELECT id FROM facts WHERE id = ? AND active = 1", (int(fact_id),))
        if not row:
            return False
        self._execute(
            "UPDATE facts SET active = 0, updated_at = ? WHERE id = ?",
            (now_ts(), int(fact_id)),
        )
        self._delete_fact_fts(int(fact_id))
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
            if self.deactivate_fact(int(row["id"])):
                count += 1
        return count

    def rebuild_topics(self, *, max_facts: int = 5, max_chars: int = 650) -> int:
        rows = self._fetchall(
            """
            SELECT *
            FROM facts
            WHERE active = 1
            ORDER BY topic ASC, importance DESC, updated_at DESC
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
                if sum(len(piece) for piece in pieces) + len(content) > int(max_chars):
                    break
                pieces.append(content)
            summary = " | ".join(pieces)[: int(max_chars)]
            category = str(top_facts[0]["category"]) if top_facts else "general"
            importance = max(int(fact["importance"]) for fact in top_facts) if top_facts else 5
            updated_at = max(float(fact["updated_at"]) for fact in top_facts) if top_facts else now_ts()
            title = pretty_topic(slug)
            self._execute(
                """
                INSERT INTO topics(slug, title, category, summary, metadata_json, importance, updated_at)
                VALUES (?, ?, ?, ?, '{}', ?, ?)
                ON CONFLICT(slug) DO UPDATE SET
                    title = excluded.title,
                    category = excluded.category,
                    summary = excluded.summary,
                    importance = excluded.importance,
                    updated_at = excluded.updated_at
                """,
                (slug, title, category, summary, int(importance), float(updated_at)),
            )
            topic_row = self._fetchone("SELECT id, slug, title, summary, category, importance, updated_at FROM topics WHERE slug = ?", (slug,)) or {}
            if topic_row:
                self._upsert_topic_fts(topic_row)
                for fact in top_facts:
                    self._execute(
                        "INSERT OR IGNORE INTO topic_membership(topic_id, fact_id) VALUES (?, ?)",
                        (int(topic_row["id"]), int(fact["id"])),
                    )

        stale_topics = self._fetchall("SELECT id, slug FROM topics")
        for row in stale_topics:
            if str(row["slug"]) in live_slugs:
                continue
            self._execute("DELETE FROM topics WHERE id = ?", (int(row["id"]),))
            self._delete_topic_fts(int(row["id"]))

        return len(grouped)

    def recent_items(self, *, limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        facts = self._fetchall(
            """
            SELECT id, content, category, topic, importance, confidence, updated_at, subject_key, value_key, polarity, exclusive
            FROM facts
            WHERE active = 1
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        topics = self._fetchall(
            """
            SELECT id, slug, title, summary, category, importance, updated_at
            FROM topics
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        episodes = self._fetchall(
            """
            SELECT id, session_id, digest, topic_hint, created_at
            FROM episodes
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        contradictions = self.recent_contradictions(limit=limit)
        return {"facts": facts, "topics": topics, "episodes": episodes, "contradictions": contradictions}

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

    def search(self, query: str, *, scope: str = "all", limit: int = 8) -> Dict[str, List[Dict[str, Any]]]:
        clean = normalize_whitespace(query)
        if not clean:
            return self.recent_items(limit=limit)
        results = {"facts": [], "topics": [], "episodes": []}
        if scope in ("all", "facts"):
            results["facts"] = self._search_facts(clean, limit=limit)
        if scope in ("all", "topics"):
            results["topics"] = self._search_topics(clean, limit=max(3, min(limit, 6)))
        if scope in ("all", "episodes"):
            results["episodes"] = self._search_episodes(clean, limit=max(3, min(limit, 6)))
        return results

    def _search_facts(self, query: str, *, limit: int) -> List[Dict[str, Any]]:
        if self._fts_enabled:
            try:
                return self._fetchall(
                    """
                    SELECT f.id, f.content, f.category, f.topic, f.importance, f.confidence, f.updated_at, f.subject_key, f.value_key, f.polarity, f.exclusive
                    FROM facts_fts idx
                    JOIN facts f ON f.id = idx.fact_id
                    WHERE f.active = 1
                      AND facts_fts MATCH ?
                    ORDER BY bm25(facts_fts), f.importance DESC, f.updated_at DESC
                    LIMIT ?
                    """,
                    (query, int(limit)),
                )
            except sqlite3.OperationalError:
                pass
        like = f"%{query}%"
        return self._fetchall(
            """
            SELECT id, content, category, topic, importance, confidence, updated_at, subject_key, value_key, polarity, exclusive
            FROM facts
            WHERE active = 1
              AND (content LIKE ? OR topic LIKE ? OR category LIKE ?)
            ORDER BY importance DESC, updated_at DESC
            LIMIT ?
            """,
            (like, like, like, int(limit)),
        )

    def _search_topics(self, query: str, *, limit: int) -> List[Dict[str, Any]]:
        if self._fts_enabled:
            try:
                return self._fetchall(
                    """
                    SELECT t.id, t.slug, t.title, t.summary, t.category, t.importance, t.updated_at
                    FROM topics_fts idx
                    JOIN topics t ON t.id = idx.topic_id
                    WHERE topics_fts MATCH ?
                    ORDER BY bm25(topics_fts), t.importance DESC, t.updated_at DESC
                    LIMIT ?
                    """,
                    (query, int(limit)),
                )
            except sqlite3.OperationalError:
                pass
        like = f"%{query}%"
        return self._fetchall(
            """
            SELECT id, slug, title, summary, category, importance, updated_at
            FROM topics
            WHERE title LIKE ? OR summary LIKE ? OR slug LIKE ?
            ORDER BY importance DESC, updated_at DESC
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
        try:
            row["stats"] = json.loads(row.pop("stats_json"))
        except Exception:
            row["stats"] = {}
        return row

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

    def _upsert_episode_fts(
        self,
        *,
        episode_id: int,
        digest: str,
        user_content: str,
        assistant_content: str,
    ) -> None:
        if not self._fts_enabled:
            return
        self._execute("DELETE FROM episodes_fts WHERE episode_id = ?", (int(episode_id),))
        self._execute(
            """
            INSERT INTO episodes_fts(episode_id, digest, user_content, assistant_content)
            VALUES (?, ?, ?, ?)
            """,
            (int(episode_id), digest, user_content, assistant_content),
        )
