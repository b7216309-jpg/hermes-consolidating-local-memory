from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
import threading
import time
import unicodedata
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
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
    "user",
    "users",
    "using",
    "we",
    "with",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "how",
    "know",
    "about",
    "you",
    "your",
}


def now_ts() -> float:
    return time.time()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_text(text: str) -> str:
    clean = unicodedata.normalize("NFKC", normalize_whitespace(text)).casefold()
    clean = "".join(char for char in clean if char.isalnum() or char.isspace() or char in "-_/.:")
    return normalize_whitespace(clean)


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", normalize_text(text))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    slug = re.sub(r"[^\w]+", "-", normalized, flags=re.UNICODE)
    slug = slug.strip("-_")
    return slug or "general"


def pretty_topic(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").title()


def fingerprint_text(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def text_signature(text: str) -> str:
    tokens = re.findall(r"\w+", normalize_text(text), flags=re.UNICODE)
    keep = [token for token in tokens if token not in STOPWORDS]
    return " ".join(keep[:6])


def fts_query(text: str) -> str:
    """Convert natural-language input into a safe, recall-friendly FTS5 query."""
    tokens = re.findall(r"\w+", normalize_text(text), flags=re.UNICODE)
    useful = [token for token in tokens if token not in STOPWORDS]
    if not useful:
        useful = tokens
    synonyms = {
        "shell": ("powershell", "pwsh", "bash", "zsh", "fish"),
        "database": ("postgresql", "postgres", "mysql", "sqlite", "redis"),
        "editor": ("vscode", "vim", "neovim", "emacs", "sublime"),
        "operating": ("windows", "linux", "macos", "ubuntu", "debian", "wsl"),
    }
    expanded: List[str] = []
    for token in useful:
        expanded.append(token)
        expanded.extend(synonyms.get(token, ()))
    unique = list(dict.fromkeys(expanded))[:20]
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in unique)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(low, min(high, parsed))


def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    return max(low, min(high, parsed))


def _timestamp(value: Any, default: float = 0.0) -> float:
    """Return a finite, non-negative Unix timestamp at the storage boundary."""
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        parsed = float(default)
    if not math.isfinite(parsed) or parsed < 0:
        parsed = float(default)
    return parsed if math.isfinite(parsed) and parsed >= 0 else 0.0


_CREDENTIAL_PATTERNS = (
    r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b",
    r"\bgh[pousr]_[A-Za-z0-9]{20,}\b",
    r"\bAKIA[0-9A-Z]{16}\b",
    r"\bAIza[0-9A-Za-z_-]{20,}\b",
    r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
    r"\bbearer\s+[A-Za-z0-9._~+/=-]{12,}",
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
    r"\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]{4,}@",
    r"\b(?:auth(?:orization)?[_ -]?token|token)\s*(?:is|[:=])\s*['\"]?\S{8,}",
)


def _looks_like_credential(text: str) -> bool:
    raw = str(text or "")
    return any(re.search(pattern, raw, flags=re.IGNORECASE) for pattern in _CREDENTIAL_PATTERNS)


# Subject keys that can hold multiple coexisting values (e.g. WSL = Windows + Linux).
# State resolution is scoped to a value/facet for these keys rather than the whole
# subject.
COEXIST_SUBJECT_KEYS = frozenset(
    {
        "environment:os",
        "environment:shell",
        "user:physical_attributes",
        "user:interest:media",
    }
)

# Subject-key prefixes for transient / stats-like data that should never
# generate contradictions.  These change every session and create noise.
_SUPERSEDE_ONLY_PREFIXES = (
    "workflow:next_step",
    "workflow:current_",
    "project:memory_stats",
    "project:memory_usage",
)


def _transactional(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.transaction():
            return method(self, *args, **kwargs)

    return wrapper


def _row_to_dict(row: Any | None) -> Dict[str, Any] | None:
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


def _first_review_offset_seconds(review_intervals_days: Sequence[float] | None) -> float:
    if not review_intervals_days:
        return 86400.0
    for value in review_intervals_days:
        try:
            clean = float(value)
        except Exception:
            continue
        if clean > 0:
            return clean * 86400.0
    return 86400.0


def _next_review_offset_seconds(review_count: int, review_intervals_days: Sequence[float] | None) -> float:
    if not review_intervals_days:
        return 86400.0
    clean_intervals: List[float] = []
    for value in review_intervals_days:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(parsed) and parsed > 0:
            clean_intervals.append(parsed)
    if not clean_intervals:
        return 86400.0
    index = max(0, min(int(review_count), len(clean_intervals) - 1))
    return clean_intervals[index] * 86400.0


def _looks_sensitive_for_export(row: Dict[str, Any]) -> bool:
    def carries_sensitive_label(value: Any) -> bool:
        if isinstance(value, dict):
            sensitivity = normalize_text(str(value.get("sensitivity") or "normal"))
            if sensitivity != "normal":
                return True
            return any(carries_sensitive_label(child) for child in value.values())
        if isinstance(value, (list, tuple, set)):
            return any(carries_sensitive_label(child) for child in value)
        return False

    if carries_sensitive_label(row):
        return True
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    subject = normalize_text(str(row.get("subject_key") or metadata.get("subject_key") or ""))
    sensitive_fields = {
        key: value
        for key, value in row.items()
        if key
        in {
            "content",
            "summary",
            "value",
            "candidate",
            "intention",
            "condition_text",
            "steps",
            "prerequisites",
            "success_criteria",
            "failure_recovery",
            "payload",
            "metadata",
        }
    }
    content = normalize_text(json.dumps(sensitive_fields, ensure_ascii=False, sort_keys=True, default=str))
    combined = f"{subject} {content}"
    if _looks_like_credential(json.dumps(sensitive_fields, ensure_ascii=False, sort_keys=True, default=str)):
        return True
    return any(
        marker in combined
        for marker in (
            "password",
            "passphrase",
            "api key",
            "access token",
            "private key",
            "medical",
            "diagnosis",
            "medication",
            "financial",
            "bank",
            "iban",
            "credit card",
            "date of birth",
            "passport",
            "social security",
            "exact location",
            "home address",
        )
    )


class MemoryStore:
    _REFERENCE_TABLES = {
        "fact": ("facts", "id"),
        "topic": ("topics", "id"),
        "episode": ("episodes", "id"),
        "session": ("memory_sessions", "session_id"),
        "trace": ("memory_traces", "id"),
        "journal": ("memory_journals", "id"),
        "summary": ("memory_summaries", "id"),
        "preference": ("memory_preferences", "id"),
        "policy": ("memory_policies", "id"),
        "autobiographical_event": ("autobiographical_events", "id"),
        "working": ("working_memory", "id"),
        "procedure": ("memory_procedures", "id"),
        "intention": ("prospective_memories", "id"),
    }
    SEARCH_SCOPES = (
        "facts",
        "topics",
        "episodes",
        "summaries",
        "journals",
        "preferences",
        "policies",
    )

    def __init__(self, db_path: str | Path, *, encryption_key: str = "", conflict_policy: str = "evidence"):
        self.db_path = str(Path(db_path).expanduser())
        db_parent = Path(self.db_path).parent
        db_parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(db_parent, 0o700)
        except OSError:
            pass
        self._lock = threading.RLock()
        self._transaction_depth = 0
        self.conflict_policy = conflict_policy if conflict_policy in {"evidence", "newest"} else "evidence"
        self._encryption_key = str(encryption_key or "")
        self._dbapi = sqlite3
        if encryption_key:
            try:
                from sqlcipher3 import dbapi2 as sqlcipher
            except ImportError as exc:
                raise RuntimeError(
                    "Database encryption was requested, but the optional sqlcipher3 package is not installed"
                ) from exc
            self._dbapi = sqlcipher
        self._operational_errors = tuple(dict.fromkeys((sqlite3.OperationalError, self._dbapi.OperationalError)))
        self._conn = self._dbapi.connect(self.db_path, check_same_thread=False)
        self._closed = False
        try:
            self._conn.row_factory = self._dbapi.Row
            self._conn.create_function("memory_now", 0, now_ts)
            if encryption_key:
                escaped_key = str(encryption_key).replace("'", "''")
                self._conn.execute(f"PRAGMA key = '{escaped_key}'")
                cipher_row = self._conn.execute("PRAGMA cipher_version").fetchone()
                if not cipher_row or not str(cipher_row[0] or "").strip():
                    raise RuntimeError("The selected SQLite driver does not provide SQLCipher encryption")
                self._conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
            self._conn.execute("PRAGMA busy_timeout = 5000")
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA synchronous = NORMAL")
            self._conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except self._operational_errors:
                pass
            self._fts_enabled = False
            self._init_schema()
        except BaseException:
            self._conn.close()
            self._closed = True
            raise
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            pass

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.close()
            self._closed = True

    @contextmanager
    def transaction(self):
        with self._lock:
            outermost = self._transaction_depth == 0
            savepoint = f"memory_sp_{self._transaction_depth}"
            if outermost:
                self._conn.execute("BEGIN IMMEDIATE")
            else:
                self._conn.execute(f"SAVEPOINT {savepoint}")
            self._transaction_depth += 1
            try:
                yield
            except BaseException:
                self._transaction_depth -= 1
                if outermost:
                    self._conn.rollback()
                else:
                    self._conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    self._conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                raise
            else:
                self._transaction_depth -= 1
                if outermost:
                    self._conn.commit()
                else:
                    self._conn.execute(f"RELEASE SAVEPOINT {savepoint}")

    def _execute(self, sql: str, params: Iterable[Any] = ()) -> Any:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            if self._transaction_depth == 0:
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
                review_count INTEGER NOT NULL DEFAULT 0,
                next_review_at REAL NOT NULL DEFAULT 0,
                reconsolidation_until REAL NOT NULL DEFAULT 0,
                decay_half_life_days REAL NOT NULL DEFAULT 45,
                belief_score REAL NOT NULL DEFAULT 0.5,
                observation_count INTEGER NOT NULL DEFAULT 1,
                valid_from REAL NOT NULL DEFAULT 0,
                valid_until REAL NOT NULL DEFAULT 0,
                temporal_kind TEXT NOT NULL DEFAULT 'atemporal',
                event_at REAL NOT NULL DEFAULT 0,
                temporal_precision TEXT NOT NULL DEFAULT 'unknown',
                temporal_timezone TEXT NOT NULL DEFAULT '',
                temporal_confidence REAL NOT NULL DEFAULT 0,
                sensitivity TEXT NOT NULL DEFAULT 'normal',
                memory_class TEXT NOT NULL DEFAULT 'semantic',
                pinned INTEGER NOT NULL DEFAULT 0,
                revision INTEGER NOT NULL DEFAULT 1,
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
                sensitivity TEXT NOT NULL DEFAULT 'normal',
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
                sensitivity TEXT NOT NULL DEFAULT 'normal',
                operation_key TEXT NOT NULL DEFAULT '',
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
                sensitivity TEXT NOT NULL DEFAULT 'normal',
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
                sensitivity TEXT NOT NULL DEFAULT 'normal',
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
                sensitivity TEXT NOT NULL DEFAULT 'normal',
                importance INTEGER NOT NULL DEFAULT 6,
                salience REAL NOT NULL DEFAULT 0.6,
                last_recalled_at REAL NOT NULL DEFAULT 0,
                review_count INTEGER NOT NULL DEFAULT 0,
                next_review_at REAL NOT NULL DEFAULT 0,
                reconsolidation_until REAL NOT NULL DEFAULT 0,
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
                sensitivity TEXT NOT NULL DEFAULT 'normal',
                importance INTEGER NOT NULL DEFAULT 7,
                salience REAL NOT NULL DEFAULT 0.65,
                last_recalled_at REAL NOT NULL DEFAULT 0,
                review_count INTEGER NOT NULL DEFAULT 0,
                next_review_at REAL NOT NULL DEFAULT 0,
                reconsolidation_until REAL NOT NULL DEFAULT 0,
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
                sensitivity TEXT NOT NULL DEFAULT 'normal',
                source_session_id TEXT NOT NULL DEFAULT '',
                importance INTEGER NOT NULL DEFAULT 8,
                salience REAL NOT NULL DEFAULT 0.9,
                last_recalled_at REAL NOT NULL DEFAULT 0,
                review_count INTEGER NOT NULL DEFAULT 0,
                next_review_at REAL NOT NULL DEFAULT 0,
                reconsolidation_until REAL NOT NULL DEFAULT 0,
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
                sensitivity TEXT NOT NULL DEFAULT 'normal',
                source_session_id TEXT NOT NULL DEFAULT '',
                importance INTEGER NOT NULL DEFAULT 9,
                salience REAL NOT NULL DEFAULT 0.95,
                last_recalled_at REAL NOT NULL DEFAULT 0,
                review_count INTEGER NOT NULL DEFAULT 0,
                next_review_at REAL NOT NULL DEFAULT 0,
                reconsolidation_until REAL NOT NULL DEFAULT 0,
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
            """
            CREATE TABLE IF NOT EXISTS belief_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                source_role TEXT NOT NULL DEFAULT 'unknown',
                session_id TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.5,
                reliability REAL NOT NULL DEFAULT 0.5,
                explicit_correction INTEGER NOT NULL DEFAULT 0,
                observed_at REAL NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                FOREIGN KEY(fact_id) REFERENCES facts(id) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_belief_evidence_fact ON belief_evidence(fact_id, observed_at DESC)",
            """
            CREATE TABLE IF NOT EXISTS working_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                content TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 5,
                expires_at REAL NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                sensitivity TEXT NOT NULL DEFAULT 'normal',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(session_id, memory_key)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_working_memory_active ON working_memory(session_id, expires_at, priority DESC)",
            """
            CREATE TABLE IF NOT EXISTS memory_procedures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                procedure_key TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                steps_json TEXT NOT NULL DEFAULT '[]',
                prerequisites_json TEXT NOT NULL DEFAULT '[]',
                success_criteria TEXT NOT NULL DEFAULT '',
                failure_recovery TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.6,
                use_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                last_used_at REAL NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                sensitivity TEXT NOT NULL DEFAULT 'normal',
                active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS prospective_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intention TEXT NOT NULL,
                due_at REAL NOT NULL DEFAULT 0,
                condition_text TEXT NOT NULL DEFAULT '',
                recurrence TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                importance INTEGER NOT NULL DEFAULT 6,
                session_id TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                sensitivity TEXT NOT NULL DEFAULT 'normal',
                last_triggered_at REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_prospective_due ON prospective_memories(status, due_at, importance DESC)",
            """
            CREATE TABLE IF NOT EXISTS autobiographical_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_key TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL,
                event_at REAL NOT NULL DEFAULT 0,
                valid_from REAL NOT NULL DEFAULT 0,
                valid_until REAL NOT NULL DEFAULT 0,
                people_json TEXT NOT NULL DEFAULT '[]',
                places_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                sensitivity TEXT NOT NULL DEFAULT 'normal',
                importance INTEGER NOT NULL DEFAULT 6,
                salience REAL NOT NULL DEFAULT 0.65,
                active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_autobiographical_time ON autobiographical_events(active, event_at DESC)",
            """
            CREATE TABLE IF NOT EXISTS memory_associations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                left_kind TEXT NOT NULL,
                left_id TEXT NOT NULL,
                right_kind TEXT NOT NULL,
                right_id TEXT NOT NULL,
                relation TEXT NOT NULL DEFAULT 'associated',
                weight REAL NOT NULL DEFAULT 0.5,
                cooccurrence_count INTEGER NOT NULL DEFAULT 1,
                last_activated_at REAL NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(left_kind, left_id, right_kind, right_id, relation)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_memory_associations_left ON memory_associations(left_kind, left_id, weight DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memory_associations_right ON memory_associations(right_kind, right_id, weight DESC)",
            """
            CREATE TABLE IF NOT EXISTS memory_approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_json TEXT NOT NULL,
                candidate_fingerprint TEXT NOT NULL DEFAULT '',
                sensitivity TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                session_id TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                resolved_at REAL NOT NULL DEFAULT 0,
                resolution TEXT NOT NULL DEFAULT ''
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_memory_approvals_status ON memory_approvals(status, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memory_approvals_fingerprint ON memory_approvals(status, candidate_fingerprint)",
            """
            CREATE TABLE IF NOT EXISTS pending_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_type TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                available_at REAL NOT NULL DEFAULT 0,
                claimed_at REAL NOT NULL DEFAULT 0,
                error TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_pending_operations_ready ON pending_operations(status, available_at, id)",
            """
            CREATE TABLE IF NOT EXISTS maintenance_leases (
                lease_name TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                expires_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at REAL NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}'
            )
            """,
        ]
        for sql in schema:
            self._execute(sql)

        self._ensure_column("facts", "salience", "REAL NOT NULL DEFAULT 0.55")
        self._ensure_column("facts", "source_session_id", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("facts", "last_recalled_at", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("facts", "review_count", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("facts", "next_review_at", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("facts", "reconsolidation_until", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("facts", "decay_half_life_days", "REAL NOT NULL DEFAULT 45")
        self._ensure_column("facts", "belief_score", "REAL NOT NULL DEFAULT 0.5")
        self._ensure_column("facts", "observation_count", "INTEGER NOT NULL DEFAULT 1")
        self._ensure_column("facts", "valid_from", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("facts", "valid_until", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("facts", "temporal_kind", "TEXT NOT NULL DEFAULT 'atemporal'")
        self._ensure_column("facts", "event_at", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("facts", "temporal_precision", "TEXT NOT NULL DEFAULT 'unknown'")
        self._ensure_column("facts", "temporal_timezone", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("facts", "temporal_confidence", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("facts", "sensitivity", "TEXT NOT NULL DEFAULT 'normal'")
        self._ensure_column("facts", "memory_class", "TEXT NOT NULL DEFAULT 'semantic'")
        self._ensure_column("facts", "pinned", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("facts", "revision", "INTEGER NOT NULL DEFAULT 1")
        self._ensure_column("autobiographical_events", "sensitivity", "TEXT NOT NULL DEFAULT 'normal'")
        self._ensure_column("memory_sessions", "sensitivity", "TEXT NOT NULL DEFAULT 'normal'")
        self._ensure_column("memory_traces", "sensitivity", "TEXT NOT NULL DEFAULT 'normal'")
        self._ensure_column("episodes", "sensitivity", "TEXT NOT NULL DEFAULT 'normal'")
        self._ensure_column("episodes", "operation_key", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("memory_approvals", "candidate_fingerprint", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("topics", "salience", "REAL NOT NULL DEFAULT 0.55")
        self._ensure_column("topics", "source_session_id", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("topics", "last_recalled_at", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("topics", "decay_half_life_days", "REAL NOT NULL DEFAULT 60")
        self._ensure_column("topics", "sensitivity", "TEXT NOT NULL DEFAULT 'normal'")
        self._ensure_column("memory_journals", "review_count", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("memory_journals", "next_review_at", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("memory_journals", "reconsolidation_until", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("memory_journals", "sensitivity", "TEXT NOT NULL DEFAULT 'normal'")
        self._ensure_column("memory_summaries", "review_count", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("memory_summaries", "next_review_at", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("memory_summaries", "reconsolidation_until", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("memory_summaries", "sensitivity", "TEXT NOT NULL DEFAULT 'normal'")
        self._ensure_column("memory_preferences", "source_session_id", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("memory_preferences", "review_count", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("memory_preferences", "next_review_at", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("memory_preferences", "reconsolidation_until", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("memory_preferences", "sensitivity", "TEXT NOT NULL DEFAULT 'normal'")
        self._ensure_column("memory_policies", "source_session_id", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("memory_policies", "review_count", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("memory_policies", "next_review_at", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("memory_policies", "reconsolidation_until", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("memory_policies", "sensitivity", "TEXT NOT NULL DEFAULT 'normal'")
        self._ensure_column("working_memory", "sensitivity", "TEXT NOT NULL DEFAULT 'normal'")
        self._ensure_column("memory_procedures", "sensitivity", "TEXT NOT NULL DEFAULT 'normal'")
        self._ensure_column("prospective_memories", "sensitivity", "TEXT NOT NULL DEFAULT 'normal'")
        self._execute("CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(source_session_id, updated_at DESC)")
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_preferences_session ON memory_preferences(source_session_id, updated_at DESC)"
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_policies_session ON memory_policies(source_session_id, updated_at DESC)"
        )
        self._execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_operation_key ON episodes(operation_key) WHERE operation_key != ''"
        )
        self._execute("CREATE INDEX IF NOT EXISTS idx_facts_temporal ON facts(temporal_kind, event_at DESC, active)")
        evidence_backfilled = self._backfill_belief_evidence()
        self._execute(
            "INSERT OR IGNORE INTO schema_migrations(version, name, applied_at, details_json) VALUES(2, ?, ?, ?)",
            (
                "evidence_temporal_brain_systems",
                now_ts(),
                json.dumps({"rollback": "restore a pre-v2 backup", "evidence_backfilled": evidence_backfilled}),
            ),
        )
        self._backfill_source_sessions("memory_preferences")
        self._backfill_source_sessions("memory_policies")
        self._backfill_memory_sessions()
        self._backfill_review_schedule()
        temporal_migration = self._fetchone("SELECT version FROM schema_migrations WHERE version = 3")
        if not temporal_migration:
            temporal_backfilled = self._backfill_temporal_facts()
            self._execute(
                "INSERT INTO schema_migrations(version, name, applied_at, details_json) VALUES(3, ?, ?, ?)",
                (
                    "structured_temporal_context",
                    now_ts(),
                    json.dumps(
                        {"rollback": "restore a pre-v3 backup", "facts_backfilled": temporal_backfilled},
                        sort_keys=True,
                    ),
                ),
            )
        self._init_fts()

    def _init_fts(self) -> None:
        try:
            with self._lock:
                fact_columns = {
                    str(row["name"]) for row in self._conn.execute("PRAGMA table_info(facts_fts)").fetchall()
                }
            if fact_columns and "subject_key" not in fact_columns:
                self._execute("DROP TABLE facts_fts")
            self._execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(fact_id UNINDEXED, content, topic, category, subject_key)"
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
            self._rebuild_fts_if_needed()
        except self._operational_errors:
            self._fts_enabled = False

    def _rebuild_fts_if_needed(self) -> None:
        """Repair missing/stale FTS rows after upgrades or interrupted writes."""
        if not self._fts_enabled:
            return
        mappings = {
            "facts_fts": ("facts", "active = 1"),
            "topics_fts": ("topics", "1 = 1"),
            "episodes_fts": ("episodes", "1 = 1"),
            "memory_summaries_fts": ("memory_summaries", "active = 1"),
            "memory_journals_fts": ("memory_journals", "active = 1"),
            "memory_preferences_fts": ("memory_preferences", "active = 1"),
            "memory_policies_fts": ("memory_policies", "active = 1"),
            "memory_traces_fts": ("memory_traces", "active = 1"),
        }
        needs_rebuild = self.get_state("fts_schema_version") != "3"
        if not needs_rebuild:
            for index, (table, condition) in mappings.items():
                source = self._fetchone(f"SELECT COUNT(*) AS count FROM {table} WHERE {condition}") or {}
                indexed = self._fetchone(f"SELECT COUNT(*) AS count FROM {index}") or {}
                if int(source.get("count") or 0) != int(indexed.get("count") or 0):
                    needs_rebuild = True
                    break
        if not needs_rebuild:
            return

        for index in mappings:
            self._execute(f"DELETE FROM {index}")
        for row in self._fetchall("SELECT * FROM facts WHERE active = 1"):
            self._upsert_fact_fts(row)
        for row in self._fetchall("SELECT * FROM topics"):
            self._upsert_topic_fts(row)
        for row in self._fetchall("SELECT * FROM episodes"):
            self._upsert_episode_fts(
                episode_id=int(row["id"]),
                digest=str(row.get("digest") or ""),
                user_content=str(row.get("user_content") or ""),
                assistant_content=str(row.get("assistant_content") or ""),
            )
        for row in self._fetchall("SELECT * FROM memory_summaries WHERE active = 1"):
            self._upsert_summary_fts(row)
        for row in self._fetchall("SELECT * FROM memory_journals WHERE active = 1"):
            self._upsert_journal_fts(row)
        for row in self._fetchall("SELECT * FROM memory_preferences WHERE active = 1"):
            self._upsert_preference_fts(row)
        for row in self._fetchall("SELECT * FROM memory_policies WHERE active = 1"):
            self._upsert_policy_fts(row)
        for row in self._fetchall("SELECT * FROM memory_traces WHERE active = 1"):
            self._upsert_trace_fts(row)
        self.set_state("fts_schema_version", "3")

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
        existing_ids = {
            str(row.get("session_id") or "") for row in self._fetchall("SELECT session_id FROM memory_sessions")
        }
        for session_id in sorted(session_ids - existing_ids):
            self.ensure_memory_session(session_id, label=session_id)

    def _backfill_review_schedule(self) -> None:
        default_offset = _first_review_offset_seconds((1.0, 3.0, 7.0, 14.0, 30.0))
        tables = {
            "facts": "SELECT id, COALESCE(last_seen_at, updated_at, created_at, 0) AS anchor FROM facts WHERE COALESCE(next_review_at, 0) <= 0",
            "memory_journals": "SELECT id, COALESCE(updated_at, created_at, 0) AS anchor FROM memory_journals WHERE COALESCE(next_review_at, 0) <= 0",
            "memory_summaries": "SELECT id, COALESCE(updated_at, created_at, 0) AS anchor FROM memory_summaries WHERE COALESCE(next_review_at, 0) <= 0",
            "memory_preferences": "SELECT id, COALESCE(updated_at, created_at, 0) AS anchor FROM memory_preferences WHERE COALESCE(next_review_at, 0) <= 0",
            "memory_policies": "SELECT id, COALESCE(updated_at, created_at, 0) AS anchor FROM memory_policies WHERE COALESCE(next_review_at, 0) <= 0",
        }
        for table, sql in tables.items():
            for row in self._fetchall(sql):
                anchor = float(row.get("anchor") or now_ts())
                self._execute(
                    f"UPDATE {table} SET next_review_at = ? WHERE id = ?",
                    (anchor + default_offset, int(row["id"])),
                )

    def _backfill_temporal_facts(self) -> int:
        """Classify pre-v3 facts without inventing event dates or expiry times."""
        rows = self._fetchall("SELECT * FROM facts")
        updated = 0
        allowed_kinds = {"atemporal", "current", "event", "scheduled", "temporary"}
        allowed_precision = {"unknown", "year", "month", "day", "hour", "minute", "second"}
        for row in rows:
            metadata = dict(row.get("metadata") or {})
            kind = normalize_text(str(metadata.get("temporal_kind") or row.get("temporal_kind") or ""))
            try:
                event_at = float(metadata.get("event_at", row.get("event_at")) or 0.0)
            except (TypeError, ValueError, OverflowError):
                event_at = 0.0
            try:
                valid_until = float(row.get("valid_until") or 0.0)
            except (TypeError, ValueError, OverflowError):
                valid_until = 0.0
            if event_at != event_at or event_at < 0:
                event_at = 0.0
            if valid_until != valid_until or valid_until < 0:
                valid_until = 0.0
            if kind not in allowed_kinds or kind == "atemporal":
                if str(row.get("memory_class") or "") == "autobiographical":
                    kind = "event"
                elif valid_until > 0:
                    kind = "temporary"
                elif int(row.get("exclusive") or 0) == 1:
                    kind = "current"
                else:
                    kind = "atemporal"
            precision = normalize_text(
                str(metadata.get("temporal_precision") or row.get("temporal_precision") or "unknown")
            )
            if precision not in allowed_precision:
                precision = "unknown"
            timezone_name = normalize_whitespace(
                str(metadata.get("temporal_timezone") or row.get("temporal_timezone") or "")
            )[:80]
            temporal_confidence = _clamp_float(
                metadata.get("temporal_confidence", row.get("temporal_confidence")), 0.0, 1.0, 0.0
            )
            metadata["temporal_kind"] = kind
            metadata["temporal_precision"] = precision
            metadata["temporal_confidence"] = temporal_confidence
            if event_at > 0:
                metadata["event_at"] = event_at
            self._execute(
                """UPDATE facts
                   SET temporal_kind=?, event_at=?, temporal_precision=?, temporal_timezone=?,
                       temporal_confidence=?, metadata_json=?
                   WHERE id=?""",
                (
                    kind,
                    event_at,
                    precision,
                    timezone_name,
                    temporal_confidence,
                    json.dumps(metadata, sort_keys=True),
                    int(row["id"]),
                ),
            )
            updated += 1
        return updated

    def _backfill_belief_evidence(self) -> int:
        """Give pre-v2 facts one provenance observation without duplicating it."""
        rows = self._fetchall(
            """SELECT f.* FROM facts f
               WHERE NOT EXISTS(SELECT 1 FROM belief_evidence e WHERE e.fact_id=f.id)"""
        )
        count = 0
        for row in rows:
            metadata = dict(row.get("metadata") or {})
            source = str(row.get("source") or "legacy")
            source_name = normalize_text(source)
            role = normalize_text(str(metadata.get("source_role") or ""))
            if not role:
                if source_name in {"user", "manual", "memory-tool"}:
                    role = "user"
                elif source_name == "tool" or source_name.startswith("builtin_memory"):
                    role = "tool"
                elif source_name in {"llm", "consolidation"}:
                    role = "assistant"
                else:
                    role = "unknown"
            confidence = _clamp_float(row.get("confidence"), 0, 1, 0.7)
            reliability = self._source_reliability(role, source)
            explicit_correction = _as_bool(metadata.get("explicit_correction"))
            observed_at = float(row.get("last_seen_at") or row.get("updated_at") or row.get("created_at") or now_ts())
            self._record_fact_evidence(
                fact_id=int(row["id"]),
                content=str(row.get("content") or ""),
                source=source,
                source_role=role,
                session_id=str(row.get("source_session_id") or ""),
                confidence=confidence,
                reliability=reliability,
                explicit_correction=explicit_correction,
                observed_at=observed_at,
                metadata={**metadata, "migration": "v1_evidence_backfill"},
            )
            belief_score = self._belief_score(
                confidence=confidence,
                reliability=reliability,
                explicit_correction=explicit_correction,
                observations=1,
            )
            self._execute(
                "UPDATE facts SET belief_score=?, observation_count=MAX(1, observation_count) WHERE id=?",
                (belief_score, int(row["id"])),
            )
            count += 1
        return count

    def get_state(self, key: str, default: str = "") -> str:
        row = self._fetchone("SELECT value FROM provider_state WHERE key = ?", (key,))
        if not row:
            return default
        return str(row["value"])

    @_transactional
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
            "facts": """SELECT COUNT(*) AS count FROM facts WHERE active = 1
                        AND (valid_from=0 OR valid_from <= memory_now())
                        AND (valid_until=0 OR valid_until > memory_now())""",
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
            "evidence": "SELECT COUNT(*) AS count FROM belief_evidence",
            "working": "SELECT COUNT(*) AS count FROM working_memory WHERE expires_at=0 OR expires_at > memory_now()",
            "procedures": "SELECT COUNT(*) AS count FROM memory_procedures WHERE active=1",
            "intentions": "SELECT COUNT(*) AS count FROM prospective_memories WHERE status='pending'",
            "autobiographical_events": """SELECT COUNT(*) AS count FROM autobiographical_events WHERE active=1
                                           AND (valid_from=0 OR valid_from <= memory_now())
                                           AND (valid_until=0 OR valid_until > memory_now())""",
            "associations": "SELECT COUNT(*) AS count FROM memory_associations",
            "approvals": "SELECT COUNT(*) AS count FROM memory_approvals WHERE status='pending'",
            "pending_operations": "SELECT COUNT(*) AS count FROM pending_operations WHERE status IN ('pending','running')",
            "failed_operations": "SELECT COUNT(*) AS count FROM pending_operations WHERE status='failed'",
        }
        counts: Dict[str, int] = {}
        for key, sql in tables.items():
            row = self._fetchone(sql) or {"count": 0}
            counts[key] = int(row["count"])
        return counts

    @_transactional
    def ensure_memory_session(
        self,
        session_id: str,
        *,
        label: str = "",
        summary: str = "",
        status: str | None = None,
        sensitivity: str = "normal",
    ) -> Dict[str, Any]:
        clean_id = normalize_whitespace(session_id)
        if not clean_id:
            raise ValueError("session_id is required")
        clean_sensitivity = normalize_text(sensitivity) or "normal"
        if clean_sensitivity == "normal" and summary and _looks_sensitive_for_export({"summary": summary}):
            clean_sensitivity = "sensitive"
        now = now_ts()
        existing = self._fetchone("SELECT * FROM memory_sessions WHERE session_id = ?", (clean_id,))
        if existing:
            next_label = label or str(existing.get("label") or "")
            next_summary = summary or str(existing.get("summary") or "")
            next_status = status or str(existing.get("status") or "open")
            next_sensitivity = (
                clean_sensitivity if clean_sensitivity != "normal" else str(existing.get("sensitivity") or "normal")
            )
            ended_at = 0.0 if status == "open" else float(existing.get("ended_at") or 0.0)
            self._execute(
                """
                UPDATE memory_sessions
                SET label = ?, summary = ?, status = ?, sensitivity = ?, ended_at = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (next_label, next_summary, next_status, next_sensitivity, ended_at, now, clean_id),
            )
            return self._fetchone("SELECT * FROM memory_sessions WHERE session_id = ?", (clean_id,)) or {}

        self._execute(
            """
            INSERT INTO memory_sessions(
                session_id, label, summary, status, sensitivity, started_at,
                ended_at, last_activity_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
            """,
            (
                clean_id,
                label or clean_id,
                summary or "",
                status or "open",
                clean_sensitivity,
                now,
                now,
                now,
                now,
            ),
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

    @_transactional
    def close_memory_session(
        self,
        session_id: str,
        *,
        summary: str = "",
        sensitivity: str = "normal",
    ) -> Dict[str, Any]:
        session = self.ensure_memory_session(
            session_id,
            summary=summary or "",
            status="closed",
            sensitivity=sensitivity,
        )
        effective_sensitivity = normalize_text(sensitivity) or "normal"
        if effective_sensitivity == "normal" and str(session.get("sensitivity") or "normal") != "normal":
            effective_sensitivity = str(session.get("sensitivity") or "normal")
        if effective_sensitivity == "normal" and summary and _looks_sensitive_for_export({"summary": summary}):
            effective_sensitivity = "sensitive"
        now = now_ts()
        self._execute(
            """
            UPDATE memory_sessions
            SET summary = ?, status = 'closed', sensitivity = ?,
                ended_at = ?, last_activity_at = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (
                summary or str(session.get("summary") or ""),
                effective_sensitivity,
                now,
                now,
                now,
                normalize_whitespace(session_id),
            ),
        )
        closed = (
            self._fetchone("SELECT * FROM memory_sessions WHERE session_id = ?", (normalize_whitespace(session_id),))
            or {}
        )
        self.record_history(
            entity_kind="session",
            entity_id=normalize_whitespace(session_id),
            action="closed",
            reason="session_end",
            source="session",
            payload=closed,
        )
        return closed

    @_transactional
    def append_episode(
        self,
        *,
        session_id: str,
        user_content: str,
        assistant_content: str,
        topic_hint: str = "",
        created_at: float | None = None,
        sensitivity: str = "normal",
        operation_key: str = "",
    ) -> Dict[str, Any]:
        created_at = float(created_at or now_ts())
        clean_session = normalize_whitespace(session_id)
        clean_operation_key = normalize_whitespace(operation_key)
        if clean_operation_key:
            existing = self._fetchone("SELECT * FROM episodes WHERE operation_key=?", (clean_operation_key,))
            if existing:
                return existing
        self.ensure_memory_session(clean_session, status="open")
        digest_source = normalize_whitespace(f"{user_content} {assistant_content}")[:240]
        digest = digest_source or "(empty turn)"
        clean_sensitivity = normalize_text(sensitivity) or "normal"
        if clean_sensitivity == "normal" and _looks_sensitive_for_export(
            {"content": f"{user_content} {assistant_content}"}
        ):
            clean_sensitivity = "sensitive"
        cur = self._execute(
            """
            INSERT INTO episodes(
                session_id, user_content, assistant_content, digest, topic_hint, sensitivity, operation_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean_session,
                user_content or "",
                assistant_content or "",
                digest,
                topic_hint or "",
                clean_sensitivity,
                clean_operation_key,
                created_at,
            ),
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
            "sensitivity": clean_sensitivity,
            "operation_key": clean_operation_key,
            "created_at": created_at,
        }

    @_transactional
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
            self._execute(
                "DELETE FROM memory_links WHERE (source_kind = 'episode' AND source_id = ?) OR (target_kind = 'episode' AND target_id = ?)",
                (str(episode_id), str(episode_id)),
            )
            self._execute(
                "UPDATE memory_traces SET source_episode_id = 0 WHERE source_episode_id = ?",
                (episode_id,),
            )
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

    def pending_episode_count(self, episode_id: int) -> int:
        row = self._fetchone(
            "SELECT COUNT(*) AS count FROM episodes WHERE id > ?",
            (int(episode_id),),
        ) or {"count": 0}
        return int(row["count"])

    def latest_episode_id(self) -> int:
        row = self._fetchone("SELECT COALESCE(MAX(id), 0) AS id FROM episodes") or {"id": 0}
        return int(row["id"])

    def get_session_artifacts(self, session_id: str, *, limit: int = 20) -> Dict[str, Any]:
        clean_id = normalize_whitespace(session_id)
        like_session = f'%"session_id": "{clean_id}"%'
        artifacts = {
            "session": self._fetchone("SELECT * FROM memory_sessions WHERE session_id = ?", (clean_id,)) or {},
            "episodes": self._fetchall(
                """
                SELECT id, session_id, digest, topic_hint, sensitivity, created_at
                FROM episodes
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (clean_id, int(limit)),
            ),
            "traces": self._fetchall(
                """
                SELECT id, session_id, label, content, trace_type, sensitivity, importance, salience, updated_at
                FROM memory_traces
                WHERE session_id = ? AND active = 1 AND sensitivity = 'normal'
                ORDER BY id DESC
                LIMIT ?
                """,
                (clean_id, int(limit)),
            ),
            "journals": self._fetchall(
                """
                SELECT id, session_id, label, content, journal_type, sensitivity, importance, salience, updated_at
                FROM memory_journals
                WHERE session_id = ? AND active = 1 AND sensitivity = 'normal'
                ORDER BY id DESC
                LIMIT ?
                """,
                (clean_id, int(limit)),
            ),
            "summaries": self._fetchall(
                """
                SELECT id, session_id, label, summary, summary_type, sensitivity, importance, salience, updated_at
                FROM memory_summaries
                WHERE session_id = ? AND active = 1 AND sensitivity = 'normal'
                ORDER BY id DESC
                LIMIT ?
                """,
                (clean_id, int(limit)),
            ),
            "preferences": self._fetchall(
                """
                SELECT id, preference_key, label, value, content, sensitivity, source_session_id, importance, salience, updated_at
                FROM memory_preferences
                WHERE active = 1 AND sensitivity = 'normal'
                  AND (source_session_id = ? OR (source_session_id = '' AND metadata_json LIKE ?))
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (clean_id, like_session, int(limit)),
            ),
            "policies": self._fetchall(
                """
                SELECT id, policy_key, label, content, sensitivity, source_session_id, importance, salience, updated_at
                FROM memory_policies
                WHERE active = 1 AND sensitivity = 'normal'
                  AND (source_session_id = ? OR (source_session_id = '' AND metadata_json LIKE ?))
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (clean_id, like_session, int(limit)),
            ),
            "facts": self._fetchall(
                """
                SELECT id, content, category, topic, importance, salience, belief_score, sensitivity,
                       memory_class, valid_from, valid_until, temporal_kind, event_at,
                       temporal_precision, temporal_timezone, temporal_confidence,
                       created_at, updated_at, subject_key, value_key, source_session_id
                FROM facts
                WHERE source_session_id = ? AND active = 1
                  AND sensitivity = 'normal'
                  AND (valid_from=0 OR valid_from <= memory_now())
                  AND (valid_until=0 OR valid_until > memory_now())
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (clean_id, int(limit)),
            ),
        }
        # v1 traces did not carry a sensitivity label. Filter their content as a
        # migration safeguard so an old secret cannot be copied into a v2 summary.
        artifacts["traces"] = [row for row in artifacts["traces"] if not _looks_sensitive_for_export(row)]
        return artifacts

    def list_sessions(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        return self._fetchall(
            """
            SELECT session_id, label, summary, status, sensitivity,
                   started_at, ended_at, last_activity_at, created_at, updated_at
            FROM memory_sessions
            ORDER BY updated_at DESC, session_id ASC
            LIMIT ?
            """,
            (int(limit),),
        )

    def list_topics(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        return self._fetchall(
            """
            SELECT id, slug, title, summary, category, sensitivity, importance,
                   salience, source_session_id, last_recalled_at, decay_half_life_days, updated_at
            FROM topics
            ORDER BY salience DESC, importance DESC, updated_at DESC, slug ASC
            LIMIT ?
            """,
            (int(limit),),
        )

    def list_preferences(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        return self._fetchall(
            """
            SELECT id, preference_key, label, value, content, metadata_json, sensitivity, source_session_id, importance, salience, updated_at
            FROM memory_preferences
            WHERE active = 1
            ORDER BY salience DESC, importance DESC, updated_at DESC, preference_key ASC
            LIMIT ?
            """,
            (int(limit),),
        )

    def list_active_facts(self, *, limit: int = 500) -> List[Dict[str, Any]]:
        return self._fetchall(
            """
            SELECT id, content, category, topic, subject_key, value_key, importance, confidence, salience,
                   belief_score, observation_count, valid_from, valid_until, temporal_kind, event_at,
                   temporal_precision, temporal_timezone, temporal_confidence, sensitivity, memory_class, pinned,
                   created_at, updated_at, source_session_id
            FROM facts
            WHERE active = 1
              AND (valid_from=0 OR valid_from <= memory_now())
              AND (valid_until=0 OR valid_until > memory_now())
            ORDER BY category ASC, pinned DESC, belief_score DESC, importance DESC, salience DESC
            LIMIT ?
            """,
            (int(limit),),
        )

    def list_policies(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        return self._fetchall(
            """
            SELECT id, policy_key, label, content, metadata_json, sensitivity, source_session_id, importance, salience, updated_at
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
            SELECT f.id, f.content, f.category, f.topic, f.importance, f.confidence, f.salience,
                   f.belief_score, f.sensitivity, f.memory_class, f.valid_from, f.valid_until,
                   f.temporal_kind, f.event_at, f.temporal_precision, f.temporal_timezone,
                   f.temporal_confidence, f.created_at, f.updated_at, f.subject_key, f.value_key,
                   f.source_session_id
            FROM topic_membership tm
            JOIN facts f ON f.id = tm.fact_id
            WHERE tm.topic_id = ? AND f.active = 1
              AND (f.valid_from=0 OR f.valid_from <= memory_now())
              AND (f.valid_until=0 OR f.valid_until > memory_now())
            ORDER BY f.belief_score DESC, f.salience DESC, f.importance DESC, f.updated_at DESC, f.id ASC
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

    @_transactional
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
        sensitivity: str = "normal",
    ) -> Dict[str, Any]:
        clean_session = normalize_whitespace(session_id)
        clean_content = normalize_whitespace(content)
        if not clean_content:
            raise ValueError("Trace content cannot be empty.")
        self.ensure_memory_session(clean_session)
        clean_sensitivity = normalize_text(sensitivity) or "normal"
        if clean_sensitivity == "normal" and _looks_sensitive_for_export({"content": clean_content}):
            clean_sensitivity = "sensitive"
        now = now_ts()
        cur = self._execute(
            """
            INSERT INTO memory_traces(
                session_id, label, content, trace_type, metadata_json, sensitivity,
                importance, salience, last_recalled_at, source_episode_id, active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 1, ?, ?)
            """,
            (
                clean_session,
                normalize_whitespace(label) or trace_type,
                clean_content,
                trace_type or "turn",
                json.dumps(dict(metadata or {}), sort_keys=True),
                clean_sensitivity,
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

    @_transactional
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
        sensitivity: str = "normal",
    ) -> Dict[str, Any]:
        clean_content = normalize_whitespace(content)
        if not clean_content:
            raise ValueError("Journal content cannot be empty.")
        clean_session = normalize_whitespace(session_id)
        if clean_session:
            self.ensure_memory_session(clean_session)
        now = now_ts()
        next_review_at = now + _first_review_offset_seconds((1.0, 3.0, 7.0, 14.0, 30.0))
        cur = self._execute(
            """
            INSERT INTO memory_journals(session_id, label, content, journal_type, metadata_json, sensitivity, importance, salience, last_recalled_at, review_count, next_review_at, reconsolidation_until, active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, 0, 1, ?, ?)
            """,
            (
                clean_session,
                normalize_whitespace(label) or "Journal",
                clean_content,
                journal_type or "note",
                json.dumps(dict(metadata or {}), sort_keys=True),
                normalize_text(sensitivity) or "normal",
                int(importance),
                float(salience),
                next_review_at,
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

    @_transactional
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
        sensitivity: str = "normal",
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
            action = "reconsolidated" if float(existing.get("reconsolidation_until") or 0.0) > now else "updated"
            self._execute(
                """
                UPDATE memory_summaries
                SET session_id = ?, label = ?, summary = ?, content = ?, summary_type = ?, metadata_json = ?, sensitivity = ?, importance = MAX(importance, ?), salience = MAX(salience, ?), next_review_at = ?, active = 1, updated_at = ?
                WHERE source_hash = ?
                """,
                (
                    clean_session,
                    normalize_whitespace(label) or pretty_topic(summary_type),
                    clean_summary,
                    normalize_whitespace(content),
                    summary_type or "session",
                    json.dumps(meta, sort_keys=True),
                    normalize_text(sensitivity) or "normal",
                    int(importance),
                    float(salience),
                    now + _first_review_offset_seconds((1.0, 3.0, 7.0, 14.0, 30.0)),
                    now,
                    source_hash,
                ),
            )
            summary_id = int(existing["id"])
        else:
            cur = self._execute(
                """
                INSERT INTO memory_summaries(session_id, label, summary, content, summary_type, source_hash, metadata_json, sensitivity, importance, salience, last_recalled_at, review_count, next_review_at, reconsolidation_until, active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, 0, 1, ?, ?)
                """,
                (
                    clean_session,
                    normalize_whitespace(label) or pretty_topic(summary_type),
                    clean_summary,
                    normalize_whitespace(content),
                    summary_type or "session",
                    source_hash,
                    json.dumps(meta, sort_keys=True),
                    normalize_text(sensitivity) or "normal",
                    int(importance),
                    float(salience),
                    now + _first_review_offset_seconds((1.0, 3.0, 7.0, 14.0, 30.0)),
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
            self.add_link(
                "summary", summary_id, str(ref.get("kind") or "memory"), str(ref.get("id") or ""), "summarizes"
            )
        self.record_history(
            entity_kind="summary",
            entity_id=summary_id,
            action=action,
            reason=reason,
            source="summary",
            payload=row,
        )
        return row

    @_transactional
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
        sensitivity: str = "normal",
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
            action = "reconsolidated" if float(existing.get("reconsolidation_until") or 0.0) > now else "updated"
            # Weighted merge instead of pure MAX
            _old_imp = int(existing.get("importance") or 5)
            _new_imp = int(importance)
            next_importance = max(_new_imp, round(_old_imp * 0.7 + _new_imp * 0.3))
            _old_sal = float(existing.get("salience") or 0.5)
            next_salience = min(1.0, _old_sal * 0.6 + float(salience) * 0.4 + 0.05)
            self._execute(
                """
                UPDATE memory_preferences
                SET label = ?, value = ?, content = ?, metadata_json = ?, sensitivity = ?, source_session_id = ?, importance = ?, salience = ?, next_review_at = ?, active = 1, updated_at = ?
                WHERE preference_key = ?
                """,
                (
                    pref_label,
                    pref_value,
                    pref_content,
                    json.dumps(meta, sort_keys=True),
                    normalize_text(sensitivity) or "normal",
                    source_session_id,
                    next_importance,
                    next_salience,
                    now + _first_review_offset_seconds((1.0, 3.0, 7.0, 14.0, 30.0)),
                    now,
                    pref_key,
                ),
            )
            pref_id = int(existing["id"])
        else:
            cur = self._execute(
                """
                INSERT INTO memory_preferences(preference_key, label, value, content, metadata_json, sensitivity, source_session_id, importance, salience, last_recalled_at, review_count, next_review_at, reconsolidation_until, active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, 0, 1, ?, ?)
                """,
                (
                    pref_key,
                    pref_label,
                    pref_value,
                    pref_content,
                    json.dumps(meta, sort_keys=True),
                    normalize_text(sensitivity) or "normal",
                    source_session_id,
                    int(importance),
                    float(salience),
                    now + _first_review_offset_seconds((1.0, 3.0, 7.0, 14.0, 30.0)),
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

    @_transactional
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
        sensitivity: str = "normal",
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
            action = "reconsolidated" if float(existing.get("reconsolidation_until") or 0.0) > now else "updated"
            self._execute(
                """
                UPDATE memory_policies
                SET label = ?, content = ?, metadata_json = ?, sensitivity = ?, source_session_id = ?, importance = MAX(importance, ?), salience = MAX(salience, ?), next_review_at = ?, active = 1, updated_at = ?
                WHERE policy_key = ?
                """,
                (
                    policy_label,
                    clean_content,
                    json.dumps(meta, sort_keys=True),
                    normalize_text(sensitivity) or "normal",
                    source_session_id,
                    int(importance),
                    float(salience),
                    now + _first_review_offset_seconds((1.0, 3.0, 7.0, 14.0, 30.0)),
                    now,
                    policy_key,
                ),
            )
            policy_id = int(existing["id"])
        else:
            cur = self._execute(
                """
                INSERT INTO memory_policies(policy_key, label, content, metadata_json, sensitivity, source_session_id, importance, salience, last_recalled_at, review_count, next_review_at, reconsolidation_until, active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, 0, 1, ?, ?)
                """,
                (
                    policy_key,
                    policy_label,
                    clean_content,
                    json.dumps(meta, sort_keys=True),
                    normalize_text(sensitivity) or "normal",
                    source_session_id,
                    int(importance),
                    float(salience),
                    now + _first_review_offset_seconds((1.0, 3.0, 7.0, 14.0, 30.0)),
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

    @_transactional
    def add_link(
        self,
        source_kind: str,
        source_id: Any,
        target_kind: str,
        target_id: Any,
        link_type: str,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        source_kind = normalize_text(source_kind)
        target_kind = normalize_text(target_kind)
        link_type = normalize_whitespace(link_type)
        source_id_text = str(source_id)
        target_id_text = str(target_id)
        if not (source_kind and target_kind and link_type and source_id_text and target_id_text):
            raise ValueError("Link fields cannot be empty.")
        self._require_reference(source_kind, source_id_text)
        self._require_reference(target_kind, target_id_text)
        if source_kind == target_kind and source_id_text == target_id_text:
            raise ValueError("An entity cannot link to itself.")
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
            return (
                self._fetchone(
                    """
                SELECT * FROM memory_links
                WHERE source_kind = ? AND source_id = ? AND target_kind = ? AND target_id = ? AND link_type = ?
                """,
                    (source_kind, source_id_text, target_kind, target_id_text, link_type),
                )
                or {}
            )
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

    def _require_reference(self, kind: str, entity_id: Any) -> None:
        clean_kind = normalize_text(kind)
        reference = self._REFERENCE_TABLES.get(clean_kind)
        if not reference:
            raise ValueError(f"Unsupported memory reference kind: {kind}")
        table, primary_key = reference
        row = self._fetchone(
            f"SELECT 1 AS present FROM {table} WHERE CAST({primary_key} AS TEXT)=? LIMIT 1",
            (str(entity_id),),
        )
        if not row:
            raise ValueError(f"Unknown {clean_kind} reference: {entity_id}")

    @_transactional
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

    @_transactional
    def compact_history(self, *, max_per_entity: int = 10, max_age_days: int = 90) -> int:
        """Delete old history rows, keeping only the newest *max_per_entity* per entity."""
        cutoff = now_ts() - (int(max_age_days) * 86400)
        # Find entities with too many rows.
        heavy = self._fetchall(
            """
            SELECT entity_kind, entity_id, COUNT(*) AS cnt
            FROM memory_history
            GROUP BY entity_kind, entity_id
            HAVING cnt > ?
            """,
            (int(max_per_entity),),
        )
        deleted = 0
        for row in heavy:
            kind = str(row["entity_kind"])
            eid = str(row["entity_id"])
            # Find the Nth-newest row id to use as the cutoff.
            boundary = self._fetchone(
                """
                SELECT id FROM memory_history
                WHERE entity_kind = ? AND entity_id = ?
                ORDER BY id DESC
                LIMIT 1 OFFSET ?
                """,
                (kind, eid, int(max_per_entity) - 1),
            )
            if not boundary:
                continue
            cur = self._execute(
                """
                DELETE FROM memory_history
                WHERE entity_kind = ? AND entity_id = ? AND id < ?
                """,
                (kind, eid, int(boundary["id"])),
            )
            deleted += cur.rowcount
        # Also delete very old rows regardless.
        cur = self._execute(
            "DELETE FROM memory_history WHERE created_at < ? AND action = 'updated'",
            (cutoff,),
        )
        deleted += cur.rowcount
        return deleted

    def _default_fact_salience(self, category: str, metadata: Dict[str, Any]) -> float:
        subject_key = str(metadata.get("subject_key") or "")
        if subject_key.startswith("user:"):
            # Differentiate: core identity facts get high salience,
            # transient/activity facts start lower.
            _HIGH_SALIENCE_PREFIXES = (
                "user:name",
                "user:date_of_birth",
                "user:occupation",
                "user:location",
                "user:origin",
                "user:hometown",
                "user:condition",
                "user:diet",
                "user:pronouns",
                "user:physical_attributes",
                "user:response_style",
                "user:response_tone",
            )
            if any(subject_key.startswith(p) for p in _HIGH_SALIENCE_PREFIXES):
                return 0.92
            if subject_key.startswith("user:preference:"):
                return 0.72
            if subject_key.startswith("user:favorite:"):
                return 0.78
            return 0.80
        if category == "workflow":
            return 0.85
        if category == "project":
            return 0.72
        if category == "environment":
            return 0.65
        return 0.50

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

    @staticmethod
    def _source_reliability(source_role: str, source: str) -> float:
        role = normalize_text(source_role)
        source_name = normalize_text(source)
        if role == "user":
            return 0.95
        if role in {"tool", "system"}:
            return 0.82
        if role == "assistant":
            return 0.38
        if source_name in {"manual", "user", "memory-tool"}:
            return 0.92
        if source_name in {"tool", "observation"}:
            return 0.78
        if source_name in {"llm", "consolidation"}:
            return 0.42
        return 0.55

    @staticmethod
    def _belief_score(*, confidence: float, reliability: float, explicit_correction: bool, observations: int) -> float:
        corroboration = min(math.log2(max(1, observations)) / 10.0, 0.12)
        return _clamp_float(
            (0.42 * reliability) + (0.33 * confidence) + (0.2 if explicit_correction else 0.0) + corroboration,
            0.0,
            1.0,
            0.5,
        )

    def _record_fact_evidence(
        self,
        *,
        fact_id: int,
        content: str,
        source: str,
        source_role: str,
        session_id: str,
        confidence: float,
        reliability: float,
        explicit_correction: bool,
        observed_at: float,
        metadata: Dict[str, Any],
    ) -> None:
        self._execute(
            """
            INSERT INTO belief_evidence(
                fact_id, content, source, source_role, session_id, confidence,
                reliability, explicit_correction, observed_at, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(fact_id),
                content,
                source,
                source_role,
                session_id,
                confidence,
                reliability,
                int(explicit_correction),
                observed_at,
                json.dumps(metadata, sort_keys=True),
                now_ts(),
            ),
        )

    @_transactional
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
        source_role: str = "",
        reliability: float | None = None,
        explicit_correction: bool = False,
        valid_from: float | None = None,
        valid_until: float | None = None,
        temporal_kind: str = "",
        event_at: float | None = None,
        temporal_precision: str = "",
        temporal_timezone: str = "",
        temporal_confidence: float | None = None,
        sensitivity: str = "normal",
        memory_class: str = "semantic",
        pinned: bool = False,
    ) -> Dict[str, Any]:
        clean = normalize_whitespace(content)
        if not clean:
            raise ValueError("Fact content cannot be empty.")
        observed_at = _timestamp(observed_at, now_ts()) or now_ts()
        fingerprint = fingerprint_text(clean)
        signature = text_signature(clean)
        topic_slug = slugify(topic)
        existing = self._fetchone("SELECT * FROM facts WHERE fingerprint = ?", (fingerprint,))
        meta = _merge_json_dict(existing.get("metadata") if existing else {}, metadata)
        subject_key = normalize_whitespace(str(meta.get("subject_key") or ""))
        value_key = normalize_text(str(meta.get("value_key") or "")) if meta.get("value_key") else ""
        polarity_raw = str(meta.get("polarity", 1)).strip().casefold()
        polarity = -1 if polarity_raw in {"-1", "false", "neg", "negative", "no"} else 1
        exclusive = 1 if subject_key and _as_bool(meta.get("exclusive")) else 0
        source_session = normalize_whitespace(source_session_id or str(meta.get("source_session_id") or ""))
        if source_session:
            self.ensure_memory_session(source_session)
        importance_value = _clamp_int(importance, 1, 10, 5)
        confidence_value = _clamp_float(confidence, 0.0, 1.0, 0.7)
        role = normalize_text(source_role or str(meta.get("source_role") or "unknown")) or "unknown"
        reliability_value = _clamp_float(
            reliability if reliability is not None else self._source_reliability(role, source), 0.0, 1.0, 0.5
        )
        is_correction = bool(explicit_correction or _as_bool(meta.get("explicit_correction")))
        valid_from_value = _timestamp(valid_from if valid_from is not None else meta.get("valid_from"), observed_at)
        valid_until_value = _timestamp(valid_until if valid_until is not None else meta.get("valid_until"), 0.0)
        allowed_temporal_kinds = {"atemporal", "current", "event", "scheduled", "temporary"}
        temporal_kind_value = normalize_text(temporal_kind or str(meta.get("temporal_kind") or ""))
        if temporal_kind_value not in allowed_temporal_kinds:
            if valid_until_value > 0:
                temporal_kind_value = "temporary"
            elif subject_key and exclusive:
                temporal_kind_value = "current"
            else:
                temporal_kind_value = "atemporal"
        event_at_value = _timestamp(event_at if event_at is not None else meta.get("event_at"), 0.0)
        allowed_temporal_precision = {"unknown", "year", "month", "day", "hour", "minute", "second"}
        temporal_precision_value = normalize_text(
            temporal_precision or str(meta.get("temporal_precision") or "unknown")
        )
        if temporal_precision_value not in allowed_temporal_precision:
            temporal_precision_value = "unknown"
        temporal_timezone_value = normalize_whitespace(temporal_timezone or str(meta.get("temporal_timezone") or ""))[
            :80
        ]
        temporal_confidence_value = _clamp_float(
            temporal_confidence if temporal_confidence is not None else meta.get("temporal_confidence"),
            0.0,
            1.0,
            0.0,
        )
        meta["temporal_kind"] = temporal_kind_value
        meta["temporal_precision"] = temporal_precision_value
        meta["temporal_confidence"] = temporal_confidence_value
        if event_at_value > 0:
            meta["event_at"] = event_at_value
        else:
            meta.pop("event_at", None)
        if valid_from_value > 0:
            meta["valid_from"] = valid_from_value
        if valid_until_value > 0:
            meta["valid_until"] = valid_until_value
        else:
            meta.pop("valid_until", None)
        if temporal_timezone_value:
            meta["temporal_timezone"] = temporal_timezone_value
        else:
            meta.pop("temporal_timezone", None)
        sensitivity_value = normalize_text(sensitivity or str(meta.get("sensitivity") or "normal")) or "normal"
        memory_class_value = normalize_text(memory_class or str(meta.get("memory_class") or "semantic")) or "semantic"
        if temporal_kind_value in {"event", "scheduled"} and memory_class_value == "semantic":
            memory_class_value = "autobiographical"
        salience_value = _clamp_float(
            salience if salience is not None else self._default_fact_salience(category, meta),
            0.0,
            1.0,
            0.55,
        )
        half_life = _clamp_float(
            decay_half_life_days if decay_half_life_days is not None else self._default_fact_half_life(category, meta),
            0.01,
            3650.0,
            45.0,
        )
        metadata_json = json.dumps(meta, sort_keys=True)
        if existing:
            history_action = (
                "reconsolidated" if float(existing.get("reconsolidation_until") or 0.0) > observed_at else "updated"
            )
            next_subject = subject_key or str(existing.get("subject_key") or "")
            next_value = value_key or str(existing.get("value_key") or "")
            next_polarity = polarity if subject_key else int(existing.get("polarity") or 1)
            next_exclusive = exclusive if subject_key else int(existing.get("exclusive") or 0)
            # Weighted merge: blend existing and new values instead of pure MAX.
            # This allows importance/salience to drift down when new observations
            # provide lower values, while still giving weight to the historical peak.
            _old_imp = int(existing.get("importance") or 5)
            _new_imp = importance_value
            # 70% old + 30% new, but never drop below new value minus 1
            next_importance = max(_new_imp, round(_old_imp * 0.7 + _new_imp * 0.3))
            _old_conf = float(existing.get("confidence") or 0.5)
            _new_conf = confidence_value
            next_confidence = min(1.0, max(_new_conf, _old_conf * 0.7 + _new_conf * 0.3))
            # Salience: blend with slight upward bias for re-observation
            _old_sal = float(existing.get("salience") or 0.5)
            next_salience = _old_sal * 0.6 + salience_value * 0.4
            # Bump slightly for being re-observed (reconsolidation reward)
            next_salience = min(1.0, next_salience + 0.05)
            next_half_life = max(float(existing.get("decay_half_life_days") or 0.0), half_life)
            next_session = source_session or str(existing.get("source_session_id") or "")
            next_review_at = observed_at + _first_review_offset_seconds((1.0, 3.0, 7.0, 14.0, 30.0))
            observation_count = int(existing.get("observation_count") or 1) + 1
            belief_score = self._belief_score(
                confidence=next_confidence,
                reliability=reliability_value,
                explicit_correction=is_correction,
                observations=observation_count,
            )
            self._execute(
                """
                UPDATE facts
                SET active = 1,
                    superseded_by = NULL,
                    importance = ?,
                    confidence = ?,
                    salience = ?,
                    updated_at = ?,
                    last_seen_at = ?,
                    metadata_json = ?,
                    subject_key = ?,
                    value_key = ?,
                    polarity = ?,
                    exclusive = ?,
                    source_session_id = ?,
                    next_review_at = ?,
                    decay_half_life_days = ?,
                    belief_score = MAX(belief_score, ?),
                    observation_count = ?,
                    valid_from = CASE WHEN valid_from = 0 THEN ? ELSE MIN(valid_from, ?) END,
                    valid_until = ?,
                    temporal_kind = ?,
                    event_at = ?,
                    temporal_precision = ?,
                    temporal_timezone = ?,
                    temporal_confidence = ?,
                    sensitivity = ?,
                    memory_class = ?,
                    pinned = MAX(pinned, ?),
                    revision = revision + 1
                WHERE id = ?
                """,
                (
                    next_importance,
                    next_confidence,
                    next_salience,
                    observed_at,
                    observed_at,
                    metadata_json,
                    next_subject,
                    next_value,
                    next_polarity,
                    next_exclusive,
                    next_session,
                    next_review_at,
                    next_half_life,
                    belief_score,
                    observation_count,
                    valid_from_value,
                    valid_from_value,
                    valid_until_value,
                    temporal_kind_value,
                    event_at_value,
                    temporal_precision_value,
                    temporal_timezone_value,
                    temporal_confidence_value,
                    sensitivity_value,
                    memory_class_value,
                    int(pinned),
                    int(existing["id"]),
                ),
            )
            updated = self._fetchone("SELECT * FROM facts WHERE id = ?", (int(existing["id"]),)) or {}
            self._record_fact_evidence(
                fact_id=int(existing["id"]),
                content=clean,
                source=source,
                source_role=role,
                session_id=source_session,
                confidence=confidence_value,
                reliability=reliability_value,
                explicit_correction=is_correction,
                observed_at=observed_at,
                metadata=meta,
            )
            self._upsert_fact_fts(updated)
            contradictions = self._resolve_subject_state(updated)
            updated = self._fetchone("SELECT * FROM facts WHERE id = ?", (int(existing["id"]),)) or updated
            if next_session:
                self.add_link("fact", updated["id"], "session", next_session, "captured_in")
            self.record_history(
                entity_kind="fact",
                entity_id=updated["id"],
                subject_key=next_subject,
                action=history_action,
                reason=history_reason or source,
                source=source,
                payload=updated,
            )
            return {
                "action": "updated" if int(updated.get("active") or 0) == 1 else "superseded",
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
                review_count,
                next_review_at,
                reconsolidation_until,
                decay_half_life_days,
                belief_score,
                observation_count,
                valid_from,
                valid_until,
                temporal_kind,
                event_at,
                temporal_precision,
                temporal_timezone,
                temporal_confidence,
                sensitivity,
                memory_class,
                pinned,
                revision,
                created_at,
                updated_at,
                last_seen_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, 0, 0, ?, 0, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
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
                importance_value,
                confidence_value,
                salience_value,
                subject_key,
                value_key,
                int(polarity),
                int(exclusive),
                source_session,
                observed_at + _first_review_offset_seconds((1.0, 3.0, 7.0, 14.0, 30.0)),
                half_life,
                self._belief_score(
                    confidence=confidence_value,
                    reliability=reliability_value,
                    explicit_correction=is_correction,
                    observations=1,
                ),
                valid_from_value,
                valid_until_value,
                temporal_kind_value,
                event_at_value,
                temporal_precision_value,
                temporal_timezone_value,
                temporal_confidence_value,
                sensitivity_value,
                memory_class_value,
                int(pinned),
                observed_at,
                observed_at,
                observed_at,
            ),
        )
        fact_id = int(cur.lastrowid)
        self._record_fact_evidence(
            fact_id=fact_id,
            content=clean,
            source=source,
            source_role=role,
            session_id=source_session,
            confidence=confidence_value,
            reliability=reliability_value,
            explicit_correction=is_correction,
            observed_at=observed_at,
            metadata=meta,
        )
        inserted = self._fetchone("SELECT * FROM facts WHERE id = ?", (fact_id,)) or {}
        superseded = self._supersede_older_facts(inserted)
        inserted = self._fetchone("SELECT * FROM facts WHERE id = ?", (fact_id,)) or inserted
        contradictions = (
            self._resolve_subject_state(inserted)
            if int(inserted.get("active") or 0) == 1
            else {"superseded": [], "contradictions": []}
        )
        inserted = self._fetchone("SELECT * FROM facts WHERE id = ?", (fact_id,)) or inserted
        if int(inserted.get("active") or 0) == 1:
            self._upsert_fact_fts(inserted)
        else:
            self._delete_fact_fts(fact_id)
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
            "action": "inserted" if int(inserted.get("active") or 0) == 1 else "superseded",
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
            SELECT id, subject_key, belief_score, observation_count, updated_at
            FROM facts
            WHERE id != ?
              AND active = 1
              AND signature = ?
              AND category = ?
              AND topic = ?
            ORDER BY updated_at DESC
            """,
            (
                int(new_fact["id"]),
                signature,
                str(new_fact.get("category") or "general"),
                str(new_fact.get("topic") or "general"),
            ),
        )
        superseded_ids: List[int] = []
        new_subject = str(new_fact.get("subject_key") or "")
        for row in older:
            old_subject = str(row.get("subject_key") or "")
            if new_subject and old_subject and new_subject != old_subject:
                continue
            if self.conflict_policy == "evidence" and float(row.get("belief_score") or 0.0) > float(
                new_fact.get("belief_score") or 0.0
            ):
                self._soft_supersede_fact(
                    int(new_fact["id"]),
                    int(row["id"]),
                    float(new_fact["updated_at"]),
                    subject_key=new_subject,
                    reason="stronger_existing_evidence",
                )
                return [int(new_fact["id"])]
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

    @_transactional
    def merge_duplicate_subjects(self) -> int:
        """Consolidation-time dedup: for each subject_key that has multiple
        active facts with the same value_key, keep only the one with the
        highest (importance, salience, updated_at) and supersede the rest."""
        merged = 0
        rows = self._fetchall(
            """
            SELECT subject_key, value_key, COUNT(*) AS cnt
            FROM facts
            WHERE active = 1 AND subject_key != ''
            GROUP BY subject_key, value_key
            HAVING cnt > 1
            ORDER BY subject_key, value_key
            """
        )
        for row in rows:
            sk = str(row["subject_key"])
            vk = str(row.get("value_key") or "")
            if vk:
                dupes = self._fetchall(
                    """SELECT id, importance, salience, updated_at FROM facts
                       WHERE active=1 AND subject_key=? AND value_key=?
                       ORDER BY belief_score DESC, observation_count DESC, importance DESC, salience DESC, updated_at DESC""",
                    (sk, vk),
                )
            else:
                dupes = self._fetchall(
                    """SELECT id, importance, salience, updated_at FROM facts
                       WHERE active=1 AND subject_key=? AND (value_key IS NULL OR value_key='')
                       ORDER BY belief_score DESC, observation_count DESC, importance DESC, salience DESC, updated_at DESC""",
                    (sk,),
                )
            if len(dupes) <= 1:
                continue
            winner_id = int(dupes[0]["id"])
            for loser in dupes[1:]:
                self._soft_supersede_fact(
                    int(loser["id"]),
                    winner_id,
                    float(dupes[0]["updated_at"]),
                    subject_key=sk,
                    reason="consolidation_dedup",
                )
                merged += 1
        return merged

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
        supersede_only = any(subject_key.startswith(p) for p in _SUPERSEDE_ONLY_PREFIXES)
        new_value = str(new_fact.get("value_key") or "")
        new_polarity = int(new_fact.get("polarity") or 1)

        # Faceted subjects may hold several values. All other exclusive
        # subjects have one current state and therefore compare against every
        # active fact for the subject.
        if subject_key in COEXIST_SUBJECT_KEYS and new_value:
            others = self._fetchall(
                """
                SELECT id, content, normalized_content, value_key, polarity, belief_score, observation_count, updated_at
                FROM facts
                WHERE id != ?
                  AND active = 1
                  AND subject_key = ?
                  AND exclusive = 1
                  AND value_key = ?
                ORDER BY updated_at DESC
                """,
                (int(new_fact["id"]), subject_key, new_value),
            )
        elif subject_key in COEXIST_SUBJECT_KEYS:
            others = self._fetchall(
                """
                SELECT id, content, normalized_content, value_key, polarity, belief_score, observation_count, updated_at
                FROM facts
                WHERE id != ?
                  AND active = 1
                  AND subject_key = ?
                  AND exclusive = 1
                  AND (value_key IS NULL OR value_key = '')
                ORDER BY updated_at DESC
                """,
                (int(new_fact["id"]), subject_key),
            )
        else:
            others = self._fetchall(
                """
                SELECT id, content, normalized_content, value_key, polarity, belief_score, observation_count, updated_at
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
        stronger = next(
            (
                row
                for row in others
                if self.conflict_policy == "evidence"
                and float(row.get("belief_score") or 0.0) > float(new_fact.get("belief_score") or 0.0)
            ),
            None,
        )
        if stronger is not None:
            contradictory = str(stronger.get("normalized_content") or "") != str(
                new_fact.get("normalized_content") or ""
            )
            self._soft_supersede_fact(
                int(new_fact["id"]),
                int(stronger["id"]),
                float(new_fact["updated_at"]),
                subject_key=subject_key,
                reason="stronger_existing_evidence",
            )
            if contradictory and not supersede_only:
                contradictions.append(
                    self._record_contradiction(
                        subject_key=subject_key,
                        winner_fact_id=int(stronger["id"]),
                        loser_fact_id=int(new_fact["id"]),
                        resolution="existing belief retained because its evidence score is stronger",
                    )
                )
                self.add_link("fact", int(stronger["id"]), "fact", int(new_fact["id"]), "contradicts")
            return {"superseded": [int(new_fact["id"])], "contradictions": contradictions}
        for row in others:
            row_value = str(row.get("value_key") or "")
            row_polarity = int(row.get("polarity") or 1)
            same_content = str(row.get("normalized_content") or "") == str(new_fact.get("normalized_content") or "")
            same_value = bool(new_value or row_value) and row_value == new_value and row_polarity == new_polarity
            contradictory = not (same_content or same_value)
            self._soft_supersede_fact(
                int(row["id"]),
                int(new_fact["id"]),
                float(new_fact["updated_at"]),
                subject_key=subject_key,
                reason="exclusive_subject",
            )
            superseded.append(int(row["id"]))
            if contradictory and not supersede_only:
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
                   w.sensitivity AS winner_sensitivity,
                   l.id AS loser_fact_id,
                   l.content AS loser_content,
                   l.sensitivity AS loser_sensitivity,
                   CASE
                       WHEN COALESCE(w.sensitivity, 'normal') = 'normal'
                        AND COALESCE(l.sensitivity, 'normal') = 'normal' THEN 'normal'
                       WHEN COALESCE(w.sensitivity, 'normal') = COALESCE(l.sensitivity, 'normal')
                           THEN COALESCE(w.sensitivity, 'normal')
                       ELSE 'sensitive'
                   END AS sensitivity
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

    @_transactional
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

    @_transactional
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

    @_transactional
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

    @_transactional
    def prune_stale_facts(self, max_age_days: int = 90) -> int:
        cutoff = now_ts() - (int(max_age_days) * 86400)
        rows = self._fetchall(
            """
            SELECT id
            FROM facts
            WHERE active = 1
              AND importance <= 4
              AND pinned = 0
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

    @_transactional
    def rebuild_topics(self, *, max_facts: int = 5, max_chars: int = 650) -> int:
        rows = self._fetchall(
            """
            SELECT *
            FROM facts
            WHERE active = 1
              AND (valid_from=0 OR valid_from <= memory_now())
              AND (valid_until=0 OR valid_until > memory_now())
            ORDER BY topic ASC, pinned DESC, belief_score DESC, salience DESC, importance DESC, updated_at DESC
            """
        )
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["topic"])].append(row)

        self._execute("DELETE FROM topic_membership")
        live_slugs = set(grouped.keys())

        for slug, facts in grouped.items():
            membership_facts = facts[: max(1, int(max_facts))]
            normal_facts = [fact for fact in facts if str(fact.get("sensitivity") or "normal") == "normal"]
            summary_candidates = normal_facts or facts
            top_facts = summary_candidates[: max(1, int(max_facts))]
            sensitivity_values = {str(fact.get("sensitivity") or "normal") for fact in top_facts}
            topic_sensitivity = (
                "normal"
                if normal_facts
                else (next(iter(sensitivity_values)) if len(sensitivity_values) == 1 else "sensitive")
            )
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
            topic_local_only = any(_as_bool(dict(fact.get("metadata") or {}).get("local_only")) for fact in top_facts)
            topic_metadata_json = json.dumps({"local_only": True} if topic_local_only else {}, sort_keys=True)
            summary = " | ".join(pieces)[: int(max_chars)]
            category = str(top_facts[0]["category"]) if top_facts else "general"
            importance = max(int(fact["importance"]) for fact in top_facts) if top_facts else 5
            salience = max(float(fact.get("salience") or 0.0) for fact in top_facts) if top_facts else 0.55
            updated_at = max(float(fact["updated_at"]) for fact in top_facts) if top_facts else now_ts()
            source_session_id = next(
                (str(fact.get("source_session_id") or "") for fact in top_facts if fact.get("source_session_id")), ""
            )
            decay_half_life_days = (
                max(float(fact.get("decay_half_life_days") or 0.0) for fact in top_facts) if top_facts else 60.0
            )
            title = pretty_topic(slug)
            existing = self._fetchone("SELECT * FROM topics WHERE slug = ?", (slug,))
            if existing:
                self._execute(
                    """
                    UPDATE topics
                    SET title = ?, category = ?, summary = ?, metadata_json = ?,
                        sensitivity = ?, importance = ?, salience = ?, source_session_id = ?,
                        decay_half_life_days = ?, updated_at = ?
                    WHERE slug = ?
                    """,
                    (
                        title,
                        category,
                        summary,
                        topic_metadata_json,
                        topic_sensitivity,
                        int(importance),
                        salience,
                        source_session_id,
                        decay_half_life_days,
                        updated_at,
                        slug,
                    ),
                )
                topic_id = int(existing["id"])
                action = "updated"
            else:
                cur = self._execute(
                    """
                    INSERT INTO topics(
                        slug, title, category, summary, metadata_json, sensitivity,
                        importance, salience, source_session_id, last_recalled_at,
                        decay_half_life_days, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                    """,
                    (
                        slug,
                        title,
                        category,
                        summary,
                        topic_metadata_json,
                        topic_sensitivity,
                        int(importance),
                        salience,
                        source_session_id,
                        decay_half_life_days,
                        updated_at,
                    ),
                )
                topic_id = int(cur.lastrowid)
                action = "inserted"
            topic_row = (
                self._fetchone(
                    """SELECT id, slug, title, summary, category, sensitivity, importance,
                              salience, source_session_id, last_recalled_at,
                              decay_half_life_days, updated_at
                       FROM topics WHERE slug = ?""",
                    (slug,),
                )
                or {}
            )
            if topic_row:
                self._upsert_topic_fts(topic_row)
                self.delete_links(source_kind="topic", source_id=topic_id, link_types=("supports",))
                for fact in membership_facts:
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
                SELECT id, content, category, topic, importance, confidence, salience, belief_score,
                       observation_count, valid_from, valid_until, sensitivity, memory_class, pinned,
                       updated_at, subject_key, value_key, polarity, exclusive, source_session_id,
                       metadata_json
                FROM facts
                WHERE active = 1
                  AND (valid_from=0 OR valid_from <= memory_now())
                  AND (valid_until=0 OR valid_until > memory_now())
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ),
            "topics": self._fetchall(
                """
                SELECT id, slug, title, summary, category, sensitivity,
                       importance, salience, updated_at, source_session_id, metadata_json
                FROM topics
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ),
            "episodes": self._fetchall(
                """
                SELECT id, session_id, digest, topic_hint, sensitivity, created_at
                FROM episodes
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ),
            "summaries": self._fetchall(
                """
                SELECT id, session_id, label, summary, summary_type, sensitivity, importance, salience, updated_at
                FROM memory_summaries
                WHERE active = 1
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ),
            "journals": self._fetchall(
                """
                SELECT id, session_id, label, content, journal_type, sensitivity, importance, salience, updated_at
                FROM memory_journals
                WHERE active = 1
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ),
            "preferences": self._fetchall(
                """
                SELECT id, preference_key, label, value, content, sensitivity, importance, salience,
                       updated_at, metadata_json
                FROM memory_preferences
                WHERE active = 1
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ),
            "policies": self._fetchall(
                """
                SELECT id, policy_key, label, content, sensitivity, importance, salience, updated_at,
                       metadata_json
                FROM memory_policies
                WHERE active = 1
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ),
            "contradictions": self.recent_contradictions(limit=limit),
        }

    def prompt_snapshot_rows(
        self,
        *,
        user_limit: int = 24,
        memory_limit: int = 36,
        preference_limit: int = 10,
        policy_limit: int = 16,
    ) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "user_facts": self._fetchall(
                """
                SELECT id,
                       content,
                       category,
                       topic,
                       source,
                       metadata_json,
                       importance,
                       salience,
                       updated_at,
                       subject_key,
                       value_key,
                       polarity,
                       exclusive,
                       source_session_id
                FROM facts
                WHERE active = 1
                  AND sensitivity = 'normal'
                  AND (valid_from=0 OR valid_from <= memory_now())
                  AND (valid_until=0 OR valid_until > memory_now())
                  AND subject_key LIKE 'user:%'
                  AND (exclusive = 1 OR importance >= 7 OR salience >= 0.82)
                  AND (source NOT LIKE 'builtin_memory:%' OR subject_key != '')
                ORDER BY exclusive DESC, importance DESC, salience DESC, updated_at DESC
                LIMIT ?
                """,
                (int(user_limit),),
            ),
            "memory_facts": self._fetchall(
                """
                SELECT id,
                       content,
                       category,
                       topic,
                       source,
                       metadata_json,
                       importance,
                       salience,
                       updated_at,
                       subject_key,
                       value_key,
                       polarity,
                       exclusive,
                       source_session_id
                FROM facts
                WHERE active = 1
                  AND sensitivity = 'normal'
                  AND (valid_from=0 OR valid_from <= memory_now())
                  AND (valid_until=0 OR valid_until > memory_now())
                  AND subject_key NOT LIKE 'user:%'
                  AND subject_key != ''
                  AND (source NOT LIKE 'builtin_memory:%' OR subject_key != '' OR category != 'user_pref')
                ORDER BY exclusive DESC, importance DESC, salience DESC, updated_at DESC
                LIMIT ?
                """,
                (int(memory_limit),),
            ),
            "preferences": self._fetchall(
                """
                SELECT id,
                       preference_key,
                       label,
                       value,
                       content,
                       metadata_json,
                       sensitivity,
                       source_session_id,
                       importance,
                       salience,
                       updated_at
                FROM memory_preferences
                WHERE active = 1 AND sensitivity = 'normal'
                ORDER BY importance DESC, salience DESC, updated_at DESC
                LIMIT ?
                """,
                (int(preference_limit),),
            ),
            "policies": self._fetchall(
                """
                SELECT id,
                       policy_key,
                       label,
                       content,
                       metadata_json,
                       sensitivity,
                       source_session_id,
                       importance,
                       salience,
                       updated_at
                FROM memory_policies
                WHERE active = 1 AND sensitivity = 'normal'
                ORDER BY importance DESC, salience DESC, updated_at DESC
                LIMIT ?
                """,
                (int(policy_limit),),
            ),
        }

    def scoped_recent_items(self, *, scope: str = "all", limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        recent = self.recent_items(limit=limit)
        if scope == "all":
            return {name: recent.get(name, []) for name in self.SEARCH_SCOPES}
        return {name: (recent.get(name, []) if name == scope else []) for name in self.SEARCH_SCOPES}

    def recent_contradictions(
        self,
        *,
        limit: int = 5,
        max_age_days: int | None = None,
        subject_keys: Sequence[str] | None = None,
    ) -> List[Dict[str, Any]]:
        params: List[Any] = []
        clauses: List[str] = []
        if max_age_days is not None:
            clauses.append("c.created_at >= ?")
            params.append(now_ts() - (int(max_age_days) * 86400))
        cleaned_subjects = [
            normalize_whitespace(str(item)) for item in (subject_keys or []) if normalize_whitespace(str(item))
        ]
        if cleaned_subjects:
            placeholders = ", ".join("?" for _ in cleaned_subjects)
            clauses.append(f"c.subject_key IN ({placeholders})")
            params.extend(cleaned_subjects)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
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
                   w.sensitivity AS winner_sensitivity,
                   l.id AS loser_fact_id,
                   l.content AS loser_content,
                   l.topic AS loser_topic,
                   l.sensitivity AS loser_sensitivity,
                   CASE
                       WHEN COALESCE(w.sensitivity, 'normal') = 'normal'
                        AND COALESCE(l.sensitivity, 'normal') = 'normal' THEN 'normal'
                       WHEN COALESCE(w.sensitivity, 'normal') = COALESCE(l.sensitivity, 'normal')
                           THEN COALESCE(w.sensitivity, 'normal')
                       ELSE 'sensitive'
                   END AS sensitivity
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
            results["summaries"] = self._search_summaries(
                clean, limit=max(1, min(limit, 6)), include_inactive=include_inactive
            )
        if scope in ("all", "journals"):
            results["journals"] = self._search_journals(
                clean, limit=max(1, min(limit, 6)), include_inactive=include_inactive
            )
        if scope in ("all", "preferences"):
            results["preferences"] = self._search_preferences(
                clean, limit=max(1, min(limit, 6)), include_inactive=include_inactive
            )
        if scope in ("all", "policies"):
            results["policies"] = self._search_policies(
                clean, limit=max(1, min(limit, 6)), include_inactive=include_inactive
            )
        return results

    def _search_facts(self, query: str, *, limit: int, include_inactive: bool) -> List[Dict[str, Any]]:
        active_clause = (
            ""
            if include_inactive
            else """AND f.active = 1
                      AND (f.valid_from=0 OR f.valid_from <= memory_now())
                      AND (f.valid_until=0 OR f.valid_until > memory_now())"""
        )
        if self._fts_enabled:
            try:
                return self._fetchall(
                    f"""
                    SELECT f.*
                    FROM facts_fts idx
                    JOIN facts f ON f.id = idx.fact_id
                    WHERE facts_fts MATCH ?
                      {active_clause}
                    ORDER BY bm25(facts_fts), f.salience DESC, f.importance DESC, f.updated_at DESC
                    LIMIT ?
                    """,
                    (fts_query(query), int(limit)),
                )
            except self._operational_errors:
                pass
        like = f"%{query}%"
        active_sql = (
            ""
            if include_inactive
            else """AND active = 1
                      AND (valid_from=0 OR valid_from <= memory_now())
                      AND (valid_until=0 OR valid_until > memory_now())"""
        )
        return self._fetchall(
            f"""
            SELECT *
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
                    SELECT t.id, t.slug, t.title, t.summary, t.category, t.sensitivity,
                           t.importance, t.salience, t.updated_at, t.source_session_id, t.metadata_json
                    FROM topics_fts idx
                    JOIN topics t ON t.id = idx.topic_id
                    WHERE topics_fts MATCH ?
                    ORDER BY bm25(topics_fts), t.salience DESC, t.importance DESC, t.updated_at DESC
                    LIMIT ?
                    """,
                    (fts_query(query), int(limit)),
                )
            except self._operational_errors:
                pass
        like = f"%{query}%"
        return self._fetchall(
            """
            SELECT id, slug, title, summary, category, sensitivity, importance,
                   salience, updated_at, source_session_id, metadata_json
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
                    SELECT e.id, e.session_id, e.digest, e.topic_hint, e.sensitivity, e.created_at
                    FROM episodes_fts idx
                    JOIN episodes e ON e.id = idx.episode_id
                    WHERE episodes_fts MATCH ?
                    ORDER BY bm25(episodes_fts), e.created_at DESC
                    LIMIT ?
                    """,
                    (fts_query(query), int(limit)),
                )
            except self._operational_errors:
                pass
        like = f"%{query}%"
        return self._fetchall(
            """
            SELECT id, session_id, digest, topic_hint, sensitivity, created_at
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
                    SELECT s.id, s.session_id, s.label, s.summary, s.summary_type, s.sensitivity, s.importance, s.salience, s.updated_at
                    FROM memory_summaries_fts idx
                    JOIN memory_summaries s ON s.id = idx.summary_id
                    WHERE memory_summaries_fts MATCH ?
                      {active_clause}
                    ORDER BY bm25(memory_summaries_fts), s.salience DESC, s.importance DESC, s.updated_at DESC
                    LIMIT ?
                    """,
                    (fts_query(query), int(limit)),
                )
            except self._operational_errors:
                pass
        like = f"%{query}%"
        active_sql = "" if include_inactive else "AND active = 1"
        return self._fetchall(
            f"""
            SELECT id, session_id, label, summary, summary_type, sensitivity, importance, salience, updated_at
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
                    SELECT j.id, j.session_id, j.label, j.content, j.journal_type, j.sensitivity, j.importance, j.salience, j.updated_at
                    FROM memory_journals_fts idx
                    JOIN memory_journals j ON j.id = idx.journal_id
                    WHERE memory_journals_fts MATCH ?
                      {active_clause}
                    ORDER BY bm25(memory_journals_fts), j.salience DESC, j.importance DESC, j.updated_at DESC
                    LIMIT ?
                    """,
                    (fts_query(query), int(limit)),
                )
            except self._operational_errors:
                pass
        like = f"%{query}%"
        active_sql = "" if include_inactive else "AND active = 1"
        return self._fetchall(
            f"""
            SELECT id, session_id, label, content, journal_type, sensitivity, importance, salience, updated_at
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
                    SELECT p.id, p.preference_key, p.label, p.value, p.content, p.sensitivity,
                           p.source_session_id, p.importance, p.salience, p.updated_at, p.metadata_json
                    FROM memory_preferences_fts idx
                    JOIN memory_preferences p ON p.id = idx.preference_id
                    WHERE memory_preferences_fts MATCH ?
                      {active_clause}
                    ORDER BY bm25(memory_preferences_fts), p.salience DESC, p.importance DESC, p.updated_at DESC
                    LIMIT ?
                    """,
                    (fts_query(query), int(limit)),
                )
            except self._operational_errors:
                pass
        like = f"%{query}%"
        active_sql = "" if include_inactive else "AND active = 1"
        return self._fetchall(
            f"""
            SELECT id, preference_key, label, value, content, sensitivity, source_session_id,
                   importance, salience, updated_at, metadata_json
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
                    SELECT p.id, p.policy_key, p.label, p.content, p.sensitivity, p.source_session_id,
                           p.importance, p.salience, p.updated_at, p.metadata_json
                    FROM memory_policies_fts idx
                    JOIN memory_policies p ON p.id = idx.policy_id
                    WHERE memory_policies_fts MATCH ?
                      {active_clause}
                    ORDER BY bm25(memory_policies_fts), p.salience DESC, p.importance DESC, p.updated_at DESC
                    LIMIT ?
                    """,
                    (fts_query(query), int(limit)),
                )
            except self._operational_errors:
                pass
        like = f"%{query}%"
        active_sql = "" if include_inactive else "AND active = 1"
        return self._fetchall(
            f"""
            SELECT id, policy_key, label, content, sensitivity, source_session_id, importance,
                   salience, updated_at, metadata_json
            FROM memory_policies
            WHERE (policy_key LIKE ? OR label LIKE ? OR content LIKE ?)
              {active_sql}
            ORDER BY salience DESC, importance DESC, updated_at DESC
            LIMIT ?
            """,
            (like, like, like, int(limit)),
        )

    @_transactional
    def touch_recall(
        self,
        kind: str,
        ids: Sequence[Any],
        *,
        session_id: str = "",
        review_intervals_days: Sequence[float] | None = None,
        reconsolidation_window_hours: float = 6.0,
        cues: Dict[str, Any] | None = None,
    ) -> None:
        if not ids:
            return
        now = now_ts()
        clean_kind = normalize_whitespace(kind)
        clean_session = normalize_whitespace(session_id)
        table_map = {
            "fact": ("facts", "id", True),
            "topic": ("topics", "id", False),
            "summary": ("memory_summaries", "id", True),
            "journal": ("memory_journals", "id", True),
            "preference": ("memory_preferences", "id", True),
            "policy": ("memory_policies", "id", True),
            "trace": ("memory_traces", "id", False),
        }
        table_info = table_map.get(clean_kind)
        if not table_info:
            return
        table, id_col, reviewable = table_info
        unique_ids = list(dict.fromkeys(str(item) for item in ids if str(item)))
        if not unique_ids:
            return
        reconsolidation_until = now + max(float(reconsolidation_window_hours), 0.0) * 3600.0
        for raw_id in unique_ids:
            row: Dict[str, Any] | None = None
            if reviewable:
                row = self._fetchone(
                    f"SELECT {id_col} AS id, salience, review_count FROM {table} WHERE {id_col} = ?",
                    (raw_id,),
                )
                if not row:
                    continue
                review_count = int(row.get("review_count") or 0) + 1
                next_review_at = now + _next_review_offset_seconds(review_count, review_intervals_days)
                boosted_salience = min(1.0, float(row.get("salience") or 0.0) + 0.04)
                self._execute(
                    f"""
                    UPDATE {table}
                    SET last_recalled_at = ?, salience = ?, review_count = ?, next_review_at = ?, reconsolidation_until = ?
                    WHERE {id_col} = ?
                    """,
                    (now, boosted_salience, review_count, next_review_at, reconsolidation_until, raw_id),
                )
            else:
                row = self._fetchone(f"SELECT {id_col} AS id FROM {table} WHERE {id_col} = ?", (raw_id,))
                if not row:
                    continue
                self._execute(f"UPDATE {table} SET last_recalled_at = ? WHERE {id_col} = ?", (now, raw_id))
            if clean_session:
                self.add_link("session", clean_session, clean_kind, raw_id, "recalls")
            if reviewable:
                self.record_history(
                    entity_kind=clean_kind,
                    entity_id=raw_id,
                    action="recalled",
                    reason="retrieval_practice",
                    source="recall",
                    payload={"session_id": clean_session, "cues": dict(cues or {})},
                )

    @_transactional
    def touch_recall_batch(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        *,
        session_id: str = "",
        review_intervals_days: Sequence[float] | None = None,
        reconsolidation_window_hours: float = 6.0,
        cues: Dict[str, Any] | None = None,
    ) -> None:
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
            self.touch_recall(
                kind,
                ids,
                session_id=session_id,
                review_intervals_days=review_intervals_days,
                reconsolidation_window_hours=reconsolidation_window_hours,
                cues=cues,
            )

    def review_due(self, *, scope: str = "all", limit: int = 8) -> Dict[str, List[Dict[str, Any]]]:
        now = now_ts()
        sections = ("facts", "summaries", "journals", "preferences", "policies")
        if scope != "all":
            sections = tuple(section for section in sections if section == scope)
        results = {name: [] for name in self.SEARCH_SCOPES}
        queries = {
            "facts": """
                SELECT id, content, category, topic, subject_key, source_session_id,
                       sensitivity, importance, salience, review_count, next_review_at, updated_at
                FROM facts
                WHERE active = 1 AND next_review_at > 0 AND next_review_at <= ?
                  AND (valid_from=0 OR valid_from <= memory_now())
                  AND (valid_until=0 OR valid_until > memory_now())
                ORDER BY next_review_at ASC, salience DESC, importance DESC, updated_at DESC
                LIMIT ?
            """,
            "summaries": """
                SELECT id, session_id, label, summary, summary_type, sensitivity, importance, salience, review_count, next_review_at, updated_at
                FROM memory_summaries
                WHERE active = 1 AND next_review_at > 0 AND next_review_at <= ?
                ORDER BY next_review_at ASC, salience DESC, importance DESC, updated_at DESC
                LIMIT ?
            """,
            "journals": """
                SELECT id, session_id, label, content, journal_type, sensitivity, importance, salience, review_count, next_review_at, updated_at
                FROM memory_journals
                WHERE active = 1 AND next_review_at > 0 AND next_review_at <= ?
                ORDER BY next_review_at ASC, salience DESC, importance DESC, updated_at DESC
                LIMIT ?
            """,
            "preferences": """
                SELECT id, preference_key, label, value, content, sensitivity, source_session_id, importance, salience, review_count, next_review_at, updated_at
                FROM memory_preferences
                WHERE active = 1 AND next_review_at > 0 AND next_review_at <= ?
                ORDER BY next_review_at ASC, salience DESC, importance DESC, updated_at DESC
                LIMIT ?
            """,
            "policies": """
                SELECT id, policy_key, label, content, sensitivity, source_session_id, importance, salience, review_count, next_review_at, updated_at
                FROM memory_policies
                WHERE active = 1 AND next_review_at > 0 AND next_review_at <= ?
                ORDER BY next_review_at ASC, salience DESC, importance DESC, updated_at DESC
                LIMIT ?
            """,
        }
        for section in sections:
            rows = self._fetchall(queries[section], (now, int(limit)))
            for row in rows:
                row["review_overdue_days"] = round(
                    max((now - float(row.get("next_review_at") or 0.0)) / 86400.0, 0.0), 3
                )
            results[section] = rows
        return results

    def review_status(self) -> Dict[str, Any]:
        now = now_ts()
        sections = {
            "facts": """SELECT COUNT(*) AS count, MIN(next_review_at) AS next_due_at
                        FROM facts WHERE active = 1 AND next_review_at > 0 AND next_review_at <= ?
                          AND (valid_from=0 OR valid_from <= memory_now())
                          AND (valid_until=0 OR valid_until > memory_now())""",
            "summaries": "SELECT COUNT(*) AS count, MIN(next_review_at) AS next_due_at FROM memory_summaries WHERE active = 1 AND next_review_at > 0 AND next_review_at <= ?",
            "journals": "SELECT COUNT(*) AS count, MIN(next_review_at) AS next_due_at FROM memory_journals WHERE active = 1 AND next_review_at > 0 AND next_review_at <= ?",
            "preferences": "SELECT COUNT(*) AS count, MIN(next_review_at) AS next_due_at FROM memory_preferences WHERE active = 1 AND next_review_at > 0 AND next_review_at <= ?",
            "policies": "SELECT COUNT(*) AS count, MIN(next_review_at) AS next_due_at FROM memory_policies WHERE active = 1 AND next_review_at > 0 AND next_review_at <= ?",
        }
        results: Dict[str, Any] = {"due_counts": {}, "total_due": 0, "next_due_at": 0.0}
        next_due_values: List[float] = []
        for name, sql in sections.items():
            row = self._fetchone(sql, (now,)) or {"count": 0, "next_due_at": 0}
            count = int(row.get("count") or 0)
            next_due = float(row.get("next_due_at") or 0.0)
            results["due_counts"][name] = count
            results["total_due"] += count
            if next_due > 0:
                next_due_values.append(next_due)
        results["next_due_at"] = min(next_due_values) if next_due_values else 0.0
        return results

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
            SELECT id, session_id, label, summary, summary_type, sensitivity, importance, salience, updated_at
            FROM memory_summaries
            WHERE active = 1 AND summary_type = 'session'
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (int(limit),),
        )

    @_transactional
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

    @_transactional
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
            SELECT id, category, importance, salience, last_recalled_at, last_seen_at, updated_at, decay_half_life_days, pinned
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
            if (
                next_salience < threshold
                and int(row.get("importance") or 0) <= 4
                and str(row.get("category") or "") == "general"
                and not int(row.get("pinned") or 0)
            ):
                if self.deactivate_fact(int(row["id"]), reason="decay", source="decay"):
                    stats["facts_deactivated"] += 1

        stats["topics_decayed"] = self._decay_table(
            "topics", now=now, half_life=half_life, threshold=threshold, last_decay_at=last_decay_at
        )
        journal_stats = self._decay_table(
            "memory_journals",
            now=now,
            half_life=half_life,
            threshold=threshold,
            last_decay_at=last_decay_at,
            deactivate=True,
            max_keep_importance=5,
        )
        trace_stats = self._decay_table(
            "memory_traces",
            now=now,
            half_life=half_life,
            threshold=threshold,
            last_decay_at=last_decay_at,
            deactivate=True,
            max_keep_importance=4,
        )
        summary_stats = self._decay_table(
            "memory_summaries",
            now=now,
            half_life=half_life,
            threshold=threshold,
            last_decay_at=last_decay_at,
            deactivate=True,
            max_keep_importance=5,
        )
        stats["journals_decayed"], stats["journals_deactivated"] = journal_stats
        stats["traces_decayed"], stats["traces_deactivated"] = trace_stats
        stats["summaries_decayed"], stats["summaries_deactivated"] = summary_stats
        stats["preferences_decayed"] = self._decay_table(
            "memory_preferences",
            now=now,
            half_life=max(half_life * 2.0, 1.0),
            threshold=0.0,
            last_decay_at=last_decay_at,
        )
        stats["policies_decayed"] = self._decay_table(
            "memory_policies", now=now, half_life=max(half_life * 3.0, 1.0), threshold=0.0, last_decay_at=last_decay_at
        )
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
            anchor = max(
                float(row.get("updated_at") or 0), float(row.get("last_recalled_at") or 0), float(last_decay_at or 0)
            )
            age_days = max((now - anchor) / 86400.0, 0.0)
            next_salience = max(
                0.01, float(row.get("salience") or 0.0) * math.pow(0.5, age_days / max(half_life, 0.01))
            )
            self._execute(f"UPDATE {table} SET salience = ? WHERE id = ?", (next_salience, int(row["id"])))
            changed += 1
            if deactivate and next_salience < threshold and int(row.get("importance") or 0) <= max_keep_importance:
                self._execute(f"UPDATE {table} SET active = 0, updated_at = ? WHERE id = ?", (now, int(row["id"])))
                if table == "memory_journals":
                    self._delete_journal_fts(int(row["id"]))
                elif table == "memory_traces":
                    self._delete_trace_fts(int(row["id"]))
                elif table == "memory_summaries":
                    self._delete_summary_fts(int(row["id"]))
                deactivated += 1
        if deactivate:
            return changed, deactivated
        return changed

    def _upsert_fact_fts(self, fact: Dict[str, Any]) -> None:
        if not self._fts_enabled:
            return
        self._delete_fact_fts(int(fact["id"]))
        self._execute(
            "INSERT INTO facts_fts(fact_id, content, topic, category, subject_key) VALUES (?, ?, ?, ?, ?)",
            (
                int(fact["id"]),
                fact["content"],
                fact["topic"],
                fact["category"],
                fact.get("subject_key", ""),
            ),
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
            (
                int(summary["id"]),
                summary["label"],
                summary["summary"],
                summary.get("content", ""),
                summary["summary_type"],
            ),
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

    # ------------------------------------------------------------------
    # Evidence, temporal state, and higher-level memory systems

    def explain_fact(self, fact_id: int) -> Dict[str, Any]:
        fact = self._fetchone("SELECT * FROM facts WHERE id = ?", (int(fact_id),))
        if not fact:
            raise ValueError(f"Unknown fact id: {fact_id}")
        evidence = self._fetchall(
            "SELECT * FROM belief_evidence WHERE fact_id = ? ORDER BY observed_at DESC, id DESC",
            (int(fact_id),),
        )
        history = self.list_history(memory_type="fact", entity_id=str(fact_id), limit=100)
        links = self.list_links(source_kind="fact", source_id=str(fact_id), limit=100)
        return {
            "fact": fact,
            "evidence": evidence,
            "history": history,
            "links": links,
            "explanation": (
                f"Belief score {float(fact.get('belief_score') or 0):.3f} from "
                f"{int(fact.get('observation_count') or 0)} observation(s)."
            ),
        }

    @_transactional
    def pin_fact(self, fact_id: int, pinned: bool = True) -> Dict[str, Any]:
        self._execute(
            "UPDATE facts SET pinned = ?, updated_at = ?, revision = revision + 1 WHERE id = ?",
            (int(bool(pinned)), now_ts(), int(fact_id)),
        )
        row = self._fetchone("SELECT * FROM facts WHERE id = ?", (int(fact_id),))
        if not row:
            raise ValueError(f"Unknown fact id: {fact_id}")
        self.record_history(
            entity_kind="fact",
            entity_id=fact_id,
            subject_key=str(row.get("subject_key") or ""),
            action="pinned" if pinned else "unpinned",
            reason="manual",
            source="memory-tool",
            payload=row,
        )
        return row

    @_transactional
    def set_working_memory(
        self,
        *,
        session_id: str,
        memory_key: str,
        content: str,
        priority: int = 5,
        ttl_seconds: float = 3600,
        metadata: Dict[str, Any] | None = None,
        capacity: int = 12,
        sensitivity: str = "normal",
    ) -> Dict[str, Any]:
        clean_session = normalize_whitespace(session_id)
        clean_key = slugify(memory_key)
        clean_content = normalize_whitespace(content)
        if not clean_session or not clean_content:
            raise ValueError("Working memory requires a session id and content.")
        now = now_ts()
        expires_at = now + max(1.0, float(ttl_seconds)) if ttl_seconds else 0.0
        self._execute("DELETE FROM working_memory WHERE expires_at > 0 AND expires_at <= ?", (now,))
        self._execute(
            """
            INSERT INTO working_memory(
                session_id, memory_key, content, priority, expires_at, metadata_json, sensitivity, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, memory_key) DO UPDATE SET
                content = excluded.content, priority = excluded.priority,
                expires_at = excluded.expires_at, metadata_json = excluded.metadata_json,
                sensitivity = excluded.sensitivity,
                updated_at = excluded.updated_at
            """,
            (
                clean_session,
                clean_key,
                clean_content,
                _clamp_int(priority, 1, 10, 5),
                expires_at,
                json.dumps(metadata or {}, sort_keys=True),
                normalize_text(sensitivity) or "normal",
                now,
                now,
            ),
        )
        keep = max(1, int(capacity))
        overflow = self._fetchall(
            """
            SELECT id FROM working_memory WHERE session_id = ?
            ORDER BY priority DESC, updated_at DESC LIMIT -1 OFFSET ?
            """,
            (clean_session, keep),
        )
        if overflow:
            placeholders = ",".join("?" for _ in overflow)
            self._execute(
                f"DELETE FROM working_memory WHERE id IN ({placeholders})",
                tuple(int(row["id"]) for row in overflow),
            )
        return (
            self._fetchone(
                "SELECT * FROM working_memory WHERE session_id = ? AND memory_key = ?",
                (clean_session, clean_key),
            )
            or {}
        )

    def list_working_memory(self, session_id: str, *, limit: int = 12) -> List[Dict[str, Any]]:
        now = now_ts()
        return self._fetchall(
            """
            SELECT * FROM working_memory
            WHERE session_id = ? AND (expires_at = 0 OR expires_at > ?)
            ORDER BY priority DESC, updated_at DESC LIMIT ?
            """,
            (normalize_whitespace(session_id), now, max(1, int(limit))),
        )

    @_transactional
    def clear_working_memory(self, session_id: str, memory_key: str = "") -> int:
        if memory_key:
            cur = self._execute(
                "DELETE FROM working_memory WHERE session_id = ? AND memory_key = ?",
                (normalize_whitespace(session_id), slugify(memory_key)),
            )
        else:
            cur = self._execute("DELETE FROM working_memory WHERE session_id = ?", (normalize_whitespace(session_id),))
        return int(cur.rowcount or 0)

    @_transactional
    def upsert_procedure(
        self,
        *,
        procedure_key: str,
        label: str,
        steps: Sequence[str],
        prerequisites: Sequence[str] | None = None,
        success_criteria: str = "",
        failure_recovery: str = "",
        confidence: float = 0.6,
        metadata: Dict[str, Any] | None = None,
        sensitivity: str = "normal",
    ) -> Dict[str, Any]:
        key = slugify(procedure_key)
        clean_steps = [normalize_whitespace(str(step)) for step in steps if normalize_whitespace(str(step))]
        if not clean_steps:
            raise ValueError("A procedure requires at least one step.")
        now = now_ts()
        self._execute(
            """
            INSERT INTO memory_procedures(
                procedure_key, label, steps_json, prerequisites_json, success_criteria,
                failure_recovery, confidence, metadata_json, sensitivity, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(procedure_key) DO UPDATE SET
                label = excluded.label, steps_json = excluded.steps_json,
                prerequisites_json = excluded.prerequisites_json,
                success_criteria = excluded.success_criteria,
                failure_recovery = excluded.failure_recovery,
                confidence = excluded.confidence, metadata_json = excluded.metadata_json,
                sensitivity = excluded.sensitivity,
                active = 1, updated_at = excluded.updated_at
            """,
            (
                key,
                normalize_whitespace(label) or pretty_topic(key),
                json.dumps(clean_steps),
                json.dumps(list(prerequisites or [])),
                normalize_whitespace(success_criteria),
                normalize_whitespace(failure_recovery),
                _clamp_float(confidence, 0, 1, 0.6),
                json.dumps(metadata or {}, sort_keys=True),
                normalize_text(sensitivity) or "normal",
                now,
                now,
            ),
        )
        return self._fetchone("SELECT * FROM memory_procedures WHERE procedure_key = ?", (key,)) or {}

    def list_procedures(self, query: str = "", *, limit: int = 20) -> List[Dict[str, Any]]:
        if query:
            pattern = f"%{normalize_whitespace(query)}%"
            return self._fetchall(
                """SELECT * FROM memory_procedures WHERE active = 1
                   AND (label LIKE ? OR procedure_key LIKE ? OR steps_json LIKE ?)
                   ORDER BY confidence DESC, success_count DESC, updated_at DESC LIMIT ?""",
                (pattern, pattern, pattern, max(1, int(limit))),
            )
        return self._fetchall(
            "SELECT * FROM memory_procedures WHERE active = 1 ORDER BY confidence DESC, updated_at DESC LIMIT ?",
            (max(1, int(limit)),),
        )

    @_transactional
    def record_procedure_result(self, procedure_key: str, *, success: bool) -> Dict[str, Any]:
        key = slugify(procedure_key)
        row = self._fetchone("SELECT * FROM memory_procedures WHERE procedure_key = ?", (key,))
        if not row:
            raise ValueError(f"Unknown procedure: {procedure_key}")
        uses = int(row.get("use_count") or 0) + 1
        successes = int(row.get("success_count") or 0) + int(bool(success))
        observed_rate = successes / uses
        confidence = (float(row.get("confidence") or 0.6) * 0.8) + (observed_rate * 0.2)
        self._execute(
            "UPDATE memory_procedures SET use_count=?, success_count=?, confidence=?, last_used_at=?, updated_at=? WHERE id=?",
            (uses, successes, confidence, now_ts(), now_ts(), int(row["id"])),
        )
        return self._fetchone("SELECT * FROM memory_procedures WHERE id = ?", (int(row["id"]),)) or {}

    @_transactional
    def add_intention(
        self,
        *,
        intention: str,
        due_at: float = 0,
        condition_text: str = "",
        recurrence: str = "",
        importance: int = 6,
        session_id: str = "",
        metadata: Dict[str, Any] | None = None,
        sensitivity: str = "normal",
    ) -> Dict[str, Any]:
        clean = normalize_whitespace(intention)
        if not clean:
            raise ValueError("Intention cannot be empty.")
        now = now_ts()
        cur = self._execute(
            """
            INSERT INTO prospective_memories(
                intention, due_at, condition_text, recurrence, status, importance,
                session_id, metadata_json, sensitivity, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
            """,
            (
                clean,
                max(0.0, float(due_at)),
                normalize_whitespace(condition_text),
                normalize_whitespace(recurrence),
                _clamp_int(importance, 1, 10, 6),
                normalize_whitespace(session_id),
                json.dumps(metadata or {}, sort_keys=True),
                normalize_text(sensitivity) or "normal",
                now,
                now,
            ),
        )
        return self._fetchone("SELECT * FROM prospective_memories WHERE id = ?", (int(cur.lastrowid),)) or {}

    def list_intentions(self, *, due_only: bool = False, limit: int = 20) -> List[Dict[str, Any]]:
        if due_only:
            return self._fetchall(
                """SELECT * FROM prospective_memories WHERE status = 'pending'
                   AND ((due_at > 0 AND due_at <= ?) OR (due_at = 0 AND condition_text = ''))
                   ORDER BY importance DESC, due_at ASC, id ASC LIMIT ?""",
                (now_ts(), max(1, int(limit))),
            )
        return self._fetchall(
            "SELECT * FROM prospective_memories WHERE status = 'pending' ORDER BY importance DESC, due_at ASC LIMIT ?",
            (max(1, int(limit)),),
        )

    def intentions_for_context(self, query: str, *, limit: int = 20) -> List[Dict[str, Any]]:
        candidates = self._fetchall(
            "SELECT * FROM prospective_memories WHERE status='pending' ORDER BY importance DESC, due_at ASC LIMIT 100"
        )
        clean_query = normalize_text(query)
        now = now_ts()
        selected: List[Dict[str, Any]] = []
        for row in candidates:
            due_at = float(row.get("due_at") or 0)
            condition = normalize_text(str(row.get("condition_text") or ""))
            due = due_at > 0 and due_at <= now
            unconditional = due_at == 0 and not condition
            condition_match = bool(condition and clean_query and condition in clean_query)
            if due or unconditional or condition_match:
                row["trigger_reason"] = "due" if due else ("condition" if condition_match else "unconditional")
                selected.append(row)
            if len(selected) >= max(1, int(limit)):
                break
        return selected

    @_transactional
    def resolve_intention(self, intention_id: int, *, status: str = "completed") -> Dict[str, Any]:
        clean_status = normalize_text(status)
        if clean_status not in {"completed", "cancelled", "pending"}:
            raise ValueError("Intention status must be completed, cancelled, or pending.")
        row = self._fetchone("SELECT * FROM prospective_memories WHERE id = ?", (int(intention_id),))
        if not row:
            raise ValueError(f"Unknown intention id: {intention_id}")
        now = now_ts()
        recurrence = normalize_text(str(row.get("recurrence") or ""))
        recurrence_seconds = {"daily": 86400.0, "weekly": 604800.0, "monthly": 2592000.0}.get(recurrence, 0.0)
        if clean_status == "completed" and recurrence_seconds:
            next_due = max(now, float(row.get("due_at") or now)) + recurrence_seconds
            self._execute(
                """UPDATE prospective_memories SET status='pending', due_at=?,
                   last_triggered_at=?, updated_at=? WHERE id=?""",
                (next_due, now, now, int(intention_id)),
            )
        else:
            self._execute(
                "UPDATE prospective_memories SET status = ?, last_triggered_at = ?, updated_at = ? WHERE id = ?",
                (clean_status, now, now, int(intention_id)),
            )
        return self._fetchone("SELECT * FROM prospective_memories WHERE id = ?", (int(intention_id),)) or {}

    @_transactional
    def upsert_autobiographical_event(
        self,
        *,
        event_key: str,
        content: str,
        event_at: float = 0,
        valid_from: float = 0,
        valid_until: float = 0,
        people: Sequence[str] | None = None,
        places: Sequence[str] | None = None,
        importance: int = 6,
        metadata: Dict[str, Any] | None = None,
        sensitivity: str = "normal",
    ) -> Dict[str, Any]:
        key = slugify(event_key or fingerprint_text(content)[:16])
        clean = normalize_whitespace(content)
        if not clean:
            raise ValueError("Autobiographical event cannot be empty.")
        now = now_ts()
        self._execute(
            """
            INSERT INTO autobiographical_events(
                event_key, content, event_at, valid_from, valid_until, people_json,
                places_json, metadata_json, sensitivity, importance, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_key) DO UPDATE SET
                content=excluded.content, event_at=excluded.event_at,
                valid_from=excluded.valid_from, valid_until=excluded.valid_until,
                people_json=excluded.people_json, places_json=excluded.places_json,
                metadata_json=excluded.metadata_json, sensitivity=excluded.sensitivity,
                importance=excluded.importance,
                active=1, updated_at=excluded.updated_at
            """,
            (
                key,
                clean,
                _timestamp(event_at),
                _timestamp(valid_from),
                _timestamp(valid_until),
                json.dumps(list(people or [])),
                json.dumps(list(places or [])),
                json.dumps(metadata or {}, sort_keys=True),
                normalize_text(sensitivity) or "normal",
                _clamp_int(importance, 1, 10, 6),
                now,
                now,
            ),
        )
        return self._fetchone("SELECT * FROM autobiographical_events WHERE event_key = ?", (key,)) or {}

    def list_autobiographical_events(self, query: str = "", *, limit: int = 20) -> List[Dict[str, Any]]:
        if query:
            pattern = f"%{normalize_whitespace(query)}%"
            return self._fetchall(
                """SELECT * FROM autobiographical_events WHERE active=1
                   AND (valid_from=0 OR valid_from <= memory_now())
                   AND (valid_until=0 OR valid_until > memory_now())
                   AND (content LIKE ? OR people_json LIKE ? OR places_json LIKE ?)
                   ORDER BY event_at DESC, importance DESC LIMIT ?""",
                (pattern, pattern, pattern, max(1, int(limit))),
            )
        return self._fetchall(
            """SELECT * FROM autobiographical_events WHERE active=1
               AND (valid_from=0 OR valid_from <= memory_now())
               AND (valid_until=0 OR valid_until > memory_now())
               ORDER BY event_at DESC, importance DESC LIMIT ?""",
            (max(1, int(limit)),),
        )

    @_transactional
    def associate(
        self,
        left_kind: str,
        left_id: str | int,
        right_kind: str,
        right_id: str | int,
        relation: str = "associated",
        *,
        weight: float = 0.5,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        left = (normalize_text(left_kind), str(left_id))
        right = (normalize_text(right_kind), str(right_id))
        if not all((*left, *right)):
            raise ValueError("Association kinds and identifiers cannot be empty.")
        self._require_reference(left[0], left[1])
        self._require_reference(right[0], right[1])
        if left == right:
            raise ValueError("An entity cannot be associated with itself.")
        if right < left:
            left, right = right, left
        now = now_ts()
        self._execute(
            """
            INSERT INTO memory_associations(
                left_kind, left_id, right_kind, right_id, relation, weight,
                cooccurrence_count, last_activated_at, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
            ON CONFLICT(left_kind, left_id, right_kind, right_id, relation) DO UPDATE SET
                weight = MIN(1.0, memory_associations.weight + (excluded.weight * 0.2)),
                cooccurrence_count = memory_associations.cooccurrence_count + 1,
                last_activated_at = excluded.last_activated_at,
                metadata_json = excluded.metadata_json, updated_at = excluded.updated_at
            """,
            (
                left[0],
                left[1],
                right[0],
                right[1],
                normalize_text(relation) or "associated",
                _clamp_float(weight, 0, 1, 0.5),
                now,
                json.dumps(metadata or {}, sort_keys=True),
                now,
                now,
            ),
        )
        return (
            self._fetchone(
                """SELECT * FROM memory_associations WHERE left_kind=? AND left_id=?
               AND right_kind=? AND right_id=? AND relation=?""",
                (left[0], left[1], right[0], right[1], normalize_text(relation) or "associated"),
            )
            or {}
        )

    def list_associations(self, kind: str, entity_id: str | int, *, limit: int = 20) -> List[Dict[str, Any]]:
        return self._fetchall(
            """SELECT * FROM memory_associations
               WHERE (left_kind=? AND left_id=?) OR (right_kind=? AND right_id=?)
               ORDER BY weight DESC, cooccurrence_count DESC LIMIT ?""",
            (normalize_text(kind), str(entity_id), normalize_text(kind), str(entity_id), max(1, int(limit))),
        )

    def associated_facts(self, fact_ids: Sequence[int], *, limit: int = 8) -> List[Dict[str, Any]]:
        ids = list(dict.fromkeys(str(int(value)) for value in fact_ids if int(value) > 0))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        return self._fetchall(
            f"""
            SELECT DISTINCT f.*, a.weight AS association_weight, a.relation,
                   a.cooccurrence_count
            FROM memory_associations a
            JOIN facts f ON CAST(f.id AS TEXT) = CASE
                WHEN a.left_kind='fact' AND a.left_id IN ({placeholders}) THEN a.right_id
                ELSE a.left_id END
            WHERE f.active=1
              AND (f.valid_from=0 OR f.valid_from <= memory_now())
              AND (f.valid_until=0 OR f.valid_until > memory_now())
              AND ((a.left_kind='fact' AND a.left_id IN ({placeholders}) AND a.right_kind='fact')
                OR (a.right_kind='fact' AND a.right_id IN ({placeholders}) AND a.left_kind='fact'))
            ORDER BY a.weight DESC, a.cooccurrence_count DESC, f.belief_score DESC
            LIMIT ?
            """,
            tuple(ids + ids + ids + [max(1, int(limit))]),
        )

    @_transactional
    def associate_fact_group(self, fact_ids: Sequence[int], *, relation: str = "coobserved") -> int:
        ids = list(dict.fromkeys(int(value) for value in fact_ids if int(value) > 0))[:30]
        count = 0
        for index, left_id in enumerate(ids):
            for right_id in ids[index + 1 :]:
                self.associate("fact", left_id, "fact", right_id, relation, weight=0.35)
                count += 1
        return count

    @_transactional
    def merge_facts(self, winner_fact_id: int, loser_fact_ids: Sequence[int]) -> Dict[str, Any]:
        winner = self._fetchone("SELECT * FROM facts WHERE id=?", (int(winner_fact_id),))
        if not winner:
            raise ValueError(f"Unknown winner fact id: {winner_fact_id}")
        if int(winner.get("active") or 0) != 1:
            raise ValueError(f"Winner fact {winner_fact_id} is inactive")
        clean_loser_ids = list(
            dict.fromkeys(int(value) for value in loser_fact_ids if int(value) != int(winner_fact_id))
        )
        if not clean_loser_ids:
            raise ValueError("At least one distinct loser fact id is required")
        losers: Dict[int, Dict[str, Any]] = {}
        for loser_id in clean_loser_ids:
            loser = self._fetchone("SELECT * FROM facts WHERE id=?", (loser_id,))
            if not loser:
                raise ValueError(f"Unknown loser fact id: {loser_id}")
            if int(loser.get("active") or 0) != 1:
                raise ValueError(f"Loser fact {loser_id} is inactive")
            losers[loser_id] = loser
        merged: List[int] = []
        for loser_id in losers:
            self._execute("UPDATE belief_evidence SET fact_id=? WHERE fact_id=?", (int(winner_fact_id), loser_id))
            self._soft_supersede_fact(
                loser_id,
                int(winner_fact_id),
                now_ts(),
                subject_key=str(winner.get("subject_key") or ""),
                reason="manual_merge",
            )
            merged.append(loser_id)
        evidence = (
            self._fetchone(
                "SELECT COUNT(*) AS count, AVG(confidence) AS confidence, MAX(reliability) AS reliability FROM belief_evidence WHERE fact_id=?",
                (int(winner_fact_id),),
            )
            or {}
        )
        observations = int(evidence.get("count") or 1)
        confidence = float(evidence.get("confidence") or winner.get("confidence") or 0.5)
        reliability = float(evidence.get("reliability") or 0.5)
        self._execute(
            "UPDATE facts SET observation_count=?, confidence=?, belief_score=?, revision=revision+1, updated_at=? WHERE id=?",
            (
                observations,
                confidence,
                self._belief_score(
                    confidence=confidence, reliability=reliability, explicit_correction=False, observations=observations
                ),
                now_ts(),
                int(winner_fact_id),
            ),
        )
        return {"winner": self._fetchone("SELECT * FROM facts WHERE id=?", (int(winner_fact_id),)), "merged": merged}

    @_transactional
    def split_fact(self, fact_id: int, contents: Sequence[str]) -> Dict[str, Any]:
        original = self._fetchone("SELECT * FROM facts WHERE id=?", (int(fact_id),))
        if not original:
            raise ValueError(f"Unknown fact id: {fact_id}")
        clean_contents = list(
            dict.fromkeys(normalize_whitespace(value) for value in contents if normalize_whitespace(value))
        )
        if len(clean_contents) < 2:
            raise ValueError("Splitting requires at least two non-empty replacement facts.")
        created: List[Dict[str, Any]] = []
        split_metadata = dict(original.get("metadata") or {})
        for state_key in ("subject_key", "value_key", "exclusive", "polarity"):
            split_metadata.pop(state_key, None)
        split_metadata["split_from_fact_id"] = int(fact_id)
        for content in clean_contents:
            result = self.upsert_fact(
                content=content,
                category=str(original.get("category") or "general"),
                topic=str(original.get("topic") or "general"),
                source="manual_split",
                importance=int(original.get("importance") or 5),
                confidence=float(original.get("confidence") or 0.7),
                metadata=split_metadata,
                source_session_id=str(original.get("source_session_id") or ""),
                valid_from=float(original.get("valid_from") or 0),
                valid_until=float(original.get("valid_until") or 0),
                temporal_kind=str(original.get("temporal_kind") or "atemporal"),
                event_at=float(original.get("event_at") or 0),
                temporal_precision=str(original.get("temporal_precision") or "unknown"),
                temporal_timezone=str(original.get("temporal_timezone") or ""),
                temporal_confidence=float(original.get("temporal_confidence") or 0),
                sensitivity=str(original.get("sensitivity") or "normal"),
                memory_class=str(original.get("memory_class") or "semantic"),
                pinned=bool(original.get("pinned")),
                history_reason=f"split_from:{fact_id}",
            )
            child = dict(result.get("fact") or {})
            if child.get("id") is not None:
                self.add_link("fact", child["id"], "fact", fact_id, "split_from")
                created.append(child)
        self.deactivate_fact(int(fact_id), reason="manual_split", source="memory-tool")
        return {"original_fact_id": int(fact_id), "created": created}

    @_transactional
    def request_approval(
        self,
        *,
        candidate: Dict[str, Any],
        sensitivity: str,
        reason: str,
        session_id: str = "",
    ) -> Dict[str, Any]:
        now = now_ts()
        identity = {
            key: candidate.get(key)
            for key in ("content", "category", "topic", "metadata", "_memory_type")
            if key in candidate
        }
        candidate_fingerprint = fingerprint_text(json.dumps(identity, sort_keys=True, default=str))
        existing = self._fetchone(
            "SELECT * FROM memory_approvals WHERE status='pending' AND candidate_fingerprint=? ORDER BY id DESC LIMIT 1",
            (candidate_fingerprint,),
        )
        if existing:
            return existing
        cur = self._execute(
            """INSERT INTO memory_approvals(
                   candidate_json, candidate_fingerprint, sensitivity, reason, status, session_id, created_at
               ) VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
            (
                json.dumps(candidate, sort_keys=True, default=str),
                candidate_fingerprint,
                normalize_text(sensitivity) or "sensitive",
                normalize_whitespace(reason),
                normalize_whitespace(session_id),
                now,
            ),
        )
        return self._fetchone("SELECT * FROM memory_approvals WHERE id = ?", (int(cur.lastrowid),)) or {}

    def list_approvals(self, *, status: str = "pending", limit: int = 50) -> List[Dict[str, Any]]:
        return self._fetchall(
            "SELECT * FROM memory_approvals WHERE status = ? ORDER BY created_at DESC LIMIT ?",
            (normalize_text(status), max(1, int(limit))),
        )

    @_transactional
    def resolve_approval(self, approval_id: int, *, approved: bool, resolution: str = "") -> Dict[str, Any]:
        existing = self._fetchone("SELECT * FROM memory_approvals WHERE id = ?", (int(approval_id),))
        if not existing:
            raise ValueError(f"Unknown approval id: {approval_id}")
        if str(existing.get("status") or "") != "pending":
            raise ValueError(f"Approval {approval_id} has already been resolved")
        status = "approved" if approved else "rejected"
        self._execute(
            "UPDATE memory_approvals SET status=?, resolved_at=?, resolution=? WHERE id=? AND status='pending'",
            (status, now_ts(), normalize_whitespace(resolution), int(approval_id)),
        )
        row = self._fetchone("SELECT * FROM memory_approvals WHERE id = ?", (int(approval_id),))
        if not row or str(row.get("status") or "") != status:
            raise RuntimeError(f"Approval {approval_id} could not be resolved")
        return row

    # ------------------------------------------------------------------
    # Durable work queue and cross-process maintenance coordination

    @_transactional
    def enqueue_operation(self, operation_type: str, payload: Dict[str, Any], *, available_at: float = 0) -> int:
        now = now_ts()
        cur = self._execute(
            """INSERT INTO pending_operations(
                   operation_type, payload_json, status, available_at, created_at, updated_at
               ) VALUES (?, ?, 'pending', ?, ?, ?)""",
            (
                normalize_text(operation_type),
                json.dumps(payload, sort_keys=True, default=str),
                max(0.0, float(available_at)),
                now,
                now,
            ),
        )
        return int(cur.lastrowid)

    @_transactional
    def claim_operations(
        self,
        *,
        limit: int = 25,
        stale_after_seconds: float = 300,
        max_attempts: int = 5,
    ) -> List[Dict[str, Any]]:
        now = now_ts()
        attempt_limit = max(1, int(max_attempts))
        stale_before = now - max(1.0, float(stale_after_seconds))
        self._execute(
            """UPDATE pending_operations
               SET status='failed', available_at=0, claimed_at=0,
                   error=CASE WHEN error='' THEN 'Worker lease expired after maximum attempts' ELSE error END,
                   updated_at=?
               WHERE status='running' AND claimed_at < ? AND attempts >= ?""",
            (now, stale_before, attempt_limit),
        )
        self._execute(
            """UPDATE pending_operations SET status='pending', claimed_at=0, updated_at=?
               WHERE status='running' AND claimed_at < ? AND attempts < ?""",
            (now, stale_before, attempt_limit),
        )
        # Migrate any legacy poison rows that reached the limit while pending.
        self._execute(
            """UPDATE pending_operations
               SET status='failed', available_at=0, claimed_at=0,
                   error=CASE WHEN error='' THEN 'Maximum retry attempts reached' ELSE error END,
                   updated_at=?
               WHERE status='pending' AND attempts >= ?""",
            (now, attempt_limit),
        )
        rows = self._fetchall(
            """SELECT * FROM pending_operations
               WHERE status='pending' AND attempts < ? AND available_at <= ?
               ORDER BY id ASC LIMIT ?""",
            (attempt_limit, now, max(1, int(limit))),
        )
        for row in rows:
            self._execute(
                "UPDATE pending_operations SET status='running', attempts=attempts+1, claimed_at=?, updated_at=? WHERE id=?",
                (now, now, int(row["id"])),
            )
            row["status"] = "running"
            row["attempts"] = int(row.get("attempts") or 0) + 1
            row["claimed_at"] = now
        return rows

    @_transactional
    def complete_operation(self, operation_id: int) -> None:
        self._execute("DELETE FROM pending_operations WHERE id = ?", (int(operation_id),))

    @_transactional
    def fail_operation(
        self,
        operation_id: int,
        error: str,
        *,
        retry_delay_seconds: float = 30,
        max_attempts: int = 5,
    ) -> Dict[str, Any]:
        row = self._fetchone("SELECT * FROM pending_operations WHERE id = ?", (int(operation_id),))
        if not row:
            raise KeyError(f"Unknown durable operation: {operation_id}")
        attempts = int(row.get("attempts") or 0)
        attempt_limit = max(1, int(max_attempts))
        failed = attempts >= attempt_limit
        now = now_ts()
        base_delay = max(0.0, float(retry_delay_seconds))
        retry_delay = min(3600.0, base_delay * (2 ** max(0, attempts - 1)))
        self._execute(
            """UPDATE pending_operations SET status=?, available_at=?, claimed_at=0,
               error=?, updated_at=? WHERE id=?""",
            (
                "failed" if failed else "pending",
                0.0 if failed else now + retry_delay,
                normalize_whitespace(error)[:1000],
                now,
                int(operation_id),
            ),
        )
        return self._fetchone("SELECT * FROM pending_operations WHERE id = ?", (int(operation_id),)) or {}

    def pending_operation_count(self) -> int:
        row = (
            self._fetchone("SELECT COUNT(*) AS count FROM pending_operations WHERE status IN ('pending','running')")
            or {}
        )
        return int(row.get("count") or 0)

    def failed_operation_count(self) -> int:
        row = self._fetchone("SELECT COUNT(*) AS count FROM pending_operations WHERE status='failed'") or {}
        return int(row.get("count") or 0)

    def list_failed_operations(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        return self._fetchall(
            """SELECT id, operation_type, status, attempts, available_at,
                      claimed_at, error, created_at, updated_at
               FROM pending_operations WHERE status='failed'
               ORDER BY updated_at DESC, id DESC LIMIT ?""",
            (max(1, min(int(limit), 1000)),),
        )

    @_transactional
    def retry_failed_operations(self, *, limit: int = 100) -> int:
        rows = self._fetchall(
            """SELECT id FROM pending_operations WHERE status='failed'
               ORDER BY updated_at ASC, id ASC LIMIT ?""",
            (max(1, min(int(limit), 1000)),),
        )
        if not rows:
            return 0
        placeholders = ",".join("?" for _ in rows)
        ids = [int(row["id"]) for row in rows]
        return int(
            self._execute(
                f"""UPDATE pending_operations
                    SET status='pending', attempts=0, available_at=0, claimed_at=0,
                        error='', updated_at=?
                    WHERE status='failed' AND id IN ({placeholders})""",
                (now_ts(), *ids),
            ).rowcount
            or 0
        )

    @_transactional
    def acquire_lease(self, lease_name: str, owner_id: str, *, ttl_seconds: float = 300) -> bool:
        now = now_ts()
        clean_name = slugify(lease_name)
        clean_owner = normalize_whitespace(owner_id)
        current = self._fetchone("SELECT * FROM maintenance_leases WHERE lease_name = ?", (clean_name,))
        if current and float(current.get("expires_at") or 0) > now and str(current.get("owner_id")) != clean_owner:
            return False
        self._execute(
            """INSERT INTO maintenance_leases(lease_name, owner_id, expires_at, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(lease_name) DO UPDATE SET owner_id=excluded.owner_id,
               expires_at=excluded.expires_at, updated_at=excluded.updated_at""",
            (clean_name, clean_owner, now + max(1.0, float(ttl_seconds)), now),
        )
        return True

    @_transactional
    def release_lease(self, lease_name: str, owner_id: str) -> bool:
        cur = self._execute(
            "DELETE FROM maintenance_leases WHERE lease_name = ? AND owner_id = ?",
            (slugify(lease_name), normalize_whitespace(owner_id)),
        )
        return int(cur.rowcount or 0) > 0

    # ------------------------------------------------------------------
    # Repair, retention, backups, and portable export

    def database_size_bytes(self) -> int:
        return sum(
            path.stat().st_size
            for path in (Path(self.db_path), Path(self.db_path + "-wal"), Path(self.db_path + "-shm"))
            if path.exists()
        )

    def logical_database_size_bytes(self) -> int:
        """Return live SQLite pages, excluding reusable free pages and WAL timing noise."""
        with self._lock:
            page_count = int(self._conn.execute("PRAGMA page_count").fetchone()[0])
            free_pages = int(self._conn.execute("PRAGMA freelist_count").fetchone()[0])
            page_size = int(self._conn.execute("PRAGMA page_size").fetchone()[0])
        return max(0, page_count - free_pages) * max(0, page_size)

    @classmethod
    def _missing_reference_clause(cls, alias: str, kind_column: str, id_column: str) -> str:
        known_kinds = ", ".join(f"'{kind}'" for kind in cls._REFERENCE_TABLES)
        missing_known = " OR ".join(
            f"({alias}.{kind_column}='{kind}' AND NOT EXISTS("
            f"SELECT 1 FROM {table} referenced WHERE CAST(referenced.{primary_key} AS TEXT)={alias}.{id_column}))"
            for kind, (table, primary_key) in cls._REFERENCE_TABLES.items()
        )
        return f"({alias}.{kind_column} NOT IN ({known_kinds})) OR {missing_known}"

    def _dangling_reference_counts(self) -> Dict[str, int]:
        link_clause = " OR ".join(
            (
                self._missing_reference_clause("l", "source_kind", "source_id"),
                self._missing_reference_clause("l", "target_kind", "target_id"),
            )
        )
        association_clause = " OR ".join(
            (
                self._missing_reference_clause("a", "left_kind", "left_id"),
                self._missing_reference_clause("a", "right_kind", "right_id"),
            )
        )
        return {
            "links": int(
                (self._fetchone(f"SELECT COUNT(*) AS count FROM memory_links l WHERE {link_clause}") or {}).get("count")
                or 0
            ),
            "associations": int(
                (
                    self._fetchone(f"SELECT COUNT(*) AS count FROM memory_associations a WHERE {association_clause}")
                    or {}
                ).get("count")
                or 0
            ),
            "topic_membership": int(
                (
                    self._fetchone(
                        """SELECT COUNT(*) AS count FROM topic_membership membership
                           WHERE NOT EXISTS(SELECT 1 FROM topics WHERE topics.id=membership.topic_id)
                              OR NOT EXISTS(SELECT 1 FROM facts WHERE facts.id=membership.fact_id)"""
                    )
                    or {}
                ).get("count")
                or 0
            ),
            "trace_sources": int(
                (
                    self._fetchone(
                        """SELECT COUNT(*) AS count FROM memory_traces traces
                           WHERE traces.source_episode_id != 0
                             AND NOT EXISTS(
                                 SELECT 1 FROM episodes WHERE episodes.id=traces.source_episode_id
                             )"""
                    )
                    or {}
                ).get("count")
                or 0
            ),
            "fact_supersession": int(
                (
                    self._fetchone(
                        """SELECT COUNT(*) AS count FROM facts child
                           WHERE child.superseded_by IS NOT NULL
                             AND NOT EXISTS(SELECT 1 FROM facts parent WHERE parent.id=child.superseded_by)"""
                    )
                    or {}
                ).get("count")
                or 0
            ),
            "contradictions": int(
                (
                    self._fetchone(
                        """SELECT COUNT(*) AS count FROM contradictions contradiction
                           WHERE NOT EXISTS(
                                     SELECT 1 FROM facts WHERE facts.id=contradiction.winner_fact_id
                                 )
                              OR NOT EXISTS(
                                     SELECT 1 FROM facts WHERE facts.id=contradiction.loser_fact_id
                                 )"""
                    )
                    or {}
                ).get("count")
                or 0
            ),
        }

    def _delete_dangling_references(self) -> Dict[str, int]:
        link_clause = " OR ".join(
            (
                self._missing_reference_clause("memory_links", "source_kind", "source_id"),
                self._missing_reference_clause("memory_links", "target_kind", "target_id"),
            )
        )
        association_clause = " OR ".join(
            (
                self._missing_reference_clause("memory_associations", "left_kind", "left_id"),
                self._missing_reference_clause("memory_associations", "right_kind", "right_id"),
            )
        )
        removed = {
            "links": int(self._execute(f"DELETE FROM memory_links WHERE {link_clause}").rowcount or 0),
            "associations": int(
                self._execute(f"DELETE FROM memory_associations WHERE {association_clause}").rowcount or 0
            ),
            "topic_membership": int(
                self._execute(
                    """DELETE FROM topic_membership
                       WHERE NOT EXISTS(SELECT 1 FROM topics WHERE topics.id=topic_membership.topic_id)
                          OR NOT EXISTS(SELECT 1 FROM facts WHERE facts.id=topic_membership.fact_id)"""
                ).rowcount
                or 0
            ),
            "trace_sources": int(
                self._execute(
                    """UPDATE memory_traces SET source_episode_id=0
                       WHERE source_episode_id != 0
                         AND NOT EXISTS(
                             SELECT 1 FROM episodes WHERE episodes.id=memory_traces.source_episode_id
                         )"""
                ).rowcount
                or 0
            ),
            "fact_supersession": int(
                self._execute(
                    """UPDATE facts SET superseded_by=NULL
                       WHERE superseded_by IS NOT NULL
                         AND NOT EXISTS(SELECT 1 FROM facts parent WHERE parent.id=facts.superseded_by)"""
                ).rowcount
                or 0
            ),
            "contradictions": int(
                self._execute(
                    """DELETE FROM contradictions
                       WHERE NOT EXISTS(SELECT 1 FROM facts WHERE facts.id=contradictions.winner_fact_id)
                          OR NOT EXISTS(SELECT 1 FROM facts WHERE facts.id=contradictions.loser_fact_id)"""
                ).rowcount
                or 0
            ),
        }
        return removed

    @_transactional
    def maintain(
        self,
        *,
        episode_retention_hours: float = 168,
        trace_retention_days: float = 30,
        history_retention_days: float = 180,
        sensitive_retention_days: float = 30,
        max_database_mb: float = 512,
    ) -> Dict[str, Any]:
        now = now_ts()
        cutoffs = {
            "episodes": now - max(1.0, float(episode_retention_hours)) * 3600,
            "traces": now - max(1.0, float(trace_retention_days)) * 86400,
            "history": now - max(1.0, float(history_retention_days)) * 86400,
            "sensitive": now - max(1.0, float(sensitive_retention_days)) * 86400,
        }
        counts: Dict[str, int] = {}
        for name, sql, params in (
            ("working_memory", "DELETE FROM working_memory WHERE expires_at > 0 AND expires_at <= ?", (now,)),
            ("episodes", "DELETE FROM episodes WHERE created_at < ?", (cutoffs["episodes"],)),
            ("traces", "DELETE FROM memory_traces WHERE active=0 AND updated_at < ?", (cutoffs["traces"],)),
            ("history", "DELETE FROM memory_history WHERE created_at < ?", (cutoffs["history"],)),
            (
                "inactive_facts",
                "DELETE FROM facts WHERE active=0 AND pinned=0 AND updated_at < ?",
                (cutoffs["history"],),
            ),
            (
                "inactive_journals",
                "DELETE FROM memory_journals WHERE active=0 AND updated_at < ?",
                (cutoffs["history"],),
            ),
            (
                "inactive_summaries",
                "DELETE FROM memory_summaries WHERE active=0 AND updated_at < ?",
                (cutoffs["history"],),
            ),
            (
                "inactive_preferences",
                "DELETE FROM memory_preferences WHERE active=0 AND updated_at < ?",
                (cutoffs["history"],),
            ),
            (
                "inactive_policies",
                "DELETE FROM memory_policies WHERE active=0 AND updated_at < ?",
                (cutoffs["history"],),
            ),
            (
                "inactive_procedures",
                "DELETE FROM memory_procedures WHERE active=0 AND updated_at < ?",
                (cutoffs["history"],),
            ),
            (
                "inactive_events",
                "DELETE FROM autobiographical_events WHERE active=0 AND updated_at < ?",
                (cutoffs["history"],),
            ),
            (
                "sensitive_facts",
                "DELETE FROM facts WHERE pinned=0 AND sensitivity != 'normal' AND active=0 AND updated_at < ?",
                (cutoffs["sensitive"],),
            ),
            (
                "sensitive_journals",
                "DELETE FROM memory_journals WHERE sensitivity != 'normal' AND active=0 AND updated_at < ?",
                (cutoffs["sensitive"],),
            ),
            (
                "sensitive_traces",
                "DELETE FROM memory_traces WHERE sensitivity != 'normal' AND active=0 AND updated_at < ?",
                (cutoffs["sensitive"],),
            ),
            (
                "sensitive_summaries",
                "DELETE FROM memory_summaries WHERE sensitivity != 'normal' AND active=0 AND updated_at < ?",
                (cutoffs["sensitive"],),
            ),
            (
                "sensitive_preferences",
                "DELETE FROM memory_preferences WHERE sensitivity != 'normal' AND active=0 AND updated_at < ?",
                (cutoffs["sensitive"],),
            ),
            (
                "sensitive_policies",
                "DELETE FROM memory_policies WHERE sensitivity != 'normal' AND active=0 AND updated_at < ?",
                (cutoffs["sensitive"],),
            ),
            (
                "sensitive_events",
                "DELETE FROM autobiographical_events WHERE sensitivity != 'normal' AND active=0 AND updated_at < ?",
                (cutoffs["sensitive"],),
            ),
            (
                "sensitive_history",
                """DELETE FROM memory_history WHERE created_at < ?
                   AND payload_json LIKE '%\"sensitivity\": %'
                   AND payload_json NOT LIKE '%\"sensitivity\": \"normal\"%'""",
                (cutoffs["sensitive"],),
            ),
            (
                "approvals",
                """DELETE FROM memory_approvals
                   WHERE (status='pending' AND created_at < ?)
                      OR (status!='pending' AND resolved_at < ?)""",
                (cutoffs["sensitive"], cutoffs["sensitive"]),
            ),
            (
                "sensitive_working",
                "DELETE FROM working_memory WHERE sensitivity != 'normal' AND updated_at < ?",
                (cutoffs["sensitive"],),
            ),
            (
                "sensitive_procedures",
                "DELETE FROM memory_procedures WHERE sensitivity != 'normal' AND active=0 AND updated_at < ?",
                (cutoffs["sensitive"],),
            ),
            (
                "sensitive_intentions",
                "DELETE FROM prospective_memories WHERE sensitivity != 'normal' AND status!='pending' AND updated_at < ?",
                (cutoffs["sensitive"],),
            ),
            (
                "completed_intentions",
                "DELETE FROM prospective_memories WHERE status!='pending' AND updated_at < ?",
                (cutoffs["history"],),
            ),
            (
                "contradictions",
                """DELETE FROM contradictions
                   WHERE created_at < ?
                      OR NOT EXISTS(SELECT 1 FROM facts WHERE facts.id=contradictions.winner_fact_id)
                      OR NOT EXISTS(SELECT 1 FROM facts WHERE facts.id=contradictions.loser_fact_id)""",
                (cutoffs["history"],),
            ),
            (
                "consolidation_runs",
                "DELETE FROM consolidation_runs WHERE finished_at < ?",
                (cutoffs["history"],),
            ),
            (
                "failed_operations",
                "DELETE FROM pending_operations WHERE status='failed' AND updated_at < ?",
                (cutoffs["history"],),
            ),
        ):
            counts[name] = int(self._execute(sql, params).rowcount or 0)
        self._execute(
            "DELETE FROM episodes_fts WHERE episode_id NOT IN (SELECT id FROM episodes)"
        ) if self._fts_enabled else None
        self._execute(
            "DELETE FROM memory_traces_fts WHERE trace_id NOT IN (SELECT id FROM memory_traces)"
        ) if self._fts_enabled else None
        max_bytes = max(16.0, float(max_database_mb)) * 1024 * 1024
        budget_pruned = 0
        logical_size = self.logical_database_size_bytes()
        if logical_size > max_bytes:
            for table, time_col in (
                ("memory_history", "created_at"),
                ("memory_traces", "updated_at"),
                ("episodes", "created_at"),
            ):
                while logical_size > max_bytes:
                    cur = self._execute(
                        f"DELETE FROM {table} WHERE id IN (SELECT id FROM {table} ORDER BY {time_col} ASC LIMIT 500)"
                    )
                    removed = int(cur.rowcount or 0)
                    budget_pruned += removed
                    if not removed:
                        break
                    if self._fts_enabled and table == "memory_traces":
                        self._execute(
                            "DELETE FROM memory_traces_fts WHERE trace_id NOT IN (SELECT id FROM memory_traces)"
                        )
                    if self._fts_enabled and table == "episodes":
                        self._execute("DELETE FROM episodes_fts WHERE episode_id NOT IN (SELECT id FROM episodes)")
                    logical_size = self.logical_database_size_bytes()
                if logical_size <= max_bytes:
                    break
        if self._fts_enabled:
            for fts_table, id_column, source_table in (
                ("facts_fts", "fact_id", "facts"),
                ("topics_fts", "topic_id", "topics"),
                ("episodes_fts", "episode_id", "episodes"),
                ("memory_summaries_fts", "summary_id", "memory_summaries"),
                ("memory_journals_fts", "journal_id", "memory_journals"),
                ("memory_preferences_fts", "preference_id", "memory_preferences"),
                ("memory_policies_fts", "policy_id", "memory_policies"),
                ("memory_traces_fts", "trace_id", "memory_traces"),
            ):
                self._execute(f"DELETE FROM {fts_table} WHERE {id_column} NOT IN (SELECT id FROM {source_table})")
        dangling_removed = self._delete_dangling_references()
        counts.update({f"dangling_{key}": value for key, value in dangling_removed.items()})
        self._execute("PRAGMA incremental_vacuum(200)")
        counts["budget_pruned"] = budget_pruned
        counts["size_bytes"] = self.database_size_bytes()
        counts["logical_size_bytes"] = self.logical_database_size_bytes()
        counts["over_budget"] = int(counts["logical_size_bytes"] > max_bytes)
        return counts

    def checkpoint(self, *, truncate: bool = False) -> Dict[str, Any]:
        mode = "TRUNCATE" if truncate else "PASSIVE"
        with self._lock:
            row = self._conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
        return {"busy": int(row[0]), "log_frames": int(row[1]), "checkpointed_frames": int(row[2])} if row else {}

    def doctor(self, *, repair: bool = False) -> Dict[str, Any]:
        integrity_rows = self._fetchall("PRAGMA integrity_check")
        integrity_values = [str(next(iter(row.values()), "")) for row in integrity_rows]
        source_counts = self.counts()
        fts_counts: Dict[str, int] = {}
        fts_expected: Dict[str, int] = {}
        fts_mismatches: Dict[str, Dict[str, int]] = {}
        if self._fts_enabled:
            mappings = {
                "facts_fts": "SELECT COUNT(*) AS count FROM facts WHERE active=1",
                "topics_fts": "SELECT COUNT(*) AS count FROM topics",
                "episodes_fts": "SELECT COUNT(*) AS count FROM episodes",
                "memory_summaries_fts": "SELECT COUNT(*) AS count FROM memory_summaries WHERE active=1",
                "memory_journals_fts": "SELECT COUNT(*) AS count FROM memory_journals WHERE active=1",
                "memory_preferences_fts": "SELECT COUNT(*) AS count FROM memory_preferences WHERE active=1",
                "memory_policies_fts": "SELECT COUNT(*) AS count FROM memory_policies WHERE active=1",
                "memory_traces_fts": "SELECT COUNT(*) AS count FROM memory_traces WHERE active=1",
            }
            for table, expected_sql in mappings.items():
                fts_counts[table] = int(
                    (self._fetchone(f"SELECT COUNT(*) AS count FROM {table}") or {}).get("count") or 0
                )
                fts_expected[table] = int((self._fetchone(expected_sql) or {}).get("count") or 0)
                if fts_counts[table] != fts_expected[table]:
                    fts_mismatches[table] = {
                        "indexed": fts_counts[table],
                        "expected": fts_expected[table],
                    }
        dangling = self._dangling_reference_counts()
        repaired: Dict[str, Any] = {}
        if repair:
            if self._fts_enabled:
                self.set_state("fts_schema_version", "")
                self._rebuild_fts_if_needed()
                repaired["fts"] = True
            repaired.update({f"dangling_{key}": value for key, value in self._delete_dangling_references().items()})
            repaired["checkpoint"] = self.checkpoint(truncate=False)
            verified = self.doctor(repair=False)
            verified["repaired"] = repaired
            return verified
        failed_operations = self.failed_operation_count()
        return {
            "ok": (
                integrity_values == ["ok"]
                and not fts_mismatches
                and not any(dangling.values())
                and failed_operations == 0
            ),
            "integrity": integrity_values,
            "fts_enabled": self._fts_enabled,
            "fts_counts": fts_counts,
            "fts_expected": fts_expected,
            "fts_mismatches": fts_mismatches,
            "source_counts": source_counts,
            "dangling_links": dangling["links"],
            "dangling_associations": dangling["associations"],
            "dangling_references": dangling,
            "pending_operations": self.pending_operation_count(),
            "failed_operations": failed_operations,
            "failed_operation_details": self.list_failed_operations(limit=10),
            "database_size_bytes": self.database_size_bytes(),
            "logical_database_size_bytes": self.logical_database_size_bytes(),
            "repaired": repaired,
        }

    def backup_to(self, destination: str | Path) -> str:
        target = Path(destination).expanduser().resolve()
        if target == Path(self.db_path).expanduser().resolve():
            raise ValueError("Backup destination must differ from the active database")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        destination_connection = None
        moved_sidecars: List[tuple[Path, Path]] = []
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".db", delete=False) as handle:
                temporary = Path(handle.name)
            with self._lock:
                destination_connection = self._dbapi.connect(str(temporary))
                if self._encryption_key:
                    escaped_key = self._encryption_key.replace("'", "''")
                    destination_connection.execute(f"PRAGMA key = '{escaped_key}'")
                self._conn.backup(destination_connection)
                destination_connection.commit()
                integrity = destination_connection.execute("PRAGMA integrity_check").fetchone()
                if not integrity or str(integrity[0]) != "ok":
                    raise RuntimeError(f"Backup failed integrity_check: {integrity}")
                destination_connection.close()
                destination_connection = None

            # Replacing a previously opened backup must not leave its old WAL
            # or SHM next to the new main file. Move them out of the way first
            # so a failed replace can still restore the original sidecars.
            try:
                for suffix in ("-wal", "-shm"):
                    sidecar = Path(str(target) + suffix)
                    if sidecar.exists():
                        held = Path(str(temporary) + suffix + ".old")
                        os.replace(sidecar, held)
                        moved_sidecars.append((sidecar, held))
            except Exception:
                for sidecar, held in reversed(moved_sidecars):
                    if held.exists():
                        os.replace(held, sidecar)
                moved_sidecars.clear()
                raise
            try:
                os.replace(temporary, target)
                temporary = None
            except Exception:
                for sidecar, held in reversed(moved_sidecars):
                    if held.exists():
                        os.replace(held, sidecar)
                moved_sidecars.clear()
                raise
            for _, held in moved_sidecars:
                if held.exists():
                    held.unlink()
            moved_sidecars.clear()
        finally:
            if destination_connection is not None:
                destination_connection.close()
            if temporary is not None and temporary.exists():
                temporary.unlink()
            for _, held in moved_sidecars:
                if held.exists():
                    held.unlink()
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
        return str(target)

    def export_data(self, *, redact_sensitive: bool = True) -> Dict[str, Any]:
        tables = (
            "facts",
            "belief_evidence",
            "topics",
            "topic_membership",
            "memory_sessions",
            "memory_traces",
            "memory_journals",
            "memory_summaries",
            "memory_preferences",
            "memory_policies",
            "contradictions",
            "memory_history",
            "memory_links",
            "working_memory",
            "memory_procedures",
            "prospective_memories",
            "autobiographical_events",
            "memory_associations",
            "schema_migrations",
        )
        result: Dict[str, Any] = {"format": "hermes-consolidating-memory", "version": 2, "exported_at": now_ts()}
        for table in tables:
            rows = self._fetchall(f"SELECT * FROM {table}")
            if redact_sensitive:
                rows = [row for row in rows if not _looks_sensitive_for_export(row)]
            if redact_sensitive and table == "belief_evidence":
                allowed = {int(row["id"]) for row in result.get("facts", [])}
                rows = [row for row in rows if int(row.get("fact_id") or 0) in allowed]
            result[table] = rows
        if redact_sensitive:
            allowed_ids = {
                "fact": {str(row["id"]) for row in result["facts"]},
                "topic": {str(row["id"]) for row in result["topics"]},
                "episode": set(),  # Raw episode bodies are deliberately not portable.
                "session": {str(row["session_id"]) for row in result["memory_sessions"]},
                "trace": {str(row["id"]) for row in result["memory_traces"]},
                "journal": {str(row["id"]) for row in result["memory_journals"]},
                "summary": {str(row["id"]) for row in result["memory_summaries"]},
                "preference": {str(row["id"]) for row in result["memory_preferences"]},
                "policy": {str(row["id"]) for row in result["memory_policies"]},
                "autobiographical_event": {str(row["id"]) for row in result["autobiographical_events"]},
                "working": {str(row["id"]) for row in result["working_memory"]},
                "procedure": {str(row["id"]) for row in result["memory_procedures"]},
                "intention": {str(row["id"]) for row in result["prospective_memories"]},
            }

            def reference_allowed(kind: Any, entity_id: Any) -> bool:
                clean_kind = normalize_text(str(kind or ""))
                return clean_kind not in allowed_ids or str(entity_id) in allowed_ids[clean_kind]

            result["topic_membership"] = [
                row
                for row in result["topic_membership"]
                if str(row.get("fact_id")) in allowed_ids["fact"] and str(row.get("topic_id")) in allowed_ids["topic"]
            ]
            result["contradictions"] = [
                row
                for row in result["contradictions"]
                if str(row.get("winner_fact_id")) in allowed_ids["fact"]
                and str(row.get("loser_fact_id")) in allowed_ids["fact"]
            ]
            result["memory_links"] = [
                row
                for row in result["memory_links"]
                if reference_allowed(row.get("source_kind"), row.get("source_id"))
                and reference_allowed(row.get("target_kind"), row.get("target_id"))
            ]
            result["memory_associations"] = [
                row
                for row in result["memory_associations"]
                if reference_allowed(row.get("left_kind"), row.get("left_id"))
                and reference_allowed(row.get("right_kind"), row.get("right_id"))
            ]
            result["memory_history"] = [
                row
                for row in result["memory_history"]
                if reference_allowed(row.get("entity_kind"), row.get("entity_id"))
            ]
        return result

    @_transactional
    def import_data(self, data: Dict[str, Any], *, source: str = "json_import") -> Dict[str, int]:
        if str(data.get("format") or "") != "hermes-consolidating-memory":
            raise ValueError("Unsupported memory export format.")
        if int(data.get("version") or 0) > 2:
            raise ValueError("This memory export was created by a newer unsupported format version.")
        history_start_id = int(
            (self._fetchone("SELECT COALESCE(MAX(id), 0) AS id FROM memory_history") or {}).get("id") or 0
        )
        contradiction_start_id = int(
            (self._fetchone("SELECT COALESCE(MAX(id), 0) AS id FROM contradictions") or {}).get("id") or 0
        )
        counts = {
            "sessions": 0,
            "facts": 0,
            "evidence": 0,
            "traces": 0,
            "journals": 0,
            "summaries": 0,
            "preferences": 0,
            "policies": 0,
            "working": 0,
            "procedures": 0,
            "intentions": 0,
            "events": 0,
            "links": 0,
            "associations": 0,
            "contradictions": 0,
            "history": 0,
        }
        id_maps: Dict[str, Dict[str, str]] = defaultdict(dict)
        for row in data.get("memory_sessions") or []:
            if not isinstance(row, dict) or not normalize_whitespace(str(row.get("session_id") or "")):
                continue
            session = self.ensure_memory_session(
                str(row["session_id"]),
                label=str(row.get("label") or ""),
                summary=str(row.get("summary") or ""),
                status=str(row.get("status") or "open"),
                sensitivity=str(row.get("sensitivity") or "normal"),
            )
            self._execute(
                """UPDATE memory_sessions
                   SET started_at=?, ended_at=?, last_activity_at=?, created_at=?, updated_at=?
                   WHERE session_id=?""",
                (
                    float(row.get("started_at") or session.get("started_at") or now_ts()),
                    float(row.get("ended_at") or 0),
                    float(row.get("last_activity_at") or session.get("last_activity_at") or now_ts()),
                    float(row.get("created_at") or session.get("created_at") or now_ts()),
                    float(row.get("updated_at") or session.get("updated_at") or now_ts()),
                    str(session["session_id"]),
                ),
            )
            id_maps["session"][str(row["session_id"])] = str(session["session_id"])
            counts["sessions"] += 1

        evidence_by_fact: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for evidence in data.get("belief_evidence") or []:
            if isinstance(evidence, dict):
                evidence_by_fact[str(evidence.get("fact_id") or "")].append(evidence)
        for row in data.get("facts") or []:
            if not isinstance(row, dict) or not normalize_whitespace(str(row.get("content") or "")):
                continue
            stored_result = self.upsert_fact(
                content=str(row["content"]),
                category=str(row.get("category") or "general"),
                topic=str(row.get("topic") or "general"),
                source=source,
                importance=int(row.get("importance") or 5),
                confidence=float(row.get("confidence") or 0.7),
                salience=float(row.get("salience") or 0.55),
                metadata=dict(row.get("metadata") or {}),
                observed_at=float(row.get("updated_at") or row.get("created_at") or now_ts()),
                source_session_id=str(row.get("source_session_id") or ""),
                valid_from=float(row.get("valid_from") or 0),
                valid_until=float(row.get("valid_until") or 0),
                temporal_kind=str(row.get("temporal_kind") or "atemporal"),
                event_at=float(row.get("event_at") or 0),
                temporal_precision=str(row.get("temporal_precision") or "unknown"),
                temporal_timezone=str(row.get("temporal_timezone") or ""),
                temporal_confidence=float(row.get("temporal_confidence") or 0),
                sensitivity=str(row.get("sensitivity") or "normal"),
                memory_class=str(row.get("memory_class") or "semantic"),
                pinned=bool(row.get("pinned")),
                history_reason=source,
            )
            stored = dict(stored_result.get("fact") or {})
            if not stored.get("id"):
                continue
            old_id = str(row.get("id") or "")
            new_id = int(stored["id"])
            id_maps["fact"][old_id] = str(new_id)
            original_evidence = evidence_by_fact.get(old_id, [])
            if original_evidence:
                self._execute("DELETE FROM belief_evidence WHERE fact_id = ?", (new_id,))
                for evidence in original_evidence:
                    self._execute(
                        """INSERT INTO belief_evidence(
                               fact_id, content, source, source_role, session_id, confidence,
                               reliability, explicit_correction, observed_at, metadata_json, created_at
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            new_id,
                            normalize_whitespace(str(evidence.get("content") or row["content"])),
                            str(evidence.get("source") or source),
                            str(evidence.get("source_role") or "unknown"),
                            str(evidence.get("session_id") or ""),
                            _clamp_float(evidence.get("confidence"), 0, 1, 0.5),
                            _clamp_float(evidence.get("reliability"), 0, 1, 0.5),
                            int(_as_bool(evidence.get("explicit_correction"))),
                            float(evidence.get("observed_at") or now_ts()),
                            json.dumps(dict(evidence.get("metadata") or {}), sort_keys=True),
                            float(evidence.get("created_at") or now_ts()),
                        ),
                    )
                    counts["evidence"] += 1
            self._execute(
                """UPDATE facts SET active=?, belief_score=?, observation_count=?, revision=?,
                       last_recalled_at=?, review_count=?, next_review_at=?, reconsolidation_until=?,
                       created_at=?, updated_at=? WHERE id=?""",
                (
                    int(_as_bool(row.get("active", 1))),
                    _clamp_float(row.get("belief_score"), 0, 1, float(stored.get("belief_score") or 0.5)),
                    max(1, int(row.get("observation_count") or len(original_evidence) or 1)),
                    max(1, int(row.get("revision") or 1)),
                    float(row.get("last_recalled_at") or 0),
                    max(0, int(row.get("review_count") or 0)),
                    float(row.get("next_review_at") or 0),
                    float(row.get("reconsolidation_until") or 0),
                    float(row.get("created_at") or now_ts()),
                    float(row.get("updated_at") or now_ts()),
                    new_id,
                ),
            )
            counts["facts"] += 1

        # Later facts can change earlier active/superseded state while the
        # normal conflict resolver runs. Restore the exported final state only
        # after every fact identifier is known.
        for row in data.get("facts") or []:
            if not isinstance(row, dict):
                continue
            new_id = id_maps["fact"].get(str(row.get("id") or ""))
            if not new_id:
                continue
            superseded_by = None
            if row.get("superseded_by") is not None:
                superseded_by = id_maps["fact"].get(str(row.get("superseded_by")))
            self._execute(
                """UPDATE facts
                   SET active=?, superseded_by=?, created_at=?, updated_at=?
                   WHERE id=?""",
                (
                    int(_as_bool(row.get("active", 1))),
                    int(superseded_by) if superseded_by else None,
                    float(row.get("created_at") or now_ts()),
                    float(row.get("updated_at") or now_ts()),
                    int(new_id),
                ),
            )

        for row in data.get("memory_traces") or []:
            if not isinstance(row, dict) or not normalize_whitespace(str(row.get("content") or "")):
                continue
            trace = self.append_trace(
                session_id=str(row.get("session_id") or "imported"),
                label=str(row.get("label") or "Imported trace"),
                content=str(row["content"]),
                trace_type=str(row.get("trace_type") or "turn"),
                metadata=dict(row.get("metadata") or {}),
                importance=int(row.get("importance") or 4),
                salience=float(row.get("salience") or 0.45),
                sensitivity=str(row.get("sensitivity") or "normal"),
            )
            id_maps["trace"][str(row.get("id") or "")] = str(trace["id"])
            if not _as_bool(row.get("active", 1)):
                self.deactivate_memory_item("trace", int(trace["id"]), reason=source, source=source)
            counts["traces"] += 1

        for row in data.get("memory_journals") or []:
            if not isinstance(row, dict) or not normalize_whitespace(str(row.get("content") or "")):
                continue
            journal = self.add_journal(
                session_id=str(row.get("session_id") or ""),
                label=str(row.get("label") or "Imported journal"),
                content=str(row["content"]),
                journal_type=str(row.get("journal_type") or "note"),
                metadata=dict(row.get("metadata") or {}),
                importance=int(row.get("importance") or 6),
                salience=float(row.get("salience") or 0.6),
                sensitivity=str(row.get("sensitivity") or "normal"),
            )
            id_maps["journal"][str(row.get("id") or "")] = str(journal["id"])
            if not _as_bool(row.get("active", 1)):
                self.deactivate_memory_item("journal", int(journal["id"]), reason=source, source=source)
            counts["journals"] += 1

        for row in data.get("memory_summaries") or []:
            if not isinstance(row, dict) or not normalize_whitespace(str(row.get("summary") or "")):
                continue
            metadata = dict(row.get("metadata") or {})
            metadata.pop("source_refs", None)
            summary = self.upsert_summary(
                session_id=str(row.get("session_id") or ""),
                label=str(row.get("label") or "Imported summary"),
                summary=str(row["summary"]),
                content=str(row.get("content") or ""),
                summary_type=str(row.get("summary_type") or "session"),
                metadata=metadata,
                importance=int(row.get("importance") or 7),
                salience=float(row.get("salience") or 0.65),
                reason=source,
                sensitivity=str(row.get("sensitivity") or "normal"),
            )
            id_maps["summary"][str(row.get("id") or "")] = str(summary["id"])
            if not _as_bool(row.get("active", 1)):
                self.deactivate_memory_item("summary", int(summary["id"]), reason=source, source=source)
            counts["summaries"] += 1

        for row in data.get("memory_preferences") or []:
            if not isinstance(row, dict):
                continue
            metadata = dict(row.get("metadata") or {})
            if row.get("source_session_id"):
                metadata["session_id"] = str(row["source_session_id"])
            preference = self.upsert_preference(
                key=str(row.get("preference_key") or ""),
                label=str(row.get("label") or "Preference"),
                value=str(row.get("value") or ""),
                content=str(row.get("content") or row.get("value") or ""),
                metadata=metadata,
                importance=int(row.get("importance") or 8),
                salience=float(row.get("salience") or 0.9),
                reason=source,
                sensitivity=str(row.get("sensitivity") or "normal"),
            )
            id_maps["preference"][str(row.get("id") or "")] = str(preference["id"])
            if not _as_bool(row.get("active", 1)):
                self.deactivate_memory_item("preference", int(preference["id"]), reason=source, source=source)
            counts["preferences"] += 1
        for row in data.get("memory_policies") or []:
            if not isinstance(row, dict):
                continue
            metadata = dict(row.get("metadata") or {})
            if row.get("source_session_id"):
                metadata["session_id"] = str(row["source_session_id"])
            policy = self.upsert_policy(
                key=str(row.get("policy_key") or ""),
                label=str(row.get("label") or "Policy"),
                content=str(row.get("content") or ""),
                metadata=metadata,
                importance=int(row.get("importance") or 9),
                salience=float(row.get("salience") or 0.95),
                reason=source,
                sensitivity=str(row.get("sensitivity") or "normal"),
            )
            id_maps["policy"][str(row.get("id") or "")] = str(policy["id"])
            if not _as_bool(row.get("active", 1)):
                self.deactivate_memory_item("policy", int(policy["id"]), reason=source, source=source)
            counts["policies"] += 1

        for row in data.get("working_memory") or []:
            if not isinstance(row, dict) or not normalize_whitespace(str(row.get("content") or "")):
                continue
            expires_at = float(row.get("expires_at") or 0)
            if expires_at and expires_at <= now_ts():
                continue
            working = self.set_working_memory(
                session_id=str(row.get("session_id") or "imported"),
                memory_key=str(row.get("memory_key") or "imported"),
                content=str(row["content"]),
                priority=int(row.get("priority") or 5),
                ttl_seconds=max(1.0, expires_at - now_ts()) if expires_at else 0,
                metadata=dict(row.get("metadata") or {}),
                capacity=1000,
                sensitivity=str(row.get("sensitivity") or "normal"),
            )
            id_maps["working"][str(row.get("id") or "")] = str(working["id"])
            self._execute(
                "UPDATE working_memory SET created_at=?, updated_at=? WHERE id=?",
                (
                    float(row.get("created_at") or working.get("created_at") or now_ts()),
                    float(row.get("updated_at") or working.get("updated_at") or now_ts()),
                    int(working["id"]),
                ),
            )
            counts["working"] += 1
        for row in data.get("memory_procedures") or []:
            if not isinstance(row, dict):
                continue
            procedure = self.upsert_procedure(
                procedure_key=str(row.get("procedure_key") or ""),
                label=str(row.get("label") or "Procedure"),
                steps=list(row.get("steps") or []),
                prerequisites=list(row.get("prerequisites") or []),
                success_criteria=str(row.get("success_criteria") or ""),
                failure_recovery=str(row.get("failure_recovery") or ""),
                confidence=float(row.get("confidence") or 0.6),
                metadata=dict(row.get("metadata") or {}),
                sensitivity=str(row.get("sensitivity") or "normal"),
            )
            id_maps["procedure"][str(row.get("id") or "")] = str(procedure["id"])
            self._execute(
                """UPDATE memory_procedures
                   SET active=?, use_count=?, success_count=?, last_used_at=?, created_at=?, updated_at=?
                   WHERE id=?""",
                (
                    int(_as_bool(row.get("active", 1))),
                    max(0, int(row.get("use_count") or 0)),
                    max(0, int(row.get("success_count") or 0)),
                    float(row.get("last_used_at") or 0),
                    float(row.get("created_at") or procedure.get("created_at") or now_ts()),
                    float(row.get("updated_at") or procedure.get("updated_at") or now_ts()),
                    int(procedure["id"]),
                ),
            )
            counts["procedures"] += 1
        for row in data.get("prospective_memories") or []:
            if not isinstance(row, dict) or not normalize_whitespace(str(row.get("intention") or "")):
                continue
            intention = self.add_intention(
                intention=str(row.get("intention") or ""),
                due_at=float(row.get("due_at") or 0),
                condition_text=str(row.get("condition_text") or ""),
                recurrence=str(row.get("recurrence") or ""),
                importance=int(row.get("importance") or 6),
                session_id=str(row.get("session_id") or ""),
                metadata=dict(row.get("metadata") or {}),
                sensitivity=str(row.get("sensitivity") or "normal"),
            )
            id_maps["intention"][str(row.get("id") or "")] = str(intention["id"])
            status = normalize_text(str(row.get("status") or "pending"))
            if status not in {"pending", "completed", "cancelled"}:
                status = "pending"
            self._execute(
                """UPDATE prospective_memories
                   SET status=?, last_triggered_at=?, created_at=?, updated_at=? WHERE id=?""",
                (
                    status,
                    float(row.get("last_triggered_at") or 0),
                    float(row.get("created_at") or intention.get("created_at") or now_ts()),
                    float(row.get("updated_at") or intention.get("updated_at") or now_ts()),
                    int(intention["id"]),
                ),
            )
            counts["intentions"] += 1
        for row in data.get("autobiographical_events") or []:
            if not isinstance(row, dict):
                continue
            event = self.upsert_autobiographical_event(
                event_key=str(row.get("event_key") or ""),
                content=str(row.get("content") or ""),
                event_at=float(row.get("event_at") or 0),
                valid_from=float(row.get("valid_from") or 0),
                valid_until=float(row.get("valid_until") or 0),
                people=list(row.get("people") or []),
                places=list(row.get("places") or []),
                importance=int(row.get("importance") or 6),
                metadata=dict(row.get("metadata") or {}),
                sensitivity=str(row.get("sensitivity") or "normal"),
            )
            id_maps["autobiographical_event"][str(row.get("id") or "")] = str(event["id"])
            if not _as_bool(row.get("active", 1)):
                self._execute("UPDATE autobiographical_events SET active=0 WHERE id=?", (int(event["id"]),))
            counts["events"] += 1
        self.rebuild_topics()

        topics_by_slug = {str(row.get("slug") or ""): row for row in self.list_topics(limit=100000)}
        for row in data.get("topics") or []:
            if not isinstance(row, dict):
                continue
            current = topics_by_slug.get(str(row.get("slug") or ""))
            if current:
                id_maps["topic"][str(row.get("id") or "")] = str(current["id"])

        def mapped_reference(kind: Any, entity_id: Any) -> str | None:
            clean_kind = normalize_text(str(kind or ""))
            raw_id = str(entity_id)
            if clean_kind == "episode":
                return None
            if clean_kind in id_maps:
                return id_maps[clean_kind].get(raw_id)
            return raw_id

        for row in data.get("memory_links") or []:
            if not isinstance(row, dict):
                continue
            source_id = mapped_reference(row.get("source_kind"), row.get("source_id"))
            target_id = mapped_reference(row.get("target_kind"), row.get("target_id"))
            if source_id is None or target_id is None:
                continue
            self.add_link(
                str(row.get("source_kind") or "memory"),
                source_id,
                str(row.get("target_kind") or "memory"),
                target_id,
                str(row.get("link_type") or "related"),
                dict(row.get("metadata") or {}),
            )
            counts["links"] += 1

        for row in data.get("memory_associations") or []:
            if not isinstance(row, dict):
                continue
            left_id = mapped_reference(row.get("left_kind"), row.get("left_id"))
            right_id = mapped_reference(row.get("right_kind"), row.get("right_id"))
            if left_id is None or right_id is None:
                continue
            self.associate(
                str(row.get("left_kind") or "fact"),
                left_id,
                str(row.get("right_kind") or "fact"),
                right_id,
                str(row.get("relation") or "associated"),
                weight=float(row.get("weight") or 0.5),
                metadata=dict(row.get("metadata") or {}),
            )
            counts["associations"] += 1

        exported_contradictions = [row for row in data.get("contradictions") or [] if isinstance(row, dict)]
        if exported_contradictions:
            self._execute("DELETE FROM contradictions WHERE id > ?", (contradiction_start_id,))
        for row in exported_contradictions:
            if not isinstance(row, dict):
                continue
            winner_id = mapped_reference("fact", row.get("winner_fact_id"))
            loser_id = mapped_reference("fact", row.get("loser_fact_id"))
            if winner_id is None or loser_id is None:
                continue
            created_at = float(row.get("created_at") or now_ts())
            existing = self._fetchone(
                """SELECT id FROM contradictions
                   WHERE subject_key=? AND winner_fact_id=? AND loser_fact_id=?
                     AND resolution=? AND created_at=?""",
                (
                    normalize_whitespace(str(row.get("subject_key") or "")),
                    int(winner_id),
                    int(loser_id),
                    normalize_whitespace(str(row.get("resolution") or "imported")),
                    created_at,
                ),
            )
            if existing:
                continue
            self._execute(
                """INSERT INTO contradictions(
                       subject_key, winner_fact_id, loser_fact_id, resolution, created_at
                   ) VALUES (?, ?, ?, ?, ?)""",
                (
                    normalize_whitespace(str(row.get("subject_key") or "")),
                    int(winner_id),
                    int(loser_id),
                    normalize_whitespace(str(row.get("resolution") or "imported")),
                    created_at,
                ),
            )
            counts["contradictions"] += 1

        exported_history = [row for row in data.get("memory_history") or [] if isinstance(row, dict)]
        if exported_history:
            # The public upsert methods record useful import actions. A v2 export
            # already carries the original audit trail, so replace only the rows
            # created by this transaction with that remapped history.
            self._execute("DELETE FROM memory_history WHERE id > ?", (history_start_id,))
            for row in exported_history:
                entity_kind = normalize_whitespace(str(row.get("entity_kind") or "memory"))
                entity_id = mapped_reference(entity_kind, row.get("entity_id"))
                if entity_id is None:
                    continue
                self._execute(
                    """INSERT INTO memory_history(
                           entity_kind, entity_id, subject_key, action, reason,
                           source, payload_json, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entity_kind,
                        entity_id,
                        normalize_whitespace(str(row.get("subject_key") or "")),
                        normalize_whitespace(str(row.get("action") or "imported")),
                        normalize_whitespace(str(row.get("reason") or "")),
                        normalize_whitespace(str(row.get("source") or source)),
                        json.dumps(dict(row.get("payload") or {}), sort_keys=True, default=str),
                        float(row.get("created_at") or now_ts()),
                    ),
                )
                counts["history"] += 1
        if self._fts_enabled:
            self.set_state("fts_schema_version", "")
            self._rebuild_fts_if_needed()
        return counts
