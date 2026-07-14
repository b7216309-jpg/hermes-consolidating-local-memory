from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import queue
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

try:
    from agent.memory_provider import MemoryProvider
except ModuleNotFoundError as exc:
    if exc.name not in {"agent", "agent.memory_provider"}:
        raise

    class MemoryProvider:  # type: ignore[override]
        pass


from .consolidation import (
    build_consolidation_plan,
    message_content_text,
    normalize_candidate_fact,
    run_consolidation,
)
from .llm_client import OpenAICompatibleEmbeddings, OpenAICompatibleLLM, env_or_blank
from .store import (
    MemoryStore,
    _as_bool,
    _looks_like_credential,
    _looks_sensitive_for_export,
    fingerprint_text,
    normalize_text,
    normalize_whitespace,
    pretty_topic,
    slugify,
)
from .wiki_export import export_compiled_wiki

logger = logging.getLogger(__name__)
__version__ = "3.2.0"


def _flag(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().casefold() in {"1", "true", "yes", "on"}


TOOL_SCHEMA = {
    "name": "consolidating_memory",
    "description": (
        "Layered local memory with session summaries, durable facts, "
        "preferences, policies, provenance, and background consolidation.\n\n"
        "Actions:\n"
        "- search: lookup across facts, topics, summaries, journals, preferences, policies, and episode buffers.\n"
        "- remember: store a durable fact or preference.\n"
        "- forget: deactivate a fact or other memory entry by id or matching text.\n"
        "- recent: inspect the latest memory objects.\n"
        "- contradictions: inspect recently resolved assumption changes.\n"
        "- status: show memory counts, retrieval backend, and consolidation state.\n"
        "- consolidate: force a consolidation pass now.\n"
        "- journal: write a narrative note.\n"
        "- distill: create or refresh a summary.\n"
        "- history: inspect append-only memory history.\n"
        "- policy: set or get a memory policy.\n"
        "- review: inspect and advance due spaced-review items.\n"
        "- decay: apply salience decay now.\n"
        "- export: write the compiled markdown wiki mirror."
        "\n- explain: show a fact's evidence, provenance, and revision history."
        "\n- working/procedure/intention/timeline: manage working, procedural, prospective, and autobiographical memory."
        "\n- approval: inspect or resolve sensitive-memory consent requests."
        "\n- associate/merge/split/pin: curate links and fact structure."
        "\n- doctor/maintain/backup/export_json: operate and repair the store."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "search",
                    "remember",
                    "forget",
                    "recent",
                    "contradictions",
                    "status",
                    "consolidate",
                    "journal",
                    "distill",
                    "history",
                    "policy",
                    "review",
                    "decay",
                    "export",
                    "explain",
                    "working",
                    "procedure",
                    "intention",
                    "timeline",
                    "approval",
                    "associate",
                    "merge",
                    "split",
                    "pin",
                    "doctor",
                    "maintain",
                    "backup",
                    "export_json",
                ],
            },
            "query": {"type": "string", "description": "Search or forget query."},
            "scope": {
                "type": "string",
                "enum": ["all", "facts", "topics", "episodes", "summaries", "journals", "preferences", "policies"],
                "description": "Search scope for action=search.",
            },
            "limit": {"type": "integer", "description": "Maximum number of results."},
            "content": {"type": "string", "description": "Content to store or update."},
            "category": {
                "type": "string",
                "enum": ["user_pref", "project", "environment", "workflow", "general"],
            },
            "topic": {"type": "string", "description": "Topic bucket for remembered content."},
            "importance": {"type": "integer", "description": "Importance score from 1 to 10."},
            "fact_id": {"type": "integer", "description": "Specific fact id to forget."},
            "memory_type": {"type": "string", "description": "fact, preference, journal, summary, or policy."},
            "session_id": {
                "type": "string",
                "description": "Session identifier for journals, distillation, and recall links.",
            },
            "subject_key": {"type": "string", "description": "Exclusive subject key or history filter."},
            "since_days": {"type": "integer", "description": "History or contradiction age filter in days."},
            "include_inactive": {"type": "boolean", "description": "Whether to include inactive memory items."},
            "key": {"type": "string", "description": "Preference or policy key."},
            "value": {"type": "string", "description": "Preference value."},
            "label": {
                "type": "string",
                "description": "Optional label for journals, summaries, preferences, or policies.",
            },
            "dry_run": {"type": "boolean", "description": "Preview a destructive or expensive action."},
            "confirm": {"type": "boolean", "description": "Confirm an exact destructive operation."},
            "approved": {"type": "boolean", "description": "Explicit consent for a sensitive memory."},
            "explicit_correction": {
                "type": "boolean",
                "description": "Mark a direct user correction as strong evidence.",
            },
            "pinned": {"type": "boolean", "description": "Protect a fact from decay and budget pruning."},
            "status": {"type": "string", "description": "Status filter or target status."},
            "due_at": {"type": "number", "description": "Unix timestamp for a prospective memory."},
            "ttl_seconds": {"type": "number", "description": "Working-memory lifetime."},
            "steps": {"type": "array", "items": {"type": "string"}},
            "prerequisites": {"type": "array", "items": {"type": "string"}},
            "ids": {"type": "array", "items": {"type": "integer"}},
            "contents": {"type": "array", "items": {"type": "string"}},
            "left_kind": {"type": "string"},
            "left_id": {"type": "string"},
            "right_kind": {"type": "string"},
            "right_id": {"type": "string"},
            "relation": {"type": "string"},
            "destination": {"type": "string", "description": "Backup or JSON export destination."},
        },
        "required": ["action"],
    },
}

PLUGIN_CONFIG_KEY = "consolidating-local-memory"
AUTO_MEMORY_BLOCK_START = "<!-- consolidating_local:auto:start -->"
AUTO_MEMORY_BLOCK_END = "<!-- consolidating_local:auto:end -->"
SUMMARY_SNAPSHOT_SUBJECTS = (
    "user:timezone",
    "environment:shell",
    "project:database",
    "project:deploy_method",
    "project:cache_backend",
    "workflow:manual_edits",
)
WORKFLOW_SNAPSHOT_SUBJECTS = (
    "environment:shell",
    "project:test_command",
    "project:deploy_method",
    "workflow:docker_sudo",
)


def _load_plugin_config() -> dict:
    try:
        from hermes_constants import get_hermes_home
    except Exception:
        return {}

    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        with open(config_path, encoding="utf-8") as handle:
            all_config = yaml.safe_load(handle) or {}
        return all_config.get("plugins", {}).get(PLUGIN_CONFIG_KEY, {}) or {}
    except Exception:
        return {}


class ConsolidatingLocalMemoryProvider(MemoryProvider):
    def __init__(self, config: dict | None = None):
        self._config = dict(config) if config is not None else _load_plugin_config()
        self._store: MemoryStore | None = None
        self._llm: OpenAICompatibleLLM | None = None
        self._embedder: OpenAICompatibleEmbeddings | None = None
        self._hermes_home = Path("~/.hermes").expanduser()
        self._retrieval_backend = "fts"
        self._session_id = ""
        self._task_queue: queue.Queue[tuple[str, Dict[str, Any]] | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._prefetch_cache: Dict[str, Dict[str, Any]] = {}
        self._prefetch_lock = threading.Lock()
        self._consolidation_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._consolidation_requested = False
        self._accepting_tasks = False
        self._draining = False
        self._write_enabled = True
        self._last_scan_at = 0.0
        self._scope_id = "legacy"
        self._owner_id = f"{os.getpid()}-{uuid.uuid4().hex}"
        self._queue_metrics = {"enqueued": 0, "dropped_prefetch": 0, "spooled": 0, "failed": 0}

    @property
    def name(self) -> str:
        return "consolidating_local"

    def is_available(self) -> bool:
        if self._cfg_bool("database_encryption", False):
            if not env_or_blank("CONSOLIDATING_MEMORY_DB_KEY"):
                return False
            try:
                from sqlcipher3 import dbapi2 as _sqlcipher  # noqa: F401
            except ImportError:
                return False
        return True

    def get_config_schema(self) -> List[Dict[str, Any]]:
        # The provider is usable without configuration. Keeping Hermes' setup
        # wizard empty avoids prompting for every optional tuning knob.
        return []

    def get_advanced_config_schema(self) -> List[Dict[str, Any]]:
        """Machine-readable reference for optional config.yaml settings."""
        return [
            {
                "key": "db_path",
                "description": "SQLite database path",
                "default": "$HERMES_HOME/consolidating_memory.db",
            },
            {
                "key": "memory_scope",
                "description": "Isolation boundary for gateway users and agents",
                "default": "user",
                "choices": ["user", "agent", "global"],
            },
            {
                "key": "sensitive_memory",
                "description": "Admission policy for health, financial, identity, location, or credential memories",
                "default": "ask",
                "choices": ["deny", "ask", "allow"],
            },
            {
                "key": "allow_credential_memory",
                "description": "Permit credentials to follow sensitive_memory instead of always denying them",
                "default": "false",
            },
            {
                "key": "allow_sensitive_model_processing",
                "description": "Permit configured LLM and embedding endpoints to receive sensitive memory text",
                "default": "false",
            },
            {
                "key": "conflict_policy",
                "description": "Choose evidence-weighted or last-write-wins contradiction resolution",
                "default": "evidence",
                "choices": ["evidence", "newest"],
            },
            {
                "key": "never_remember_categories",
                "description": "Comma-separated categories rejected before storage",
                "default": "",
            },
            {
                "key": "queue_max_size",
                "description": "Maximum in-memory background tasks before durable spooling",
                "default": "256",
            },
            {
                "key": "queue_max_attempts",
                "description": "Attempts before a poison durable task moves to the recoverable dead-letter queue",
                "default": "5",
            },
            {
                "key": "shutdown_timeout_seconds",
                "description": "Maximum graceful worker-drain wait before remaining queued work stays durable",
                "default": "10",
            },
            {
                "key": "max_database_mb",
                "description": "Soft database size budget used by maintenance",
                "default": "512",
            },
            {
                "key": "trace_retention_days",
                "description": "Retention for inactive turn traces",
                "default": "30",
            },
            {
                "key": "history_retention_days",
                "description": "Retention for append-only operational history",
                "default": "180",
            },
            {
                "key": "sensitive_retention_days",
                "description": "Retention for inactive sensitive facts",
                "default": "30",
            },
            {
                "key": "consolidation_max_batches",
                "description": "Maximum immediate passes while a backlog remains",
                "default": "4",
            },
            {
                "key": "consolidation_batch_size",
                "description": "Episode buffers processed per atomic consolidation pass",
                "default": "250",
            },
            {
                "key": "working_memory_capacity",
                "description": "Maximum working-memory slots per session",
                "default": "12",
            },
            {
                "key": "database_encryption",
                "description": "Require SQLCipher using CONSOLIDATING_MEMORY_DB_KEY",
                "default": "false",
            },
            {
                "key": "export_redact_sensitive",
                "description": "Omit sensitive memories from portable and wiki exports",
                "default": "true",
            },
            {
                "key": "min_hours",
                "description": "Minimum hours between background consolidations",
                "default": "24",
            },
            {
                "key": "min_sessions",
                "description": "Minimum distinct sessions since the last consolidation",
                "default": "5",
            },
            {
                "key": "scan_cooldown_seconds",
                "description": "How often the provider re-checks the consolidation gate during active use",
                "default": "600",
            },
            {
                "key": "prefetch_limit",
                "description": "How many memory lines to inject into context",
                "default": "8",
            },
            {
                "key": "max_topic_facts",
                "description": "How many top facts to pack into each topic summary",
                "default": "5",
            },
            {
                "key": "topic_summary_chars",
                "description": "Maximum characters per topic summary",
                "default": "650",
            },
            {
                "key": "session_summary_chars",
                "description": "Maximum characters per session or handoff summary",
                "default": "900",
            },
            {
                "key": "prune_after_days",
                "description": "Age threshold for pruning low-value extracted facts",
                "default": "90",
            },
            {
                "key": "episode_body_retention_hours",
                "description": "How long raw episode buffers are kept after consolidation",
                "default": "24",
            },
            {
                "key": "decay_half_life_days",
                "description": "Default salience half life in days",
                "default": "90",
            },
            {
                "key": "reconsolidation_window_hours",
                "description": "How long a recalled memory stays open to reconsolidation updates",
                "default": "6",
            },
            {
                "key": "review_intervals_days",
                "description": "Comma-separated spaced review intervals in days",
                "default": "1,3,7,14,30",
            },
            {
                "key": "decay_min_salience",
                "description": "Minimum salience before low-priority items are deactivated",
                "default": "0.15",
            },
            {
                "key": "builtin_snapshot_sync_enabled",
                "description": "Keep Hermes bounded USER.md and MEMORY.md aligned with the plugin's current-state winners",
                "default": "false",
            },
            {
                "key": "builtin_memory_dir",
                "description": "Directory containing Hermes USER.md and MEMORY.md files",
                "default": "$HERMES_HOME/memories",
            },
            {
                "key": "builtin_snapshot_user_chars",
                "description": "Character budget for USER.md snapshot updates",
                "default": "1375",
            },
            {
                "key": "builtin_snapshot_memory_chars",
                "description": "Character budget for MEMORY.md snapshot updates",
                "default": "2200",
            },
            {
                "key": "wiki_export_enabled",
                "description": "Enable compiled markdown wiki export",
                "default": "false",
            },
            {
                "key": "wiki_export_dir",
                "description": "Directory for compiled wiki export",
                "default": "$HERMES_HOME/consolidating_memory_wiki",
            },
            {
                "key": "wiki_export_on_consolidate",
                "description": "Refresh the wiki mirror after successful consolidation",
                "default": "true",
            },
            {
                "key": "wiki_export_session_limit",
                "description": "Maximum number of session pages to export",
                "default": "50",
            },
            {
                "key": "wiki_export_topic_limit",
                "description": "Maximum number of topic pages to export",
                "default": "100",
            },
            {
                "key": "llm_model",
                "description": "OpenAI-compatible model for automatic fact extraction (requires llm_base_url)",
                "default": "",
            },
            {
                "key": "llm_base_url",
                "description": "Opt-in OpenAI-compatible base URL (requires llm_model)",
                "default": "",
            },
            {
                "key": "llm_disable_thinking",
                "description": "Ask compatible extraction endpoints to disable reasoning and require visible output",
                "default": "false",
            },
            {
                "key": "llm_timeout_seconds",
                "description": "Timeout for local LLM extraction calls",
                "default": "45",
            },
            {
                "key": "llm_failure_cooldown_seconds",
                "description": "Base cooldown after three consecutive model endpoint failures",
                "default": "120",
            },
            {
                "key": "llm_max_input_chars",
                "description": "Maximum input chars sent to local extractor prompts",
                "default": "4000",
            },
            {
                "key": "retrieval_backend",
                "description": "Recall backend",
                "default": "fts",
                "choices": ["fts", "hybrid"],
            },
            {
                "key": "embedding_model",
                "description": "Opt-in OpenAI-compatible embedding model name",
                "default": "",
            },
            {
                "key": "embedding_base_url",
                "description": "Opt-in embedding base URL (requires embedding_model)",
                "default": "",
            },
            {
                "key": "embedding_timeout_seconds",
                "description": "Timeout for embedding calls",
                "default": "20",
            },
            {
                "key": "embedding_candidate_limit",
                "description": "How many text candidates hybrid retrieval reranks",
                "default": "16",
            },
            {
                "key": "prefetch_cache_ttl_seconds",
                "description": "Maximum age of a synchronous recall cache entry",
                "default": "120",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        config_path = Path(hermes_home) / "config.yaml"
        temp_path: Path | None = None
        try:
            import yaml

            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8") as handle:
                    existing = yaml.safe_load(handle) or {}
            if not isinstance(existing, dict):
                existing = {}
            if not isinstance(existing.get("plugins"), dict):
                existing["plugins"] = {}
            existing["plugins"][PLUGIN_CONFIG_KEY] = values
            config_path.parent.mkdir(parents=True, exist_ok=True)
            rendered = yaml.safe_dump(existing, default_flow_style=False, sort_keys=False, allow_unicode=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=config_path.parent,
                prefix=".config.yaml.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)
            os.replace(temp_path, config_path)
        except Exception as exc:
            logger.warning("Failed to save config for %s: %s", self.name, exc)
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()

    def initialize(self, session_id: str, **kwargs) -> None:
        if self._store is not None or (self._worker and self._worker.is_alive()):
            self.shutdown()
        if self._worker and self._worker.is_alive():
            raise RuntimeError("Previous memory worker did not stop; refusing unsafe reinitialization")
        hermes_home = Path(str(kwargs.get("hermes_home") or Path("~/.hermes").expanduser()))
        self._hermes_home = hermes_home
        agent_context = str(kwargs.get("agent_context") or "primary").strip().lower()
        platform = str(kwargs.get("platform") or "cli").strip().lower()
        self._write_enabled = agent_context == "primary" and platform != "cron"
        db_path = str(self._config.get("db_path", "$HERMES_HOME/consolidating_memory.db"))
        db_path = db_path.replace("$HERMES_HOME", str(hermes_home))
        base_db_path = Path(db_path).expanduser()
        scope_mode = str(self._config.get("memory_scope") or "user").strip().lower()
        if scope_mode not in {"user", "agent", "global"}:
            scope_mode = "user"
        user_id = normalize_whitespace(
            str(
                kwargs.get("user_id")
                or kwargs.get("user_id_alt")
                or kwargs.get("platform_user_id")
                or kwargs.get("owner_id")
                or kwargs.get("chat_id")
                or ""
            )
        )
        agent_identity = normalize_whitespace(
            str(kwargs.get("agent_identity") or kwargs.get("agent_name") or kwargs.get("agent_workspace") or "")
        )
        scope_parts: List[str] = []
        local_platform = platform in {"cli", "cron", "local", "terminal", "shell"}
        if scope_mode == "user" and not user_id and not local_platform:
            raise RuntimeError(
                f"memory_scope=user requires user_id for platform {platform!r}; refusing a shared database"
            )
        if scope_mode == "agent" and not (user_id or agent_identity) and not local_platform:
            raise RuntimeError(f"memory_scope=agent requires an agent or user identity for platform {platform!r}")
        if scope_mode == "user" and user_id:
            scope_parts = ["user", platform, user_id]
        elif scope_mode == "agent" and (user_id or agent_identity):
            scope_parts = ["agent", platform, user_id or "anonymous", agent_identity or "default"]
        if scope_parts:
            raw_scope = "\x1f".join(scope_parts)
            digest = hashlib.sha256(raw_scope.encode("utf-8")).hexdigest()[:24]
            self._scope_id = f"{scope_mode}:{digest}"
            scopes_dir = base_db_path.parent / f"{base_db_path.stem}_scopes"
            db_path = str(scopes_dir / f"{digest}{base_db_path.suffix or '.db'}")
        else:
            self._scope_id = "global" if scope_mode == "global" else "legacy-local"
        llm_model = str(self._config.get("llm_model") or "").strip()
        llm_base_url = str(self._config.get("llm_base_url") or "").strip()
        llm_api_key = env_or_blank("CONSOLIDATING_MEMORY_LLM_API_KEY")
        embedding_model = str(self._config.get("embedding_model") or "").strip()
        embedding_base_url = str(self._config.get("embedding_base_url") or "").strip()
        embedding_api_key = env_or_blank("CONSOLIDATING_MEMORY_EMBEDDING_API_KEY") or llm_api_key
        self._llm = OpenAICompatibleLLM(
            model=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            timeout_seconds=self._cfg_int("llm_timeout_seconds", 45, 1, 300),
            failure_cooldown_seconds=self._cfg_int("llm_failure_cooldown_seconds", 120, 1, 86400),
            disable_thinking=self._cfg_bool("llm_disable_thinking", False),
        )
        self._embedder = OpenAICompatibleEmbeddings(
            model=embedding_model,
            base_url=embedding_base_url,
            api_key=embedding_api_key,
            timeout_seconds=self._cfg_int("embedding_timeout_seconds", 20, 1, 300),
            failure_cooldown_seconds=self._cfg_int("llm_failure_cooldown_seconds", 120, 1, 86400),
        )
        self._retrieval_backend = str(self._config.get("retrieval_backend", "fts") or "fts").strip().lower()
        if self._retrieval_backend not in {"fts", "hybrid"}:
            self._retrieval_backend = "fts"
        self._session_id = session_id
        encryption_key = ""
        if self._cfg_bool("database_encryption", False):
            encryption_key = env_or_blank("CONSOLIDATING_MEMORY_DB_KEY")
            if not encryption_key:
                raise RuntimeError("database_encryption is enabled but CONSOLIDATING_MEMORY_DB_KEY is not set")
        self._store = MemoryStore(
            db_path=db_path,
            encryption_key=encryption_key,
            conflict_policy=str(self._config.get("conflict_policy") or "evidence").strip().lower(),
        )
        self._store.set_state("memory_scope", self._scope_id)
        if self._write_enabled:
            self._store.ensure_memory_session(session_id, label=session_id, status="open")
            self._sync_builtin_snapshot(reason="initialize")
        self._task_queue = queue.Queue(maxsize=self._cfg()["queue_max_size"])
        self._queue_metrics = {"enqueued": 0, "dropped_prefetch": 0, "spooled": 0, "failed": 0}
        self._stop_event.clear()
        self._draining = False
        self._accepting_tasks = True
        self._worker = threading.Thread(target=self._worker_loop, name="consolidating-memory", daemon=True)
        self._worker.start()
        if self._write_enabled:
            self._enqueue("maintenance")

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        counts = self._store.counts()
        cfg = self._cfg()
        last = self._store.last_consolidation()
        last_text = "never"
        if last:
            last_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(last["finished_at"])))
        backend_desc = (
            f"LLM via {self._llm.model}" if self._llm and self._llm.enabled else "disabled (tool writes only)"
        )
        retrieval_desc = self._retrieval_backend
        if self._retrieval_backend == "hybrid" and (not self._embedder or not self._embedder.supports_embeddings):
            retrieval_desc += " (embeddings unavailable, using FTS fallback)"
        wiki_desc = "disabled"
        if self._cfg()["wiki_export_enabled"]:
            wiki_desc = f"enabled at {self._wiki_export_dir()}"
        return (
            "# Consolidating Memory\n"
            f"Active. {counts['facts']} facts, {counts['topics']} topics, {counts['summaries']} summaries, "
            f"{counts['journals']} journals, {counts['preferences']} preferences, {counts['policies']} policies, "
            f"{counts['episodes']} episode buffers, and {counts['contradictions']} contradiction resolutions logged.\n"
            f"Background consolidation gate: {cfg['min_hours']}h + {cfg['min_sessions']} sessions.\n"
            f"Extractor backend: {backend_desc}.\n"
            f"Retrieval backend: {retrieval_desc}.\n"
            f"Compiled wiki export: {wiki_desc}.\n"
            f"Last consolidation: {last_text}.\n"
            "Use consolidating_memory to search, remember, journal, distill, inspect history/policies, export the wiki mirror, or force consolidation."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._store:
            return ""
        key = session_id or self._session_id
        clean = normalize_whitespace(query)
        if not clean:
            return ""
        with self._prefetch_lock:
            cached = self._prefetch_cache.get(key)
            cache_age = time.time() - float(cached.get("created_at") or 0.0) if cached else float("inf")
            if cached and cached.get("query") == clean and cache_age <= self._cfg()["prefetch_cache_ttl_seconds"]:
                return str(cached.get("rendered") or "")
        cues = self._build_retrieval_cues(query=clean, args={}, session_id=key)
        results = self._search_memory(
            clean,
            scope="all",
            limit=self._cfg()["prefetch_limit"],
            session_id=key,
            cues=cues,
            touch_recall=self._write_enabled,
            allow_embeddings=False,
        )
        rendered = self._render_prefetch(clean, results, cues=cues)
        self._cache_prefetch(key, clean, rendered)
        return rendered

    def get_context(self, *, session_id: str = "", query: str = "") -> str:
        effective_query = normalize_whitespace(query)
        if not effective_query:
            effective_query = (
                "Give me a provenance summary of every fact, preference, policy, "
                "journal note, and changed assumption you know about me."
            )
        return self.prefetch(effective_query, session_id=session_id or self._session_id)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        clean = normalize_whitespace(query)
        if not clean or not self._store:
            return
        self._enqueue("prefetch", query=clean, session_id=session_id or self._session_id)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: List[Dict[str, Any]] | None = None,
    ) -> None:
        if not self._write_enabled:
            return
        self._enqueue(
            "sync_turn",
            session_id=session_id or self._session_id,
            user_content=user_content or "",
            assistant_content=assistant_content or "",
            messages=list(messages or []),
        )

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        if not self._store or not self._write_enabled:
            return
        cooldown = float(self._cfg()["scan_cooldown_seconds"])
        now = time.time()
        if now - self._last_scan_at < cooldown:
            return
        self._last_scan_at = now
        last_maintenance = float(self._store.get_state("last_maintenance_at", "0") or 0)
        if now - last_maintenance >= 86400:
            self._enqueue("maintenance")
        plan = build_consolidation_plan(
            self._store,
            min_hours=self._cfg()["min_hours"],
            min_sessions=self._cfg()["min_sessions"],
        )
        if plan["should_run"]:
            self._request_consolidation(reason="turn_gate")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._store or not self._write_enabled:
            return
        self._enqueue("extract_messages", session_id=self._session_id, messages=messages or [], source="session_end")
        self._request_consolidation(reason="session_end")

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not self._store or not self._write_enabled:
            return ""
        candidates = self._extract_messages_facts(messages or [])
        inserted = 0
        preserved_candidates: List[Dict[str, Any]] = []
        source_refs: List[Dict[str, Any]] = []
        for candidate in candidates[:6]:
            result = self._store_candidate(
                candidate,
                source="precompress_extract",
                session_id=self._session_id,
            )
            fact_id = dict(result.get("fact") or {}).get("id")
            if fact_id is not None:
                preserved_candidates.append(candidate)
                if len(source_refs) < 3:
                    source_refs.append({"kind": "fact", "id": fact_id})
            if result["action"] == "inserted":
                inserted += 1
        if inserted > 0:
            self._store.rebuild_topics(
                max_facts=self._cfg()["max_topic_facts"],
                max_chars=self._cfg()["topic_summary_chars"],
            )
        if not preserved_candidates:
            return ""
        summary = "; ".join(str(item["content"]) for item in preserved_candidates[:3])
        if self._store:
            artifacts = self._store.get_session_artifacts(self._session_id, limit=8)
            summary_sensitivity, _ = self._classify_sensitivity(summary)
            self._store.upsert_summary(
                label="Pre-compression Handoff",
                summary=summary[: self._cfg()["session_summary_chars"]],
                session_id=self._session_id,
                content=summary[: self._cfg()["session_summary_chars"]],
                summary_type="handoff",
                metadata={"source": "precompress"},
                importance=7,
                salience=0.72,
                source_refs=source_refs or self._collect_summary_refs(artifacts, per_section=2),
                reason="precompress",
                sensitivity=summary_sensitivity,
            )
        return (
            "Memory provider preserved pre-compression signals. Preserve these points in the summary: "
            + summary[:500]
            + ("" if inserted == 0 else f" ({inserted} new durable facts stored)")
        )

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        if not self._write_enabled:
            return
        provenance = dict(metadata or {})
        execution_context = str(provenance.get("execution_context") or "").strip().lower()
        if execution_context in {"cron", "flush", "subagent"}:
            return
        self._enqueue(
            "mirror_memory",
            action=action,
            target=target,
            content=content,
            metadata=provenance,
        )

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        clean_id = normalize_whitespace(new_session_id)
        if not clean_id:
            return
        previous = self._session_id
        self._session_id = clean_id
        self._invalidate_prefetch_cache(previous, clean_id)
        if not self._store or not self._write_enabled:
            return
        self._store.ensure_memory_session(clean_id, label=clean_id, status="open")
        parent = normalize_whitespace(parent_session_id or previous)
        if parent and parent != clean_id and not reset:
            self._store.add_link(
                "session",
                clean_id,
                "session",
                parent,
                "rewound_from" if rewound else "continues_from",
            )

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        if not self._write_enabled:
            return
        content = f"Delegated task completed. Task: {normalize_whitespace(task)} Result: {normalize_whitespace(result)}"
        self._enqueue(
            "remember_fact",
            content=content[:500],
            category="workflow",
            topic="delegation-results",
            source="delegation",
            importance=5,
            confidence=0.45,
            metadata={"child_session_id": child_session_id},
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [TOOL_SCHEMA]

    def backup_paths(self) -> List[str]:
        try:
            from hermes_constants import get_hermes_home

            hermes_home = Path(get_hermes_home()).expanduser().resolve()
        except Exception:
            hermes_home = Path("~/.hermes").expanduser().resolve()
        paths: List[str] = []
        configured_db = str(self._config.get("db_path") or "$HERMES_HOME/consolidating_memory.db")
        configured = [configured_db]
        if str(self._config.get("memory_scope") or "user").strip().lower() in {"user", "agent"}:
            base = Path(configured_db.replace("$HERMES_HOME", str(hermes_home))).expanduser()
            configured.append(str(base.parent / f"{base.stem}_scopes"))
        if self._store:
            configured.append(self._store.db_path)
        if self._cfg_bool("wiki_export_enabled", False):
            configured.append(str(self._config.get("wiki_export_dir") or "$HERMES_HOME/consolidating_memory_wiki"))
        for raw in configured:
            path = Path(raw.replace("$HERMES_HOME", str(hermes_home))).expanduser().resolve()
            if path != hermes_home and hermes_home not in path.parents and str(path) not in paths:
                paths.append(str(path))
        return paths

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name != TOOL_SCHEMA["name"]:
            raise NotImplementedError(f"{self.name} does not handle tool {tool_name}")
        if not self._store:
            return json.dumps({"success": False, "error": "Provider not initialized."})

        action = str(args.get("action") or "").strip()
        try:
            limit = max(1, min(int(args.get("limit") or 8), 50))
        except (TypeError, ValueError, OverflowError):
            limit = 8
        session_id = str(args.get("session_id") or self._session_id).strip()
        include_inactive = _flag(args.get("include_inactive"))
        valid_scopes = {"all", "facts", "topics", "episodes", "summaries", "journals", "preferences", "policies"}
        valid_memory_types = {"fact", "summary", "journal", "preference", "policy"}
        mutating = action in {
            "remember",
            "journal",
            "distill",
            "review",
            "decay",
            "export",
            "associate",
            "merge",
            "split",
            "pin",
            "maintain",
            "backup",
            "export_json",
        }
        mutating = mutating or (action == "forget" and not _flag(args.get("dry_run")))
        mutating = mutating or (action == "consolidate" and not _flag(args.get("dry_run")))
        mutating = mutating or (action == "working" and bool(args.get("content") or args.get("status") == "clear"))
        mutating = mutating or (action == "procedure" and bool(args.get("steps") or args.get("status")))
        mutating = mutating or (action == "intention" and bool(args.get("content") or args.get("fact_id") is not None))
        mutating = mutating or (action == "timeline" and bool(args.get("content")))
        mutating = mutating or action == "approval"
        mutating = mutating or (action == "doctor" and _flag(args.get("confirm")))
        mutating = mutating or (action == "policy" and bool(str(args.get("content") or "").strip()))
        if mutating and not self._write_enabled:
            return json.dumps(
                {
                    "success": False,
                    "error": "Sensitive or mutating memory actions are disabled outside the primary agent context.",
                }
            )
        if mutating:
            self._invalidate_prefetch_cache()

        try:
            if action == "search":
                query = str(args.get("query") or "").strip()
                scope = str(args.get("scope") or "all")
                if scope not in valid_scopes:
                    return json.dumps({"success": False, "error": f"Unsupported scope: {scope}"})
                cues = self._build_retrieval_cues(query=query, args=args, session_id=session_id)
                results = self._search_memory(
                    query,
                    scope=scope,
                    limit=limit,
                    session_id=session_id,
                    include_inactive=include_inactive,
                    cues=cues,
                    touch_recall=self._write_enabled,
                )
                payload: Dict[str, Any] = {"success": True, "action": action, "results": results}
                if str(cues.get("mode") or "") == "provenance" and str(cues.get("subject_key") or ""):
                    payload["provenance"] = self._subject_provenance_entries(
                        subject_key=str(cues.get("subject_key") or ""),
                        facts=list(results.get("facts", [])),
                        limit=max(3, min(limit, 6)),
                        query=query,
                    )
                if str(cues.get("mode") or "") in {"summary", "workflow"}:
                    mode_snapshot = self._mode_snapshot_entries(
                        str(cues.get("mode") or ""),
                        max_items=max(4, min(limit, 8)),
                    )
                    if mode_snapshot:
                        payload["current_snapshot"] = mode_snapshot
                return json.dumps(payload)

            if action == "remember":
                result = self._remember_from_tool(args, session_id=session_id)
                return json.dumps({"success": True, "action": action, "result": result})

            if action == "forget":
                fact_id = args.get("fact_id")
                memory_type = str(args.get("memory_type") or "fact").strip().lower()
                if memory_type not in valid_memory_types:
                    return json.dumps({"success": False, "error": f"Unsupported memory_type: {memory_type}"})
                if fact_id is not None:
                    if _flag(args.get("dry_run")):
                        preview = (
                            self._store.explain_fact(int(fact_id))
                            if memory_type == "fact"
                            else {"id": int(fact_id), "memory_type": memory_type}
                        )
                        return json.dumps({"success": True, "action": action, "dry_run": True, "candidate": preview})
                    removed = self._store.deactivate_memory_item(
                        memory_type, int(fact_id), reason="tool_forget", source="tool"
                    )
                    self._store.rebuild_topics(
                        max_facts=self._cfg()["max_topic_facts"],
                        max_chars=self._cfg()["topic_summary_chars"],
                    )
                    if removed:
                        self._sync_builtin_snapshot(reason="tool_forget")
                    return json.dumps(
                        {"success": removed, "action": action, "fact_id": int(fact_id), "memory_type": memory_type}
                    )
                query = str(args.get("query") or "").strip()
                if not query:
                    return json.dumps({"success": False, "error": "query or fact_id is required for forget"})
                section = (
                    "facts"
                    if memory_type == "fact"
                    else {
                        "summary": "summaries",
                        "journal": "journals",
                        "preference": "preferences",
                        "policy": "policies",
                    }[memory_type]
                )
                preview = self._search_memory(
                    query,
                    scope=section,
                    limit=limit,
                    session_id=session_id,
                    include_inactive=False,
                    touch_recall=False,
                ).get(section, [])
                if _flag(args.get("dry_run")) or not _flag(args.get("confirm")):
                    return json.dumps(
                        {
                            "success": True,
                            "action": action,
                            "dry_run": True,
                            "requires_confirmation": True,
                            "candidates": preview,
                            "instruction": "Repeat with fact_id for an exact deletion. Query deletion is refused when ambiguous.",
                        }
                    )
                if len(preview) != 1 or preview[0].get("id") is None:
                    return json.dumps(
                        {
                            "success": False,
                            "action": action,
                            "error": "Query deletion requires exactly one match; use fact_id.",
                            "candidates": preview,
                        }
                    )
                removed = self._store.deactivate_memory_item(
                    memory_type, int(preview[0]["id"]), reason="tool_forget", source="tool"
                )
                if removed:
                    self._store.rebuild_topics(
                        max_facts=self._cfg()["max_topic_facts"], max_chars=self._cfg()["topic_summary_chars"]
                    )
                    self._sync_builtin_snapshot(reason="tool_forget")
                return json.dumps({"success": removed, "action": action, "removed_id": preview[0]["id"]})

            if action == "recent":
                recent = self._store.recent_items(limit=limit)
                for section, rows in recent.items():
                    recent[section] = [row for row in rows if not _looks_sensitive_for_export(row)]
                return json.dumps({"success": True, "action": action, "results": recent})

            if action == "contradictions":
                contradiction_query = str(args.get("query") or "")
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "results": self._visible_sensitive_rows(
                            self._store.recent_contradictions(
                                limit=limit,
                                max_age_days=args.get("since_days"),
                            ),
                            contradiction_query,
                        ),
                    }
                )

            if action == "status":
                plan = build_consolidation_plan(
                    self._store,
                    min_hours=self._cfg()["min_hours"],
                    min_sessions=self._cfg()["min_sessions"],
                )
                review_status = self._store.review_status()
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "counts": self._store.counts(),
                        "plan": plan,
                        "last_consolidation": self._store.last_consolidation(),
                        "recent_contradictions": self._visible_sensitive_rows(
                            self._store.recent_contradictions(limit=3), ""
                        ),
                        "automatic_extraction": {
                            "enabled": bool(self._llm and self._llm.enabled),
                            "backend": "llm" if self._llm and self._llm.enabled else "disabled",
                        },
                        "retrieval_backend": self._effective_retrieval_backend(),
                        "llm_model": self._llm.model if self._llm else "",
                        "llm_base_url": self._llm.base_url if self._llm else "",
                        "llm_disable_thinking": self._llm.disable_thinking if self._llm else False,
                        "llm_circuit": self._llm.circuit_state if self._llm else {},
                        "embedding_model": self._embedder.model if self._embedder else "",
                        "embedding_base_url": self._embedder.base_url if self._embedder else "",
                        "embedding_enabled": bool(self._embedder and self._embedder.supports_embeddings),
                        "embedding_circuit": self._embedder.circuit_state if self._embedder else {},
                        "last_decay_at": self._store.get_state("last_decay_at", ""),
                        "latest_session_summaries": self._visible_sensitive_rows(
                            self._store.latest_session_summaries(limit=3), ""
                        ),
                        "review": review_status,
                        "wiki_export": {
                            "enabled": self._cfg()["wiki_export_enabled"],
                            "on_consolidate": self._cfg()["wiki_export_on_consolidate"],
                            "root": str(self._wiki_export_dir()),
                            "last_export_at": self._store.get_state("last_wiki_export_at", ""),
                            "last_export_stats": self._load_state_json("last_wiki_export_stats"),
                        },
                        "builtin_snapshot": self._load_state_json("last_builtin_snapshot_sync"),
                        "scope": self._scope_id,
                        "database_path": self._store.db_path,
                        "database_size_bytes": self._store.database_size_bytes(),
                        "logical_database_size_bytes": self._store.logical_database_size_bytes(),
                        "queue": {
                            **self._queue_metrics,
                            "in_memory": self._task_queue.qsize(),
                            "durable": self._store.pending_operation_count(),
                            "dead_letter": self._store.failed_operation_count(),
                        },
                        "working_memory": self._visible_sensitive_rows(
                            self._store.list_working_memory(session_id, limit=4), ""
                        ),
                        "due_intentions": self._visible_sensitive_rows(
                            self._store.list_intentions(due_only=True, limit=4), ""
                        ),
                        "pending_approvals": [
                            {
                                key: row.get(key)
                                for key in ("id", "sensitivity", "reason", "status", "session_id", "created_at")
                            }
                            for row in self._store.list_approvals(limit=4)
                        ],
                        "config": self._cfg(),
                    }
                )

            if action == "consolidate":
                if _flag(args.get("dry_run")):
                    return json.dumps(
                        {
                            "success": True,
                            "action": action,
                            "dry_run": True,
                            "plan": build_consolidation_plan(
                                self._store,
                                min_hours=self._cfg()["min_hours"],
                                min_sessions=self._cfg()["min_sessions"],
                            ),
                        }
                    )
                result = self._run_consolidation(force=True, reason="manual")
                return json.dumps({"success": True, "action": action, "result": result})

            if action == "journal":
                label = str(args.get("label") or "Journal").strip()
                content = str(args.get("content") or "").strip()
                if not content:
                    return json.dumps({"success": False, "error": "content is required for journal"})
                journal_candidate = {
                    "content": content,
                    "category": "general",
                    "topic": "journal",
                    "importance": int(args.get("importance") or 6),
                    "confidence": 1.0,
                    "metadata": {"source_role": "user"},
                    "_memory_type": "journal",
                    "_tool_args": dict(args),
                }
                admission = self._admit_candidate(
                    journal_candidate,
                    source="tool_journal",
                    session_id=session_id,
                    approved=_flag(args.get("approved")),
                )
                if admission["decision"] != "allowed":
                    return json.dumps({"success": True, "action": action, "result": admission})
                result = self._store.add_journal(
                    label=label,
                    content=content,
                    session_id=session_id,
                    journal_type=str(args.get("memory_type") or "note"),
                    metadata={"session_id": session_id} if session_id else None,
                    importance=int(args.get("importance") or 6),
                    salience=0.62,
                    sensitivity=str(admission.get("sensitivity") or "normal"),
                )
                return json.dumps({"success": True, "action": action, "result": result})

            if action == "distill":
                result = self._distill_memory(args, session_id=session_id)
                return json.dumps({"success": True, "action": action, "result": result})

            if action == "history":
                results = self._store.list_history(
                    memory_type=str(args.get("memory_type") or ""),
                    entity_id=args.get("fact_id"),
                    subject_key=str(args.get("subject_key") or ""),
                    limit=limit,
                    since_days=args.get("since_days"),
                )
                results = [row for row in results if not _looks_sensitive_for_export(row)]
                return json.dumps({"success": True, "action": action, "results": results})

            if action == "policy":
                key = str(args.get("key") or args.get("query") or "").strip()
                content = str(args.get("content") or "").strip()
                if content:
                    policy_candidate = {
                        "content": content,
                        "category": "workflow",
                        "topic": "policies",
                        "importance": int(args.get("importance") or 9),
                        "confidence": 1.0,
                        "metadata": {"source_role": "user"},
                        "_memory_type": "policy",
                        "_tool_args": dict(args),
                    }
                    admission = self._admit_candidate(
                        policy_candidate,
                        source="tool_policy",
                        session_id=session_id,
                        approved=_flag(args.get("approved")),
                    )
                    if admission["decision"] != "allowed":
                        return json.dumps({"success": True, "action": action, "result": admission})
                    result = self._store.upsert_policy(
                        key=key or slugify(args.get("label") or content[:40]),
                        label=str(args.get("label") or key or "Policy"),
                        content=content,
                        metadata={"session_id": session_id},
                        importance=int(args.get("importance") or 9),
                        sensitivity=str(admission.get("sensitivity") or "normal"),
                    )
                    if session_id:
                        self._store.add_link("policy", result["id"], "session", session_id, "captured_in")
                    self._sync_builtin_snapshot(reason="tool_policy")
                    return json.dumps({"success": True, "action": action, "result": result})
                if not key:
                    return json.dumps(
                        {
                            "success": True,
                            "action": action,
                            "results": self._visible_sensitive_rows(
                                self._store.recent_items(limit=limit).get("policies", []), ""
                            ),
                        }
                    )
                results = self._search_memory(
                    key, scope="policies", limit=limit, session_id=session_id, include_inactive=include_inactive
                )
                return json.dumps({"success": True, "action": action, "results": results.get("policies", [])})

            if action == "review":
                scope = str(args.get("scope") or "all").strip()
                if scope not in valid_scopes:
                    return json.dumps({"success": False, "error": f"Unsupported scope: {scope}"})
                if scope not in {"all", "facts", "summaries", "journals", "preferences", "policies"}:
                    return json.dumps({"success": False, "error": f"Scope {scope} is not reviewable"})
                review_scope = scope
                due = self._store.review_due(scope=review_scope, limit=limit)
                review_query = str(args.get("query") or "")
                for section in ("facts", "summaries", "journals", "preferences", "policies"):
                    due[section] = self._visible_sensitive_rows(due.get(section, []), review_query)
                cues = self._build_retrieval_cues(query=review_query.strip(), args=args, session_id=session_id)
                for section, rows in due.items():
                    for row in rows:
                        row["review_prompt"] = self._review_prompt(section, row)
                self._store.touch_recall_batch(
                    due,
                    session_id=session_id,
                    review_intervals_days=self._review_intervals_days(),
                    reconsolidation_window_hours=float(self._cfg()["reconsolidation_window_hours"]),
                    cues={**cues, "mode": "review"},
                )
                return json.dumps({"success": True, "action": action, "results": due, "reviewed": True})

            if action == "decay":
                result = self._store.apply_decay(
                    half_life_days=float(self._cfg()["decay_half_life_days"]),
                    min_salience=float(self._cfg()["decay_min_salience"]),
                )
                if int(result.get("facts_deactivated") or 0) > 0:
                    self._store.rebuild_topics(
                        max_facts=self._cfg()["max_topic_facts"],
                        max_chars=self._cfg()["topic_summary_chars"],
                    )
                    self._sync_builtin_snapshot(reason="decay")
                return json.dumps({"success": True, "action": action, "result": result})

            if action == "export":
                if not self._cfg()["export_redact_sensitive"] and not _flag(args.get("confirm")):
                    return json.dumps(
                        {
                            "success": False,
                            "error": "Unredacted wiki export requires confirm=true because it may expose private memory",
                        }
                    )
                result = self._export_compiled_wiki(reason="tool")
                return json.dumps({"success": True, "action": action, "result": result})

            if action == "explain":
                if args.get("fact_id") is None:
                    return json.dumps({"success": False, "error": "fact_id is required for explain"})
                return json.dumps(
                    {"success": True, "action": action, "result": self._store.explain_fact(int(args["fact_id"]))}
                )

            if action == "working":
                content = str(args.get("content") or "").strip()
                if str(args.get("status") or "").lower() == "clear":
                    count = self._store.clear_working_memory(session_id, str(args.get("key") or ""))
                    return json.dumps({"success": True, "action": action, "cleared": count})
                if content:
                    working_candidate = {
                        "content": content,
                        "category": "workflow",
                        "topic": "working-memory",
                        "importance": int(args.get("importance") or 7),
                        "confidence": 1.0,
                        "metadata": {"source_role": "user"},
                        "_memory_type": "working",
                        "_tool_args": dict(args),
                    }
                    admission = self._admit_candidate(
                        working_candidate,
                        source="tool_working",
                        session_id=session_id,
                        approved=_flag(args.get("approved")),
                    )
                    if admission["decision"] != "allowed":
                        return json.dumps({"success": True, "action": action, "result": admission})
                    result = self._store.set_working_memory(
                        session_id=session_id,
                        memory_key=str(args.get("key") or args.get("label") or "focus"),
                        content=content,
                        priority=int(args.get("importance") or 7),
                        ttl_seconds=float(args.get("ttl_seconds") or 3600),
                        capacity=self._cfg()["working_memory_capacity"],
                        sensitivity=str(admission.get("sensitivity") or "normal"),
                    )
                    return json.dumps({"success": True, "action": action, "result": result})
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "results": self._visible_sensitive_rows(
                            self._store.list_working_memory(session_id, limit=limit), str(args.get("query") or "")
                        ),
                    }
                )

            if action == "procedure":
                raw_steps = args.get("steps")
                raw_prerequisites = args.get("prerequisites")
                if raw_steps is not None and not isinstance(raw_steps, (list, tuple)):
                    return json.dumps({"success": False, "error": "steps must be an array of strings"})
                if raw_prerequisites is not None and not isinstance(raw_prerequisites, (list, tuple)):
                    return json.dumps({"success": False, "error": "prerequisites must be an array of strings"})
                steps = list(raw_steps or [])
                prerequisites = list(raw_prerequisites or [])
                if steps:
                    procedure_content = " ".join(
                        [
                            str(args.get("label") or args.get("key") or "Procedure"),
                            *(str(step) for step in steps),
                            *(str(item) for item in prerequisites),
                            str(args.get("content") or ""),
                            str(args.get("value") or ""),
                        ]
                    )
                    procedure_candidate = {
                        "content": procedure_content,
                        "category": "workflow",
                        "topic": "procedures",
                        "importance": int(args.get("importance") or 7),
                        "confidence": 1.0,
                        "metadata": {"source_role": "user"},
                        "_memory_type": "procedure",
                        "_tool_args": dict(args),
                    }
                    admission = self._admit_candidate(
                        procedure_candidate,
                        source="tool_procedure",
                        session_id=session_id,
                        approved=_flag(args.get("approved")),
                    )
                    if admission["decision"] != "allowed":
                        return json.dumps({"success": True, "action": action, "result": admission})
                    result = self._store.upsert_procedure(
                        procedure_key=str(args.get("key") or args.get("label") or args.get("query") or "procedure"),
                        label=str(args.get("label") or args.get("key") or "Procedure"),
                        steps=steps,
                        prerequisites=prerequisites,
                        success_criteria=str(args.get("content") or ""),
                        failure_recovery=str(args.get("value") or ""),
                        sensitivity=str(admission.get("sensitivity") or "normal"),
                    )
                    return json.dumps({"success": True, "action": action, "result": result})
                if str(args.get("status") or "").lower() in {"success", "failed", "failure"}:
                    result = self._store.record_procedure_result(
                        str(args.get("key") or args.get("query") or ""),
                        success=str(args.get("status") or "").lower() == "success",
                    )
                    return json.dumps({"success": True, "action": action, "result": result})
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "results": self._visible_sensitive_rows(
                            self._store.list_procedures(str(args.get("query") or ""), limit=limit),
                            str(args.get("query") or ""),
                        ),
                    }
                )

            if action == "intention":
                if args.get("fact_id") is not None and args.get("status"):
                    result = self._store.resolve_intention(int(args["fact_id"]), status=str(args["status"]))
                    return json.dumps({"success": True, "action": action, "result": result})
                if str(args.get("content") or "").strip():
                    intention_candidate = {
                        "content": " ".join(
                            (str(args["content"]), str(args.get("query") or ""), str(args.get("value") or ""))
                        ),
                        "category": "workflow",
                        "topic": "intentions",
                        "importance": int(args.get("importance") or 6),
                        "confidence": 1.0,
                        "metadata": {"source_role": "user"},
                        "_memory_type": "intention",
                        "_tool_args": dict(args),
                    }
                    admission = self._admit_candidate(
                        intention_candidate,
                        source="tool_intention",
                        session_id=session_id,
                        approved=_flag(args.get("approved")),
                    )
                    if admission["decision"] != "allowed":
                        return json.dumps({"success": True, "action": action, "result": admission})
                    result = self._store.add_intention(
                        intention=str(args["content"]),
                        due_at=float(args.get("due_at") or 0),
                        condition_text=str(args.get("query") or ""),
                        recurrence=str(args.get("value") or ""),
                        importance=int(args.get("importance") or 6),
                        session_id=session_id,
                        sensitivity=str(admission.get("sensitivity") or "normal"),
                    )
                    return json.dumps({"success": True, "action": action, "result": result})
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "results": self._visible_sensitive_rows(
                            self._store.list_intentions(due_only=bool(args.get("status") == "due"), limit=limit),
                            str(args.get("query") or ""),
                        ),
                    }
                )

            if action == "timeline":
                if str(args.get("content") or "").strip():
                    timeline_candidate = {
                        "content": str(args["content"]),
                        "category": "general",
                        "topic": "life-events",
                        "importance": int(args.get("importance") or 6),
                        "confidence": 1.0,
                        "metadata": {"kind": "life_event", "source_role": "user"},
                        "_memory_type": "timeline",
                        "_tool_args": dict(args),
                    }
                    admission = self._admit_candidate(
                        timeline_candidate,
                        source="tool_timeline",
                        session_id=session_id,
                        approved=_flag(args.get("approved")),
                    )
                    if admission["decision"] != "allowed":
                        return json.dumps({"success": True, "action": action, "result": admission})
                    result = self._store.upsert_autobiographical_event(
                        event_key=str(
                            args.get("key") or args.get("label") or fingerprint_text(str(args["content"]))[:16]
                        ),
                        content=str(args["content"]),
                        event_at=float(args.get("due_at") or time.time()),
                        importance=int(args.get("importance") or 6),
                        sensitivity=str(admission.get("sensitivity") or "normal"),
                    )
                    return json.dumps({"success": True, "action": action, "result": result})
                timeline_query = str(args.get("query") or "")
                events = [
                    row
                    for row in self._store.list_autobiographical_events(timeline_query, limit=limit)
                    if str(row.get("sensitivity") or "normal") == "normal"
                    or self._query_allows_sensitive(timeline_query, str(row.get("sensitivity") or ""))
                ]
                return json.dumps({"success": True, "action": action, "results": events})

            if action == "approval":
                if args.get("fact_id") is None:
                    return json.dumps(
                        {
                            "success": True,
                            "action": action,
                            "results": self._store.list_approvals(
                                status=str(args.get("status") or "pending"), limit=limit
                            ),
                        }
                    )
                if "approved" not in args:
                    return json.dumps(
                        {"success": False, "error": "approved=true or false is required to resolve an approval"}
                    )
                approval = self._store.resolve_approval(
                    int(args["fact_id"]),
                    approved=_flag(args.get("approved")),
                    resolution=str(args.get("content") or ""),
                )
                stored: Dict[str, Any] | None = None
                if _flag(args.get("approved")):
                    candidate = dict(approval.get("candidate") or {})
                    source = str(candidate.pop("_source", "approved_sensitive"))
                    source_session = str(candidate.pop("_session_id", session_id))
                    tool_args = candidate.pop("_tool_args", None)
                    special_type = candidate.pop("_memory_type", "")
                    if special_type == "preference" and isinstance(tool_args, dict):
                        stored = self._remember_from_tool({**tool_args, "approved": True}, session_id=source_session)
                    elif special_type == "journal" and isinstance(tool_args, dict):
                        stored = self._store.add_journal(
                            label=str(tool_args.get("label") or "Journal"),
                            content=str(tool_args.get("content") or ""),
                            session_id=source_session,
                            journal_type=str(tool_args.get("memory_type") or "note"),
                            metadata={"session_id": source_session},
                            importance=int(tool_args.get("importance") or 6),
                            salience=0.62,
                            sensitivity=str(approval.get("sensitivity") or "normal"),
                        )
                    elif special_type == "timeline" and isinstance(tool_args, dict):
                        timeline_content = str(tool_args.get("content") or "")
                        stored = self._store.upsert_autobiographical_event(
                            event_key=str(
                                tool_args.get("key")
                                or tool_args.get("label")
                                or fingerprint_text(timeline_content)[:16]
                            ),
                            content=timeline_content,
                            event_at=float(tool_args.get("due_at") or time.time()),
                            importance=int(tool_args.get("importance") or 6),
                            sensitivity=str(approval.get("sensitivity") or "normal"),
                        )
                    elif special_type == "distill" and isinstance(tool_args, dict):
                        stored = self._distill_memory({**tool_args, "approved": True}, session_id=source_session)
                    elif special_type == "policy" and isinstance(tool_args, dict):
                        policy_content = str(tool_args.get("content") or "")
                        stored = self._store.upsert_policy(
                            key=str(tool_args.get("key") or tool_args.get("query") or slugify(policy_content[:40])),
                            label=str(tool_args.get("label") or tool_args.get("key") or "Policy"),
                            content=policy_content,
                            metadata={"session_id": source_session},
                            importance=int(tool_args.get("importance") or 9),
                            sensitivity=str(approval.get("sensitivity") or "normal"),
                            reason="approved_sensitive_policy",
                        )
                    elif special_type == "working" and isinstance(tool_args, dict):
                        stored = self._store.set_working_memory(
                            session_id=source_session,
                            memory_key=str(tool_args.get("key") or tool_args.get("label") or "focus"),
                            content=str(tool_args.get("content") or ""),
                            priority=int(tool_args.get("importance") or 7),
                            ttl_seconds=float(tool_args.get("ttl_seconds") or 3600),
                            capacity=self._cfg()["working_memory_capacity"],
                            sensitivity=str(approval.get("sensitivity") or "normal"),
                        )
                    elif special_type == "procedure" and isinstance(tool_args, dict):
                        stored = self._store.upsert_procedure(
                            procedure_key=str(
                                tool_args.get("key") or tool_args.get("label") or tool_args.get("query") or "procedure"
                            ),
                            label=str(tool_args.get("label") or tool_args.get("key") or "Procedure"),
                            steps=list(tool_args.get("steps") or []),
                            prerequisites=list(tool_args.get("prerequisites") or []),
                            success_criteria=str(tool_args.get("content") or ""),
                            failure_recovery=str(tool_args.get("value") or ""),
                            sensitivity=str(approval.get("sensitivity") or "normal"),
                        )
                    elif special_type == "intention" and isinstance(tool_args, dict):
                        stored = self._store.add_intention(
                            intention=str(tool_args.get("content") or ""),
                            due_at=float(tool_args.get("due_at") or 0),
                            condition_text=str(tool_args.get("query") or ""),
                            recurrence=str(tool_args.get("value") or ""),
                            importance=int(tool_args.get("importance") or 6),
                            session_id=source_session,
                            sensitivity=str(approval.get("sensitivity") or "normal"),
                        )
                    else:
                        stored = self._store_candidate(
                            candidate, source=source, session_id=source_session, approved=True
                        )
                return json.dumps({"success": True, "action": action, "approval": approval, "stored": stored})

            if action == "associate":
                result = self._store.associate(
                    str(args.get("left_kind") or "fact"),
                    str(args.get("left_id") or args.get("fact_id") or ""),
                    str(args.get("right_kind") or "fact"),
                    str(args.get("right_id") or ""),
                    str(args.get("relation") or "associated"),
                )
                return json.dumps({"success": True, "action": action, "result": result})

            if action == "merge":
                loser_ids = args.get("ids")
                if args.get("fact_id") is None or not isinstance(loser_ids, (list, tuple)) or not loser_ids:
                    return json.dumps({"success": False, "error": "fact_id (winner) and ids (losers) are required"})
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "result": self._store.merge_facts(int(args["fact_id"]), list(loser_ids)),
                    }
                )

            if action == "split":
                replacement_contents = args.get("contents")
                if (
                    args.get("fact_id") is None
                    or not isinstance(replacement_contents, (list, tuple))
                    or len(replacement_contents) < 2
                ):
                    return json.dumps(
                        {"success": False, "error": "fact_id and at least two replacement contents are required"}
                    )
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "result": self._store.split_fact(int(args["fact_id"]), list(replacement_contents)),
                    }
                )

            if action == "pin":
                if args.get("fact_id") is None:
                    return json.dumps({"success": False, "error": "fact_id is required for pin"})
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "result": self._store.pin_fact(int(args["fact_id"]), _flag(args.get("pinned"), True)),
                    }
                )

            if action == "doctor":
                return json.dumps(
                    {"success": True, "action": action, "result": self._store.doctor(repair=_flag(args.get("confirm")))}
                )

            if action == "maintain":
                result = self._store.maintain(
                    episode_retention_hours=float(self._cfg()["episode_body_retention_hours"]),
                    trace_retention_days=float(self._cfg()["trace_retention_days"]),
                    history_retention_days=float(self._cfg()["history_retention_days"]),
                    sensitive_retention_days=float(self._cfg()["sensitive_retention_days"]),
                    max_database_mb=float(self._cfg()["max_database_mb"]),
                )
                return json.dumps({"success": True, "action": action, "result": result})

            if action == "backup":
                destination = str(args.get("destination") or "").strip()
                if not destination:
                    scope_suffix = self._scope_id.replace(":", "-")
                    destination = str(
                        self._hermes_home / "backups" / f"consolidating-memory-{scope_suffix}-{time.time_ns()}.db"
                    )
                target = Path(destination).expanduser().resolve()
                if not target.is_relative_to(self._hermes_home.resolve()) and not _flag(args.get("confirm")):
                    return json.dumps(
                        {
                            "success": False,
                            "error": "Backing up outside HERMES_HOME requires confirm=true because backups contain unredacted memory",
                        }
                    )
                return json.dumps({"success": True, "action": action, "path": self._store.backup_to(destination)})

            if action == "export_json":
                destination = str(args.get("destination") or "").strip()
                redact_sensitive = bool(self._cfg()["export_redact_sensitive"])
                if not redact_sensitive and not _flag(args.get("confirm")):
                    return json.dumps(
                        {
                            "success": False,
                            "error": "Unredacted export requires confirm=true because it may expose private memory",
                        }
                    )
                data = self._store.export_data(redact_sensitive=redact_sensitive)
                if not destination:
                    return json.dumps({"success": True, "action": action, "result": data})
                path = Path(destination).expanduser().resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                temp_export: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
                        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
                        handle.flush()
                        os.fsync(handle.fileno())
                        temp_export = Path(handle.name)
                    os.replace(temp_export, path)
                    temp_export = None
                    try:
                        os.chmod(path, 0o600)
                    except OSError:
                        pass
                finally:
                    if temp_export and temp_export.exists():
                        temp_export.unlink()
                return json.dumps({"success": True, "action": action, "path": str(path)})

            return json.dumps({"success": False, "error": f"Unknown action: {action}"})
        except Exception as exc:
            logger.exception("Tool call failed for %s", self.name)
            return json.dumps({"success": False, "error": str(exc)})

    def shutdown(self) -> None:
        with self._state_lock:
            if not self._accepting_tasks and self._store is None:
                return
            self._accepting_tasks = False
            self._draining = True
        try:
            self._task_queue.put_nowait(None)
        except queue.Full:
            self._spool_queued_tasks(preserve_sentinel=False)
            self._task_queue.put_nowait(None)
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=float(self._cfg()["shutdown_timeout_seconds"]))
        if self._worker and self._worker.is_alive():
            spooled = self._spool_queued_tasks(preserve_sentinel=True)
            logger.error(
                "Memory worker did not drain before the shutdown deadline; left the database open and spooled %s queued tasks",
                spooled,
            )
            return
        self._stop_event.set()
        if self._store:
            self._store.close()
            self._store = None
        self._worker = None
        self._invalidate_prefetch_cache()

    def _cfg(self) -> Dict[str, Any]:
        sensitive_memory = str(self._config.get("sensitive_memory") or "ask").strip().lower()
        if sensitive_memory not in {"deny", "ask", "allow"}:
            sensitive_memory = "ask"
        return {
            "min_hours": self._cfg_int("min_hours", 24, 0, 8760),
            "min_sessions": self._cfg_int("min_sessions", 5, 1, 10000),
            "scan_cooldown_seconds": self._cfg_int("scan_cooldown_seconds", 600, 1, 86400),
            "prefetch_limit": self._cfg_int("prefetch_limit", 8, 1, 50),
            "max_topic_facts": self._cfg_int("max_topic_facts", 5, 1, 100),
            "topic_summary_chars": self._cfg_int("topic_summary_chars", 650, 100, 10000),
            "session_summary_chars": self._cfg_int("session_summary_chars", 900, 100, 20000),
            "prune_after_days": self._cfg_int("prune_after_days", 90, 1, 36500),
            "episode_body_retention_hours": self._cfg_float("episode_body_retention_hours", 24, 0, 87600),
            "decay_half_life_days": self._cfg_float("decay_half_life_days", 90, 0.01, 36500),
            "reconsolidation_window_hours": self._cfg_float("reconsolidation_window_hours", 6, 0, 8760),
            "review_intervals_days": str(self._config.get("review_intervals_days", "1,3,7,14,30")),
            "decay_min_salience": self._cfg_float("decay_min_salience", 0.15, 0, 1),
            "builtin_snapshot_sync_enabled": self._cfg_bool("builtin_snapshot_sync_enabled", False),
            "builtin_memory_dir": str(self._config.get("builtin_memory_dir", "$HERMES_HOME/memories")),
            "builtin_snapshot_user_chars": self._cfg_int("builtin_snapshot_user_chars", 1375, 100, 100000),
            "builtin_snapshot_memory_chars": self._cfg_int("builtin_snapshot_memory_chars", 2200, 100, 100000),
            "wiki_export_enabled": self._cfg_bool("wiki_export_enabled", False),
            "wiki_export_dir": str(self._config.get("wiki_export_dir", "$HERMES_HOME/consolidating_memory_wiki")),
            "wiki_export_on_consolidate": self._cfg_bool("wiki_export_on_consolidate", True),
            "wiki_export_session_limit": self._cfg_int("wiki_export_session_limit", 50, 1, 10000),
            "wiki_export_topic_limit": self._cfg_int("wiki_export_topic_limit", 100, 1, 10000),
            "llm_timeout_seconds": self._cfg_int("llm_timeout_seconds", 45, 1, 300),
            "llm_max_input_chars": self._cfg_int("llm_max_input_chars", 4000, 256, 100000),
            "llm_disable_thinking": self._cfg_bool("llm_disable_thinking", False),
            "retrieval_backend": str(self._config.get("retrieval_backend", "fts") or "fts").strip().lower(),
            "embedding_timeout_seconds": self._cfg_int("embedding_timeout_seconds", 20, 1, 300),
            "embedding_candidate_limit": self._cfg_int("embedding_candidate_limit", 16, 1, 100),
            "prefetch_cache_ttl_seconds": self._cfg_float("prefetch_cache_ttl_seconds", 120, 0, 86400),
            "memory_scope": str(self._config.get("memory_scope") or "user").strip().lower(),
            "sensitive_memory": sensitive_memory,
            "allow_credential_memory": self._cfg_bool("allow_credential_memory", False),
            "allow_sensitive_model_processing": self._cfg_bool("allow_sensitive_model_processing", False),
            "conflict_policy": str(self._config.get("conflict_policy") or "evidence").strip().lower(),
            "queue_max_size": self._cfg_int("queue_max_size", 256, 8, 100000),
            "queue_max_attempts": self._cfg_int("queue_max_attempts", 5, 1, 100),
            "shutdown_timeout_seconds": self._cfg_float("shutdown_timeout_seconds", 10, 1, 60),
            "max_database_mb": self._cfg_float("max_database_mb", 512, 16, 102400),
            "trace_retention_days": self._cfg_float("trace_retention_days", 30, 1, 36500),
            "history_retention_days": self._cfg_float("history_retention_days", 180, 1, 36500),
            "sensitive_retention_days": self._cfg_float("sensitive_retention_days", 30, 1, 36500),
            "consolidation_max_batches": self._cfg_int("consolidation_max_batches", 4, 1, 100),
            "consolidation_batch_size": self._cfg_int("consolidation_batch_size", 250, 1, 5000),
            "working_memory_capacity": self._cfg_int("working_memory_capacity", 12, 1, 100),
            "database_encryption": self._cfg_bool("database_encryption", False),
            "export_redact_sensitive": self._cfg_bool("export_redact_sensitive", True),
            "llm_failure_cooldown_seconds": self._cfg_int("llm_failure_cooldown_seconds", 120, 1, 86400),
        }

    def _cfg_int(self, key: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(self._config.get(key, default))
        except (TypeError, ValueError, OverflowError):
            value = default
        return max(minimum, min(maximum, value))

    def _cfg_float(self, key: str, default: float, minimum: float, maximum: float) -> float:
        try:
            value = float(self._config.get(key, default))
        except (TypeError, ValueError, OverflowError):
            value = default
        if not math.isfinite(value):
            value = default
        return max(minimum, min(maximum, value))

    def _cfg_bool(self, key: str, default: bool) -> bool:
        raw = self._config.get(key, default)
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def _wiki_export_dir(self) -> Path:
        raw = str(self._cfg()["wiki_export_dir"] or "$HERMES_HOME/consolidating_memory_wiki")
        root = Path(raw.replace("$HERMES_HOME", str(self._hermes_home))).expanduser()
        if self._scope_id.startswith(("user:", "agent:")):
            return root / "scopes" / self._scope_id.split(":", 1)[1]
        return root

    def _builtin_memory_dir(self) -> Path:
        raw = str(self._cfg()["builtin_memory_dir"] or "$HERMES_HOME/memories")
        return Path(raw.replace("$HERMES_HOME", str(self._hermes_home))).expanduser()

    def _builtin_memory_path(self, target: str) -> Path:
        name = "USER.md" if target == "user" else "MEMORY.md"
        return self._builtin_memory_dir() / name

    def _strip_auto_memory_block(self, text: str) -> str:
        if not text:
            return ""
        pattern = rf"{re.escape(AUTO_MEMORY_BLOCK_START)}.*?{re.escape(AUTO_MEMORY_BLOCK_END)}\s*"
        return re.sub(pattern, "", text, flags=re.DOTALL).strip()

    def _select_snapshot_entries(self, entries: List[Dict[str, Any]], *, limit_chars: int) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        used = 0
        for entry in sorted(
            entries,
            key=lambda item: (
                int(item.get("importance") or 0),
                float(item.get("salience") or 0.0),
                float(item.get("updated_at") or 0.0),
                1 if str(item.get("subject_key") or "") else 0,
            ),
            reverse=True,
        ):
            text = normalize_whitespace(str(entry.get("text") or ""))
            if not text:
                continue
            cost = len(f"- {text}\n")
            if used + cost > max(int(limit_chars), 0):
                continue
            kept.append({**entry, "text": text})
            used += cost
        return kept

    def _build_snapshot_block(self, entries: List[Dict[str, Any]]) -> str:
        if not entries:
            return ""
        lines = [AUTO_MEMORY_BLOCK_START]
        lines.extend(f"- {entry['text']}" for entry in entries if str(entry.get("text") or "").strip())
        lines.append(AUTO_MEMORY_BLOCK_END)
        return "\n".join(lines).strip()

    def _mirror_memory_candidates(self, content: str) -> List[Dict[str, Any]]:
        clean = normalize_whitespace(content)
        if not clean:
            return []
        candidate = normalize_candidate_fact(
            {
                "content": clean,
                "category": "general",
                "topic": "hermes-memory",
                "importance": 7,
                "confidence": 0.95,
            },
            source_role="tool",
        )
        return [candidate] if candidate else []

    def _build_builtin_snapshot_entries(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self._store:
            return {"user": [], "memory": []}
        snapshot = self._store.prompt_snapshot_rows()
        entries: Dict[str, List[Dict[str, Any]]] = {"user": [], "memory": []}
        seen_subjects: Dict[str, set[str]] = {"user": set(), "memory": set()}
        seen_texts: Dict[str, set[str]] = {"user": set(), "memory": set()}

        def add_entry(
            target: str,
            *,
            text: str,
            subject_key: str = "",
            importance: Any = 5,
            salience: Any = 0.5,
            updated_at: Any = 0.0,
        ) -> None:
            clean_text = normalize_whitespace(text)
            clean_subject = normalize_whitespace(subject_key)
            if not clean_text:
                return
            normalized_text = normalize_text(clean_text)
            if normalized_text in seen_texts[target]:
                return
            if clean_subject and clean_subject in seen_subjects[target]:
                return
            seen_texts[target].add(normalized_text)
            if clean_subject:
                seen_subjects[target].add(clean_subject)
            entries[target].append(
                {
                    "text": clean_text,
                    "subject_key": clean_subject,
                    "importance": int(importance or 0),
                    "salience": float(salience or 0.0),
                    "updated_at": float(updated_at or 0.0),
                }
            )

        for row in snapshot.get("user_facts", []):
            add_entry(
                "user",
                text=str(row.get("content") or ""),
                subject_key=str(row.get("subject_key") or ""),
                importance=row.get("importance"),
                salience=row.get("salience"),
                updated_at=row.get("updated_at"),
            )
        for row in snapshot.get("preferences", []):
            metadata = dict(row.get("metadata") or {})
            subject_key = str(metadata.get("subject_key") or "")
            pref_key = str(row.get("preference_key") or "")
            text = str(row.get("content") or row.get("label") or row.get("value") or "")
            if not subject_key.startswith("user:"):
                continue
            # Cross-table dedup: skip if this preference's subject_key
            # OR preference_key was already added by a fact row.
            if subject_key and subject_key in seen_subjects.get("user", set()):
                continue
            if pref_key and pref_key != subject_key and pref_key in seen_subjects.get("user", set()):
                continue
            add_entry(
                "user",
                text=text,
                subject_key=subject_key or pref_key,
                importance=row.get("importance"),
                salience=row.get("salience"),
                updated_at=row.get("updated_at"),
            )
        for row in snapshot.get("memory_facts", []):
            add_entry(
                "memory",
                text=str(row.get("content") or ""),
                subject_key=str(row.get("subject_key") or ""),
                importance=row.get("importance"),
                salience=row.get("salience"),
                updated_at=row.get("updated_at"),
            )
        for row in snapshot.get("policies", []):
            metadata = dict(row.get("metadata") or {})
            add_entry(
                "memory",
                text=str(row.get("content") or row.get("label") or ""),
                subject_key=str(metadata.get("subject_key") or row.get("policy_key") or ""),
                importance=row.get("importance"),
                salience=row.get("salience"),
                updated_at=row.get("updated_at"),
            )
        return entries

    def _line_should_be_replaced(
        self,
        raw_line: str,
        *,
        normalized_contents: set[str],
    ) -> bool:
        clean_line = raw_line.strip()
        if not clean_line:
            return False
        if clean_line in {AUTO_MEMORY_BLOCK_START, AUTO_MEMORY_BLOCK_END}:
            return True
        text = clean_line[2:].strip() if clean_line.startswith("- ") else clean_line
        if normalize_text(text) in normalized_contents:
            return True
        return False

    def _write_builtin_snapshot_file(
        self, target: str, entries: List[Dict[str, Any]], *, limit_chars: int
    ) -> Dict[str, Any]:
        path = self._builtin_memory_path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if path.exists():
            try:
                existing = path.read_text(encoding="utf-8")
            except Exception:
                existing = path.read_text(encoding="utf-8", errors="ignore")
        stripped = self._strip_auto_memory_block(existing)
        normalized_contents = {
            normalize_text(str(entry.get("text") or "")) for entry in entries if str(entry.get("text") or "").strip()
        }
        preserved_lines = [
            line.rstrip()
            for line in stripped.splitlines()
            if not self._line_should_be_replaced(line, normalized_contents=normalized_contents)
        ]
        selected = self._select_snapshot_entries(entries, limit_chars=max(int(limit_chars), 0))
        block = self._build_snapshot_block(selected)
        preserved_text = "\n".join(preserved_lines).strip()
        combined = block
        if preserved_text:
            combined = f"{block}\n\n{preserved_text}" if block else preserved_text
        while preserved_lines and len(combined) > max(int(limit_chars), 0):
            preserved_lines.pop()
            while preserved_lines and not preserved_lines[-1].strip():
                preserved_lines.pop()
            preserved_text = "\n".join(preserved_lines).strip()
            combined = f"{block}\n\n{preserved_text}" if block and preserved_text else (block or preserved_text)
        while selected and len(combined) > max(int(limit_chars), 0):
            selected.pop()
            block = self._build_snapshot_block(selected)
            preserved_text = "\n".join(preserved_lines).strip()
            combined = f"{block}\n\n{preserved_text}" if block and preserved_text else (block or preserved_text)
        normalized = combined.strip()
        if normalized:
            normalized += "\n"
        changed = normalize_whitespace(existing) != normalize_whitespace(normalized)
        if changed:
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    dir=path.parent,
                    delete=False,
                ) as handle:
                    handle.write(normalized)
                    handle.flush()
                    os.fsync(handle.fileno())
                    temporary = Path(handle.name)
                os.replace(temporary, path)
                temporary = None
                try:
                    os.chmod(path, 0o600)
                except OSError:
                    pass
            finally:
                if temporary and temporary.exists():
                    temporary.unlink()
        return {
            "path": str(path),
            "changed": changed,
            "chars": len(normalized),
            "entries": len(selected),
        }

    def _sync_builtin_snapshot(self, *, reason: str) -> Dict[str, Any]:
        if not self._store or not self._cfg()["builtin_snapshot_sync_enabled"]:
            return {"success": False, "reason": "disabled"}
        if self._scope_id.startswith(("user:", "agent:")):
            result = {
                "success": False,
                "reason": "scoped memory cannot safely write shared Hermes USER.md or MEMORY.md files",
            }
            self._store.set_state("last_builtin_snapshot_sync", json.dumps(result, sort_keys=True))
            return result
        try:
            entries = self._build_builtin_snapshot_entries()
            result = {
                "success": True,
                "reason": reason,
                "user": self._write_builtin_snapshot_file(
                    "user",
                    entries.get("user", []),
                    limit_chars=int(self._cfg()["builtin_snapshot_user_chars"]),
                ),
                "memory": self._write_builtin_snapshot_file(
                    "memory",
                    entries.get("memory", []),
                    limit_chars=int(self._cfg()["builtin_snapshot_memory_chars"]),
                ),
            }
        except Exception as exc:
            logger.warning("Builtin snapshot sync failed: %s", exc)
            result = {"success": False, "reason": reason, "error": str(exc)}
        try:
            self._store.set_state("last_builtin_snapshot_sync", json.dumps(result, sort_keys=True))
        except Exception:
            logger.debug("Failed to persist builtin snapshot sync metadata", exc_info=True)
        return result

    def _load_state_json(self, key: str) -> Dict[str, Any]:
        if not self._store:
            return {}
        raw = str(self._store.get_state(key, "") or "").strip()
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _effective_retrieval_backend(self) -> str:
        if self._retrieval_backend == "hybrid" and self._embedder and self._embedder.supports_embeddings:
            return "hybrid"
        return "fts"

    def _section_limit(self, section: str, limit: int) -> int:
        return int(limit) if section == "facts" else max(1, min(int(limit), 6))

    def _review_intervals_days(self) -> List[float]:
        raw = str(self._cfg()["review_intervals_days"] or "1,3,7,14,30")
        values: List[float] = []
        for chunk in raw.split(","):
            clean = normalize_whitespace(chunk)
            if not clean:
                continue
            try:
                value = float(clean)
            except Exception:
                continue
            if value > 0:
                values.append(value)
        return values or [1.0, 3.0, 7.0, 14.0, 30.0]

    def _json_dict(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if not value:
            return {}
        try:
            data = json.loads(str(value))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _result_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        metadata = self._json_dict(row.get("metadata"))
        if metadata:
            return metadata
        return self._json_dict(row.get("metadata_json"))

    @staticmethod
    def _query_allows_sensitive(query: str, sensitivity: str) -> bool:
        clean = normalize_text(query)
        markers = {
            "credential": ("password", "passphrase", "credential", "api key", "token", "private key", "secret"),
            "health": ("health", "medical", "diagnosis", "medication", "surgery", "allergy"),
            "financial": ("financial", "bank", "iban", "credit card", "salary", "income", "debt"),
            "identity": ("identity", "date of birth", "dob", "passport", "social security", "national id"),
            "location": ("address", "exact location", "home location", "where do i live"),
        }
        return any(marker in clean for marker in markers.get(normalize_text(sensitivity), ()))

    def _visible_sensitive_rows(self, rows: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        return [
            row
            for row in rows
            if str(row.get("sensitivity") or "normal") == "normal"
            or self._query_allows_sensitive(query, str(row.get("sensitivity") or ""))
        ]

    def _decorate_search_results(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        decorated: Dict[str, List[Dict[str, Any]]] = {}
        for section, rows in results.items():
            enriched: List[Dict[str, Any]] = []
            for raw_row in rows:
                row = dict(raw_row)
                metadata = self._result_metadata(row)
                if metadata:
                    row["metadata"] = metadata
                    if not row.get("subject_key") and metadata.get("subject_key"):
                        row["subject_key"] = str(metadata.get("subject_key") or "")
                    if not row.get("source_session_id") and metadata.get("source_session_id"):
                        row["source_session_id"] = str(metadata.get("source_session_id") or "")
                    if metadata.get("source_label"):
                        row["source_label"] = str(metadata.get("source_label") or "")
                    if metadata.get("turn_id"):
                        row["turn_id"] = str(metadata.get("turn_id") or "")
                enriched.append(row)
            decorated[section] = enriched
        return decorated

    def _merge_prefetch_rows(
        self,
        section: str,
        *row_groups: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        merged: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()
        seen_texts: set[str] = set()
        for rows in row_groups:
            for raw_row in rows:
                row = dict(raw_row)
                metadata = self._result_metadata(row)
                if metadata and "metadata" not in row:
                    row["metadata"] = metadata
                subject_key = normalize_whitespace(
                    str(
                        row.get("subject_key")
                        or metadata.get("subject_key")
                        or row.get("preference_key")
                        or row.get("policy_key")
                        or ""
                    )
                )
                if section == "facts":
                    stable_key = subject_key or normalize_whitespace(str(row.get("id") or ""))
                elif section == "preferences":
                    stable_key = normalize_whitespace(
                        str(row.get("preference_key") or subject_key or row.get("label") or row.get("id") or "")
                    )
                elif section == "policies":
                    stable_key = normalize_whitespace(
                        str(row.get("policy_key") or subject_key or row.get("label") or row.get("id") or "")
                    )
                else:
                    stable_key = normalize_whitespace(str(row.get("id") or row.get("label") or ""))
                text = normalize_whitespace(
                    str(
                        row.get("content")
                        or row.get("summary")
                        or row.get("title")
                        or row.get("label")
                        or row.get("value")
                        or ""
                    )
                )
                text_key = normalize_text(text)
                dedupe_key = stable_key or text_key
                if dedupe_key and dedupe_key in seen_keys:
                    continue
                if text_key and text_key in seen_texts:
                    continue
                merged.append(row)
                if dedupe_key:
                    seen_keys.add(dedupe_key)
                if text_key:
                    seen_texts.add(text_key)
                if len(merged) >= max(int(limit), 0):
                    return merged
        return merged

    def _global_prefetch_results(
        self,
        *,
        scope: str,
        limit: int,
        include_inactive: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        empty_results = {name: [] for name in MemoryStore.SEARCH_SCOPES}
        if not self._store:
            return empty_results
        candidate_limit = max(int(limit), int(self._cfg()["prefetch_limit"]), 8)
        recent = self._decorate_search_results(
            self._store.search("", scope=scope, limit=candidate_limit, include_inactive=include_inactive)
        )
        snapshot_rows = self._store.prompt_snapshot_rows(
            user_limit=max(candidate_limit, 10),
            memory_limit=max(candidate_limit * 2, 14),
            preference_limit=max(candidate_limit, 8),
            policy_limit=max(candidate_limit, 8),
        )
        snapshot = self._decorate_search_results(
            {
                "facts": list(snapshot_rows.get("user_facts", [])) + list(snapshot_rows.get("memory_facts", [])),
                "topics": [],
                "episodes": [],
                "summaries": [],
                "journals": [],
                "preferences": list(snapshot_rows.get("preferences", [])),
                "policies": list(snapshot_rows.get("policies", [])),
            }
        )
        merged = {name: list(recent.get(name, [])) for name in MemoryStore.SEARCH_SCOPES}
        if scope in {"all", "facts"}:
            merged["facts"] = self._merge_prefetch_rows(
                "facts",
                list(snapshot.get("facts", [])),
                list(recent.get("facts", [])),
                limit=self._section_limit("facts", limit),
            )
        if scope in {"all", "preferences"}:
            merged["preferences"] = self._merge_prefetch_rows(
                "preferences",
                list(snapshot.get("preferences", [])),
                list(recent.get("preferences", [])),
                limit=self._section_limit("preferences", limit),
            )
        if scope in {"all", "policies"}:
            merged["policies"] = self._merge_prefetch_rows(
                "policies",
                list(snapshot.get("policies", [])),
                list(recent.get("policies", [])),
                limit=self._section_limit("policies", limit),
            )
        if scope in {"all", "summaries"}:
            merged["summaries"] = list(recent.get("summaries", []))[: self._section_limit("summaries", limit)]
        if scope in {"all", "journals"}:
            merged["journals"] = list(recent.get("journals", []))[: self._section_limit("journals", limit)]
        if scope in {"all", "topics"}:
            merged["topics"] = list(recent.get("topics", []))[: self._section_limit("topics", limit)]
        if scope in {"all", "episodes"}:
            merged["episodes"] = list(recent.get("episodes", []))[: self._section_limit("episodes", limit)]
        return merged

    def _current_snapshot_entries(self, *, max_items: int = 8) -> List[Dict[str, Any]]:
        combined = self._build_builtin_snapshot_entries()
        entries = list(combined.get("user", [])) + list(combined.get("memory", []))
        selected = self._select_snapshot_entries(entries, limit_chars=max(1600, max_items * 140))
        return selected[:max_items]

    def _snapshot_entry_for_subject(
        self,
        subject_key: str,
        *,
        snapshot_entries: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any] | None:
        clean_subject = normalize_whitespace(subject_key)
        if not clean_subject:
            return None
        pool = list(snapshot_entries or [])
        for entry in pool:
            if normalize_whitespace(str(entry.get("subject_key") or "")) == clean_subject:
                return dict(entry)
        if not self._store:
            return None
        fallback = self._decorate_search_results(
            self._store.search(clean_subject, scope="all", limit=8, include_inactive=False)
        )
        for section in ("facts", "preferences", "policies"):
            for row in fallback.get(section, []):
                metadata = self._result_metadata(row)
                row_subject = normalize_whitespace(
                    str(
                        row.get("subject_key")
                        or metadata.get("subject_key")
                        or row.get("preference_key")
                        or row.get("policy_key")
                        or ""
                    )
                )
                if row_subject != clean_subject:
                    continue
                text = normalize_whitespace(str(row.get("content") or row.get("label") or row.get("value") or ""))
                if not text:
                    continue
                return {
                    "text": text,
                    "subject_key": clean_subject,
                    "importance": int(row.get("importance") or 0),
                    "salience": float(row.get("salience") or 0.0),
                    "updated_at": float(row.get("updated_at") or row.get("created_at") or 0.0),
                }
        return None

    def _current_subject_snapshot_entries(
        self,
        subject_keys: List[str] | tuple[str, ...],
        *,
        max_items: int | None = None,
    ) -> List[Dict[str, Any]]:
        combined = self._build_builtin_snapshot_entries()
        pool = list(combined.get("user", [])) + list(combined.get("memory", []))
        entries: List[Dict[str, Any]] = []
        seen_subjects: set[str] = set()
        for subject_key in subject_keys:
            clean_subject = normalize_whitespace(subject_key)
            if not clean_subject or clean_subject in seen_subjects:
                continue
            entry = self._snapshot_entry_for_subject(clean_subject, snapshot_entries=pool)
            if not entry:
                continue
            entries.append(entry)
            seen_subjects.add(clean_subject)
            if max_items is not None and len(entries) >= max_items:
                break
        return entries

    def _mode_snapshot_entries(self, mode: str, *, max_items: int = 8) -> List[Dict[str, Any]]:
        clean_mode = normalize_whitespace(mode)
        if clean_mode == "workflow":
            return self._current_subject_snapshot_entries(
                list(WORKFLOW_SNAPSHOT_SUBJECTS),
                max_items=min(max_items, len(WORKFLOW_SNAPSHOT_SUBJECTS)),
            )
        if clean_mode == "summary":
            selected = self._current_subject_snapshot_entries(
                list(SUMMARY_SNAPSHOT_SUBJECTS),
                max_items=min(max_items, len(SUMMARY_SNAPSHOT_SUBJECTS)),
            )
            if len(selected) >= max_items:
                return selected[:max_items]
            supplement = self._current_snapshot_entries(max_items=max_items * 2)
            seen_subjects = {
                normalize_whitespace(str(entry.get("subject_key") or ""))
                for entry in selected
                if normalize_whitespace(str(entry.get("subject_key") or ""))
            }
            seen_texts = {
                normalize_text(str(entry.get("text") or ""))
                for entry in selected
                if normalize_text(str(entry.get("text") or ""))
            }
            for entry in supplement:
                clean_subject = normalize_whitespace(str(entry.get("subject_key") or ""))
                clean_text = normalize_text(str(entry.get("text") or ""))
                if (clean_subject and clean_subject in seen_subjects) or (clean_text and clean_text in seen_texts):
                    continue
                selected.append(entry)
                if clean_subject:
                    seen_subjects.add(clean_subject)
                if clean_text:
                    seen_texts.add(clean_text)
                if len(selected) >= max_items:
                    break
            return selected[:max_items]
        return self._current_snapshot_entries(max_items=max_items)

    def _subject_provenance_entries(
        self,
        *,
        subject_key: str,
        facts: List[Dict[str, Any]] | None = None,
        limit: int = 4,
        query: str = "",
    ) -> List[Dict[str, Any]]:
        clean_subject = normalize_whitespace(subject_key)
        if not clean_subject or not self._store:
            return []
        entries: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        def push(
            *,
            content: str,
            source_label: str = "",
            source_session_id: str = "",
            action: str = "",
            source: str = "",
            created_at: Any = 0.0,
            turn_id: str = "",
            current: bool = False,
        ) -> None:
            clean_content = normalize_whitespace(content)
            clean_label = normalize_whitespace(source_label)
            clean_session = normalize_whitespace(source_session_id)
            clean_turn = normalize_whitespace(turn_id)
            key = (clean_label, clean_session, clean_content)
            if key in seen:
                return
            if not clean_label and not clean_session and not clean_turn:
                return
            seen.add(key)
            entries.append(
                {
                    "subject_key": clean_subject,
                    "content": clean_content,
                    "source_label": clean_label,
                    "source_session_id": clean_session,
                    "turn_id": clean_turn,
                    "action": normalize_whitespace(action),
                    "source": normalize_whitespace(source),
                    "created_at": float(created_at or 0.0),
                    "current": bool(current),
                }
            )

        for fact in facts or []:
            if normalize_whitespace(str(fact.get("subject_key") or "")) != clean_subject:
                continue
            metadata = self._result_metadata(fact)
            push(
                content=str(fact.get("content") or ""),
                source_label=str(fact.get("source_label") or metadata.get("source_label") or ""),
                source_session_id=str(fact.get("source_session_id") or metadata.get("source_session_id") or ""),
                action="current",
                source=str(fact.get("source") or ""),
                created_at=fact.get("updated_at") or fact.get("created_at") or 0.0,
                turn_id=str(fact.get("turn_id") or metadata.get("turn_id") or ""),
                current=True,
            )
            if len(entries) >= limit:
                return entries[:limit]

        history_rows = self._store.list_history(memory_type="fact", subject_key=clean_subject, limit=max(limit * 4, 10))
        for row in history_rows:
            payload = self._json_dict(row.get("payload"))
            if not payload:
                payload = self._json_dict(row.get("payload_json"))
            metadata = self._json_dict(payload.get("metadata"))
            if not metadata:
                metadata = self._json_dict(payload.get("metadata_json"))
            sensitivity = str(payload.get("sensitivity") or "normal")
            if sensitivity == "normal":
                sensitivity, _ = self._classify_sensitivity(str(payload.get("content") or ""), metadata)
            if sensitivity != "normal" and not self._query_allows_sensitive(query, sensitivity):
                continue
            push(
                content=str(payload.get("content") or ""),
                source_label=str(metadata.get("source_label") or ""),
                source_session_id=str(payload.get("source_session_id") or metadata.get("source_session_id") or ""),
                action=str(row.get("action") or ""),
                source=str(row.get("source") or ""),
                created_at=row.get("created_at") or 0.0,
                turn_id=str(metadata.get("turn_id") or ""),
            )
            if len(entries) >= limit:
                break
        return entries[:limit]

    def _infer_subject_key_from_query(self, query: str) -> str:
        clean = normalize_text(query)
        if not clean:
            return ""
        checks = (
            (("timezone", "local morning", "clock zone"), "user:timezone"),
            (("shell", "terminal environment"), "environment:shell"),
            (("primary database", "main datastore", "database", "datastore"), "project:database"),
            (("deployment path", "deploy", "deployment", "orchestration path", "release"), "project:deploy_method"),
            (("test command", "run the tests", "test invocation", "tests in one command"), "project:test_command"),
            (("docker commands", "container commands", "sudo"), "workflow:docker_sudo"),
        )
        for markers, subject_key in checks:
            if any(marker in clean for marker in markers):
                return subject_key
        return ""

    def _infer_recall_mode(self, *, query: str, args: Dict[str, Any] | None = None) -> str:
        clean = normalize_text(query)
        if not clean:
            return "general"
        provenance_markers = (
            "provenance",
            "source label",
            "source update",
            "source batch",
            "source session",
            "source of",
            "which update batch label",
            "where did",
            "why do we know",
            "captured in",
        )
        history_markers = (
            "previous",
            "older value",
            "before the correction",
            "before the current",
            "used to be",
            "changed over time",
            "history",
            "immediately previous",
            "prior value",
        )
        summary_markers = (
            "summary",
            "snapshot",
            "keep in mind",
            "overview",
            "profile",
            "recap",
            "synthesis",
        )
        workflow_markers = (
            "checklist",
            "runbook",
            "operating checklist",
        )
        if any(marker in clean for marker in provenance_markers):
            return "provenance"
        if any(marker in clean for marker in history_markers):
            return "history"
        if any(marker in clean for marker in summary_markers):
            return "summary"
        if any(marker in clean for marker in workflow_markers):
            return "workflow"
        return "current_state"

    def _build_retrieval_cues(self, *, query: str, args: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        subject_key = normalize_whitespace(str(args.get("subject_key") or ""))
        category = normalize_whitespace(str(args.get("category") or ""))
        topic = normalize_whitespace(str(args.get("topic") or ""))
        if query and not subject_key:
            subject_key = self._infer_subject_key_from_query(query)
        return {
            "query": normalize_whitespace(query),
            "session_id": normalize_whitespace(session_id),
            "subject_key": subject_key,
            "category": category,
            "topic": slugify(topic) if topic else "",
            "mode": self._infer_recall_mode(query=query, args=args),
        }

    def _cue_bonus(self, section: str, row: Dict[str, Any], cues: Dict[str, Any]) -> float:
        bonus = 0.0
        session_cue = normalize_whitespace(str(cues.get("session_id") or ""))
        topic_cue = normalize_whitespace(str(cues.get("topic") or ""))
        category_cue = normalize_whitespace(str(cues.get("category") or ""))
        subject_key_cue = normalize_whitespace(str(cues.get("subject_key") or ""))
        row_session = normalize_whitespace(str(row.get("source_session_id") or row.get("session_id") or ""))
        if session_cue and row_session and row_session == session_cue:
            bonus += 0.22
        row_topic = slugify(str(row.get("topic") or row.get("slug") or ""))
        if topic_cue and row_topic and row_topic == topic_cue:
            bonus += 0.16
        row_category = normalize_whitespace(str(row.get("category") or ""))
        if category_cue and row_category and row_category == category_cue:
            bonus += 0.08
        row_subject = normalize_whitespace(
            str(row.get("subject_key") or row.get("preference_key") or row.get("policy_key") or "")
        )
        if subject_key_cue and row_subject and row_subject == subject_key_cue:
            bonus += 0.2
        return bonus

    def _section_mode_adjustment(self, section: str, row: Dict[str, Any], cues: Dict[str, Any]) -> float:
        mode = str(cues.get("mode") or "")
        if not mode:
            return 0.0
        base: Dict[str, Dict[str, float]] = {
            "current_state": {
                "facts": 0.16,
                "preferences": 0.14,
                "policies": 0.14,
                "topics": 0.04,
                "summaries": -0.14,
                "journals": -0.18,
                "episodes": -0.2,
            },
            "summary": {
                "topics": 0.14,
                "facts": 0.12,
                "preferences": 0.1,
                "policies": 0.1,
                "summaries": -0.04,
                "journals": -0.1,
                "episodes": -0.12,
            },
            "workflow": {
                "topics": 0.12,
                "facts": 0.12,
                "preferences": 0.12,
                "policies": 0.12,
                "summaries": -0.04,
                "journals": -0.1,
                "episodes": -0.12,
            },
            "history": {
                "facts": 0.1,
                "summaries": 0.08,
                "journals": 0.04,
                "topics": 0.04,
            },
            "provenance": {
                "facts": 0.12,
                "summaries": 0.08,
                "journals": 0.04,
                "topics": 0.05,
            },
        }
        adjustment = float(base.get(mode, {}).get(section, 0.0))
        if mode == "current_state":
            if section == "facts" and int(row.get("exclusive") or 0) == 1:
                adjustment += 0.04
            if section == "summaries" and str(row.get("summary_type") or "") in {"session", "handoff"}:
                adjustment -= 0.06
        if mode in {"summary", "workflow"} and section == "facts" and int(row.get("exclusive") or 0) == 1:
            adjustment += 0.02
        return adjustment

    def _filter_results_for_mode(
        self, results: Dict[str, List[Dict[str, Any]]], cues: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        mode = str(cues.get("mode") or "")
        filtered = {section: list(rows) for section, rows in results.items()}
        has_direct = any(filtered.get(section) for section in ("facts", "preferences", "policies"))
        if mode == "current_state":
            if has_direct:
                for section in ("topics", "summaries", "journals", "episodes"):
                    filtered[section] = []
            else:
                for section in ("summaries", "journals", "episodes"):
                    filtered[section] = []
        elif mode in {"summary", "workflow"}:
            for section in ("journals", "episodes"):
                filtered[section] = []
            if mode == "workflow":
                filtered["topics"] = []
                filtered["summaries"] = []
            elif filtered.get("topics") or has_direct:
                filtered["summaries"] = []
                if self._mode_snapshot_entries("summary", max_items=6):
                    filtered["topics"] = []
        return filtered

    def _review_prompt(self, section: str, row: Dict[str, Any]) -> str:
        if section == "facts":
            subject_key = normalize_whitespace(str(row.get("subject_key") or ""))
            if subject_key:
                return f"What is the current memory for `{subject_key}`?"
            topic = pretty_topic(str(row.get("topic") or "memory"))
            return f"What key fact should we remember about {topic}?"
        if section == "summaries":
            return f"What is the current summary for {str(row.get('label') or 'this session')}?"
        if section == "journals":
            return f"What note matters from {str(row.get('label') or 'this journal entry')}?"
        if section == "preferences":
            return (
                f"What preference do we hold for {str(row.get('label') or row.get('preference_key') or 'this item')}?"
            )
        if section == "policies":
            return f"What policy should guide {str(row.get('label') or row.get('policy_key') or 'this workflow')}?"
        return f"What should we recall about {section}?"

    def _memory_text(self, section: str, row: Dict[str, Any]) -> str:
        if section == "topics":
            return f"{row.get('title', '')} {row.get('summary', '')}".strip()
        if section == "summaries":
            return f"{row.get('label', '')} {row.get('summary', '')}".strip()
        if section == "journals":
            return f"{row.get('label', '')} {row.get('content', '')}".strip()
        if section == "preferences":
            return f"{row.get('label', '')} {row.get('value', '')} {row.get('content', '')}".strip()
        if section == "policies":
            return f"{row.get('label', '')} {row.get('content', '')}".strip()
        if section == "episodes":
            return f"{row.get('digest', '')} {row.get('topic_hint', '')}".strip()
        return f"{row.get('content', '')} {row.get('topic', '')}".strip()

    def _cosine_similarity(self, left: List[float], right: List[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for left_value, right_value in zip(left, right):
            numerator += float(left_value) * float(right_value)
            left_norm += float(left_value) * float(left_value)
            right_norm += float(right_value) * float(right_value)
        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0
        return numerator / ((left_norm**0.5) * (right_norm**0.5))

    def _search_memory(
        self,
        query: str,
        *,
        scope: str,
        limit: int,
        session_id: str,
        include_inactive: bool = False,
        cues: Dict[str, Any] | None = None,
        touch_recall: bool = True,
        allow_embeddings: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        if not self._store:
            return {}
        clean = normalize_whitespace(query)
        cue_map = dict(cues or {})
        if session_id and not cue_map.get("session_id"):
            cue_map["session_id"] = normalize_whitespace(session_id)
        candidate_limit = (
            self._cfg()["embedding_candidate_limit"] if self._effective_retrieval_backend() == "hybrid" else limit
        )
        results = self._store.search(clean, scope=scope, limit=int(candidate_limit), include_inactive=include_inactive)
        results = self._decorate_search_results(results)
        results["facts"] = [
            row
            for row in results.get("facts", [])
            if str(row.get("sensitivity") or "normal") == "normal"
            or self._query_allows_sensitive(clean, str(row.get("sensitivity") or ""))
        ]
        results["journals"] = [
            row
            for row in results.get("journals", [])
            if str(row.get("sensitivity") or "normal") == "normal"
            or self._query_allows_sensitive(clean, str(row.get("sensitivity") or ""))
        ]
        results["summaries"] = [
            row
            for row in results.get("summaries", [])
            if str(row.get("sensitivity") or "normal") == "normal"
            or self._query_allows_sensitive(clean, str(row.get("sensitivity") or ""))
        ]
        for sensitive_section in ("topics", "episodes", "preferences", "policies"):
            results[sensitive_section] = [
                row
                for row in results.get(sensitive_section, [])
                if str(row.get("sensitivity") or "normal") == "normal"
                or self._query_allows_sensitive(clean, str(row.get("sensitivity") or ""))
            ]
        if str(cue_map.get("subject_key") or "") and not results.get("facts"):
            subject_results = self._store.search(
                str(cue_map.get("subject_key") or ""),
                scope="facts",
                limit=max(int(candidate_limit), 3),
                include_inactive=include_inactive,
            )
            subject_facts = self._decorate_search_results(subject_results).get("facts", [])
            results["facts"] = [
                row
                for row in subject_facts
                if str(row.get("sensitivity") or "normal") == "normal"
                or self._query_allows_sensitive(clean, str(row.get("sensitivity") or ""))
            ][: self._section_limit("facts", limit)]
        if (
            str(cue_map.get("mode") or "") in {"current_state", "provenance"}
            and not str(cue_map.get("subject_key") or "")
            and not any(results.get(section) for section in MemoryStore.SEARCH_SCOPES)
        ):
            results = self._global_prefetch_results(
                scope=scope,
                limit=limit,
                include_inactive=include_inactive,
            )
        results["facts"] = [
            row
            for row in results.get("facts", [])
            if str(row.get("sensitivity") or "normal") == "normal"
            or self._query_allows_sensitive(clean, str(row.get("sensitivity") or ""))
        ]
        results["journals"] = [
            row
            for row in results.get("journals", [])
            if str(row.get("sensitivity") or "normal") == "normal"
            or self._query_allows_sensitive(clean, str(row.get("sensitivity") or ""))
        ]
        results["summaries"] = [
            row
            for row in results.get("summaries", [])
            if str(row.get("sensitivity") or "normal") == "normal"
            or self._query_allows_sensitive(clean, str(row.get("sensitivity") or ""))
        ]
        for sensitive_section in ("topics", "episodes", "preferences", "policies"):
            results[sensitive_section] = [
                row
                for row in results.get(sensitive_section, [])
                if str(row.get("sensitivity") or "normal") == "normal"
                or self._query_allows_sensitive(clean, str(row.get("sensitivity") or ""))
            ]
        embedding_sensitivities = [self._classify_sensitivity(clean)[0]]
        local_only_result = False
        for section, rows in results.items():
            for row in rows:
                if _as_bool(self._result_metadata(row).get("local_only")):
                    local_only_result = True
                embedding_sensitivities.append(str(row.get("sensitivity") or "normal"))
                embedding_sensitivities.append(
                    self._classify_sensitivity(self._memory_text(section, row), self._result_metadata(row))[0]
                )
        remote_embedding_allowed = not local_only_result and self._remote_processing_allowed(*embedding_sensitivities)
        if (
            clean
            and allow_embeddings
            and remote_embedding_allowed
            and self._effective_retrieval_backend() == "hybrid"
            and self._embedder
            and self._embedder.supports_embeddings
        ):
            query_vector = self._embedder.embed_texts([clean])
            if query_vector:
                query_embedding = query_vector[0]
                for section, rows in results.items():
                    if not rows:
                        continue
                    texts = [self._memory_text(section, row) for row in rows]
                    vectors = self._embedder.embed_texts(texts)
                    if not vectors or len(vectors) != len(rows):
                        logger.debug(
                            "Hybrid retrieval: embedding mismatch for section %s (%s vectors vs %s rows), falling back to FTS scoring",
                            section,
                            len(vectors) if vectors else 0,
                            len(rows),
                        )
                        continue
                    scored: List[Dict[str, Any]] = []
                    for index, (row, vector) in enumerate(zip(rows, vectors)):
                        similarity = self._cosine_similarity(query_embedding, vector)
                        salience = float(row.get("salience") or 0.4)
                        importance = float(row.get("importance") or 5) / 10.0
                        updated_at = float(row.get("updated_at") or row.get("created_at") or 0)
                        age_days = max((time.time() - updated_at) / 86400.0, 0.0) if updated_at > 0 else 365.0
                        recency = 1.0 / (1.0 + age_days / 7.0)
                        rank_prior = max(0.0, 1.0 - (index / max(len(rows), 1)))
                        cue_bonus = self._cue_bonus(section, row, cue_map)
                        mode_adjustment = self._section_mode_adjustment(section, row, cue_map)
                        score = (
                            (0.5 * similarity)
                            + (0.2 * salience)
                            + (0.1 * importance)
                            + (0.08 * recency)
                            + (0.07 * rank_prior)
                            + cue_bonus
                            + mode_adjustment
                        )
                        item = dict(row)
                        item["hybrid_score"] = round(score, 5)
                        item["cue_match_score"] = round(cue_bonus, 5)
                        item["mode_adjustment_score"] = round(mode_adjustment, 5)
                        scored.append(item)
                    scored.sort(key=lambda item: float(item.get("hybrid_score") or 0.0), reverse=True)
                    results[section] = scored[: self._section_limit(section, limit)]
        else:
            for section, rows in list(results.items()):
                scored: List[Dict[str, Any]] = []
                for index, row in enumerate(rows):
                    salience = float(row.get("salience") or 0.4)
                    importance = float(row.get("importance") or 5) / 10.0
                    updated_at = float(row.get("updated_at") or row.get("created_at") or 0)
                    age_days = max((time.time() - updated_at) / 86400.0, 0.0) if updated_at > 0 else 365.0
                    recency = 1.0 / (1.0 + age_days / 7.0)
                    rank_prior = max(0.0, 1.0 - (index / max(len(rows), 1)))
                    cue_bonus = self._cue_bonus(section, row, cue_map)
                    mode_adjustment = self._section_mode_adjustment(section, row, cue_map)
                    score = (
                        (0.38 * rank_prior)
                        + (0.3 * salience)
                        + (0.16 * importance)
                        + (0.16 * recency)
                        + cue_bonus
                        + mode_adjustment
                    )
                    item = dict(row)
                    item["lexical_score"] = round(score, 5)
                    item["cue_match_score"] = round(cue_bonus, 5)
                    item["mode_adjustment_score"] = round(mode_adjustment, 5)
                    scored.append(item)
                scored.sort(key=lambda item: float(item.get("lexical_score") or 0.0), reverse=True)
                results[section] = scored[: self._section_limit(section, limit)]
        if scope == "all":
            results = self._filter_results_for_mode(results, cue_map)
        direct_fact_ids = [int(row["id"]) for row in results.get("facts", []) if row.get("id") is not None]
        if direct_fact_ids:
            associated = self._store.associated_facts(direct_fact_ids, limit=max(2, min(limit, 6)))
            seen_fact_ids = set(direct_fact_ids)
            for row in associated:
                if str(row.get("sensitivity") or "normal") != "normal" and not self._query_allows_sensitive(
                    clean, str(row.get("sensitivity") or "")
                ):
                    continue
                fact_id = int(row.get("id") or 0)
                if fact_id <= 0 or fact_id in seen_fact_ids:
                    continue
                row["retrieval_reason"] = "associative_pattern_completion"
                results.setdefault("facts", []).append(row)
                seen_fact_ids.add(fact_id)
                if len(results["facts"]) >= self._section_limit("facts", limit):
                    break
        results["working"] = (
            self._visible_sensitive_rows(self._store.list_working_memory(session_id, limit=4), clean)
            if session_id
            else []
        )
        results["intentions"] = self._visible_sensitive_rows(self._store.intentions_for_context(clean, limit=4), clean)
        results["procedures"] = (
            self._visible_sensitive_rows(self._store.list_procedures(clean, limit=3), clean) if clean else []
        )
        results["timeline"] = (
            [
                row
                for row in self._store.list_autobiographical_events(clean, limit=3)
                if str(row.get("sensitivity") or "normal") == "normal"
                or self._query_allows_sensitive(clean, str(row.get("sensitivity") or ""))
            ]
            if clean
            else []
        )
        if touch_recall and clean:
            self._store.touch_recall_batch(
                results,
                session_id=session_id,
                review_intervals_days=self._review_intervals_days(),
                reconsolidation_window_hours=float(self._cfg()["reconsolidation_window_hours"]),
                cues=cue_map,
            )
        return results

    # Subject key prefixes that warrant a preference record (behavioral directives,
    # response style, explicit likes/dislikes, favorites).  Other user:* facts
    # (physical attributes, schedule, financials, etc.) stay as facts only.
    _PREFERENCE_WORTHY_PREFIXES = (
        "user:preference:",
        "user:favorite:",
        "user:response_style",
        "user:response_tone",
        "user:answer_format",
        "user:vibe",
        "user:diet",
        "user:allergy:",
        "user:pronouns",
    )

    def _candidate_to_preference(self, candidate: Dict[str, Any], fact: Dict[str, Any]) -> None:
        if not self._store:
            return
        if str(fact.get("sensitivity") or "normal") != "normal":
            return
        metadata = dict(candidate.get("metadata") or fact.get("metadata") or {})
        subject_key = str(metadata.get("subject_key") or "")
        if not subject_key.startswith("user:"):
            return
        # Only promote to preference if it's a behavioral/preference pattern,
        # not every user:* fact (avoids duplication of profile facts).
        if not any(subject_key.startswith(p) for p in self._PREFERENCE_WORTHY_PREFIXES):
            return
        key = subject_key or slugify(str(metadata.get("item_label") or fact.get("content") or "")[:48])
        # ── Build distinct label / value / content fields ──
        # value = the short, concrete datum (e.g. "coffee", "Paris", "light")
        value = str(
            metadata.get("value_label")
            or metadata.get("item_label")
            or metadata.get("trait_label")
            or metadata.get("location_label")
            or metadata.get("origin_label")
            or metadata.get("hometown_label")
            or metadata.get("diet_label")
            or metadata.get("relationship_label")
            or metadata.get("pronouns_label")
            or metadata.get("name_label")
            or metadata.get("pet_name")
            or metadata.get("hobby_label")
            or metadata.get("height_label")
            or metadata.get("weight_label")
            or metadata.get("eye_color_label")
            or metadata.get("hair_label")
            or metadata.get("dob_label")
            or metadata.get("value_key")
            or ""
        )
        # label = short human-readable description (distinct from full content)
        fact_content = str(fact.get("content") or key)
        label = fact_content
        # content = full sentence for context injection
        content = fact_content
        if not value or value == fact_content:
            value = fact_content
        preference = self._store.upsert_preference(
            key=key,
            label=label,
            value=value,
            content=content,
            metadata={
                **metadata,
                **({"session_id": str(fact.get("source_session_id") or "")} if fact.get("source_session_id") else {}),
            },
            importance=int(fact.get("importance") or 6),
            salience=float(fact.get("salience") or 0.7),
            reason="fact_extract",
        )
        if fact.get("id") is not None:
            self._store.add_link("preference", preference["id"], "fact", fact["id"], "supports")
        if fact.get("source_session_id"):
            self._store.add_link("preference", preference["id"], "session", fact["source_session_id"], "captured_in")

    def _classify_sensitivity(self, content: str, metadata: Dict[str, Any] | None = None) -> tuple[str, str]:
        meta = dict(metadata or {})
        subject = normalize_text(str(meta.get("subject_key") or ""))
        text = normalize_text(content)
        combined = f"{subject} {text}"
        if _looks_like_credential(content) or re.search(
            r"\b(password|passphrase|api[_ -]?key|access[_ -]?token|private[_ -]?key|secret)\b", combined
        ):
            return "credential", "credential or secret material"
        if any(token in combined for token in ("health", "medical", "diagnosis", "medication", "surgery", "allerg")):
            return "health", "health information"
        if any(token in combined for token in ("financial", "bank", "iban", "credit card", "salary", "income", "debt")):
            return "financial", "financial information"
        if any(token in combined for token in ("date of birth", "dob", "passport", "social security", "national id")):
            return "identity", "identity information"
        if any(token in subject for token in ("address", "exact_location", "home_location")):
            return "location", "precise location"
        return "normal", ""

    def _remote_processing_allowed(self, *sensitivities: str) -> bool:
        kinds = {normalize_text(value) or "normal" for value in sensitivities}
        if kinds <= {"normal"}:
            return True
        if not self._cfg()["allow_sensitive_model_processing"]:
            return False
        return "credential" not in kinds or self._cfg()["allow_credential_memory"]

    def _category_is_blocked(self, category: str) -> bool:
        blocked = {
            normalize_text(value)
            for value in str(self._config.get("never_remember_categories") or "").split(",")
            if normalize_text(value)
        }
        clean_category = normalize_text(category)
        if clean_category in blocked:
            return True
        if not self._store:
            return False
        for policy in self._store.list_policies(limit=200):
            key = normalize_text(str(policy.get("policy_key") or ""))
            if key in {f"never_remember:{clean_category}", f"never-remember-{clean_category}"}:
                return True
        return False

    def _admit_candidate(
        self,
        candidate: Dict[str, Any],
        *,
        source: str,
        session_id: str,
        approved: bool = False,
    ) -> Dict[str, Any]:
        category = str(candidate.get("category") or "general")
        if self._category_is_blocked(category):
            return {
                "decision": "denied",
                "reason": f"category {category!r} is blocked by policy",
                "sensitivity": "normal",
            }
        sensitivity, reason = self._classify_sensitivity(
            str(candidate.get("content") or ""), dict(candidate.get("metadata") or {})
        )
        if sensitivity == "credential" and not self._cfg()["allow_credential_memory"]:
            return {
                "decision": "denied",
                "reason": "credential memory is disabled; set allow_credential_memory only for an intentional exception",
                "sensitivity": sensitivity,
            }
        if sensitivity == "normal" or approved or self._cfg()["sensitive_memory"] == "allow":
            return {"decision": "allowed", "reason": reason, "sensitivity": sensitivity}
        if self._cfg()["sensitive_memory"] == "deny":
            return {"decision": "denied", "reason": reason, "sensitivity": sensitivity}
        if not self._store:
            return {"decision": "denied", "reason": "store unavailable", "sensitivity": sensitivity}
        approval = self._store.request_approval(
            candidate={**candidate, "_source": source, "_session_id": session_id},
            sensitivity=sensitivity,
            reason=reason,
            session_id=session_id,
        )
        return {"decision": "pending", "reason": reason, "sensitivity": sensitivity, "approval": approval}

    def _store_candidate(
        self,
        candidate: Dict[str, Any],
        *,
        source: str,
        session_id: str,
        observed_at: float | None = None,
        approved: bool = False,
    ) -> Dict[str, Any]:
        if not self._store:
            return {}
        self._invalidate_prefetch_cache()
        admission = self._admit_candidate(candidate, source=source, session_id=session_id, approved=approved)
        if admission["decision"] != "allowed":
            return {"action": admission["decision"], "fact": {}, **admission}
        metadata = dict(candidate.get("metadata") or {})
        source_role = str(metadata.get("source_role") or candidate.get("source_role") or "unknown")
        result = self._store.upsert_fact(
            content=str(candidate["content"]),
            category=str(candidate["category"]),
            topic=str(candidate["topic"]),
            source=source,
            importance=int(candidate["importance"]),
            confidence=float(candidate["confidence"]),
            metadata=metadata,
            observed_at=observed_at,
            source_session_id=session_id,
            history_reason=source,
            source_role=source_role,
            explicit_correction=bool(metadata.get("explicit_correction")),
            sensitivity=str(admission.get("sensitivity") or "normal"),
            memory_class="autobiographical" if str(metadata.get("kind") or "") == "life_event" else "semantic",
            pinned=bool(candidate.get("pinned") or metadata.get("pinned")),
        )
        self._candidate_to_preference(candidate, dict(result.get("fact") or {}))
        fact = dict(result.get("fact") or {})
        if str(metadata.get("kind") or "") == "life_event" and fact.get("id") is not None:
            event = self._store.upsert_autobiographical_event(
                event_key=f"fact-{fact['id']}",
                content=str(candidate.get("content") or ""),
                event_at=float(observed_at or time.time()),
                importance=int(candidate.get("importance") or 6),
                metadata={"fact_id": fact["id"], "session_id": session_id},
                sensitivity=str(admission.get("sensitivity") or "normal"),
            )
            self._store.add_link("fact", fact["id"], "autobiographical_event", event["id"], "represented_by")
        return result

    def _remember_from_tool(self, args: Dict[str, Any], *, session_id: str) -> Dict[str, Any]:
        if not self._store:
            return {}
        memory_type = str(args.get("memory_type") or "fact").strip().lower()
        content = str(args.get("content") or "").strip()
        importance = int(args.get("importance") or 6)
        if memory_type == "preference":
            value = str(args.get("value") or content or "").strip()
            if not value:
                raise ValueError("value or content is required for remember memory_type=preference")
            candidate = {
                "content": content or value,
                "category": "user_pref",
                "topic": str(args.get("topic") or "preferences"),
                "importance": importance,
                "confidence": 0.9,
                "metadata": {"subject_key": str(args.get("subject_key") or ""), "source_role": "user"},
                "_memory_type": "preference",
                "_tool_args": dict(args),
            }
            admission = self._admit_candidate(
                candidate, source="tool", session_id=session_id, approved=_flag(args.get("approved"))
            )
            if admission["decision"] != "allowed":
                return {"memory_type": "preference", "action": admission["decision"], **admission}
            result = self._store.upsert_preference(
                key=str(args.get("key") or args.get("subject_key") or slugify(str(args.get("label") or value)[:48])),
                label=str(args.get("label") or content or value),
                value=value,
                content=content or value,
                metadata={"subject_key": str(args.get("subject_key") or ""), "session_id": session_id},
                importance=importance,
                salience=0.9,
                reason="tool_remember",
                sensitivity=str(admission.get("sensitivity") or "normal"),
            )
            if session_id:
                self._store.add_link("preference", result["id"], "session", session_id, "captured_in")
            self._sync_builtin_snapshot(reason="tool_remember_preference")
            return {"memory_type": "preference", "entry": result}
        if not content:
            raise ValueError("content is required for remember")
        metadata = {"via_tool": True}
        subject_key = str(args.get("subject_key") or "").strip()
        if subject_key:
            metadata["subject_key"] = subject_key
            metadata["exclusive"] = True
            if args.get("value"):
                metadata["value_key"] = str(args.get("value"))
        category = str(args.get("category") or "general")
        topic = str(args.get("topic") or category)
        candidate = {
            "content": content,
            "category": category,
            "topic": topic,
            "importance": importance,
            "confidence": 0.9,
            "pinned": _flag(args.get("pinned")),
            "metadata": {
                **metadata,
                "source_role": "user",
                "explicit_correction": _flag(args.get("explicit_correction")) or "correction" in content.lower(),
            },
        }
        admission = self._admit_candidate(
            candidate, source="tool", session_id=session_id, approved=_flag(args.get("approved"))
        )
        if admission["decision"] != "allowed":
            return {"memory_type": "fact", "action": admission["decision"], **admission}
        result = self._store.upsert_fact(
            content=content,
            category=category,
            topic=topic,
            source="tool",
            importance=importance,
            confidence=0.9,
            metadata=metadata,
            source_session_id=session_id,
            history_reason="tool_remember",
            source_role="user",
            explicit_correction=_flag(args.get("explicit_correction")) or "correction" in content.lower(),
            sensitivity=str(admission.get("sensitivity") or "normal"),
            pinned=_flag(args.get("pinned")),
        )
        self._candidate_to_preference(
            {
                "content": content,
                "category": category,
                "topic": topic,
                "importance": importance,
                "confidence": 0.9,
                "metadata": metadata,
            },
            dict(result.get("fact") or {}),
        )
        self._store.rebuild_topics(
            max_facts=self._cfg()["max_topic_facts"],
            max_chars=self._cfg()["topic_summary_chars"],
        )
        self._sync_builtin_snapshot(reason="tool_remember_fact")
        return {"memory_type": "fact", **result}

    def _singular_kind(self, section: str) -> str:
        mapping = {
            "facts": "fact",
            "topics": "topic",
            "summaries": "summary",
            "journals": "journal",
            "preferences": "preference",
            "policies": "policy",
            "episodes": "episode",
        }
        return mapping.get(section, section.rstrip("s"))

    def _build_summary_text(
        self,
        *,
        artifacts: Dict[str, Any],
        messages: List[Dict[str, Any]] | None = None,
    ) -> str:
        max_chars = self._cfg()["session_summary_chars"]
        parts: List[str] = []
        facts = [str(item.get("content") or "") for item in artifacts.get("facts", [])[:4] if item.get("content")]
        if facts:
            parts.append("Facts: " + "; ".join(facts))
        journals = [str(item.get("content") or "") for item in artifacts.get("journals", [])[:2] if item.get("content")]
        if journals:
            parts.append("Notes: " + " | ".join(journals))
        traces = [str(item.get("content") or "") for item in artifacts.get("traces", [])[:3] if item.get("content")]
        if traces:
            parts.append("Recent flow: " + " | ".join(traces))
        preferences = [
            str(item.get("content") or item.get("label") or "")
            for item in artifacts.get("preferences", [])[:2]
            if item.get("content") or item.get("label")
        ]
        if preferences:
            parts.append("Preferences: " + " | ".join(preferences))
        policies = [
            str(item.get("content") or item.get("label") or "")
            for item in artifacts.get("policies", [])[:2]
            if item.get("content") or item.get("label")
        ]
        if policies:
            parts.append("Policies: " + " | ".join(policies))
        if messages and not parts:
            snippets = []
            for message in messages[-4:]:
                content = message.get("content", "")
                if isinstance(content, list):
                    content = " ".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
                clean = normalize_whitespace(str(content or ""))
                sensitivity, _ = self._classify_sensitivity(clean)
                sensitive_allowed = self._cfg()["sensitive_memory"] == "allow" and (
                    sensitivity != "credential" or self._cfg()["allow_credential_memory"]
                )
                if sensitivity != "normal" and not sensitive_allowed:
                    continue
                if clean:
                    snippets.append(clean[:160])
            if snippets:
                parts.append("Conversation: " + " | ".join(snippets))
        text = " ".join(part for part in parts if part).strip()
        return text[:max_chars] if text else ""

    def _collect_summary_refs(self, artifacts: Dict[str, Any], *, per_section: int = 4) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for section in ("facts", "journals", "traces", "episodes", "preferences", "policies"):
            for item in artifacts.get(section, [])[:per_section]:
                if item.get("id") is None:
                    continue
                refs.append({"kind": self._singular_kind(section), "id": item["id"]})
        return refs

    def _distill_memory(self, args: Dict[str, Any], *, session_id: str) -> Dict[str, Any]:
        if not self._store:
            return {}
        clean_session = str(args.get("session_id") or session_id or self._session_id).strip()
        label = str(args.get("label") or "Session Summary").strip()
        summary_type = str(args.get("memory_type") or "session").strip() or "session"
        content = str(args.get("content") or "").strip()
        try:
            limit = max(8, min(int(args.get("limit") or 8), 50))
        except (TypeError, ValueError, OverflowError):
            limit = 8
        artifacts = self._store.get_session_artifacts(clean_session, limit=limit) if clean_session else {}
        if not content and clean_session:
            content = self._build_summary_text(artifacts=artifacts)
        if not content:
            query = str(args.get("query") or "").strip()
            search_results = self._search_memory(
                query, scope="all", limit=limit, session_id=clean_session or session_id
            )
            parts = []
            for section in ("summaries", "facts", "journals"):
                texts = [self._memory_text(section, item) for item in search_results.get(section, [])[:3]]
                if texts:
                    parts.append(f"{section}: " + " | ".join(texts))
            content = " ".join(parts)[: self._cfg()["session_summary_chars"]]
            artifacts = search_results
        if not content:
            raise ValueError("Nothing available to distill.")
        distill_candidate = {
            "content": content,
            "category": "general",
            "topic": "summary",
            "importance": max(int(args.get("importance") or 7), 7),
            "confidence": 1.0,
            "metadata": {"source_role": "user"},
            "_memory_type": "distill",
            "_tool_args": dict(args),
        }
        admission = self._admit_candidate(
            distill_candidate,
            source="tool_distill",
            session_id=clean_session or session_id,
            approved=_flag(args.get("approved")),
        )
        if admission["decision"] != "allowed":
            return admission
        refs: List[Dict[str, Any]] = []
        for section in ("facts", "journals", "traces", "summaries", "preferences", "policies", "episodes"):
            for item in artifacts.get(section, [])[:8]:
                if item.get("id") is None:
                    continue
                refs.append({"kind": self._singular_kind(section), "id": item["id"]})
        result = self._store.upsert_summary(
            label=label,
            summary=content,
            session_id=clean_session,
            content=content,
            summary_type=summary_type,
            metadata={"source_session_id": clean_session},
            importance=max(int(args.get("importance") or 7), 7),
            salience=0.7,
            source_refs=refs,
            reason="tool_distill",
            sensitivity=str(admission.get("sensitivity") or "normal"),
        )
        return result

    def _export_compiled_wiki(self, *, reason: str) -> Dict[str, Any]:
        if not self._store:
            return {"status": "uninitialized"}
        export_root = self._wiki_export_dir().resolve()
        if export_root == self._hermes_home.resolve():
            raise ValueError("wiki_export_dir must be a dedicated subdirectory, not HERMES_HOME itself")
        result = export_compiled_wiki(
            self._store,
            export_dir=export_root,
            session_limit=self._cfg()["wiki_export_session_limit"],
            topic_limit=self._cfg()["wiki_export_topic_limit"],
            redact_sensitive=self._cfg()["export_redact_sensitive"],
        )
        result["reason"] = reason
        result["enabled"] = self._cfg()["wiki_export_enabled"]
        now = time.time()
        self._store.set_state("last_wiki_export_at", now)
        self._store.set_state("last_wiki_export_root", result["root"])
        self._store.set_state("last_wiki_export_stats", json.dumps(result, sort_keys=True))
        return result

    def _enqueue(self, kind: str, **payload: Any) -> bool:
        if kind == "sync_turn" and not payload.get("_operation_key"):
            payload["_operation_key"] = uuid.uuid4().hex
        with self._state_lock:
            if not self._accepting_tasks:
                return False
            if kind == "mirror_memory" and self._store:
                self._store.enqueue_operation(kind, self._durable_payload(kind, payload))
                self._queue_metrics["spooled"] += 1
                try:
                    self._task_queue.put_nowait(("drain_durable", {}))
                    self._queue_metrics["enqueued"] += 1
                except queue.Full:
                    pass
                return True
            try:
                self._task_queue.put_nowait((kind, payload))
                self._queue_metrics["enqueued"] += 1
                return True
            except queue.Full:
                if kind == "prefetch":
                    self._queue_metrics["dropped_prefetch"] += 1
                    return False
                if self._store:
                    self._store.enqueue_operation(kind, self._durable_payload(kind, payload))
                    self._queue_metrics["spooled"] += 1
                    return True
                self._queue_metrics["failed"] += 1
                return False

    def _durable_payload(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Remove raw sensitive transcripts before an overflow task reaches SQLite."""
        durable = dict(payload)
        allow_sensitive = self._cfg()["sensitive_memory"] == "allow"
        allow_credentials = self._cfg()["allow_credential_memory"]
        if kind == "sync_turn":
            messages = [item for item in durable.get("messages", []) if isinstance(item, dict)]
            raw = " ".join(
                (
                    str(durable.get("user_content") or ""),
                    str(durable.get("assistant_content") or ""),
                    json.dumps(messages, ensure_ascii=False, default=str),
                )
            )
            sensitivity, _ = self._classify_sensitivity(raw)
            if sensitivity != "normal" and not (allow_sensitive and (sensitivity != "credential" or allow_credentials)):
                durable["user_content"] = "[Sensitive user content omitted from durable queue]"
                durable["assistant_content"] = "[Sensitive assistant content omitted from durable queue]"
                durable["messages"] = []
        elif kind == "extract_messages":
            messages = [item for item in durable.get("messages", []) if isinstance(item, dict)]
            sensitivity, _ = self._classify_sensitivity(json.dumps(messages, ensure_ascii=False, default=str))
            if sensitivity != "normal" and not (allow_sensitive and (sensitivity != "credential" or allow_credentials)):
                durable["messages"] = []
        elif kind in {"mirror_memory", "remember_fact"}:
            sensitivity, _ = self._classify_sensitivity(
                str(durable.get("content") or ""), dict(durable.get("metadata") or {})
            )
            if (sensitivity == "credential" and not allow_credentials) or (
                sensitivity != "normal" and self._cfg()["sensitive_memory"] == "deny"
            ):
                return {"_privacy_denied": True}
        return durable

    def _request_consolidation(self, *, reason: str) -> None:
        with self._state_lock:
            if not self._accepting_tasks or self._consolidation_requested:
                return
            self._consolidation_requested = True
            try:
                self._task_queue.put_nowait(("consolidate", {"reason": reason}))
                self._queue_metrics["enqueued"] += 1
            except queue.Full:
                if self._store:
                    self._store.enqueue_operation("consolidate", {"reason": reason})
                    self._queue_metrics["spooled"] += 1
                else:
                    self._queue_metrics["failed"] += 1
                    self._consolidation_requested = False

    def _spool_queued_tasks(self, *, preserve_sentinel: bool) -> int:
        """Move accepted FIFO work to SQLite without waiting for a queue slot."""
        spooled = 0
        saw_sentinel = False
        while True:
            try:
                item = self._task_queue.get_nowait()
            except queue.Empty:
                break
            try:
                if item is None:
                    saw_sentinel = True
                    continue
                kind, payload = item
                if kind == "prefetch":
                    self._queue_metrics["dropped_prefetch"] += 1
                    continue
                if kind == "drain_durable":
                    continue
                if self._store:
                    self._store.enqueue_operation(kind, self._durable_payload(kind, payload))
                    self._queue_metrics["spooled"] += 1
                    spooled += 1
                else:
                    self._queue_metrics["failed"] += 1
            finally:
                self._task_queue.task_done()
        if preserve_sentinel and saw_sentinel:
            self._task_queue.put_nowait(None)
        return spooled

    def _dispatch_task(self, kind: str, payload: Dict[str, Any]) -> None:
        if _flag(payload.get("_privacy_denied")):
            return
        if kind == "sync_turn":
            self._handle_sync_turn(payload)
        elif kind == "prefetch":
            self._handle_prefetch(payload)
        elif kind == "mirror_memory":
            self._handle_mirror_memory(payload)
        elif kind == "remember_fact":
            self._handle_remember_fact(payload)
        elif kind == "extract_messages":
            self._handle_extract_messages(payload)
        elif kind == "consolidate":
            self._run_consolidation(force=False, reason=str(payload.get("reason") or "auto"))
        elif kind == "drain_durable":
            self._drain_durable_operations(limit=100)
        elif kind == "maintenance":
            if not self._store:
                return
            stats = self._store.maintain(
                episode_retention_hours=float(self._cfg()["episode_body_retention_hours"]),
                trace_retention_days=float(self._cfg()["trace_retention_days"]),
                history_retention_days=float(self._cfg()["history_retention_days"]),
                sensitive_retention_days=float(self._cfg()["sensitive_retention_days"]),
                max_database_mb=float(self._cfg()["max_database_mb"]),
            )
            self._store.set_state("last_maintenance_at", time.time())
            self._store.set_state("last_maintenance_stats", json.dumps(stats, sort_keys=True))
        else:
            raise ValueError(f"Unknown durable memory operation: {kind}")

    def _drain_durable_operations(self, *, limit: int = 100) -> int:
        if not self._store:
            return 0
        drained = 0
        max_attempts = int(self._cfg()["queue_max_attempts"])
        for operation in self._store.claim_operations(limit=limit, max_attempts=max_attempts):
            operation_kind = str(operation.get("operation_type") or "")
            try:
                self._dispatch_task(
                    operation_kind,
                    dict(operation.get("payload") or {}),
                )
            except Exception as exc:
                failed = self._store.fail_operation(
                    int(operation["id"]),
                    str(exc),
                    max_attempts=max_attempts,
                )
                self._queue_metrics["failed"] += 1
                logger.warning(
                    "Durable memory operation %s %s after attempt %s: %s",
                    operation.get("id"),
                    "moved to dead letter" if failed.get("status") == "failed" else "will retry",
                    operation.get("attempts"),
                    exc,
                )
            else:
                self._store.complete_operation(int(operation["id"]))
                drained += 1
            finally:
                if operation_kind == "consolidate":
                    with self._state_lock:
                        self._consolidation_requested = False
        return drained

    def _worker_loop(self) -> None:
        while True:
            try:
                item = self._task_queue.get(timeout=1.0)
            except queue.Empty:
                # A failed durable operation may become eligible after its
                # backoff while Hermes is otherwise idle. Wake it without
                # requiring a new user turn or process restart.
                if self._store and self._store.pending_operation_count():
                    self._drain_durable_operations(limit=10)
                continue
            if item is None:
                while self._store and self._store.pending_operation_count():
                    if self._drain_durable_operations(limit=1000) == 0:
                        break
                self._task_queue.task_done()
                break
            kind, payload = item
            try:
                self._dispatch_task(kind, payload)
                if self._store and self._store.pending_operation_count():
                    self._drain_durable_operations(limit=10)
            except Exception as exc:
                self._queue_metrics["failed"] += 1
                replay_id: int | None = None
                if self._store and kind not in {"prefetch", "drain_durable"}:
                    try:
                        replay_id = self._store.enqueue_operation(kind, self._durable_payload(kind, payload))
                        self._queue_metrics["spooled"] += 1
                    except Exception:
                        logger.exception("Failed to spool memory worker task %s after dispatch failure", kind)
                logger.warning(
                    "Memory worker task %s failed%s: %s",
                    kind,
                    f"; queued durable replay {replay_id}" if replay_id is not None else "",
                    exc,
                )
            finally:
                if kind == "consolidate":
                    with self._state_lock:
                        self._consolidation_requested = False
                self._task_queue.task_done()

        # When shutdown timed out because a model request was still in flight,
        # the caller cannot safely close this connection. The worker owns the
        # final close once it has consumed the sentinel and all ready durable
        # work, preventing a permanent connection/thread leak.
        if self._draining:
            store = self._store
            if store:
                store.close()
            with self._state_lock:
                if self._store is store:
                    self._store = None
            self._stop_event.set()
            self._invalidate_prefetch_cache()

    def _invalidate_prefetch_cache(self, *session_ids: str) -> None:
        with self._prefetch_lock:
            if not session_ids:
                self._prefetch_cache.clear()
                return
            for session_id in session_ids:
                if session_id:
                    self._prefetch_cache.pop(session_id, None)

    def _cache_prefetch(self, session_id: str, query: str, rendered: str) -> None:
        with self._prefetch_lock:
            self._prefetch_cache[session_id] = {
                "query": query,
                "rendered": rendered,
                "created_at": time.time(),
            }
            if len(self._prefetch_cache) > 64:
                oldest = min(
                    self._prefetch_cache,
                    key=lambda key: float(self._prefetch_cache[key].get("created_at") or 0.0),
                )
                self._prefetch_cache.pop(oldest, None)

    def _handle_sync_turn(self, payload: Dict[str, Any]) -> None:
        if not self._store:
            return
        self._invalidate_prefetch_cache()
        session_id = str(payload.get("session_id") or self._session_id)
        original_user_content = str(payload.get("user_content") or "")
        original_assistant_content = str(payload.get("assistant_content") or "")
        capture_user_content = original_user_content
        capture_assistant_content = original_assistant_content
        raw_sensitivity, _ = self._classify_sensitivity(f"{original_user_content} {original_assistant_content}")
        raw_sensitive_allowed = self._cfg()["sensitive_memory"] == "allow" and (
            raw_sensitivity != "credential" or self._cfg()["allow_credential_memory"]
        )
        if raw_sensitivity != "normal" and not raw_sensitive_allowed:
            capture_user_content = "[Sensitive user content omitted from raw episode storage]"
            capture_assistant_content = "[Sensitive assistant content omitted from raw episode storage]"
        episode = self._store.append_episode(
            session_id=session_id,
            user_content=capture_user_content,
            assistant_content=capture_assistant_content,
            sensitivity=raw_sensitivity,
            operation_key=str(payload.get("_operation_key") or ""),
        )
        extracted = 0
        extracted_ids: List[int] = []
        for candidate in self._extract_turn_facts(
            user_content=original_user_content,
            assistant_content=original_assistant_content,
            created_at=float(episode.get("created_at") or time.time()),
        )[:10]:
            result = self._store_candidate(
                candidate,
                source="turn_extract",
                session_id=session_id,
                observed_at=float(episode.get("created_at") or time.time()),
            )
            fact_id = dict(result.get("fact") or {}).get("id")
            if fact_id is not None:
                self._store.add_link("fact", fact_id, "episode", int(episode["id"]), "derived_from_episode")
                extracted += 1
                extracted_ids.append(int(fact_id))
        if len(extracted_ids) > 1:
            self._store.associate_fact_group(extracted_ids, relation="same_turn")
        if extracted:
            self._store.rebuild_topics(
                max_facts=self._cfg()["max_topic_facts"],
                max_chars=self._cfg()["topic_summary_chars"],
            )
        trace_parts = []
        user_content = normalize_whitespace(capture_user_content)
        assistant_content = normalize_whitespace(capture_assistant_content)
        if user_content:
            self._store.set_working_memory(
                session_id=session_id,
                memory_key="current-request",
                content=user_content[:1000],
                priority=9,
                ttl_seconds=6 * 3600,
                capacity=self._cfg()["working_memory_capacity"],
                metadata={"episode_id": episode.get("id"), "kind": "current_request"},
                sensitivity=raw_sensitivity,
            )
        if user_content:
            trace_parts.append(f"user: {user_content}")
        if assistant_content:
            trace_parts.append(f"assistant: {assistant_content[:300]}")
        if trace_parts:
            messages = [item for item in payload.get("messages", []) if isinstance(item, dict)]
            roles = [str(item.get("role") or item.get("type") or "") for item in messages]
            tool_names: List[str] = []
            for message in messages:
                for call in message.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function") if isinstance(call.get("function"), dict) else {}
                    name = str(function.get("name") or call.get("name") or "").strip()
                    if name and name not in tool_names:
                        tool_names.append(name)
            self._store.append_trace(
                session_id=session_id,
                label="Turn Trace",
                content=" | ".join(trace_parts),
                trace_type="turn",
                salience=0.48,
                source_episode_id=int(episode.get("id") or 0),
                metadata={"message_roles": roles, "tool_names": tool_names[:12], "facts_extracted": extracted},
                sensitivity=raw_sensitivity,
            )

    def _handle_prefetch(self, payload: Dict[str, Any]) -> None:
        if not self._store:
            return
        query = normalize_whitespace(str(payload.get("query") or ""))
        session_id = str(payload.get("session_id") or self._session_id)
        if not query:
            return
        cues = self._build_retrieval_cues(query=query, args={}, session_id=session_id)
        results = self._search_memory(
            query,
            scope="all",
            limit=self._cfg()["prefetch_limit"],
            session_id=session_id,
            cues=cues,
            touch_recall=self._write_enabled,
        )
        rendered = self._render_prefetch(query, results, cues=cues)
        self._cache_prefetch(session_id, query, rendered)

    def _handle_mirror_memory(self, payload: Dict[str, Any]) -> None:
        if not self._store:
            return
        self._invalidate_prefetch_cache()
        action = str(payload.get("action") or "")
        target = str(payload.get("target") or "")
        content = str(payload.get("content") or "").strip()
        provenance = dict(payload.get("metadata") or {})
        old_text = normalize_whitespace(str(provenance.get("old_text") or ""))
        session_id = normalize_whitespace(str(provenance.get("session_id") or self._session_id))
        if not content and not old_text:
            return
        candidates = self._mirror_memory_candidates(old_text if action == "remove" and old_text else content)
        if action == "remove":
            removal_texts = [normalize_whitespace(str(candidate.get("content") or "")) for candidate in candidates]
            if not removal_texts:
                removal_texts = [normalize_whitespace(content)]
            for clean in removal_texts:
                if not clean:
                    continue
                self._store.deactivate_matching(clean, limit=10)
                matches = self._store.search(clean, scope="preferences", limit=10).get("preferences", [])
                for row in matches:
                    values = {
                        normalize_whitespace(str(row.get("content") or "")),
                        normalize_whitespace(str(row.get("label") or "")),
                        normalize_whitespace(str(row.get("value") or "")),
                    }
                    if clean not in values or row.get("id") is None:
                        continue
                    self._store.deactivate_memory_item(
                        "preference",
                        int(row["id"]),
                        reason="mirror_memory_remove",
                        source="builtin_memory",
                    )
            self._store.rebuild_topics(
                max_facts=self._cfg()["max_topic_facts"],
                max_chars=self._cfg()["topic_summary_chars"],
            )
            self._sync_builtin_snapshot(reason="mirror_memory_remove")
        else:
            if action == "replace" and old_text:
                self._store.deactivate_matching(old_text, limit=10)
                for old_candidate in self._mirror_memory_candidates(old_text):
                    old_content = normalize_whitespace(str(old_candidate.get("content") or ""))
                    if old_content:
                        self._store.deactivate_matching(old_content, limit=10)
            if not candidates:
                return
            for candidate in candidates:
                effective_target = "user" if target == "user" else "memory"
                metadata = {
                    **dict(candidate.get("metadata") or {}),
                    "target": target,
                    "action": action,
                    "snapshot_target": effective_target,
                    "hermes_write": provenance,
                }
                self._store_candidate(
                    {
                        **candidate,
                        "topic": str(candidate.get("topic") or "hermes-memory"),
                        "metadata": {**metadata, "source_role": "tool"},
                    },
                    source=f"builtin_memory:{effective_target}",
                    session_id=session_id,
                )
            self._store.rebuild_topics(
                max_facts=self._cfg()["max_topic_facts"],
                max_chars=self._cfg()["topic_summary_chars"],
            )
            self._sync_builtin_snapshot(reason="mirror_memory_write")

    def _handle_remember_fact(self, payload: Dict[str, Any]) -> None:
        if not self._store:
            return
        self._invalidate_prefetch_cache()
        source = str(payload.get("source") or "manual")
        self._store_candidate(
            {
                "content": str(payload.get("content") or ""),
                "category": str(payload.get("category") or "general"),
                "topic": str(payload.get("topic") or "general"),
                "importance": int(payload.get("importance") or 5),
                "confidence": float(payload.get("confidence") or 0.7),
                "metadata": {
                    **dict(payload.get("metadata") or {}),
                    "source_role": "assistant" if source == "delegation" else "tool",
                },
            },
            source=source,
            session_id=str(payload.get("session_id") or self._session_id),
        )
        self._store.rebuild_topics(
            max_facts=self._cfg()["max_topic_facts"],
            max_chars=self._cfg()["topic_summary_chars"],
        )
        self._sync_builtin_snapshot(reason=str(payload.get("source") or "remember_fact"))

    def _handle_extract_messages(self, payload: Dict[str, Any]) -> None:
        if not self._store:
            return
        self._invalidate_prefetch_cache()
        session_id = str(payload.get("session_id") or self._session_id)
        messages = list(payload.get("messages") or [])
        source = str(payload.get("source") or "messages")
        inserted_ids: List[int] = []
        for candidate in self._extract_messages_facts(messages):
            result = self._store_candidate(candidate, source=source, session_id=session_id)
            fact_id = dict(result.get("fact") or {}).get("id")
            if fact_id is not None:
                inserted_ids.append(int(fact_id))
        if len(inserted_ids) > 1:
            self._store.associate_fact_group(inserted_ids, relation="same_session_extract")
        self._store.rebuild_topics(
            max_facts=self._cfg()["max_topic_facts"],
            max_chars=self._cfg()["topic_summary_chars"],
        )
        if session_id:
            artifacts = self._store.get_session_artifacts(session_id, limit=8)
            summary = self._build_summary_text(artifacts=artifacts, messages=messages)
            summary_sensitivity = "normal"
            if summary:
                summary_sensitivity, _ = self._classify_sensitivity(summary)
                self._store.upsert_summary(
                    label="Session Summary",
                    summary=summary,
                    session_id=session_id,
                    content=summary,
                    summary_type="session",
                    metadata={"source": source},
                    importance=8,
                    salience=0.72,
                    source_refs=self._collect_summary_refs(artifacts, per_section=4),
                    reason=source,
                    sensitivity=summary_sensitivity,
                )
            # Always close the session at session_end, even without a summary.
            if source == "session_end" or summary:
                self._store.close_memory_session(
                    session_id,
                    summary=summary or "",
                    sensitivity=summary_sensitivity,
                )
        self._sync_builtin_snapshot(reason=f"extract_messages:{source}")

    def _run_consolidation(self, *, force: bool, reason: str) -> Dict[str, Any]:
        if not self._store:
            return {"status": "uninitialized"}
        self._invalidate_prefetch_cache()
        if not self._consolidation_lock.acquire(blocking=False):
            return {"status": "busy"}
        lease_acquired = False
        try:
            lease_acquired = self._store.acquire_lease(
                "consolidation", self._owner_id, ttl_seconds=max(300, self._cfg()["llm_timeout_seconds"] * 4)
            )
            if not lease_acquired:
                return {"status": "busy", "reason": "another process owns the consolidation lease"}
            batches: List[Dict[str, Any]] = []
            result: Dict[str, Any] = {"status": "skipped"}
            for batch_index in range(self._cfg()["consolidation_max_batches"]):
                self._store.acquire_lease(
                    "consolidation",
                    self._owner_id,
                    ttl_seconds=max(300, self._cfg()["llm_timeout_seconds"] * 4),
                )
                result = run_consolidation(
                    self._store,
                    min_hours=self._cfg()["min_hours"],
                    min_sessions=self._cfg()["min_sessions"],
                    max_topic_facts=self._cfg()["max_topic_facts"],
                    topic_summary_chars=self._cfg()["topic_summary_chars"],
                    prune_after_days=self._cfg()["prune_after_days"],
                    session_summary_chars=self._cfg()["session_summary_chars"],
                    episode_retention_hours=float(self._cfg()["episode_body_retention_hours"]),
                    decay_half_life_days=float(self._cfg()["decay_half_life_days"]),
                    decay_min_salience=float(self._cfg()["decay_min_salience"]),
                    episode_batch_size=self._cfg()["consolidation_batch_size"],
                    force=force or batch_index > 0,
                    reason=reason if batch_index == 0 else f"{reason}:backlog",
                )
                batches.append(dict(result))
                if result.get("status") != "completed" or int(result.get("backlog_remaining") or 0) <= 0:
                    break
            result = {**result, "batches_completed": len(batches), "batch_stats": batches}
            result["maintenance"] = self._store.maintain(
                episode_retention_hours=float(self._cfg()["episode_body_retention_hours"]),
                trace_retention_days=float(self._cfg()["trace_retention_days"]),
                history_retention_days=float(self._cfg()["history_retention_days"]),
                sensitive_retention_days=float(self._cfg()["sensitive_retention_days"]),
                max_database_mb=float(self._cfg()["max_database_mb"]),
            )
            if (
                result.get("status") == "completed"
                and self._cfg()["wiki_export_enabled"]
                and self._cfg()["wiki_export_on_consolidate"]
            ):
                try:
                    result["wiki_export"] = self._export_compiled_wiki(reason=f"consolidation:{reason}")
                except Exception as exc:
                    logger.warning("Wiki export failed after consolidation: %s", exc)
                    result["wiki_export"] = {"success": False, "error": str(exc)}
            if result.get("status") == "completed":
                result["builtin_snapshot"] = self._sync_builtin_snapshot(reason=f"consolidation:{reason}")
            return result
        finally:
            if lease_acquired and self._store:
                self._store.release_lease("consolidation", self._owner_id)
            self._consolidation_lock.release()

    def _extract_messages_facts(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        pending_user = ""
        for message in messages:
            role = str(message.get("role") or message.get("type") or "").strip().casefold()
            text = message_content_text(message.get("content", ""))
            if not text:
                continue
            if role in {"user", "human"}:
                if pending_user:
                    candidates.extend(self._extract_turn_facts(user_content=pending_user, assistant_content=""))
                pending_user = text
            elif role in {"assistant", "ai"}:
                candidates.extend(self._extract_turn_facts(user_content=pending_user, assistant_content=text))
                pending_user = ""
        if pending_user:
            candidates.extend(self._extract_turn_facts(user_content=pending_user, assistant_content=""))
        return self._dedupe_candidates(candidates)

    def _extract_turn_facts(
        self,
        *,
        user_content: str,
        assistant_content: str,
        created_at: float | None = None,
    ) -> List[Dict[str, Any]]:
        if not self._llm or not self._llm.enabled:
            return []
        sensitivity = self._classify_sensitivity(f"{user_content} {assistant_content}")[0]
        if not self._remote_processing_allowed(sensitivity):
            return []
        return self._llm_extract_turn_facts(
            user_content=user_content,
            assistant_content=assistant_content,
            created_at=created_at,
        )

    def _llm_extract_turn_facts(
        self,
        *,
        user_content: str,
        assistant_content: str,
        created_at: float | None = None,
    ) -> List[Dict[str, Any]]:
        if not self._llm or not self._llm.enabled:
            return []
        max_chars = self._cfg()["llm_max_input_chars"]
        user_text = normalize_whitespace(user_content)[:max_chars]
        assistant_text = normalize_whitespace(assistant_content)[: max(0, max_chars - len(user_text))]
        system_prompt = (
            "You extract durable long-term memory facts for a personal AI assistant. "
            "Return JSON only, no markdown. "
            "Output schema: "
            '{"facts":[{"content":string,"category":"user_pref|project|environment|workflow|general",'
            '"topic":string,"importance":1-10,"confidence":0-1,"subject_key":string,'
            '"value_key":string,"exclusive":boolean,"polarity":-1|1,'
            '"source_role":"user|assistant"}]}. '
            "Keep facts atomic, durable, and useful across sessions. "
            "ALWAYS assign a subject_key and value_key — never leave them empty. "
            "Canonical subject keys: "
            "user:name, user:date_of_birth, user:age, user:occupation, user:location:current, "
            "user:origin, user:hometown, user:pronouns, user:relationship_status, "
            "user:family:<relation> (father/mother/brother/sister/partner/child), "
            "user:pet:<name>, user:personality:<trait>, user:physical_attributes (value_key=height/weight/eye_color/hair), "
            "user:daily_schedule:<aspect> (wake_up/work_hours/bedtime), "
            "user:hobby:<slug>, user:gaming:current, user:interest:<slug>, "
            "user:condition, user:diet, user:diet_aversion, user:allergy:<slug>, "
            "user:preference:<slug>, user:favorite:<kind>, user:response_style, user:response_tone, "
            "user:answer_format, user:vibe, user:belief:<slug>, "
            "user:financial:<aspect>, user:living_situation, user:language:<lang>, "
            "environment:shell, environment:editor, environment:os, environment:ssh_port, "
            "environment:cpu, environment:ram, environment:gpu, environment:wsl, "
            "workflow:docker_sudo, project:test_command, project:deploy_method, project:database, "
            "project:primary_language, project:cache_backend. "
            "Category rules: personal details, preferences, traits, identity -> user_pref. "
            "Family, social, philosophical -> general. Technical setup -> environment. "
            "Use exclusive=true when a newer fact should replace older values for the same subject. "
            "Extract ALL personal details: family members by name, pets, hobbies, physical traits, "
            "daily routines, food preferences, personality, beliefs, finances. "
            "Never treat an assistant guess or suggestion as a user fact; assistant-supported facts require explicit confirmation. "
            "Drop ephemeral task chatter (current activity, greetings, session meta). "
            "Convert relative dates to absolute dates when possible. "
            "Set source_role to the message that supports each fact. Return at most 10 facts."
        )
        user_prompt = json.dumps(
            {
                "reference_unix_time": created_at or time.time(),
                "user_message": user_text,
                "assistant_message": assistant_text,
            },
            ensure_ascii=False,
        )
        data = self._llm.chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=2000,
        )
        if data is None and self._llm.last_request_succeeded is False:
            state = self._llm.circuit_state
            raise RuntimeError(f"automatic memory extraction failed: {state.get('last_error') or 'model unavailable'}")
        if not data or not isinstance(data.get("facts"), list):
            raise RuntimeError("automatic memory extractor returned an invalid facts payload")
            return []
        facts: List[Dict[str, Any]] = []
        for raw in data.get("facts", [])[:10]:
            if not isinstance(raw, dict):
                continue
            raw_role = str(raw.get("source_role") or "").strip().casefold()
            source_role = raw_role if raw_role in {"user", "assistant"} else ("user" if user_text else "assistant")
            normalized = normalize_candidate_fact(raw, source_role=source_role)
            if normalized:
                facts.append(normalized)
        return self._canonicalize_candidates(self._dedupe_candidates(facts))

    def _dedupe_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for candidate in candidates:
            metadata = dict(candidate.get("metadata") or {})
            key = (
                candidate.get("content", "").lower(),
                metadata.get("subject_key", ""),
                metadata.get("value_key", ""),
                metadata.get("polarity", 1),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _canonicalize_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for candidate in candidates:
            metadata = dict(candidate.get("metadata") or {})
            source_role = str(metadata.get("source_role") or candidate.get("source_role") or "assistant")
            raw = {
                **dict(candidate),
                "subject_key": candidate.get("subject_key", metadata.get("subject_key", "")),
                "value_key": candidate.get("value_key", metadata.get("value_key", "")),
                "exclusive": candidate.get("exclusive", metadata.get("exclusive", False)),
                "polarity": candidate.get("polarity", metadata.get("polarity", 1)),
                "metadata": metadata,
            }
            item = normalize_candidate_fact(raw, source_role=source_role)
            if item:
                normalized.append(item)
        return self._dedupe_candidates(normalized)

    def _render_prefetch(
        self, query: str, results: Dict[str, List[Dict[str, Any]]], *, cues: Dict[str, Any] | None = None
    ) -> str:
        cue_map = dict(cues or {})
        mode = str(cue_map.get("mode") or "current_state")
        snapshot_lines: List[str] = []
        if mode in {"summary", "workflow"}:
            snapshot_lines = [f"- {entry.get('text')}" for entry in self._mode_snapshot_entries(mode, max_items=6)]
        topic_lines = []
        for item in results.get("topics", [])[:2]:
            topic_lines.append(f"- {item.get('title')}: {item.get('summary')}")
        summary_lines = []
        for item in results.get("summaries", [])[:3]:
            summary_lines.append(f"- {item.get('label')}: {item.get('summary')}")

        # ── Collect preference content for dedup against facts ──
        preference_lines = []
        _pref_content_seen: set[str] = set()
        for item in results.get("preferences", [])[:3]:
            pref_text = str(item.get("content") or "").strip().lower()
            _pref_content_seen.add(pref_text)
            preference_lines.append(f"- {item.get('content')}")
        workflow_lines = []
        for item in results.get("policies", [])[:3]:
            workflow_lines.append(f"- {item.get('content')}")
        for fact in results.get("facts", []):
            if str(fact.get("category") or "") == "workflow":
                workflow_lines.append(f"- [{fact['topic']}] {fact['content']}")
            if len(workflow_lines) >= 3:
                break

        fact_lines = []
        for fact in results.get("facts", [])[: self._cfg()["prefetch_limit"]]:
            if str(fact.get("category") or "") == "workflow":
                continue
            # Skip facts already represented in preferences (avoid double-injection)
            fact_text = str(fact.get("content") or "").strip().lower()
            if fact_text in _pref_content_seen:
                continue
            fact_lines.append(f"- [{fact['category']}/{fact['topic']}] {fact['content']}")

        journal_lines = []
        for item in results.get("journals", [])[:2]:
            journal_lines.append(f"- {item.get('label')}: {item.get('content')}")
        working_lines = [f"- {item.get('content')}" for item in results.get("working", [])[:4]]
        intention_lines = [f"- {item.get('intention')}" for item in results.get("intentions", [])[:4]]
        procedure_lines = [
            f"- {item.get('label')}: " + " -> ".join(str(step) for step in item.get("steps", [])[:6])
            for item in results.get("procedures", [])[:3]
        ]
        timeline_lines = [f"- {item.get('content')}" for item in results.get("timeline", [])[:3]]

        contradiction_subjects = set()
        if cue_map.get("subject_key"):
            contradiction_subjects.add(str(cue_map["subject_key"]))
        for fact in results.get("facts", []):
            subject_key = normalize_whitespace(str(fact.get("subject_key") or ""))
            if subject_key:
                contradiction_subjects.add(subject_key)
        contradiction_lines = []
        provenance_lines: List[str] = []
        if mode in {"history", "provenance"} and self._store:
            rows = self._store.recent_contradictions(
                limit=3,
                max_age_days=14,
                subject_keys=sorted(contradiction_subjects) if contradiction_subjects else None,
            )
            rows = self._visible_sensitive_rows(rows, query)
            for row in rows:
                winner = normalize_whitespace(str(row.get("winner_content") or ""))
                loser = normalize_whitespace(str(row.get("loser_content") or ""))
                contradiction_lines.append(f"- {row.get('subject_key')}: {loser} -> {winner}")
        if mode == "provenance":
            subject_keys: List[str] = []
            if cue_map.get("subject_key"):
                subject_keys.append(str(cue_map.get("subject_key") or ""))
            for fact in results.get("facts", []):
                subject_key = normalize_whitespace(str(fact.get("subject_key") or ""))
                if subject_key and subject_key not in subject_keys:
                    subject_keys.append(subject_key)
            for subject_key in subject_keys[:3]:
                for entry in self._subject_provenance_entries(
                    subject_key=subject_key,
                    facts=list(results.get("facts", [])),
                    limit=3,
                    query=query,
                ):
                    label = str(entry.get("source_label") or "")
                    session_text = str(entry.get("source_session_id") or "")
                    turn_text = str(entry.get("turn_id") or "")
                    content = str(entry.get("content") or "")
                    origin = label or session_text or turn_text or "unknown source"
                    detail = f"{subject_key} -> {origin}"
                    if content:
                        detail += f" ({content})"
                    provenance_lines.append(f"- {detail}")
            deduped: List[str] = []
            seen = set()
            for line in provenance_lines:
                if line in seen:
                    continue
                seen.add(line)
                deduped.append(line)
            provenance_lines = deduped[:6]

        if (
            not topic_lines
            and not summary_lines
            and not preference_lines
            and not workflow_lines
            and not fact_lines
            and not journal_lines
            and not contradiction_lines
            and not provenance_lines
            and not snapshot_lines
            and not working_lines
            and not intention_lines
            and not procedure_lines
            and not timeline_lines
        ):
            return ""

        lines = [f"## Consolidating Memory Recall for: {query}"]
        if mode in {"current_state", "summary", "workflow"}:
            lines.append(
                "Current-state guidance: prefer active current memory and avoid mentioning older conflicting values unless the user explicitly asks for history or provenance."
            )
        if mode == "summary":
            lines.append("Do not mention obsolete or superseded values, even as exclusions, contrasts, or examples.")
            lines.append(
                "When you answer, list only the current winner facts and stop. Do not append an exclusions note."
            )
            lines.append(
                "Use the winner snapshot as the source of truth. Do not add replacement history, caveats, or extra workflow items that are not in the snapshot."
            )
        if mode == "workflow":
            lines.append(
                "Use only the current workflow winners below for shell, test command, deploy method, and Docker sudo behavior."
            )
            lines.append("Do not append an obsolete-values note, contrast list, or superseded examples.")
        if mode == "current_state":
            if working_lines:
                lines.append("Active working memory:")
                lines.extend(working_lines)
            if intention_lines:
                lines.append("Due intentions:")
                lines.extend(intention_lines)
            if procedure_lines:
                lines.append("Relevant learned procedures:")
                lines.extend(procedure_lines)
            if timeline_lines:
                lines.append("Relevant timeline events:")
                lines.extend(timeline_lines)
            if preference_lines or workflow_lines:
                lines.append("Active preferences and workflow rules:")
                lines.extend(preference_lines + workflow_lines)
            if fact_lines:
                lines.append("Current direct matches:")
                lines.extend(fact_lines)
            elif topic_lines:
                lines.append("Current topic snapshots:")
                lines.extend(topic_lines)
            return "\n".join(lines)
        if mode in {"summary", "workflow"}:
            if snapshot_lines:
                lines.append("Current workflow winners:" if mode == "workflow" else "Current winner snapshot:")
                lines.extend(snapshot_lines)
            if not snapshot_lines:
                if topic_lines:
                    lines.append("Current topic snapshots:")
                    lines.extend(topic_lines)
                elif summary_lines:
                    lines.append("Relevant summaries:")
                    lines.extend(summary_lines)
                if preference_lines or workflow_lines:
                    lines.append("Active preferences and workflow rules:")
                    lines.extend(preference_lines + workflow_lines)
                if fact_lines:
                    lines.append("Active direct matches:")
                    lines.extend(fact_lines)
            return "\n".join(lines)
        if provenance_lines:
            lines.append("Provenance trail:")
            lines.extend(provenance_lines)
        if working_lines:
            lines.append("Active working memory:")
            lines.extend(working_lines)
        if intention_lines:
            lines.append("Due intentions:")
            lines.extend(intention_lines)
        if procedure_lines:
            lines.append("Relevant learned procedures:")
            lines.extend(procedure_lines)
        if timeline_lines:
            lines.append("Relevant timeline events:")
            lines.extend(timeline_lines)
        if summary_lines:
            lines.append("Relevant summaries:")
            lines.extend(summary_lines)
        if preference_lines or workflow_lines:
            lines.append("Active preferences and workflow rules:")
            lines.extend(preference_lines + workflow_lines)
        if fact_lines:
            lines.append("Direct matches:")
            lines.extend(fact_lines)
        if journal_lines:
            lines.append("Recent journal notes:")
            lines.extend(journal_lines)
        if contradiction_lines:
            lines.append("Changed assumptions:")
            lines.extend(contradiction_lines)
        return "\n".join(lines)


def register(ctx) -> None:
    ctx.register_memory_provider(ConsolidatingLocalMemoryProvider())


ConsolidatingLocalProvider = ConsolidatingLocalMemoryProvider
