from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    from agent.memory_provider import MemoryProvider
except ModuleNotFoundError:
    class MemoryProvider:  # type: ignore[override]
        pass

from .consolidator import (
    build_candidate,
    build_consolidation_plan,
    extract_candidate_facts_from_messages,
    extract_candidate_facts_from_turn,
    infer_category,
    infer_topic,
    normalize_candidate_fact,
    run_consolidation,
)
from .llm_client import OpenAICompatibleEmbeddings, OpenAICompatibleLLM, env_or_blank, load_hermes_model_defaults
from .store import MemoryStore, normalize_text, normalize_whitespace, pretty_topic, slugify
from .wiki_export import export_compiled_wiki

logger = logging.getLogger(__name__)

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
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "remember", "forget", "recent", "contradictions", "status", "consolidate", "journal", "distill", "history", "policy", "review", "decay", "export"],
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
            "session_id": {"type": "string", "description": "Session identifier for journals, distillation, and recall links."},
            "subject_key": {"type": "string", "description": "Exclusive subject key or history filter."},
            "since_days": {"type": "integer", "description": "History or contradiction age filter in days."},
            "include_inactive": {"type": "boolean", "description": "Whether to include inactive memory items."},
            "key": {"type": "string", "description": "Preference or policy key."},
            "value": {"type": "string", "description": "Preference value."},
            "label": {"type": "string", "description": "Optional label for journals, summaries, preferences, or policies."},
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
        self._config = config or _load_plugin_config()
        self._store: MemoryStore | None = None
        self._llm: OpenAICompatibleLLM | None = None
        self._embedder: OpenAICompatibleEmbeddings | None = None
        self._hermes_home = Path("~/.hermes").expanduser()
        self._llm_backend = "heuristic"
        self._retrieval_backend = "fts"
        self._session_id = ""
        self._task_queue: queue.Queue[tuple[str, Dict[str, Any]] | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._prefetch_cache: Dict[str, Dict[str, Any]] = {}
        self._prefetch_lock = threading.Lock()
        self._consolidation_lock = threading.Lock()
        self._consolidation_requested = False
        self._last_scan_at = 0.0

    @property
    def name(self) -> str:
        return "consolidating_local"

    def is_available(self) -> bool:
        return True

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "db_path",
                "description": "SQLite database path",
                "default": "$HERMES_HOME/consolidating_memory.db",
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
                "key": "extractor_backend",
                "description": "Fact extraction backend",
                "default": "hybrid",
                "choices": ["heuristic", "hybrid", "llm"],
            },
            {
                "key": "llm_model",
                "description": "OpenAI-compatible local model name (blank = Hermes default model)",
                "default": "",
            },
            {
                "key": "llm_base_url",
                "description": "OpenAI-compatible base URL (blank = Hermes default base_url)",
                "default": "",
            },
            {
                "key": "llm_timeout_seconds",
                "description": "Timeout for local LLM extraction calls",
                "default": "45",
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
                "description": "OpenAI-compatible embedding model name (blank = use llm model/default model)",
                "default": "",
            },
            {
                "key": "embedding_base_url",
                "description": "OpenAI-compatible embedding base URL (blank = use llm base_url/default base_url)",
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
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml

            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8") as handle:
                    existing = yaml.safe_load(handle) or {}
            existing.setdefault("plugins", {})
            existing["plugins"][PLUGIN_CONFIG_KEY] = values
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.dump(existing, handle, default_flow_style=False, sort_keys=False)
        except Exception as exc:
            logger.warning("Failed to save config for %s: %s", self.name, exc)

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = Path(str(kwargs.get("hermes_home") or Path("~/.hermes").expanduser()))
        self._hermes_home = hermes_home
        db_path = str(self._config.get("db_path", "$HERMES_HOME/consolidating_memory.db"))
        db_path = db_path.replace("$HERMES_HOME", str(hermes_home))
        defaults = load_hermes_model_defaults(hermes_home)
        llm_model = str(self._config.get("llm_model") or defaults.get("model") or "").strip()
        llm_base_url = str(self._config.get("llm_base_url") or defaults.get("base_url") or "").strip()
        llm_api_key = env_or_blank("CONSOLIDATING_MEMORY_LLM_API_KEY")
        embedding_model = str(self._config.get("embedding_model") or llm_model or defaults.get("model") or "").strip()
        embedding_base_url = str(self._config.get("embedding_base_url") or llm_base_url or defaults.get("base_url") or "").strip()
        embedding_api_key = env_or_blank("CONSOLIDATING_MEMORY_EMBEDDING_API_KEY") or llm_api_key
        self._llm = OpenAICompatibleLLM(
            model=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            timeout_seconds=int(self._config.get("llm_timeout_seconds", 120)),
        )
        self._llm_backend = str(self._config.get("extractor_backend", "hybrid") or "hybrid").strip().lower()
        self._embedder = OpenAICompatibleEmbeddings(
            model=embedding_model,
            base_url=embedding_base_url,
            api_key=embedding_api_key,
            timeout_seconds=int(self._config.get("embedding_timeout_seconds", 20)),
        )
        self._retrieval_backend = str(self._config.get("retrieval_backend", "fts") or "fts").strip().lower()
        self._session_id = session_id
        self._store = MemoryStore(db_path=db_path)
        self._store.ensure_memory_session(session_id, label=session_id)
        self._sync_builtin_snapshot(reason="initialize")
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._worker_loop, name="consolidating-memory", daemon=True)
        self._worker.start()

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        counts = self._store.counts()
        cfg = self._cfg()
        last = self._store.last_consolidation()
        last_text = "never"
        if last:
            last_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(last["finished_at"])))
        backend_desc = self._llm_backend
        if self._llm and self._llm.enabled:
            backend_desc += f" via {self._llm.model}"
        elif self._llm_backend != "heuristic":
            backend_desc += " (LLM unavailable, using fallback)"
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
            if cached and cached.get("query") == clean:
                return str(cached.get("rendered") or "")
        cues = self._build_retrieval_cues(query=clean, args={}, session_id=key)
        results = self._search_memory(clean, scope="all", limit=self._cfg()["prefetch_limit"], session_id=key, cues=cues)
        rendered = self._render_prefetch(clean, results, cues=cues)
        with self._prefetch_lock:
            self._prefetch_cache[key] = {"query": clean, "rendered": rendered, "created_at": time.time()}
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

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        self._enqueue(
            "sync_turn",
            session_id=session_id or self._session_id,
            user_content=user_content or "",
            assistant_content=assistant_content or "",
        )

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        if not self._store:
            return
        cooldown = float(self._cfg()["scan_cooldown_seconds"])
        now = time.time()
        if now - self._last_scan_at < cooldown:
            return
        self._last_scan_at = now
        plan = build_consolidation_plan(
            self._store,
            min_hours=self._cfg()["min_hours"],
            min_sessions=self._cfg()["min_sessions"],
        )
        if plan["should_run"]:
            self._request_consolidation(reason="turn_gate")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._store:
            return
        self._enqueue("extract_messages", session_id=self._session_id, messages=messages or [], source="session_end")
        self._request_consolidation(reason="session_end")

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not self._store:
            return ""
        candidates = extract_candidate_facts_from_messages(messages or [])
        if self._llm_backend != "heuristic":
            candidates = self._extract_messages_facts(messages or [])
        inserted = 0
        source_refs: List[Dict[str, Any]] = []
        for candidate in candidates[:6]:
            result = self._store_candidate(
                candidate,
                source="precompress_extract",
                session_id=self._session_id,
            )
            fact_id = dict(result.get("fact") or {}).get("id")
            if fact_id is not None and len(source_refs) < 3:
                source_refs.append({"kind": "fact", "id": fact_id})
            if result["action"] == "inserted":
                inserted += 1
        if inserted > 0:
            self._store.rebuild_topics(
                max_facts=self._cfg()["max_topic_facts"],
                max_chars=self._cfg()["topic_summary_chars"],
            )
        if not candidates:
            return ""
        summary = "; ".join(str(item["content"]) for item in candidates[:3])
        if self._store:
            artifacts = self._store.get_session_artifacts(self._session_id, limit=8)
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
            )
        return (
            "Memory provider preserved pre-compression signals. Preserve these points in the summary: "
            + summary[:500]
            + ("" if inserted == 0 else f" ({inserted} new durable facts stored)")
        )

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        self._enqueue("mirror_memory", action=action, target=target, content=content)

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
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

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name != TOOL_SCHEMA["name"]:
            raise NotImplementedError(f"{self.name} does not handle tool {tool_name}")
        if not self._store:
            return json.dumps({"success": False, "error": "Provider not initialized."})

        action = str(args.get("action") or "").strip()
        limit = max(1, min(int(args.get("limit") or 8), 50))
        session_id = str(args.get("session_id") or self._session_id).strip()
        include_inactive = bool(args.get("include_inactive") or False)
        valid_scopes = {"all", "facts", "topics", "episodes", "summaries", "journals", "preferences", "policies"}
        valid_memory_types = {"fact", "summary", "journal", "preference", "policy"}

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
                )
                payload: Dict[str, Any] = {"success": True, "action": action, "results": results}
                if str(cues.get("mode") or "") == "provenance" and str(cues.get("subject_key") or ""):
                    payload["provenance"] = self._subject_provenance_entries(
                        subject_key=str(cues.get("subject_key") or ""),
                        facts=list(results.get("facts", [])),
                        limit=max(3, min(limit, 6)),
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
                    removed = self._store.deactivate_memory_item(memory_type, int(fact_id), reason="tool_forget", source="tool")
                    self._store.rebuild_topics(
                        max_facts=self._cfg()["max_topic_facts"],
                        max_chars=self._cfg()["topic_summary_chars"],
                    )
                    if removed:
                        self._sync_builtin_snapshot(reason="tool_forget")
                    return json.dumps({"success": removed, "action": action, "fact_id": int(fact_id), "memory_type": memory_type})
                query = str(args.get("query") or "").strip()
                if not query:
                    return json.dumps({"success": False, "error": "query or fact_id is required for forget"})
                if memory_type == "fact":
                    removed_count = self._store.deactivate_matching(query, limit=limit)
                    self._store.rebuild_topics(
                        max_facts=self._cfg()["max_topic_facts"],
                        max_chars=self._cfg()["topic_summary_chars"],
                    )
                else:
                    section = f"{memory_type}s" if not memory_type.endswith("s") else memory_type
                    if memory_type == "summary":
                        section = "summaries"
                    elif memory_type == "journal":
                        section = "journals"
                    elif memory_type == "preference":
                        section = "preferences"
                    elif memory_type == "policy":
                        section = "policies"
                    results = self._search_memory(
                        query,
                        scope=section,
                        limit=limit,
                        session_id=session_id,
                        include_inactive=include_inactive,
                        touch_recall=False,
                    )
                    removed_count = 0
                    for row in results.get(section, []):
                        if row.get("id") is None:
                            continue
                        if self._store.deactivate_memory_item(memory_type, int(row["id"]), reason="tool_forget", source="tool"):
                            removed_count += 1
                if removed_count > 0:
                    self._sync_builtin_snapshot(reason="tool_forget")
                return json.dumps({"success": True, "action": action, "removed_count": removed_count})

            if action == "recent":
                return json.dumps(
                    {"success": True, "action": action, "results": self._store.recent_items(limit=limit)}
                )

            if action == "contradictions":
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "results": self._store.recent_contradictions(limit=limit, max_age_days=args.get("since_days")),
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
                        "recent_contradictions": self._store.recent_contradictions(limit=3),
                        "extractor_backend": self._llm_backend,
                        "retrieval_backend": self._effective_retrieval_backend(),
                        "llm_model": self._llm.model if self._llm else "",
                        "llm_base_url": self._llm.base_url if self._llm else "",
                        "embedding_model": self._embedder.model if self._embedder else "",
                        "embedding_base_url": self._embedder.base_url if self._embedder else "",
                        "embedding_enabled": bool(self._embedder and self._embedder.supports_embeddings),
                        "last_decay_at": self._store.get_state("last_decay_at", ""),
                        "latest_session_summaries": self._store.latest_session_summaries(limit=3),
                        "review": review_status,
                        "wiki_export": {
                            "enabled": self._cfg()["wiki_export_enabled"],
                            "on_consolidate": self._cfg()["wiki_export_on_consolidate"],
                            "root": str(self._wiki_export_dir()),
                            "last_export_at": self._store.get_state("last_wiki_export_at", ""),
                            "last_export_stats": self._load_state_json("last_wiki_export_stats"),
                        },
                        "builtin_snapshot": self._load_state_json("last_builtin_snapshot_sync"),
                        "config": self._cfg(),
                    }
                )

            if action == "consolidate":
                result = self._run_consolidation(force=True, reason="manual")
                return json.dumps({"success": True, "action": action, "result": result})

            if action == "journal":
                label = str(args.get("label") or "Journal").strip()
                content = str(args.get("content") or "").strip()
                if not content:
                    return json.dumps({"success": False, "error": "content is required for journal"})
                result = self._store.add_journal(
                    label=label,
                    content=content,
                    session_id=session_id,
                    journal_type=str(args.get("memory_type") or "note"),
                    metadata={"session_id": session_id} if session_id else None,
                    importance=int(args.get("importance") or 6),
                    salience=0.62,
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
                return json.dumps({"success": True, "action": action, "results": results})

            if action == "policy":
                key = str(args.get("key") or args.get("query") or "").strip()
                content = str(args.get("content") or "").strip()
                if content:
                    result = self._store.upsert_policy(
                        key=key or slugify(args.get("label") or content[:40]),
                        label=str(args.get("label") or key or "Policy"),
                        content=content,
                        metadata={"session_id": session_id},
                        importance=int(args.get("importance") or 9),
                    )
                    if session_id:
                        self._store.add_link("policy", result["id"], "session", session_id, "captured_in")
                    self._sync_builtin_snapshot(reason="tool_policy")
                    return json.dumps({"success": True, "action": action, "result": result})
                if not key:
                    return json.dumps({"success": True, "action": action, "results": self._store.recent_items(limit=limit).get("policies", [])})
                results = self._search_memory(key, scope="policies", limit=limit, session_id=session_id, include_inactive=include_inactive)
                return json.dumps({"success": True, "action": action, "results": results.get("policies", [])})

            if action == "review":
                scope = str(args.get("scope") or "all").strip()
                if scope not in valid_scopes:
                    return json.dumps({"success": False, "error": f"Unsupported scope: {scope}"})
                if scope not in {"all", "facts", "summaries", "journals", "preferences", "policies"}:
                    return json.dumps({"success": False, "error": f"Scope {scope} is not reviewable"})
                review_scope = scope
                due = self._store.review_due(scope=review_scope, limit=limit)
                cues = self._build_retrieval_cues(query=str(args.get("query") or "").strip(), args=args, session_id=session_id)
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
                result = self._export_compiled_wiki(reason="tool")
                return json.dumps({"success": True, "action": action, "result": result})

            return json.dumps({"success": False, "error": f"Unknown action: {action}"})
        except Exception as exc:
            logger.exception("Tool call failed for %s", self.name)
            return json.dumps({"success": False, "error": str(exc)})

    def shutdown(self) -> None:
        self._stop_event.set()
        self._task_queue.put(None)
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=5.0)
        if self._store:
            self._store.close()

    def _cfg(self) -> Dict[str, Any]:
        return {
            "min_hours": int(self._config.get("min_hours", 24)),
            "min_sessions": int(self._config.get("min_sessions", 5)),
            "scan_cooldown_seconds": int(self._config.get("scan_cooldown_seconds", 600)),
            "prefetch_limit": int(self._config.get("prefetch_limit", 8)),
            "max_topic_facts": int(self._config.get("max_topic_facts", 5)),
            "topic_summary_chars": int(self._config.get("topic_summary_chars", 650)),
            "session_summary_chars": int(self._config.get("session_summary_chars", 900)),
            "prune_after_days": int(self._config.get("prune_after_days", 90)),
            "episode_body_retention_hours": float(self._config.get("episode_body_retention_hours", 24)),
            "decay_half_life_days": float(self._config.get("decay_half_life_days", 90)),
            "reconsolidation_window_hours": float(self._config.get("reconsolidation_window_hours", 6)),
            "review_intervals_days": str(self._config.get("review_intervals_days", "1,3,7,14,30")),
            "decay_min_salience": float(self._config.get("decay_min_salience", 0.15)),
            "builtin_snapshot_sync_enabled": self._cfg_bool("builtin_snapshot_sync_enabled", True),
            "builtin_memory_dir": str(self._config.get("builtin_memory_dir", "$HERMES_HOME/memories")),
            "builtin_snapshot_user_chars": int(self._config.get("builtin_snapshot_user_chars", 1375)),
            "builtin_snapshot_memory_chars": int(self._config.get("builtin_snapshot_memory_chars", 2200)),
            "wiki_export_enabled": self._cfg_bool("wiki_export_enabled", False),
            "wiki_export_dir": str(self._config.get("wiki_export_dir", "$HERMES_HOME/consolidating_memory_wiki")),
            "wiki_export_on_consolidate": self._cfg_bool("wiki_export_on_consolidate", True),
            "wiki_export_session_limit": int(self._config.get("wiki_export_session_limit", 50)),
            "wiki_export_topic_limit": int(self._config.get("wiki_export_topic_limit", 100)),
            "llm_timeout_seconds": int(self._config.get("llm_timeout_seconds", 120)),
            "llm_max_input_chars": int(self._config.get("llm_max_input_chars", 4000)),
            "retrieval_backend": str(self._config.get("retrieval_backend", "fts") or "fts").strip().lower(),
            "embedding_timeout_seconds": int(self._config.get("embedding_timeout_seconds", 20)),
            "embedding_candidate_limit": int(self._config.get("embedding_candidate_limit", 16)),
        }

    def _cfg_bool(self, key: str, default: bool) -> bool:
        raw = self._config.get(key, default)
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def _wiki_export_dir(self) -> Path:
        raw = str(self._cfg()["wiki_export_dir"] or "$HERMES_HOME/consolidating_memory_wiki")
        return Path(raw.replace("$HERMES_HOME", str(self._hermes_home))).expanduser()

    def _builtin_memory_dir(self) -> Path:
        raw = str(self._cfg()["builtin_memory_dir"] or "$HERMES_HOME/memories")
        return Path(raw.replace("$HERMES_HOME", str(self._hermes_home))).expanduser()

    def _builtin_memory_path(self, target: str) -> Path:
        name = "USER.md" if target == "user" else "MEMORY.md"
        return self._builtin_memory_dir() / name

    def _candidate_for_memory_line(self, text: str) -> Dict[str, Any] | None:
        clean = normalize_whitespace(text)
        if not clean:
            return None
        facts: List[Dict[str, Any]] = []
        for raw in extract_candidate_facts_from_messages([{"role": "assistant", "content": clean}]):
            if not isinstance(raw, dict):
                continue
            normalized = normalize_candidate_fact(raw, source_role="assistant")
            if normalized:
                facts.append(normalized)
        canonical = self._canonicalize_candidates(facts)
        if not canonical:
            return None
        target = normalize_text(clean)
        for candidate in canonical:
            if normalize_text(str(candidate.get("content") or "")) == target:
                return candidate
        return canonical[0]

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

    def _looks_like_user_profile_text(self, text: str) -> bool:
        clean = normalize_text(text)
        return clean.startswith(
            (
                "user prefers ",
                "users favorite ",
                "user likes ",
                "user dislikes ",
                "user is allergic to ",
                "user is not allergic to ",
                "user is from ",
                "user grew up in ",
                "user lives in ",
                "user pronouns are ",
                "user is ",
                "users timezone is ",
                "user timezone is ",
            )
        )

    def _sanitize_mirror_memory_text(self, text: str) -> str:
        clean = normalize_whitespace(text)
        if not clean:
            return ""
        clean = re.sub(
            r"^(?:please\s+remember\s+this(?:\s+long-term\s+project\s+fact)?(?:\s+(?:workflow|safety)\s+rule)?\s+for\s+future\s+sessions:|"
            r"correction\s+for\s+future\s+sessions:|"
            r"lower[- ]priority(?:\s+(?:note|preference|personal\s+detail|environment\s+note))?:)\s*",
            "",
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(r"^\[[A-Z0-9-]+\]\s*", "", clean)
        clean = re.sub(
            r"\b(?:please\s+(?:acknowledge|confirm|reply|say)\b.*|reply\s+briefly\b.*|"
            r"acknowledge\s+it\s+briefly\b.*|confirm\s+once\s+you\s+stored\s+it\b.*|"
            r"once\s+you\s+stored\s+it\b.*|after\s+handling\s+it\b.*|once\s+you\s+understand\b.*|"
            r"in\s+one\s+short\s+sentence\b.*|in\s+one\s+sentence\b.*)$",
            "",
            clean,
            flags=re.IGNORECASE,
        )
        clean = normalize_whitespace(clean.strip(" -:"))
        return clean

    def _extract_mirror_memory_candidates(self, content: str) -> List[Dict[str, Any]]:
        clean = self._sanitize_mirror_memory_text(content)
        if not clean:
            return []
        candidates: List[Dict[str, Any]] = []
        for raw in extract_candidate_facts_from_messages([{"role": "user", "content": clean}]):
            if not isinstance(raw, dict):
                continue
            normalized = normalize_candidate_fact(raw, source_role="user")
            if normalized:
                candidates.append(normalized)
        if not candidates:
            return []
        canonical = self._canonicalize_candidates(self._dedupe_candidates(candidates))
        filtered: List[Dict[str, Any]] = []
        for candidate in canonical:
            metadata = dict(candidate.get("metadata") or {})
            subject_key = normalize_whitespace(str(metadata.get("subject_key") or ""))
            if not subject_key:
                continue
            filtered.append(candidate)
        return filtered

    def _mirror_candidate_target(self, candidate: Dict[str, Any], default_target: str) -> str:
        metadata = dict(candidate.get("metadata") or {})
        subject_key = normalize_whitespace(str(metadata.get("subject_key") or ""))
        if subject_key.startswith("user:"):
            return "user"
        if subject_key:
            return "memory"
        return "user" if default_target == "user" else "memory"

    @staticmethod
    def _is_snapshot_worthy(text: str, subject_key: str = "") -> bool:
        """Return False for content that should never appear in MEMORY.md / USER.md."""
        lower = (text or "").lower()
        if len(lower) < 6:
            return False
        # Reject raw conversation fragments (questions, broken grammar, meta-talk).
        if lower.endswith("?") and not any(
            k in lower for k in ("prefer", "schedule", "allerg", "born", "live")
        ):
            return False
        # Reject agent self-description / system status.
        if any(phrase in lower for phrase in (
            "plugin is running",
            "plugin is operational",
            "memory is active",
            "sql db consolidation",
            "system confirms",
            "memory usage is not",
            "hermes prioritizes",
            "hermes avoids",
            "hermes is a",
            "hermes optimizes",
            "agent views its internal",
            "nothing to save",
        )):
            return False
        # Reject subject keys that describe transient system state.
        subj = (subject_key or "").lower()
        if any(subj.startswith(p) for p in (
            "system:", "plugin:", "memory:sql",
            "general:roleplay", "assistant:",
        )):
            return False

        # --- Fix 2: Reject unresolved variable placeholders ---
        clean = (text or "").strip()
        if re.match(
            r'^(?:User|The user)\s+(?:is|lives in|prefers|has)\s+\w+\.\s*$',
            clean, re.IGNORECASE,
        ):
            word = clean.rstrip('.').split()[-1].lower()
            if word in ('status', 'location', 'value', 'unknown', 'none', 'default'):
                return False

        # Reject "User likes <slug>" where the value looks like a DB key
        m = re.match(r'^User\s+(?:likes|dislikes)\s+(\S+)\.\s*$', clean, re.IGNORECASE)
        if m:
            val = m.group(1)
            if '_' in val or val.lower() in (
                'food', 'hobby', 'enjoyment', 'status', 'location',
            ):
                return False

        # --- Fix 3: Ephemeral subject keys that should not persist ---
        _EPHEMERAL_SNAPSHOT_KEYS = {
            'user:mood', 'user:mood:easter', 'user:daily_activity',
            'user:game_phase', 'user:resource:tokens',
            'user:session_pattern', 'user:coding_state',
            'user:log_focus', 'user:file_request',
            'user:next_activity_choice', 'user:hobby_status',
        }
        _EPHEMERAL_SNAPSHOT_PREFIXES = (
            'memory:durable_storage',
            'environment:memory_plugin_status',
            'environment:last_reflection_date',
            'environment:philosophical_log',
            'environment:agent_memory_structure',
            'general:identity_file_location',
        )
        if subj in _EPHEMERAL_SNAPSHOT_KEYS:
            return False
        for prefix in _EPHEMERAL_SNAPSHOT_PREFIXES:
            if subj.startswith(prefix):
                return False

        # --- Fix 5: Reject meta/self-referential content about memory system ---
        meta_patterns = (
            'memory and sql db consolidation',
            'consolidating_memory plugin',
            'durable memory is being stored',
            'dedicated .soul. file structure',
            'memory system is categorized',
            'memory.md and user.md',
        )
        for pattern in meta_patterns:
            if pattern in lower:
                return False

        return True

    # Subject-key prefixes that should share a single snapshot slot.
    # Only the highest-salience entry from each group is kept.
    _SNAPSHOT_SUBJECT_GROUPS: Dict[str, str] = {
        "hardware:gpu": "hardware:rig",
        "hardware:cpu": "hardware:rig",
        "hardware:ram": "hardware:rig",
        "hardware:rig": "hardware:rig",
        "hardware:monitor": "hardware:rig",
        "hardware:storage": "hardware:rig",
        "hardware:motherboard": "hardware:rig",
        "environment:os": "environment:setup",
        "environment:shell": "environment:setup",
        "environment:editor": "environment:setup",
        "environment:ide": "environment:setup",
        "environment:terminal": "environment:setup",
    }

    @classmethod
    def _snapshot_group(cls, subject_key: str) -> str:
        """Return the dedup group for a subject_key, or the key itself."""
        lower = (subject_key or "").lower()
        for prefix, group in cls._SNAPSHOT_SUBJECT_GROUPS.items():
            if lower == prefix or lower.startswith(prefix + "_") or lower.startswith(prefix + ":"):
                return group
        return subject_key

    def _build_builtin_snapshot_entries(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self._store:
            return {"user": [], "memory": []}
        snapshot = self._store.prompt_snapshot_rows()
        entries: Dict[str, List[Dict[str, Any]]] = {"user": [], "memory": []}
        seen_subjects: Dict[str, set[str]] = {"user": set(), "memory": set()}
        seen_groups: Dict[str, set[str]] = {"user": set(), "memory": set()}
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
            if not self._is_snapshot_worthy(clean_text, clean_subject):
                return
            normalized_text = normalize_text(clean_text)
            if normalized_text in seen_texts[target]:
                return
            if clean_subject and clean_subject in seen_subjects[target]:
                return
            # Group dedup: only one entry per hardware/environment group.
            group = self._snapshot_group(clean_subject)
            if group and group != clean_subject and group in seen_groups[target]:
                return
            seen_texts[target].add(normalized_text)
            if clean_subject:
                seen_subjects[target].add(clean_subject)
            if group:
                seen_groups[target].add(group)
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
        target: str,
        raw_line: str,
        *,
        subject_keys: set[str],
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
        candidate = self._candidate_for_memory_line(text)
        metadata = dict(candidate.get("metadata") or {}) if candidate else {}
        subject_key = normalize_whitespace(str(metadata.get("subject_key") or ""))
        if not subject_key:
            subject_key = self._infer_subject_key_from_query(text)
        if not subject_key or subject_key not in subject_keys:
            return False
        candidate_target = "user" if subject_key.startswith("user:") or str((candidate or {}).get("category") or "") == "user_pref" else "memory"
        return candidate_target == target

    def _write_builtin_snapshot_file(self, target: str, entries: List[Dict[str, Any]], *, limit_chars: int) -> Dict[str, Any]:
        path = self._builtin_memory_path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if path.exists():
            try:
                existing = path.read_text(encoding="utf-8")
            except Exception:
                existing = path.read_text(encoding="utf-8", errors="ignore")
        stripped = self._strip_auto_memory_block(existing)
        subject_keys = {normalize_whitespace(str(entry.get("subject_key") or "")) for entry in entries if str(entry.get("subject_key") or "").strip()}
        normalized_contents = {normalize_text(str(entry.get("text") or "")) for entry in entries if str(entry.get("text") or "").strip()}
        preserved_lines = [
            line.rstrip()
            for line in stripped.splitlines()
            if not self._line_should_be_replaced(target, line, subject_keys=subject_keys, normalized_contents=normalized_contents)
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
            path.write_text(normalized, encoding="utf-8")
        return {
            "path": str(path),
            "changed": changed,
            "chars": len(normalized),
            "entries": len(selected),
        }

    def _sync_builtin_snapshot(self, *, reason: str) -> Dict[str, Any]:
        if not self._store or not self._cfg()["builtin_snapshot_sync_enabled"]:
            return {"success": False, "reason": "disabled"}
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
                text = normalize_whitespace(
                    str(row.get("content") or row.get("label") or row.get("value") or "")
                )
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
        if query and not category:
            inferred_category = infer_category(query)
            if inferred_category in {"user_pref", "project", "environment", "workflow"}:
                category = inferred_category
        if query and not topic:
            inferred_topic = infer_topic(query, category or "general", subject_key) or ""
            if inferred_topic and slugify(inferred_topic) not in {"general", "notes", "memory"}:
                topic = inferred_topic
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
        row_subject = normalize_whitespace(str(row.get("subject_key") or row.get("preference_key") or row.get("policy_key") or ""))
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

    def _filter_results_for_mode(self, results: Dict[str, List[Dict[str, Any]]], cues: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
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
            return f"What preference do we hold for {str(row.get('label') or row.get('preference_key') or 'this item')}?"
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
        return numerator / ((left_norm ** 0.5) * (right_norm ** 0.5))

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
    ) -> Dict[str, List[Dict[str, Any]]]:
        if not self._store:
            return {}
        clean = normalize_whitespace(query)
        cue_map = dict(cues or {})
        if session_id and not cue_map.get("session_id"):
            cue_map["session_id"] = normalize_whitespace(session_id)
        candidate_limit = self._cfg()["embedding_candidate_limit"] if self._effective_retrieval_backend() == "hybrid" else limit
        results = self._store.search(clean, scope=scope, limit=int(candidate_limit), include_inactive=include_inactive)
        results = self._decorate_search_results(results)
        if str(cue_map.get("mode") or "") == "provenance" and str(cue_map.get("subject_key") or "") and not results.get("facts"):
            subject_results = self._store.search(
                str(cue_map.get("subject_key") or ""),
                scope="facts",
                limit=max(int(candidate_limit), 3),
                include_inactive=include_inactive,
            )
            results["facts"] = self._decorate_search_results(subject_results).get("facts", [])[: self._section_limit("facts", limit)]
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
        if clean and self._effective_retrieval_backend() == "hybrid" and self._embedder and self._embedder.supports_embeddings:
            query_vector = self._embedder.embed_texts([clean])
            if query_vector:
                query_embedding = query_vector[0]
                for section, rows in results.items():
                    if not rows:
                        continue
                    texts = [self._memory_text(section, row) for row in rows]
                    vectors = self._embedder.embed_texts(texts)
                    if not vectors or len(vectors) != len(rows):
                        logger.debug("Hybrid retrieval: embedding mismatch for section %s (%s vectors vs %s rows), falling back to FTS scoring", section, len(vectors) if vectors else 0, len(rows))
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
                        score = (0.5 * similarity) + (0.2 * salience) + (0.1 * importance) + (0.08 * recency) + (0.07 * rank_prior) + cue_bonus + mode_adjustment
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
                    score = (0.38 * rank_prior) + (0.3 * salience) + (0.16 * importance) + (0.16 * recency) + cue_bonus + mode_adjustment
                    item = dict(row)
                    item["heuristic_score"] = round(score, 5)
                    item["cue_match_score"] = round(cue_bonus, 5)
                    item["mode_adjustment_score"] = round(mode_adjustment, 5)
                    scored.append(item)
                scored.sort(key=lambda item: float(item.get("heuristic_score") or 0.0), reverse=True)
                results[section] = scored[: self._section_limit(section, limit)]
        results = self._filter_results_for_mode(results, cue_map)
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
        "user:preference:", "user:favorite:", "user:response_style",
        "user:response_tone", "user:answer_format", "user:vibe",
        "user:diet", "user:allergy:", "user:pronouns",
    )

    def _candidate_to_preference(self, candidate: Dict[str, Any], fact: Dict[str, Any]) -> None:
        if not self._store:
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
        # value = the short, concrete datum (e.g. "coffee", "Caudry", "light")
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
        # If value is the same as content, try to shorten label to "key: value" form
        if value and value != fact_content:
            # We have a distinct short value — good
            pass
        else:
            # Fallback: extract value from content by stripping common prefixes
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

    def _store_candidate(
        self,
        candidate: Dict[str, Any],
        *,
        source: str,
        session_id: str,
        observed_at: float | None = None,
    ) -> Dict[str, Any]:
        if not self._store:
            return {}
        result = self._store.upsert_fact(
            content=str(candidate["content"]),
            category=str(candidate["category"]),
            topic=str(candidate["topic"]),
            source=source,
            importance=int(candidate["importance"]),
            confidence=float(candidate["confidence"]),
            metadata=dict(candidate.get("metadata") or {}),
            observed_at=observed_at,
            source_session_id=session_id,
            history_reason=source,
        )
        self._candidate_to_preference(candidate, dict(result.get("fact") or {}))
        return result

    def _write_consolidation_fact(self, candidate: Dict[str, Any], episode: Dict[str, Any]) -> Dict[str, Any]:
        result = self._store_candidate(
            candidate,
            source="episode_extract",
            session_id=str(episode.get("session_id") or self._session_id),
            observed_at=float(episode.get("created_at") or time.time()),
        )
        if self._store and dict(result.get("fact") or {}).get("id") is not None and episode.get("id") is not None:
            self._store.add_link("fact", result["fact"]["id"], "episode", int(episode["id"]), "derived_from_episode")
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
            result = self._store.upsert_preference(
                key=str(args.get("key") or args.get("subject_key") or slugify(str(args.get("label") or value)[:48])),
                label=str(args.get("label") or content or value),
                value=value,
                content=content or value,
                metadata={"subject_key": str(args.get("subject_key") or ""), "session_id": session_id},
                importance=importance,
                salience=0.9,
                reason="tool_remember",
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
        preferences = [str(item.get("content") or item.get("label") or "") for item in artifacts.get("preferences", [])[:2] if item.get("content") or item.get("label")]
        if preferences:
            parts.append("Preferences: " + " | ".join(preferences))
        policies = [str(item.get("content") or item.get("label") or "") for item in artifacts.get("policies", [])[:2] if item.get("content") or item.get("label")]
        if policies:
            parts.append("Policies: " + " | ".join(policies))
        if messages and not parts:
            snippets = []
            for message in messages[-4:]:
                content = message.get("content", "")
                if isinstance(content, list):
                    content = " ".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
                clean = normalize_whitespace(str(content or ""))
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
        limit = max(int(args.get("limit") or 8), 8)
        artifacts = self._store.get_session_artifacts(clean_session, limit=limit) if clean_session else {}
        if not content and clean_session:
            content = self._build_summary_text(artifacts=artifacts)
        if not content:
            query = str(args.get("query") or "").strip()
            search_results = self._search_memory(query, scope="all", limit=limit, session_id=clean_session or session_id)
            parts = []
            for section in ("summaries", "facts", "journals"):
                texts = [self._memory_text(section, item) for item in search_results.get(section, [])[:3]]
                if texts:
                    parts.append(f"{section}: " + " | ".join(texts))
            content = " ".join(parts)[: self._cfg()["session_summary_chars"]]
            artifacts = search_results
        if not content:
            raise ValueError("Nothing available to distill.")
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
        )
        return result

    def _export_compiled_wiki(self, *, reason: str) -> Dict[str, Any]:
        if not self._store:
            return {"status": "uninitialized"}
        result = export_compiled_wiki(
            self._store,
            export_dir=self._wiki_export_dir(),
            session_limit=self._cfg()["wiki_export_session_limit"],
            topic_limit=self._cfg()["wiki_export_topic_limit"],
        )
        result["reason"] = reason
        result["enabled"] = self._cfg()["wiki_export_enabled"]
        now = time.time()
        self._store.set_state("last_wiki_export_at", now)
        self._store.set_state("last_wiki_export_root", result["root"])
        self._store.set_state("last_wiki_export_stats", json.dumps(result, sort_keys=True))
        return result

    def _enqueue(self, kind: str, **payload: Any) -> None:
        if self._stop_event.is_set():
            return
        self._task_queue.put((kind, payload))

    def _request_consolidation(self, *, reason: str) -> None:
        if self._consolidation_requested:
            return
        self._consolidation_requested = True
        self._enqueue("consolidate", reason=reason)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            kind, payload = item
            try:
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
            except Exception as exc:
                logger.warning("Memory worker task %s failed: %s", kind, exc)
            finally:
                if kind == "consolidate":
                    self._consolidation_requested = False
                self._task_queue.task_done()

    def _handle_sync_turn(self, payload: Dict[str, Any]) -> None:
        if not self._store:
            return
        session_id = str(payload.get("session_id") or self._session_id)
        episode = self._store.append_episode(
            session_id=session_id,
            user_content=str(payload.get("user_content") or ""),
            assistant_content=str(payload.get("assistant_content") or ""),
        )
        trace_parts = []
        user_content = normalize_whitespace(str(payload.get("user_content") or ""))
        assistant_content = normalize_whitespace(str(payload.get("assistant_content") or ""))
        if user_content:
            trace_parts.append(f"user: {user_content}")
        if assistant_content:
            trace_parts.append(f"assistant: {assistant_content[:300]}")
        if trace_parts:
            self._store.append_trace(
                session_id=session_id,
                label="Turn Trace",
                content=" | ".join(trace_parts),
                trace_type="turn",
                salience=0.48,
                source_episode_id=int(episode.get("id") or 0),
            )

    def _handle_prefetch(self, payload: Dict[str, Any]) -> None:
        if not self._store:
            return
        query = normalize_whitespace(str(payload.get("query") or ""))
        session_id = str(payload.get("session_id") or self._session_id)
        if not query:
            return
        cues = self._build_retrieval_cues(query=query, args={}, session_id=session_id)
        results = self._search_memory(query, scope="all", limit=self._cfg()["prefetch_limit"], session_id=session_id, cues=cues)
        rendered = self._render_prefetch(query, results, cues=cues)
        with self._prefetch_lock:
            self._prefetch_cache[session_id] = {"query": query, "rendered": rendered, "created_at": time.time()}

    def _handle_mirror_memory(self, payload: Dict[str, Any]) -> None:
        if not self._store:
            return
        action = str(payload.get("action") or "")
        target = str(payload.get("target") or "")
        content = str(payload.get("content") or "").strip()
        if not content:
            return
        candidates = self._extract_mirror_memory_candidates(content)
        if action == "remove":
            removal_texts = [normalize_whitespace(str(candidate.get("content") or "")) for candidate in candidates]
            if not removal_texts:
                removal_texts = [normalize_whitespace(self._sanitize_mirror_memory_text(content) or content)]
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
            if not candidates:
                return
            for candidate in candidates:
                effective_target = self._mirror_candidate_target(candidate, target)
                metadata = {
                    **dict(candidate.get("metadata") or {}),
                    "target": target,
                    "action": action,
                    "snapshot_target": effective_target,
                }
                result = self._store.upsert_fact(
                    content=str(candidate.get("content") or ""),
                    category=str(candidate.get("category") or "general"),
                    topic=str(candidate.get("topic") or infer_topic(str(candidate.get("content") or ""), str(candidate.get("category") or "general"))),
                    source=f"builtin_memory:{effective_target}",
                    importance=int(candidate.get("importance") or 8),
                    confidence=float(candidate.get("confidence") or 0.85),
                    metadata=metadata,
                    source_session_id=self._session_id,
                    history_reason="mirror_memory",
                )
                self._candidate_to_preference(
                    {
                        "content": str(candidate.get("content") or ""),
                        "category": str(candidate.get("category") or "general"),
                        "topic": str(candidate.get("topic") or "builtin-memory"),
                        "importance": int(candidate.get("importance") or 8),
                        "confidence": float(candidate.get("confidence") or 0.85),
                        "metadata": metadata,
                    },
                    dict(result.get("fact") or {}),
                )
            self._store.rebuild_topics(
                max_facts=self._cfg()["max_topic_facts"],
                max_chars=self._cfg()["topic_summary_chars"],
            )
            self._sync_builtin_snapshot(reason="mirror_memory_write")

    def _handle_remember_fact(self, payload: Dict[str, Any]) -> None:
        if not self._store:
            return
        result = self._store.upsert_fact(
            content=str(payload.get("content") or ""),
            category=str(payload.get("category") or "general"),
            topic=str(payload.get("topic") or "general"),
            source=str(payload.get("source") or "manual"),
            importance=int(payload.get("importance") or 5),
            confidence=float(payload.get("confidence") or 0.7),
            metadata=dict(payload.get("metadata") or {}),
            source_session_id=str(payload.get("session_id") or self._session_id),
            history_reason=str(payload.get("source") or "manual"),
        )
        self._candidate_to_preference(
            {
                "content": str(payload.get("content") or ""),
                "category": str(payload.get("category") or "general"),
                "topic": str(payload.get("topic") or "general"),
                "importance": int(payload.get("importance") or 5),
                "confidence": float(payload.get("confidence") or 0.7),
                "metadata": dict(payload.get("metadata") or {}),
            },
            dict(result.get("fact") or {}),
        )
        self._store.rebuild_topics(
            max_facts=self._cfg()["max_topic_facts"],
            max_chars=self._cfg()["topic_summary_chars"],
        )
        self._sync_builtin_snapshot(reason=str(payload.get("source") or "remember_fact"))

    def _handle_extract_messages(self, payload: Dict[str, Any]) -> None:
        if not self._store:
            return
        session_id = str(payload.get("session_id") or self._session_id)
        messages = list(payload.get("messages") or [])
        source = str(payload.get("source") or "messages")
        inserted_ids: List[int] = []
        for candidate in self._extract_messages_facts(messages):
            result = self._store_candidate(candidate, source=source, session_id=session_id)
            fact_id = dict(result.get("fact") or {}).get("id")
            if fact_id is not None:
                inserted_ids.append(int(fact_id))
        self._store.rebuild_topics(
            max_facts=self._cfg()["max_topic_facts"],
            max_chars=self._cfg()["topic_summary_chars"],
        )
        if session_id:
            artifacts = self._store.get_session_artifacts(session_id, limit=8)
            summary = self._build_summary_text(artifacts=artifacts, messages=messages)
            if summary:
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
                )
            # Always close the session at session_end, even without a summary.
            if source == "session_end" or summary:
                self._store.close_memory_session(session_id, summary=summary or "")
        self._sync_builtin_snapshot(reason=f"extract_messages:{source}")

    def _run_consolidation(self, *, force: bool, reason: str) -> Dict[str, Any]:
        if not self._store:
            return {"status": "uninitialized"}
        if not self._consolidation_lock.acquire(blocking=False):
            return {"status": "busy"}
        try:
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
                extractor=self._extract_turn_facts,
                fact_writer=self._write_consolidation_fact,
                force=force,
                reason=reason,
            )
            if result.get("status") == "completed" and self._cfg()["wiki_export_enabled"] and self._cfg()["wiki_export_on_consolidate"]:
                try:
                    result["wiki_export"] = self._export_compiled_wiki(reason=f"consolidation:{reason}")
                except Exception as exc:
                    logger.warning("Wiki export failed after consolidation: %s", exc)
                    result["wiki_export"] = {"success": False, "error": str(exc)}
            if result.get("status") == "completed":
                result["builtin_snapshot"] = self._sync_builtin_snapshot(reason=f"consolidation:{reason}")
            return result
        finally:
            self._consolidation_lock.release()

    def _extract_messages_facts(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        if self._llm_backend == "heuristic":
            return extract_candidate_facts_from_messages(messages)
        for message in messages:
            role = str(message.get("role") or message.get("type") or "")
            content = message.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(block.get("text", ""))
                    for block in content
                    if isinstance(block, dict)
                )
            text = str(content or "")
            if not text:
                continue
            source_role = "assistant" if "assistant" in role else "user"
            if source_role == "assistant" and "remember" not in text.lower():
                continue
            candidates.extend(
                self._extract_turn_facts(
                    user_content=text if source_role == "user" else "",
                    assistant_content=text if source_role == "assistant" else "",
                )
            )
        return self._dedupe_candidates(candidates)

    def _extract_turn_facts(
        self,
        *,
        user_content: str,
        assistant_content: str,
        created_at: float | None = None,
    ) -> List[Dict[str, Any]]:
        heuristic = extract_candidate_facts_from_turn(
            user_content=user_content,
            assistant_content=assistant_content,
            created_at=created_at,
        )
        if self._llm_backend == "heuristic" or not self._llm or not self._llm.enabled:
            return self._canonicalize_candidates(heuristic)
        llm_facts = self._llm_extract_turn_facts(
            user_content=user_content,
            assistant_content=assistant_content,
            heuristic=heuristic,
            created_at=created_at,
        )
        if not llm_facts:
            return self._canonicalize_candidates(heuristic)
        if self._llm_backend == "llm":
            return self._canonicalize_candidates(llm_facts)
        return self._canonicalize_candidates(self._dedupe_candidates(heuristic + llm_facts))

    def _llm_extract_turn_facts(
        self,
        *,
        user_content: str,
        assistant_content: str,
        heuristic: List[Dict[str, Any]],
        created_at: float | None = None,
    ) -> List[Dict[str, Any]]:
        if not self._llm or not self._llm.enabled:
            return []
        max_chars = self._cfg()["llm_max_input_chars"]
        user_text = normalize_whitespace(user_content)[:max_chars]
        assistant_text = normalize_whitespace(assistant_content)[:max_chars]
        seed_facts = [
            {
                "content": item["content"],
                "category": item["category"],
                "topic": item["topic"],
                "subject_key": dict(item.get("metadata") or {}).get("subject_key", ""),
                "value_key": dict(item.get("metadata") or {}).get("value_key", ""),
                "exclusive": bool(dict(item.get("metadata") or {}).get("exclusive")),
                "polarity": dict(item.get("metadata") or {}).get("polarity", 1),
            }
            for item in heuristic[:10]
        ]
        system_prompt = (
            "You extract durable long-term memory facts for a personal AI assistant. "
            "Return JSON only, no markdown. "
            "Output schema: "
            "{\"facts\":[{\"content\":string,\"category\":\"user_pref|project|environment|workflow|general\","
            "\"topic\":string,\"importance\":1-10,\"confidence\":0-1,\"subject_key\":string,"
            "\"value_key\":string,\"exclusive\":boolean,\"polarity\":-1|1}]}. "
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
            "Drop ephemeral task chatter (current activity, greetings, session meta). "
            "Convert relative dates to absolute dates when possible. "
            "Return at most 10 facts."
        )
        user_prompt = json.dumps(
            {
                "reference_unix_time": created_at or time.time(),
                "user_message": user_text,
                "assistant_message": assistant_text,
                "heuristic_seed_facts": seed_facts,
            },
            ensure_ascii=True,
        )
        data = self._llm.chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=16384,
        )
        if not data or not isinstance(data.get("facts"), list):
            return []
        facts: List[Dict[str, Any]] = []
        for raw in data.get("facts", [])[:10]:
            if not isinstance(raw, dict):
                continue
            normalized = normalize_candidate_fact(raw, source_role="assistant")
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
            item = dict(candidate)
            metadata = dict(item.get("metadata") or {})
            content = normalize_whitespace(str(item.get("content") or ""))
            subject_key = str(metadata.get("subject_key") or "")
            value_key = str(metadata.get("value_key") or "")
            lowered = content.lower()

            if subject_key == "project:deploy_method":
                if "docker compose" in lowered:
                    metadata["value_key"] = "docker-compose"
                    item["content"] = "Project deploys with Docker Compose."
                elif "nomad" in lowered:
                    metadata["value_key"] = "nomad"
                    item["content"] = "Project deploys with Nomad."
                elif "kubernetes" in lowered or "k8s" in lowered:
                    metadata["value_key"] = "kubernetes"
                    item["content"] = "Project deploys with Kubernetes."
                elif "docker" in lowered:
                    metadata["value_key"] = "docker"
                    item["content"] = "Project deploys with Docker."
            elif subject_key == "project:test_command":
                command_label = str(metadata.get("command_label") or "")
                if "uv run pytest -q" in lowered:
                    metadata["value_key"] = "uv-run-pytest-q"
                    metadata["command_label"] = "uv run pytest -q"
                    item["content"] = "Project tests run with uv run pytest -q."
                elif "python -m unittest -q" in lowered:
                    metadata["value_key"] = "python-m-unittest-q"
                    metadata["command_label"] = "python -m unittest -q"
                    item["content"] = "Project tests run with python -m unittest -q."
                elif "pytest -q" in lowered:
                    metadata["value_key"] = "pytest-q"
                    metadata["command_label"] = "pytest -q"
                    item["content"] = "Project tests run with pytest -q."
                elif command_label:
                    metadata["value_key"] = slugify(command_label)
                    item["content"] = f"Project tests run with {command_label}."
            elif subject_key == "project:database":
                if "postgres" in lowered:
                    metadata["value_key"] = "postgresql"
                    item["content"] = "Primary project database is PostgreSQL."
                elif "mysql" in lowered:
                    metadata["value_key"] = "mysql"
                    item["content"] = "Primary project database is MySQL."
                elif "sqlite" in lowered:
                    metadata["value_key"] = "sqlite"
                    item["content"] = "Primary project database is SQLite."
                elif "redis" in lowered:
                    metadata["value_key"] = "redis"
                    item["content"] = "Primary project database is Redis."
            elif subject_key == "user:response_style":
                if "detailed" in lowered or "verbose" in lowered:
                    metadata["value_key"] = "detailed"
                    item["content"] = "User prefers detailed responses."
                elif "concise" in lowered or "brief" in lowered or "short" in lowered or "terse" in lowered:
                    metadata["value_key"] = "concise"
                    item["content"] = "User prefers concise responses."
            elif subject_key.startswith("user:favorite:"):
                trait = str(metadata.get("trait_label") or subject_key.split(":", 2)[-1].replace("-", " "))
                value = str(metadata.get("value_label") or value_key.replace("-", " "))
                trait = normalize_whitespace(trait)
                value = normalize_whitespace(value)
                if trait and value:
                    metadata["value_key"] = value_key or value.lower().replace(" ", "-")
                    item["content"] = f"User's favorite {trait} is {value}."
            elif subject_key.startswith("user:preference:"):
                item_label = str(metadata.get("item_label") or "")
                if not item_label:
                    # Fallback to slug, but reject if it looks like a DB key
                    fallback = subject_key.split(":", 2)[-1].replace("-", " ")
                    # Skip canonicalization if the fallback is a single generic word
                    if fallback and " " not in fallback and fallback.lower() in (
                        "food", "hobby", "enjoyment", "status", "location",
                        "things", "stuff", "setting", "preference",
                    ):
                        continue
                    item_label = normalize_whitespace(fallback)
                item_label = normalize_whitespace(item_label)
                if item_label:
                    if int(metadata.get("polarity", 1) or 1) < 0 or value_key == "dislike":
                        metadata["value_key"] = "dislike"
                        item["content"] = f"User dislikes {item_label}."
                    else:
                        metadata["value_key"] = "like"
                        item["content"] = f"User likes {item_label}."
            elif subject_key == "user:diet":
                diet = str(metadata.get("diet_label") or value_key.replace("-", " "))
                diet = normalize_whitespace(diet)
                if diet:
                    metadata["value_key"] = value_key or diet.lower().replace(" ", "-")
                    item["content"] = f"User is {diet}."
            elif subject_key == "user:origin":
                origin = str(metadata.get("origin_label") or value_key.replace("-", " "))
                origin = normalize_whitespace(origin)
                if origin:
                    metadata["value_key"] = value_key or origin.lower().replace(" ", "-")
                    item["content"] = f"User is from {origin}."
            elif subject_key == "user:hometown":
                hometown = str(metadata.get("hometown_label") or value_key.replace("-", " "))
                hometown = normalize_whitespace(hometown)
                if hometown:
                    metadata["value_key"] = value_key or hometown.lower().replace(" ", "-")
                    item["content"] = f"User grew up in {hometown}."
            elif subject_key == "user:location:current":
                location = str(metadata.get("location_label") or value_key.replace("-", " "))
                location = normalize_whitespace(location)
                if location:
                    metadata["value_key"] = value_key or location.lower().replace(" ", "-")
                    item["content"] = f"User lives in {location}."
            elif subject_key == "user:pronouns":
                pronouns = str(metadata.get("pronouns_label") or value_key.replace("-", "/"))
                pronouns = normalize_whitespace(pronouns)
                if pronouns:
                    metadata["value_key"] = value_key or pronouns.lower().replace(" ", "-")
                    item["content"] = f"User pronouns are {pronouns}."
            elif subject_key == "user:relationship_status":
                status = str(metadata.get("relationship_label") or value_key.replace("-", " "))
                status = normalize_whitespace(status)
                if status:
                    metadata["value_key"] = value_key or status.lower().replace(" ", "-")
                    item["content"] = f"User is {status}."
            elif subject_key == "user:timezone":
                timezone = str(metadata.get("timezone_label") or value_key.replace("-", " "))
                timezone = normalize_whitespace(timezone)
                if timezone:
                    label = timezone if "/" in timezone else timezone.upper()
                    metadata["timezone_label"] = label
                    metadata["value_key"] = slugify(label)
                    item["content"] = f"User's timezone is {label}."
            elif subject_key.startswith("user:allergy:"):
                item_label = str(metadata.get("item_label") or subject_key.split(":", 2)[-1].replace("-", " "))
                item_label = normalize_whitespace(item_label)
                if item_label:
                    if int(metadata.get("polarity", 1) or 1) < 0 or value_key == "not-allergic":
                        metadata["value_key"] = "not-allergic"
                        item["content"] = f"User is not allergic to {item_label}."
                    else:
                        metadata["value_key"] = "allergic"
                        item["content"] = f"User is allergic to {item_label}."
            elif subject_key == "environment:shell":
                for shell in ("bash", "zsh", "fish", "powershell"):
                    if shell in lowered:
                        metadata["value_key"] = shell
                        item["content"] = f"Environment shell is {shell}."
                        break
            elif subject_key == "workflow:docker_sudo":
                if "do not use sudo" in lowered or "don't use sudo" in lowered or "no sudo" in lowered:
                    metadata["value_key"] = "no-sudo"
                    item["content"] = "Do not use sudo for Docker commands."
                elif "use sudo" in lowered:
                    metadata["value_key"] = "sudo-required"
                    item["content"] = "Use sudo for Docker commands."
            elif subject_key == "workflow:manual_edits":
                metadata["value_key"] = "apply-patch"
                item["content"] = "Use apply_patch for manual file edits."
            elif subject_key == "workflow:git_safety":
                metadata["value_key"] = "avoid-git-reset-hard"
                item["content"] = "Never use git reset --hard."

            if subject_key and not metadata.get("value_key"):
                metadata["value_key"] = value_key
            item["metadata"] = metadata
            normalized.append(item)
        return self._dedupe_candidates(normalized)

    def _render_prefetch(self, query: str, results: Dict[str, List[Dict[str, Any]]], *, cues: Dict[str, Any] | None = None) -> str:
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

        if not topic_lines and not summary_lines and not preference_lines and not workflow_lines and not fact_lines and not journal_lines and not contradiction_lines and not provenance_lines and not snapshot_lines:
            return ""

        lines = [f"## Consolidating Memory Recall for: {query}"]
        if mode in {"current_state", "summary", "workflow"}:
            lines.append("Current-state guidance: prefer active current memory and avoid mentioning older conflicting values unless the user explicitly asks for history or provenance.")
        if mode == "summary":
            lines.append("Do not mention obsolete or superseded values, even as exclusions, contrasts, or examples.")
            lines.append("When you answer, list only the current winner facts and stop. Do not append an exclusions note.")
            lines.append("Use the winner snapshot as the source of truth. Do not add replacement history, caveats, or extra workflow items that are not in the snapshot.")
        if mode == "workflow":
            lines.append("Use only the current workflow winners below for shell, test command, deploy method, and Docker sudo behavior.")
            lines.append("Do not append an obsolete-values note, contrast list, or superseded examples.")
        if mode == "current_state":
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
