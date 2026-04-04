from __future__ import annotations

import json
import logging
import queue
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
from .store import MemoryStore, normalize_whitespace, slugify

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
        "- decay: apply salience decay now."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "remember", "forget", "recent", "contradictions", "status", "consolidate", "journal", "distill", "history", "policy", "decay"],
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
                "key": "decay_min_salience",
                "description": "Minimum salience before low-priority items are deactivated",
                "default": "0.15",
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
            timeout_seconds=int(self._config.get("llm_timeout_seconds", 45)),
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
        if self._retrieval_backend == "hybrid" and (not self._embedder or not self._embedder.enabled):
            retrieval_desc += " (embeddings unavailable, using FTS fallback)"
        return (
            "# Consolidating Memory\n"
            f"Active. {counts['facts']} facts, {counts['topics']} topics, {counts['summaries']} summaries, "
            f"{counts['journals']} journals, {counts['preferences']} preferences, {counts['policies']} policies, "
            f"{counts['episodes']} episode buffers, and {counts['contradictions']} contradiction resolutions logged.\n"
            f"Background consolidation gate: {cfg['min_hours']}h + {cfg['min_sessions']} sessions.\n"
            f"Extractor backend: {backend_desc}.\n"
            f"Retrieval backend: {retrieval_desc}.\n"
            f"Last consolidation: {last_text}.\n"
            "Use consolidating_memory to search, remember, journal, distill, inspect history/policies, or force consolidation."
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
        results = self._search_memory(clean, scope="all", limit=self._cfg()["prefetch_limit"], session_id=key)
        rendered = self._render_prefetch(clean, results)
        with self._prefetch_lock:
            self._prefetch_cache[key] = {"query": clean, "rendered": rendered, "created_at": time.time()}
        return rendered

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
                results = self._search_memory(
                    query,
                    scope=scope,
                    limit=limit,
                    session_id=session_id,
                    include_inactive=include_inactive,
                )
                return json.dumps({"success": True, "action": action, "results": results})

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
                    results = self._search_memory(query, scope=section, limit=limit, session_id=session_id, include_inactive=include_inactive)
                    removed_count = 0
                    for row in results.get(section, []):
                        if row.get("id") is None:
                            continue
                        if self._store.deactivate_memory_item(memory_type, int(row["id"]), reason="tool_forget", source="tool"):
                            removed_count += 1
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
                        "embedding_enabled": bool(self._embedder and self._embedder.enabled),
                        "last_decay_at": self._store.get_state("last_decay_at", ""),
                        "latest_session_summaries": self._store.latest_session_summaries(limit=3),
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
                    return json.dumps({"success": True, "action": action, "result": result})
                if not key:
                    return json.dumps({"success": True, "action": action, "results": self._store.recent_items(limit=limit).get("policies", [])})
                results = self._search_memory(key, scope="policies", limit=limit, session_id=session_id, include_inactive=include_inactive)
                return json.dumps({"success": True, "action": action, "results": results.get("policies", [])})

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
            "decay_min_salience": float(self._config.get("decay_min_salience", 0.15)),
            "llm_timeout_seconds": int(self._config.get("llm_timeout_seconds", 45)),
            "llm_max_input_chars": int(self._config.get("llm_max_input_chars", 4000)),
            "retrieval_backend": str(self._config.get("retrieval_backend", "fts") or "fts").strip().lower(),
            "embedding_timeout_seconds": int(self._config.get("embedding_timeout_seconds", 20)),
            "embedding_candidate_limit": int(self._config.get("embedding_candidate_limit", 16)),
        }

    def _effective_retrieval_backend(self) -> str:
        if self._retrieval_backend == "hybrid" and self._embedder and self._embedder.enabled:
            return "hybrid"
        return "fts"

    def _section_limit(self, section: str, limit: int) -> int:
        return int(limit) if section == "facts" else max(1, min(int(limit), 6))

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
    ) -> Dict[str, List[Dict[str, Any]]]:
        if not self._store:
            return {}
        clean = normalize_whitespace(query)
        candidate_limit = self._cfg()["embedding_candidate_limit"] if self._effective_retrieval_backend() == "hybrid" else limit
        results = self._store.search(clean, scope=scope, limit=int(candidate_limit), include_inactive=include_inactive)
        if clean and self._effective_retrieval_backend() == "hybrid" and self._embedder and self._embedder.enabled:
            query_vector = self._embedder.embed_texts([clean])
            if query_vector:
                query_embedding = query_vector[0]
                for section, rows in results.items():
                    if not rows:
                        continue
                    texts = [self._memory_text(section, row) for row in rows]
                    vectors = self._embedder.embed_texts(texts)
                    if not vectors or len(vectors) != len(rows):
                        continue
                    scored: List[Dict[str, Any]] = []
                    for row, vector in zip(rows, vectors):
                        similarity = self._cosine_similarity(query_embedding, vector)
                        salience = float(row.get("salience") or 0.4)
                        importance = float(row.get("importance") or 5) / 10.0
                        updated_at = float(row.get("updated_at") or row.get("created_at") or time.time())
                        age_days = max((time.time() - updated_at) / 86400.0, 0.0)
                        recency = 1.0 / (1.0 + age_days / 7.0)
                        score = (0.55 * similarity) + (0.25 * salience) + (0.1 * importance) + (0.1 * recency)
                        item = dict(row)
                        item["hybrid_score"] = round(score, 5)
                        scored.append(item)
                    scored.sort(key=lambda item: float(item.get("hybrid_score") or 0.0), reverse=True)
                    results[section] = scored[: self._section_limit(section, limit)]
        else:
            for section, rows in list(results.items()):
                results[section] = rows[: self._section_limit(section, limit)]
        self._store.touch_recall_batch(results, session_id=session_id)
        return results

    def _candidate_to_preference(self, candidate: Dict[str, Any], fact: Dict[str, Any]) -> None:
        if not self._store:
            return
        metadata = dict(candidate.get("metadata") or fact.get("metadata") or {})
        category = str(candidate.get("category") or fact.get("category") or "")
        subject_key = str(metadata.get("subject_key") or "")
        if category != "user_pref" and not subject_key.startswith("user:"):
            return
        key = subject_key or slugify(str(metadata.get("item_label") or fact.get("content") or "")[:48])
        value = str(metadata.get("value_key") or metadata.get("item_label") or fact.get("content") or "")
        preference = self._store.upsert_preference(
            key=key,
            label=str(fact.get("content") or key),
            value=value or str(fact.get("content") or key),
            content=str(fact.get("content") or key),
            metadata={
                **metadata,
                **({"session_id": str(fact.get("source_session_id") or "")} if fact.get("source_session_id") else {}),
            },
            importance=max(int(fact.get("importance") or 6), 7),
            salience=max(float(fact.get("salience") or 0.0), 0.8),
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
        results = self._search_memory(query, scope="all", limit=self._cfg()["prefetch_limit"], session_id=session_id)
        rendered = self._render_prefetch(query, results)
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
        if action == "remove":
            self._store.deactivate_matching(content, limit=10)
        else:
            category = "user_pref" if target == "user" else "workflow"
            topic = "user-profile" if target == "user" else "builtin-memory"
            result = self._store.upsert_fact(
                content=content,
                category=category,
                topic=topic,
                source=f"builtin_memory:{target}",
                importance=8,
                confidence=0.95,
                metadata={"target": target, "action": action},
                source_session_id=self._session_id,
                history_reason="mirror_memory",
            )
            if target == "user":
                self._candidate_to_preference(
                    {
                        "content": content,
                        "category": category,
                        "topic": topic,
                        "importance": 8,
                        "confidence": 0.95,
                        "metadata": {"target": target, "action": action},
                    },
                    dict(result.get("fact") or {}),
                )
            self._store.rebuild_topics(
                max_facts=self._cfg()["max_topic_facts"],
                max_chars=self._cfg()["topic_summary_chars"],
            )

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
                self._store.close_memory_session(session_id, summary=summary)

    def _run_consolidation(self, *, force: bool, reason: str) -> Dict[str, Any]:
        if not self._store:
            return {"status": "uninitialized"}
        if not self._consolidation_lock.acquire(blocking=False):
            return {"status": "busy"}
        try:
            return run_consolidation(
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
            "You extract durable long-term memory facts for a coding agent. "
            "Return JSON only, no markdown. "
            "Output schema: "
            "{\"facts\":[{\"content\":string,\"category\":\"user_pref|project|environment|workflow|general\","
            "\"topic\":string,\"importance\":1-10,\"confidence\":0-1,\"subject_key\":string,"
            "\"value_key\":string,\"exclusive\":boolean,\"polarity\":-1|1}]}. "
            "Keep facts atomic, durable, and useful across sessions. "
            "Prefer canonical subject keys like user:response_style, user:answer_format, user:role, "
            "user:timezone, user:preference:<slug>, user:favorite:<kind>, user:diet, user:origin, "
            "user:hometown, user:location:current, user:pronouns, user:relationship_status, "
            "user:allergy:<slug>, user:event:<slug>, environment:shell, environment:editor, environment:os, environment:ssh_port, "
            "workflow:docker_sudo, project:test_command, project:deploy_method, project:database, "
            "project:cache_backend. "
            "Use exclusive=true when a newer fact should replace older values for the same subject. "
            "Remember user likes/dislikes, favorite things, stable personal details, and notable life events when they are likely to matter later. "
            "Drop ephemeral task chatter. Convert relative dates to absolute dates when possible. "
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
            max_tokens=700,
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
                elif "kubernetes" in lowered or "k8s" in lowered:
                    metadata["value_key"] = "kubernetes"
                    item["content"] = "Project deploys with Kubernetes."
                elif "docker" in lowered:
                    metadata["value_key"] = "docker"
                    item["content"] = "Project deploys with Docker."
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
                item_label = str(metadata.get("item_label") or subject_key.split(":", 2)[-1].replace("-", " "))
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

            if subject_key and not metadata.get("value_key"):
                metadata["value_key"] = value_key
            item["metadata"] = metadata
            normalized.append(item)
        return self._dedupe_candidates(normalized)

    def _render_prefetch(self, query: str, results: Dict[str, List[Dict[str, Any]]]) -> str:
        summary_lines = []
        for item in results.get("summaries", [])[:3]:
            summary_lines.append(f"- {item.get('label')}: {item.get('summary')}")

        preference_lines = []
        for item in results.get("preferences", [])[:3]:
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
            fact_lines.append(f"- [{fact['category']}/{fact['topic']}] {fact['content']}")

        journal_lines = []
        for item in results.get("journals", [])[:2]:
            journal_lines.append(f"- {item.get('label')}: {item.get('content')}")

        if not summary_lines and not preference_lines and not workflow_lines and not fact_lines and not journal_lines:
            return ""

        lines = [f"## Consolidating Memory Recall for: {query}"]
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
        contradiction_lines = []
        for row in self._store.recent_contradictions(limit=2, max_age_days=7) if self._store else []:
            winner = normalize_whitespace(str(row.get("winner_content") or ""))
            loser = normalize_whitespace(str(row.get("loser_content") or ""))
            contradiction_lines.append(f"- {row.get('subject_key')}: {loser} -> {winner}")
        if contradiction_lines:
            lines.append("Changed assumptions:")
            lines.extend(contradiction_lines)
        return "\n".join(lines)


def register(ctx) -> None:
    ctx.register_memory_provider(ConsolidatingLocalMemoryProvider())
