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
from .llm_client import OpenAICompatibleLLM, env_or_blank, load_hermes_model_defaults
from .store import MemoryStore, normalize_whitespace

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "consolidating_memory",
    "description": (
        "Local memory with episodic storage, durable facts, "
        "topic summaries, and background consolidation.\n\n"
        "Actions:\n"
        "- search: lookup across durable facts, topics, and past episodes.\n"
        "- remember: store a durable fact directly.\n"
        "- forget: deactivate a fact by id or matching text.\n"
        "- recent: inspect the latest facts/topics/episodes.\n"
        "- contradictions: inspect recently resolved assumption changes.\n"
        "- status: show memory counts and consolidation gates.\n"
        "- consolidate: force a consolidation pass now."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "remember", "forget", "recent", "contradictions", "status", "consolidate"],
            },
            "query": {"type": "string", "description": "Search or forget query."},
            "scope": {
                "type": "string",
                "enum": ["all", "facts", "topics", "episodes"],
                "description": "Search scope for action=search.",
            },
            "limit": {"type": "integer", "description": "Maximum number of results."},
            "content": {"type": "string", "description": "Durable fact to remember."},
            "category": {
                "type": "string",
                "enum": ["user_pref", "project", "environment", "workflow", "general"],
            },
            "topic": {"type": "string", "description": "Topic bucket for remembered content."},
            "importance": {"type": "integer", "description": "Importance score from 1 to 10."},
            "fact_id": {"type": "integer", "description": "Specific fact id to forget."},
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
        self._llm_backend = "heuristic"
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
                "key": "prune_after_days",
                "description": "Age threshold for pruning low-value extracted facts",
                "default": "90",
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
        self._llm = OpenAICompatibleLLM(
            model=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            timeout_seconds=int(self._config.get("llm_timeout_seconds", 45)),
        )
        self._llm_backend = str(self._config.get("extractor_backend", "hybrid") or "hybrid").strip().lower()
        self._session_id = session_id
        self._store = MemoryStore(db_path=db_path)
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
        return (
            "# Consolidating Memory\n"
            f"Active. {counts['facts']} facts, {counts['topics']} topics, {counts['episodes']} episodes stored, "
            f"{counts['contradictions']} contradiction resolutions logged.\n"
            f"Background consolidation gate: {cfg['min_hours']}h + {cfg['min_sessions']} sessions.\n"
            f"Extractor backend: {backend_desc}.\n"
            f"Last consolidation: {last_text}.\n"
            "Use consolidating_memory to search, remember, forget, inspect status, or force consolidation."
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
        results = self._store.search(clean, scope="all", limit=self._cfg()["prefetch_limit"])
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
        self._enqueue("extract_messages", messages=messages or [], source="session_end")
        self._request_consolidation(reason="session_end")

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not self._store:
            return ""
        candidates = extract_candidate_facts_from_messages(messages or [])
        if self._llm_backend != "heuristic":
            candidates = self._extract_messages_facts(messages or [])
        inserted = 0
        for candidate in candidates[:6]:
            result = self._store.upsert_fact(
                content=str(candidate["content"]),
                category=str(candidate["category"]),
                topic=str(candidate["topic"]),
                source="precompress_extract",
                importance=int(candidate["importance"]),
                confidence=float(candidate["confidence"]),
                metadata=dict(candidate.get("metadata") or {}),
            )
            if result["action"] == "inserted":
                inserted += 1
        if not candidates:
            return ""
        summary = "; ".join(str(item["content"]) for item in candidates[:3])
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
        limit = int(args.get("limit") or 8)

        try:
            if action == "search":
                query = str(args.get("query") or "").strip()
                scope = str(args.get("scope") or "all")
                results = self._store.search(query, scope=scope, limit=limit)
                return json.dumps({"success": True, "action": action, "results": results})

            if action == "remember":
                content = str(args.get("content") or "").strip()
                if not content:
                    return json.dumps({"success": False, "error": "content is required for remember"})
                category = str(args.get("category") or "general")
                topic = str(args.get("topic") or category)
                importance = int(args.get("importance") or 6)
                result = self._store.upsert_fact(
                    content=content,
                    category=category,
                    topic=topic,
                    source="tool",
                    importance=importance,
                    confidence=0.9,
                    metadata={"via_tool": True},
                )
                self._store.rebuild_topics(
                    max_facts=self._cfg()["max_topic_facts"],
                    max_chars=self._cfg()["topic_summary_chars"],
                )
                return json.dumps({"success": True, "action": action, "result": result})

            if action == "forget":
                fact_id = args.get("fact_id")
                if fact_id is not None:
                    removed = self._store.deactivate_fact(int(fact_id))
                    self._store.rebuild_topics(
                        max_facts=self._cfg()["max_topic_facts"],
                        max_chars=self._cfg()["topic_summary_chars"],
                    )
                    return json.dumps({"success": removed, "action": action, "fact_id": int(fact_id)})
                query = str(args.get("query") or "").strip()
                removed_count = self._store.deactivate_matching(query, limit=limit)
                self._store.rebuild_topics(
                    max_facts=self._cfg()["max_topic_facts"],
                    max_chars=self._cfg()["topic_summary_chars"],
                )
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
                        "results": self._store.recent_contradictions(limit=limit),
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
                        "llm_model": self._llm.model if self._llm else "",
                        "llm_base_url": self._llm.base_url if self._llm else "",
                        "config": self._cfg(),
                    }
                )

            if action == "consolidate":
                result = self._run_consolidation(force=True, reason="manual")
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

    def _cfg(self) -> Dict[str, int]:
        return {
            "min_hours": int(self._config.get("min_hours", 24)),
            "min_sessions": int(self._config.get("min_sessions", 5)),
            "scan_cooldown_seconds": int(self._config.get("scan_cooldown_seconds", 600)),
            "prefetch_limit": int(self._config.get("prefetch_limit", 8)),
            "max_topic_facts": int(self._config.get("max_topic_facts", 5)),
            "topic_summary_chars": int(self._config.get("topic_summary_chars", 650)),
            "prune_after_days": int(self._config.get("prune_after_days", 90)),
            "llm_timeout_seconds": int(self._config.get("llm_timeout_seconds", 45)),
            "llm_max_input_chars": int(self._config.get("llm_max_input_chars", 4000)),
        }

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
        self._store.append_episode(
            session_id=str(payload.get("session_id") or self._session_id),
            user_content=str(payload.get("user_content") or ""),
            assistant_content=str(payload.get("assistant_content") or ""),
        )

    def _handle_prefetch(self, payload: Dict[str, Any]) -> None:
        if not self._store:
            return
        query = normalize_whitespace(str(payload.get("query") or ""))
        session_id = str(payload.get("session_id") or self._session_id)
        if not query:
            return
        results = self._store.search(query, scope="all", limit=self._cfg()["prefetch_limit"])
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
            self._store.upsert_fact(
                content=content,
                category=category,
                topic=topic,
                source=f"builtin_memory:{target}",
                importance=8,
                confidence=0.95,
                metadata={"target": target, "action": action},
            )
            self._store.rebuild_topics(
                max_facts=self._cfg()["max_topic_facts"],
                max_chars=self._cfg()["topic_summary_chars"],
            )

    def _handle_remember_fact(self, payload: Dict[str, Any]) -> None:
        if not self._store:
            return
        self._store.upsert_fact(
            content=str(payload.get("content") or ""),
            category=str(payload.get("category") or "general"),
            topic=str(payload.get("topic") or "general"),
            source=str(payload.get("source") or "manual"),
            importance=int(payload.get("importance") or 5),
            confidence=float(payload.get("confidence") or 0.7),
            metadata=dict(payload.get("metadata") or {}),
        )

    def _handle_extract_messages(self, payload: Dict[str, Any]) -> None:
        if not self._store:
            return
        messages = list(payload.get("messages") or [])
        source = str(payload.get("source") or "messages")
        for candidate in self._extract_messages_facts(messages):
            self._store.upsert_fact(
                content=str(candidate["content"]),
                category=str(candidate["category"]),
                topic=str(candidate["topic"]),
                source=source,
                importance=int(candidate["importance"]),
                confidence=float(candidate["confidence"]),
                metadata=dict(candidate.get("metadata") or {}),
            )
        self._store.rebuild_topics(
            max_facts=self._cfg()["max_topic_facts"],
            max_chars=self._cfg()["topic_summary_chars"],
        )

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
                extractor=self._extract_turn_facts,
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
            "user:timezone, environment:shell, environment:editor, environment:os, environment:ssh_port, "
            "workflow:docker_sudo, project:test_command, project:deploy_method, project:database, "
            "project:cache_backend. "
            "Use exclusive=true when a newer fact should replace older values for the same subject. "
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
        topic_lines = []
        for topic in results.get("topics", [])[:3]:
            topic_lines.append(f"- {topic['title']}: {topic['summary']}")

        fact_lines = []
        for fact in results.get("facts", [])[: self._cfg()["prefetch_limit"]]:
            fact_lines.append(f"- [{fact['category']}/{fact['topic']}] {fact['content']}")

        if not topic_lines and not fact_lines:
            return ""

        lines = [f"## Consolidating Memory Recall for: {query}"]
        if topic_lines:
            lines.append("Top topics:")
            lines.extend(topic_lines)
        if fact_lines:
            lines.append("Direct matches:")
            lines.extend(fact_lines)
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
