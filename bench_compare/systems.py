from __future__ import annotations

import importlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from bench_compare.utils.facts_corpus import FactSeed, SessionSeed
from bench_compare.utils.hermes_home import (
    BASELINE_CHAR_LIMITS,
    DEFAULT_ADDON_CONFIG,
    addon_db_path,
    ensure_fresh_home,
    write_config,
)
from bench_compare.utils.memory_reader import (
    joined_char_count,
    memory_paths,
    query_rows,
    read_baseline_snapshot,
    write_memory_entries,
)


@dataclass(frozen=True)
class RuntimeComponents:
    provider_cls: type


@dataclass
class BenchmarkContext:
    repo_root: Path
    model: str
    scale_facts: int
    overflow_facts: int
    timeout_seconds: int
    addon_config: dict[str, Any]
    baseline: "BaselineSystem"
    addon: "AddonSystem"
    wsl_settings: dict[str, Any]


def resolve_runtime() -> RuntimeComponents:
    try:
        plugin_module = importlib.import_module("plugins.memory.consolidating_local")
    except Exception as exc:
        raise RuntimeError(
            f"Unable to import consolidating_local plugin: {type(exc).__name__}: {exc}"
        ) from exc

    provider_cls = getattr(plugin_module, "ConsolidatingLocalProvider", None) or getattr(
        plugin_module, "ConsolidatingLocalMemoryProvider", None
    )
    if provider_cls is None:
        raise RuntimeError("consolidating_local plugin does not export a provider class.")

    return RuntimeComponents(provider_cls=provider_cls)


def timestamp_iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or time.time()))


def empty_record(dim_id: str, system: str, injected_facts: list[str]) -> dict[str, Any]:
    return {
        "dim_id": dim_id,
        "system": system,
        "timestamp_start": timestamp_iso(),
        "duration_ms": 0,
        "raw_injected_facts": list(injected_facts),
        "raw_recalled_items": [],
        "fuzzy_matches": [],
        "system_prompt_chars": 0,
        "llm_calls_made": 0,
        "tokens_estimated": 0,
        "errors": [],
    }


class BaselineSystem:
    def __init__(self, *, hermes_home: Path, runtime: RuntimeComponents, runtime_seed: Mapping[str, Any] | None = None):
        self.hermes_home = hermes_home
        self.runtime = runtime
        self.runtime_seed = dict(runtime_seed or {})

    def reset(self) -> None:
        ensure_fresh_home(self.hermes_home)
        write_config(self.hermes_home, runtime_model_config=self.runtime_seed.get("model_config"))
        self._seed_runtime_files()
        self._write_snapshot({"memory": [], "user": []})

    def _write_snapshot(self, entries_by_target: dict[str, list[str]]) -> None:
        paths = memory_paths(self.hermes_home)
        write_memory_entries(paths["memory"], entries_by_target.get("memory", []))
        write_memory_entries(paths["user"], entries_by_target.get("user", []))

    def seed_direct_snapshot(self, facts: list[FactSeed]) -> dict[str, Any]:
        kept: list[FactSeed] = []
        dropped: list[FactSeed] = []
        entries_by_target = {"memory": [], "user": []}
        for fact in facts:
            target = fact.target
            proposed = entries_by_target[target] + [fact.text]
            if joined_char_count(proposed) <= BASELINE_CHAR_LIMITS[target]:
                entries_by_target[target].append(fact.text)
                kept.append(fact)
            else:
                dropped.append(fact)
        self._write_snapshot(entries_by_target)
        return {"kept": kept, "dropped": dropped, "entries_by_target": entries_by_target}

    def seed_via_memory_tool(self, facts: list[FactSeed]) -> dict[str, Any]:
        outputs: list[str] = []
        parsed_outputs: list[dict[str, Any]] = []
        kept: list[FactSeed] = []
        state = {"memory": [], "user": []}
        for fact in facts:
            parsed = self._memory_add(state=state, target=fact.target, content=fact.text)
            outputs.append(json.dumps(parsed))
            parsed_outputs.append(parsed)
            if bool(parsed.get("success")):
                kept.append(fact)
        self._write_snapshot(state)
        return {"kept": kept, "outputs": outputs, "parsed_outputs": parsed_outputs}

    def seed_sessions(self, sessions: list[SessionSeed]) -> dict[str, Any]:
        flattened: list[FactSeed] = []
        outputs: list[str] = []
        parsed_outputs: list[dict[str, Any]] = []
        state = {"memory": [], "user": []}
        for session in sessions:
            for fact in session.facts:
                flattened.append(fact)
                parsed = self._memory_add(state=state, target=fact.target, content=fact.text)
                outputs.append(json.dumps(parsed))
                parsed_outputs.append(parsed)
        self._write_snapshot(state)
        return {"facts": flattened, "outputs": outputs, "parsed_outputs": parsed_outputs}

    def snapshot_state(self) -> dict[str, Any]:
        state = read_baseline_snapshot(self.hermes_home)
        state["paths"] = memory_paths(self.hermes_home)
        return state

    def prompt_snapshot(self) -> dict[str, Any]:
        snapshot = self.snapshot_state()
        memory_block = self._render_prompt_block("memory", snapshot["memory_entries"])
        user_block = self._render_prompt_block("user", snapshot["user_entries"])
        return {
            "memory_block": memory_block,
            "user_block": user_block,
            "chars": len(memory_block) + len(user_block),
        }

    def close(self) -> None:
        return None

    def _seed_runtime_files(self) -> None:
        for source_key, target_name in (("source_auth_path", "auth.json"), ("source_env_path", ".env")):
            raw_source = str(self.runtime_seed.get(source_key) or "").strip()
            if not raw_source:
                continue
            source = Path(raw_source)
            if not source.exists():
                continue
            shutil.copy2(source, self.hermes_home / target_name)

    def _memory_add(self, *, state: dict[str, list[str]], target: str, content: str) -> dict[str, Any]:
        content = str(content or "").strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}
        entries = state[target]
        if content in entries:
            return self._success_response(target, entries, "Entry already exists (no duplicate added).")
        new_entries = entries + [content]
        new_total = joined_char_count(new_entries)
        limit = BASELINE_CHAR_LIMITS[target]
        if new_total > limit:
            current = joined_char_count(entries)
            return {
                "success": False,
                "error": (
                    f"Memory at {current:,}/{limit:,} chars. "
                    f"Adding this entry ({len(content)} chars) would exceed the limit. "
                    "Replace or remove existing entries first."
                ),
                "current_entries": list(entries),
                "usage": f"{current:,}/{limit:,}",
            }
        entries.append(content)
        return self._success_response(target, entries, "Entry added.")

    def _success_response(self, target: str, entries: list[str], message: str) -> dict[str, Any]:
        current = joined_char_count(entries)
        limit = BASELINE_CHAR_LIMITS[target]
        pct = min(100, int((current / limit) * 100)) if limit else 0
        return {
            "success": True,
            "target": target,
            "entries": list(entries),
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
            "message": message,
        }

    def _render_prompt_block(self, target: str, entries: list[str]) -> str:
        if not entries:
            return ""
        limit = BASELINE_CHAR_LIMITS[target]
        content = "\n§\n".join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit else 0
        if target == "user":
            header = f"USER PROFILE (who the user is) [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"MEMORY (your personal notes) [{pct}% — {current:,}/{limit:,} chars]"
        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"


class AddonSystem:
    def __init__(
        self,
        *,
        hermes_home: Path,
        runtime: RuntimeComponents,
        addon_config: Mapping[str, Any] | None = None,
        runtime_seed: Mapping[str, Any] | None = None,
    ):
        self.hermes_home = hermes_home
        self.runtime = runtime
        self.addon_config = {**DEFAULT_ADDON_CONFIG, **dict(addon_config or {})}
        self.runtime_seed = dict(runtime_seed or {})
        self.provider = None

    @property
    def db_path(self) -> Path:
        return addon_db_path(self.hermes_home, self.addon_config)

    @property
    def store(self) -> Any:
        if self.provider is None or getattr(self.provider, "_store", None) is None:
            raise RuntimeError("Provider store is not initialized.")
        return getattr(self.provider, "_store")

    def reset(self) -> None:
        self.close()
        ensure_fresh_home(self.hermes_home)
        write_config(
            self.hermes_home,
            self.addon_config,
            runtime_model_config=self.runtime_seed.get("model_config"),
        )
        self._seed_runtime_files()
        self.provider = self.runtime.provider_cls(dict(self.addon_config))
        self.provider.initialize(session_id="bench-bootstrap", hermes_home=str(self.hermes_home))

    def remember_fact(self, fact: FactSeed, *, session_id: str) -> dict[str, Any]:
        if self.provider is None:
            raise RuntimeError("Provider not initialized.")
        raw = self.provider.handle_tool_call(
            "consolidating_memory",
            {**fact.addon_args(), "session_id": session_id},
        )
        parsed = json.loads(raw)
        if not bool(parsed.get("success")):
            raise RuntimeError(str(parsed.get("error") or raw))
        return parsed

    def remember_facts(self, facts: list[FactSeed], *, session_id: str) -> list[dict[str, Any]]:
        return [self.remember_fact(fact, session_id=session_id) for fact in facts]

    def seed_sessions(self, sessions: list[SessionSeed], *, create_summaries: bool = True) -> None:
        for session in sessions:
            self.remember_facts(list(session.facts), session_id=session.session_id)
            if create_summaries:
                summary_text = " ".join(fact.text for fact in session.facts[:4])[:900]
                self.provider.handle_tool_call(
                    "consolidating_memory",
                    {
                        "action": "distill",
                        "session_id": session.session_id,
                        "label": f"Summary {session.session_id}",
                        "content": summary_text,
                    },
                )
        self.consolidate()

    def store_fact_direct(self, fact: FactSeed, *, session_id: str = "bench-direct") -> dict[str, Any]:
        observed_at = time.time() - float(fact.observed_days_ago) * 86400.0
        metadata: dict[str, Any] = {}
        if fact.subject_key:
            metadata["subject_key"] = fact.subject_key
            metadata["exclusive"] = True
            if fact.value:
                metadata["value_key"] = fact.value
        return self.store.upsert_fact(
            content=fact.text,
            category=fact.category,
            topic=fact.topic,
            source="bench-direct",
            importance=fact.importance,
            confidence=0.9,
            metadata=metadata,
            observed_at=observed_at,
            source_session_id=session_id,
        )

    def consolidate(self) -> dict[str, Any]:
        if self.provider is None:
            raise RuntimeError("Provider not initialized.")
        raw = self.provider.handle_tool_call("consolidating_memory", {"action": "consolidate"})
        return json.loads(raw)

    def decay(self) -> dict[str, Any]:
        if self.provider is None:
            raise RuntimeError("Provider not initialized.")
        raw = self.provider.handle_tool_call("consolidating_memory", {"action": "decay"})
        return json.loads(raw)

    def get_context(self, *, session_id: str, query: str = "") -> str:
        if self.provider is None:
            raise RuntimeError("Provider not initialized.")
        getter = getattr(self.provider, "get_context", None)
        if callable(getter):
            return str(getter(session_id=session_id, query=query) or "")
        return str(self.provider.prefetch(query, session_id=session_id) or "")

    def active_facts(self) -> list[dict[str, Any]]:
        return query_rows(
            self.db_path,
            """
            SELECT id, content, category, topic, importance, confidence, salience, active,
                   subject_key, value_key, source_session_id, created_at, updated_at
            FROM facts
            WHERE active = 1
            ORDER BY id ASC
            """,
        )

    def close(self) -> None:
        if self.provider is not None:
            self.provider.shutdown()
            self.provider = None

    def _seed_runtime_files(self) -> None:
        for source_key, target_name in (("source_auth_path", "auth.json"), ("source_env_path", ".env")):
            raw_source = str(self.runtime_seed.get(source_key) or "").strip()
            if not raw_source:
                continue
            source = Path(raw_source)
            if not source.exists():
                continue
            shutil.copy2(source, self.hermes_home / target_name)
