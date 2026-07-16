from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from contextvars import copy_context
from pathlib import Path
from types import SimpleNamespace

from consolidating_local import TOOL_SCHEMA, ConsolidatingLocalMemoryProvider, register
from consolidating_local.origin import (
    classify_turn,
    mark_gateway_user_dispatch,
    note_llm_turn,
    reset_origin_state,
)
from consolidating_local.store import MemoryStore


def _provider(tmp_path, **config):
    values = {"db_path": str(tmp_path / "memory.db")}
    values.update(config)
    provider = ConsolidatingLocalMemoryProvider(values)
    provider.initialize("session-old", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
    return provider


def test_current_hermes_hook_signatures_are_supported():
    sync = inspect.signature(ConsolidatingLocalMemoryProvider.sync_turn)
    memory_write = inspect.signature(ConsolidatingLocalMemoryProvider.on_memory_write)
    assert "messages" in sync.parameters
    assert "metadata" in memory_write.parameters
    assert hasattr(ConsolidatingLocalMemoryProvider, "on_session_switch")


def test_plugin_registers_authoritative_gateway_origin_hooks():
    class FakeContext:
        def __init__(self):
            self.providers = []
            self.hooks = {}

        def register_memory_provider(self, provider):
            self.providers.append(provider)

        def register_hook(self, name, handler):
            self.hooks[name] = handler

    context = FakeContext()
    register(context)
    assert len(context.providers) == 1
    assert set(context.hooks) == {"pre_gateway_dispatch", "pre_llm_call"}


def test_register_supports_hermes_separate_provider_and_hook_contexts():
    class ProviderContext:
        def __init__(self):
            self.providers = []

        def register_memory_provider(self, provider):
            self.providers.append(provider)

    class HookContext:
        def __init__(self):
            self.hooks = {}

        def register_hook(self, name, handler):
            self.hooks[name] = handler

    provider_context = ProviderContext()
    hook_context = HookContext()
    register(provider_context)
    register(hook_context)
    assert len(provider_context.providers) == 1
    assert set(hook_context.hooks) == {"pre_gateway_dispatch", "pre_llm_call"}


def test_origin_state_is_shared_across_hermes_module_namespaces():
    origin_path = Path(__file__).resolve().parents[1] / "plugins" / "memory" / "consolidating_local" / "origin.py"

    def load_copy(name):
        spec = importlib.util.spec_from_file_location(name, origin_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    first = load_copy("_test_memory_origin_general")
    second = load_copy("_test_memory_origin_provider")
    try:
        first.reset_origin_state()
        first.mark_gateway_user_dispatch(SimpleNamespace(internal=False))
        assert (
            second.classify_turn(
                session_id="shared-session",
                user_message="real inbound",
                platform="telegram",
            )
            == "user"
        )
        second.note_llm_turn(
            session_id="shared-session",
            user_message="remember this turn",
            platform="cli",
        )
        assert first.recorded_origin("shared-session", "remember this turn") == "user"
    finally:
        first.reset_origin_state()
        sys.modules.pop("_test_memory_origin_general", None)
        sys.modules.pop("_test_memory_origin_provider", None)


def test_gateway_marker_is_single_use_across_copied_contexts():
    reset_origin_state()
    mark_gateway_user_dispatch(SimpleNamespace(internal=False))
    copied = copy_context()
    assert (
        copied.run(
            classify_turn,
            session_id="gateway-session",
            user_message="real inbound",
            platform="mattermost",
        )
        == "user"
    )
    assert (
        classify_turn(
            session_id="gateway-session",
            user_message="nested synthetic turn",
            platform="mattermost",
        )
        == "internal"
    )
    reset_origin_state()


def test_gateway_internal_turns_never_enter_memory(tmp_path):
    reset_origin_state()
    provider = ConsolidatingLocalMemoryProvider({"db_path": str(tmp_path / "memory.db"), "memory_scope": "global"})
    provider.initialize(
        "telegram-session",
        hermes_home=str(tmp_path),
        platform="telegram",
        agent_context="primary",
    )
    try:
        note_llm_turn(
            session_id="telegram-session",
            user_message="internal process result",
            platform="telegram",
        )
        provider.sync_turn(
            "internal process result",
            "internal acknowledgement",
            session_id="telegram-session",
        )
        provider.queue_prefetch("internal process result", session_id="telegram-session")
        assert provider.prefetch("internal process result", session_id="telegram-session") == ""
        provider._task_queue.join()
        assert provider._store.counts()["episodes"] == 0
    finally:
        provider.shutdown()
        reset_origin_state()


def test_real_gateway_turn_is_captured_once(tmp_path):
    reset_origin_state()
    provider = ConsolidatingLocalMemoryProvider({"db_path": str(tmp_path / "memory.db"), "memory_scope": "global"})
    provider.initialize(
        "telegram-session",
        hermes_home=str(tmp_path),
        platform="telegram",
        agent_context="primary",
    )
    try:
        mark_gateway_user_dispatch(SimpleNamespace(internal=False))
        note_llm_turn(
            session_id="telegram-session",
            user_message="a genuine human turn",
            platform="telegram",
        )
        provider.sync_turn(
            "a genuine human turn",
            "one answer",
            session_id="telegram-session",
        )
        provider._task_queue.join()
        assert provider._store.counts()["episodes"] == 1
    finally:
        provider.shutdown()
        reset_origin_state()


def test_background_review_and_its_assistant_pair_are_excluded_from_history_extraction(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    calls = []
    monkeypatch.setattr(
        provider,
        "_extract_turn_facts",
        lambda **kwargs: calls.append(kwargs) or [],
    )
    review_prompt = "Review the conversation above and consider saving to memory if appropriate."
    try:
        provider._extract_messages_facts(
            [
                {"role": "user", "content": "a genuine human turn"},
                {"role": "assistant", "content": "a genuine answer"},
                {"role": "user", "content": review_prompt},
                {"role": "assistant", "content": "hidden review result"},
            ],
            session_id="session-old",
        )
        assert calls == [{"user_content": "a genuine human turn", "assistant_content": "a genuine answer"}]
    finally:
        provider.shutdown()


def test_defaults_are_local_and_do_not_rewrite_builtin_memory():
    provider = ConsolidatingLocalMemoryProvider()
    assert provider._cfg()["builtin_snapshot_sync_enabled"] is False
    assert provider.get_config_schema() == []
    advanced = provider.get_advanced_config_schema()
    builtin_sync = next(item for item in advanced if item["key"] == "builtin_snapshot_sync_enabled")
    assert builtin_sync["default"] == "false"
    assert all(item["key"] != "extractor_backend" for item in advanced)
    llm_model = next(item for item in advanced if item["key"] == "llm_model")
    assert llm_model["default"] == ""
    llm_disable_thinking = next(item for item in advanced if item["key"] == "llm_disable_thinking")
    assert llm_disable_thinking["default"] == "false"


def test_disabled_builtin_sync_removes_only_plugin_owned_snapshot_blocks(tmp_path):
    memories = tmp_path / "memories"
    memories.mkdir()
    for name in ("USER.md", "MEMORY.md"):
        (memories / name).write_text(
            "<!-- consolidating_local:auto:start -->\n"
            "- stale generated state\n"
            "<!-- consolidating_local:auto:end -->\n\n"
            "Manual operator note.\n",
            encoding="utf-8",
        )

    provider = _provider(tmp_path, builtin_snapshot_sync_enabled=False)
    try:
        for name in ("USER.md", "MEMORY.md"):
            content = (memories / name).read_text(encoding="utf-8")
            assert "consolidating_local:auto" not in content
            assert "stale generated state" not in content
            assert content.strip() == "Manual operator note."
        status = json.loads(provider._store.get_state("last_builtin_snapshot_sync"))
        assert status["success"] is True
        assert status["reason"] == "disabled_cleanup"
    finally:
        provider.shutdown()


def test_model_memory_surface_is_compact_and_excludes_operator_maintenance(tmp_path):
    encoded = json.dumps(TOOL_SCHEMA, separators=(",", ":"))
    actions = set(TOOL_SCHEMA["parameters"]["properties"]["action"]["enum"])
    assert len(encoded) < 3800
    assert len(TOOL_SCHEMA["description"]) < 200
    assert actions.isdisjoint(
        {
            "status",
            "consolidate",
            "review",
            "decay",
            "export",
            "doctor",
            "maintain",
            "backup",
            "export_json",
        }
    )

    provider = _provider(tmp_path)
    try:
        prompt = provider.system_prompt_block()
        assert len(prompt) < 120
        assert "facts" not in prompt
        assert "backend" not in prompt
        results = {
            "facts": [
                {
                    "content": f"fact-{number} " + "x" * 4000,
                    "category": "general",
                    "topic": "load-test",
                }
                for number in range(30)
            ]
        }
        recall = provider._render_prefetch("bounded recall", results, cues={"mode": "current_state"})
        assert len(recall) <= 4500
        assert max(map(len, recall.splitlines())) <= 500
        assert recall.count("superseded") == 1
    finally:
        provider.shutdown()


def test_encrypted_provider_reports_unavailable_without_required_key(monkeypatch):
    monkeypatch.delenv("CONSOLIDATING_MEMORY_DB_KEY", raising=False)
    provider = ConsolidatingLocalMemoryProvider({"database_encryption": True})
    assert provider.is_available() is False


def test_malformed_numeric_config_falls_back_and_is_bounded():
    provider = ConsolidatingLocalMemoryProvider(
        {"prefetch_limit": "not-a-number", "decay_min_salience": float("nan"), "llm_timeout_seconds": 99999}
    )
    config = provider._cfg()
    assert config["prefetch_limit"] == 8
    assert config["decay_min_salience"] == 0.15
    assert config["llm_timeout_seconds"] == 300


def test_malformed_tool_limit_falls_back_to_default(tmp_path):
    provider = _provider(tmp_path)
    try:
        response = json.loads(
            provider.handle_tool_call(
                "consolidating_memory",
                {"action": "search", "query": "anything", "limit": "not-a-number"},
            )
        )
        assert response["success"] is True
    finally:
        provider.shutdown()


def test_shutdown_drains_accepted_turns_and_session_switch_updates_target(tmp_path):
    provider = _provider(tmp_path)
    for number in range(12):
        provider.sync_turn(f"Durable turn {number}", "Acknowledged", session_id="session-old")
    provider.on_session_switch("session-new", parent_session_id="session-old")
    provider.on_memory_write(
        "add",
        "user",
        "My name is Alice",
        metadata={"session_id": "session-new", "write_origin": "assistant_tool"},
    )
    provider.shutdown()

    store = MemoryStore(tmp_path / "memory.db")
    try:
        assert store.counts()["episodes"] == 12
        names = store.search("Alice name", scope="facts")["facts"]
        assert names and names[0]["source_session_id"] == "session-new"
        assert store.get_session_artifacts("session-new")["session"]
    finally:
        store.close()


def test_prefetch_is_immediately_useful_and_cache_is_invalidated(tmp_path):
    provider = _provider(tmp_path)
    try:
        provider.on_memory_write(
            "add",
            "memory",
            "My shell is PowerShell.",
            metadata={"session_id": "session-old", "write_origin": "assistant_tool"},
        )
        provider._task_queue.join()
        first = provider.prefetch("What shell do I use?", session_id="session-old")
        assert "PowerShell" in first or "powershell" in first

        provider.on_memory_write(
            "add",
            "user",
            "My name is Alice",
            metadata={"session_id": "session-old"},
        )
        provider._task_queue.join()
        assert "Alice" in provider.prefetch("What is my name?", session_id="session-old")
    finally:
        provider.shutdown()


def test_automatic_prefetch_does_not_fall_back_to_unrelated_global_memory(tmp_path):
    provider = _provider(tmp_path)
    try:
        provider.on_memory_write(
            "add",
            "memory",
            "My shell is PowerShell.",
            metadata={"session_id": "session-old", "write_origin": "assistant_tool"},
        )
        provider._task_queue.join()

        automatic = provider.prefetch("A quiet afternoon with nothing needed.", session_id="session-new")
        explicit = provider.get_context(query="A quiet afternoon with nothing needed.", session_id="session-old")

        assert automatic == ""
        assert "PowerShell" in explicit or "powershell" in explicit
    finally:
        provider.shutdown()


def test_non_primary_context_is_read_only(tmp_path):
    provider = ConsolidatingLocalMemoryProvider({"db_path": str(tmp_path / "memory.db")})
    provider.initialize("cron", hermes_home=str(tmp_path), platform="cron", agent_context="cron")
    try:
        provider.sync_turn("My name is Wrong", "Okay")
        response = json.loads(
            provider.handle_tool_call("consolidating_memory", {"action": "remember", "content": "bad"})
        )
        assert response["success"] is False
        assert provider._store.counts()["episodes"] == 0
        assert provider._store.counts()["facts"] == 0
    finally:
        provider.shutdown()


def test_non_primary_write_provenance_is_ignored_by_primary_provider(tmp_path):
    provider = _provider(tmp_path)
    try:
        provider.on_memory_write(
            "add",
            "user",
            "My name is Cron Poison",
            metadata={"execution_context": "cron", "session_id": "session-old"},
        )
        provider._task_queue.join()
        assert not provider._store.search("Cron Poison", scope="facts")["facts"]
    finally:
        provider.shutdown()
