from __future__ import annotations

import inspect
import json

from consolidating_local import ConsolidatingLocalMemoryProvider
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
