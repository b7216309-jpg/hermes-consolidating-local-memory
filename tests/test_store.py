from __future__ import annotations

import pytest

from consolidating_local.store import MemoryStore, normalize_text, slugify


def test_exclusive_subject_replaces_old_value_and_records_contradiction(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        store.upsert_fact(
            content="Primary project database is PostgreSQL",
            category="project",
            topic="project-data",
            source="test",
            metadata={"subject_key": "project:database", "value_key": "postgresql", "exclusive": True},
        )
        result = store.upsert_fact(
            content="Primary project database is SQLite",
            category="project",
            topic="project-data",
            source="test",
            metadata={"subject_key": "project:database", "value_key": "sqlite", "exclusive": True},
        )

        assert [fact["content"] for fact in store.list_active_facts()] == ["Primary project database is SQLite"]
        assert len(result["contradictions"]) == 1
        assert len(store.recent_contradictions()) == 1
    finally:
        store.close()


def test_coexisting_subject_values_remain_active(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        for shell in ("powershell", "bash"):
            store.upsert_fact(
                content=f"Environment shell is {shell}",
                category="environment",
                topic="environment",
                source="test",
                metadata={"subject_key": "environment:shell", "value_key": shell, "exclusive": True},
            )
        assert {fact["value_key"] for fact in store.list_active_facts()} == {"powershell", "bash"}
        assert not store.recent_contradictions()
    finally:
        store.close()


def test_closed_session_is_not_reopened_by_backfill(tmp_path):
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    store.ensure_memory_session("closed-session", status="open")
    store.close_memory_session("closed-session")
    store.close()

    reopened = MemoryStore(path)
    try:
        session = reopened.get_session_artifacts("closed-session")["session"]
        assert session["status"] == "closed"
        assert session["ended_at"] > 0
    finally:
        reopened.close()


def test_unicode_normalization_and_natural_language_fts(tmp_path):
    assert normalize_text("你好，世界") == "你好世界"
    assert slugify("Café 東京") == "cafe-東京"
    store = MemoryStore(tmp_path / "memory.db")
    try:
        store.upsert_fact(
            content="Environment shell is PowerShell",
            category="environment",
            topic="environment",
            source="test",
            metadata={"subject_key": "environment:shell", "value_key": "powershell", "exclusive": True},
        )
        results = store.search("What shell do I use?", scope="facts")["facts"]
        assert results and results[0]["content"] == "Environment shell is PowerShell"
    finally:
        store.close()


def test_fts_repairs_missing_rows_on_reopen(tmp_path):
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    store.upsert_fact(
        content="User prefers concise answers",
        category="user_pref",
        topic="preferences",
        source="test",
    )
    store._execute("DELETE FROM facts_fts")
    store.close()

    repaired = MemoryStore(path)
    try:
        assert repaired.search("concise", scope="facts")["facts"]
    finally:
        repaired.close()


def test_read_only_store_does_not_initialize_or_modify_database(tmp_path):
    path = tmp_path / "memory.db"
    writable = MemoryStore(path)
    writable.upsert_fact(
        content="Synthetic read-only proof",
        category="test",
        topic="test",
        source="test",
    )
    writable.close()
    before = path.stat().st_mtime_ns

    read_only = MemoryStore(path, read_only=True)
    try:
        assert read_only.search("read-only", scope="facts")["facts"]
        with pytest.raises(RuntimeError, match="read-only"):
            read_only.set_state("forbidden", "write")
    finally:
        read_only.close()

    assert path.stat().st_mtime_ns == before
