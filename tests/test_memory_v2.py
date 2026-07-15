from __future__ import annotations

import json
import sqlite3
import threading
import time
from argparse import ArgumentParser
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

from consolidating_local import ConsolidatingLocalMemoryProvider
from consolidating_local.admin import _restore
from consolidating_local.cli import register_cli
from consolidating_local.llm_client import OpenAICompatibleLLM
from consolidating_local.origin import mark_gateway_user_dispatch, note_llm_turn
from consolidating_local.store import MemoryStore


def _fact(store: MemoryStore, content: str, *, value: str, role: str, confidence: float = 0.9, **kwargs):
    return store.upsert_fact(
        content=content,
        category="project",
        topic="project-data",
        source="user" if role == "user" else "llm",
        source_role=role,
        confidence=confidence,
        importance=8,
        metadata={"subject_key": "project:database", "value_key": value, "exclusive": True},
        **kwargs,
    )


def test_evidence_prevents_weak_newer_inference_from_overwriting_user(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        postgres = _fact(store, "Primary project database is PostgreSQL", value="postgresql", role="user")
        mysql = _fact(store, "Primary project database is MySQL", value="mysql", role="assistant", confidence=0.35)

        assert postgres["fact"]["active"] == 1
        assert mysql["fact"]["active"] == 0
        active = store.search("project database", scope="facts", limit=10)["facts"]
        assert [row["content"] for row in active] == ["Primary project database is PostgreSQL"]
        explanation = store.explain_fact(int(postgres["fact"]["id"]))
        assert explanation["fact"]["belief_score"] > mysql["fact"]["belief_score"]
        assert len(explanation["evidence"]) == 1
    finally:
        store.close()


def test_nested_transactions_roll_back_every_table_including_fts(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        with pytest.raises(RuntimeError):
            with store.transaction():
                _fact(store, "Primary project database is PostgreSQL", value="postgresql", role="user")
                raise RuntimeError("abort logical write")
        assert store.counts()["facts"] == 0
        assert store.search("PostgreSQL", scope="facts", limit=10)["facts"] == []
        assert store.counts()["history"] == 0
        assert store.counts()["evidence"] == 0

        with pytest.raises(KeyboardInterrupt):
            with store.transaction():
                _fact(store, "Primary project database is SQLite", value="sqlite", role="user")
                raise KeyboardInterrupt
        assert store.counts()["facts"] == 0
        assert store._transaction_depth == 0
    finally:
        store.close()


def test_doctor_repairs_all_internal_reference_types(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        winner = _fact(store, "Primary project database is PostgreSQL", value="postgresql", role="user")
        loser = _fact(
            store,
            "Primary project database is MySQL",
            value="mysql",
            role="assistant",
            confidence=0.2,
        )
        store.rebuild_topics()
        episode = store.append_episode(
            session_id="repair",
            user_content="Reference repair",
            assistant_content="Acknowledged",
        )
        trace = store.append_trace(
            session_id="repair",
            label="Repair trace",
            content="Trace with episode provenance",
            source_episode_id=episode["id"],
        )

        store._execute("DELETE FROM episodes WHERE id=?", (episode["id"],))
        store._execute("DELETE FROM facts WHERE id=?", (winner["fact"]["id"],))
        store._execute(
            """INSERT INTO memory_links(
                   source_kind, source_id, target_kind, target_id, link_type, metadata_json, created_at
               ) VALUES ('legacy-unknown', '1', 'fact', ?, 'legacy', '{}', ?)""",
            (str(loser["fact"]["id"]), time.time()),
        )

        degraded = store.doctor()
        assert degraded["ok"] is False
        assert degraded["dangling_references"]["topic_membership"] > 0
        assert degraded["dangling_references"]["trace_sources"] == 1
        assert degraded["dangling_references"]["fact_supersession"] == 1
        assert degraded["dangling_references"]["contradictions"] == 1

        repaired = store.doctor(repair=True)
        assert repaired["ok"] is True
        assert not any(repaired["dangling_references"].values())
        assert (
            store._fetchone("SELECT source_episode_id FROM memory_traces WHERE id=?", (trace["id"],))[
                "source_episode_id"
            ]
            == 0
        )
        assert (
            store._fetchone("SELECT superseded_by FROM facts WHERE id=?", (loser["fact"]["id"],))["superseded_by"]
            is None
        )
    finally:
        store.close()


def test_v1_database_migrates_columns_evidence_and_fts_without_losing_facts(tmp_path):
    path = tmp_path / "legacy.db"
    store = MemoryStore(path)
    try:
        legacy = store.upsert_fact(
            content="Legacy user prefers concise answers",
            category="user_pref",
            topic="preferences",
            source="user",
            source_role="user",
            metadata={"subject_key": "user:response_style", "value_key": "concise", "exclusive": True},
        )["fact"]
    finally:
        store.close()

    connection = sqlite3.connect(path)
    try:
        connection.execute("DELETE FROM belief_evidence")
        for column in (
            "belief_score",
            "observation_count",
            "valid_from",
            "valid_until",
            "sensitivity",
            "memory_class",
            "pinned",
            "revision",
        ):
            connection.execute(f"ALTER TABLE facts DROP COLUMN {column}")
        connection.execute("ALTER TABLE memory_sessions DROP COLUMN sensitivity")
        connection.execute("ALTER TABLE memory_traces DROP COLUMN sensitivity")
        connection.execute("ALTER TABLE topics DROP COLUMN sensitivity")
        connection.execute("ALTER TABLE episodes DROP COLUMN sensitivity")
        connection.execute("DROP TABLE facts_fts")
        connection.execute("CREATE VIRTUAL TABLE facts_fts USING fts5(fact_id UNINDEXED, content, topic, category)")
        connection.execute(
            "INSERT INTO facts_fts(fact_id, content, topic, category) VALUES (?, ?, ?, ?)",
            (legacy["id"], legacy["content"], legacy["topic"], legacy["category"]),
        )
        connection.commit()
    finally:
        connection.close()

    migrated = MemoryStore(path)
    try:
        result = migrated.search("concise answers", scope="facts", limit=5)["facts"]
        assert len(result) == 1
        explanation = migrated.explain_fact(result[0]["id"])
        assert explanation["evidence"][0]["source_role"] == "user"
        assert explanation["fact"]["belief_score"] > 0.6
        assert explanation["fact"]["revision"] == 1
        assert "sensitivity" in {row["name"] for row in migrated._fetchall("PRAGMA table_info(memory_sessions)")}
        assert "sensitivity" in {row["name"] for row in migrated._fetchall("PRAGMA table_info(memory_traces)")}
        assert "sensitivity" in {row["name"] for row in migrated._fetchall("PRAGMA table_info(topics)")}
        assert "sensitivity" in {row["name"] for row in migrated._fetchall("PRAGMA table_info(episodes)")}
        assert "operation_key" in {row["name"] for row in migrated._fetchall("PRAGMA table_info(episodes)")}
        assert migrated.doctor()["ok"] is True
    finally:
        migrated.close()


def test_temporal_working_procedural_prospective_and_associative_memory(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        expired = store.upsert_fact(
            content="Temporary release flag is blue",
            category="project",
            topic="release",
            source="user",
            source_role="user",
            valid_until=time.time() - 1,
        )
        assert store.search("release flag", scope="facts", limit=10)["facts"] == []
        assert store.explain_fact(expired["fact"]["id"])["fact"]["valid_until"] > 0

        future = store.upsert_fact(
            content="The scheduled release window is open",
            category="project",
            topic="release",
            source="user",
            source_role="user",
            valid_from=time.time() + 3600,
        )["fact"]
        assert store.search("scheduled release window", scope="facts", limit=10)["facts"] == []
        assert future["id"] not in {row["id"] for row in store.list_active_facts()}
        assert future["id"] not in {row["id"] for row in store.recent_items()["facts"]}
        store._execute("UPDATE facts SET valid_from=? WHERE id=?", (time.time() - 1, future["id"]))
        assert store.search("scheduled release window", scope="facts", limit=10)["facts"]

        store.set_working_memory(session_id="s", memory_key="one", content="first", capacity=2)
        store.set_working_memory(session_id="s", memory_key="two", content="second", priority=9, capacity=2)
        store.set_working_memory(session_id="s", memory_key="three", content="third", priority=8, capacity=2)
        assert [row["content"] for row in store.list_working_memory("s")] == ["second", "third"]

        procedure = store.upsert_procedure(
            procedure_key="release",
            label="Release",
            steps=["Run tests", "Deploy"],
            prerequisites=["Clean tree"],
            success_criteria="Health check passes",
        )
        result = store.record_procedure_result("release", success=True)
        assert procedure["steps"] == ["Run tests", "Deploy"]
        assert result["use_count"] == result["success_count"] == 1

        intention = store.add_intention(intention="Send the release note", due_at=time.time() - 1, session_id="s")
        assert store.list_intentions(due_only=True)[0]["id"] == intention["id"]
        assert store.resolve_intention(intention["id"])["status"] == "completed"

        event = store.upsert_autobiographical_event(
            event_key="launch", content="The team launched version 2", event_at=time.time()
        )
        assert store.list_autobiographical_events("launched")[0]["id"] == event["id"]

        left = store.upsert_fact(content="Uses Python", category="project", topic="stack", source="user")
        right = store.upsert_fact(content="Uses pytest", category="project", topic="stack", source="user")
        store.associate_fact_group([left["fact"]["id"], right["fact"]["id"]])
        associated = store.associated_facts([left["fact"]["id"]])
        assert associated[0]["id"] == right["fact"]["id"]
        with pytest.raises(ValueError, match="Unknown fact reference"):
            store.associate("fact", 999999, "fact", left["fact"]["id"])
        with pytest.raises(ValueError, match="Unsupported memory reference kind"):
            store.associate("imaginary", 1, "fact", left["fact"]["id"])
        with pytest.raises(ValueError, match="itself"):
            store.associate("fact", left["fact"]["id"], "fact", left["fact"]["id"])

        compound = store.upsert_fact(content="Uses Ruff and mypy", category="project", topic="quality", source="user")
        with pytest.raises(ValueError, match="at least two"):
            store.split_fact(compound["fact"]["id"], ["Uses Ruff"])
        assert store.explain_fact(compound["fact"]["id"])["fact"]["active"] == 1
        split = store.split_fact(compound["fact"]["id"], ["Uses Ruff", "Uses mypy"])
        assert len(split["created"]) == 2
        assert all(row["active"] == 1 for row in split["created"])
        with pytest.raises(ValueError, match="Unknown loser"):
            store.merge_facts(split["created"][0]["id"], [999999])
        assert all(store.explain_fact(row["id"])["fact"]["active"] == 1 for row in split["created"])
        merged = store.merge_facts(split["created"][0]["id"], [split["created"][1]["id"]])
        assert merged["merged"] == [split["created"][1]["id"]]
    finally:
        store.close()


def test_durable_queue_leases_doctor_backup_and_portable_import(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        operation_id = store.enqueue_operation("remember_fact", {"content": "durable"})
        claimed = store.claim_operations(limit=1)
        assert claimed[0]["id"] == operation_id
        store.complete_operation(operation_id)
        assert store.pending_operation_count() == 0

        poison_id = store.enqueue_operation("unknown-operation", {"content": "poison"})
        for attempt in range(1, 4):
            poison = store.claim_operations(limit=1, max_attempts=3)[0]
            assert poison["id"] == poison_id
            failed = store.fail_operation(
                poison_id,
                f"failure {attempt}",
                retry_delay_seconds=0,
                max_attempts=3,
            )
        assert failed["status"] == "failed"
        assert store.pending_operation_count() == 0
        assert store.failed_operation_count() == 1
        assert store.list_failed_operations()[0]["error"] == "failure 3"
        degraded = store.doctor()
        assert not degraded["ok"]
        assert degraded["failed_operations"] == 1
        assert store.retry_failed_operations(limit=1) == 1
        assert store.failed_operation_count() == 0
        assert store.pending_operation_count() == 1
        retried = store.claim_operations(limit=1, max_attempts=3)[0]
        assert retried["attempts"] == 1
        store.complete_operation(poison_id)
        assert store.doctor()["ok"]

        assert store.acquire_lease("consolidation", "a", ttl_seconds=60)
        assert not store.acquire_lease("consolidation", "b", ttl_seconds=60)
        assert store.release_lease("consolidation", "a")

        approval = store.request_approval(candidate={"content": "private"}, sensitivity="health", reason="test")
        store.resolve_approval(approval["id"], approved=False)
        with pytest.raises(ValueError, match="already been resolved"):
            store.resolve_approval(approval["id"], approved=True)

        primary = _fact(store, "Primary project database is PostgreSQL", value="postgresql", role="user")
        _fact(
            store,
            "Primary project database is MySQL",
            value="mysql",
            role="assistant",
            confidence=0.2,
        )
        preference = store.upsert_preference(
            key="private-health", label="Private health", value="private", sensitivity="health"
        )
        policy = store.upsert_policy(
            key="credential-policy", label="Credential policy", content="Password is private", sensitivity="credential"
        )
        summary = store.upsert_summary(
            label="Private summary", summary="Medical diagnosis summary", sensitivity="health"
        )
        assert preference["sensitivity"] == "health"
        assert policy["sensitivity"] == "credential"
        assert summary["sensitivity"] == "health"
        store.append_trace(
            session_id="legacy",
            label="Legacy trace",
            content="Old medical diagnosis must not be summarized",
        )
        assert store.get_session_artifacts("legacy")["traces"] == []
        store.ensure_memory_session("portable")
        portable_journal = store.add_journal(
            label="Release note", content="Version two is ready for the team", session_id="portable"
        )
        portable_summary = store.upsert_summary(
            label="Release summary",
            summary="The release passed its verification",
            session_id="portable",
            source_refs=[{"kind": "fact", "id": primary["fact"]["id"]}],
        )
        portable_working = store.set_working_memory(
            session_id="portable", memory_key="focus", content="Prepare the release", ttl_seconds=3600
        )
        portable_procedure = store.upsert_procedure(procedure_key="release", label="Release", steps=["Test", "Publish"])
        store.record_procedure_result("release", success=True)
        portable_intention = store.add_intention(intention="Announce the release", session_id="portable")
        store.resolve_intention(portable_intention["id"], status="completed")
        portable_event = store.upsert_autobiographical_event(
            event_key="version-two", content="The team completed version two"
        )
        store.associate("fact", primary["fact"]["id"], "autobiographical_event", portable_event["id"], "supported_by")
        store.add_link("procedure", portable_procedure["id"], "fact", primary["fact"]["id"], "uses")
        store.add_link("working", portable_working["id"], "intention", portable_intention["id"], "supports")
        store.record_history(
            entity_kind="memory",
            entity_id="nested-sensitive",
            action="updated",
            payload={"nested": [{"sensitivity": "health", "content": "nested private marker"}]},
        )
        store.close_memory_session("private-session", summary="My salary details", sensitivity="financial")
        assert portable_journal["id"] and portable_summary["id"]
        doctor = store.doctor()
        assert doctor["ok"] is True
        backup = store.backup_to(tmp_path / "backup.db")
        backup_store = MemoryStore(backup)
        try:
            assert backup_store.doctor()["ok"] is True
        finally:
            backup_store.close()

        restore_path = tmp_path / "restored.db"
        stale_store = MemoryStore(restore_path)
        try:
            stale_store.upsert_fact(content="Stale destination", category="general", topic="stale", source="test")
        finally:
            stale_store.close()
        restored = _restore(Path(backup), restore_path)
        assert restored["database"] == str(restore_path.resolve())
        restored_store = MemoryStore(restore_path)
        try:
            assert restored_store.doctor()["ok"] is True
            assert restored_store.search("PostgreSQL", scope="facts", limit=5)["facts"]
            assert restored_store.search("Stale destination", scope="facts", limit=5)["facts"] == []
        finally:
            restored_store.close()

        exported = store.export_data()
        rendered_export = json.dumps(exported).lower()
        assert "medical diagnosis summary" not in rendered_export
        assert "password is private" not in rendered_export
        assert "old medical diagnosis" not in rendered_export
        assert "nested private marker" not in rendered_export
        assert "my salary details" not in rendered_export
        imported = MemoryStore(tmp_path / "imported.db")
        try:
            counts = imported.import_data(exported)
            assert counts["facts"] == 2
            assert counts["evidence"] == 2
            assert counts["journals"] == 1
            assert counts["summaries"] == 1
            assert counts["working"] == 1
            assert counts["procedures"] == 1
            assert counts["intentions"] == 1
            assert counts["events"] == 1
            assert counts["associations"] == 1
            assert counts["contradictions"] == 1
            assert counts["links"] >= 2
            assert counts["history"] > 0
            assert imported.search("PostgreSQL", scope="facts", limit=5)["facts"]
            imported_fact = imported.search("PostgreSQL", scope="facts", limit=5)["facts"][0]
            assert imported.explain_fact(imported_fact["id"])["evidence"][0]["source_role"] == "user"
            imported_mysql = imported._fetchone("SELECT * FROM facts WHERE content LIKE '%MySQL%'")
            assert imported_mysql["active"] == 0
            assert imported_mysql["superseded_by"] == imported_fact["id"]
            assert imported.list_working_memory("portable")[0]["content"] == "Prepare the release"
            assert imported.list_procedures("release")[0]["steps"] == ["Test", "Publish"]
            assert imported.list_procedures("release")[0]["use_count"] == 1
            imported_intention = imported._fetchone(
                "SELECT * FROM prospective_memories WHERE intention='Announce the release'"
            )
            assert imported_intention["status"] == "completed"
            assert imported._fetchone(
                "SELECT id FROM memory_links WHERE source_kind='procedure' AND source_id=?",
                (str(imported.list_procedures("release")[0]["id"]),),
            )
            assert imported.list_autobiographical_events("version two")
            assert imported.recent_contradictions()[0]["subject_key"] == "project:database"
            assert imported.doctor()["ok"] is True
        finally:
            imported.close()
    finally:
        store.close()


def test_provider_user_isolation_sensitive_consent_and_brain_tools(tmp_path):
    config = {
        "db_path": str(tmp_path / "memory.db"),
        "memory_scope": "user",
        "sensitive_memory": "ask",
        "queue_max_size": 8,
        "builtin_snapshot_sync_enabled": True,
        "wiki_export_enabled": True,
    }
    first = ConsolidatingLocalMemoryProvider(config)
    second = ConsolidatingLocalMemoryProvider(config)
    try:
        first.initialize("s1", hermes_home=str(tmp_path), platform="gateway", agent_context="primary", user_id="alice")
        second.initialize("s2", hermes_home=str(tmp_path), platform="gateway", agent_context="primary", user_id="bob")
        assert first._store is not None and second._store is not None
        assert first._store.db_path != second._store.db_path
        assert first._wiki_export_dir() != second._wiki_export_dir()
        assert first._wiki_export_dir().parent.name == "scopes"
        snapshot_status = json.loads(first._store.get_state("last_builtin_snapshot_sync"))
        assert snapshot_status["success"] is False
        assert "scoped memory" in snapshot_status["reason"]
        assert not (tmp_path / "memories" / "USER.md").exists()

        remembered = json.loads(
            first.handle_tool_call(
                "consolidating_memory",
                {"action": "remember", "content": "My medical diagnosis is private", "category": "general"},
            )
        )
        assert remembered["result"]["action"] == "pending"
        approval_id = remembered["result"]["approval"]["id"]
        assert second._store.counts()["approvals"] == 0
        approved = json.loads(
            first.handle_tool_call(
                "consolidating_memory", {"action": "approval", "fact_id": approval_id, "approved": True}
            )
        )
        assert approved["stored"]["fact"]["sensitivity"] == "health"
        string_false_candidate = json.loads(
            first.handle_tool_call(
                "consolidating_memory",
                {"action": "remember", "content": "My medical allergy is private", "category": "general"},
            )
        )
        string_false_resolution = json.loads(
            first.handle_tool_call(
                "consolidating_memory",
                {
                    "action": "approval",
                    "fact_id": string_false_candidate["result"]["approval"]["id"],
                    "approved": "false",
                },
            )
        )
        assert string_false_resolution["approval"]["status"] == "rejected"
        assert string_false_resolution["stored"] is None
        generic = json.loads(first.handle_tool_call("consolidating_memory", {"action": "search", "query": "private"}))
        assert generic["results"]["facts"] == []
        explicit = json.loads(
            first.handle_tool_call("consolidating_memory", {"action": "search", "query": "medical diagnosis"})
        )
        assert explicit["results"]["facts"]
        approved_fact_id = approved["stored"]["fact"]["id"]
        first._store._execute("UPDATE facts SET next_review_at=? WHERE id=?", (time.time() - 1, approved_fact_id))
        hidden_review = json.loads(first.handle_tool_call("consolidating_memory", {"action": "review"}))
        assert all(row["id"] != approved_fact_id for row in hidden_review["results"]["facts"])
        visible_review = json.loads(
            first.handle_tool_call("consolidating_memory", {"action": "review", "query": "health"})
        )
        assert any(row["id"] == approved_fact_id for row in visible_review["results"]["facts"])
        first._store.upsert_policy(
            key="private-care",
            label="Private care",
            content="Use the private care plan",
            sensitivity="health",
        )
        hidden_policy_list = json.loads(first.handle_tool_call("consolidating_memory", {"action": "policy"}))
        assert all(row["policy_key"] != "private-care" for row in hidden_policy_list["results"])
        first._store.upsert_fact(
            content="Private health state is alpha",
            category="general",
            topic="health",
            source="user",
            source_role="user",
            source_session_id="s1",
            sensitivity="health",
            metadata={"subject_key": "user:health-state", "value_key": "alpha", "exclusive": True},
        )
        first._store.upsert_fact(
            content="Private health state is beta",
            category="general",
            topic="health",
            source="user",
            source_role="user",
            source_session_id="s1",
            sensitivity="health",
            metadata={"subject_key": "user:health-state", "value_key": "beta", "exclusive": True},
            explicit_correction=True,
        )
        first._store.rebuild_topics()
        hidden_topic = json.loads(
            first.handle_tool_call(
                "consolidating_memory",
                {"action": "search", "query": "alpha", "scope": "topics"},
            )
        )
        assert hidden_topic["results"]["topics"] == []
        visible_topic = json.loads(
            first.handle_tool_call(
                "consolidating_memory",
                {"action": "search", "query": "health", "scope": "topics"},
            )
        )
        assert any(row["sensitivity"] == "health" for row in visible_topic["results"]["topics"])
        hidden_contradictions = json.loads(first.handle_tool_call("consolidating_memory", {"action": "contradictions"}))
        assert all(row["subject_key"] != "user:health-state" for row in hidden_contradictions["results"])
        visible_contradictions = json.loads(
            first.handle_tool_call("consolidating_memory", {"action": "contradictions", "query": "health"})
        )
        assert any(row["subject_key"] == "user:health-state" for row in visible_contradictions["results"])
        hidden_provenance = json.loads(
            first.handle_tool_call(
                "consolidating_memory",
                {
                    "action": "search",
                    "query": "where did this come from",
                    "scope": "facts",
                    "subject_key": "user:health-state",
                },
            )
        )
        assert hidden_provenance.get("provenance", []) == []
        visible_provenance = json.loads(
            first.handle_tool_call(
                "consolidating_memory",
                {
                    "action": "search",
                    "query": "health provenance source",
                    "scope": "facts",
                    "subject_key": "user:health-state",
                },
            )
        )
        assert visible_provenance.get("provenance")
        portable = json.loads(first.handle_tool_call("consolidating_memory", {"action": "export_json"}))
        assert all("medical diagnosis" not in row["content"].lower() for row in portable["result"]["facts"])
        first._config["export_redact_sensitive"] = False
        unconfirmed_export = json.loads(
            first.handle_tool_call("consolidating_memory", {"action": "export_json", "confirm": "false"})
        )
        assert unconfirmed_export["success"] is False
        assert "confirm=true" in unconfirmed_export["error"]
        first._config["export_redact_sensitive"] = True

        journal = json.loads(
            first.handle_tool_call(
                "consolidating_memory", {"action": "journal", "content": "Medical medication note is private"}
            )
        )
        assert journal["result"]["decision"] == "pending"
        journal_approval_id = journal["result"]["approval"]["id"]
        approved_journal = json.loads(
            first.handle_tool_call(
                "consolidating_memory", {"action": "approval", "fact_id": journal_approval_id, "approved": True}
            )
        )
        assert approved_journal["stored"]["sensitivity"] == "health"
        hidden_journal = json.loads(
            first.handle_tool_call(
                "consolidating_memory", {"action": "search", "query": "private note", "scope": "journals"}
            )
        )
        assert hidden_journal["results"]["journals"] == []
        visible_journal = json.loads(
            first.handle_tool_call(
                "consolidating_memory", {"action": "search", "query": "medication", "scope": "journals"}
            )
        )
        assert visible_journal["results"]["journals"]

        mark_gateway_user_dispatch(SimpleNamespace(internal=False))
        note_llm_turn(
            session_id="s1",
            user_message="My bank IBAN is FR00 PRIVATE",
            platform="gateway",
        )
        first.sync_turn("My bank IBAN is FR00 PRIVATE", "I understand", session_id="s1")
        first._task_queue.join()
        raw_episode = first._store._fetchone(
            "SELECT user_content, assistant_content, sensitivity FROM episodes ORDER BY id DESC"
        )
        assert "FR00" not in raw_episode["user_content"]
        assert "Sensitive user content omitted" in raw_episode["user_content"]
        assert raw_episode["sensitivity"] == "financial"
        recent_after_sensitive_turn = json.loads(first.handle_tool_call("consolidating_memory", {"action": "recent"}))
        assert all(row["sensitivity"] == "normal" for row in recent_after_sensitive_turn["results"]["episodes"])

        denied_credential = json.loads(
            first.handle_tool_call(
                "consolidating_memory",
                {"action": "remember", "content": "My password is hunter2", "category": "general"},
            )
        )
        assert denied_credential["result"]["action"] == "denied"
        assert first._store.search("hunter2", scope="facts", limit=5)["facts"] == []

        pending_before = len(first._store.list_approvals(status="pending"))
        first.on_memory_write(
            "add",
            "user",
            "I am allergic to penicillin",
            {"session_id": "s1", "execution_context": "primary"},
        )
        first._task_queue.join()
        assert len(first._store.list_approvals(status="pending")) == pending_before + 1
        assert first._store.search("penicillin", scope="facts", limit=5)["facts"] == []

        handoff = first.on_pre_compress([{"role": "user", "content": "My medical diagnosis is asthma"}])
        assert "asthma" not in handoff.lower()
        assert all(
            "asthma" not in str(row.get("summary") or "").lower()
            for row in first._store.recent_items(limit=20)["summaries"]
        )

        durable = first._durable_payload(
            "sync_turn",
            {
                "session_id": "s1",
                "user_content": "My IBAN is FR00 SECRET",
                "assistant_content": "noted",
                "messages": [{"role": "user", "content": "My IBAN is FR00 SECRET"}],
            },
        )
        assert "FR00" not in json.dumps(durable)

        outside_backup = tmp_path.parent / f"{tmp_path.name}-outside.db"
        refused_backup = json.loads(
            first.handle_tool_call("consolidating_memory", {"action": "backup", "destination": str(outside_backup)})
        )
        assert refused_backup["success"] is False
        allowed_backup = json.loads(
            first.handle_tool_call(
                "consolidating_memory",
                {"action": "backup", "destination": str(outside_backup), "confirm": True},
            )
        )
        assert allowed_backup["success"] is True and outside_backup.is_file()

        sensitive_working = json.loads(
            first.handle_tool_call(
                "consolidating_memory",
                {"action": "working", "key": "health", "content": "Track my medication schedule"},
            )
        )
        assert sensitive_working["result"]["decision"] == "pending"
        approved_working = json.loads(
            first.handle_tool_call(
                "consolidating_memory",
                {
                    "action": "approval",
                    "fact_id": sensitive_working["result"]["approval"]["id"],
                    "approved": True,
                },
            )
        )
        assert approved_working["stored"]["sensitivity"] == "health"
        hidden_working = json.loads(first.handle_tool_call("consolidating_memory", {"action": "working"}))
        assert all(row["memory_key"] != "health" for row in hidden_working["results"])
        visible_working = json.loads(
            first.handle_tool_call("consolidating_memory", {"action": "working", "query": "medication"})
        )
        assert any(row["memory_key"] == "health" for row in visible_working["results"])

        sensitive_procedure = json.loads(
            first.handle_tool_call(
                "consolidating_memory",
                {"action": "procedure", "key": "health-check", "steps": ["Review medication"]},
            )
        )
        assert sensitive_procedure["result"]["decision"] == "pending"
        sensitive_intention = json.loads(
            first.handle_tool_call(
                "consolidating_memory", {"action": "intention", "content": "Take medication tomorrow"}
            )
        )
        assert sensitive_intention["result"]["decision"] == "pending"

        status = json.loads(first.handle_tool_call("consolidating_memory", {"action": "status"}))
        assert status["pending_approvals"]
        assert all("candidate" not in row for row in status["pending_approvals"])
        recent = json.loads(first.handle_tool_call("consolidating_memory", {"action": "recent"}))
        assert all(str(row.get("sensitivity") or "normal") == "normal" for row in recent["results"]["facts"])

        working = json.loads(
            first.handle_tool_call(
                "consolidating_memory",
                {"action": "working", "key": "focus", "content": "Finish migration", "ttl_seconds": 30},
            )
        )
        assert working["success"]
        intention = json.loads(
            first.handle_tool_call(
                "consolidating_memory",
                {"action": "intention", "content": "Review the migration", "due_at": time.time() - 1},
            )
        )
        assert intention["result"]["status"] == "pending"
    finally:
        first.shutdown()
        second.shutdown()


def test_provider_fails_closed_when_multi_user_scope_has_no_identity(tmp_path):
    provider = ConsolidatingLocalMemoryProvider({"db_path": str(tmp_path / "memory.db"), "memory_scope": "user"})
    with pytest.raises(RuntimeError, match="requires user_id"):
        provider.initialize("s", hermes_home=str(tmp_path), platform="gateway", agent_context="primary")


def test_credentials_never_enter_raw_or_durable_storage_without_explicit_exception(tmp_path):
    provider = ConsolidatingLocalMemoryProvider(
        {
            "db_path": str(tmp_path / "memory.db"),
            "sensitive_memory": "allow",
            "allow_credential_memory": False,
        }
    )
    try:
        provider.initialize("s", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
        durable = provider._durable_payload(
            "remember_fact",
            {"content": "My password is ultra-secret", "category": "general"},
        )
        assert durable == {"_privacy_denied": True}
        assert "ultra-secret" not in json.dumps(durable)
        provider.sync_turn("My password is ultra-secret", "I will not retain it", session_id="s")
        provider._task_queue.join()
        raw = provider._store._fetchone("SELECT user_content, assistant_content FROM episodes ORDER BY id DESC")
        assert "ultra-secret" not in raw["user_content"]
        assert provider._store.search("ultra secret", scope="facts", limit=5)["facts"] == []

        bare_token = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
        assert provider._classify_sensitivity(bare_token)[0] == "credential"
        token_payload = provider._durable_payload("remember_fact", {"content": bare_token})
        assert token_payload == {"_privacy_denied": True}
        assert bare_token not in json.dumps(token_payload)
    finally:
        provider.shutdown()

    direct_store = MemoryStore(tmp_path / "legacy-import.db")
    try:
        bare_token = "ghp_abcdefghijklmnopqrstuvwxyz123456"
        direct_store.upsert_fact(
            content=f"Legacy unlabeled token {bare_token}",
            category="general",
            topic="legacy",
            source="legacy-import",
        )
        assert bare_token not in json.dumps(direct_store.export_data(redact_sensitive=True))
    finally:
        direct_store.close()


def test_provider_accepts_current_hermes_alternate_user_identity_and_registers_cli(tmp_path):
    provider = ConsolidatingLocalMemoryProvider({"db_path": str(tmp_path / "memory.db"), "memory_scope": "user"})
    try:
        provider.initialize(
            "s",
            hermes_home=str(tmp_path),
            platform="gateway",
            agent_context="primary",
            user_id_alt="alternate-user",
        )
        assert provider._scope_id.startswith("user:")
    finally:
        provider.shutdown()

    parser = ArgumentParser()
    register_cli(parser)
    args = parser.parse_args(["--db", str(tmp_path / "memory.db"), "doctor"])
    assert args.consolidating_local_command == "doctor"
    assert callable(args.func)


def test_full_in_memory_queue_can_be_spooled_without_blocking_shutdown(tmp_path):
    provider = ConsolidatingLocalMemoryProvider({"db_path": str(tmp_path / "memory.db"), "queue_max_size": 8})
    try:
        provider._store = MemoryStore(tmp_path / "memory.db")
        provider._task_queue = Queue(maxsize=8)
        for index in range(8):
            provider._task_queue.put_nowait(
                (
                    "remember_fact",
                    {
                        "content": f"Queued item {index}",
                        "category": "workflow",
                        "topic": "queue",
                        "source": "test",
                    },
                )
            )
        spooled = provider._spool_queued_tasks(preserve_sentinel=False)
        assert spooled == 8
        assert provider._task_queue.empty()
        assert provider._store is not None and provider._store.pending_operation_count() == 8
    finally:
        provider.shutdown()


def test_idle_worker_wakes_durable_operations_without_a_new_turn(tmp_path):
    provider = ConsolidatingLocalMemoryProvider({"db_path": str(tmp_path / "memory.db"), "queue_max_size": 8})
    try:
        provider.initialize("idle", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
        assert provider._store is not None
        provider._store.enqueue_operation(
            "remember_fact",
            {
                "content": "Idle durable work completed",
                "category": "workflow",
                "topic": "queue",
                "source": "test",
                "session_id": "idle",
            },
        )
        deadline = time.time() + 4
        while provider._store.pending_operation_count() and time.time() < deadline:
            time.sleep(0.05)
        assert provider._store.pending_operation_count() == 0
        assert provider._store.search("Idle durable work", scope="facts", limit=5)["facts"]
    finally:
        provider.shutdown()


def test_failed_in_memory_write_is_replayed_from_durable_queue(tmp_path):
    provider = ConsolidatingLocalMemoryProvider({"db_path": str(tmp_path / "memory.db"), "queue_max_size": 8})
    try:
        provider.initialize("replay", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
        provider._task_queue.join()
        original = provider._handle_remember_fact
        attempts = 0

        def fail_once(payload):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("transient dispatch failure")
            original(payload)

        provider._handle_remember_fact = fail_once
        assert provider._enqueue(
            "remember_fact",
            content="Recovered after a transient worker failure",
            category="workflow",
            topic="queue",
            source="test",
            session_id="replay",
        )
        deadline = time.time() + 5
        while time.time() < deadline:
            if (
                provider._store.pending_operation_count() == 0
                and provider._store.search("transient worker failure", scope="facts", limit=5)["facts"]
            ):
                break
            time.sleep(0.05)
        assert attempts >= 2
        assert provider._store.pending_operation_count() == 0
        assert provider._store.search("transient worker failure", scope="facts", limit=5)["facts"]
        assert provider._queue_metrics["spooled"] >= 1
    finally:
        provider.shutdown()


def test_sync_turn_replay_does_not_duplicate_episode_capture(tmp_path):
    provider = ConsolidatingLocalMemoryProvider({"db_path": str(tmp_path / "memory.db"), "queue_max_size": 8})
    try:
        provider.initialize("replay", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
        provider._task_queue.join()
        original_extract = provider._extract_turn_facts
        attempts = 0

        def fail_once(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("failure after episode commit")
            return original_extract(**kwargs)

        provider._extract_turn_facts = fail_once
        provider.sync_turn("This exact turn must only be captured once", "Acknowledged", session_id="replay")
        deadline = time.time() + 5
        while time.time() < deadline:
            if attempts >= 2 and provider._store.pending_operation_count() == 0:
                break
            time.sleep(0.05)
        assert attempts >= 2
        assert provider._store.pending_operation_count() == 0
        assert provider._store.counts()["episodes"] == 1
        episode = provider._store._fetchone("SELECT operation_key FROM episodes")
        assert episode and episode["operation_key"]
    finally:
        provider.shutdown()


def test_worker_closes_database_after_bounded_shutdown_returns(tmp_path):
    provider = ConsolidatingLocalMemoryProvider(
        {
            "db_path": str(tmp_path / "memory.db"),
            "queue_max_size": 8,
            "shutdown_timeout_seconds": 1,
        }
    )
    release = threading.Event()
    started = threading.Event()
    provider.initialize("slow", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
    original_dispatch = provider._dispatch_task

    def delayed_dispatch(kind, payload):
        if kind == "delayed-test":
            started.set()
            release.wait(timeout=5)
            return
        original_dispatch(kind, payload)

    provider._dispatch_task = delayed_dispatch
    provider._task_queue.put_nowait(("delayed-test", {}))
    assert started.wait(timeout=2)
    provider.shutdown()
    assert provider._store is not None
    release.set()
    assert provider._worker is not None
    provider._worker.join(timeout=3)
    assert not provider._worker.is_alive()
    assert provider._store is None


def test_retention_purges_sensitive_tombstones_but_preserves_pinned_facts(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        journal = store.add_journal(label="Private", content="Medical note", sensitivity="health")
        store.deactivate_memory_item("journal", journal["id"], reason="test", source="test")
        pinned = store.upsert_fact(
            content="Medical record retained by explicit pin",
            category="general",
            topic="private",
            source="user",
            sensitivity="health",
            pinned=True,
        )["fact"]
        store.deactivate_fact(pinned["id"], reason="test", source="test")
        inactive = store.upsert_fact(
            content="Obsolete unpinned memory",
            category="general",
            topic="obsolete",
            source="test",
        )["fact"]
        store.deactivate_fact(inactive["id"], reason="test", source="test")
        approval = store.request_approval(
            candidate={"content": "Old medical candidate"}, sensitivity="health", reason="test"
        )
        old = time.time() - (3 * 86400)
        very_old = time.time() - (200 * 86400)
        with store.transaction():
            store._execute("UPDATE memory_journals SET updated_at=? WHERE id=?", (old, journal["id"]))
            store._execute("UPDATE facts SET updated_at=? WHERE id=?", (old, pinned["id"]))
            store._execute("UPDATE facts SET updated_at=? WHERE id=?", (very_old, inactive["id"]))
            store._execute("UPDATE memory_approvals SET created_at=? WHERE id=?", (old, approval["id"]))
        stats = store.maintain(sensitive_retention_days=1)
        assert stats["sensitive_journals"] == 1
        assert stats["approvals"] == 1
        assert stats["inactive_facts"] == 1
        assert store._fetchone("SELECT id FROM memory_journals WHERE id=?", (journal["id"],)) is None
        assert store._fetchone("SELECT id FROM facts WHERE id=?", (pinned["id"],)) is not None
        assert store._fetchone("SELECT id FROM facts WHERE id=?", (inactive["id"],)) is None
        assert store.doctor()["ok"] is True
    finally:
        store.close()


def test_decay_deactivation_keeps_non_fact_fts_consistent(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        journal = store.add_journal(label="Transient", content="Temporary scratch note", importance=1, salience=0.1)
        store._execute(
            "UPDATE memory_journals SET updated_at=?, last_recalled_at=0 WHERE id=?",
            (time.time() - 86400, journal["id"]),
        )
        result = store.apply_decay(half_life_days=0.01, min_salience=0.5)
        assert result["journals_deactivated"] == 1
        assert store.doctor()["ok"] is True
    finally:
        store.close()


def test_doctor_repairs_non_fact_fts_and_dangling_links(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        preference = store.upsert_preference(
            key="editor", label="Editor", value="VS Code", metadata={"session_id": "s"}
        )
        with store.transaction():
            store._execute("DELETE FROM memory_preferences WHERE id=?", (preference["id"],))
        broken = store.doctor()
        assert broken["ok"] is False
        assert broken["dangling_links"] >= 1
        assert "memory_preferences_fts" in broken["fts_mismatches"]
        repaired = store.doctor(repair=True)
        assert repaired["ok"] is True
        assert repaired["dangling_links"] == 0
    finally:
        store.close()


def test_llm_circuit_breaker_opens_and_recovers_without_network():
    client = OpenAICompatibleLLM(model="test", base_url="http://127.0.0.1:1", failure_cooldown_seconds=1)
    client._record_failure()
    client._record_failure()
    client._record_failure()
    assert client.circuit_state["open"] is True
    client._record_success()
    assert client.circuit_state["open"] is False


def test_sensitive_text_requires_separate_opt_in_for_model_and_embedding_endpoints(tmp_path):
    provider = ConsolidatingLocalMemoryProvider(
        {
            "db_path": str(tmp_path / "memory.db"),
            "llm_model": "test-model",
            "llm_base_url": "http://model.invalid/v1",
            "retrieval_backend": "hybrid",
            "embedding_model": "test-embedding",
            "embedding_base_url": "http://embedding.invalid/v1",
            "sensitive_memory": "allow",
        }
    )
    try:
        provider.initialize("privacy", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
        provider._task_queue.join()
        model_calls = []
        embedding_calls = []

        def fake_chat_json(**kwargs):
            model_calls.append(kwargs["user_prompt"])
            return {"facts": []}

        def fake_embed(texts):
            embedding_calls.append(list(texts))
            return [[1.0, 0.0] for _ in texts]

        provider._llm.chat_json = fake_chat_json
        provider._embedder.embed_texts = fake_embed

        secret = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
        provider._extract_turn_facts(user_content=f"My token is {secret}", assistant_content="")
        assert model_calls == []
        provider._extract_turn_facts(user_content="My shell is PowerShell", assistant_content="")
        assert len(model_calls) == 1

        provider._store.upsert_fact(
            content="Medical diagnosis is private",
            category="general",
            topic="health",
            source="user",
            sensitivity="health",
        )
        provider._search_memory(
            "medical diagnosis",
            scope="facts",
            limit=5,
            session_id="privacy",
            allow_embeddings=True,
        )
        assert embedding_calls == []

        provider._config["allow_sensitive_model_processing"] = True
        provider._config["allow_credential_memory"] = True
        provider._extract_turn_facts(user_content=f"My token is {secret}", assistant_content="")
        assert len(model_calls) == 2
        provider._search_memory(
            "medical diagnosis",
            scope="facts",
            limit=5,
            session_id="privacy",
            allow_embeddings=True,
        )
        assert embedding_calls
    finally:
        provider.shutdown()
