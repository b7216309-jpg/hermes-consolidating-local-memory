from __future__ import annotations

import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from plugins.memory.consolidating_local import ConsolidatingLocalMemoryProvider
from plugins.memory.consolidating_local.consolidator import build_consolidation_plan
from plugins.memory.consolidating_local.store import MemoryStore, fingerprint_text, normalize_text, text_signature


def flush_provider(provider: ConsolidatingLocalMemoryProvider) -> None:
    provider._task_queue.join()  # type: ignore[attr-defined]
    time.sleep(0.05)


class ConsolidatingLocalMemoryTests(unittest.TestCase):
    def make_provider(self, tmpdir: str, **overrides: object) -> ConsolidatingLocalMemoryProvider:
        config = {
            "db_path": str(Path(tmpdir) / "memory.db"),
            "min_hours": 0,
            "min_sessions": 0,
            "scan_cooldown_seconds": 0,
            "prefetch_limit": 8,
            "max_topic_facts": 5,
            "topic_summary_chars": 650,
            "session_summary_chars": 900,
            "prune_after_days": 90,
            "episode_body_retention_hours": 24,
            "decay_half_life_days": 90,
            "decay_min_salience": 0.15,
            "extractor_backend": "heuristic",
            "retrieval_backend": "fts",
            "llm_timeout_seconds": 10,
            "llm_max_input_chars": 2000,
            "wiki_export_enabled": False,
            "wiki_export_on_consolidate": True,
            "wiki_export_session_limit": 50,
            "wiki_export_topic_limit": 100,
        }
        config.update(overrides)
        provider = ConsolidatingLocalMemoryProvider(config=config)
        provider.initialize("test-session")
        return provider

    def test_store_migrates_existing_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "legacy.db"
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE facts (
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
                    active INTEGER NOT NULL DEFAULT 1,
                    superseded_by INTEGER,
                    subject_key TEXT NOT NULL DEFAULT '',
                    value_key TEXT NOT NULL DEFAULT '',
                    polarity INTEGER NOT NULL DEFAULT 1,
                    exclusive INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL
                )
                """
            )
            now = time.time()
            content = "User prefers concise responses."
            conn.execute(
                """
                INSERT INTO facts(content, normalized_content, fingerprint, signature, category, topic, source, metadata_json, importance, confidence, active, subject_key, value_key, polarity, exclusive, created_at, updated_at, last_seen_at)
                VALUES (?, ?, ?, ?, 'user_pref', 'user-profile', 'legacy', '{}', 8, 0.9, 1, 'user:response_style', 'concise', 1, 1, ?, ?, ?)
                """,
                (
                    content,
                    normalize_text(content),
                    fingerprint_text(content),
                    text_signature(content),
                    now,
                    now,
                    now,
                ),
            )
            conn.commit()
            conn.close()

            store = MemoryStore(db_path=db_path)
            try:
                counts = store.counts()
                self.assertEqual(counts["facts"], 1)
                columns = {row["name"] for row in store._conn.execute("PRAGMA table_info(facts)").fetchall()}
                self.assertIn("salience", columns)
                self.assertIn("source_session_id", columns)
                self.assertIn("decay_half_life_days", columns)
                self.assertGreaterEqual(counts["sessions"], 0)
            finally:
                store.close()

    def test_store_backfills_missing_memory_sessions_from_existing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            store = MemoryStore(db_path=db_path)
            try:
                store.upsert_fact(
                    content="Project uses SQLite.",
                    category="project",
                    topic="database",
                    source="test",
                    importance=6,
                    confidence=0.8,
                    source_session_id="orphan-session",
                )
                store._execute("DELETE FROM memory_sessions WHERE session_id = ?", ("orphan-session",))
            finally:
                store.close()

            reopened = MemoryStore(db_path=db_path)
            try:
                session = reopened._fetchone(
                    "SELECT session_id FROM memory_sessions WHERE session_id = ?",
                    ("orphan-session",),
                )
                self.assertIsNotNone(session)
            finally:
                reopened.close()

    def test_lifecycle_creates_traces_summaries_history_and_preferences(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.make_provider(tmpdir)
            try:
                provider.sync_turn(
                    "I prefer concise answers and I live in Paris.",
                    "Understood, I will keep replies concise.",
                )
                flush_provider(provider)
                provider.on_session_end(
                    [
                        {"role": "user", "content": "I prefer concise answers and I live in Paris."},
                        {"role": "assistant", "content": "Understood."},
                    ]
                )
                flush_provider(provider)
                provider.on_memory_write("write", "user", "User likes jasmine tea.")
                provider.on_delegation("Collect CI logs", "Captured the latest failing job output.", child_session_id="child-1")
                flush_provider(provider)

                store = provider._store
                assert store is not None
                counts = store.counts()
                self.assertGreaterEqual(counts["traces"], 1)
                self.assertGreaterEqual(counts["summaries"], 1)
                self.assertGreaterEqual(counts["preferences"], 1)
                self.assertGreater(counts["history"], 0)
                delegation = store.search("failing job output", scope="facts", limit=5)
                self.assertTrue(any(item["topic"] == "delegation-results" for item in delegation.get("facts", [])))
            finally:
                provider.shutdown()

    def test_contradictions_create_history_and_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.make_provider(tmpdir)
            try:
                store = provider._store
                assert store is not None
                first = store.upsert_fact(
                    content="User prefers concise responses.",
                    category="user_pref",
                    topic="user-profile",
                    source="test",
                    importance=8,
                    confidence=0.9,
                    metadata={"subject_key": "user:response_style", "value_key": "concise", "exclusive": True},
                    source_session_id="test-session",
                )
                second = store.upsert_fact(
                    content="User prefers detailed responses.",
                    category="user_pref",
                    topic="user-profile",
                    source="test",
                    importance=8,
                    confidence=0.9,
                    metadata={"subject_key": "user:response_style", "value_key": "detailed", "exclusive": True},
                    source_session_id="test-session",
                )

                self.assertEqual(first["action"], "inserted")
                self.assertEqual(second["action"], "inserted")
                contradictions = store.recent_contradictions(limit=5)
                self.assertTrue(any(item["subject_key"] == "user:response_style" for item in contradictions))
                history = store.list_history(memory_type="fact", subject_key="user:response_style", limit=10)
                self.assertTrue(any(item["action"] == "superseded" for item in history))
                links = store._fetchall(
                    "SELECT link_type FROM memory_links WHERE source_kind = 'fact' AND target_kind = 'fact'"
                )
                self.assertTrue(any(link["link_type"] in {"supersedes", "contradicts"} for link in links))
            finally:
                provider.shutdown()

    def test_tool_actions_cover_journal_distill_policy_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.make_provider(tmpdir)
            try:
                provider.sync_turn("The project uses PostgreSQL.", "Noted.")
                flush_provider(provider)
                provider.on_session_end([{"role": "user", "content": "The project uses PostgreSQL."}])
                flush_provider(provider)

                journal = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {"action": "journal", "label": "Daily note", "content": "Investigated the database shape."},
                    )
                )
                self.assertTrue(journal["success"])

                policy = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {"action": "policy", "key": "retention", "label": "Retention", "content": "Keep handoff summaries for longer than raw buffers."},
                    )
                )
                self.assertTrue(policy["success"])

                distill = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {"action": "distill", "session_id": "test-session", "label": "Checkpoint"},
                    )
                )
                self.assertTrue(distill["success"])

                history = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {"action": "history", "memory_type": "summary", "limit": 10},
                    )
                )
                self.assertTrue(history["success"])
                self.assertTrue(history["results"])
            finally:
                provider.shutdown()

    def test_review_action_surfaces_due_items_and_advances_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.make_provider(tmpdir)
            try:
                store = provider._store
                assert store is not None
                fact = store.upsert_fact(
                    content="User prefers concise responses.",
                    category="user_pref",
                    topic="user-profile",
                    source="test",
                    importance=8,
                    confidence=0.9,
                    metadata={"subject_key": "user:response_style", "value_key": "concise", "exclusive": True},
                    source_session_id="test-session",
                )["fact"]
                store._execute(
                    "UPDATE facts SET next_review_at = ?, review_count = 0, reconsolidation_until = 0 WHERE id = ?",
                    (time.time() - 60, int(fact["id"])),
                )

                review = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {"action": "review", "scope": "facts", "limit": 5},
                    )
                )
                self.assertTrue(review["success"])
                self.assertTrue(review["results"]["facts"])
                self.assertIn("review_prompt", review["results"]["facts"][0])

                refreshed = store._fetchone(
                    "SELECT review_count, next_review_at, reconsolidation_until FROM facts WHERE id = ?",
                    (int(fact["id"]),),
                )
                assert refreshed is not None
                self.assertEqual(int(refreshed["review_count"]), 1)
                self.assertGreater(float(refreshed["next_review_at"]), time.time())
                self.assertGreater(float(refreshed["reconsolidation_until"]), time.time())
                history = store.list_history(memory_type="fact", entity_id=fact["id"], limit=5)
                self.assertTrue(any(item["action"] == "recalled" for item in history))
            finally:
                provider.shutdown()

    def test_recalled_fact_reconsolidates_on_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=Path(tmpdir) / "memory.db")
            try:
                fact = store.upsert_fact(
                    content="Project deploys with Docker Compose.",
                    category="project",
                    topic="delivery",
                    source="test",
                    importance=7,
                    confidence=0.8,
                    metadata={"subject_key": "project:deploy_method", "value_key": "docker-compose", "exclusive": True},
                    source_session_id="session-a",
                )["fact"]
                store.touch_recall(
                    "fact",
                    [fact["id"]],
                    session_id="session-a",
                    review_intervals_days=[1.0, 3.0, 7.0],
                    reconsolidation_window_hours=6.0,
                    cues={"query": "docker compose"},
                )

                store.upsert_fact(
                    content="Project deploys with Docker Compose.",
                    category="project",
                    topic="delivery",
                    source="test",
                    importance=7,
                    confidence=0.8,
                    metadata={"subject_key": "project:deploy_method", "value_key": "docker-compose", "exclusive": True},
                    observed_at=time.time() + 1,
                    source_session_id="session-a",
                    history_reason="retest",
                )

                history = store.list_history(memory_type="fact", entity_id=fact["id"], limit=6)
                self.assertTrue(any(item["action"] == "reconsolidated" for item in history))
            finally:
                store.close()

    def test_search_uses_session_cues_for_reranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.make_provider(tmpdir)
            try:
                store = provider._store
                assert store is not None
                store.upsert_fact(
                    content="Redis handles queued background jobs.",
                    category="project",
                    topic="infrastructure",
                    source="test",
                    importance=6,
                    confidence=0.8,
                    source_session_id="focus-session",
                )
                time.sleep(0.01)
                store.upsert_fact(
                    content="Redis stores analytics cache snapshots.",
                    category="project",
                    topic="infrastructure",
                    source="test",
                    importance=6,
                    confidence=0.8,
                    source_session_id="other-session",
                )

                search = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {
                            "action": "search",
                            "query": "redis",
                            "scope": "facts",
                            "session_id": "focus-session",
                        },
                    )
                )
                self.assertTrue(search["success"])
                facts = search["results"]["facts"]
                self.assertGreaterEqual(len(facts), 2)
                self.assertEqual(facts[0]["source_session_id"], "focus-session")
                self.assertGreater(float(facts[0]["cue_match_score"]), 0.0)
            finally:
                provider.shutdown()

    def test_blank_search_does_not_count_as_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.make_provider(tmpdir)
            try:
                remember = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {
                            "action": "remember",
                            "content": "Favorite language is Python.",
                            "category": "user_pref",
                            "subject_key": "user:favorite_language",
                            "session_id": "blank-search-session",
                        },
                    )
                )
                self.assertTrue(remember["success"])
                store = provider._store
                assert store is not None
                fact_id = int(remember["result"]["fact"]["id"])
                before = store._fetchone(
                    "SELECT review_count, next_review_at FROM facts WHERE id = ?",
                    (fact_id,),
                )
                assert before is not None

                search = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {"action": "search", "query": "", "scope": "facts", "session_id": "blank-search-session"},
                    )
                )
                self.assertTrue(search["success"])

                after = store._fetchone(
                    "SELECT review_count, next_review_at FROM facts WHERE id = ?",
                    (fact_id,),
                )
                assert after is not None
                self.assertEqual(int(after["review_count"]), int(before["review_count"]))
                self.assertEqual(float(after["next_review_at"]), float(before["next_review_at"]))
                history = store.list_history(memory_type="fact", entity_id=fact_id, limit=5)
                self.assertFalse(any(item["action"] == "recalled" for item in history))
            finally:
                provider.shutdown()

    def test_touch_recall_advances_to_next_review_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=Path(tmpdir) / "memory.db")
            try:
                fact = store.upsert_fact(
                    content="Project uses Redis for cache.",
                    category="project",
                    topic="infrastructure",
                    source="test",
                    importance=6,
                    confidence=0.8,
                    source_session_id="session-a",
                )["fact"]
                before = store._fetchone(
                    "SELECT review_count, next_review_at FROM facts WHERE id = ?",
                    (int(fact["id"]),),
                )
                assert before is not None

                store.touch_recall(
                    "fact",
                    [fact["id"]],
                    session_id="session-a",
                    review_intervals_days=[1.0, 3.0, 7.0],
                    reconsolidation_window_hours=6.0,
                    cues={"query": "redis"},
                )
                after = store._fetchone(
                    "SELECT review_count, next_review_at FROM facts WHERE id = ?",
                    (int(fact["id"]),),
                )
                assert after is not None
                self.assertEqual(int(after["review_count"]), 1)
                self.assertGreater(float(after["next_review_at"]) - float(before["next_review_at"]), 1.5 * 86400.0)
            finally:
                store.close()

    def test_generic_query_does_not_bias_toward_general_cues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.make_provider(tmpdir)
            try:
                remember_project = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {
                            "action": "remember",
                            "content": "Redis is used for cache.",
                            "category": "project",
                            "topic": "cache",
                            "importance": 10,
                            "session_id": "project-session",
                        },
                    )
                )
                remember_general = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {
                            "action": "remember",
                            "content": "Redis glossary note.",
                            "category": "general",
                            "topic": "general",
                            "importance": 1,
                            "session_id": "general-session",
                        },
                    )
                )
                self.assertTrue(remember_project["success"])
                self.assertTrue(remember_general["success"])

                search = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {"action": "search", "query": "redis", "scope": "facts", "session_id": "neutral-session"},
                    )
                )
                self.assertTrue(search["success"])
                facts = search["results"]["facts"]
                self.assertGreaterEqual(len(facts), 2)
                self.assertEqual(facts[0]["category"], "project")
                self.assertTrue(all(float(item["cue_match_score"]) == 0.0 for item in facts[:2]))
            finally:
                provider.shutdown()

    def test_forget_lookup_does_not_mark_memory_as_recalled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.make_provider(tmpdir)
            try:
                policy = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {
                            "action": "policy",
                            "key": "reply-style",
                            "content": "Prefer concise replies.",
                            "session_id": "policy-session",
                        },
                    )
                )
                self.assertTrue(policy["success"])

                forgot = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {
                            "action": "forget",
                            "memory_type": "policy",
                            "query": "reply-style",
                            "session_id": "policy-session",
                        },
                    )
                )
                self.assertTrue(forgot["success"])

                store = provider._store
                assert store is not None
                row = store._fetchone(
                    "SELECT id, review_count, active FROM memory_policies WHERE policy_key = ?",
                    ("reply-style",),
                )
                assert row is not None
                self.assertEqual(int(row["review_count"]), 0)
                self.assertEqual(int(row["active"]), 0)
                history = store.list_history(memory_type="policy", entity_id=int(row["id"]), limit=10)
                self.assertFalse(any(item["action"] == "recalled" for item in history))
            finally:
                provider.shutdown()

    def test_touch_recall_ignores_missing_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=Path(tmpdir) / "memory.db")
            try:
                store.touch_recall(
                    "fact",
                    ["9999"],
                    session_id="session-a",
                    review_intervals_days=[1.0, 3.0, 7.0],
                    reconsolidation_window_hours=6.0,
                    cues={"query": "missing"},
                )
                history = store._fetchall(
                    "SELECT entity_kind, entity_id, action FROM memory_history WHERE entity_kind = 'fact'"
                )
                links = store._fetchall(
                    "SELECT source_kind, target_kind, target_id, link_type FROM memory_links WHERE target_kind = 'fact'"
                )
                self.assertEqual(history, [])
                self.assertEqual(links, [])
            finally:
                store.close()

    def test_consolidation_prunes_episode_buffers_and_keeps_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.make_provider(tmpdir, episode_body_retention_hours=0)
            try:
                provider.sync_turn("I like Vim.", "Noted.")
                flush_provider(provider)
                provider.on_session_end([{"role": "user", "content": "I like Vim."}])
                flush_provider(provider)

                result = json.loads(provider.handle_tool_call("consolidating_memory", {"action": "consolidate"}))
                self.assertTrue(result["success"])
                store = provider._store
                assert store is not None
                counts = store.counts()
                self.assertEqual(counts["episodes"], 0)
                self.assertGreaterEqual(counts["summaries"], 1)
            finally:
                provider.shutdown()

    def test_hybrid_retrieval_falls_back_to_fts_and_decay_updates_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.make_provider(tmpdir, retrieval_backend="hybrid", embedding_model="", embedding_base_url="")
            try:
                remember = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {
                            "action": "remember",
                            "content": "Project deploys with Docker Compose.",
                            "category": "project",
                            "topic": "project-delivery",
                        },
                    )
                )
                self.assertTrue(remember["success"])

                search = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {"action": "search", "query": "docker compose", "scope": "facts"},
                    )
                )
                self.assertTrue(search["success"])
                self.assertTrue(search["results"]["facts"])

                decay = json.loads(provider.handle_tool_call("consolidating_memory", {"action": "decay"}))
                self.assertTrue(decay["success"])
                store = provider._store
                assert store is not None
                self.assertTrue(store.get_state("last_decay_at", ""))
            finally:
                provider.shutdown()

    def test_session_scoped_tool_writes_create_real_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.make_provider(tmpdir)
            try:
                remember_fact = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {
                            "action": "remember",
                            "content": "Project uses Redis.",
                            "category": "project",
                            "topic": "infra",
                            "session_id": "fact-session",
                        },
                    )
                )
                remember_pref = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {
                            "action": "remember",
                            "memory_type": "preference",
                            "label": "Tone",
                            "value": "Concise",
                            "session_id": "pref-session",
                        },
                    )
                )
                policy = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {
                            "action": "policy",
                            "key": "retention",
                            "label": "Retention",
                            "content": "Keep summaries.",
                            "session_id": "policy-session",
                        },
                    )
                )
                self.assertTrue(remember_fact["success"])
                self.assertTrue(remember_pref["success"])
                self.assertTrue(policy["success"])

                store = provider._store
                assert store is not None
                session_rows = store._fetchall("SELECT session_id FROM memory_sessions ORDER BY session_id ASC")
                self.assertEqual(
                    [row["session_id"] for row in session_rows],
                    ["fact-session", "policy-session", "pref-session", "test-session"],
                )
                fact_artifacts = store.get_session_artifacts("fact-session")
                pref_artifacts = store.get_session_artifacts("pref-session")
                policy_artifacts = store.get_session_artifacts("policy-session")
                self.assertEqual(len(fact_artifacts["facts"]), 1)
                self.assertEqual(fact_artifacts["facts"][0]["source_session_id"], "fact-session")
                self.assertEqual(len(pref_artifacts["preferences"]), 1)
                self.assertEqual(pref_artifacts["preferences"][0]["source_session_id"], "pref-session")
                self.assertEqual(len(policy_artifacts["policies"]), 1)
                self.assertEqual(policy_artifacts["policies"][0]["source_session_id"], "policy-session")
            finally:
                provider.shutdown()

    def test_mirror_remove_rebuilds_topics_and_preferences(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.make_provider(tmpdir)
            try:
                provider.on_memory_write("write", "user", "User likes jasmine tea.")
                flush_provider(provider)

                store = provider._store
                assert store is not None
                self.assertEqual(store.counts()["topics"], 1)
                self.assertEqual(store.counts()["preferences"], 1)

                provider.on_memory_write("remove", "user", "User likes jasmine tea.")
                flush_provider(provider)

                counts = store.counts()
                self.assertEqual(counts["facts"], 0)
                self.assertEqual(counts["topics"], 0)
                self.assertEqual(counts["preferences"], 0)
            finally:
                provider.shutdown()

    def test_decay_uses_incremental_anchor_instead_of_compounding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=Path(tmpdir) / "memory.db")
            try:
                observed_at = time.time() - (10 * 86400)
                store.upsert_fact(
                    content="Temporary note about a minor detail.",
                    category="general",
                    topic="notes",
                    source="test",
                    importance=3,
                    confidence=0.5,
                    observed_at=observed_at,
                    salience=1.0,
                    decay_half_life_days=10,
                )

                first = store.apply_decay(half_life_days=10, min_salience=0.0)
                self.assertEqual(first["facts_decayed"], 1)
                first_salience = float(store._fetchone("SELECT salience FROM facts WHERE id = 1")["salience"])

                second = store.apply_decay(half_life_days=10, min_salience=0.0)
                self.assertEqual(second["facts_decayed"], 1)
                second_salience = float(store._fetchone("SELECT salience FROM facts WHERE id = 1")["salience"])

                self.assertGreater(first_salience, 0.45)
                self.assertLess(first_salience, 0.55)
                self.assertGreater(second_salience, first_salience * 0.95)
            finally:
                store.close()

    def test_blank_scoped_search_and_forget_are_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.make_provider(tmpdir)
            try:
                journal = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {"action": "journal", "label": "Daily note", "content": "Investigated the database shape."},
                    )
                )
                self.assertTrue(journal["success"])

                scoped = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {"action": "search", "scope": "summaries", "limit": 1},
                    )
                )
                self.assertTrue(scoped["success"])
                self.assertEqual(scoped["results"]["summaries"], [])
                self.assertEqual(scoped["results"]["journals"], [])
                self.assertNotIn("contradictions", scoped["results"])

                forget = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {"action": "forget", "memory_type": "journal"},
                    )
                )
                self.assertFalse(forget["success"])

                recent = json.loads(provider.handle_tool_call("consolidating_memory", {"action": "recent"}))
                self.assertEqual(len(recent["results"]["journals"]), 1)
            finally:
                provider.shutdown()

    def test_consolidation_preserves_session_links_and_builds_session_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.make_provider(tmpdir)
            try:
                plan = build_consolidation_plan(provider._store, min_hours=0, min_sessions=0)  # type: ignore[arg-type]
                self.assertFalse(plan["should_run"])

                provider.sync_turn("I prefer concise answers.", "Understood.")
                flush_provider(provider)

                result = json.loads(provider.handle_tool_call("consolidating_memory", {"action": "consolidate"}))
                self.assertTrue(result["success"])
                self.assertGreaterEqual(result["result"]["session_summaries"], 1)

                store = provider._store
                assert store is not None
                facts = store._fetchall("SELECT id, source_session_id FROM facts")
                self.assertEqual(facts[0]["source_session_id"], "test-session")
                links = store._fetchall(
                    """
                    SELECT source_kind, target_kind, target_id, link_type
                    FROM memory_links
                    WHERE source_kind = 'fact'
                    ORDER BY id ASC
                    """
                )
                self.assertTrue(any(link["link_type"] == "derived_from_episode" for link in links))
                self.assertTrue(any(link["link_type"] == "captured_in" and link["target_kind"] == "session" for link in links))
                summaries = store.latest_session_summaries(limit=5)
                self.assertTrue(summaries)
                self.assertEqual(len(summaries), 1)

                store.set_state("last_consolidated_episode_id", 0)
                second = json.loads(provider.handle_tool_call("consolidating_memory", {"action": "consolidate"}))
                self.assertTrue(second["success"])
                self.assertGreaterEqual(second["result"]["session_summaries"], 1)
                self.assertEqual(len(store.latest_session_summaries(limit=5)), 1)
            finally:
                provider.shutdown()

    def test_prefetch_surfaces_policies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.make_provider(tmpdir)
            try:
                policy = json.loads(
                    provider.handle_tool_call(
                        "consolidating_memory",
                        {
                            "action": "policy",
                            "key": "retention",
                            "label": "Retention",
                            "content": "Keep handoff summaries longer than raw buffers.",
                        },
                    )
                )
                self.assertTrue(policy["success"])
                rendered = provider.prefetch("handoff summaries")
                self.assertIn("Active preferences and workflow rules:", rendered)
                self.assertIn("Keep handoff summaries longer than raw buffers.", rendered)
            finally:
                provider.shutdown()

    def test_export_writes_compiled_wiki_and_prunes_stale_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_dir = Path(tmpdir) / "wiki"
            provider = self.make_provider(
                tmpdir,
                wiki_export_enabled=True,
                wiki_export_dir=str(wiki_dir),
            )
            try:
                provider.sync_turn(
                    "I prefer concise answers and the project uses PostgreSQL.",
                    "Noted, I will keep replies concise.",
                )
                flush_provider(provider)
                provider.on_session_end(
                    [
                        {"role": "user", "content": "I prefer concise answers and the project uses PostgreSQL."},
                        {"role": "assistant", "content": "Noted."},
                    ]
                )
                flush_provider(provider)
                provider.handle_tool_call(
                    "consolidating_memory",
                    {"action": "policy", "key": "retention", "label": "Retention", "content": "Keep handoff summaries longer than raw buffers."},
                )

                export = json.loads(provider.handle_tool_call("consolidating_memory", {"action": "export"}))
                self.assertTrue(export["success"])
                result = export["result"]
                self.assertGreater(result["generated_files"], 0)

                index_path = wiki_dir / "index.md"
                session_pages = sorted((wiki_dir / "sessions").glob("*.md"))
                topic_pages = sorted((wiki_dir / "topics").glob("*.md"))
                self.assertTrue(index_path.exists())
                self.assertTrue(session_pages)
                self.assertTrue(topic_pages)
                self.assertTrue((wiki_dir / "preferences" / "index.md").exists())
                self.assertTrue((wiki_dir / "policies" / "index.md").exists())
                self.assertTrue((wiki_dir / "contradictions" / "index.md").exists())

                index_text = index_path.read_text(encoding="utf-8")
                self.assertIn("Compiled Memory Wiki", index_text)
                self.assertIn("topics/", index_text)
                self.assertIn("sessions/", index_text)

                session_text = session_pages[0].read_text(encoding="utf-8")
                self.assertIn("## Facts", session_text)
                self.assertIn("../topics/", session_text)

                topic_text = topic_pages[0].read_text(encoding="utf-8")
                self.assertIn("## Related Sessions", topic_text)
                self.assertIn("../sessions/", topic_text)

                stale = wiki_dir / "topics" / "stale.md"
                stale.write_text("old page\n", encoding="utf-8")
                second = json.loads(provider.handle_tool_call("consolidating_memory", {"action": "export"}))
                self.assertTrue(second["success"])
                self.assertEqual(second["result"]["written_files"], 0)
                self.assertEqual(second["result"]["pruned_files"], 1)
                self.assertFalse(stale.exists())
            finally:
                provider.shutdown()

    def test_consolidation_auto_exports_and_status_reports_wiki_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_dir = Path(tmpdir) / "wiki"
            provider = self.make_provider(
                tmpdir,
                wiki_export_enabled=True,
                wiki_export_on_consolidate=True,
                wiki_export_dir=str(wiki_dir),
            )
            try:
                provider.sync_turn("I like jasmine tea.", "Noted.")
                flush_provider(provider)

                consolidate = json.loads(provider.handle_tool_call("consolidating_memory", {"action": "consolidate"}))
                self.assertTrue(consolidate["success"])
                self.assertIn("wiki_export", consolidate["result"])
                self.assertTrue((wiki_dir / "index.md").exists())

                status = json.loads(provider.handle_tool_call("consolidating_memory", {"action": "status"}))
                self.assertTrue(status["success"])
                wiki_status = status["wiki_export"]
                self.assertTrue(wiki_status["enabled"])
                self.assertEqual(Path(wiki_status["root"]), wiki_dir)
                self.assertTrue(wiki_status["last_export_at"])
                self.assertGreater(wiki_status["last_export_stats"]["generated_files"], 0)
            finally:
                provider.shutdown()

    def test_tool_schema_exposes_v2_actions_and_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.make_provider(tmpdir)
            try:
                schemas = provider.get_tool_schemas()
                self.assertEqual(len(schemas), 1)
                params = schemas[0]["parameters"]["properties"]
                actions = set(params["action"]["enum"])
                scopes = set(params["scope"]["enum"])
                self.assertTrue(
                    {
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
                    }.issubset(actions)
                )
                self.assertTrue(
                    {
                        "all",
                        "facts",
                        "topics",
                        "episodes",
                        "summaries",
                        "journals",
                        "preferences",
                        "policies",
                    }.issubset(scopes)
                )
            finally:
                provider.shutdown()


if __name__ == "__main__":
    unittest.main()
