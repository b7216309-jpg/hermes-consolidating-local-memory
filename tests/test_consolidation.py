from __future__ import annotations

from consolidating_local.consolidation import message_content_text, normalize_candidate_fact, run_consolidation
from consolidating_local.store import MemoryStore


def test_message_content_text_accepts_supported_text_blocks_only():
    assert (
        message_content_text(
            [
                {"type": "input_text", "input_text": "first"},
                {"type": "output_text", "output_text": "second"},
                {"type": "image", "url": "ignored"},
                "third",
            ]
        )
        == "first second third"
    )
    assert message_content_text({"text": "not-a-list"}) == ""


def test_candidate_normalization_validates_model_output_without_content_guessing():
    candidate = normalize_candidate_fact(
        {
            "content": "Project database is SQLite",
            "category": "invalid-category",
            "topic": "Primary Database",
            "subject_key": "project:database",
            "value_key": "sqlite",
            "exclusive": "false",
            "polarity": "NEGATIVE",
            "importance": 99,
            "confidence": float("nan"),
        },
        source_role="user",
    )
    assert candidate is not None
    assert candidate["content"] == "Project database is SQLite"
    assert candidate["category"] == "general"
    assert candidate["topic"] == "primary-database"
    assert candidate["importance"] == 10
    assert candidate["confidence"] == 0.75
    assert candidate["metadata"]["subject_key"] == "project:database"
    assert "exclusive" not in candidate["metadata"]
    assert candidate["metadata"]["polarity"] == -1


def test_consolidation_has_no_implicit_fact_extractor(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        store.append_episode(session_id="session", user_content="My name is Alice", assistant_content="Okay")
        result = run_consolidation(
            store,
            min_hours=0,
            min_sessions=1,
            max_topic_facts=5,
            topic_summary_chars=500,
            prune_after_days=90,
            episode_retention_hours=99999,
            force=True,
            reason="test",
        )
        assert result["episodes_scanned"] == 1
        assert result["facts_added"] == 0
        assert store.counts()["facts"] == 0
    finally:
        store.close()


def test_consolidation_uses_only_an_explicit_extractor(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        store.append_episode(session_id="session", user_content="My name is Alice", assistant_content="Okay")

        def extractor(**_):
            return [
                {
                    "content": "User's name is Alice",
                    "category": "user_pref",
                    "topic": "user-profile",
                    "importance": 8,
                    "confidence": 0.95,
                    "metadata": {"subject_key": "user:name", "value_key": "alice", "exclusive": True},
                }
            ]

        result = run_consolidation(
            store,
            min_hours=0,
            min_sessions=1,
            max_topic_facts=5,
            topic_summary_chars=500,
            prune_after_days=90,
            episode_retention_hours=99999,
            extractor=extractor,
            force=True,
            reason="test",
        )
        assert result["facts_added"] == 1
        assert store.search("Alice", scope="facts")["facts"]
    finally:
        store.close()


def test_consolidation_does_not_skip_episodes_after_batch_limit(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        for number in range(501):
            store.append_episode(session_id="session", user_content=f"turn {number}", assistant_content="ok")
        kwargs = {
            "min_hours": 0,
            "min_sessions": 1,
            "max_topic_facts": 5,
            "topic_summary_chars": 500,
            "prune_after_days": 90,
            "episode_retention_hours": 99999,
            "force": True,
            "reason": "test",
        }
        first = run_consolidation(store, **kwargs)
        assert first["episodes_scanned"] == 500
        assert first["processed_episode_id"] == 500
        assert first["backlog_remaining"] == 1

        second = run_consolidation(store, **kwargs)
        assert second["episodes_scanned"] == 1
        assert second["processed_episode_id"] == 501
        assert second["backlog_remaining"] == 0
    finally:
        store.close()
