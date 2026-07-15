from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from consolidating_local import ConsolidatingLocalMemoryProvider
from consolidating_local.consolidation import normalize_candidate_fact
from consolidating_local.store import MemoryStore


def test_model_temporal_fields_are_normalized_and_one_time_schedules_expire():
    reference = datetime(2026, 7, 15, 9, 0, tzinfo=UTC).timestamp()
    candidate = normalize_candidate_fact(
        {
            "content": "The user is welding train parts tomorrow",
            "category": "general",
            "topic": "work",
            "temporal_kind": "scheduled",
            "event_at": "2026-07-16",
            "temporal_precision": "day",
            "temporal_timezone": "Europe/Paris",
            "temporal_confidence": 0.95,
            "reference_unix_time": reference,
            "reference_timezone": "Europe/Paris",
        },
        source_role="user",
    )

    assert candidate is not None
    metadata = candidate["metadata"]
    assert metadata["temporal_kind"] == "scheduled"
    assert metadata["temporal_precision"] == "day"
    assert metadata["temporal_timezone"] == "Europe/Paris"
    assert metadata["event_at"] > reference
    assert metadata["valid_until"] > metadata["event_at"]


def test_temporal_fact_creates_a_persistent_timeline_and_renders_age(tmp_path):
    provider = ConsolidatingLocalMemoryProvider({"db_path": str(tmp_path / "memory.db"), "timezone": "Europe/Paris"})
    try:
        provider.initialize("temporal", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
        observed = datetime(2026, 7, 15, 7, 0, tzinfo=UTC).timestamp()
        candidate = normalize_candidate_fact(
            {
                "content": "The user welded SNCF train parts",
                "category": "general",
                "topic": "work",
                "temporal_kind": "event",
                "event_at": "2026-07-14T22:25:00+02:00",
                "temporal_precision": "minute",
                "temporal_timezone": "Europe/Paris",
                "temporal_confidence": 0.98,
                "reference_unix_time": observed,
            },
            source_role="user",
        )
        result = provider._store_candidate(candidate, source="test", session_id="temporal", observed_at=observed)
        fact = result["fact"]
        events = provider._store.list_autobiographical_events("SNCF", limit=5)

        assert fact["temporal_kind"] == "event"
        assert fact["memory_class"] == "autobiographical"
        assert fact["event_at"] > 0
        assert len(events) == 1
        annotation = provider._temporal_annotation("facts", fact, now_timestamp=observed)
        assert "event 2026-07-14 22:25 CEST" in annotation
        assert "hours ago" in annotation
        timeline_annotation = provider._temporal_annotation("timeline", events[0], now_timestamp=observed)
        assert "event 2026-07-14 22:25 CEST" in timeline_annotation
    finally:
        provider.shutdown()


def test_scheduled_fact_expires_as_current_state_but_timeline_survives(tmp_path):
    provider = ConsolidatingLocalMemoryProvider({"db_path": str(tmp_path / "memory.db"), "timezone": "Europe/Paris"})
    try:
        provider.initialize("scheduled", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
        now = datetime.now(UTC)
        candidate = normalize_candidate_fact(
            {
                "content": "The user has a one-time welding shift tomorrow",
                "category": "general",
                "topic": "work",
                "temporal_kind": "scheduled",
                "event_at": (now + timedelta(days=1)).isoformat(),
                "temporal_precision": "minute",
                "temporal_timezone": "Europe/Paris",
                "reference_unix_time": now.timestamp(),
            },
            source_role="user",
        )
        result = provider._store_candidate(
            candidate,
            source="test",
            session_id="scheduled",
            observed_at=now.timestamp(),
        )
        fact = result["fact"]
        assert provider._store.search("welding shift", scope="facts")["facts"]

        provider._store._execute("UPDATE facts SET valid_until=? WHERE id=?", (now.timestamp() - 1, fact["id"]))
        assert provider._store.search("welding shift", scope="facts")["facts"] == []
        timeline = provider._store.list_autobiographical_events("welding shift")
        assert len(timeline) == 1
        assert timeline[0]["event_at"] > now.timestamp()
    finally:
        provider.shutdown()


def test_direct_remember_scheduled_fact_links_timeline_and_sets_expiry(tmp_path):
    provider = ConsolidatingLocalMemoryProvider({"db_path": str(tmp_path / "memory.db"), "timezone": "Europe/Paris"})
    try:
        provider.initialize("tool-time", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
        event_at = datetime.now(UTC).timestamp() + 3600
        result = provider._remember_from_tool(
            {
                "content": "The user has a synthetic inspection in one hour",
                "category": "general",
                "topic": "tests",
                "temporal_kind": "scheduled",
                "event_at": event_at,
                "temporal_precision": "minute",
                "temporal_timezone": "Europe/Paris",
            },
            session_id="tool-time",
        )
        fact = result["fact"]
        timeline = provider._store.list_autobiographical_events("synthetic inspection")

        assert fact["temporal_kind"] == "scheduled"
        assert fact["valid_until"] > fact["event_at"]
        assert len(timeline) == 1
        assert timeline[0]["event_key"] == f"fact-{fact['id']}"
    finally:
        provider.shutdown()


def test_unknown_event_date_is_not_fabricated_from_observation_time(tmp_path):
    provider = ConsolidatingLocalMemoryProvider({"db_path": str(tmp_path / "memory.db"), "timezone": "Europe/Paris"})
    try:
        provider.initialize("unknown-event", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
        candidate = normalize_candidate_fact(
            {
                "content": "The user completed an undated synthetic milestone",
                "category": "general",
                "topic": "tests",
                "temporal_kind": "event",
                "temporal_precision": "unknown",
            },
            source_role="user",
        )
        result = provider._store_candidate(
            candidate,
            source="test",
            session_id="unknown-event",
            observed_at=datetime.now(UTC).timestamp(),
        )

        assert result["fact"]["temporal_kind"] == "event"
        assert result["fact"]["event_at"] == 0
        assert provider._store.list_autobiographical_events("undated synthetic milestone") == []
        try:
            provider._remember_from_tool(
                {
                    "content": "The user completed another undated milestone",
                    "temporal_kind": "event",
                },
                session_id="unknown-event",
            )
        except ValueError as exc:
            assert "requires event_at" in str(exc)
        else:
            raise AssertionError("explicit event without event_at was accepted")
    finally:
        provider.shutdown()


def test_v3_migration_backfills_legacy_temporal_classification(tmp_path):
    path = tmp_path / "legacy.db"
    store = MemoryStore(path)
    try:
        fact = store.upsert_fact(
            content="The user currently lives in Paris",
            category="user_pref",
            topic="profile",
            source="legacy",
            metadata={"subject_key": "user:location:current", "exclusive": True},
        )["fact"]
    finally:
        store.close()

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE facts SET metadata_json=? WHERE id=?",
            ('{"subject_key":"user:location:current","exclusive":true}', fact["id"]),
        )
        connection.execute("DELETE FROM schema_migrations WHERE version=3")
        connection.execute("DROP INDEX idx_facts_temporal")
        for column in (
            "temporal_kind",
            "event_at",
            "temporal_precision",
            "temporal_timezone",
            "temporal_confidence",
        ):
            connection.execute(f"ALTER TABLE facts DROP COLUMN {column}")
        connection.commit()
    finally:
        connection.close()

    migrated = MemoryStore(path)
    try:
        row = migrated._fetchone("SELECT * FROM facts WHERE id=?", (fact["id"],))
        assert row["temporal_kind"] == "current"
        assert migrated._fetchone("SELECT name FROM schema_migrations WHERE version=3")["name"] == (
            "structured_temporal_context"
        )
        assert migrated.doctor()["ok"] is True
    finally:
        migrated.close()


def test_storage_rejects_nonfinite_temporal_numbers(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        fact = store.upsert_fact(
            content="A malformed integration supplied temporal numbers.",
            category="test",
            topic="temporal",
            source="test",
            observed_at=float("nan"),
            event_at=float("inf"),
            valid_until="not-a-number",
        )["fact"]
        event = store.upsert_autobiographical_event(
            event_key="bad-temporal-values",
            content="A malformed timeline event.",
            event_at=float("nan"),
            valid_from=float("-inf"),
            valid_until="invalid",
        )

        assert fact["last_seen_at"] > 0
        assert fact["event_at"] == 0
        assert fact["valid_until"] == 0
        assert event["event_at"] == 0
        assert event["valid_from"] == 0
        assert event["valid_until"] == 0
    finally:
        store.close()
