from __future__ import annotations

import json
from argparse import ArgumentParser

import pytest

from consolidating_local import ConsolidatingLocalMemoryProvider
from consolidating_local.cli import _scoped_database_path, register_cli
from consolidating_local.onboarding import (
    apply_onboarding,
    build_onboarding_plan,
    load_answers,
    run_onboarding,
    write_answer_template,
)
from consolidating_local.store import MemoryStore


def _row_count(store: MemoryStore, table: str) -> int:
    row = store._fetchone(f"SELECT COUNT(*) AS count FROM {table}")
    return int((row or {}).get("count") or 0)


def test_onboarding_builds_brain_memory_types_and_never_retains_credentials():
    secret = "sk-" + "proj-" + "abcdefghijklmnopqrstuvwxyz123456"
    plan = build_onboarding_plan(
        {
            "preferred_name": "Ada",
            "languages": "French, English",
            "response_style": "Lead with the outcome",
            "active_projects": "Hermes memory, Atlas",
            "current_goals": "Ship onboarding; improve retrieval",
            "approval_rules": "Ask before pushing; Ask before deleting data",
            "never_remember": "passwords, API keys, recovery codes",
            "recurring_workflow": "Release | run tests; update changelog; publish",
            "additional_context": f"Temporary credential {secret}",
        }
    )

    kinds = {item["memory_type"] for item in plan["items"]}
    assert kinds == {"fact", "preference", "policy", "procedure", "intention"}
    assert all(item["metadata"]["local_only"] is True for item in plan["items"])
    exclusion = next(item for item in plan["items"] if item["key"] == "onboarding-never-remember")
    assert exclusion["sensitivity"] == "normal"
    assert any(entry["key"].startswith("user:context:") for entry in plan["skipped"])
    assert secret not in json.dumps(plan)


def test_onboarding_strips_terminal_bom_from_first_answer():
    plan = build_onboarding_plan({"preferred_name": "\ufeffAda"})
    assert plan["items"][0]["content"] == "The user's preferred name is Ada."


def test_onboarding_preview_and_cancellation_never_write(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    output = []
    try:
        preview = run_onboarding(
            store,
            {"preferred_name": "Ada"},
            preview_only=True,
            input_fn=lambda _: pytest.fail("preview must not prompt"),
            output_fn=output.append,
        )
        assert preview["status"] == "preview"
        assert _row_count(store, "facts") == 0

        cancelled = run_onboarding(
            store,
            {"preferred_name": "Ada"},
            input_fn=lambda _: "no",
            output_fn=output.append,
        )
        assert cancelled["status"] == "cancelled"
        assert _row_count(store, "facts") == 0
        assert "local-only" in "\n".join(output)
    finally:
        store.close()


def test_onboarding_apply_is_atomic_idempotent_and_doctor_clean(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    plan = build_onboarding_plan(
        {
            "preferred_name": "Ada",
            "timezone": "Europe/Paris",
            "languages": "French, English",
            "response_tone": "Direct and collaborative",
            "active_projects": "Hermes memory",
            "current_goals": "Ship onboarding",
            "approval_rules": "Ask before pushing",
            "recurring_workflow": "Release | run tests; publish",
        }
    )
    try:
        first = apply_onboarding(store, plan)
        initial_history = _row_count(store, "memory_history")
        initial_evidence = _row_count(store, "belief_evidence")
        initial_counts = {
            table: _row_count(store, table)
            for table in (
                "facts",
                "memory_preferences",
                "memory_policies",
                "memory_procedures",
                "prospective_memories",
            )
        }
        second = apply_onboarding(store, plan)
        repeated_counts = {table: _row_count(store, table) for table in initial_counts}

        assert first["stored"] == len(plan["items"])
        assert second["stored"] == 0
        assert second["unchanged"] == len(plan["items"])
        assert repeated_counts == initial_counts
        assert _row_count(store, "memory_history") == initial_history
        assert _row_count(store, "belief_evidence") == initial_evidence
        assert store.doctor()["ok"] is True
        fact = store._fetchone("SELECT * FROM facts WHERE subject_key='user:name'")
        assert fact and fact["pinned"] == 1
        assert fact["metadata"]["local_only"] is True
    finally:
        store.close()


def test_onboarding_local_only_blocks_remote_embeddings_and_propagates_to_topics(tmp_path):
    provider = ConsolidatingLocalMemoryProvider(
        {
            "db_path": str(tmp_path / "memory.db"),
            "retrieval_backend": "hybrid",
            "embedding_model": "test-embedding",
            "embedding_base_url": "http://embedding.invalid/v1",
        }
    )
    try:
        provider.initialize("onboarding", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
        provider._task_queue.join()
        apply_onboarding(
            provider._store,
            build_onboarding_plan({"preferred_name": "Zephyr", "response_tone": "Direct and collaborative"}),
        )
        embedding_calls = []

        def fake_embed(texts):
            embedding_calls.append(list(texts))
            return [[1.0, 0.0] for _ in texts]

        provider._embedder.embed_texts = fake_embed
        results = provider._search_memory(
            "Zephyr", scope="facts", limit=5, session_id="onboarding", allow_embeddings=True
        )
        assert results["facts"]
        assert embedding_calls == []

        preference_results = provider._search_memory(
            "Direct collaborative response tone",
            scope="preferences",
            limit=5,
            session_id="onboarding",
            allow_embeddings=True,
        )
        assert preference_results["preferences"]
        assert embedding_calls == []

        provider._store.rebuild_topics()
        topic = provider._store._fetchone("SELECT * FROM topics WHERE slug='user-profile'")
        assert topic and topic["metadata"]["local_only"] is True
        topic_results = provider._search_memory(
            "Zephyr user profile", scope="topics", limit=5, session_id="onboarding", allow_embeddings=True
        )
        assert topic_results["topics"]
        assert embedding_calls == []

        provider._store.upsert_fact(
            content="PowerShell is available in the development environment.",
            category="environment",
            topic="shell",
            source="test",
        )
        provider._search_memory(
            "PowerShell development environment",
            scope="facts",
            limit=5,
            session_id="onboarding",
            allow_embeddings=True,
        )
        assert embedding_calls
    finally:
        provider.shutdown()


def test_onboarding_answer_file_template_validation_and_native_cli(tmp_path):
    template = write_answer_template(tmp_path / "answers.json")
    assert template.stat().st_size > 0
    answers = json.loads(template.read_text(encoding="utf-8"))
    answers["preferred_name"] = "Ada"
    template.write_text(json.dumps(answers), encoding="utf-8")
    assert load_answers(template)["preferred_name"] == "Ada"

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"unknown_field": "value"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown onboarding answer keys"):
        load_answers(bad)

    parser = ArgumentParser()
    register_cli(parser)
    args = parser.parse_args(
        [
            "--db",
            str(tmp_path / "memory.db"),
            "onboard",
            "--answers",
            str(template),
            "--preview-only",
            "--skip-sensitive",
        ]
    )
    assert args.consolidating_local_command == "onboard"
    assert args.preview_only is True
    assert args.skip_sensitive is True
    assert callable(args.func)


def test_onboarding_cli_scope_path_exactly_matches_provider_scope(tmp_path):
    base = tmp_path / "memory.db"
    provider = ConsolidatingLocalMemoryProvider({"db_path": str(base), "memory_scope": "user"})
    try:
        provider.initialize(
            "scope-test",
            hermes_home=str(tmp_path),
            platform="telegram",
            agent_context="primary",
            user_id="123456",
        )
        resolved = _scoped_database_path(
            base,
            scope_mode="user",
            platform="telegram",
            user_id="123456",
        )
        assert provider._store.db_path == resolved
    finally:
        provider.shutdown()

    parser = ArgumentParser()
    register_cli(parser)
    args = parser.parse_args(
        [
            "--db",
            str(base),
            "--scope-platform",
            "telegram",
            "--scope-user-id",
            "123456",
            "onboard",
            "--preview-only",
        ]
    )
    assert args.scope_platform == "telegram"
    assert args.scope_user_id == "123456"

    with pytest.raises(ValueError, match="scope-platform"):
        _scoped_database_path(base, scope_mode="user", user_id="123456")
