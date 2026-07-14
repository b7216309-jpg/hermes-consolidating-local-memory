from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterable, List

from .store import MemoryStore, normalize_whitespace, slugify

ALLOWED_CATEGORIES = {"user_pref", "project", "environment", "workflow", "general"}


def message_content_text(content: Any) -> str:
    """Return the textual portions of a Hermes/OpenAI message payload."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        for key in ("text", "input_text", "output_text"):
            value = block.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
                break
    return " ".join(parts)


def _bounded_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(low, min(high, parsed))


def _bounded_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        parsed = default
    return max(low, min(high, parsed))


def _is_true(value: Any) -> bool:
    return value is True or str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def normalize_candidate_fact(raw: Dict[str, Any], *, source_role: str = "assistant") -> Dict[str, Any] | None:
    """Validate and normalize a model-produced fact without guessing its meaning."""

    content = normalize_whitespace(str(raw.get("content") or "")).strip()
    if not content:
        return None
    content = content[:8000]
    category = normalize_whitespace(str(raw.get("category") or "general")).casefold()
    if category not in ALLOWED_CATEGORIES:
        category = "general"
    topic = slugify(str(raw.get("topic") or category)) or category
    metadata_raw = raw.get("metadata")
    metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
    subject_key = normalize_whitespace(str(raw.get("subject_key") or metadata.get("subject_key") or ""))[:255]
    value_key = normalize_whitespace(str(raw.get("value_key") or metadata.get("value_key") or ""))[:255]
    if subject_key:
        metadata["subject_key"] = subject_key
        if value_key:
            metadata["value_key"] = value_key
        else:
            metadata.pop("value_key", None)
        if _is_true(raw.get("exclusive", metadata.get("exclusive"))):
            metadata["exclusive"] = True
        else:
            metadata.pop("exclusive", None)
    else:
        metadata.pop("subject_key", None)
        metadata.pop("value_key", None)
        metadata.pop("exclusive", None)
    polarity_raw = raw.get("polarity", metadata.get("polarity", 1))
    metadata["polarity"] = -1 if str(polarity_raw).strip().casefold() in {"-1", "false", "neg", "negative", "no"} else 1
    clean_role = str(source_role or "assistant").strip().casefold()
    metadata["source_role"] = clean_role if clean_role in {"user", "assistant", "tool"} else "assistant"
    return {
        "content": content,
        "category": category,
        "topic": topic,
        "importance": _bounded_int(raw.get("importance"), 1, 10, 6),
        "confidence": _bounded_float(
            raw.get("confidence"),
            0.05,
            1.0,
            0.75 if metadata["source_role"] == "user" else 0.6,
        ),
        "metadata": metadata,
    }


def build_consolidation_plan(store: MemoryStore, *, min_hours: int, min_sessions: int) -> Dict[str, Any]:
    last_at = float(store.get_state("last_consolidated_at", "0") or 0)
    last_episode_id = int(store.get_state("last_consolidated_episode_id", "0") or 0)
    hours_since = (time.time() - last_at) / 3600 if last_at else float("inf")
    pending_sessions = store.sessions_since_episode(last_episode_id)
    pending_episodes = store.pending_episode_count(last_episode_id)
    return {
        "last_consolidated_at": last_at,
        "last_consolidated_episode_id": last_episode_id,
        "hours_since_last": None if hours_since == float("inf") else round(hours_since, 2),
        "pending_sessions": int(max(pending_sessions, 0)),
        "pending_episodes": int(max(pending_episodes, 0)),
        "min_hours": int(min_hours),
        "min_sessions": int(min_sessions),
        "should_run": (
            pending_episodes > 0
            and (hours_since == float("inf") or hours_since >= min_hours)
            and pending_sessions >= min_sessions
        ),
    }


def _build_session_summary_text(artifacts: Dict[str, Any], *, max_chars: int) -> str:
    parts: List[str] = []
    facts = [str(item.get("content") or "") for item in artifacts.get("facts", [])[:4] if item.get("content")]
    if facts:
        parts.append("Facts: " + "; ".join(facts))
    journals = [str(item.get("content") or "") for item in artifacts.get("journals", [])[:2] if item.get("content")]
    if journals:
        parts.append("Notes: " + " | ".join(journals))
    traces = [str(item.get("content") or "") for item in artifacts.get("traces", [])[:3] if item.get("content")]
    if traces:
        parts.append("Recent flow: " + " | ".join(traces))
    preferences = [
        str(item.get("content") or item.get("label") or "")
        for item in artifacts.get("preferences", [])[:2]
        if item.get("content") or item.get("label")
    ]
    if preferences:
        parts.append("Preferences: " + " | ".join(preferences))
    policies = [
        str(item.get("content") or item.get("label") or "")
        for item in artifacts.get("policies", [])[:2]
        if item.get("content") or item.get("label")
    ]
    if policies:
        parts.append("Policies: " + " | ".join(policies))
    summary = " ".join(part for part in parts if part).strip()
    return summary[:max_chars] if summary else ""


Extractor = Callable[..., Iterable[Dict[str, Any]]]
FactWriter = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


def run_consolidation(
    store: MemoryStore,
    *,
    min_hours: int,
    min_sessions: int,
    max_topic_facts: int,
    topic_summary_chars: int,
    prune_after_days: int,
    session_summary_chars: int = 900,
    episode_retention_hours: float = 24.0,
    decay_half_life_days: float = 90.0,
    decay_min_salience: float = 0.15,
    episode_batch_size: int = 500,
    extractor: Extractor | None = None,
    fact_writer: FactWriter | None = None,
    force: bool = False,
    reason: str = "auto",
) -> Dict[str, Any]:
    started_at = time.time()
    plan = build_consolidation_plan(store, min_hours=min_hours, min_sessions=min_sessions)
    if not force and not plan["should_run"]:
        plan["status"] = "skipped"
        return plan

    last_episode_id = int(plan["last_consolidated_episode_id"])
    episodes = store.episodes_since_episode(last_episode_id, limit=max(1, int(episode_batch_size)))
    facts_added = 0
    facts_updated = 0
    facts_superseded = 0
    contradictions_resolved = 0
    touched_sessions = set()

    for episode in episodes:
        session_id = normalize_whitespace(str(episode.get("session_id") or ""))
        if session_id:
            touched_sessions.add(session_id)
        candidates = (
            extractor(
                user_content=str(episode.get("user_content", "")),
                assistant_content=str(episode.get("assistant_content", "")),
                created_at=float(episode.get("created_at") or started_at),
            )
            if extractor
            else []
        )
        for candidate in candidates:
            if fact_writer:
                result = fact_writer(candidate, episode)
            else:
                result = store.upsert_fact(
                    content=str(candidate["content"]),
                    category=str(candidate["category"]),
                    topic=str(candidate["topic"]),
                    source="episode_extract",
                    importance=int(candidate["importance"]),
                    confidence=float(candidate["confidence"]),
                    metadata=dict(candidate.get("metadata") or {}),
                    observed_at=float(episode.get("created_at") or started_at),
                    source_session_id=session_id,
                    history_reason="episode_extract",
                )
                fact_id = dict(result.get("fact") or {}).get("id")
                if fact_id is not None and episode.get("id") is not None:
                    store.add_link("fact", fact_id, "episode", int(episode["id"]), "derived_from_episode")
            if result.get("action") == "inserted":
                facts_added += 1
            elif result.get("action") == "updated":
                facts_updated += 1
            facts_superseded += len(result.get("superseded", []))
            contradictions_resolved += len(result.get("contradictions", []))

    subjects_merged = store.merge_duplicate_subjects()
    pruned = store.prune_stale_facts(max_age_days=prune_after_days)
    decay_stats = store.apply_decay(half_life_days=decay_half_life_days, min_salience=decay_min_salience)
    topics_rebuilt = store.rebuild_topics(max_facts=max_topic_facts, max_chars=topic_summary_chars)
    session_summaries = 0
    for session_id in sorted(touched_sessions):
        artifacts = store.get_session_artifacts(session_id, limit=max(8, max_topic_facts * 2))
        summary = _build_session_summary_text(artifacts, max_chars=int(session_summary_chars))
        if not summary:
            continue
        refs: List[Dict[str, Any]] = []
        kind_map = {
            "facts": "fact",
            "journals": "journal",
            "traces": "trace",
            "episodes": "episode",
            "preferences": "preference",
            "policies": "policy",
        }
        for section in ("facts", "journals", "traces", "episodes", "preferences", "policies"):
            for item in artifacts.get(section, [])[:4]:
                if item.get("id") is not None:
                    refs.append({"kind": kind_map[section], "id": item["id"]})
        store.upsert_summary(
            label="Session Summary",
            summary=summary,
            session_id=session_id,
            content=summary,
            summary_type="session",
            metadata={"source": "consolidation"},
            importance=8,
            salience=0.72,
            source_refs=refs,
            reason="consolidation_distill",
        )
        store.ensure_memory_session(session_id, summary=summary)
        session_summaries += 1
    latest_episode_id = store.latest_episode_id()
    processed_episode_id = max((int(item["id"]) for item in episodes), default=last_episode_id)
    episodes_pruned = store.purge_episode_buffers(
        retention_hours=episode_retention_hours,
        max_episode_id=processed_episode_id,
    )
    backlog_remaining = store.pending_episode_count(processed_episode_id)
    history_compacted = store.compact_history(max_per_entity=10, max_age_days=90)
    finished_at = time.time()

    stats = {
        "status": "completed",
        "reason": reason,
        "episodes_scanned": len(episodes),
        "facts_added": facts_added,
        "facts_updated": facts_updated,
        "facts_superseded": facts_superseded,
        "subjects_merged": subjects_merged,
        "contradictions_resolved": contradictions_resolved,
        "facts_pruned": pruned,
        "topics_rebuilt": topics_rebuilt,
        "session_summaries": session_summaries,
        "episodes_pruned": episodes_pruned,
        "history_compacted": history_compacted,
        "decay": decay_stats,
        "latest_episode_id": latest_episode_id,
        "processed_episode_id": processed_episode_id,
        "backlog_remaining": backlog_remaining,
        "counts": store.counts(),
        "duration_seconds": round(finished_at - started_at, 3),
    }
    store.record_consolidation(
        reason=reason,
        started_at=started_at,
        finished_at=finished_at,
        source_episode_id=processed_episode_id,
        stats=stats,
    )
    return stats
