from __future__ import annotations

import calendar
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .store import MemoryStore, normalize_whitespace, slugify

ALLOWED_CATEGORIES = {"user_pref", "project", "environment", "workflow", "general"}
TEMPORAL_KINDS = {"atemporal", "current", "event", "scheduled", "temporary"}
TEMPORAL_PRECISIONS = {"unknown", "year", "month", "day", "hour", "minute", "second"}


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


def _clean_timezone(value: Any) -> str:
    name = normalize_whitespace(str(value or ""))[:80]
    if not name:
        return ""
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ""
    return name


def _infer_temporal_precision(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "second"
    text = str(value or "").strip()
    if not text:
        return "unknown"
    date_part, _, time_part = text.replace("Z", "+00:00").partition("T")
    if len(date_part) == 4 and date_part.isdigit():
        return "year"
    if len(date_part) == 7 and date_part[4] == "-":
        return "month"
    if not time_part:
        return "day"
    clock = time_part.split("+", 1)[0].rsplit("-", 1)[0]
    fields = clock.split(":")
    if len(fields) <= 1:
        return "hour"
    if len(fields) == 2:
        return "minute"
    return "second"


def _parse_temporal_value(
    value: Any,
    *,
    timezone_name: str = "",
    boundary: str = "event",
) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if parsed == parsed and 0 < parsed < 253402300800 else 0.0

    text = str(value).strip()
    if not text:
        return 0.0
    try:
        numeric = float(text)
    except ValueError:
        numeric = 0.0
    if numeric:
        return numeric if numeric == numeric and 0 < numeric < 253402300800 else 0.0

    zone = ZoneInfo(timezone_name) if timezone_name else UTC
    precision = _infer_temporal_precision(text)
    try:
        if precision == "year":
            year = int(text)
            if boundary == "end":
                parsed_dt = datetime(year + 1, 1, 1, tzinfo=zone)
            elif boundary == "event":
                parsed_dt = datetime(year, 1, 1, 12, tzinfo=zone)
            else:
                parsed_dt = datetime(year, 1, 1, tzinfo=zone)
        elif precision == "month":
            year, month = (int(part) for part in text.split("-", 1))
            if boundary == "end":
                next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
                parsed_dt = datetime(next_year, next_month, 1, tzinfo=zone)
            elif boundary == "event":
                parsed_dt = datetime(year, month, 1, 12, tzinfo=zone)
            else:
                parsed_dt = datetime(year, month, 1, tzinfo=zone)
        elif precision == "day":
            parsed_date = datetime.fromisoformat(text).date()
            if boundary == "end":
                parsed_dt = datetime.combine(parsed_date + timedelta(days=1), datetime.min.time(), tzinfo=zone)
            elif boundary == "event":
                parsed_dt = datetime.combine(parsed_date, datetime.min.time(), tzinfo=zone) + timedelta(hours=12)
            else:
                parsed_dt = datetime.combine(parsed_date, datetime.min.time(), tzinfo=zone)
        else:
            parsed_dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.replace(tzinfo=zone)
        timestamp = parsed_dt.timestamp()
    except (OverflowError, TypeError, ValueError):
        return 0.0
    return timestamp if 0 < timestamp < 253402300800 else 0.0


def _scheduled_expiry(event_at: float, precision: str, timezone_name: str) -> float:
    if event_at <= 0:
        return 0.0
    zone = ZoneInfo(timezone_name) if timezone_name else UTC
    event = datetime.fromtimestamp(event_at, zone)
    if precision == "year":
        return datetime(event.year + 1, 1, 1, tzinfo=zone).timestamp()
    if precision == "month":
        _, last_day = calendar.monthrange(event.year, event.month)
        return datetime(event.year, event.month, last_day, 23, 59, 59, tzinfo=zone).timestamp() + 1
    if precision == "day":
        return datetime.combine(event.date() + timedelta(days=1), datetime.min.time(), tzinfo=zone).timestamp()
    return event_at + 86400.0


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

    reference_time = _parse_temporal_value(raw.get("reference_unix_time") or metadata.get("reference_unix_time"))
    timezone_name = _clean_timezone(
        raw.get("temporal_timezone")
        or metadata.get("temporal_timezone")
        or raw.get("reference_timezone")
        or metadata.get("reference_timezone")
    )
    event_raw = raw.get("event_at", metadata.get("event_at"))
    precision = normalize_whitespace(
        str(raw.get("temporal_precision") or metadata.get("temporal_precision") or "")
    ).casefold()
    if precision not in TEMPORAL_PRECISIONS:
        precision = _infer_temporal_precision(event_raw)
    event_at = _parse_temporal_value(event_raw, timezone_name=timezone_name, boundary="event")
    valid_from = _parse_temporal_value(
        raw.get("valid_from", metadata.get("valid_from")),
        timezone_name=timezone_name,
        boundary="start",
    )
    valid_until = _parse_temporal_value(
        raw.get("valid_until", metadata.get("valid_until")),
        timezone_name=timezone_name,
        boundary="end",
    )
    temporal_kind = normalize_whitespace(
        str(raw.get("temporal_kind") or metadata.get("temporal_kind") or "")
    ).casefold()
    explicit_temporal_kind = temporal_kind in TEMPORAL_KINDS
    if not explicit_temporal_kind:
        if valid_until:
            temporal_kind = "temporary"
        elif event_at:
            temporal_kind = "scheduled" if reference_time and event_at > reference_time else "event"
        elif subject_key and _is_true(raw.get("exclusive", metadata.get("exclusive"))):
            temporal_kind = "current"
        else:
            temporal_kind = "atemporal"
    if temporal_kind == "scheduled" and event_at and not valid_until:
        valid_until = _scheduled_expiry(event_at, precision, timezone_name)
    if valid_from and valid_until and valid_until <= valid_from:
        valid_until = 0.0
    default_temporal_confidence = 0.9 if explicit_temporal_kind else (0.7 if event_at or valid_until else 0.5)
    temporal_confidence = _bounded_float(
        raw.get("temporal_confidence", metadata.get("temporal_confidence")),
        0.0,
        1.0,
        default_temporal_confidence,
    )
    metadata["temporal_kind"] = temporal_kind
    metadata["temporal_precision"] = precision
    metadata["temporal_confidence"] = temporal_confidence
    if timezone_name:
        metadata["temporal_timezone"] = timezone_name
    else:
        metadata.pop("temporal_timezone", None)
    for key, value in (("event_at", event_at), ("valid_from", valid_from), ("valid_until", valid_until)):
        if value > 0:
            metadata[key] = value
        else:
            metadata.pop(key, None)
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
