from __future__ import annotations

import hashlib
import html
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set

from .store import MemoryStore, _looks_sensitive_for_export, normalize_whitespace, pretty_topic, slugify

MANAGED_SUBDIRS = ("topics", "sessions", "preferences", "policies", "contradictions")
MANIFEST_NAME = ".consolidating-memory-manifest.json"

# ── Category display names & ordering ──
_CATEGORY_ORDER = ["user_pref", "general", "project", "environment", "workflow"]
_CATEGORY_LABELS = {
    "user_pref": "Personal Profile",
    "general": "General Knowledge",
    "project": "Projects",
    "environment": "Environment & Setup",
    "workflow": "Workflow & Rules",
}


def _safe_page_name(value: str, *, fallback: str) -> str:
    clean = normalize_whitespace(value)
    slug = (slugify(clean) if clean else fallback)[:80].rstrip("-_.") or fallback
    digest = hashlib.sha1(clean.encode("utf-8")).hexdigest()[:8] if clean else "00000000"
    return f"{slug}-{digest}.md"


def _topic_page_path(slug: str) -> str:
    return f"topics/{_safe_page_name(slug, fallback='topic')}"


def _normalize_markdown(text: str) -> str:
    clean = str(text or "").replace("\r\n", "\n").strip()
    return clean + "\n"


def _md(value: Any, *, limit: int | None = None) -> str:
    text = str(value or "")
    if limit is not None:
        text = text[:limit]
    text = html.escape(text, quote=False).replace("\r", " ").replace("\n", " ")
    return re.sub(r"([\\`*_{}\[\]|])", r"\\\1", text)


def _is_sensitive_item(item: Dict[str, Any]) -> bool:
    return _looks_sensitive_for_export(item)


def _redact_rows(rows: List[Dict[str, Any]], enabled: bool) -> List[Dict[str, Any]]:
    return [row for row in rows if not _is_sensitive_item(row)] if enabled else rows


def _write_if_changed(path: Path, content: str) -> bool:
    normalized = _normalize_markdown(content)
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current == normalized:
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(normalized)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
    return True


def _safe_output_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe wiki output path: {relative}")
    candidate = root / rel
    resolved_parent = candidate.parent.resolve()
    if resolved_parent != root and root not in resolved_parent.parents:
        raise ValueError(f"Wiki output escapes the export root: {relative}")
    return candidate


def _relative_link(from_rel: str, to_rel: str) -> str:
    return Path(os.path.relpath(to_rel, start=str(Path(from_rel).parent))).as_posix()


def _bullet_link(label: str, target_rel: str, *, from_rel: str) -> str:
    return f"[{_md(label)}]({_relative_link(from_rel, target_rel)})"


def _fmt_ts(value: Any) -> str:
    try:
        ts = float(value or 0)
    except Exception:
        return ""
    if ts <= 0:
        return ""
    import time

    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _fmt_ts_short(value: Any) -> str:
    try:
        ts = float(value or 0)
    except Exception:
        return ""
    if ts <= 0:
        return ""
    import time

    return time.strftime("%Y-%m-%d", time.localtime(ts))


def _fact_temporal_label(row: Dict[str, Any]) -> str:
    kind = str(row.get("temporal_kind") or "atemporal")
    if kind in {"event", "scheduled"} and float(row.get("event_at") or 0) > 0:
        return f"{kind}: {_fmt_ts_short(row.get('event_at'))}"
    if float(row.get("valid_until") or 0) > 0:
        return f"{kind}: until {_fmt_ts_short(row.get('valid_until'))}"
    anchor = row.get("valid_from") or row.get("updated_at") or row.get("created_at")
    return f"{kind}: {_fmt_ts_short(anchor) or 'unknown'}"


def _imp_bar(importance: int) -> str:
    """Render importance 1-10 as a compact visual bar."""
    filled = min(max(int(importance), 0), 10)
    return "█" * filled + "░" * (10 - filled)


def _salience_tag(salience: float) -> str:
    """Render salience as a colored tag word."""
    if salience >= 0.90:
        return "🔴 core"
    if salience >= 0.80:
        return "🟠 high"
    if salience >= 0.65:
        return "🟡 mid"
    if salience >= 0.50:
        return "🟢 low"
    return "⚪ faint"


# ────────────────────────────────────────────────
#  INDEX PAGE
# ────────────────────────────────────────────────


def _render_index(
    *,
    counts: Dict[str, int],
    topics: List[Dict[str, Any]],
    sessions: List[Dict[str, Any]],
    preferences: List[Dict[str, Any]],
    policies: List[Dict[str, Any]],
    contradictions: List[Dict[str, Any]],
    topic_paths: Dict[str, str],
    session_paths: Dict[str, str],
    facts_by_category: Dict[str, List[Dict[str, Any]]],
) -> str:
    rel = "index.md"
    lines = [
        "# 🧠 Memory Wiki",
        "",
        "> Auto-generated from `consolidating_memory.db`. Edit the SQLite source, not these files.",
        "",
        "---",
        "",
        "## 📊 Overview",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Facts | **{counts.get('facts', 0)}** |",
        f"| Topics | **{counts.get('topics', 0)}** |",
        f"| Sessions | **{counts.get('sessions', 0)}** |",
        f"| Summaries | **{counts.get('summaries', 0)}** |",
        f"| Preferences | **{counts.get('preferences', 0)}** |",
        f"| Policies | **{counts.get('policies', 0)}** |",
        f"| Contradictions | **{counts.get('contradictions', 0)}** |",
        "",
        "---",
        "",
    ]

    # ── Facts grouped by category ──
    lines.append("## 🗂️ Facts by Category")
    lines.append("")
    categories = _CATEGORY_ORDER + sorted(set(facts_by_category) - set(_CATEGORY_ORDER))
    for cat in categories:
        cat_facts = facts_by_category.get(cat, [])
        if not cat_facts:
            continue
        label = _CATEGORY_LABELS.get(cat, cat.title())
        lines.append(f"### {label} ({len(cat_facts)})")
        lines.append("")
        lines.append("| Subject | Value | Content | Time | Imp | Salience |")
        lines.append("| --- | --- | --- | --- | ---: | --- |")
        for f in sorted(cat_facts, key=lambda x: (-int(x.get("importance") or 0), str(x.get("subject_key") or ""))):
            sk = str(f.get("subject_key") or "—")
            # Strip common prefixes for readability
            for pfx in ("user:", "environment:", "project:", "workflow:", "general:"):
                if sk.startswith(pfx):
                    sk = sk[len(pfx) :]
                    break
            vk = str(f.get("value_key") or "—")
            content = str(f.get("content") or "")[:80]
            imp = int(f.get("importance") or 0)
            sal = float(f.get("salience") or 0)
            lines.append(
                f"| `{_md(sk)}` | `{_md(vk)}` | {_md(content)} | {_md(_fact_temporal_label(f))} | "
                f"{imp} | {_salience_tag(sal)} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")

    # ── Topics ──
    lines.append("## 📚 Topics")
    lines.append("")
    if topics:
        lines.append("| Topic | Category | Salience | Updated |")
        lines.append("| --- | --- | --- | --- |")
        for topic in topics[:20]:
            target = topic_paths.get(str(topic.get("slug") or ""))
            title = str(topic.get("title") or pretty_topic(str(topic.get("slug") or "topic")))
            link = _bullet_link(title, target, from_rel=rel) if target else _md(title)
            cat = str(topic.get("category") or "general")
            sal = float(topic.get("salience") or 0)
            updated = _fmt_ts_short(topic.get("updated_at")) or "—"
            lines.append(f"| {link} | `{_md(cat)}` | {_salience_tag(sal)} | {updated} |")
    else:
        lines.append("*No topics exported yet.*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Sessions ──
    lines.append("## 💬 Latest Sessions")
    lines.append("")
    if sessions:
        for session in sessions[:10]:
            session_id = str(session.get("session_id") or "")
            target = session_paths.get(session_id)
            summary = normalize_whitespace(str(session.get("summary") or "")) or "*No summary yet.*"
            started = _fmt_ts_short(session.get("started_at")) or ""
            link = _bullet_link(f"{session_id[:12]}…", target, from_rel=rel) if target else _md(session_id[:12])
            lines.append(f"- **{link}** ({started}): {_md(summary, limit=120)}")
    else:
        lines.append("*No session pages exported yet.*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Preferences ──
    lines.append("## ⭐ Active Preferences")
    lines.append("")
    if preferences:
        lines.append("| Key | Value | Importance |")
        lines.append("| --- | --- | ---: |")
        for item in preferences:
            pk = str(item.get("preference_key") or item.get("label") or "")
            val = str(item.get("value") or item.get("content") or "")[:60]
            imp = int(item.get("importance") or 0)
            lines.append(f"| `{_md(pk)}` | {_md(val)} | {imp} |")
    else:
        lines.append("*No active preferences.*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Policies ──
    lines.append("## 📋 Active Policies")
    lines.append("")
    if policies:
        for item in policies[:10]:
            lines.append(
                f"- **{_md(item.get('label') or item.get('policy_key') or 'Policy')}**: {_md(item.get('content'))}"
            )
    else:
        lines.append("*No active policies.*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Contradictions ──
    lines.append("## ⚡ Recent Contradictions")
    lines.append("")
    if contradictions:
        lines.append("| Subject | Before | After | Date |")
        lines.append("| --- | --- | --- | --- |")
        for row in contradictions[:10]:
            sk = str(row.get("subject_key") or "unknown")
            winner = normalize_whitespace(str(row.get("winner_content") or ""))[:50]
            loser = normalize_whitespace(str(row.get("loser_content") or ""))[:50]
            created = _fmt_ts_short(row.get("created_at")) or "—"
            lines.append(f"| `{_md(sk)}` | ~~{_md(loser)}~~ | {_md(winner)} | {created} |")
    else:
        lines.append("*No contradictions logged.*")

    return "\n".join(lines)


# ────────────────────────────────────────────────
#  TOPIC PAGE
# ────────────────────────────────────────────────


def _render_topic_page(
    topic: Dict[str, Any],
    *,
    facts: List[Dict[str, Any]],
    contradictions: List[Dict[str, Any]],
    session_paths: Dict[str, str],
) -> str:
    rel = _topic_page_path(str(topic.get("slug") or "topic"))
    title = str(topic.get("title") or pretty_topic(str(topic.get("slug") or "topic")))
    sal = float(topic.get("salience") or 0)
    lines = [
        f"# {_md(title)}",
        "",
        "| Property | Value |",
        "| --- | --- |",
        f"| Slug | `{_md(topic.get('slug'))}` |",
        f"| Category | `{_md(topic.get('category') or 'general')}` |",
        f"| Salience | {_salience_tag(sal)} ({sal:.2f}) |",
        f"| Updated | {_fmt_ts(topic.get('updated_at')) or 'unknown'} |",
        "",
        "---",
        "",
        "## Summary",
        "",
        _md(topic.get("summary")) if topic.get("summary") else "*No summary available.*",
        "",
        "---",
        "",
        "## Supporting Facts",
        "",
    ]
    if facts:
        lines.append("| Content | Time | Importance | Confidence |")
        lines.append("| --- | --- | ---: | ---: |")
        for fact in facts:
            content = str(fact.get("content") or "")[:100]
            imp = int(fact.get("importance") or 0)
            conf = float(fact.get("confidence") or 0)
            lines.append(f"| {_md(content)} | {_md(_fact_temporal_label(fact))} | {imp} | {conf:.0%} |")
    else:
        lines.append("*No supporting facts linked.*")

    lines.extend(["", "---", "", "## Related Sessions", ""])
    seen_sessions: Set[str] = set()
    session_lines: List[str] = []
    for fact in facts:
        session_id = str(fact.get("source_session_id") or "")
        target = session_paths.get(session_id)
        if not session_id or not target or session_id in seen_sessions:
            continue
        seen_sessions.add(session_id)
        session_lines.append(f"- {_bullet_link(session_id[:12] + '…', target, from_rel=rel)}")
    if session_lines:
        lines.extend(session_lines)
    else:
        lines.append("*No related session pages.*")

    lines.extend(["", "---", "", "## Contradictions", ""])
    topic_contradictions = [
        row
        for row in contradictions
        if str(row.get("winner_topic") or "") == str(topic.get("slug") or "")
        or str(row.get("loser_topic") or "") == str(topic.get("slug") or "")
    ]
    if topic_contradictions:
        lines.append("| Subject | Before | After |")
        lines.append("| --- | --- | --- |")
        for row in topic_contradictions[:10]:
            winner = normalize_whitespace(str(row.get("winner_content") or ""))[:50]
            loser = normalize_whitespace(str(row.get("loser_content") or ""))[:50]
            lines.append(f"| `{_md(row.get('subject_key') or 'unknown')}` | ~~{_md(loser)}~~ | {_md(winner)} |")
    else:
        lines.append("*No contradictions for this topic.*")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Navigation",
            "",
            f"- [← Back to Wiki]({_relative_link(rel, 'index.md')})",
            f"- [Contradictions]({_relative_link(rel, 'contradictions/index.md')})",
        ]
    )
    return "\n".join(lines)


# ────────────────────────────────────────────────
#  SESSION PAGE
# ────────────────────────────────────────────────


def _render_session_page(
    session: Dict[str, Any],
    *,
    artifacts: Dict[str, Any],
    topic_paths: Dict[str, str],
) -> str:
    session_id = str(session.get("session_id") or "")
    rel = f"sessions/{_safe_page_name(session_id, fallback='session')}"
    started = _fmt_ts(session.get("started_at")) or "unknown"
    ended = _fmt_ts(session.get("ended_at")) or "ongoing"
    lines = [
        f"# Session `{_md(session_id[:16])}…`",
        "",
        "| Property | Value |",
        "| --- | --- |",
        f"| Status | `{_md(session.get('status') or 'unknown')}` |",
        f"| Started | {started} |",
        f"| Ended | {ended} |",
        f"| Last activity | {_fmt_ts(session.get('last_activity_at')) or 'unknown'} |",
        "",
        "---",
        "",
        "## Summary",
        "",
        _md(session.get("summary")) if session.get("summary") else "*No session summary available.*",
        "",
        "---",
        "",
    ]

    # Facts
    facts = list(artifacts.get("facts", []))
    lines.append("## Facts Extracted")
    lines.append("")
    if facts:
        lines.append("| Content | Time | Topic |")
        lines.append("| --- | --- | --- |")
        for fact in facts:
            content = str(fact.get("content") or "")[:90]
            topic_slug = str(fact.get("topic") or "")
            topic_target = topic_paths.get(topic_slug)
            topic_link = (
                _bullet_link(pretty_topic(topic_slug), topic_target, from_rel=rel)
                if topic_target
                else pretty_topic(topic_slug)
            )
            lines.append(f"| {_md(content)} | {_md(_fact_temporal_label(fact))} | {topic_link} |")
    else:
        lines.append("*No extracted facts linked to this session.*")

    # Preferences
    preferences = list(artifacts.get("preferences", []))
    lines.extend(["", "---", "", "## Preferences", ""])
    if preferences:
        for item in preferences:
            lines.append(f"- {_md(item.get('content') or item.get('label'))}")
    else:
        lines.append("*No session-specific preferences.*")

    # Policies
    policies = list(artifacts.get("policies", []))
    lines.extend(["", "## Policies", ""])
    if policies:
        for item in policies:
            lines.append(f"- {_md(item.get('content') or item.get('label'))}")
    else:
        lines.append("*No session-specific policies.*")

    # Journals
    journals = list(artifacts.get("journals", []))
    lines.extend(["", "## Journals", ""])
    if journals:
        for item in journals:
            lines.append(f"- **{_md(item.get('label') or 'Journal')}**: {_md(item.get('content'))}")
    else:
        lines.append("*No journals for this session.*")

    # Traces
    traces = list(artifacts.get("traces", []))
    lines.extend(["", "## Traces", ""])
    if traces:
        for item in traces[:10]:
            lines.append(f"- {_md(item.get('content'))}")
    else:
        lines.append("*No traces for this session.*")

    # Navigation
    lines.extend(["", "---", "", "## Navigation", ""])
    linked_topics: List[str] = []
    for fact in facts:
        topic_slug = str(fact.get("topic") or "")
        topic_target = topic_paths.get(topic_slug)
        if not topic_target or topic_slug in linked_topics:
            continue
        linked_topics.append(topic_slug)
        lines.append(f"- {_bullet_link(pretty_topic(topic_slug), topic_target, from_rel=rel)}")
    if not linked_topics:
        lines.append("*No linked topic pages.*")
    lines.append(f"- [← Back to Wiki]({_relative_link(rel, 'index.md')})")
    return "\n".join(lines)


# ────────────────────────────────────────────────
#  PREFERENCES INDEX
# ────────────────────────────────────────────────


def _render_preferences_index(
    preferences: List[Dict[str, Any]],
    *,
    session_paths: Dict[str, str],
) -> str:
    rel = "preferences/index.md"
    lines = [
        "# ⭐ Preferences",
        "",
        "> Durable user preferences extracted from conversations.",
        "",
        "---",
        "",
    ]
    if preferences:
        lines.append("| Key | Label | Value | Importance | Salience |")
        lines.append("| --- | --- | --- | ---: | --- |")
        for item in preferences:
            pk = str(item.get("preference_key") or "")
            label = str(item.get("label") or "")[:40]
            val = str(item.get("value") or "")[:40]
            imp = int(item.get("importance") or 0)
            sal = float(item.get("salience") or 0)
            lines.append(f"| `{_md(pk)}` | {_md(label)} | {_md(val)} | {imp} | {_salience_tag(sal)} |")
    else:
        lines.append("*No active preferences.*")
    lines.extend(["", "---", "", f"[← Back to Wiki]({_relative_link(rel, 'index.md')})"])
    return "\n".join(lines)


# ────────────────────────────────────────────────
#  POLICIES INDEX
# ────────────────────────────────────────────────


def _render_policies_index(
    policies: List[Dict[str, Any]],
    *,
    session_paths: Dict[str, str],
) -> str:
    rel = "policies/index.md"
    lines = [
        "# 📋 Policies",
        "",
        "> Active workflow rules and operating constraints.",
        "",
        "---",
        "",
    ]
    if policies:
        lines.append("| Policy | Content | Importance |")
        lines.append("| --- | --- | ---: |")
        for item in policies:
            label = str(item.get("label") or item.get("policy_key") or "Policy")
            content = str(item.get("content") or "")[:80]
            imp = int(item.get("importance") or 0)
            lines.append(f"| **{_md(label)}** | {_md(content)} | {imp} |")
    else:
        lines.append("*No active policies.*")
    lines.extend(["", "---", "", f"[← Back to Wiki]({_relative_link(rel, 'index.md')})"])
    return "\n".join(lines)


# ────────────────────────────────────────────────
#  CONTRADICTIONS INDEX
# ────────────────────────────────────────────────


def _render_contradictions_index(contradictions: List[Dict[str, Any]]) -> str:
    rel = "contradictions/index.md"
    lines = [
        "# ⚡ Contradictions",
        "",
        "> Changelog of superseded assumptions — what the system once believed vs. what it knows now.",
        "",
        "---",
        "",
    ]
    if contradictions:
        lines.append("| Subject | Before | After | Date |")
        lines.append("| --- | --- | --- | --- |")
        for row in contradictions:
            sk = str(row.get("subject_key") or "unknown")
            winner = normalize_whitespace(str(row.get("winner_content") or ""))[:60]
            loser = normalize_whitespace(str(row.get("loser_content") or ""))[:60]
            created = _fmt_ts_short(row.get("created_at")) or "—"
            lines.append(f"| `{_md(sk)}` | ~~{_md(loser)}~~ | {_md(winner)} | {created} |")
    else:
        lines.append("*No contradictions logged.*")
    lines.extend(["", "---", "", f"[← Back to Wiki]({_relative_link(rel, 'index.md')})"])
    return "\n".join(lines)


# ────────────────────────────────────────────────
#  MAIN EXPORT
# ────────────────────────────────────────────────


def export_compiled_wiki(
    store: MemoryStore,
    *,
    export_dir: str | Path,
    session_limit: int = 50,
    topic_limit: int = 100,
    redact_sensitive: bool = True,
) -> Dict[str, Any]:
    root = Path(export_dir).expanduser().resolve()
    if root == Path(root.anchor):
        raise ValueError("Refusing to export the memory wiki to a filesystem root")
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    manifest_path = root / MANIFEST_NAME
    previously_owned: Set[str] = set()
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            candidates = manifest.get("generated_files", []) if isinstance(manifest, dict) else []
            previously_owned = {
                str(rel)
                for rel in candidates
                if isinstance(rel, str)
                and rel.endswith(".md")
                and not Path(rel).is_absolute()
                and ".." not in Path(rel).parts
            }
        except (OSError, ValueError, TypeError):
            previously_owned = set()

    topics = _redact_rows(store.list_topics(limit=topic_limit), redact_sensitive)
    sessions = store.list_sessions(limit=session_limit)
    if redact_sensitive:
        sessions = [{**session, "summary": ""} if _is_sensitive_item(session) else session for session in sessions]
    preferences = _redact_rows(store.list_preferences(limit=200), redact_sensitive)
    policies = _redact_rows(store.list_policies(limit=200), redact_sensitive)
    contradictions = _redact_rows(store.recent_contradictions(limit=200), redact_sensitive)
    counts = store.counts()

    # Gather all active facts grouped by category for the index page
    facts_by_category: Dict[str, List[Dict[str, Any]]] = {}
    try:
        all_facts = _redact_rows(store.list_active_facts(limit=500), redact_sensitive)
    except AttributeError:
        # Fallback: query directly if list_active_facts doesn't exist
        all_facts = []
    for f in all_facts:
        cat = str(f.get("category") or "general")
        facts_by_category.setdefault(cat, []).append(f)

    topic_paths = {
        str(topic.get("slug") or ""): _topic_page_path(str(topic.get("slug") or ""))
        for topic in topics
        if topic.get("slug")
    }
    session_paths = {
        str(
            session.get("session_id") or ""
        ): f"sessions/{_safe_page_name(str(session.get('session_id') or ''), fallback='session')}"
        for session in sessions
        if session.get("session_id")
    }

    expected: Dict[str, str] = {}
    expected["index.md"] = _render_index(
        counts=counts,
        topics=topics,
        sessions=sessions,
        preferences=preferences,
        policies=policies,
        contradictions=contradictions,
        topic_paths=topic_paths,
        session_paths=session_paths,
        facts_by_category=facts_by_category,
    )
    expected["preferences/index.md"] = _render_preferences_index(preferences, session_paths=session_paths)
    expected["policies/index.md"] = _render_policies_index(policies, session_paths=session_paths)
    expected["contradictions/index.md"] = _render_contradictions_index(contradictions)

    for topic in topics:
        slug = str(topic.get("slug") or "")
        if not slug:
            continue
        topic_id = topic.get("id")
        if topic_id is None:
            continue
        expected[topic_paths[slug]] = _render_topic_page(
            topic,
            facts=_redact_rows(store.topic_supporting_facts(int(topic_id), limit=20), redact_sensitive),
            contradictions=contradictions,
            session_paths=session_paths,
        )

    for session in sessions:
        session_id = str(session.get("session_id") or "")
        if not session_id:
            continue
        rel = session_paths.get(session_id)
        if not rel:
            continue
        artifacts = store.get_session_artifacts(session_id, limit=20)
        for section in ("facts", "preferences", "journals", "summaries", "traces"):
            artifacts[section] = _redact_rows(list(artifacts.get(section, [])), redact_sensitive)
        expected[rel] = _render_session_page(
            session,
            artifacts=artifacts,
            topic_paths=topic_paths,
        )

    written = 0
    for rel, content in sorted(expected.items()):
        if _write_if_changed(_safe_output_path(root, rel), content):
            written += 1

    for subdir in MANAGED_SUBDIRS:
        managed_dir = _safe_output_path(root, f"{subdir}/.directory-safety-check").parent
        managed_dir.mkdir(parents=True, exist_ok=True)

    expected_paths = set(expected.keys())
    stale_paths = sorted(previously_owned - expected_paths)
    pruned = 0
    for rel in stale_paths:
        stale_file = (root / rel).resolve()
        if root in stale_file.parents and stale_file.is_file():
            stale_file.unlink()
            pruned += 1

    _write_if_changed(
        manifest_path,
        json.dumps(
            {"version": 1, "generated_files": sorted(expected_paths)},
            ensure_ascii=False,
            indent=2,
        ),
    )

    return {
        "root": str(root),
        "generated_files": len(expected_paths),
        "written_files": int(written),
        "pruned_files": pruned,
        "topic_pages": len(topics),
        "session_pages": len(sessions),
        "index_files": 4,
    }
