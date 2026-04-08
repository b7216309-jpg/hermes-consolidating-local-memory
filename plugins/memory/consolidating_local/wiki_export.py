from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from .store import MemoryStore, normalize_whitespace, pretty_topic, slugify

MANAGED_SUBDIRS = ("topics", "sessions", "preferences", "policies", "contradictions")

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
    slug = slugify(clean) if clean else fallback
    digest = hashlib.sha1(clean.encode("utf-8")).hexdigest()[:8] if clean else "00000000"
    return f"{slug}-{digest}.md"


def _normalize_markdown(text: str) -> str:
    clean = str(text or "").replace("\r\n", "\n").strip()
    return clean + "\n"


def _write_if_changed(path: Path, content: str) -> bool:
    normalized = _normalize_markdown(content)
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current == normalized:
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normalized, encoding="utf-8")
    return True


def _relative_link(from_rel: str, to_rel: str) -> str:
    return Path(os.path.relpath(to_rel, start=str(Path(from_rel).parent))).as_posix()


def _bullet_link(label: str, target_rel: str, *, from_rel: str) -> str:
    return f"[{label}]({_relative_link(from_rel, target_rel)})"


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
    for cat in _CATEGORY_ORDER:
        cat_facts = facts_by_category.get(cat, [])
        if not cat_facts:
            continue
        label = _CATEGORY_LABELS.get(cat, cat.title())
        lines.append(f"### {label} ({len(cat_facts)})")
        lines.append("")
        lines.append("| Subject | Value | Content | Imp | Salience |")
        lines.append("| --- | --- | --- | ---: | --- |")
        for f in sorted(cat_facts, key=lambda x: (-int(x.get("importance") or 0), str(x.get("subject_key") or ""))):
            sk = str(f.get("subject_key") or "—")
            # Strip common prefixes for readability
            for pfx in ("user:", "environment:", "project:", "workflow:", "general:"):
                if sk.startswith(pfx):
                    sk = sk[len(pfx):]
                    break
            vk = str(f.get("value_key") or "—")
            content = str(f.get("content") or "")[:80]
            imp = int(f.get("importance") or 0)
            sal = float(f.get("salience") or 0)
            lines.append(f"| `{sk}` | `{vk}` | {content} | {imp} | {_salience_tag(sal)} |")
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
            link = _bullet_link(title, target, from_rel=rel) if target else title
            cat = str(topic.get("category") or "general")
            sal = float(topic.get("salience") or 0)
            updated = _fmt_ts_short(topic.get("updated_at")) or "—"
            lines.append(f"| {link} | `{cat}` | {_salience_tag(sal)} | {updated} |")
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
            link = _bullet_link(f"**{session_id[:12]}…**", target, from_rel=rel) if target else f"**{session_id[:12]}**"
            lines.append(f"- {link} ({started}): {summary[:120]}")
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
            lines.append(f"| `{pk}` | {val} | {imp} |")
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
            lines.append(f"- **{str(item.get('label') or item.get('policy_key') or 'Policy')}**: {str(item.get('content') or '')}")
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
            lines.append(f"| `{sk}` | ~~{loser}~~ | {winner} | {created} |")
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
    rel = f"topics/{topic['slug']}.md"
    title = str(topic.get("title") or pretty_topic(str(topic.get("slug") or "topic")))
    sal = float(topic.get("salience") or 0)
    lines = [
        f"# {title}",
        "",
        "| Property | Value |",
        "| --- | --- |",
        f"| Slug | `{str(topic.get('slug') or '')}` |",
        f"| Category | `{str(topic.get('category') or 'general')}` |",
        f"| Salience | {_salience_tag(sal)} ({sal:.2f}) |",
        f"| Updated | {_fmt_ts(topic.get('updated_at')) or 'unknown'} |",
        "",
        "---",
        "",
        "## Summary",
        "",
        str(topic.get("summary") or "*No summary available.*"),
        "",
        "---",
        "",
        "## Supporting Facts",
        "",
    ]
    if facts:
        lines.append("| Content | Importance | Confidence |")
        lines.append("| --- | ---: | ---: |")
        for fact in facts:
            content = str(fact.get("content") or "")[:100]
            imp = int(fact.get("importance") or 0)
            conf = float(fact.get("confidence") or 0)
            lines.append(f"| {content} | {imp} | {conf:.0%} |")
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
        row for row in contradictions
        if str(row.get("winner_topic") or "") == str(topic.get("slug") or "")
        or str(row.get("loser_topic") or "") == str(topic.get("slug") or "")
    ]
    if topic_contradictions:
        lines.append("| Subject | Before | After |")
        lines.append("| --- | --- | --- |")
        for row in topic_contradictions[:10]:
            winner = normalize_whitespace(str(row.get("winner_content") or ""))[:50]
            loser = normalize_whitespace(str(row.get("loser_content") or ""))[:50]
            lines.append(f"| `{str(row.get('subject_key') or 'unknown')}` | ~~{loser}~~ | {winner} |")
    else:
        lines.append("*No contradictions for this topic.*")

    lines.extend([
        "", "---", "",
        "## Navigation", "",
        f"- [← Back to Wiki]({_relative_link(rel, 'index.md')})",
        f"- [Contradictions]({_relative_link(rel, 'contradictions/index.md')})",
    ])
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
        f"# Session `{session_id[:16]}…`",
        "",
        "| Property | Value |",
        "| --- | --- |",
        f"| Status | `{str(session.get('status') or 'unknown')}` |",
        f"| Started | {started} |",
        f"| Ended | {ended} |",
        f"| Last activity | {_fmt_ts(session.get('last_activity_at')) or 'unknown'} |",
        "",
        "---",
        "",
        "## Summary",
        "",
        str(session.get("summary") or "*No session summary available.*"),
        "",
        "---",
        "",
    ]

    # Facts
    facts = list(artifacts.get("facts", []))
    lines.append("## Facts Extracted")
    lines.append("")
    if facts:
        lines.append("| Content | Topic |")
        lines.append("| --- | --- |")
        for fact in facts:
            content = str(fact.get("content") or "")[:90]
            topic_slug = str(fact.get("topic") or "")
            topic_target = topic_paths.get(topic_slug)
            topic_link = _bullet_link(pretty_topic(topic_slug), topic_target, from_rel=rel) if topic_target else pretty_topic(topic_slug)
            lines.append(f"| {content} | {topic_link} |")
    else:
        lines.append("*No extracted facts linked to this session.*")

    # Preferences
    preferences = list(artifacts.get("preferences", []))
    lines.extend(["", "---", "", "## Preferences", ""])
    if preferences:
        for item in preferences:
            lines.append(f"- {str(item.get('content') or item.get('label') or '')}")
    else:
        lines.append("*No session-specific preferences.*")

    # Policies
    policies = list(artifacts.get("policies", []))
    lines.extend(["", "## Policies", ""])
    if policies:
        for item in policies:
            lines.append(f"- {str(item.get('content') or item.get('label') or '')}")
    else:
        lines.append("*No session-specific policies.*")

    # Journals
    journals = list(artifacts.get("journals", []))
    lines.extend(["", "## Journals", ""])
    if journals:
        for item in journals:
            lines.append(f"- **{str(item.get('label') or 'Journal')}**: {str(item.get('content') or '')}")
    else:
        lines.append("*No journals for this session.*")

    # Traces
    traces = list(artifacts.get("traces", []))
    lines.extend(["", "## Traces", ""])
    if traces:
        for item in traces[:10]:
            lines.append(f"- {str(item.get('content') or '')}")
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
            session_id = str(item.get("source_session_id") or dict(item.get("metadata") or {}).get("session_id") or "")
            lines.append(f"| `{pk}` | {label} | {val} | {imp} | {_salience_tag(sal)} |")
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
            lines.append(f"| **{label}** | {content} | {imp} |")
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
            lines.append(f"| `{sk}` | ~~{loser}~~ | {winner} | {created} |")
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
) -> Dict[str, Any]:
    root = Path(export_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    topics = store.list_topics(limit=topic_limit)
    sessions = store.list_sessions(limit=session_limit)
    preferences = store.list_preferences(limit=200)
    policies = store.list_policies(limit=200)
    contradictions = store.recent_contradictions(limit=200)
    counts = store.counts()

    # Gather all active facts grouped by category for the index page
    facts_by_category: Dict[str, List[Dict[str, Any]]] = {}
    try:
        all_facts = store.list_active_facts(limit=500)
    except AttributeError:
        # Fallback: query directly if list_active_facts doesn't exist
        all_facts = []
    for f in all_facts:
        cat = str(f.get("category") or "general")
        facts_by_category.setdefault(cat, []).append(f)

    topic_paths = {str(topic.get("slug") or ""): f"topics/{str(topic.get('slug') or '')}.md" for topic in topics if topic.get("slug")}
    session_paths = {
        str(session.get("session_id") or ""): f"sessions/{_safe_page_name(str(session.get('session_id') or ''), fallback='session')}"
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
        expected[f"topics/{slug}.md"] = _render_topic_page(
            topic,
            facts=store.topic_supporting_facts(int(topic_id), limit=20),
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
        expected[rel] = _render_session_page(
            session,
            artifacts=store.get_session_artifacts(session_id, limit=20),
            topic_paths=topic_paths,
        )

    written = 0
    for rel, content in sorted(expected.items()):
        if _write_if_changed(root / rel, content):
            written += 1

    for subdir in MANAGED_SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)

    existing_files: Set[str] = set()
    if (root / "index.md").exists():
        existing_files.add("index.md")
    for subdir in MANAGED_SUBDIRS:
        base = root / subdir
        if not base.exists():
            continue
        for file_path in sorted(base.rglob("*.md")):
            existing_files.add(file_path.relative_to(root).as_posix())

    expected_paths = set(expected.keys())
    stale_paths = sorted(existing_files - expected_paths)
    for rel in stale_paths:
        stale_file = root / rel
        if stale_file.exists():
            stale_file.unlink()

    return {
        "root": str(root),
        "generated_files": len(expected_paths),
        "written_files": int(written),
        "pruned_files": len(stale_paths),
        "topic_pages": len(topics),
        "session_pages": len(sessions),
        "index_files": 4,
    }
