from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

ENTRY_DELIMITER = "\n§\n"

SECTION_MAP = {
    "Relevant summaries": "summaries",
    "Active preferences and workflow rules": "preferences",
    "Direct matches": "facts",
    "Active direct matches": "facts",
    "Current direct matches": "facts",
    "Current topic snapshots": "topics",
    "Current winner snapshot": "snapshot",
    "Current workflow winners": "snapshot",
    "Recent journal notes": "journals",
    "Changed assumptions": "contradictions",
    "Provenance trail": "provenance",
}


def memory_paths(hermes_home: Path) -> dict[str, Path]:
    return {
        "memory": hermes_home / "memories" / "MEMORY.md",
        "user": hermes_home / "memories" / "USER.md",
    }


def read_memory_entries(path: Path) -> list[str]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    return [entry.strip() for entry in raw.split(ENTRY_DELIMITER) if entry.strip()]


def write_memory_entries(path: Path, entries: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_entries = [str(entry).strip() for entry in entries if str(entry).strip()]
    raw = ENTRY_DELIMITER.join(clean_entries) if clean_entries else ""
    path.write_text(raw, encoding="utf-8")


def joined_char_count(entries: Iterable[str]) -> int:
    clean_entries = [str(entry).strip() for entry in entries if str(entry).strip()]
    if not clean_entries:
        return 0
    return len(ENTRY_DELIMITER.join(clean_entries))


def read_baseline_snapshot(hermes_home: Path) -> dict[str, Any]:
    paths = memory_paths(hermes_home)
    memory_entries = read_memory_entries(paths["memory"])
    user_entries = read_memory_entries(paths["user"])
    return {
        "memory_entries": memory_entries,
        "user_entries": user_entries,
        "memory_chars": joined_char_count(memory_entries),
        "user_chars": joined_char_count(user_entries),
        "entries": memory_entries + user_entries,
    }


def query_rows(db_path: Path, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def query_value(db_path: Path, sql: str, params: Iterable[Any] = (), default: Any = 0) -> Any:
    rows = query_rows(db_path, sql, params)
    if not rows:
        return default
    row = rows[0]
    if not row:
        return default
    return next(iter(row.values()), default)


def parse_context_sections(context: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = ""
    for raw_line in str(context or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        key = SECTION_MAP.get(line.rstrip(":"))
        if key:
            current = key
            sections.setdefault(current, [])
            continue
        if line.startswith("- ") and current:
            sections.setdefault(current, []).append(clean_context_item(line))
    return sections


def flattened_context_items(context: str) -> list[str]:
    items: list[str] = []
    for section_items in parse_context_sections(context).values():
        items.extend(section_items)
    return items


def clean_context_item(line: str) -> str:
    text = re.sub(r"^\-\s*", "", str(line or "").strip())
    text = re.sub(r"^\[[^\]]+\]\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def salience_stats(values: Iterable[float]) -> dict[str, float]:
    values_list = sorted(float(value) for value in values)
    if not values_list:
        return {"count": 0.0, "min": 0.0, "max": 0.0, "avg": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "count": float(len(values_list)),
        "min": round(values_list[0], 4),
        "max": round(values_list[-1], 4),
        "avg": round(sum(values_list) / len(values_list), 4),
        "p50": round(_percentile(values_list, 0.50), 4),
        "p90": round(_percentile(values_list, 0.90), 4),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction
