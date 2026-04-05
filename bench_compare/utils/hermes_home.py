from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

BASELINE_CHAR_LIMITS = {"memory": 2200, "user": 1375}

DEFAULT_ADDON_CONFIG = {
    "db_path": "$HERMES_HOME/consolidating_memory.db",
    "extractor_backend": "heuristic",
    "retrieval_backend": "fts",
    "prefetch_limit": 8,
    "max_topic_facts": 5,
    "topic_summary_chars": 650,
    "session_summary_chars": 900,
    "min_hours": 0,
    "min_sessions": 0,
    "decay_half_life_days": 90,
    "decay_min_salience": 0.15,
    "episode_body_retention_hours": 0,
    "builtin_snapshot_sync_enabled": True,
    "builtin_snapshot_memory_chars": BASELINE_CHAR_LIMITS["memory"],
    "builtin_snapshot_user_chars": BASELINE_CHAR_LIMITS["user"],
    "wiki_export_enabled": False,
    "wiki_export_on_consolidate": False,
}


def ensure_fresh_home(hermes_home: Path) -> Path:
    home = hermes_home.expanduser().resolve()
    default_home = Path("~/.hermes").expanduser().resolve()
    if home == default_home:
        raise ValueError("Refusing to touch ~/.hermes during the benchmark.")
    if str(home) in {home.anchor, ""}:
        raise ValueError(f"Refusing to use unsafe HERMES_HOME path: {home}")
    if home.exists():
        shutil.rmtree(home)
    (home / "memories").mkdir(parents=True, exist_ok=True)
    return home


def addon_db_path(hermes_home: Path, addon_config: Mapping[str, Any]) -> Path:
    raw = str(addon_config.get("db_path") or "$HERMES_HOME/consolidating_memory.db")
    return Path(raw.replace("$HERMES_HOME", str(hermes_home))).expanduser().resolve()


def write_config(
    hermes_home: Path,
    addon_config: Mapping[str, Any] | None = None,
    runtime_model_config: Mapping[str, Any] | None = None,
) -> Path:
    lines: list[str] = []
    if runtime_model_config:
        lines.extend(
            [
                "model:",
                f"  default: {_yaml_scalar(runtime_model_config.get('default', ''))}",
                f"  provider: {_yaml_scalar(runtime_model_config.get('provider', ''))}",
                f"  base_url: {_yaml_scalar(runtime_model_config.get('base_url', ''))}",
            ]
        )
    lines.extend(
        [
            "memory:",
            "  memory_enabled: true",
            "  user_profile_enabled: true",
            f"  memory_char_limit: {BASELINE_CHAR_LIMITS['memory']}",
            f"  user_char_limit: {BASELINE_CHAR_LIMITS['user']}",
        ]
    )
    if addon_config:
        lines.append("  provider: consolidating_local")
        lines.append("plugins:")
        lines.append("  consolidating-local-memory:")
        for key, value in addon_config.items():
            lines.append(f"    {key}: {_yaml_scalar(value)}")
    config_path = hermes_home / "config.yaml"
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    env_path = hermes_home / ".env"
    if not env_path.exists():
        env_path.write_text("", encoding="utf-8")
    return config_path


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if value is None:
        return '""'
    return json.dumps(str(value))


@contextmanager
def temporary_hermes_home(hermes_home: Path) -> Iterator[None]:
    previous = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = str(hermes_home)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = previous
