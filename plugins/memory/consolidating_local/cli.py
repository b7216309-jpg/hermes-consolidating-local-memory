"""Native Hermes CLI commands for offline memory administration."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from .admin import main as admin_main


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def _plugin_config(hermes_home: Path | None = None) -> dict[str, Any]:
    home = hermes_home or _hermes_home()
    config_path = home / "config.yaml"
    if not config_path.is_file():
        return {}
    try:
        import yaml

        with open(config_path, encoding="utf-8-sig") as handle:
            config = yaml.safe_load(handle) or {}
        return dict(config.get("plugins", {}).get("consolidating-local-memory", {}) or {})
    except Exception:
        return {}


def _default_database_path() -> str:
    hermes_home = _hermes_home()

    configured = "$HERMES_HOME/consolidating_memory.db"
    plugin_config = _plugin_config(hermes_home)
    configured = str(plugin_config.get("db_path") or configured)
    return str(Path(configured.replace("$HERMES_HOME", str(hermes_home))).expanduser().resolve())


def _scoped_database_path(
    base_path: str | Path,
    *,
    scope_mode: str,
    platform: str = "",
    user_id: str = "",
    agent_identity: str = "",
) -> str:
    base = Path(base_path).expanduser().resolve()
    mode = str(scope_mode or "user").strip().lower()
    clean_platform = str(platform or "").strip().lower()
    clean_user = str(user_id or "").strip()
    clean_agent = str(agent_identity or "").strip()
    if not any((clean_platform, clean_user, clean_agent)):
        return str(base)
    if mode == "global":
        raise ValueError("Scope identity options cannot be used when memory_scope=global")
    if not clean_platform:
        raise ValueError("--scope-platform is required for a scoped database")
    if mode == "user":
        if not clean_user:
            raise ValueError("--scope-user-id is required when memory_scope=user")
        scope_parts = ["user", clean_platform, clean_user]
    elif mode == "agent":
        if not (clean_user or clean_agent):
            raise ValueError("--scope-user-id or --scope-agent-identity is required when memory_scope=agent")
        scope_parts = ["agent", clean_platform, clean_user or "anonymous", clean_agent or "default"]
    else:
        raise ValueError(f"Unsupported configured memory_scope: {mode}")
    digest = hashlib.sha256("\x1f".join(scope_parts).encode("utf-8")).hexdigest()[:24]
    scopes_dir = base.parent / f"{base.stem}_scopes"
    return str((scopes_dir / f"{digest}{base.suffix or '.db'}").resolve())


def _database_path_from_args(args) -> str:
    base_path = str(getattr(args, "db", "") or _default_database_path())
    plugin_config = _plugin_config()
    try:
        return _scoped_database_path(
            base_path,
            scope_mode=str(plugin_config.get("memory_scope") or "user"),
            platform=str(getattr(args, "scope_platform", "") or ""),
            user_id=str(getattr(args, "scope_user_id", "") or ""),
            agent_identity=str(getattr(args, "scope_agent_identity", "") or ""),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def consolidating_local_command(args) -> None:
    command = getattr(args, "consolidating_local_command", None) or "doctor"
    argv = ["--db", _database_path_from_args(args), command]
    if command == "doctor" and bool(getattr(args, "repair", False)):
        argv.append("--repair")
    elif command in {"backup", "export"}:
        argv.append(str(args.destination))
        if command == "export" and bool(getattr(args, "include_sensitive", False)):
            argv.append("--include-sensitive")
    elif command in {"restore", "import"}:
        argv.append(str(args.source))
        if bool(getattr(args, "confirm", False)):
            argv.append("--confirm")
    elif command == "retry-failed":
        if bool(getattr(args, "confirm", False)):
            argv.append("--confirm")
        argv.extend(["--limit", str(getattr(args, "limit", 100))])
    elif command == "onboard":
        answers = str(getattr(args, "answers", "") or "")
        template = str(getattr(args, "template", "") or "")
        if answers:
            argv.extend(["--answers", answers])
        if template:
            argv.extend(["--template", template])
        if bool(getattr(args, "preview_only", False)):
            argv.append("--preview-only")
        if bool(getattr(args, "yes", False)):
            argv.append("--yes")
        if bool(getattr(args, "skip_sensitive", False)):
            argv.append("--skip-sensitive")
    admin_main(argv)


def register_cli(subparser) -> None:
    """Build the ``hermes consolidating_local`` administration tree."""
    subparser.add_argument(
        "--db",
        default="",
        help="Database path (defaults to the configured unscoped database)",
    )
    subparser.add_argument(
        "--scope-platform",
        default="",
        help="Gateway platform used to derive a configured user/agent-scoped database",
    )
    subparser.add_argument("--scope-user-id", default="", help="Stable gateway user ID used for scoped memory")
    subparser.add_argument("--scope-agent-identity", default="", help="Agent identity used when memory_scope=agent")
    commands = subparser.add_subparsers(dest="consolidating_local_command")

    doctor = commands.add_parser("doctor", help="Check integrity, indexes, links, and queue state")
    doctor.add_argument("--repair", action="store_true", help="Rebuild indexes and remove dangling links")

    backup = commands.add_parser("backup", help="Create a consistent SQLite backup")
    backup.add_argument("destination")

    restore = commands.add_parser("restore", help="Verify and atomically restore a backup")
    restore.add_argument("source")
    restore.add_argument("--confirm", action="store_true", help="Required to replace the destination")

    export = commands.add_parser("export", help="Write a portable JSON export")
    export.add_argument("destination")
    export.add_argument("--include-sensitive", action="store_true")

    import_parser = commands.add_parser("import", help="Import a portable JSON export")
    import_parser.add_argument("source")
    import_parser.add_argument("--confirm", action="store_true", help="Required before importing")

    retry = commands.add_parser("retry-failed", help="Requeue recoverable dead-letter operations")
    retry.add_argument("--confirm", action="store_true", help="Required because work may repeat partially")
    retry.add_argument("--limit", type=int, default=100)

    onboard = commands.add_parser("onboard", help="Build a reviewed, local-only user memory profile")
    onboard.add_argument("--answers", default="", help="Read answers from a JSON object instead of prompting")
    onboard.add_argument("--template", default="", help="Create a blank answer JSON file and exit")
    onboard.add_argument("--preview-only", action="store_true", help="Show the proposed memories without writing")
    onboard.add_argument("--yes", action="store_true", help="Apply the reviewed plan without the final prompt")
    onboard.add_argument(
        "--skip-sensitive", action="store_true", help="Exclude health, financial, identity, and location entries"
    )

    commands.add_parser("maintain", help="Apply retention, size-budget, and vacuum maintenance")
    subparser.set_defaults(func=consolidating_local_command)
