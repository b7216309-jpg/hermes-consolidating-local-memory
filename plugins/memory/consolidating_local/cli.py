"""Native Hermes CLI commands for offline memory administration."""

from __future__ import annotations

import os
from pathlib import Path

from .admin import main as admin_main


def _default_database_path() -> str:
    try:
        from hermes_constants import get_hermes_home

        hermes_home = Path(get_hermes_home())
    except Exception:
        hermes_home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")

    configured = "$HERMES_HOME/consolidating_memory.db"
    config_path = hermes_home / "config.yaml"
    if config_path.is_file():
        try:
            import yaml

            with open(config_path, encoding="utf-8-sig") as handle:
                config = yaml.safe_load(handle) or {}
            plugin_config = dict(config.get("plugins", {}).get("consolidating-local-memory", {}) or {})
            configured = str(plugin_config.get("db_path") or configured)
        except Exception:
            pass
    return str(Path(configured.replace("$HERMES_HOME", str(hermes_home))).expanduser().resolve())


def consolidating_local_command(args) -> None:
    command = getattr(args, "consolidating_local_command", None) or "doctor"
    argv = ["--db", str(getattr(args, "db", "") or _default_database_path()), command]
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
    admin_main(argv)


def register_cli(subparser) -> None:
    """Build the ``hermes consolidating_local`` administration tree."""
    subparser.add_argument(
        "--db",
        default="",
        help="Database path (defaults to the configured unscoped database)",
    )
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

    commands.add_parser("maintain", help="Apply retention, size-budget, and vacuum maintenance")
    subparser.set_defaults(func=consolidating_local_command)
