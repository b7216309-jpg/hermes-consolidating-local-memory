from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _hermes_home(value: str) -> Path:
    raw = value or os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(raw).expanduser().resolve()


def _hermes_executable() -> str | None:
    executable = shutil.which("hermes")
    if executable:
        return executable
    user_executable = Path.home() / ".local" / "bin" / "hermes"
    return str(user_executable) if user_executable.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Install or update the consolidating_local Hermes memory provider.")
    parser.add_argument("--hermes-home", default="", help="Hermes home (defaults to HERMES_HOME or ~/.hermes)")
    parser.add_argument("--dry-run", action="store_true", help="Show the destination without changing files")
    parser.add_argument(
        "--no-enable",
        action="store_true",
        help="Do not enable the lifecycle observer; automatic gateway capture will remain unavailable",
    )
    args = parser.parse_args()

    source = Path(__file__).resolve().parent / "plugins" / "memory" / "consolidating_local"
    home = _hermes_home(args.hermes_home)
    plugins_dir = home / "plugins"
    destination = plugins_dir / "consolidating_local"
    if not (source / "__init__.py").is_file() or not (source / "plugin.yaml").is_file():
        print(f"Invalid source tree: {source}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"Would install {source} -> {destination}")
        return 0

    plugins_dir.mkdir(parents=True, exist_ok=True)
    stage = plugins_dir / ".consolidating_local.installing"
    backup = plugins_dir / ".consolidating_local.backup"
    updating_existing = destination.exists() or backup.exists()
    if backup.exists() and not destination.exists():
        # Recover an installation interrupted after the old plugin was moved
        # aside but before the staged replacement became active.
        os.replace(backup, destination)
    elif backup.exists():
        shutil.rmtree(backup)
    if stage.exists():
        shutil.rmtree(stage)

    shutil.copytree(source, stage, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    try:
        if destination.exists():
            os.replace(destination, backup)
        os.replace(stage, destination)
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        if backup.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)

    print(f"Installed consolidating_local to {destination}")
    if not args.no_enable:
        hermes = _hermes_executable()
        if updating_existing:
            print("Existing install updated; preserving its enablement and grant settings.")
        elif not hermes:
            print(
                "Plugin copied, but Hermes was not found. Run `hermes plugins enable consolidating_local --no-allow-tool-override` before use.",
                file=sys.stderr,
            )
            return 1
        else:
            completed = subprocess.run(
                [hermes, "plugins", "enable", "consolidating_local", "--no-allow-tool-override"],
                check=False,
            )
            if completed.returncode:
                print(
                    "Plugin copied, but its lifecycle observer could not be enabled. Automatic gateway capture is fail-closed until it is enabled.",
                    file=sys.stderr,
                )
                return completed.returncode
    print("Next: run `hermes memory setup` and select `consolidating_local`, or set memory.provider in config.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
