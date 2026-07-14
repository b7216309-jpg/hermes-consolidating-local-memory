"""Offline administration for consolidating-local memory.

Run with: python -m plugins.memory.consolidating_local.admin --db PATH <command>
Stop Hermes before restore, import, repair, or maintenance operations.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from .store import MemoryStore


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def _database_api(encryption_key: str):
    if not encryption_key:
        return sqlite3
    try:
        from sqlcipher3 import dbapi2 as sqlcipher
    except ImportError as exc:
        raise RuntimeError(
            "CONSOLIDATING_MEMORY_DB_KEY is set, but the optional sqlcipher3 package is not installed"
        ) from exc
    return sqlcipher


def _apply_encryption_key(connection: Any, encryption_key: str) -> None:
    if not encryption_key:
        return
    escaped_key = encryption_key.replace("'", "''")
    connection.execute(f"PRAGMA key = '{escaped_key}'")
    cipher_row = connection.execute("PRAGMA cipher_version").fetchone()
    if not cipher_row or not str(cipher_row[0] or "").strip():
        raise RuntimeError("The selected SQLite driver does not provide SQLCipher encryption")


def _restore(source: Path, destination: Path, *, encryption_key: str = "") -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dbapi = _database_api(encryption_key)
    temporary: Path | None = None
    source_connection = None
    destination_connection = None
    moved_sidecars: list[tuple[Path, Path]] = []
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".db", delete=False) as handle:
            temporary = Path(handle.name)
        source_connection = dbapi.connect(str(source.resolve()))
        destination_connection = dbapi.connect(str(temporary))
        _apply_encryption_key(source_connection, encryption_key)
        _apply_encryption_key(destination_connection, encryption_key)
        source_connection.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
        source_connection.execute("PRAGMA query_only = ON")
        integrity = source_connection.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise RuntimeError(f"Backup failed integrity_check: {integrity}")
        source_connection.backup(destination_connection)
        destination_connection.commit()
        restored_integrity = destination_connection.execute("PRAGMA integrity_check").fetchone()
        if not restored_integrity or restored_integrity[0] != "ok":
            raise RuntimeError(f"Restored database failed integrity_check: {restored_integrity}")
        destination_connection.close()
        destination_connection = None
        source_connection.close()
        source_connection = None
        try:
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(destination) + suffix)
                if sidecar.exists():
                    held = Path(str(temporary) + suffix + ".old")
                    os.replace(sidecar, held)
                    moved_sidecars.append((sidecar, held))
        except Exception:
            for sidecar, held in reversed(moved_sidecars):
                if held.exists():
                    os.replace(held, sidecar)
            moved_sidecars.clear()
            raise
        try:
            os.replace(temporary, destination)
            temporary = None
        except Exception:
            for sidecar, held in reversed(moved_sidecars):
                if held.exists():
                    os.replace(held, sidecar)
            moved_sidecars.clear()
            raise
        for _, held in moved_sidecars:
            if held.exists():
                held.unlink()
        moved_sidecars.clear()
        try:
            os.chmod(destination, 0o600)
        except OSError:
            pass
    finally:
        if destination_connection is not None:
            destination_connection.close()
        if source_connection is not None:
            source_connection.close()
        if temporary is not None and temporary.exists():
            temporary.unlink()
        for _, held in moved_sidecars:
            if held.exists():
                held.unlink()
    return {"restored_from": str(source.resolve()), "database": str(destination.resolve())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline consolidating-memory administration")
    parser.add_argument("--db", required=True, help="SQLite database path")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--repair", action="store_true")
    backup_parser = subparsers.add_parser("backup")
    backup_parser.add_argument("destination")
    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("source")
    restore_parser.add_argument("--confirm", action="store_true", help="Required to replace --db")
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("destination")
    export_parser.add_argument("--include-sensitive", action="store_true")
    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("source")
    import_parser.add_argument("--confirm", action="store_true")
    retry_parser = subparsers.add_parser("retry-failed")
    retry_parser.add_argument("--confirm", action="store_true", help="Required to retry dead-letter operations")
    retry_parser.add_argument("--limit", type=int, default=100)
    subparsers.add_parser("maintain")
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser().resolve()
    encryption_key = os.environ.get("CONSOLIDATING_MEMORY_DB_KEY", "")
    if args.command == "restore":
        if not args.confirm:
            parser.error("restore requires --confirm and Hermes must be stopped")
        result = _restore(Path(args.source).expanduser(), db_path, encryption_key=encryption_key)
        print(json.dumps({"success": True, **result}, indent=2))
        return 0

    store = MemoryStore(db_path, encryption_key=encryption_key)
    try:
        if args.command == "doctor":
            result = store.doctor(repair=bool(args.repair))
        elif args.command == "backup":
            result = {"path": store.backup_to(args.destination)}
        elif args.command == "export":
            target = Path(args.destination).expanduser().resolve()
            _write_json(target, store.export_data(redact_sensitive=not args.include_sensitive))
            result = {"path": str(target)}
        elif args.command == "import":
            if not args.confirm:
                parser.error("import requires --confirm and Hermes must be stopped")
            with open(Path(args.source).expanduser(), encoding="utf-8") as handle:
                payload = json.load(handle)
            result = store.import_data(payload)
        elif args.command == "maintain":
            result = store.maintain()
        elif args.command == "retry-failed":
            if not args.confirm:
                parser.error("retry-failed requires --confirm; a retried operation may repeat partial work")
            result = {
                "retried": store.retry_failed_operations(limit=max(1, min(int(args.limit), 1000))),
                "remaining_failed": store.failed_operation_count(),
                "pending": store.pending_operation_count(),
            }
        else:
            parser.error(f"unknown command: {args.command}")
        print(json.dumps({"success": True, "result": result}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
