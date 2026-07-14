from __future__ import annotations

import pytest

from consolidating_local.store import MemoryStore

pytest.importorskip("sqlcipher3", reason="optional SQLCipher dependency is not installed")


def test_sqlcipher_database_wrong_key_and_backup(tmp_path):
    path = tmp_path / "encrypted.db"
    backup = tmp_path / "encrypted-backup.db"
    key = "test-key-with-'quote"

    store = MemoryStore(path, encryption_key=key)
    try:
        store.upsert_fact(
            content="Encrypted memory remains readable with the correct key",
            category="general",
            topic="encryption",
            source="test",
        )
        store.backup_to(backup)
    finally:
        store.close()

    assert path.read_bytes()[:16] != b"SQLite format 3\x00"
    with pytest.raises(Exception):
        MemoryStore(path, encryption_key="wrong-key")

    restored = MemoryStore(backup, encryption_key=key)
    try:
        assert restored.search("correct key", scope="facts")["facts"]
        assert restored.doctor()["ok"] is True
    finally:
        restored.close()
