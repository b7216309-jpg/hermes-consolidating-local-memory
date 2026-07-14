from __future__ import annotations

import json

import pytest

from consolidating_local.llm_client import OpenAICompatibleEmbeddings, extract_json_object
from consolidating_local.store import MemoryStore
from consolidating_local.wiki_export import MANIFEST_NAME, _safe_output_path, export_compiled_wiki


def test_json_extraction_handles_surrounding_braces():
    assert extract_json_object('analysis {not json} then {"facts": [{"content": "ok"}]} trailing') == {
        "facts": [{"content": "ok"}]
    }


def test_embeddings_are_ordered_and_validated():
    client = OpenAICompatibleEmbeddings(model="embed", base_url="http://localhost")
    client._post_json = lambda *_args, **_kwargs: {
        "data": [
            {"index": 1, "embedding": [3, 4]},
            {"index": 0, "embedding": [1, 2]},
        ]
    }
    assert client.embed_texts(["a", "b"]) == [[1.0, 2.0], [3.0, 4.0]]

    client._post_json = lambda *_args, **_kwargs: {
        "data": [{"index": 0, "embedding": [1, 2]}, {"index": 1, "embedding": [float("nan"), 4]}]
    }
    assert client.embed_texts(["a", "b"]) is None


def test_wiki_output_paths_cannot_escape_export_root(tmp_path):
    root = (tmp_path / "wiki").resolve()
    root.mkdir()
    with pytest.raises(ValueError, match="Unsafe wiki output path"):
        _safe_output_path(root, "../outside.md")


def test_wiki_export_escapes_content_and_only_prunes_manifest_owned_files(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    wiki = tmp_path / "wiki"
    try:
        result = store.upsert_fact(
            content="User likes <script>alert(1)</script> | tables",
            category="custom_category",
            topic="unsafe-topic",
            source="test",
        )
        store.upsert_fact(
            content="Medical diagnosis is private",
            category="general",
            topic="private-topic",
            source="test",
            sensitivity="health",
        )
        store.upsert_policy(
            key="private-policy",
            label="Private policy",
            content="Password is private",
            sensitivity="credential",
        )
        store.ensure_memory_session("private-session", summary="Medical diagnosis summary")
        store.rebuild_topics(max_facts=5, max_chars=500)
        export_compiled_wiki(store, export_dir=wiki)
        index = (wiki / "index.md").read_text(encoding="utf-8")
        assert "&lt;script&gt;" in index
        assert "\\|" in index
        assert "Custom_Category" in index
        rendered_wiki = "\n".join(path.read_text(encoding="utf-8") for path in wiki.rglob("*.md")).lower()
        assert "medical diagnosis" not in rendered_wiki
        assert "password is private" not in rendered_wiki

        user_note = wiki / "topics" / "my-team-note.md"
        user_note.write_text("keep me", encoding="utf-8")
        store.deactivate_memory_item("fact", int(result["fact"]["id"]), reason="test", source="test")
        store.rebuild_topics(max_facts=5, max_chars=500)
        second = export_compiled_wiki(store, export_dir=wiki)

        assert user_note.read_text(encoding="utf-8") == "keep me"
        assert second["pruned_files"] >= 1
        manifest = json.loads((wiki / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert "topics/my-team-note.md" not in manifest["generated_files"]
    finally:
        store.close()
