from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from consolidating_local import ConsolidatingLocalMemoryProvider


class _ExtractionEndpoint:
    def __init__(self, *, fail_first: bool = False):
        self.fail_first = fail_first
        self.requests: list[dict] = []
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                endpoint.requests.append(payload)
                if endpoint.fail_first and len(endpoint.requests) == 1:
                    body = json.dumps({"error": "transient"}).encode("utf-8")
                    self.send_response(503)
                else:
                    content = json.dumps(
                        {
                            "facts": [
                                {
                                    "content": "User's name is Alice",
                                    "category": "user_pref",
                                    "topic": "user-profile",
                                    "importance": 8,
                                    "confidence": 0.98,
                                    "subject_key": "user:name",
                                    "value_key": "alice",
                                    "exclusive": True,
                                    "polarity": 1,
                                    "source_role": "user",
                                }
                            ]
                        }
                    )
                    body = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def test_no_model_means_episode_capture_without_automatic_fact_guessing(tmp_path):
    provider = ConsolidatingLocalMemoryProvider({"db_path": str(tmp_path / "memory.db")})
    try:
        provider.initialize("session", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
        provider.sync_turn("My name is Alice", "Understood", session_id="session")
        provider._task_queue.join()
        assert provider._store.counts()["episodes"] == 1
        assert provider._store.counts()["facts"] == 0
    finally:
        provider.shutdown()


def test_openai_compatible_extraction_retries_durably_after_transient_http_failure(tmp_path):
    with _ExtractionEndpoint(fail_first=True) as endpoint:
        provider = ConsolidatingLocalMemoryProvider(
            {
                "db_path": str(tmp_path / "memory.db"),
                "llm_model": "memory-extractor",
                "llm_base_url": endpoint.base_url,
                "llm_failure_cooldown_seconds": 1,
                "queue_max_attempts": 5,
            }
        )
        try:
            provider.initialize("session", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
            provider._task_queue.join()
            provider.sync_turn("My name is Alice", "Understood", session_id="session")
            deadline = time.time() + 8
            while time.time() < deadline:
                if len(endpoint.requests) >= 2 and provider._store.pending_operation_count() == 0:
                    break
                time.sleep(0.05)

            assert len(endpoint.requests) >= 2
            assert provider._store.pending_operation_count() == 0
            assert provider._store.counts()["episodes"] == 1
            facts = provider._store.search("Alice", scope="facts")["facts"]
            assert len(facts) == 1
            prompt = endpoint.requests[-1]["messages"][-1]["content"]
            assert "seed_facts" not in prompt
            status = json.loads(provider.handle_tool_call("consolidating_memory", {"action": "status"}))
            assert status["automatic_extraction"] == {"enabled": True, "backend": "llm"}
        finally:
            provider.shutdown()
