from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Sequence
from urllib import error, request

logger = logging.getLogger(__name__)
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def is_codex_backend(base_url: str) -> bool:
    clean = str(base_url or "").strip().rstrip("/").lower()
    return "/backend-api/codex" in clean


def extract_json_object(text: str) -> Dict[str, Any] | None:
    text = str(text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str = "",
        timeout_seconds: int = 45,
        failure_cooldown_seconds: int = 120,
    ):
        self.model = str(model or "").strip()
        self.base_url = str(base_url or "").rstrip("/")
        self.api_key = str(api_key or "")
        self.timeout_seconds = max(1, min(int(timeout_seconds), 300))
        self.failure_cooldown_seconds = max(1, min(int(failure_cooldown_seconds), 86400))
        self._last_http_error_code: int | None = None
        self._last_http_error_detail = ""
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._last_request_succeeded: bool | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.model and self.base_url)

    @property
    def backend_kind(self) -> str:
        if is_codex_backend(self.base_url):
            return "codex"
        return "openai_compatible"

    @property
    def circuit_state(self) -> Dict[str, Any]:
        return {
            "open": self._circuit_open_until > time.time(),
            "open_until": self._circuit_open_until,
            "consecutive_failures": self._consecutive_failures,
            "last_error_code": self._last_http_error_code,
            "last_error": self._last_http_error_detail[:500],
        }

    @property
    def last_request_succeeded(self) -> bool | None:
        return self._last_request_succeeded

    def _record_success(self) -> None:
        self._last_request_succeeded = True
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _record_failure(self) -> None:
        self._last_request_succeeded = False
        self._consecutive_failures += 1
        if self._consecutive_failures >= 3:
            multiplier = min(8, 2 ** (self._consecutive_failures - 3))
            self._circuit_open_until = time.time() + (self.failure_cooldown_seconds * multiplier)

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any] | None:
        if not self.enabled:
            self._last_request_succeeded = None
            return None
        if self._circuit_open_until > time.time():
            self._last_request_succeeded = False
            self._last_http_error_detail = "model endpoint circuit is open"
            return None
        self._last_http_error_code = None
        self._last_http_error_detail = ""
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ValueError("Model response exceeded the 2 MiB safety limit")
                decoded = json.loads(raw.decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise ValueError("Model response must be a JSON object")
                self._record_success()
                return decoded
        except error.HTTPError as exc:
            try:
                detail = exc.read(4096).decode("utf-8", errors="ignore")
            except Exception:
                detail = str(exc)
            self._last_http_error_code = int(exc.code)
            self._last_http_error_detail = str(detail or "")
            self._record_failure()
            logger.warning("Local model request failed (%s): %s", exc.code, detail[:500])
            return None
        except Exception as exc:
            self._last_http_error_detail = str(exc)
            self._record_failure()
            logger.warning("Local model request failed: %s", exc)
            return None

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 700,
    ) -> Dict[str, Any] | None:
        content = self.chat_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not content:
            return None
        data = extract_json_object(content)
        if data is None:
            self._last_http_error_detail = "model response did not contain a JSON object"
            self._record_failure()
        return data

    def chat_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 700,
    ) -> str:
        if self.backend_kind == "codex":
            return self._codex_responses_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        body = self._post_json("/chat/completions", payload)
        if not body:
            return ""
        return _extract_openai_chat_text(body)

    def _codex_responses_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        if not self.enabled:
            self._last_request_succeeded = None
            return ""
        if self._circuit_open_until > time.time():
            self._last_request_succeeded = False
            self._last_http_error_detail = "model endpoint circuit is open"
            return ""
        url = f"{self.base_url}/responses"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "instructions": str(system_prompt or ""),
            "input": [
                {
                    "role": "user",
                    "content": str(user_prompt or ""),
                }
            ],
            "store": False,
            "stream": True,
        }
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                response_body = response.read(MAX_RESPONSE_BYTES + 1)
                if len(response_body) > MAX_RESPONSE_BYTES:
                    raise ValueError("Model response exceeded the 2 MiB safety limit")
                raw = response_body.decode("utf-8", errors="replace")
                self._record_success()
        except error.HTTPError as exc:
            try:
                detail = exc.read(4096).decode("utf-8", errors="ignore")
            except Exception:
                detail = str(exc)
            logger.warning("Codex responses request failed (%s): %s", exc.code, detail[:500])
            self._last_http_error_code = int(exc.code)
            self._last_http_error_detail = str(detail or "")
            self._record_failure()
            return ""
        except Exception as exc:
            logger.warning("Codex responses request failed: %s", exc)
            self._last_http_error_detail = str(exc)
            self._record_failure()
            return ""
        return _extract_codex_stream_text(raw)


class OpenAICompatibleEmbeddings(OpenAICompatibleLLM):
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str = "",
        timeout_seconds: int = 45,
        failure_cooldown_seconds: int = 120,
    ):
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            failure_cooldown_seconds=failure_cooldown_seconds,
        )
        self._embedding_support_override: bool | None = None

    @property
    def supports_embeddings(self) -> bool:
        if not self.enabled or self.backend_kind == "codex":
            return False
        if self._embedding_support_override is False:
            return False
        return True

    def _mark_embeddings_unsupported_if_needed(self) -> None:
        detail = str(self._last_http_error_detail or "").lower()
        if self._last_http_error_code in {404, 405, 501}:
            self._embedding_support_override = False
            return
        if "does not support embeddings" in detail or ("embedding" in detail and "not support" in detail):
            self._embedding_support_override = False

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]] | None:
        clean = [str(text or "").strip() for text in texts]
        if not clean or not self.supports_embeddings:
            return None
        body = self._post_json(
            "/embeddings",
            {
                "model": self.model,
                "input": clean,
            },
        )
        if not body or not isinstance(body.get("data"), list):
            self._mark_embeddings_unsupported_if_needed()
            return None
        items = list(body.get("data", []))
        if items and all(isinstance(item, dict) and isinstance(item.get("index"), int) for item in items):
            indexes = [int(item["index"]) for item in items]
            if sorted(indexes) != list(range(len(clean))):
                return None
            items.sort(key=lambda item: int(item["index"]))
        vectors: List[List[float]] = []
        expected_dimensions = 0
        for item in items:
            if not isinstance(item, dict):
                return None
            vector = item.get("embedding")
            if not isinstance(vector, list):
                return None
            try:
                parsed = [float(value) for value in vector]
            except Exception:
                return None
            if not parsed or not all(math.isfinite(value) for value in parsed):
                return None
            if expected_dimensions and len(parsed) != expected_dimensions:
                return None
            expected_dimensions = len(parsed)
            vectors.append(parsed)
        if len(vectors) != len(clean):
            return None
        self._embedding_support_override = True
        return vectors


def env_or_blank(name: str) -> str:
    return os.environ.get(name, "").strip()


def _extract_codex_stream_text(raw: str) -> str:
    text = str(raw or "")
    if not text.strip():
        return ""
    pieces: List[str] = []
    final_text = ""
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            item = json.loads(payload)
        except Exception:
            continue
        event_type = str(item.get("type") or "")
        if event_type == "response.output_text.delta":
            pieces.append(str(item.get("delta") or ""))
        elif event_type == "response.output_text.done":
            final_text = str(item.get("text") or "")
        elif event_type == "response.completed":
            response = item.get("response") or {}
            for output in response.get("output") or []:
                if not isinstance(output, dict):
                    continue
                for content in output.get("content") or []:
                    if not isinstance(content, dict):
                        continue
                    if str(content.get("type") or "") == "output_text":
                        final_text = str(content.get("text") or final_text)
    if final_text:
        return final_text
    return "".join(pieces).strip()


def _extract_openai_chat_text(body: Dict[str, Any]) -> str:
    content = ""
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    raw_content = message.get("content", "")
    if isinstance(raw_content, list):
        content = " ".join(str(block.get("text", "")) for block in raw_content if isinstance(block, dict))
    else:
        content = str(raw_content or "")
    # Some reasoning models (Qwen 3.x, DeepSeek-R1, etc.) put thinking in
    # reasoning_content and the actual answer in content.  When content is
    # empty but reasoning_content contains a JSON block, extract it.
    if not content.strip():
        reasoning = str(message.get("reasoning_content") or "")
        if reasoning:
            content = reasoning
    return content
