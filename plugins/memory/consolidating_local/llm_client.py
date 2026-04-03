from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict
from urllib import error, request

logger = logging.getLogger(__name__)


def load_hermes_model_defaults(hermes_home: str | Path | None) -> Dict[str, str]:
    if not hermes_home:
        return {"model": "", "base_url": ""}
    config_path = Path(hermes_home) / "config.yaml"
    if not config_path.exists():
        return {"model": "", "base_url": ""}
    try:
        import yaml

        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        model_cfg = config.get("model", {}) or {}
        return {
            "model": str(model_cfg.get("default") or ""),
            "base_url": str(model_cfg.get("base_url") or ""),
        }
    except Exception as exc:
        logger.debug("Failed to read Hermes model defaults: %s", exc)
        return {"model": "", "base_url": ""}


def extract_json_object(text: str) -> Dict[str, Any] | None:
    text = str(text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        data = json.loads(snippet)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str = "",
        timeout_seconds: int = 45,
    ):
        self.model = str(model or "").strip()
        self.base_url = str(base_url or "").rstrip("/")
        self.api_key = str(api_key or "")
        self.timeout_seconds = int(timeout_seconds)

    @property
    def enabled(self) -> bool:
        return bool(self.model and self.base_url)

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 700,
    ) -> Dict[str, Any] | None:
        if not self.enabled:
            return None
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
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
                body = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                detail = str(exc)
            logger.warning("Local LLM request failed (%s): %s", exc.code, detail[:500])
            return None
        except Exception as exc:
            logger.warning("Local LLM request failed: %s", exc)
            return None

        content = ""
        choices = body.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            raw_content = message.get("content", "")
            if isinstance(raw_content, list):
                content = " ".join(
                    str(block.get("text", ""))
                    for block in raw_content
                    if isinstance(block, dict)
                )
            else:
                content = str(raw_content or "")
        return extract_json_object(content)


def env_or_blank(name: str) -> str:
    return os.environ.get(name, "").strip()
