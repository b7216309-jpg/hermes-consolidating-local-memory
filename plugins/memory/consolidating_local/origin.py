"""Deterministic turn-origin tracking for Hermes gateway lifecycle hooks.

Hermes invokes memory providers for both human-authored turns and synthetic
agent turns.  The gateway's ``pre_gateway_dispatch`` hook is the authoritative
boundary: it only fires for real inbound messages.  This module carries that
signal into ``pre_llm_call`` and records it for the memory provider's
background worker.
"""

from __future__ import annotations

import hashlib
import sys
import threading
import time
import types
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

_LOCAL_PLATFORMS = {"", "cli", "local", "shell", "terminal"}
_INTERNAL_PLATFORMS = {"background", "cron", "kanban", "subagent", "system"}
_INTERNAL_SESSION_PREFIXES = (
    "background_",
    "compression_",
    "cron_",
    "kanban_",
    "subagent_",
)
_INTERNAL_MESSAGE_PREFIXES = (
    "Review the conversation above and consider saving to memory if appropriate.",
    "Review the conversation above and update the skill library.",
    "[IMPORTANT: Background process ",
    "[ASYNC DELEGATION COMPLETE",
    "[ASYNC DELEGATION BATCH COMPLETE",
    "[Session was just handed off from CLI",
    "[CRITICAL — MESSAGE RECALLED]",
)
_INTERNAL_ORIGINS = {
    "background",
    "background_review",
    "compression",
    "cron",
    "delegation",
    "internal",
    "kanban",
    "recalled_message",
    "system",
}
_USER_ORIGINS = {"human", "inbound", "user", "user_message"}
_MAX_RECORDS = 2048
_RECORD_TTL_SECONDS = 6 * 3600


class _GatewayDispatchMarker:
    """A single-use marker shared by copied ContextVar contexts."""

    def __init__(self) -> None:
        self._consumed = False
        self._lock = threading.Lock()

    def available(self) -> bool:
        with self._lock:
            return not self._consumed

    def consume(self) -> bool:
        with self._lock:
            if self._consumed:
                return False
            self._consumed = True
            return True


_SHARED_STATE_MODULE = "_hermes_consolidating_memory_origin_state"
_shared = sys.modules.get(_SHARED_STATE_MODULE)
if _shared is None:
    _shared = types.ModuleType(_SHARED_STATE_MODULE)
    _shared.gateway_user_dispatch = ContextVar("consolidating_memory_gateway_user_dispatch", default=None)
    _shared.records = {}
    _shared.records_lock = threading.Lock()
    sys.modules[_SHARED_STATE_MODULE] = _shared

_gateway_user_dispatch: ContextVar[_GatewayDispatchMarker | None] = _shared.gateway_user_dispatch


@dataclass(frozen=True)
class _OriginRecord:
    origin: str
    created_at: float


_records: dict[tuple[str, str], _OriginRecord] = _shared.records
_records_lock: threading.Lock = _shared.records_lock


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _digest(value: Any) -> str:
    return hashlib.sha256(_clean_text(value).encode("utf-8")).hexdigest()


def _record_key(session_id: Any, user_message: Any) -> tuple[str, str]:
    return (_clean_text(session_id), _digest(user_message))


def _prune_locked(now: float) -> None:
    expired = [key for key, record in _records.items() if now - record.created_at > _RECORD_TTL_SECONDS]
    for key in expired:
        _records.pop(key, None)
    if len(_records) <= _MAX_RECORDS:
        return
    overflow = len(_records) - _MAX_RECORDS
    for key, _ in sorted(_records.items(), key=lambda item: item[1].created_at)[:overflow]:
        _records.pop(key, None)


def is_gateway_platform(platform: Any) -> bool:
    surface = _clean_text(platform).casefold()
    return surface not in _LOCAL_PLATFORMS and surface not in _INTERNAL_PLATFORMS


def gateway_user_dispatch_active() -> bool:
    marker = _gateway_user_dispatch.get()
    return bool(marker and marker.available())


def is_internal_harness_message(message: Any) -> bool:
    text = str(message or "").lstrip()
    return any(text.startswith(prefix) for prefix in _INTERNAL_MESSAGE_PREFIXES)


def _explicit_origin(kwargs: dict[str, Any]) -> str:
    if kwargs.get("internal") is True or kwargs.get("is_internal") is True:
        return "internal"
    if kwargs.get("delivery_visible") is False or kwargs.get("user_visible") is False:
        return "internal"
    for key in (
        "turn_origin",
        "execution_context",
        "agent_context",
        "write_context",
        "write_origin",
        "message_origin",
    ):
        value = _clean_text(kwargs.get(key)).casefold()
        if value in _USER_ORIGINS:
            return "user"
        if value in _INTERNAL_ORIGINS:
            return "internal"
    return ""


def classify_turn(*, session_id: Any, user_message: Any, platform: Any, kwargs: dict[str, Any] | None = None) -> str:
    """Return ``user`` or ``internal`` for an LLM turn."""

    metadata = dict(kwargs or {})
    explicit = _explicit_origin(metadata)
    if explicit:
        return explicit
    session = _clean_text(session_id).casefold()
    surface = _clean_text(platform).casefold()
    if session.startswith(_INTERNAL_SESSION_PREFIXES) or surface in _INTERNAL_PLATFORMS:
        return "internal"
    if is_gateway_platform(surface):
        marker = _gateway_user_dispatch.get()
        return "user" if marker and marker.consume() else "internal"
    if is_internal_harness_message(user_message):
        return "internal"
    return "user"


def mark_gateway_user_dispatch(event: Any = None, **_: Any) -> None:
    """Mark the current context when Hermes dispatches a real inbound event."""

    marker = None
    if event is not None and not getattr(event, "internal", False):
        marker = _GatewayDispatchMarker()
    _gateway_user_dispatch.set(marker)


def note_llm_turn(session_id: str = "", user_message: str = "", platform: str = "", **kwargs: Any) -> None:
    """Classify an LLM call and publish the result to memory worker threads."""

    origin = classify_turn(
        session_id=session_id,
        user_message=user_message,
        platform=platform,
        kwargs=kwargs,
    )
    now = time.monotonic()
    with _records_lock:
        _records[_record_key(session_id, user_message)] = _OriginRecord(origin, now)
        _prune_locked(now)
    # A background review can run in the same copied executor context.  It must
    # not inherit the real inbound marker from the parent conversation turn.
    _gateway_user_dispatch.set(None)


def recorded_origin(session_id: Any, user_message: Any) -> str:
    key = _record_key(session_id, user_message)
    now = time.monotonic()
    with _records_lock:
        _prune_locked(now)
        record = _records.get(key)
    return record.origin if record else "unknown"


def should_capture_memory(*, session_id: Any, user_message: Any, platform: Any) -> bool:
    """Fail closed on unknown gateway turns while preserving direct CLI use."""

    origin = recorded_origin(session_id, user_message)
    if origin != "unknown":
        return origin == "user"
    if is_gateway_platform(platform):
        return False
    return not is_internal_harness_message(user_message)


def message_was_internal(*, session_id: Any, user_message: Any) -> bool:
    origin = recorded_origin(session_id, user_message)
    return origin == "internal" or is_internal_harness_message(user_message)


def reset_origin_state() -> None:
    """Reset module state for isolated tests."""

    _gateway_user_dispatch.set(None)
    with _records_lock:
        _records.clear()
