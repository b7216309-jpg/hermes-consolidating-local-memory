from __future__ import annotations

import time


RECALL_PROMPT = (
    "List every fact, preference, and policy you know about me. "
    "Respond ONLY with a JSON array of strings, no preamble."
)

PREFETCH_AUDIT_QUERY = (
    "Give me a provenance summary of every fact, preference, policy, journal note, "
    "and changed assumption you know about me."
)


def started() -> float:
    return time.perf_counter()


def finalize_record(record: dict[str, object], started_at: float) -> None:
    record["duration_ms"] = int(round((time.perf_counter() - started_at) * 1000.0))


def append_error(record: dict[str, object], exc: Exception) -> None:
    record.setdefault("errors", [])
    record["errors"].append(f"{type(exc).__name__}: {exc}")
