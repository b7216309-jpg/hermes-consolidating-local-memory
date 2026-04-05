from __future__ import annotations

import json
from typing import Iterable

from bench_compare.utils.facts_corpus import (
    FactSeed,
    generate_contradiction_pairs,
    generate_scale_facts,
    generate_signal_noise_facts,
)

REAL_DIMENSIONS = ("REAL-1", "REAL-2", "REAL-3", "REAL-4", "REAL-5")

TASK_SUBJECT_KEYS = (
    "user:timezone",
    "user:location:current",
    "user:response_style",
    "environment:shell",
    "environment:editor",
    "project:database",
    "project:test_command",
    "project:deploy_method",
    "workflow:docker_sudo",
    "workflow:manual_edits",
)


def chunked(items: Iterable[FactSeed], size: int) -> list[list[FactSeed]]:
    rows = list(items)
    return [rows[index : index + size] for index in range(0, len(rows), max(1, int(size)))]


def build_seed_prompts(
    facts: list[FactSeed],
    *,
    batch_size: int = 5,
    session_label: str = "memory benchmark",
) -> list[str]:
    prompts: list[str] = []
    for batch_index, batch in enumerate(chunked(facts, batch_size), start=1):
        lines = [
            f"You are participating in a {session_label}.",
            "Store the stable facts below for future sessions using your normal memory facilities.",
            "Treat them as persistent unless a later message explicitly corrects them.",
            'After storing them, respond ONLY with JSON like {"stored": true, "batch": 1, "ack": "ok"}.',
            "Facts:",
        ]
        lines.extend(f"- {fact.text}" for fact in batch)
        prompts.append("\n".join(lines))
    return prompts


def build_contradiction_prompts(*, batch_size: int = 5) -> tuple[list[str], dict[str, str], dict[str, str]]:
    pairs = generate_contradiction_pairs()
    earlier = [pair.earlier for pair in pairs]
    later = [pair.later for pair in pairs]
    prompts = build_seed_prompts(earlier, batch_size=batch_size, session_label="contradiction benchmark")
    prompts.extend(
        build_seed_prompts(
            later,
            batch_size=batch_size,
            session_label="contradiction benchmark correction pass",
        )
    )
    expected_current = {
        str(pair.later.subject_key): str(pair.later.value or "").strip()
        for pair in pairs
        if pair.later.subject_key
    }
    expected_previous = {
        str(pair.earlier.subject_key): str(pair.earlier.value or "").strip()
        for pair in pairs
        if pair.earlier.subject_key
    }
    return prompts, expected_current, expected_previous


def build_current_values_prompt(subject_values: dict[str, str], *, include_history: bool = False) -> str:
    keys_json = json.dumps(list(subject_values.keys()), ensure_ascii=True)
    example = json.dumps({key: value for key, value in list(subject_values.items())[:2]}, ensure_ascii=True)
    history_line = (
        "If a value changed over time, return only the current winning value."
        if not include_history
        else "Return the current value for each key even if you also remember historical alternatives."
    )
    return "\n".join(
        [
            "Respond ONLY with a JSON object.",
            f"Use exactly these keys: {keys_json}.",
            "Return short normalized values when possible, for example `zsh`, `neovim`, `postgresql`, `pytest-x`, `europe-paris`, or `apply-patch`.",
            history_line,
            'If a value is unknown, use "".',
            f"Example format: {example}",
        ]
    )


def build_changed_subjects_prompt(subject_values: dict[str, str]) -> str:
    keys_json = json.dumps(list(subject_values.keys()), ensure_ascii=True)
    return "\n".join(
        [
            "List every subject or assumption that changed over time.",
            "Respond ONLY with a JSON array of strings.",
            f"The canonical subject keys are: {keys_json}.",
            "Prefer subject keys over prose when you know them.",
        ]
    )


def acquisition_facts(count: int) -> list[FactSeed]:
    return generate_scale_facts(count)


def task_grounding_facts() -> tuple[list[FactSeed], dict[str, str]]:
    facts = [fact for fact in generate_scale_facts(50) if fact.subject_key in TASK_SUBJECT_KEYS]
    expected = {
        str(fact.subject_key): str(fact.value or "").strip()
        for fact in facts
        if fact.subject_key and fact.value
    }
    return facts, expected


def signal_noise_facts() -> tuple[list[FactSeed], list[FactSeed]]:
    return generate_signal_noise_facts()
