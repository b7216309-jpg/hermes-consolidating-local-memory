from __future__ import annotations

import re
from typing import Iterable

from rapidfuzz import fuzz

DEFAULT_THRESHOLD = 0.82


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return float(fuzz.token_set_ratio(normalize_text(left), normalize_text(right))) / 100.0


def greedy_matches(
    expected: Iterable[str],
    observed: Iterable[str],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict[str, object]]:
    expected_list = [str(item) for item in expected if str(item).strip()]
    observed_list = [str(item) for item in observed if str(item).strip()]
    candidates: list[tuple[float, int, int]] = []
    for expected_index, injected in enumerate(expected_list):
        for observed_index, recalled in enumerate(observed_list):
            score = similarity(injected, recalled)
            if score >= threshold:
                candidates.append((score, expected_index, observed_index))
    candidates.sort(reverse=True, key=lambda item: (item[0], -item[1], -item[2]))
    used_expected: set[int] = set()
    used_observed: set[int] = set()
    matches: list[dict[str, object]] = []
    for score, expected_index, observed_index in candidates:
        if expected_index in used_expected or observed_index in used_observed:
            continue
        used_expected.add(expected_index)
        used_observed.add(observed_index)
        matches.append(
            {
                "injected": expected_list[expected_index],
                "recalled": observed_list[observed_index],
                "score": round(score, 4),
            }
        )
    return matches


def precision_recall_f1(
    expected: Iterable[str],
    observed: Iterable[str],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[float, float, float, list[dict[str, object]]]:
    expected_list = [str(item) for item in expected if str(item).strip()]
    observed_list = [str(item) for item in observed if str(item).strip()]
    matches = greedy_matches(expected_list, observed_list, threshold=threshold)
    matched = len(matches)
    precision = matched / len(observed_list) if observed_list else 0.0
    recall = matched / len(expected_list) if expected_list else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1, matches


def observed_match_ratio(
    seeded: Iterable[str],
    observed: Iterable[str],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    observed_list = [str(item) for item in observed if str(item).strip()]
    if not observed_list:
        return 0.0
    matches = greedy_matches(observed_list, seeded, threshold=threshold)
    return len(matches) / len(observed_list)
