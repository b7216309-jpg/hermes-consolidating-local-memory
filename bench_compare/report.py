from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DIM_WEIGHTS = {
    "DIM-1": 1.0,
    "DIM-2": 1.0,
    "DIM-3": 1.25,
    "DIM-4": 1.0,
    "DIM-5": 0.75,
    "DIM-6": 0.5,
    "DIM-7": 1.0,
}


def build_summary(dimensions: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    winner_per_dim: dict[str, str] = {}
    total_scores = {"baseline": 0.0, "addon": 0.0}
    total_weight = 0.0
    for dim_id, systems in dimensions.items():
        scores = score_dimension(dim_id, systems["baseline"], systems["addon"])
        winner_per_dim[dim_id] = pick_winner(scores["baseline"], scores["addon"])
        weight = DIM_WEIGHTS.get(dim_id, 1.0)
        total_scores["baseline"] += scores["baseline"] * weight
        total_scores["addon"] += scores["addon"] * weight
        total_weight += weight
    baseline_score = total_scores["baseline"] / total_weight if total_weight else 0.0
    addon_score = total_scores["addon"] / total_weight if total_weight else 0.0
    return {
        "winner_per_dim": winner_per_dim,
        "overall_winner": pick_winner(baseline_score, addon_score),
        "score_baseline": round(baseline_score, 4),
        "score_addon": round(addon_score, 4),
    }


def score_dimension(dim_id: str, baseline: dict[str, Any], addon: dict[str, Any]) -> dict[str, float]:
    if dim_id == "DIM-1":
        return {
            "baseline": float(baseline.get("retention_rate") or 0.0),
            "addon": float(addon.get("retention_rate") or 0.0),
        }
    if dim_id == "DIM-2":
        total = max(len(baseline.get("raw_injected_facts") or []), 1)
        base_kept = float(baseline.get("kept_count") or 0.0) / total
        base_penalty = float(baseline.get("memory_char_limit_errors") or 0.0) / total
        addon_kept = float(addon.get("kept_count") or 0.0) / total
        return {
            "baseline": max(base_kept - (0.25 * base_penalty), 0.0),
            "addon": addon_kept,
        }
    if dim_id == "DIM-3":
        return {"baseline": float(baseline.get("f1") or 0.0), "addon": float(addon.get("f1") or 0.0)}
    if dim_id == "DIM-4":
        return {
            "baseline": float(baseline.get("resolution_rate") or 0.0),
            "addon": float(addon.get("resolution_rate") or 0.0),
        }
    if dim_id == "DIM-5":
        base_chars = max(float(baseline.get("injected_chars") or 0.0), 1.0)
        addon_chars = max(float(addon.get("injected_chars") or 0.0), 1.0)
        min_chars = min(base_chars, addon_chars)
        return {
            "baseline": 0.8 * float(baseline.get("relevance_ratio") or 0.0) + 0.2 * (min_chars / base_chars),
            "addon": 0.8 * float(addon.get("relevance_ratio") or 0.0) + 0.2 * (min_chars / addon_chars),
        }
    if dim_id == "DIM-6":
        return {
            "baseline": 0.0,
            "addon": 1.0 if addon.get("pass_high_gt_low") else 0.0,
        }
    if dim_id == "DIM-7":
        return {
            "baseline": float(baseline.get("signal_noise_ratio") or 0.0),
            "addon": float(addon.get("signal_noise_ratio") or 0.0),
        }
    return {"baseline": 0.0, "addon": 0.0}


def pick_winner(baseline_score: float, addon_score: float, *, epsilon: float = 0.02) -> str:
    if abs(addon_score - baseline_score) <= epsilon:
        return "tie"
    return "addon" if addon_score > baseline_score else "baseline"


def write_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_stdout(dimensions: dict[str, dict[str, dict[str, Any]]], total_llm_calls: int, total_tokens: int) -> None:
    rows = [(dim_id, format_row(dim_id, systems["baseline"], "baseline"), format_row(dim_id, systems["addon"], "addon")) for dim_id, systems in dimensions.items()]
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Hermes Memory Comparative Benchmark")
        table.add_column("Dimension")
        table.add_column("BASELINE")
        table.add_column("ADDON")
        for dim_id, baseline_text, addon_text in rows:
            table.add_row(dim_id, baseline_text, addon_text)
        console = Console()
        console.print(table)
        console.print(f"Total LLM calls: {total_llm_calls}  |  Estimated tokens: ~{total_tokens}")
        return
    except Exception:
        pass

    width_dim = max(len(row[0]) for row in rows) if rows else 10
    width_base = max(len(row[1]) for row in rows) if rows else 10
    width_addon = max(len(row[2]) for row in rows) if rows else 10
    header = f"{'Dimension'.ljust(width_dim)} | {'BASELINE'.ljust(width_base)} | {'ADDON'.ljust(width_addon)}"
    print(header)
    print("-" * len(header))
    for dim_id, baseline_text, addon_text in rows:
        print(f"{dim_id.ljust(width_dim)} | {baseline_text.ljust(width_base)} | {addon_text.ljust(width_addon)}")
    print(f"Total LLM calls: {total_llm_calls}  |  Estimated tokens: ~{total_tokens}")


def format_row(dim_id: str, record: dict[str, Any], system: str) -> str:
    if record.get("errors"):
        return "ERROR"
    if dim_id == "DIM-1":
        chars = "no limit" if system == "addon" else f"{record.get('chars_used', 0)}/{record.get('chars_limit', 0)}ch"
        return f"{record.get('retained_count', 0)}/{len(record.get('raw_injected_facts', []))} {pct(record.get('retention_rate'))} {chars}"
    if dim_id == "DIM-2":
        if system == "baseline":
            return f"{record.get('kept_count', 0)} kept  {record.get('memory_char_limit_errors', 0)} errors"
        p50 = (record.get("salience_distribution") or {}).get("p50", 0.0)
        return f"{record.get('kept_count', 0)} kept  sal p50={p50:.2f}"
    if dim_id == "DIM-3":
        return f"F1={float(record.get('f1') or 0.0):.2f}"
    if dim_id == "DIM-4":
        total = int(record.get("contradictions_resolved", 0) or 0) + int(record.get("contradictions_unresolved", 0) or 0)
        return f"{record.get('contradictions_resolved', 0)}/{total} resolved"
    if dim_id == "DIM-5":
        return f"{record.get('injected_chars', 0)}ch rel={float(record.get('relevance_ratio') or 0.0):.2f}"
    if dim_id == "DIM-6":
        if system == "baseline":
            return "N/A"
        delta = float(record.get("high_minus_low_avg") or 0.0)
        status = "PASS" if record.get("pass_high_gt_low") else "FAIL"
        return f"{status} hi-lo={delta:.2f}"
    if dim_id == "DIM-7":
        return f"SNR={float(record.get('signal_noise_ratio') or 0.0):.2f}"
    return "n/a"


def pct(value: Any) -> str:
    return f"{float(value or 0.0) * 100:.0f}%"
