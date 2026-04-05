from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from bench_compare.report import score_dimension as low_token_score_dimension

PAGE_SIZE = (11.69, 8.27)
BG = "#f6f1e7"
PANEL = "#fffdf9"
TEXT = "#1d2a33"
MUTED = "#4d5b66"
GRID = "#d7d0c3"
BASELINE = "#5b7083"
ADDON = "#d96838"
ACCENT = "#2a9d8f"
WARN = "#c1661b"
SOFT_BASELINE = "#dfe7ee"
SOFT_ADDON = "#f6dfd2"
SOFT_ACCENT = "#d8efe9"
SOFT_WARN = "#f4e3cf"

DIM_ORDER = ("DIM-1", "DIM-2", "DIM-3", "DIM-4", "DIM-5", "DIM-6", "DIM-7")
REAL_DIM_ORDER = ("REAL-1", "REAL-2", "REAL-3", "REAL-4", "REAL-5")

DIM_EXPLANATIONS: dict[str, dict[str, str]] = {
    "DIM-1": {
        "title": "Retention at Scale",
        "what": "Inject 50 heterogeneous facts directly into each system without using the model, end the session, then inspect the stored memory state.",
        "why": "This isolates raw storage capacity and retention discipline. It answers how much durable memory survives before any retrieval prompt or LLM recall step is involved.",
        "readout": "Higher retention is better. Prompt footprint is a secondary efficiency signal because a memory system that stores more while injecting fewer characters leaves more room for the actual task.",
    },
    "DIM-2": {
        "title": "Overflow Behaviour",
        "what": "Push 200 facts into both systems. Baseline is measured against its flat-file ceiling and tool error behavior. Addon is measured from its fact table and salience distribution.",
        "why": "This tests what happens when memory pressure is no longer hypothetical. A useful system should degrade gracefully instead of silently discarding large portions of state.",
        "readout": "Higher kept count is better. For baseline, more character-limit errors indicate the ceiling is being hit repeatedly. For addon, the salience spread shows whether the store is still ranking facts rather than collapsing.",
    },
    "DIM-3": {
        "title": "Cross-Session Recall Quality",
        "what": "Seed 30 facts across three synthetic sessions, start a fresh session, then make one batched LLM recall call per system and score precision, recall, and F1.",
        "why": "This is the user-facing recall test. It measures what the agent can actually surface in a prompt, not just what exists in storage.",
        "readout": "Precision measures how clean the recalled set is. Recall measures how much of the seeded set comes back. F1 balances both and is the main score here.",
    },
    "DIM-4": {
        "title": "Contradiction Handling",
        "what": "Inject ten contradictory fact pairs in sequence and inspect whether the older value is resolved or whether both values remain active.",
        "why": "Durable memory is only useful if it can update beliefs. This test checks whether the system keeps a current winner instead of accumulating incompatible state.",
        "readout": "Higher resolution rate is better. For addon, supersession chains confirm that replacements are not just hidden but explicitly linked as updates.",
    },
    "DIM-5": {
        "title": "Prefetch Context Injection Audit",
        "what": "Seed 30 facts and inspect the exact memory block that would be injected for a new session without making an LLM call.",
        "why": "This bridges storage and recall. It answers what the system actually places in prompt context, how large that block is, and how much of it is relevant to the seeded set.",
        "readout": "High relevance with compact context is ideal. Section breakdown on the addon side shows whether the provider is balancing direct facts with summaries and provenance.",
    },
    "DIM-6": {
        "title": "Long-Term Salience",
        "what": "Inject 100 facts with explicit high, medium, and low importance signals, apply decay, then compare salience distributions by tier.",
        "why": "A long-term memory layer should preserve strong signals better than casual or low-value chatter. This is a direct check of that ranking behavior.",
        "readout": "The pass condition is simple: high-importance facts should keep a higher average salience than low-importance facts after decay.",
    },
    "DIM-7": {
        "title": "Recall Precision Under Noise",
        "what": "Mix 20 useful facts with 20 noisy entries, make one batched recall call per system, then compute the useful-to-total recall ratio.",
        "why": "Retrieval quality is not only about remembering more. It is also about resisting pollution from generic chatter and low-value traces.",
        "readout": "Higher signal-to-noise ratio is better. Useful and noise counts show whether the system is filtering or simply repeating everything it has ever seen.",
    },
}

REAL_DIM_EXPLANATIONS: dict[str, dict[str, str]] = {
    "REAL-1": {
        "title": "Natural Acquisition and Retention",
        "what": "Seed 50 facts through real multi-turn Hermes chats, then ask for everything remembered in one JSON recall call.",
        "why": "This is the closest benchmark to lived use. It measures whether the system can acquire stable facts naturally through conversation instead of through direct storage APIs.",
        "readout": "Precision, recall, and F1 all matter here, but F1 is the headline score because it balances remembering enough with keeping the recalled set clean.",
    },
    "REAL-2": {
        "title": "Correction Handling",
        "what": "Teach ten contradictory subject-value pairs through sequential chats, then ask for the current winning values only.",
        "why": "A practical memory system must not only remember facts. It must update them and suppress stale values after a correction.",
        "readout": "Higher field accuracy is better. Stale-value count is a direct regression signal because it shows old facts leaking into current-state answers.",
    },
    "REAL-3": {
        "title": "Task Grounding Accuracy",
        "what": "Seed operational facts about shell, deploy method, database, editing workflow, and other work-critical settings, then ask for a normalized JSON task profile.",
        "why": "This measures whether memory is useful for real work. The question is not just what was stored, but whether the agent can ground practical task decisions on that memory.",
        "readout": "Field accuracy is the main metric. Near-perfect scores indicate the memory layer is dependable for work planning and execution hints.",
    },
    "REAL-4": {
        "title": "Recall Under Noise",
        "what": "Mix useful durable facts with noisy conversational traces, then ask the model to recall everything it knows in one batched response.",
        "why": "Good memory is selective. This dimension checks whether recall stays focused on durable signal instead of repeating generic or low-value chatter.",
        "readout": "Higher signal-to-noise ratio is better. Useful recall count matters too, because a filter that removes noise but also loses important facts is not actually stronger.",
    },
    "REAL-5": {
        "title": "Changed Subject Recall",
        "what": "Teach a set of subject keys that changed over time, then ask the system to name only the changed subjects as canonical keys.",
        "why": "This isolates temporal awareness. A strong memory system should know not just facts, but which facts were revised or superseded.",
        "readout": "Precision and recall are both critical. A perfect score means the system tracks changed state cleanly and can expose it without extra noise.",
    },
}


def detect_mode(payload: dict[str, Any]) -> str:
    return str((payload.get("meta") or {}).get("benchmark_mode") or "low_token")


def score_real_dimension(dim_id: str, baseline: dict[str, Any], addon: dict[str, Any]) -> dict[str, float]:
    if dim_id in {"REAL-1", "REAL-5"}:
        return {
            "baseline": float(baseline.get("f1") or 0.0),
            "addon": float(addon.get("f1") or 0.0),
        }
    if dim_id == "REAL-2":
        return {
            "baseline": max(float(baseline.get("accuracy") or 0.0) - (0.25 * float(baseline.get("stale_value_rate") or 0.0)), 0.0),
            "addon": max(float(addon.get("accuracy") or 0.0) - (0.25 * float(addon.get("stale_value_rate") or 0.0)), 0.0),
        }
    if dim_id == "REAL-3":
        return {
            "baseline": float(baseline.get("accuracy") or 0.0),
            "addon": float(addon.get("accuracy") or 0.0),
        }
    if dim_id == "REAL-4":
        return {
            "baseline": 0.65 * float(baseline.get("signal_noise_ratio") or 0.0) + 0.35 * float(baseline.get("useful_recall_rate") or 0.0),
            "addon": 0.65 * float(addon.get("signal_noise_ratio") or 0.0) + 0.35 * float(addon.get("useful_recall_rate") or 0.0),
        }
    return {"baseline": 0.0, "addon": 0.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a PDF report from a Hermes benchmark results JSON file.")
    parser.add_argument("--input", required=True, help="Path to bench_results_*.json")
    parser.add_argument("--output", required=True, help="Path to output PDF")
    return parser.parse_args()


def load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: Any) -> str:
    return f"{float(value or 0.0) * 100:.0f}%"


def short_pct(value: Any) -> str:
    return f"{float(value or 0.0) * 100:.1f}%"


def clean_dim_title(dim_id: str) -> str:
    if dim_id in DIM_EXPLANATIONS:
        return f"{dim_id}  {DIM_EXPLANATIONS[dim_id]['title']}"
    return f"{dim_id}  {REAL_DIM_EXPLANATIONS[dim_id]['title']}"


def winner_label(summary: dict[str, Any], dim_id: str) -> str:
    return str(summary.get("winner_per_dim", {}).get(dim_id) or "tie")


def winner_color(label: str) -> str:
    if label == "baseline":
        return BASELINE
    if label == "addon":
        return ADDON
    return ACCENT


def wrap(text: str, width: int) -> str:
    return textwrap.fill(str(text or ""), width=width)


def metric_box(ax, x: float, y: float, title: str, value: str, *, fc: str, ec: str = GRID) -> None:
    ax.text(
        x,
        y,
        f"{title}\n{value}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color=TEXT,
        fontsize=11,
        linespacing=1.4,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": fc, "edgecolor": ec, "linewidth": 1.0},
    )


def add_page_title(fig: plt.Figure, title: str, subtitle: str = "") -> None:
    fig.text(0.06, 0.94, title, fontsize=24, fontweight="bold", color=TEXT)
    if subtitle:
        fig.text(0.06, 0.905, subtitle, fontsize=11, color=MUTED)


def style_chart(ax: plt.Axes, title: str, ylabel: str = "") -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, fontsize=13, color=TEXT, loc="left", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=10)


def overview_scores(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    dimensions = payload.get("dimensions", {})
    order = REAL_DIM_ORDER if detect_mode(payload) == "complete_real" else DIM_ORDER
    for dim_id in order:
        if dim_id not in dimensions:
            continue
        if detect_mode(payload) == "complete_real":
            scores[dim_id] = score_real_dimension(dim_id, dimensions[dim_id]["baseline"], dimensions[dim_id]["addon"])
        else:
            scores[dim_id] = low_token_score_dimension(dim_id, dimensions[dim_id]["baseline"], dimensions[dim_id]["addon"])
    return scores


def render_cover(pdf: PdfPages, payload: dict[str, Any], input_path: Path) -> None:
    meta = payload["meta"]
    summary = payload["summary"]
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(
        fig,
        "Hermes Memory Comparative Benchmark",
        "Baseline built-in memory vs built-in memory + consolidating_local",
    )
    fig.text(
        0.06,
        0.82,
        wrap(
            "This report summarizes a fresh benchmark run using direct storage inspection for structural checks and batched LLM calls only for recall dimensions. The source JSON below is the exact dataset used for every chart in this PDF.",
            92,
        ),
        fontsize=12,
        color=TEXT,
        linespacing=1.5,
    )
    fig.text(
        0.06,
        0.73,
        f"Model: {meta.get('model', 'unknown')}    Plugin: {meta.get('plugin_version', 'unknown')}    "
        f"LLM calls: {meta.get('total_llm_calls', 0)}    Estimated tokens: {meta.get('total_tokens_estimated', 0)}",
        fontsize=11,
        color=MUTED,
    )
    fig.text(0.06, 0.69, f"Results file: {input_path.name}", fontsize=10, color=MUTED)

    ax_text = fig.add_axes([0.06, 0.10, 0.40, 0.52])
    ax_text.axis("off")
    overall = str(summary.get("overall_winner") or "tie").upper()
    ax_text.text(
        0.0,
        1.0,
        f"Overall Winner\n{overall}",
        va="top",
        fontsize=22,
        color=winner_color(str(summary.get("overall_winner") or "tie")),
        fontweight="bold",
        linespacing=1.4,
    )
    metric_box(ax_text, 0.0, 0.70, "Composite Score", f"Baseline {summary.get('score_baseline', 0.0):.3f}", fc=SOFT_BASELINE)
    metric_box(ax_text, 0.42, 0.70, "Composite Score", f"Addon {summary.get('score_addon', 0.0):.3f}", fc=SOFT_ADDON)

    wins = summary.get("winner_per_dim", {})
    baseline_wins = sum(1 for item in wins.values() if item == "baseline")
    addon_wins = sum(1 for item in wins.values() if item == "addon")
    ties = sum(1 for item in wins.values() if item == "tie")
    metric_box(ax_text, 0.0, 0.44, "Dimensions Won", f"Baseline {baseline_wins}", fc=SOFT_BASELINE)
    metric_box(ax_text, 0.42, 0.44, "Dimensions Won", f"Addon {addon_wins}", fc=SOFT_ADDON)
    metric_box(ax_text, 0.0, 0.20, "Ties", f"{ties}", fc=SOFT_ACCENT)

    takeaways = [
        "Addon dominates the structural tests: scale retention, overflow handling, contradiction resolution, salience, and noise filtering.",
        "Baseline wins the cross-session recall F1 test in this run because its prompt snapshot preserves more of the seeded set verbatim.",
        "DIM-5 is now resolved. The addon prefetch block is no longer empty and shows a full mixed context with facts, summaries, preferences, and provenance.",
    ]
    ax_text.text(
        0.42,
        0.20,
        "Key Takeaways\n" + "\n".join(f"- {wrap(line, 45)}" for line in takeaways),
        va="top",
        fontsize=10.5,
        color=TEXT,
        linespacing=1.5,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": PANEL, "edgecolor": GRID, "linewidth": 1.0},
    )

    ax_scores = fig.add_axes([0.54, 0.18, 0.38, 0.52], facecolor=PANEL)
    labels = ["Baseline", "Addon"]
    values = [float(summary.get("score_baseline") or 0.0), float(summary.get("score_addon") or 0.0)]
    colors = [BASELINE, ADDON]
    bars = ax_scores.bar(labels, values, color=colors, width=0.58)
    style_chart(ax_scores, "Weighted Composite Score", "Score (0-1)")
    ax_scores.set_ylim(0.0, 1.05)
    for bar, value in zip(bars, values):
        ax_scores.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.03,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color=TEXT,
            fontweight="bold",
        )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_overview(pdf: PdfPages, payload: dict[str, Any]) -> None:
    meta = payload["meta"]
    summary = payload["summary"]
    dims = payload["dimensions"]
    dim_scores = overview_scores(payload)

    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(
        fig,
        "Benchmark Overview",
        "Dimension-level scores, prompt footprint, and where the LLM budget was actually spent",
    )
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.95, top=0.86, bottom=0.08, hspace=0.35, wspace=0.22)

    ax_dim = fig.add_subplot(gs[:, 0])
    labels = [dim for dim in DIM_ORDER if dim in dim_scores]
    baseline_scores = [dim_scores[dim]["baseline"] for dim in labels]
    addon_scores = [dim_scores[dim]["addon"] for dim in labels]
    x = list(range(len(labels)))
    width = 0.36
    ax_dim.bar([i - width / 2 for i in x], baseline_scores, width=width, color=BASELINE, label="Baseline")
    ax_dim.bar([i + width / 2 for i in x], addon_scores, width=width, color=ADDON, label="Addon")
    style_chart(ax_dim, "Weighted Dimension Scores", "Score (0-1)")
    ax_dim.set_xticks(x, labels)
    ax_dim.set_ylim(0.0, 1.08)
    ax_dim.legend(frameon=False, loc="upper right")

    ax_prompt = fig.add_subplot(gs[0, 1])
    prompt_baseline = [int(dims[dim]["baseline"].get("system_prompt_chars") or 0) for dim in labels]
    prompt_addon = [int(dims[dim]["addon"].get("system_prompt_chars") or 0) for dim in labels]
    ax_prompt.plot(labels, prompt_baseline, marker="o", color=BASELINE, linewidth=2.2, label="Baseline")
    ax_prompt.plot(labels, prompt_addon, marker="o", color=ADDON, linewidth=2.2, label="Addon")
    style_chart(ax_prompt, "Prompt Memory Footprint", "Injected chars")
    ax_prompt.legend(frameon=False, loc="upper left")

    ax_budget = fig.add_subplot(gs[1, 1])
    llm_calls = [
        int(dims[dim]["baseline"].get("llm_calls_made") or 0) + int(dims[dim]["addon"].get("llm_calls_made") or 0)
        for dim in labels
    ]
    tokens = [
        int(dims[dim]["baseline"].get("tokens_estimated") or 0) + int(dims[dim]["addon"].get("tokens_estimated") or 0)
        for dim in labels
    ]
    ax_budget.bar(labels, tokens, color=ACCENT, alpha=0.88, label="Estimated tokens")
    ax_budget2 = ax_budget.twinx()
    ax_budget2.plot(labels, llm_calls, color=WARN, marker="o", linewidth=2.2, label="LLM calls")
    style_chart(ax_budget, "Token Budget by Dimension", "Estimated tokens")
    ax_budget2.set_ylabel("LLM calls", color=MUTED, fontsize=10)
    ax_budget2.tick_params(colors=MUTED, labelsize=10)
    ax_budget2.spines["top"].set_visible(False)
    ax_budget2.spines["left"].set_visible(False)
    ax_budget2.spines["right"].set_color(GRID)

    fig.text(
        0.06,
        0.03,
        f"Total LLM calls: {meta.get('total_llm_calls', 0)}    Total estimated tokens: {meta.get('total_tokens_estimated', 0)}    "
        f"Overall winner: {str(summary.get('overall_winner') or 'tie').upper()}",
        fontsize=10.5,
        color=MUTED,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_methodology(pdf: PdfPages, payload: dict[str, Any]) -> None:
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(
        fig,
        "Methodology",
        "How the benchmark keeps token cost low while still measuring actual recall quality",
    )
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.80])
    ax.axis("off")

    left = (
        "1. Memory seeding is direct\n"
        "Baseline facts are written straight into USER.md and MEMORY.md. Addon facts are inserted via the provider Python API. No chat calls are used for setup.\n\n"
        "2. Structural inspection is direct\n"
        "Retention, overflow, contradictions, prefetch context, and salience are inspected from files, SQLite tables, and provider methods rather than model output.\n\n"
        "3. LLM calls are limited to recall tests\n"
        "Only DIM-3 and DIM-7 call the model, and each dimension batches the full question set into a single prompt per system."
    )
    right = (
        "4. Fresh temp homes are used\n"
        "This report comes from a clean rerun using temporary Hermes homes, so the measurements are isolated from any live user memory.\n\n"
        "5. Fuzzy matching is local\n"
        "Injected and recalled facts are matched locally with a token-set fuzzy score threshold, avoiding extra verification calls.\n\n"
        "6. Interpretation rule\n"
        "Storage-heavy wins matter most for durability. Recall wins matter most for what the user will actually see surfaced in a prompt."
    )

    ax.text(
        0.00,
        0.95,
        wrap(left, 52),
        va="top",
        fontsize=12,
        color=TEXT,
        linespacing=1.6,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": PANEL, "edgecolor": GRID, "linewidth": 1.0},
    )
    ax.text(
        0.52,
        0.95,
        wrap(right, 52),
        va="top",
        fontsize=12,
        color=TEXT,
        linespacing=1.6,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": PANEL, "edgecolor": GRID, "linewidth": 1.0},
    )

    stages = [
        ("Seed", "Direct file writes or provider API sync"),
        ("Consolidate", "Addon provider distills and updates store"),
        ("Inspect", "Files, DB tables, and get_context output"),
        ("Recall", "Single batched model call only where required"),
        ("Score", "Local fuzzy matching and weighted summary"),
    ]
    xs = [0.06, 0.25, 0.46, 0.67, 0.84]
    for (label, body), x in zip(stages, xs):
        ax.text(
            x,
            0.28,
            f"{label}\n{wrap(body, 16)}",
            ha="center",
            va="center",
            fontsize=11,
            color=TEXT,
            linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": SOFT_ACCENT, "edgecolor": GRID, "linewidth": 1.0},
        )
    for left_x, right_x in zip(xs, xs[1:]):
        ax.annotate(
            "",
            xy=(right_x - 0.055, 0.28),
            xytext=(left_x + 0.055, 0.28),
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 1.4},
        )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_dim1(pdf: PdfPages, payload: dict[str, Any]) -> None:
    baseline = payload["dimensions"]["DIM-1"]["baseline"]
    addon = payload["dimensions"]["DIM-1"]["addon"]
    summary = payload["summary"]
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(fig, clean_dim_title("DIM-1"), DIM_EXPLANATIONS["DIM-1"]["what"])
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.95, top=0.84, bottom=0.08, hspace=0.38, wspace=0.24)

    ax_ret = fig.add_subplot(gs[:, 0])
    labels = ["Baseline", "Addon"]
    retained = [int(baseline.get("retained_count") or 0), int(addon.get("retained_count") or 0)]
    bars = ax_ret.bar(labels, retained, color=[BASELINE, ADDON], width=0.58)
    style_chart(ax_ret, "Facts Retained After Seeding 50 Facts", "Retained facts")
    ax_ret.set_ylim(0, 55)
    for bar, value in zip(bars, retained):
        ax_ret.text(bar.get_x() + bar.get_width() / 2.0, value + 1.0, str(value), ha="center", va="bottom", color=TEXT)

    ax_chars = fig.add_subplot(gs[0, 1])
    prompt_chars = [int(baseline.get("system_prompt_chars") or 0), int(addon.get("system_prompt_chars") or 0)]
    bars = ax_chars.bar(labels, prompt_chars, color=[SOFT_BASELINE, SOFT_ADDON], edgecolor=[BASELINE, ADDON], linewidth=1.2)
    style_chart(ax_chars, "Prompt Footprint for the Stored Memory", "Chars")
    for bar, value in zip(bars, prompt_chars):
        ax_chars.text(bar.get_x() + bar.get_width() / 2.0, value + 50, str(value), ha="center", color=TEXT, fontsize=10)

    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis("off")
    winner = winner_label(summary, "DIM-1")
    metric_box(ax_text, 0.00, 0.95, "Winner", winner.upper(), fc=SOFT_ADDON if winner == "addon" else SOFT_BASELINE)
    metric_box(ax_text, 0.00, 0.62, "Baseline", f"{pct(baseline.get('retention_rate'))} retained\n{baseline.get('entry_count', 0)} active entries", fc=SOFT_BASELINE)
    metric_box(ax_text, 0.48, 0.62, "Addon", f"{pct(addon.get('retention_rate'))} retained\n{addon.get('entry_count', 0)} active entries", fc=SOFT_ADDON)
    ax_text.text(
        0.00,
        0.28,
        wrap(DIM_EXPLANATIONS["DIM-1"]["why"], 48) + "\n\n" + wrap(DIM_EXPLANATIONS["DIM-1"]["readout"], 48),
        va="top",
        fontsize=11,
        color=TEXT,
        linespacing=1.55,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": PANEL, "edgecolor": GRID, "linewidth": 1.0},
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_dim2(pdf: PdfPages, payload: dict[str, Any]) -> None:
    baseline = payload["dimensions"]["DIM-2"]["baseline"]
    addon = payload["dimensions"]["DIM-2"]["addon"]
    summary = payload["summary"]
    total = len(baseline.get("raw_injected_facts") or [])
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(fig, clean_dim_title("DIM-2"), DIM_EXPLANATIONS["DIM-2"]["what"])
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.95, top=0.84, bottom=0.08, hspace=0.38, wspace=0.24)

    ax_keep = fig.add_subplot(gs[:, 0])
    kept = [int(baseline.get("kept_count") or 0), int(addon.get("kept_count") or 0)]
    bars = ax_keep.bar(["Baseline", "Addon"], kept, color=[BASELINE, ADDON], width=0.58)
    style_chart(ax_keep, f"Facts Kept After Overflow Pressure ({total} injected)", "Kept facts")
    ax_keep.set_ylim(0, max(total * 1.05, 210))
    ax_keep.axhline(total, color=GRID, linestyle="--", linewidth=1.2)
    ax_keep.text(1.45, total + 3, "Injected total", color=MUTED, fontsize=10)
    for bar, value in zip(bars, kept):
        ax_keep.text(bar.get_x() + bar.get_width() / 2.0, value + 4.0, str(value), ha="center", color=TEXT)

    ax_sal = fig.add_subplot(gs[0, 1])
    sal = addon.get("salience_distribution") or {}
    labels = ["min", "avg", "p50", "p90", "max"]
    values = [float(sal.get("min") or 0.0), float(sal.get("avg") or 0.0), float(sal.get("p50") or 0.0), float(sal.get("p90") or 0.0), float(sal.get("max") or 0.0)]
    ax_sal.bar(labels, values, color=SOFT_ADDON, edgecolor=ADDON, linewidth=1.2)
    style_chart(ax_sal, "Addon Salience Distribution", "Salience")
    ax_sal.set_ylim(0.0, 1.0)

    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis("off")
    winner = winner_label(summary, "DIM-2")
    metric_box(ax_text, 0.00, 0.95, "Winner", winner.upper(), fc=SOFT_ADDON if winner == "addon" else SOFT_BASELINE)
    metric_box(
        ax_text,
        0.00,
        0.62,
        "Baseline ceiling",
        f"{baseline.get('memory_char_limit_errors', 0)} char-limit errors\n{baseline.get('chars_used', 0)}/{baseline.get('chars_limit', 0)} chars used",
        fc=SOFT_WARN,
    )
    metric_box(
        ax_text,
        0.50,
        0.62,
        "Addon spread",
        f"p50 salience {float(sal.get('p50') or 0.0):.2f}\navg salience {float(sal.get('avg') or 0.0):.2f}",
        fc=SOFT_ADDON,
    )
    ax_text.text(
        0.00,
        0.28,
        wrap(DIM_EXPLANATIONS["DIM-2"]["why"], 48) + "\n\n" + wrap(DIM_EXPLANATIONS["DIM-2"]["readout"], 48),
        va="top",
        fontsize=11,
        color=TEXT,
        linespacing=1.55,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": PANEL, "edgecolor": GRID, "linewidth": 1.0},
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_dim3(pdf: PdfPages, payload: dict[str, Any]) -> None:
    baseline = payload["dimensions"]["DIM-3"]["baseline"]
    addon = payload["dimensions"]["DIM-3"]["addon"]
    summary = payload["summary"]
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(fig, clean_dim_title("DIM-3"), DIM_EXPLANATIONS["DIM-3"]["what"])
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.95, top=0.84, bottom=0.08, hspace=0.38, wspace=0.24)

    ax_metrics = fig.add_subplot(gs[:, 0])
    labels = ["Precision", "Recall", "F1"]
    baseline_values = [float(baseline.get("precision") or 0.0), float(baseline.get("recall") or 0.0), float(baseline.get("f1") or 0.0)]
    addon_values = [float(addon.get("precision") or 0.0), float(addon.get("recall") or 0.0), float(addon.get("f1") or 0.0)]
    x = range(len(labels))
    width = 0.36
    ax_metrics.bar([i - width / 2 for i in x], baseline_values, width=width, color=BASELINE, label="Baseline")
    ax_metrics.bar([i + width / 2 for i in x], addon_values, width=width, color=ADDON, label="Addon")
    style_chart(ax_metrics, "Recall Quality from One Batched LLM Call", "Score")
    ax_metrics.set_xticks(list(x), labels)
    ax_metrics.set_ylim(0.0, 1.08)
    ax_metrics.legend(frameon=False, loc="upper left")

    ax_prompt = fig.add_subplot(gs[0, 1])
    chars = [int(baseline.get("system_prompt_memory_block_chars") or 0), int(addon.get("system_prompt_memory_block_chars") or 0)]
    bars = ax_prompt.bar(["Baseline", "Addon"], chars, color=[SOFT_BASELINE, SOFT_ADDON], edgecolor=[BASELINE, ADDON], linewidth=1.2)
    style_chart(ax_prompt, "Prompt Memory Block Size During Recall", "Chars")
    for bar, value in zip(bars, chars):
        ax_prompt.text(bar.get_x() + bar.get_width() / 2.0, value + 40, str(value), ha="center", color=TEXT, fontsize=10)

    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis("off")
    winner = winner_label(summary, "DIM-3")
    metric_box(ax_text, 0.00, 0.95, "Winner", winner.upper(), fc=SOFT_BASELINE if winner == "baseline" else SOFT_ADDON)
    metric_box(ax_text, 0.00, 0.62, "Baseline", f"Precision {short_pct(baseline.get('precision'))}\nRecall {short_pct(baseline.get('recall'))}", fc=SOFT_BASELINE)
    metric_box(ax_text, 0.50, 0.62, "Addon", f"Precision {short_pct(addon.get('precision'))}\nRecall {short_pct(addon.get('recall'))}", fc=SOFT_ADDON)
    ax_text.text(
        0.00,
        0.28,
        wrap(DIM_EXPLANATIONS["DIM-3"]["why"], 48) + "\n\n" + wrap(DIM_EXPLANATIONS["DIM-3"]["readout"], 48),
        va="top",
        fontsize=11,
        color=TEXT,
        linespacing=1.55,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": PANEL, "edgecolor": GRID, "linewidth": 1.0},
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_dim4(pdf: PdfPages, payload: dict[str, Any]) -> None:
    baseline = payload["dimensions"]["DIM-4"]["baseline"]
    addon = payload["dimensions"]["DIM-4"]["addon"]
    summary = payload["summary"]
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(fig, clean_dim_title("DIM-4"), DIM_EXPLANATIONS["DIM-4"]["what"])
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.95, top=0.84, bottom=0.08, hspace=0.38, wspace=0.24)

    ax = fig.add_subplot(gs[:, 0])
    labels = ["Baseline", "Addon"]
    resolved = [int(baseline.get("contradictions_resolved") or 0), int(addon.get("contradictions_resolved") or 0)]
    unresolved = [int(baseline.get("contradictions_unresolved") or 0), int(addon.get("contradictions_unresolved") or 0)]
    ax.bar(labels, resolved, color=ACCENT, label="Resolved")
    ax.bar(labels, unresolved, bottom=resolved, color=SOFT_WARN, edgecolor=WARN, linewidth=1.0, label="Unresolved")
    style_chart(ax, "Contradiction Resolution", "Pairs")
    ax.set_ylim(0, 11)
    ax.legend(frameon=False, loc="upper left")

    ax_text = fig.add_subplot(gs[:, 1])
    ax_text.axis("off")
    winner = winner_label(summary, "DIM-4")
    metric_box(ax_text, 0.00, 0.95, "Winner", winner.upper(), fc=SOFT_ADDON if winner == "addon" else SOFT_BASELINE)
    metric_box(ax_text, 0.00, 0.62, "Baseline", f"Resolved {baseline.get('contradictions_resolved', 0)} / 10", fc=SOFT_BASELINE)
    metric_box(ax_text, 0.50, 0.62, "Addon", f"Resolved {addon.get('contradictions_resolved', 0)} / 10\nSupersession chains {addon.get('supersession_chain_count', 0)}", fc=SOFT_ADDON)
    ax_text.text(
        0.00,
        0.34,
        wrap(DIM_EXPLANATIONS["DIM-4"]["why"], 48) + "\n\n" + wrap(DIM_EXPLANATIONS["DIM-4"]["readout"], 48),
        va="top",
        fontsize=11,
        color=TEXT,
        linespacing=1.55,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": PANEL, "edgecolor": GRID, "linewidth": 1.0},
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_dim5(pdf: PdfPages, payload: dict[str, Any]) -> None:
    baseline = payload["dimensions"]["DIM-5"]["baseline"]
    addon = payload["dimensions"]["DIM-5"]["addon"]
    summary = payload["summary"]
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(fig, clean_dim_title("DIM-5"), DIM_EXPLANATIONS["DIM-5"]["what"])
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.95, top=0.84, bottom=0.08, hspace=0.40, wspace=0.24)

    ax_chars = fig.add_subplot(gs[:, 0])
    labels = ["Baseline", "Addon"]
    chars = [int(baseline.get("injected_chars") or 0), int(addon.get("injected_chars") or 0)]
    bars = ax_chars.bar(labels, chars, color=[BASELINE, ADDON], width=0.58)
    style_chart(ax_chars, "Injected Prefetch Block Size", "Chars")
    for bar, value in zip(bars, chars):
        ax_chars.text(bar.get_x() + bar.get_width() / 2.0, value + 40, str(value), ha="center", color=TEXT, fontsize=10)

    ax_sections = fig.add_subplot(gs[0, 1])
    section_breakdown = addon.get("section_breakdown") or {}
    names = list(section_breakdown.keys())
    values = [int(section_breakdown[name] or 0) for name in names]
    if names:
        ax_sections.barh(names, values, color=SOFT_ADDON, edgecolor=ADDON, linewidth=1.2)
    style_chart(ax_sections, "Addon Context Mix", "Lines")

    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis("off")
    winner = winner_label(summary, "DIM-5")
    fc = SOFT_ACCENT if winner == "tie" else (SOFT_ADDON if winner == "addon" else SOFT_BASELINE)
    metric_box(ax_text, 0.00, 0.95, "Winner", winner.upper(), fc=fc)
    metric_box(ax_text, 0.00, 0.62, "Baseline", f"Relevance {short_pct(baseline.get('relevance_ratio'))}\nEntries {baseline.get('entry_count', 0)}", fc=SOFT_BASELINE)
    metric_box(ax_text, 0.50, 0.62, "Addon", f"Relevance {short_pct(addon.get('relevance_ratio'))}\nSections {', '.join(names) or 'n/a'}", fc=SOFT_ADDON)
    ax_text.text(
        0.00,
        0.28,
        wrap(DIM_EXPLANATIONS["DIM-5"]["why"], 48) + "\n\n" + wrap(DIM_EXPLANATIONS["DIM-5"]["readout"], 48),
        va="top",
        fontsize=11,
        color=TEXT,
        linespacing=1.55,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": PANEL, "edgecolor": GRID, "linewidth": 1.0},
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_dim6(pdf: PdfPages, payload: dict[str, Any]) -> None:
    addon = payload["dimensions"]["DIM-6"]["addon"]
    summary = payload["summary"]
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(fig, clean_dim_title("DIM-6"), DIM_EXPLANATIONS["DIM-6"]["what"])
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.95, top=0.84, bottom=0.08, hspace=0.38, wspace=0.24)

    ax = fig.add_subplot(gs[:, 0])
    salience = addon.get("salience_by_tier") or {}
    labels = ["high", "medium", "low"]
    values = [float((salience.get(label) or {}).get("avg") or 0.0) for label in labels]
    colors = [ACCENT, ADDON, WARN]
    bars = ax.bar(labels, values, color=colors, width=0.58)
    style_chart(ax, "Average Salience After Decay", "Average salience")
    ax.set_ylim(0.0, 0.75)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.02, f"{value:.3f}", ha="center", color=TEXT)

    ax_text = fig.add_subplot(gs[:, 1])
    ax_text.axis("off")
    winner = winner_label(summary, "DIM-6")
    metric_box(ax_text, 0.00, 0.95, "Winner", winner.upper(), fc=SOFT_ADDON)
    metric_box(ax_text, 0.00, 0.62, "Result", "PASS" if addon.get("pass_high_gt_low") else "FAIL", fc=SOFT_ADDON)
    metric_box(ax_text, 0.50, 0.62, "High minus low", f"{float(addon.get('high_minus_low_avg') or 0.0):.3f}", fc=SOFT_ACCENT)
    decay = addon.get("decay_result") or {}
    metric_box(
        ax_text,
        0.00,
        0.40,
        "Decay action",
        f"Facts decayed {int(decay.get('facts_decayed') or 0)}\nFacts deactivated {int(decay.get('facts_deactivated') or 0)}",
        fc=SOFT_WARN,
    )
    ax_text.text(
        0.00,
        0.18,
        wrap(DIM_EXPLANATIONS["DIM-6"]["why"], 48) + "\n\n" + wrap(DIM_EXPLANATIONS["DIM-6"]["readout"], 48),
        va="top",
        fontsize=11,
        color=TEXT,
        linespacing=1.55,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": PANEL, "edgecolor": GRID, "linewidth": 1.0},
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_dim7(pdf: PdfPages, payload: dict[str, Any]) -> None:
    baseline = payload["dimensions"]["DIM-7"]["baseline"]
    addon = payload["dimensions"]["DIM-7"]["addon"]
    summary = payload["summary"]
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(fig, clean_dim_title("DIM-7"), DIM_EXPLANATIONS["DIM-7"]["what"])
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.95, top=0.84, bottom=0.08, hspace=0.38, wspace=0.24)

    ax = fig.add_subplot(gs[:, 0])
    labels = ["Baseline", "Addon"]
    useful = [int(baseline.get("useful_recalled") or 0), int(addon.get("useful_recalled") or 0)]
    noise = [int(baseline.get("noise_recalled") or 0), int(addon.get("noise_recalled") or 0)]
    ax.bar(labels, useful, color=ACCENT, label="Useful recalled")
    ax.bar(labels, noise, bottom=useful, color=SOFT_WARN, edgecolor=WARN, linewidth=1.0, label="Noise recalled")
    style_chart(ax, "Useful vs Noise Recalled", "Items")
    ax.legend(frameon=False, loc="upper right")

    ax_text = fig.add_subplot(gs[:, 1])
    ax_text.axis("off")
    winner = winner_label(summary, "DIM-7")
    metric_box(ax_text, 0.00, 0.95, "Winner", winner.upper(), fc=SOFT_ADDON if winner == "addon" else SOFT_BASELINE)
    metric_box(ax_text, 0.00, 0.62, "Baseline", f"SNR {float(baseline.get('signal_noise_ratio') or 0.0):.2f}\nUseful {useful[0]} / Noise {noise[0]}", fc=SOFT_BASELINE)
    metric_box(ax_text, 0.50, 0.62, "Addon", f"SNR {float(addon.get('signal_noise_ratio') or 0.0):.2f}\nUseful {useful[1]} / Noise {noise[1]}", fc=SOFT_ADDON)
    ax_text.text(
        0.00,
        0.28,
        wrap(DIM_EXPLANATIONS["DIM-7"]["why"], 48) + "\n\n" + wrap(DIM_EXPLANATIONS["DIM-7"]["readout"], 48),
        va="top",
        fontsize=11,
        color=TEXT,
        linespacing=1.55,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": PANEL, "edgecolor": GRID, "linewidth": 1.0},
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_appendix(pdf: PdfPages, payload: dict[str, Any]) -> None:
    dims = payload["dimensions"]
    summary = payload["summary"]
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(fig, "Dimension Summary Table", "Compact scorecard for the benchmark run")
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.80])
    ax.axis("off")

    rows = []
    for dim_id in DIM_ORDER:
        if dim_id not in dims:
            continue
        base = dims[dim_id]["baseline"]
        addon = dims[dim_id]["addon"]
        winner = winner_label(summary, dim_id)
        if dim_id == "DIM-1":
            base_text = f"{base.get('retained_count', 0)}/50 retained"
            addon_text = f"{addon.get('retained_count', 0)}/50 retained"
        elif dim_id == "DIM-2":
            base_text = f"{base.get('kept_count', 0)} kept, {base.get('memory_char_limit_errors', 0)} errors"
            addon_text = f"{addon.get('kept_count', 0)} kept, p50 salience {float((addon.get('salience_distribution') or {}).get('p50') or 0.0):.2f}"
        elif dim_id == "DIM-3":
            base_text = f"F1 {float(base.get('f1') or 0.0):.2f}"
            addon_text = f"F1 {float(addon.get('f1') or 0.0):.2f}"
        elif dim_id == "DIM-4":
            base_text = f"{base.get('contradictions_resolved', 0)}/10 resolved"
            addon_text = f"{addon.get('contradictions_resolved', 0)}/10 resolved"
        elif dim_id == "DIM-5":
            base_text = f"{base.get('injected_chars', 0)} chars, rel {float(base.get('relevance_ratio') or 0.0):.2f}"
            addon_text = f"{addon.get('injected_chars', 0)} chars, rel {float(addon.get('relevance_ratio') or 0.0):.2f}"
        elif dim_id == "DIM-6":
            base_text = "N/A"
            addon_text = f"hi-low delta {float(addon.get('high_minus_low_avg') or 0.0):.2f}"
        else:
            base_text = f"SNR {float(base.get('signal_noise_ratio') or 0.0):.2f}"
            addon_text = f"SNR {float(addon.get('signal_noise_ratio') or 0.0):.2f}"
        rows.append([dim_id, DIM_EXPLANATIONS[dim_id]["title"], base_text, addon_text, winner.upper()])

    table = ax.table(
        cellText=rows,
        colLabels=["Dimension", "Test", "Baseline", "Addon", "Winner"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor("#e8ddca")
            cell.set_text_props(color=TEXT, weight="bold")
        else:
            cell.set_facecolor(PANEL)
            cell.set_text_props(color=TEXT)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_real_cover(pdf: PdfPages, payload: dict[str, Any], input_path: Path) -> None:
    meta = payload["meta"]
    summary = payload["summary"]
    addon_cfg = meta.get("addon_config") or {}
    retrieval_backend = str(addon_cfg.get("retrieval_backend") or "unknown")
    extractor_backend = str(addon_cfg.get("extractor_backend") or "unknown")
    llm_base = str(addon_cfg.get("llm_base_url") or "")
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(
        fig,
        "Hermes Complete Real Benchmark",
        "Real multi-turn agent conversations, real recall prompts, and LLM-backed addon extraction",
    )
    fig.text(
        0.06,
        0.82,
        wrap(
            "This report summarizes the full-fidelity benchmark run. Facts were seeded through real Hermes conversations rather than direct storage writes, then scored from structured JSON answers. The addon used LLM extraction during memory acquisition and FTS retrieval during recall because no live embeddings endpoint was available on this installation.",
            96,
        ),
        fontsize=12,
        color=TEXT,
        linespacing=1.5,
    )
    fig.text(
        0.06,
        0.73,
        f"Model: {meta.get('model', 'unknown')}    Extractor: {extractor_backend}    Retrieval: {retrieval_backend}    "
        f"LLM calls: {meta.get('total_llm_calls', 0)}    Estimated tokens: {meta.get('total_tokens_estimated', 0)}",
        fontsize=11,
        color=MUTED,
    )
    fig.text(0.06, 0.69, f"Results file: {input_path.name}", fontsize=10, color=MUTED)
    if llm_base:
        fig.text(0.06, 0.66, f"Addon extraction backend: {llm_base}", fontsize=10, color=MUTED)

    ax_text = fig.add_axes([0.06, 0.10, 0.40, 0.52])
    ax_text.axis("off")
    overall = str(summary.get("overall_winner") or "tie").upper()
    ax_text.text(
        0.0,
        1.0,
        f"Overall Winner\n{overall}",
        va="top",
        fontsize=22,
        color=winner_color(str(summary.get("overall_winner") or "tie")),
        fontweight="bold",
        linespacing=1.4,
    )
    metric_box(ax_text, 0.0, 0.70, "Composite Score", f"Baseline {summary.get('score_baseline', 0.0):.3f}", fc=SOFT_BASELINE)
    metric_box(ax_text, 0.42, 0.70, "Composite Score", f"Addon {summary.get('score_addon', 0.0):.3f}", fc=SOFT_ADDON)

    wins = summary.get("winner_per_dim", {})
    baseline_wins = sum(1 for item in wins.values() if item == "baseline")
    addon_wins = sum(1 for item in wins.values() if item == "addon")
    ties = sum(1 for item in wins.values() if item == "tie")
    metric_box(ax_text, 0.0, 0.44, "Dimensions Won", f"Baseline {baseline_wins}", fc=SOFT_BASELINE)
    metric_box(ax_text, 0.42, 0.44, "Dimensions Won", f"Addon {addon_wins}", fc=SOFT_ADDON)
    metric_box(ax_text, 0.0, 0.20, "Ties", f"{ties}", fc=SOFT_ACCENT)

    takeaways = [
        "Addon clearly wins natural retention under real chat seeding, which is the most realistic memory-acquisition test in the suite.",
        "Baseline stays very strong at exact current-value correction, matching or beating addon on contradiction-heavy JSON tasking.",
        "Both systems ground operational task facts well, but addon is much cleaner under noisy recall pressure.",
    ]
    ax_text.text(
        0.42,
        0.20,
        "Key Takeaways\n" + "\n".join(f"- {wrap(line, 45)}" for line in takeaways),
        va="top",
        fontsize=10.5,
        color=TEXT,
        linespacing=1.5,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": PANEL, "edgecolor": GRID, "linewidth": 1.0},
    )

    ax_scores = fig.add_axes([0.54, 0.18, 0.38, 0.52], facecolor=PANEL)
    labels = ["Baseline", "Addon"]
    values = [float(summary.get("score_baseline") or 0.0), float(summary.get("score_addon") or 0.0)]
    colors = [BASELINE, ADDON]
    bars = ax_scores.bar(labels, values, color=colors, width=0.58)
    style_chart(ax_scores, "Weighted Composite Score", "Score (0-1)")
    ax_scores.set_ylim(0.0, 1.05)
    for bar, value in zip(bars, values):
        ax_scores.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.03,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color=TEXT,
            fontweight="bold",
        )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_real_overview(pdf: PdfPages, payload: dict[str, Any]) -> None:
    meta = payload["meta"]
    summary = payload["summary"]
    dims = payload["dimensions"]
    dim_scores = overview_scores(payload)

    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(
        fig,
        "Complete Real Overview",
        "Dimension scores, prompt memory footprint, and actual model budget for the end-to-end run",
    )
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.95, top=0.86, bottom=0.08, hspace=0.35, wspace=0.22)

    ax_dim = fig.add_subplot(gs[:, 0])
    labels = [dim for dim in REAL_DIM_ORDER if dim in dim_scores]
    baseline_scores = [dim_scores[dim]["baseline"] for dim in labels]
    addon_scores = [dim_scores[dim]["addon"] for dim in labels]
    x = list(range(len(labels)))
    width = 0.36
    ax_dim.bar([i - width / 2 for i in x], baseline_scores, width=width, color=BASELINE, label="Baseline")
    ax_dim.bar([i + width / 2 for i in x], addon_scores, width=width, color=ADDON, label="Addon")
    style_chart(ax_dim, "Weighted Dimension Scores", "Score (0-1)")
    ax_dim.set_xticks(x, labels)
    ax_dim.set_ylim(0.0, 1.08)
    ax_dim.legend(frameon=False, loc="upper right")

    ax_prompt = fig.add_subplot(gs[0, 1])
    prompt_baseline = [int(dims[dim]["baseline"].get("system_prompt_chars") or 0) for dim in labels]
    prompt_addon = [int(dims[dim]["addon"].get("system_prompt_chars") or 0) for dim in labels]
    ax_prompt.plot(labels, prompt_baseline, marker="o", color=BASELINE, linewidth=2.2, label="Baseline")
    ax_prompt.plot(labels, prompt_addon, marker="o", color=ADDON, linewidth=2.2, label="Addon")
    style_chart(ax_prompt, "Prompt Memory Footprint", "Injected chars")
    ax_prompt.legend(frameon=False, loc="upper left")

    ax_budget = fig.add_subplot(gs[1, 1])
    llm_calls = [
        int(dims[dim]["baseline"].get("llm_calls_made") or 0) + int(dims[dim]["addon"].get("llm_calls_made") or 0)
        for dim in labels
    ]
    tokens = [
        int(dims[dim]["baseline"].get("tokens_estimated") or 0) + int(dims[dim]["addon"].get("tokens_estimated") or 0)
        for dim in labels
    ]
    ax_budget.bar(labels, tokens, color=ACCENT, alpha=0.88, label="Estimated tokens")
    ax_budget2 = ax_budget.twinx()
    ax_budget2.plot(labels, llm_calls, color=WARN, marker="o", linewidth=2.2, label="LLM calls")
    style_chart(ax_budget, "Budget by Dimension", "Estimated tokens")
    ax_budget2.set_ylabel("LLM calls", color=MUTED, fontsize=10)
    ax_budget2.tick_params(colors=MUTED, labelsize=10)
    ax_budget2.spines["top"].set_visible(False)
    ax_budget2.spines["left"].set_visible(False)
    ax_budget2.spines["right"].set_color(GRID)

    fig.text(
        0.06,
        0.03,
        f"Total LLM calls: {meta.get('total_llm_calls', 0)}    Total estimated tokens: {meta.get('total_tokens_estimated', 0)}    "
        f"Overall winner: {str(summary.get('overall_winner') or 'tie').upper()}",
        fontsize=10.5,
        color=MUTED,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_real_methodology(pdf: PdfPages, payload: dict[str, Any]) -> None:
    meta = payload["meta"]
    addon_cfg = meta.get("addon_config") or {}
    llm_base = str(addon_cfg.get("llm_base_url") or "")
    retrieval_backend = str(addon_cfg.get("retrieval_backend") or "unknown")
    extractor_backend = str(addon_cfg.get("extractor_backend") or "unknown")
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(
        fig,
        "Methodology",
        "How the complete real benchmark differs from the low-token structural suite",
    )
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.80])
    ax.axis("off")

    left = (
        "1. Seeding happens through real chats\n"
        "Every test session teaches memory through multi-turn AIAgent conversations rather than direct file writes or provider API insertion.\n\n"
        "2. Evaluation also happens through real chats\n"
        "Recall and current-state questions are asked as normal agent prompts, with responses parsed locally from JSON.\n\n"
        "3. Addon extraction is LLM-backed\n"
        f"This run used extractor_backend={extractor_backend} and fed the addon through the active backend route: {llm_base or 'unknown'}."
    )
    right = (
        "4. Retrieval stayed grounded in the installed runtime\n"
        f"Retrieval backend for this run was {retrieval_backend}. Hybrid embeddings were not used because the installation did not expose a live embedding endpoint.\n\n"
        "5. Temp Hermes homes kept the run isolated\n"
        "The benchmark used fresh temporary baseline and addon homes, so no live user memory was mutated.\n\n"
        "6. Scoring remained local and reproducible\n"
        "We still scored JSON outputs locally with exact or fuzzy matching instead of adding a separate judge model."
    )

    ax.text(
        0.00,
        0.95,
        wrap(left, 52),
        va="top",
        fontsize=12,
        color=TEXT,
        linespacing=1.6,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": PANEL, "edgecolor": GRID, "linewidth": 1.0},
    )
    ax.text(
        0.52,
        0.95,
        wrap(right, 52),
        va="top",
        fontsize=12,
        color=TEXT,
        linespacing=1.6,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": PANEL, "edgecolor": GRID, "linewidth": 1.0},
    )

    stages = [
        ("Chat Seed", "Teach facts through real turns"),
        ("Addon Distill", "sync_turn and session-end extraction"),
        ("Fresh Prompt", "Ask recall or task question"),
        ("JSON Parse", "Parse structured model output"),
        ("Local Score", "Compute accuracy, F1, and SNR"),
    ]
    xs = [0.06, 0.25, 0.46, 0.67, 0.84]
    for (label, body), x in zip(stages, xs):
        ax.text(
            x,
            0.28,
            f"{label}\n{wrap(body, 16)}",
            ha="center",
            va="center",
            fontsize=11,
            color=TEXT,
            linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": SOFT_ACCENT, "edgecolor": GRID, "linewidth": 1.0},
        )
    for left_x, right_x in zip(xs, xs[1:]):
        ax.annotate(
            "",
            xy=(right_x - 0.055, 0.28),
            xytext=(left_x + 0.055, 0.28),
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 1.4},
        )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _real_dim_text(ax: plt.Axes, dim_id: str, winner: str, baseline_text: str, addon_text: str) -> None:
    ax.axis("off")
    fc = SOFT_ACCENT if winner == "tie" else (SOFT_ADDON if winner == "addon" else SOFT_BASELINE)
    metric_box(ax, 0.00, 0.95, "Winner", winner.upper(), fc=fc)
    metric_box(ax, 0.00, 0.62, "Baseline", baseline_text, fc=SOFT_BASELINE)
    metric_box(ax, 0.50, 0.62, "Addon", addon_text, fc=SOFT_ADDON)
    ax.text(
        0.00,
        0.28,
        wrap(REAL_DIM_EXPLANATIONS[dim_id]["why"], 48) + "\n\n" + wrap(REAL_DIM_EXPLANATIONS[dim_id]["readout"], 48),
        va="top",
        fontsize=11,
        color=TEXT,
        linespacing=1.55,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": PANEL, "edgecolor": GRID, "linewidth": 1.0},
    )


def render_real_1(pdf: PdfPages, payload: dict[str, Any]) -> None:
    baseline = payload["dimensions"]["REAL-1"]["baseline"]
    addon = payload["dimensions"]["REAL-1"]["addon"]
    summary = payload["summary"]
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(fig, clean_dim_title("REAL-1"), REAL_DIM_EXPLANATIONS["REAL-1"]["what"])
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.95, top=0.84, bottom=0.08, hspace=0.38, wspace=0.24)

    ax_metrics = fig.add_subplot(gs[:, 0])
    labels = ["Precision", "Recall", "F1"]
    baseline_values = [float(baseline.get("precision") or 0.0), float(baseline.get("recall") or 0.0), float(baseline.get("f1") or 0.0)]
    addon_values = [float(addon.get("precision") or 0.0), float(addon.get("recall") or 0.0), float(addon.get("f1") or 0.0)]
    x = range(len(labels))
    width = 0.36
    ax_metrics.bar([i - width / 2 for i in x], baseline_values, width=width, color=BASELINE, label="Baseline")
    ax_metrics.bar([i + width / 2 for i in x], addon_values, width=width, color=ADDON, label="Addon")
    style_chart(ax_metrics, "Recall Quality After Real Chat Seeding", "Score")
    ax_metrics.set_xticks(list(x), labels)
    ax_metrics.set_ylim(0.0, 1.08)
    ax_metrics.legend(frameon=False, loc="upper left")

    ax_prompt = fig.add_subplot(gs[0, 1])
    chars = [int(baseline.get("system_prompt_chars") or 0), int(addon.get("system_prompt_chars") or 0)]
    bars = ax_prompt.bar(["Baseline", "Addon"], chars, color=[SOFT_BASELINE, SOFT_ADDON], edgecolor=[BASELINE, ADDON], linewidth=1.2)
    style_chart(ax_prompt, "Prompt Footprint During Recall", "Chars")
    for bar, value in zip(bars, chars):
        ax_prompt.text(bar.get_x() + bar.get_width() / 2.0, value + 40, str(value), ha="center", color=TEXT, fontsize=10)

    ax_text = fig.add_subplot(gs[1, 1])
    _real_dim_text(
        ax_text,
        "REAL-1",
        winner_label(summary, "REAL-1"),
        f"Precision {short_pct(baseline.get('precision'))}\nRecall {short_pct(baseline.get('recall'))}\nF1 {float(baseline.get('f1') or 0.0):.2f}",
        f"Precision {short_pct(addon.get('precision'))}\nRecall {short_pct(addon.get('recall'))}\nF1 {float(addon.get('f1') or 0.0):.2f}",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_real_2(pdf: PdfPages, payload: dict[str, Any]) -> None:
    baseline = payload["dimensions"]["REAL-2"]["baseline"]
    addon = payload["dimensions"]["REAL-2"]["addon"]
    summary = payload["summary"]
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(fig, clean_dim_title("REAL-2"), REAL_DIM_EXPLANATIONS["REAL-2"]["what"])
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.95, top=0.84, bottom=0.08, hspace=0.38, wspace=0.24)

    ax_acc = fig.add_subplot(gs[:, 0])
    labels = ["Baseline", "Addon"]
    accuracy = [float(baseline.get("accuracy") or 0.0), float(addon.get("accuracy") or 0.0)]
    stale = [int(baseline.get("stale_value_count") or 0), int(addon.get("stale_value_count") or 0)]
    bars = ax_acc.bar(labels, accuracy, color=[BASELINE, ADDON], width=0.58)
    style_chart(ax_acc, "Current-Value Accuracy", "Accuracy")
    ax_acc.set_ylim(0.0, 1.08)
    for bar, value, stale_count in zip(bars, accuracy, stale):
        ax_acc.text(bar.get_x() + bar.get_width() / 2.0, value + 0.03, f"{value:.2f}\nstale={stale_count}", ha="center", color=TEXT, fontsize=10)

    ax_fields = fig.add_subplot(gs[0, 1])
    correct = [int(baseline.get("correct_count") or 0), int(addon.get("correct_count") or 0)]
    total = max(int(baseline.get("total_fields") or 0), int(addon.get("total_fields") or 0), 1)
    ax_fields.bar(labels, correct, color=[SOFT_BASELINE, SOFT_ADDON], edgecolor=[BASELINE, ADDON], linewidth=1.2)
    style_chart(ax_fields, "Correct Fields", "Fields")
    ax_fields.set_ylim(0, total + 1)

    ax_text = fig.add_subplot(gs[1, 1])
    _real_dim_text(
        ax_text,
        "REAL-2",
        winner_label(summary, "REAL-2"),
        f"Accuracy {short_pct(baseline.get('accuracy'))}\nCorrect {baseline.get('correct_count', 0)}/{baseline.get('total_fields', 0)}\nStale {baseline.get('stale_value_count', 0)}",
        f"Accuracy {short_pct(addon.get('accuracy'))}\nCorrect {addon.get('correct_count', 0)}/{addon.get('total_fields', 0)}\nStale {addon.get('stale_value_count', 0)}",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_real_3(pdf: PdfPages, payload: dict[str, Any]) -> None:
    baseline = payload["dimensions"]["REAL-3"]["baseline"]
    addon = payload["dimensions"]["REAL-3"]["addon"]
    summary = payload["summary"]
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(fig, clean_dim_title("REAL-3"), REAL_DIM_EXPLANATIONS["REAL-3"]["what"])
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.95, top=0.84, bottom=0.08, hspace=0.38, wspace=0.24)

    ax_acc = fig.add_subplot(gs[:, 0])
    labels = ["Baseline", "Addon"]
    accuracy = [float(baseline.get("accuracy") or 0.0), float(addon.get("accuracy") or 0.0)]
    bars = ax_acc.bar(labels, accuracy, color=[BASELINE, ADDON], width=0.58)
    style_chart(ax_acc, "Task Grounding Accuracy", "Accuracy")
    ax_acc.set_ylim(0.0, 1.08)
    for bar, value in zip(bars, accuracy):
        ax_acc.text(bar.get_x() + bar.get_width() / 2.0, value + 0.03, f"{value:.2f}", ha="center", color=TEXT, fontsize=10)

    ax_prompt = fig.add_subplot(gs[0, 1])
    chars = [int(baseline.get("system_prompt_chars") or 0), int(addon.get("system_prompt_chars") or 0)]
    ax_prompt.bar(labels, chars, color=[SOFT_BASELINE, SOFT_ADDON], edgecolor=[BASELINE, ADDON], linewidth=1.2)
    style_chart(ax_prompt, "Prompt Footprint for Task Query", "Chars")

    ax_text = fig.add_subplot(gs[1, 1])
    _real_dim_text(
        ax_text,
        "REAL-3",
        winner_label(summary, "REAL-3"),
        f"Accuracy {short_pct(baseline.get('accuracy'))}\nCorrect {baseline.get('correct_count', 0)}/{baseline.get('total_fields', 0)}",
        f"Accuracy {short_pct(addon.get('accuracy'))}\nCorrect {addon.get('correct_count', 0)}/{addon.get('total_fields', 0)}",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_real_4(pdf: PdfPages, payload: dict[str, Any]) -> None:
    baseline = payload["dimensions"]["REAL-4"]["baseline"]
    addon = payload["dimensions"]["REAL-4"]["addon"]
    summary = payload["summary"]
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(fig, clean_dim_title("REAL-4"), REAL_DIM_EXPLANATIONS["REAL-4"]["what"])
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.95, top=0.84, bottom=0.08, hspace=0.38, wspace=0.24)

    ax = fig.add_subplot(gs[:, 0])
    labels = ["Baseline", "Addon"]
    useful = [int(baseline.get("useful_recalled") or 0), int(addon.get("useful_recalled") or 0)]
    total_recalled = [len(baseline.get("raw_recalled_items") or []), len(addon.get("raw_recalled_items") or [])]
    noise = [max(total_recalled[0] - useful[0], 0), max(total_recalled[1] - useful[1], 0)]
    ax.bar(labels, useful, color=ACCENT, label="Useful recalled")
    ax.bar(labels, noise, bottom=useful, color=SOFT_WARN, edgecolor=WARN, linewidth=1.0, label="Other recalled")
    style_chart(ax, "Useful vs Other Recall", "Items")
    ax.legend(frameon=False, loc="upper right")

    ax_text = fig.add_subplot(gs[:, 1])
    _real_dim_text(
        ax_text,
        "REAL-4",
        winner_label(summary, "REAL-4"),
        f"SNR {float(baseline.get('signal_noise_ratio') or 0.0):.2f}\nUseful {baseline.get('useful_recalled', 0)}\nRecall rate {short_pct(baseline.get('useful_recall_rate'))}",
        f"SNR {float(addon.get('signal_noise_ratio') or 0.0):.2f}\nUseful {addon.get('useful_recalled', 0)}\nRecall rate {short_pct(addon.get('useful_recall_rate'))}",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_real_5(pdf: PdfPages, payload: dict[str, Any]) -> None:
    baseline = payload["dimensions"]["REAL-5"]["baseline"]
    addon = payload["dimensions"]["REAL-5"]["addon"]
    summary = payload["summary"]
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(fig, clean_dim_title("REAL-5"), REAL_DIM_EXPLANATIONS["REAL-5"]["what"])
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.95, top=0.84, bottom=0.08, hspace=0.38, wspace=0.24)

    ax = fig.add_subplot(gs[:, 0])
    labels = ["Precision", "Recall", "F1"]
    baseline_values = [float(baseline.get("precision") or 0.0), float(baseline.get("recall") or 0.0), float(baseline.get("f1") or 0.0)]
    addon_values = [float(addon.get("precision") or 0.0), float(addon.get("recall") or 0.0), float(addon.get("f1") or 0.0)]
    x = range(len(labels))
    width = 0.36
    ax.bar([i - width / 2 for i in x], baseline_values, width=width, color=BASELINE, label="Baseline")
    ax.bar([i + width / 2 for i in x], addon_values, width=width, color=ADDON, label="Addon")
    style_chart(ax, "Changed-Subject Recall", "Score")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0.0, 1.08)
    ax.legend(frameon=False, loc="upper left")

    ax_text = fig.add_subplot(gs[:, 1])
    _real_dim_text(
        ax_text,
        "REAL-5",
        winner_label(summary, "REAL-5"),
        f"Precision {short_pct(baseline.get('precision'))}\nRecall {short_pct(baseline.get('recall'))}\nF1 {float(baseline.get('f1') or 0.0):.2f}",
        f"Precision {short_pct(addon.get('precision'))}\nRecall {short_pct(addon.get('recall'))}\nF1 {float(addon.get('f1') or 0.0):.2f}",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_real_appendix(pdf: PdfPages, payload: dict[str, Any]) -> None:
    dims = payload["dimensions"]
    summary = payload["summary"]
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=BG)
    add_page_title(fig, "Dimension Summary Table", "Compact scorecard for the complete real benchmark run")
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.80])
    ax.axis("off")

    rows = []
    for dim_id in REAL_DIM_ORDER:
        if dim_id not in dims:
            continue
        base = dims[dim_id]["baseline"]
        addon = dims[dim_id]["addon"]
        winner = winner_label(summary, dim_id)
        if dim_id in {"REAL-1", "REAL-5"}:
            base_text = f"F1 {float(base.get('f1') or 0.0):.2f}, R {float(base.get('recall') or 0.0):.2f}"
            addon_text = f"F1 {float(addon.get('f1') or 0.0):.2f}, R {float(addon.get('recall') or 0.0):.2f}"
        elif dim_id == "REAL-2":
            base_text = f"acc {float(base.get('accuracy') or 0.0):.2f}, stale {int(base.get('stale_value_count') or 0)}"
            addon_text = f"acc {float(addon.get('accuracy') or 0.0):.2f}, stale {int(addon.get('stale_value_count') or 0)}"
        elif dim_id == "REAL-3":
            base_text = f"acc {float(base.get('accuracy') or 0.0):.2f}, {base.get('correct_count', 0)}/{base.get('total_fields', 0)}"
            addon_text = f"acc {float(addon.get('accuracy') or 0.0):.2f}, {addon.get('correct_count', 0)}/{addon.get('total_fields', 0)}"
        else:
            base_text = f"SNR {float(base.get('signal_noise_ratio') or 0.0):.2f}, useful {int(base.get('useful_recalled') or 0)}"
            addon_text = f"SNR {float(addon.get('signal_noise_ratio') or 0.0):.2f}, useful {int(addon.get('useful_recalled') or 0)}"
        rows.append([dim_id, REAL_DIM_EXPLANATIONS[dim_id]["title"], base_text, addon_text, winner.upper()])

    table = ax.table(
        cellText=rows,
        colLabels=["Dimension", "Test", "Baseline", "Addon", "Winner"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor("#e8ddca")
            cell.set_text_props(color=TEXT, weight="bold")
        else:
            cell.set_facecolor(PANEL)
            cell.set_text_props(color=TEXT)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_report(input_path: Path, output_path: Path) -> None:
    payload = load_results(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "Hermes Memory Comparative Benchmark Report"
        metadata["Author"] = "Codex"
        metadata["Subject"] = "Baseline vs consolidating_local benchmark"
        metadata["Keywords"] = "Hermes memory benchmark PDF"
        if detect_mode(payload) == "complete_real":
            render_real_cover(pdf, payload, input_path)
            render_real_overview(pdf, payload)
            render_real_methodology(pdf, payload)
            render_real_1(pdf, payload)
            render_real_2(pdf, payload)
            render_real_3(pdf, payload)
            render_real_4(pdf, payload)
            render_real_5(pdf, payload)
            render_real_appendix(pdf, payload)
        else:
            render_cover(pdf, payload, input_path)
            render_overview(pdf, payload)
            render_methodology(pdf, payload)
            render_dim1(pdf, payload)
            render_dim2(pdf, payload)
            render_dim3(pdf, payload)
            render_dim4(pdf, payload)
            render_dim5(pdf, payload)
            render_dim6(pdf, payload)
            render_dim7(pdf, payload)
            render_appendix(pdf, payload)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    render_report(input_path, output_path)
    print(f"PDF report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
