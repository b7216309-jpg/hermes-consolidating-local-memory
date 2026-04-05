from __future__ import annotations

from bench_compare.dims.common import append_error, finalize_record, started
from bench_compare.systems import empty_record
from bench_compare.utils.facts_corpus import generate_recall_sessions
from bench_compare.utils.fuzzy import greedy_matches
from bench_compare.utils.memory_reader import flattened_context_items, parse_context_sections


def _relevance_items(sections: dict[str, list[str]], fallback_context: str) -> list[str]:
    items: list[str] = []
    for section_name in ("snapshot", "preferences", "facts"):
        items.extend(sections.get(section_name, []))
    if not items:
        items = flattened_context_items(fallback_context)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = " ".join(str(item).split()).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def run(ctx) -> dict[str, dict[str, object]]:
    sessions = generate_recall_sessions()
    injected = [fact.text for session in sessions for fact in session.facts]
    baseline = empty_record("DIM-5", "baseline", injected)
    addon = empty_record("DIM-5", "addon", injected)

    baseline_started = started()
    try:
        ctx.baseline.reset()
        ctx.baseline.seed_sessions(sessions)
        state = ctx.baseline.snapshot_state()
        matches = greedy_matches(injected, state["entries"])
        injected_chars = state["memory_chars"] + state["user_chars"]
        baseline.update(
            {
                "entry_count": len(state["entries"]),
                "injected_chars": injected_chars,
                "injected_token_estimate": int(round(injected_chars / 3.8)),
                "relevance_ratio": len(matches) / len(state["entries"]) if state["entries"] else 0.0,
                "fuzzy_matches": matches,
                "system_prompt_chars": ctx.baseline.prompt_snapshot()["chars"],
            }
        )
    except Exception as exc:
        append_error(baseline, exc)
    finally:
        finalize_record(baseline, baseline_started)

    addon_started = started()
    try:
        ctx.addon.reset()
        ctx.addon.seed_sessions(sessions, create_summaries=True)
        context = ctx.addon.get_context(session_id="bench-prefetch")
        sections = parse_context_sections(context)
        items = _relevance_items(sections, context)
        matches = greedy_matches(injected, items)
        addon.update(
            {
                "injected_chars": len(context),
                "injected_token_estimate": int(round(len(context) / 3.8)),
                "section_breakdown": {section: len(values) for section, values in sections.items()},
                "relevance_ratio": len(matches) / len(items) if items else 0.0,
                "fuzzy_matches": matches,
                "system_prompt_chars": len(context),
            }
        )
    except Exception as exc:
        append_error(addon, exc)
    finally:
        finalize_record(addon, addon_started)

    return {"baseline": baseline, "addon": addon}
