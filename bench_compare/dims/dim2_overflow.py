from __future__ import annotations

from bench_compare.dims.common import PREFETCH_AUDIT_QUERY, append_error, finalize_record, started
from bench_compare.systems import empty_record
from bench_compare.utils.facts_corpus import generate_overflow_facts
from bench_compare.utils.fuzzy import greedy_matches
from bench_compare.utils.memory_reader import salience_stats


def run(ctx) -> dict[str, dict[str, object]]:
    facts = generate_overflow_facts(ctx.overflow_facts)
    injected = [fact.text for fact in facts]
    baseline = empty_record("DIM-2", "baseline", injected)
    addon = empty_record("DIM-2", "addon", injected)

    baseline_started = started()
    try:
        ctx.baseline.reset()
        seeded = ctx.baseline.seed_via_memory_tool(facts)
        state = ctx.baseline.snapshot_state()
        error_count = sum(
            1
            for row in seeded["parsed_outputs"]
            if not bool(row.get("success"))
            and "would exceed the limit" in str(row.get("error") or "").lower()
        )
        matches = greedy_matches(injected, state["entries"])
        baseline.update(
            {
                "kept_count": len(seeded["kept"]),
                "retained_count": len(matches),
                "memory_char_limit_errors": error_count,
                "chars_used": state["memory_chars"] + state["user_chars"],
                "chars_limit": 3575,
                "entry_count": len(state["entries"]),
                "tool_outputs": seeded["parsed_outputs"],
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
        ctx.addon.remember_facts(facts, session_id="bench-dim2")
        ctx.addon.consolidate()
        active_facts = ctx.addon.active_facts()
        salience_distribution = salience_stats([row["salience"] for row in active_facts])
        matches = greedy_matches(injected, [row["content"] for row in active_facts])
        addon.update(
            {
                "kept_count": len(active_facts),
                "retained_count": len(matches),
                "salience_distribution": salience_distribution,
                "chars_used": None,
                "chars_limit": None,
                "fuzzy_matches": matches,
                "system_prompt_chars": len(
                    ctx.addon.get_context(session_id="bench-dim2", query=PREFETCH_AUDIT_QUERY)
                ),
            }
        )
    except Exception as exc:
        append_error(addon, exc)
    finally:
        finalize_record(addon, addon_started)

    return {"baseline": baseline, "addon": addon}
