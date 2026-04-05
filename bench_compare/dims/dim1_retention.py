from __future__ import annotations

from bench_compare.dims.common import PREFETCH_AUDIT_QUERY, append_error, finalize_record, started
from bench_compare.systems import empty_record
from bench_compare.utils.facts_corpus import generate_scale_facts
from bench_compare.utils.fuzzy import greedy_matches


def run(ctx) -> dict[str, dict[str, object]]:
    facts = generate_scale_facts(ctx.scale_facts)
    injected = [fact.text for fact in facts]
    baseline = empty_record("DIM-1", "baseline", injected)
    addon = empty_record("DIM-1", "addon", injected)

    baseline_started = started()
    try:
        ctx.baseline.reset()
        ctx.baseline.seed_direct_snapshot(facts)
        state = ctx.baseline.snapshot_state()
        matches = greedy_matches(injected, state["entries"])
        baseline.update(
            {
                "retained_count": len(matches),
                "retention_rate": len(matches) / len(injected) if injected else 0.0,
                "chars_used": state["memory_chars"] + state["user_chars"],
                "chars_limit": 3575,
                "entry_count": len(state["entries"]),
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
        ctx.addon.remember_facts(facts, session_id="bench-dim1")
        ctx.addon.consolidate()
        active_facts = ctx.addon.active_facts()
        matches = greedy_matches(injected, [row["content"] for row in active_facts])
        addon.update(
            {
                "retained_count": len(matches),
                "retention_rate": len(matches) / len(injected) if injected else 0.0,
                "chars_used": None,
                "chars_limit": None,
                "entry_count": len(active_facts),
                "fuzzy_matches": matches,
                "system_prompt_chars": len(
                    ctx.addon.get_context(session_id="bench-dim1", query=PREFETCH_AUDIT_QUERY)
                ),
            }
        )
    except Exception as exc:
        append_error(addon, exc)
    finally:
        finalize_record(addon, addon_started)

    return {"baseline": baseline, "addon": addon}
