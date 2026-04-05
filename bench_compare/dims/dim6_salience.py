from __future__ import annotations

from bench_compare.dims.common import append_error, finalize_record, started
from bench_compare.systems import empty_record
from bench_compare.utils.facts_corpus import generate_salience_facts
from bench_compare.utils.memory_reader import query_rows, salience_stats


def run(ctx) -> dict[str, dict[str, object]]:
    facts = generate_salience_facts()
    injected = [fact.text for fact in facts]
    baseline = empty_record("DIM-6", "baseline", injected)
    addon = empty_record("DIM-6", "addon", injected)

    baseline_started = started()
    try:
        baseline.update(
            {
                "status": "not_applicable",
                "not_applicable": True,
                "pass_high_gt_low": None,
            }
        )
    except Exception as exc:
        append_error(baseline, exc)
    finally:
        finalize_record(baseline, baseline_started)

    addon_started = started()
    try:
        ctx.addon.reset()
        for fact in facts:
            ctx.addon.store_fact_direct(fact, session_id="bench-dim6")
        decay = ctx.addon.decay()
        rows = query_rows(
            ctx.addon.db_path,
            """
            SELECT id, content, salience, active
            FROM facts
            ORDER BY id ASC
            """,
        )
        tier_values: dict[str, list[float]] = {"high": [], "medium": [], "low": []}
        for fact, row in zip(facts, rows):
            tier_values[fact.tier].append(float(row.get("salience") or 0.0))
        stats = {tier: salience_stats(values) for tier, values in tier_values.items()}
        high_avg = float(stats["high"]["avg"])
        low_avg = float(stats["low"]["avg"])
        addon.update(
            {
                "salience_by_tier": stats,
                "pass_high_gt_low": high_avg > low_avg,
                "high_minus_low_avg": round(high_avg - low_avg, 4),
                "decay_result": decay.get("result") if isinstance(decay, dict) else decay,
            }
        )
    except Exception as exc:
        append_error(addon, exc)
    finally:
        finalize_record(addon, addon_started)

    return {"baseline": baseline, "addon": addon}
