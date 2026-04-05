from __future__ import annotations

from bench_compare.dims.common import PREFETCH_AUDIT_QUERY, append_error, finalize_record, started
from bench_compare.systems import empty_record
from bench_compare.utils.facts_corpus import generate_contradiction_pairs
from bench_compare.utils.fuzzy import greedy_matches, similarity
from bench_compare.utils.memory_reader import query_rows


def run(ctx) -> dict[str, dict[str, object]]:
    pairs = generate_contradiction_pairs()
    injected = [item.text for pair in pairs for item in (pair.earlier, pair.later)]
    baseline = empty_record("DIM-4", "baseline", injected)
    addon = empty_record("DIM-4", "addon", injected)

    baseline_started = started()
    try:
        ctx.baseline.reset()
        ctx.baseline.seed_direct_snapshot([item for pair in pairs for item in (pair.earlier, pair.later)])
        state = ctx.baseline.snapshot_state()
        unresolved = 0
        for pair in pairs:
            older_present = _contains_match(state["entries"], pair.earlier.text)
            newer_present = _contains_match(state["entries"], pair.later.text)
            if older_present and newer_present:
                unresolved += 1
        resolved = len(pairs) - unresolved
        baseline.update(
            {
                "contradictions_resolved": resolved,
                "contradictions_unresolved": unresolved,
                "resolution_rate": resolved / len(pairs) if pairs else 0.0,
                "fuzzy_matches": greedy_matches(injected, state["entries"]),
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
        for index, pair in enumerate(pairs):
            session_id = f"bench-dim4-{index:02d}"
            ctx.addon.remember_fact(pair.earlier, session_id=session_id)
            ctx.addon.remember_fact(pair.later, session_id=session_id)
        ctx.addon.consolidate()
        fact_rows = query_rows(
            ctx.addon.db_path,
            """
            SELECT id, content, active, subject_key, value_key
            FROM facts
            ORDER BY id ASC
            """,
        )
        contradiction_rows = query_rows(
            ctx.addon.db_path,
            """
            SELECT id, subject_key, winner_fact_id, loser_fact_id, resolution
            FROM contradictions
            ORDER BY id ASC
            """,
        )
        supersedes = query_rows(
            ctx.addon.db_path,
            """
            SELECT id, source_id, target_id
            FROM memory_links
            WHERE link_type = 'supersedes'
            ORDER BY id ASC
            """,
        )
        resolved = 0
        unresolved = 0
        for pair in pairs:
            older_active = _row_active(fact_rows, pair.earlier.text)
            newer_active = _row_active(fact_rows, pair.later.text)
            contradiction_logged = any(row["subject_key"] == pair.later.subject_key for row in contradiction_rows)
            if newer_active and not older_active and contradiction_logged:
                resolved += 1
            else:
                unresolved += 1
        addon.update(
            {
                "contradictions_resolved": resolved,
                "contradictions_unresolved": unresolved,
                "resolution_rate": resolved / len(pairs) if pairs else 0.0,
                "supersession_chain_count": len(supersedes),
                "contradiction_rows": contradiction_rows,
                "fuzzy_matches": greedy_matches(injected, [row["content"] for row in fact_rows]),
                "system_prompt_chars": len(
                    ctx.addon.get_context(session_id="bench-dim4", query=PREFETCH_AUDIT_QUERY)
                ),
            }
        )
    except Exception as exc:
        append_error(addon, exc)
    finally:
        finalize_record(addon, addon_started)

    return {"baseline": baseline, "addon": addon}


def _contains_match(entries: list[str], target: str) -> bool:
    return any(similarity(entry, target) >= 0.82 for entry in entries)


def _row_active(rows: list[dict[str, object]], target: str) -> bool:
    for row in rows:
        if similarity(str(row.get("content") or ""), target) >= 0.82:
            return bool(row.get("active"))
    return False
