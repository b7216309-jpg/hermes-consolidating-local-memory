from __future__ import annotations

from bench_compare.dims.common import RECALL_PROMPT, append_error, finalize_record, started
from bench_compare.systems import empty_record
from bench_compare.utils.facts_corpus import generate_signal_noise_facts
from bench_compare.utils.fuzzy import greedy_matches
from bench_compare.utils.llm import run_agent_recall


def run(ctx) -> dict[str, dict[str, object]]:
    useful_facts, noise_facts = generate_signal_noise_facts()
    all_facts = useful_facts + noise_facts
    injected = [fact.text for fact in all_facts]
    useful_texts = [fact.text for fact in useful_facts]
    noise_texts = [fact.text for fact in noise_facts]
    baseline = empty_record("DIM-7", "baseline", injected)
    addon = empty_record("DIM-7", "addon", injected)

    baseline_started = started()
    try:
        ctx.baseline.reset()
        ctx.baseline.seed_direct_snapshot(all_facts)
        prompt_snapshot = ctx.baseline.prompt_snapshot()
        llm = run_agent_recall(
            repo_root=ctx.repo_root,
            hermes_home=ctx.baseline.hermes_home,
            model=ctx.model,
            query=RECALL_PROMPT,
            timeout_seconds=ctx.timeout_seconds,
            addon=False,
            wsl_settings=ctx.wsl_settings,
        )
        useful_matches = greedy_matches(useful_texts, llm["recalled_items"])
        noise_matches = greedy_matches(noise_texts, llm["recalled_items"])
        memory_chars = llm["system_prompt_chars"] or prompt_snapshot["chars"]
        baseline.update(
            {
                "useful_recalled": len(useful_matches),
                "noise_recalled": len(noise_matches),
                "signal_noise_ratio": len(useful_matches) / len(llm["recalled_items"])
                if llm["recalled_items"]
                else 0.0,
                "raw_recalled_items": llm["recalled_items"],
                "fuzzy_matches": useful_matches + noise_matches,
                "system_prompt_chars": memory_chars + llm["context_chars"],
                "llm_calls_made": 1,
                "tokens_estimated": llm["tokens_estimated"],
            }
        )
        baseline["errors"].extend(llm["parse_errors"])
    except Exception as exc:
        append_error(baseline, exc)
    finally:
        finalize_record(baseline, baseline_started)

    addon_started = started()
    try:
        ctx.addon.reset()
        ctx.addon.remember_facts(all_facts, session_id="bench-dim7")
        ctx.addon.consolidate()
        llm = run_agent_recall(
            repo_root=ctx.repo_root,
            hermes_home=ctx.addon.hermes_home,
            model=ctx.model,
            query=RECALL_PROMPT,
            timeout_seconds=ctx.timeout_seconds,
            addon=True,
            provider_config=ctx.addon.addon_config,
            wsl_settings=ctx.wsl_settings,
        )
        useful_matches = greedy_matches(useful_texts, llm["recalled_items"])
        noise_matches = greedy_matches(noise_texts, llm["recalled_items"])
        addon.update(
            {
                "useful_recalled": len(useful_matches),
                "noise_recalled": len(noise_matches),
                "signal_noise_ratio": len(useful_matches) / len(llm["recalled_items"])
                if llm["recalled_items"]
                else 0.0,
                "raw_recalled_items": llm["recalled_items"],
                "fuzzy_matches": useful_matches + noise_matches,
                "system_prompt_chars": llm["system_prompt_chars"] + llm["context_chars"],
                "llm_calls_made": 1,
                "tokens_estimated": llm["tokens_estimated"],
            }
        )
        addon["errors"].extend(llm["parse_errors"])
    except Exception as exc:
        append_error(addon, exc)
    finally:
        finalize_record(addon, addon_started)

    return {"baseline": baseline, "addon": addon}
