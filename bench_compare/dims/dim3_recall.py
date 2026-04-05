from __future__ import annotations

from bench_compare.dims.common import RECALL_PROMPT, append_error, finalize_record, started
from bench_compare.systems import empty_record
from bench_compare.utils.facts_corpus import generate_recall_sessions
from bench_compare.utils.fuzzy import precision_recall_f1
from bench_compare.utils.llm import run_agent_recall


def run(ctx) -> dict[str, dict[str, object]]:
    sessions = generate_recall_sessions()
    injected = [fact.text for session in sessions for fact in session.facts]
    baseline = empty_record("DIM-3", "baseline", injected)
    addon = empty_record("DIM-3", "addon", injected)

    baseline_started = started()
    try:
        ctx.baseline.reset()
        ctx.baseline.seed_sessions(sessions)
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
        precision, recall, f1, matches = precision_recall_f1(injected, llm["recalled_items"])
        memory_chars = llm["system_prompt_chars"] or prompt_snapshot["chars"]
        baseline.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "raw_recalled_items": llm["recalled_items"],
                "fuzzy_matches": matches,
                "system_prompt_memory_block_chars": memory_chars,
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
        ctx.addon.seed_sessions(sessions, create_summaries=True)
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
        precision, recall, f1, matches = precision_recall_f1(injected, llm["recalled_items"])
        total_context_chars = llm["system_prompt_chars"] + llm["context_chars"]
        addon.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "raw_recalled_items": llm["recalled_items"],
                "fuzzy_matches": matches,
                "system_prompt_memory_block_chars": total_context_chars,
                "system_prompt_chars": total_context_chars,
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
