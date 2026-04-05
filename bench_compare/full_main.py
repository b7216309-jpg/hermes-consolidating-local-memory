from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench_compare.__main__ import detect_hermes_version, detect_plugin_version
from bench_compare.dims.common import RECALL_PROMPT, append_error, finalize_record, started
from bench_compare.full_scenarios import (
    REAL_DIMENSIONS,
    acquisition_facts,
    build_changed_subjects_prompt,
    build_contradiction_prompts,
    build_current_values_prompt,
    build_seed_prompts,
    signal_noise_facts,
    task_grounding_facts,
)
from bench_compare.report import write_results
from bench_compare.systems import AddonSystem, BaselineSystem, empty_record, resolve_runtime
from bench_compare.utils.fuzzy import greedy_matches, precision_recall_f1
from bench_compare.utils.hermes_home import DEFAULT_ADDON_CONFIG
from bench_compare.utils.llm import (
    parse_json_array,
    parse_json_object,
    run_agent_conversation,
    run_agent_prompt,
    validate_agent_runtime,
)
from bench_compare.utils.wsl import collect_wsl_runtime_credentials, collect_wsl_runtime_seed
from plugins.memory.consolidating_local.llm_client import is_codex_backend

REAL_DIMENSION_WEIGHTS = {
    "REAL-1": 1.25,
    "REAL-2": 1.0,
    "REAL-3": 1.0,
    "REAL-4": 1.0,
    "REAL-5": 1.0,
}

FULL_ADDON_DEFAULTS = {
    "extractor_backend": "llm",
    "retrieval_backend": "hybrid",
    "prefetch_limit": 24,
    "max_topic_facts": 10,
    "topic_summary_chars": 1200,
    "session_summary_chars": 1800,
    "min_hours": 0,
    "min_sessions": 0,
    "decay_half_life_days": 180,
    "decay_min_salience": 0.05,
    "episode_body_retention_hours": 168,
    "wiki_export_enabled": False,
    "wiki_export_on_consolidate": False,
}


@dataclass
class FullBenchmarkContext:
    repo_root: Path
    model: str
    scale_facts: int
    timeout_seconds: int
    seed_batch_size: int
    addon_config: dict[str, Any]
    baseline: BaselineSystem
    addon: AddonSystem
    wsl_settings: dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        runtime = resolve_runtime()
    except Exception as exc:
        print(f"Full benchmark prerequisites unavailable: {exc}")
        return 2

    wsl_settings = build_wsl_settings(args)
    try:
        validate_agent_runtime(repo_root=repo_root, wsl_settings=wsl_settings)
    except Exception as exc:
        print(f"Full benchmark prerequisites unavailable: {exc}")
        return 2

    runtime_seed = collect_runtime_seed(wsl_settings)
    runtime_credentials = collect_runtime_credentials(wsl_settings)
    addon_config = build_addon_config(args, runtime_seed, runtime_credentials)
    apply_backend_env(runtime_credentials, addon_config)
    try:
        validate_full_addon_backend(addon_config)
    except Exception as exc:
        print(f"Full benchmark backend prerequisites unavailable: {exc}")
        return 2

    baseline = BaselineSystem(
        hermes_home=Path(args.hermes_home_baseline),
        runtime=runtime,
        runtime_seed=runtime_seed,
    )
    addon = AddonSystem(
        hermes_home=Path(args.hermes_home_addon),
        runtime=runtime,
        addon_config=addon_config,
        runtime_seed=runtime_seed,
    )
    context = FullBenchmarkContext(
        repo_root=repo_root,
        model=args.model,
        scale_facts=args.scale_facts,
        timeout_seconds=args.timeout,
        seed_batch_size=args.seed_batch_size,
        addon_config=addon_config,
        baseline=baseline,
        addon=addon,
        wsl_settings=wsl_settings,
    )

    dimensions: dict[str, dict[str, dict[str, Any]]] = {}
    selected_dims = parse_dims(args.dims)
    try:
        for dim_id in selected_dims:
            dimensions[dim_id] = run_dimension(dim_id, context)
    finally:
        baseline.close()
        addon.close()

    total_llm_calls = sum(int(record.get("llm_calls_made") or 0) for systems in dimensions.values() for record in systems.values())
    total_tokens = sum(int(record.get("tokens_estimated") or 0) for systems in dimensions.values() for record in systems.values())
    summary = build_summary(dimensions)
    output_path = Path(args.output) if args.output else default_output_path()
    payload = {
        "meta": {
            "benchmark_mode": "complete_real",
            "model": args.model,
            "hermes_version": detect_hermes_version(),
            "plugin_version": detect_plugin_version(repo_root),
            "timestamp": output_path.stem.replace("bench_results_full_", ""),
            "total_llm_calls": total_llm_calls,
            "total_tokens_estimated": total_tokens,
            "hermes_home_baseline": str(Path(args.hermes_home_baseline).resolve()),
            "hermes_home_addon": str(Path(args.hermes_home_addon).resolve()),
            "addon_config": {key: addon_config.get(key) for key in sorted(addon_config.keys())},
        },
        "dimensions": dimensions,
        "summary": summary,
    }
    write_results(output_path, payload)
    render_stdout(dimensions, total_llm_calls, total_tokens)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hermes complete real benchmark: baseline vs consolidating_local with LLM-heavy end-to-end evaluation"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--scale-facts", type=int, default=40)
    parser.add_argument("--seed-batch-size", type=int, default=5)
    parser.add_argument("--dims", default=",".join(REAL_DIMENSIONS))
    parser.add_argument("--output")
    parser.add_argument("--hermes-home-baseline", required=True)
    parser.add_argument("--hermes-home-addon", required=True)
    parser.add_argument("--timeout", type=int, default=600)

    parser.add_argument("--extractor-backend", default=FULL_ADDON_DEFAULTS["extractor_backend"])
    parser.add_argument("--retrieval-backend", default=FULL_ADDON_DEFAULTS["retrieval_backend"])
    parser.add_argument("--prefetch-limit", type=int, default=FULL_ADDON_DEFAULTS["prefetch_limit"])
    parser.add_argument("--max-topic-facts", type=int, default=FULL_ADDON_DEFAULTS["max_topic_facts"])
    parser.add_argument("--topic-summary-chars", type=int, default=FULL_ADDON_DEFAULTS["topic_summary_chars"])
    parser.add_argument("--session-summary-chars", type=int, default=FULL_ADDON_DEFAULTS["session_summary_chars"])
    parser.add_argument("--min-hours", type=float, default=FULL_ADDON_DEFAULTS["min_hours"])
    parser.add_argument("--min-sessions", type=int, default=FULL_ADDON_DEFAULTS["min_sessions"])
    parser.add_argument("--decay-half-life-days", type=float, default=FULL_ADDON_DEFAULTS["decay_half_life_days"])
    parser.add_argument("--decay-min-salience", type=float, default=FULL_ADDON_DEFAULTS["decay_min_salience"])
    parser.add_argument("--episode-body-retention-hours", type=float, default=FULL_ADDON_DEFAULTS["episode_body_retention_hours"])
    parser.add_argument("--addon-llm-model", default="")
    parser.add_argument("--addon-llm-base-url", default="")
    parser.add_argument("--addon-embedding-model", default="")
    parser.add_argument("--addon-embedding-base-url", default="")

    parser.add_argument("--use-wsl", action="store_true")
    parser.add_argument("--wsl-distro", default="Ubuntu")
    parser.add_argument("--wsl-hermes-root", default="~/.hermes/hermes-agent")
    parser.add_argument("--wsl-python", default="")
    return parser.parse_args(argv)


def build_wsl_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "enabled": bool(args.use_wsl),
        "distro": args.wsl_distro,
        "hermes_repo_root": args.wsl_hermes_root,
        "python_bin": args.wsl_python,
        "hermes_home": args.hermes_home_baseline,
    }


def collect_runtime_seed(wsl_settings: dict[str, Any]) -> dict[str, Any]:
    if wsl_settings.get("enabled"):
        return collect_wsl_runtime_seed(str(wsl_settings.get("distro") or "Ubuntu"))
    return {}


def collect_runtime_credentials(wsl_settings: dict[str, Any]) -> dict[str, str]:
    if not wsl_settings.get("enabled"):
        return {}
    try:
        return collect_wsl_runtime_credentials(
            distro=str(wsl_settings.get("distro") or "Ubuntu"),
            hermes_repo_root=str(wsl_settings.get("hermes_repo_root") or "~/.hermes/hermes-agent"),
            python_bin=str(wsl_settings.get("python_bin") or ""),
        )
    except Exception:
        return {}


def build_addon_config(
    args: argparse.Namespace,
    runtime_seed: dict[str, Any],
    runtime_credentials: dict[str, str],
) -> dict[str, Any]:
    model_cfg = dict(runtime_seed.get("model_config") or {})
    llm_model = str(args.addon_llm_model or args.model or "").strip()
    llm_base_url = first_supported_llm_base_url(
        str(args.addon_llm_base_url or "").strip(),
        str(runtime_credentials.get("base_url") or "").strip(),
        str(model_cfg.get("base_url") or "").strip(),
    )
    embedding_model = str(args.addon_embedding_model or llm_model or "").strip()
    embedding_base_url = first_compatible_backend_base_url(
        str(args.addon_embedding_base_url or "").strip(),
        llm_base_url,
    )
    return {
        **DEFAULT_ADDON_CONFIG,
        **FULL_ADDON_DEFAULTS,
        "extractor_backend": args.extractor_backend,
        "retrieval_backend": args.retrieval_backend,
        "prefetch_limit": args.prefetch_limit,
        "max_topic_facts": args.max_topic_facts,
        "topic_summary_chars": args.topic_summary_chars,
        "session_summary_chars": args.session_summary_chars,
        "min_hours": args.min_hours,
        "min_sessions": args.min_sessions,
        "decay_half_life_days": args.decay_half_life_days,
        "decay_min_salience": args.decay_min_salience,
        "episode_body_retention_hours": args.episode_body_retention_hours,
        "llm_model": llm_model,
        "llm_base_url": llm_base_url,
        "embedding_model": embedding_model,
        "embedding_base_url": embedding_base_url,
    }


def apply_backend_env(runtime_credentials: dict[str, str], addon_config: dict[str, Any]) -> None:
    runtime_api_key = str(runtime_credentials.get("api_key") or "").strip()
    if runtime_api_key and not os.environ.get("CONSOLIDATING_MEMORY_LLM_API_KEY") and addon_config.get("llm_base_url"):
        os.environ["CONSOLIDATING_MEMORY_LLM_API_KEY"] = runtime_api_key
    if (
        runtime_api_key
        and not os.environ.get("CONSOLIDATING_MEMORY_EMBEDDING_API_KEY")
        and is_openai_compatible_backend(str(addon_config.get("embedding_base_url") or ""))
    ):
        os.environ["CONSOLIDATING_MEMORY_EMBEDDING_API_KEY"] = runtime_api_key


def is_openai_compatible_backend(base_url: str) -> bool:
    clean = str(base_url or "").strip().rstrip("/").lower()
    if not clean:
        return False
    incompatible_markers = (
        "chatgpt.com/backend-api/codex",
        "/backend-api/codex",
    )
    return not any(marker in clean for marker in incompatible_markers)


def is_supported_llm_backend(base_url: str) -> bool:
    clean = str(base_url or "").strip()
    return bool(clean) and (is_openai_compatible_backend(clean) or is_codex_backend(clean))


def first_supported_llm_base_url(*candidates: str) -> str:
    for candidate in candidates:
        clean = str(candidate or "").strip()
        if clean and is_supported_llm_backend(clean):
            return clean
    return ""


def first_compatible_backend_base_url(*candidates: str) -> str:
    for candidate in candidates:
        clean = str(candidate or "").strip()
        if clean and is_openai_compatible_backend(clean):
            return clean
    return ""


def validate_full_addon_backend(addon_config: dict[str, Any]) -> None:
    extractor_backend = str(addon_config.get("extractor_backend") or "").strip().lower()
    retrieval_backend = str(addon_config.get("retrieval_backend") or "").strip().lower()
    llm_model = str(addon_config.get("llm_model") or "").strip()
    llm_base_url = str(addon_config.get("llm_base_url") or "").strip()
    embedding_model = str(addon_config.get("embedding_model") or "").strip()
    embedding_base_url = str(addon_config.get("embedding_base_url") or "").strip()

    if extractor_backend in {"llm", "hybrid"}:
        if not llm_model or not llm_base_url:
            raise RuntimeError(
                "The complete real benchmark requires a real extractor backend. "
                "Set --addon-llm-model and --addon-llm-base-url, or use a Hermes runtime whose base_url supports chat-completions or Codex responses."
            )
        if not is_supported_llm_backend(llm_base_url):
            raise RuntimeError(
                "The addon extraction backend is not usable with the current base URL. "
                "Provide either an OpenAI-compatible /chat/completions endpoint or a Hermes Codex /responses endpoint."
            )
    if retrieval_backend == "hybrid":
        if not embedding_model or not embedding_base_url:
            raise RuntimeError(
                "The complete real benchmark requires an embedding backend for hybrid retrieval. "
                "Set --addon-embedding-model and --addon-embedding-base-url, or rely on compatible llm defaults."
            )
        if not is_openai_compatible_backend(embedding_base_url):
            raise RuntimeError(
                "The addon embedding backend cannot use a Codex/chatgpt backend URL. "
                "Provide an OpenAI-compatible /embeddings endpoint with --addon-embedding-base-url."
            )


def parse_dims(raw: str) -> list[str]:
    dims = [item.strip().upper() for item in str(raw or "").split(",") if item.strip()]
    unknown = [dim for dim in dims if dim not in REAL_DIMENSIONS]
    if unknown:
        raise SystemExit(f"Unknown full benchmark dimensions requested: {', '.join(unknown)}")
    return dims


def run_dimension(dim_id: str, context: FullBenchmarkContext) -> dict[str, dict[str, Any]]:
    handlers = {
        "REAL-1": run_real_1,
        "REAL-2": run_real_2,
        "REAL-3": run_real_3,
        "REAL-4": run_real_4,
        "REAL-5": run_real_5,
    }
    try:
        return handlers[dim_id](context)
    except Exception as exc:
        baseline = empty_record(dim_id, "baseline", [])
        addon = empty_record(dim_id, "addon", [])
        baseline["errors"].append(f"{type(exc).__name__}: {exc}")
        addon["errors"].append(f"{type(exc).__name__}: {exc}")
        return {"baseline": baseline, "addon": addon}


def run_real_1(ctx: FullBenchmarkContext) -> dict[str, dict[str, Any]]:
    facts = acquisition_facts(ctx.scale_facts)
    injected = [fact.text for fact in facts]
    prompts = build_seed_prompts(facts, batch_size=ctx.seed_batch_size, session_label="complete real retention benchmark")
    return {
        "baseline": _run_seed_and_recall_dimension(
            ctx,
            dim_id="REAL-1",
            system="baseline",
            injected=injected,
            seed_prompts=prompts,
            recall_query=RECALL_PROMPT,
            parser="array",
            scorer="recall",
        ),
        "addon": _run_seed_and_recall_dimension(
            ctx,
            dim_id="REAL-1",
            system="addon",
            injected=injected,
            seed_prompts=prompts,
            recall_query=RECALL_PROMPT,
            parser="array",
            scorer="recall",
        ),
    }


def run_real_2(ctx: FullBenchmarkContext) -> dict[str, dict[str, Any]]:
    prompts, expected_current, expected_previous = build_contradiction_prompts(batch_size=ctx.seed_batch_size)
    injected = [prompt for prompt in prompts]
    query = build_current_values_prompt(expected_current)
    return {
        "baseline": _run_seed_and_object_dimension(
            ctx,
            dim_id="REAL-2",
            system="baseline",
            injected=injected,
            seed_prompts=prompts,
            prompt=query,
            expected=expected_current,
            stale_values=expected_previous,
        ),
        "addon": _run_seed_and_object_dimension(
            ctx,
            dim_id="REAL-2",
            system="addon",
            injected=injected,
            seed_prompts=prompts,
            prompt=query,
            expected=expected_current,
            stale_values=expected_previous,
        ),
    }


def run_real_3(ctx: FullBenchmarkContext) -> dict[str, dict[str, Any]]:
    facts, expected = task_grounding_facts()
    injected = [fact.text for fact in facts]
    prompts = build_seed_prompts(facts, batch_size=max(3, min(ctx.seed_batch_size, 4)), session_label="complete real task grounding benchmark")
    query = build_current_values_prompt(expected)
    return {
        "baseline": _run_seed_and_object_dimension(
            ctx,
            dim_id="REAL-3",
            system="baseline",
            injected=injected,
            seed_prompts=prompts,
            prompt=query,
            expected=expected,
        ),
        "addon": _run_seed_and_object_dimension(
            ctx,
            dim_id="REAL-3",
            system="addon",
            injected=injected,
            seed_prompts=prompts,
            prompt=query,
            expected=expected,
        ),
    }


def run_real_4(ctx: FullBenchmarkContext) -> dict[str, dict[str, Any]]:
    useful_facts, noise_facts = signal_noise_facts()
    mixed: list[Any] = []
    for useful, noise in zip(useful_facts, noise_facts):
        mixed.append(useful)
        mixed.append(noise)
    injected = [fact.text for fact in mixed]
    prompts = build_seed_prompts(mixed, batch_size=ctx.seed_batch_size, session_label="complete real noise benchmark")
    useful_texts = [fact.text for fact in useful_facts]
    noise_texts = [fact.text for fact in noise_facts]
    return {
        "baseline": _run_seed_and_noise_dimension(
            ctx,
            dim_id="REAL-4",
            system="baseline",
            injected=injected,
            seed_prompts=prompts,
            useful_texts=useful_texts,
            noise_texts=noise_texts,
        ),
        "addon": _run_seed_and_noise_dimension(
            ctx,
            dim_id="REAL-4",
            system="addon",
            injected=injected,
            seed_prompts=prompts,
            useful_texts=useful_texts,
            noise_texts=noise_texts,
        ),
    }


def run_real_5(ctx: FullBenchmarkContext) -> dict[str, dict[str, Any]]:
    prompts, expected_current, _expected_previous = build_contradiction_prompts(batch_size=ctx.seed_batch_size)
    injected = list(expected_current.keys())
    query = build_changed_subjects_prompt(expected_current)
    return {
        "baseline": _run_seed_and_recall_dimension(
            ctx,
            dim_id="REAL-5",
            system="baseline",
            injected=injected,
            seed_prompts=prompts,
            recall_query=query,
            parser="array",
            scorer="recall",
        ),
        "addon": _run_seed_and_recall_dimension(
            ctx,
            dim_id="REAL-5",
            system="addon",
            injected=injected,
            seed_prompts=prompts,
            recall_query=query,
            parser="array",
            scorer="recall",
        ),
    }


def _run_seed_and_recall_dimension(
    ctx: FullBenchmarkContext,
    *,
    dim_id: str,
    system: str,
    injected: list[str],
    seed_prompts: list[str],
    recall_query: str,
    parser: str,
    scorer: str,
) -> dict[str, Any]:
    record = empty_record(dim_id, system, injected)
    record["raw_seed_prompts"] = list(seed_prompts)
    started_at = started()
    try:
        seed = _seed_system(ctx, system=system, prompts=seed_prompts, session_id=f"{dim_id.lower()}-{system}-seed")
        record["raw_seed_responses"] = list(seed["responses"])
        record["raw_seed_call_error"] = str(seed.get("call_error") or "")
        record["raw_seed_agent_logs"] = str(seed.get("agent_logs") or "")
        record["seed_system_prompt_chars"] = int(seed.get("system_prompt_chars") or 0)
        if seed.get("call_error"):
            record["errors"].append(f"Seed call failed: {seed['call_error']}")
        prompt_result = _prompt_system(ctx, system=system, query=recall_query)
        record["raw_prompt_query"] = recall_query
        record["raw_prompt_response"] = str(prompt_result.get("response_text") or "")
        record["raw_prompt_call_error"] = str(prompt_result.get("call_error") or "")
        record["raw_prompt_agent_logs"] = str(prompt_result.get("agent_logs") or "")
        if prompt_result["call_error"]:
            record["errors"].append(f"Prompt call failed: {prompt_result['call_error']}")
        if parser == "array":
            observed, parse_errors = parse_json_array(prompt_result["response_text"])
            record["raw_recalled_items"] = observed
            record["errors"].extend(parse_errors)
            if scorer == "recall":
                precision, recall, f1, matches = precision_recall_f1(injected, observed)
                record.update(
                    {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "fuzzy_matches": matches,
                    }
                )
        record["system_prompt_chars"] = int(prompt_result["system_prompt_chars"] or 0) + int(prompt_result["context_chars"] or 0)
        record["llm_calls_made"] = len(seed_prompts) + 1
        record["tokens_estimated"] = int(seed["tokens_estimated"] or 0) + int(prompt_result["tokens_estimated"] or 0)
    except Exception as exc:
        append_error(record, exc)
    finally:
        finalize_record(record, started_at)
    return record


def _run_seed_and_object_dimension(
    ctx: FullBenchmarkContext,
    *,
    dim_id: str,
    system: str,
    injected: list[str],
    seed_prompts: list[str],
    prompt: str,
    expected: dict[str, str],
    stale_values: dict[str, str] | None = None,
) -> dict[str, Any]:
    record = empty_record(dim_id, system, injected)
    record["raw_seed_prompts"] = list(seed_prompts)
    started_at = started()
    try:
        seed = _seed_system(ctx, system=system, prompts=seed_prompts, session_id=f"{dim_id.lower()}-{system}-seed")
        record["raw_seed_responses"] = list(seed["responses"])
        record["raw_seed_call_error"] = str(seed.get("call_error") or "")
        record["raw_seed_agent_logs"] = str(seed.get("agent_logs") or "")
        record["seed_system_prompt_chars"] = int(seed.get("system_prompt_chars") or 0)
        if seed.get("call_error"):
            record["errors"].append(f"Seed call failed: {seed['call_error']}")
        prompt_result = _prompt_system(ctx, system=system, query=prompt)
        record["raw_prompt_query"] = prompt
        record["raw_prompt_response"] = str(prompt_result.get("response_text") or "")
        record["raw_prompt_call_error"] = str(prompt_result.get("call_error") or "")
        record["raw_prompt_agent_logs"] = str(prompt_result.get("agent_logs") or "")
        if prompt_result["call_error"]:
            record["errors"].append(f"Prompt call failed: {prompt_result['call_error']}")
        observed, parse_errors = parse_json_object(prompt_result["response_text"])
        record["errors"].extend(parse_errors)
        record["raw_object_response"] = observed
        field_results, correct_count, stale_count = _score_expected_object(
            expected=expected,
            observed=observed,
            stale_values=stale_values or {},
        )
        record["field_results"] = field_results
        record["fuzzy_matches"] = field_results
        record["correct_count"] = correct_count
        record["total_fields"] = len(expected)
        record["accuracy"] = correct_count / len(expected) if expected else 0.0
        if stale_values:
            record["stale_value_count"] = stale_count
            record["stale_value_rate"] = stale_count / len(expected) if expected else 0.0
        record["system_prompt_chars"] = int(prompt_result["system_prompt_chars"] or 0) + int(prompt_result["context_chars"] or 0)
        record["llm_calls_made"] = len(seed_prompts) + 1
        record["tokens_estimated"] = int(seed["tokens_estimated"] or 0) + int(prompt_result["tokens_estimated"] or 0)
    except Exception as exc:
        append_error(record, exc)
    finally:
        finalize_record(record, started_at)
    return record


def _run_seed_and_noise_dimension(
    ctx: FullBenchmarkContext,
    *,
    dim_id: str,
    system: str,
    injected: list[str],
    seed_prompts: list[str],
    useful_texts: list[str],
    noise_texts: list[str],
) -> dict[str, Any]:
    record = empty_record(dim_id, system, injected)
    record["raw_seed_prompts"] = list(seed_prompts)
    started_at = started()
    try:
        seed = _seed_system(ctx, system=system, prompts=seed_prompts, session_id=f"{dim_id.lower()}-{system}-seed")
        record["raw_seed_responses"] = list(seed["responses"])
        record["raw_seed_call_error"] = str(seed.get("call_error") or "")
        record["raw_seed_agent_logs"] = str(seed.get("agent_logs") or "")
        record["seed_system_prompt_chars"] = int(seed.get("system_prompt_chars") or 0)
        if seed.get("call_error"):
            record["errors"].append(f"Seed call failed: {seed['call_error']}")
        prompt_result = _prompt_system(ctx, system=system, query=RECALL_PROMPT)
        record["raw_prompt_query"] = RECALL_PROMPT
        record["raw_prompt_response"] = str(prompt_result.get("response_text") or "")
        record["raw_prompt_call_error"] = str(prompt_result.get("call_error") or "")
        record["raw_prompt_agent_logs"] = str(prompt_result.get("agent_logs") or "")
        if prompt_result["call_error"]:
            record["errors"].append(f"Prompt call failed: {prompt_result['call_error']}")
        observed, parse_errors = parse_json_array(prompt_result["response_text"])
        record["raw_recalled_items"] = observed
        record["errors"].extend(parse_errors)
        useful_matches = greedy_matches(useful_texts, observed)
        noise_matches = greedy_matches(noise_texts, observed)
        record["useful_recalled"] = len(useful_matches)
        record["noise_recalled"] = len(noise_matches)
        record["useful_recall_rate"] = len(useful_matches) / len(useful_texts) if useful_texts else 0.0
        record["signal_noise_ratio"] = len(useful_matches) / len(observed) if observed else 0.0
        record["fuzzy_matches"] = useful_matches + noise_matches
        record["system_prompt_chars"] = int(prompt_result["system_prompt_chars"] or 0) + int(prompt_result["context_chars"] or 0)
        record["llm_calls_made"] = len(seed_prompts) + 1
        record["tokens_estimated"] = int(seed["tokens_estimated"] or 0) + int(prompt_result["tokens_estimated"] or 0)
    except Exception as exc:
        append_error(record, exc)
    finally:
        finalize_record(record, started_at)
    return record


def _seed_system(
    ctx: FullBenchmarkContext,
    *,
    system: str,
    prompts: list[str],
    session_id: str,
) -> dict[str, Any]:
    if system == "baseline":
        ctx.baseline.reset()
        return run_agent_conversation(
            repo_root=ctx.repo_root,
            hermes_home=ctx.baseline.hermes_home,
            model=ctx.model,
            prompts=prompts,
            timeout_seconds=ctx.timeout_seconds,
            addon=False,
            wsl_settings=ctx.wsl_settings,
            session_id=session_id,
        )
    ctx.addon.reset()
    ctx.addon.close()
    return run_agent_conversation(
        repo_root=ctx.repo_root,
        hermes_home=ctx.addon.hermes_home,
        model=ctx.model,
        prompts=prompts,
        timeout_seconds=ctx.timeout_seconds,
        addon=True,
        provider_config=ctx.addon.addon_config,
        wsl_settings=ctx.wsl_settings,
        session_id=session_id,
        sync_provider_turns=True,
        finalize_provider_session=True,
    )


def _prompt_system(
    ctx: FullBenchmarkContext,
    *,
    system: str,
    query: str,
) -> dict[str, Any]:
    if system == "baseline":
        return run_agent_prompt(
            repo_root=ctx.repo_root,
            hermes_home=ctx.baseline.hermes_home,
            model=ctx.model,
            query=query,
            timeout_seconds=ctx.timeout_seconds,
            addon=False,
            wsl_settings=ctx.wsl_settings,
        )
    return run_agent_prompt(
        repo_root=ctx.repo_root,
        hermes_home=ctx.addon.hermes_home,
        model=ctx.model,
        query=query,
        timeout_seconds=ctx.timeout_seconds,
        addon=True,
        provider_config=ctx.addon.addon_config,
        wsl_settings=ctx.wsl_settings,
    )


def _normalize_value(value: Any) -> str:
    text = str(value or "").strip().lower()
    for old, new in (("/", "-"), ("_", "-"), (" ", "-")):
        text = text.replace(old, new)
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-")


def _score_expected_object(
    *,
    expected: dict[str, str],
    observed: dict[str, Any],
    stale_values: dict[str, str],
) -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    correct = 0
    stale = 0
    for key, expected_value in expected.items():
        observed_value = str(observed.get(key, "") or "")
        observed_norm = _normalize_value(observed_value)
        expected_norm = _normalize_value(expected_value)
        stale_norm = _normalize_value(stale_values.get(key, "")) if stale_values else ""
        status = "wrong"
        if observed_norm and observed_norm == expected_norm:
            status = "correct"
            correct += 1
        elif stale_norm and observed_norm == stale_norm:
            status = "stale"
            stale += 1
        rows.append(
            {
                "field": key,
                "expected": expected_value,
                "observed": observed_value,
                "status": status,
            }
        )
    return rows, correct, stale


def build_summary(dimensions: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    winner_per_dim: dict[str, str] = {}
    total_scores = {"baseline": 0.0, "addon": 0.0}
    total_weight = 0.0
    for dim_id, systems in dimensions.items():
        scores = score_dimension(dim_id, systems["baseline"], systems["addon"])
        winner_per_dim[dim_id] = pick_winner(scores["baseline"], scores["addon"])
        weight = REAL_DIMENSION_WEIGHTS.get(dim_id, 1.0)
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


def pick_winner(baseline_score: float, addon_score: float, *, epsilon: float = 0.02) -> str:
    if abs(addon_score - baseline_score) <= epsilon:
        return "tie"
    return "addon" if addon_score > baseline_score else "baseline"


def render_stdout(dimensions: dict[str, dict[str, dict[str, Any]]], total_llm_calls: int, total_tokens: int) -> None:
    rows = [(dim_id, format_row(dim_id, systems["baseline"], "baseline"), format_row(dim_id, systems["addon"], "addon")) for dim_id, systems in dimensions.items()]
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Hermes Complete Real Benchmark")
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
    if record.get("errors") and not _row_has_metric(dim_id, record):
        return "ERROR"
    if dim_id in {"REAL-1", "REAL-5"}:
        return f"F1={float(record.get('f1') or 0.0):.2f} R={float(record.get('recall') or 0.0):.2f}"
    if dim_id == "REAL-2":
        stale = int(record.get("stale_value_count") or 0)
        return f"acc={float(record.get('accuracy') or 0.0):.2f} stale={stale}"
    if dim_id == "REAL-3":
        return f"task={float(record.get('accuracy') or 0.0):.2f} {int(record.get('correct_count') or 0)}/{int(record.get('total_fields') or 0)}"
    if dim_id == "REAL-4":
        return f"SNR={float(record.get('signal_noise_ratio') or 0.0):.2f} useful={int(record.get('useful_recalled') or 0)}"
    return "n/a"


def _row_has_metric(dim_id: str, record: dict[str, Any]) -> bool:
    if dim_id in {"REAL-1", "REAL-5"}:
        return "f1" in record
    if dim_id in {"REAL-2", "REAL-3"}:
        return "accuracy" in record
    if dim_id == "REAL-4":
        return "signal_noise_ratio" in record
    return False


def default_output_path() -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path.cwd() / f"bench_results_full_{stamp}.json"


if __name__ == "__main__":
    raise SystemExit(main())
